from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


LABELS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
LABEL_FORMS = [f" {label}" for label in LABELS]
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SYSTEM_TEXT = "You are a careful design-evaluation annotator. Judge the visible image evidence."


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_for_row(row: dict) -> str:
    source = row["prompt"]
    context = source.split("Return only one valid JSON object.", 1)[0].strip()
    definitions = source.split("Design-intent definitions:", 1)[1].split("Scoring rules:", 1)[0].strip()
    return (
        f"{context}\n\n"
        "First inspect the visible form, use context, visual style, cultural meaning, and ambiguity. "
        "Then select the single most supported design intent.\n\n"
        f"Design-intent definitions:\n{definitions}\n\n"
        "Return exactly one lowercase label and no other text: "
        "functionality, aesthetics, usability, symbolism, or unclear."
    )


class ImageDataset(Dataset):
    def __init__(self, path: Path, image_root: Path):
        self.rows = read_jsonl(path)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = (self.image_root / row["image_file"]).resolve()
        messages = [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_for_row(row)},
            ]},
        ]
        return {
            "image_id": row["image_id"],
            "category": row["category"],
            "messages": messages,
            "human_probs": [float(row[f"human_p_{label}"]) for label in LABELS],
        }


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        texts, images = [], []
        for item in batch:
            texts.append(self.processor.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True
            ))
            image_inputs, _ = process_vision_info(item["messages"])
            images.append(image_inputs[0])
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs["human_probs"] = torch.tensor([item["human_probs"] for item in batch], dtype=torch.float32)
        inputs["image_id"] = [item["image_id"] for item in batch]
        inputs["category"] = [item["category"] for item in batch]
        return inputs


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_mode", choices=["soft", "soft_ce_only", "majority"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.10)
    parser.add_argument("--hard_weight", type=float, default=0.20)
    parser.add_argument("--entropy_weight", type=float, default=0.10)
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    for name, parameter in model.named_parameters():
        if "visual" in name.lower() and "lora_" in name:
            parameter.requires_grad = False
    model.config.use_cache = False
    return model


def label_token_ids(processor) -> list[int]:
    result = []
    for form in LABEL_FORMS:
        encoded = processor.tokenizer.encode(form, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected one token for {form!r}; got {encoded}")
        result.append(encoded[0])
    return result


def selected_label_logits(model, batch: dict, token_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
    human_probs = batch["human_probs"].to(model.device)
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key not in {"human_probs", "image_id", "category"}
    }
    output = model(**model_inputs, use_cache=False)
    positions = model_inputs["attention_mask"].long().sum(dim=1) - 1
    row_ids = torch.arange(output.logits.shape[0], device=output.logits.device)
    logits = output.logits[row_ids, positions, :].index_select(-1, token_ids)
    return logits, {"human_probs": human_probs, "model_inputs": model_inputs}


def objective_loss(logits: torch.Tensor, human_probs: torch.Tensor, mode: str,
                   hard_weight: float, entropy_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    scores = logits.float()
    majority = human_probs.argmax(dim=-1)
    if mode == "majority":
        loss = F.cross_entropy(scores, majority)
        return loss, {"total": float(loss.detach()), "hard_ce": float(loss.detach())}
    log_probs = F.log_softmax(scores, dim=-1)
    soft_loss = -(human_probs * log_probs).sum(dim=-1).mean()
    if mode == "soft_ce_only":
        return soft_loss, {"total": float(soft_loss.detach()), "soft_ce": float(soft_loss.detach())}
    hard_loss = F.cross_entropy(scores, majority)
    predicted_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    target_entropy = -(human_probs * human_probs.clamp_min(1e-8).log()).sum(dim=-1)
    entropy_loss = F.mse_loss(predicted_entropy, target_entropy)
    total = soft_loss + hard_weight * hard_loss + entropy_weight * entropy_loss
    return total, {
        "total": float(total.detach()),
        "soft_ce": float(soft_loss.detach()),
        "hard_ce": float(hard_loss.detach()),
        "entropy_mse": float(entropy_loss.detach()),
    }


def run_eval_loss(model, loader, token_ids, args) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            logits, content = selected_label_logits(model, batch, token_ids)
            loss, _ = objective_loss(
                logits, content["human_probs"], args.target_mode,
                args.hard_weight, args.entropy_weight,
            )
            losses.append(float(loss.cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def evaluate_test(model, loader, token_ids, output_path: Path, fold: int, mode: str) -> int:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits, content = selected_label_logits(model, batch, token_ids)
            probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            human = content["human_probs"].cpu().tolist()
            for image_id, category, q, p in zip(batch["image_id"], batch["category"], probs, human):
                rows.append({
                    "image_id": image_id,
                    "category": category,
                    "fold": fold,
                    "target_mode": mode,
                    "human_probs": p,
                    "predicted_probs": q,
                    "predicted_label": LABELS[max(range(len(q)), key=lambda i: q[i])],
                })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    args = parse_args()
    fold_dir = Path(args.fold_dir)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_output = output_dir / "predictions.jsonl"
    if test_output.exists():
        existing = read_jsonl(test_output)
        if len(existing) == 40 and {row["target_mode"] for row in existing} == {args.target_mode}:
            print(f"SKIP complete fold={args.fold} mode={args.target_mode} n=40", flush=True)
            return
        raise RuntimeError(f"Refusing to overwrite incomplete/unexpected predictions: {test_output}")

    train_path, val_path, test_path = [fold_dir / f"{name}.jsonl" for name in ("train", "val", "test")]
    train_rows, val_rows, test_rows = [read_jsonl(path) for path in (train_path, val_path, test_path)]
    id_sets = [{row["image_id"] for row in rows} for rows in (train_rows, val_rows, test_rows)]
    if any(id_sets[i] & id_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train/validation/test image IDs overlap")
    missing = [row["image_file"] for row in train_rows + val_rows + test_rows if not (image_root / row["image_file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image files; first={missing[0]}")

    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    token_ids = torch.tensor(label_token_ids(processor), dtype=torch.long, device="cuda")
    collator = Collator(processor)
    train_loader = DataLoader(ImageDataset(train_path, image_root), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collator)
    val_loader = DataLoader(ImageDataset(val_path, image_root), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collator)
    test_loader = DataLoader(ImageDataset(test_path, image_root), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collator)
    model = build_model(args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, updates_per_epoch * args.epochs)
    )
    best_val, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    start = time.time()
    args_record = {
        **vars(args),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "precision": "bfloat16",
        "label_order": LABELS,
        "majority_tie_rule": "torch.argmax; first label in LABELS wins exact ties",
        "soft_objective": f"soft_cross_entropy + {args.hard_weight}*plurality_cross_entropy + {args.entropy_weight}*entropy_mse",
        "soft_ce_only_objective": "soft_cross_entropy",
        "majority_objective": "cross_entropy on the training image plurality label",
        "active_objective": {
            "soft": f"soft_cross_entropy + {args.hard_weight}*plurality_cross_entropy + {args.entropy_weight}*entropy_mse",
            "soft_ce_only": "soft_cross_entropy",
            "majority": "cross_entropy on the training image plurality label",
        }[args.target_mode],
        "selection_rule": "validation objective only; test split is not used for checkpoint selection",
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "test_images": len(test_rows),
    }
    (output_dir / "training_args.json").write_text(json.dumps(args_record, indent=2), encoding="utf-8")
    log_path = output_dir / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for step, batch in enumerate(train_loader, start=1):
                logits, content = selected_label_logits(model, batch, token_ids)
                loss, parts = objective_loss(
                    logits, content["human_probs"], args.target_mode,
                    args.hard_weight, args.entropy_weight,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}")
                (loss / args.grad_accum).backward()
                losses.append(parts)
                if step % args.grad_accum == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            val_loss = run_eval_loss(model, val_loader, token_ids, args)
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_objective": sum(row["total"] for row in losses) / len(losses),
                "validation_objective": val_loss,
            }
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(f"fold={args.fold} mode={args.target_mode} {json.dumps(record)}", flush=True)
            if val_loss < best_val - args.min_delta:
                best_val, best_epoch, stale = val_loss, epoch, 0
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                torch.save(best_state, output_dir / "best_trainable_state.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
    if best_state is None:
        raise RuntimeError("No best validation checkpoint was selected")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name].to(device=parameter.device, dtype=parameter.dtype))
    model.save_pretrained(output_dir / "best_adapter")
    processor.save_pretrained(output_dir / "best_adapter")
    n_predictions = evaluate_test(model, test_loader, token_ids, test_output, args.fold, args.target_mode)
    summary = {
        **args_record,
        "best_epoch": best_epoch,
        "best_validation_objective": best_val,
        "test_predictions": n_predictions,
        "elapsed_seconds": time.time() - start,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"COMPLETE fold={args.fold} mode={args.target_mode} n={n_predictions} best_epoch={best_epoch}", flush=True)
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
