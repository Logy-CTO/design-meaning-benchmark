from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


LABELS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
LABEL_FORMS = [" functionality", " aesthetics", " usability", " symbolism", " unclear"]
SYSTEM_TEXT = "You are a careful design-evaluation annotator. Judge the visible image evidence."


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def classifier_prompt(row: dict) -> str:
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


class DistributionDataset(Dataset):
    def __init__(self, path: Path, image_root: Path):
        self.rows = load_jsonl(path)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        messages = [
            {"role": "system", "content": SYSTEM_TEXT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str((self.image_root / row["image_file"]).resolve())},
                    {"type": "text", "text": classifier_prompt(row)},
                ],
            },
        ]
        return {
            "image_id": row["image_id"],
            "messages": messages,
            "human_probs": [float(row[f"human_p_{label}"]) for label in LABELS],
        }


@dataclass
class DistributionCollator:
    processor: object

    def __call__(self, batch: list[dict]) -> dict:
        texts, images = [], []
        for item in batch:
            texts.append(
                self.processor.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=True
                )
            )
            image_inputs, _ = process_vision_info(item["messages"])
            images.append(image_inputs[0])
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs["human_probs"] = torch.tensor(
            [item["human_probs"] for item in batch], dtype=torch.float32
        )
        return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hard_weight", type=float, default=0.10)
    parser.add_argument("--entropy_weight", type=float, default=0.10)
    return parser.parse_args()


def label_token_ids(processor) -> list[int]:
    ids = []
    for form in LABEL_FORMS:
        encoded = processor.tokenizer.encode(form, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Expected one token for {form!r}, got {encoded}")
        ids.append(encoded[0])
    return ids


def load_model(args: argparse.Namespace):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
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
    model.print_trainable_parameters()
    return model


def distribution_loss(model, batch: dict, token_ids: torch.Tensor, hard_weight: float, entropy_weight: float):
    targets = batch.pop("human_probs").to(model.device)
    model_inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in batch.items()}
    output = model(**model_inputs, use_cache=False)
    positions = model_inputs["attention_mask"].long().sum(dim=1) - 1
    row_ids = torch.arange(output.logits.shape[0], device=output.logits.device)
    logits = output.logits[row_ids, positions, :].index_select(-1, token_ids)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    soft_loss = -(targets * log_probs).sum(dim=-1).mean()
    hard_loss = F.cross_entropy(logits.float(), targets.argmax(dim=-1))
    probs = log_probs.exp()
    pred_entropy = -(probs * log_probs).sum(dim=-1)
    target_entropy = -(targets * targets.clamp_min(1e-8).log()).sum(dim=-1)
    entropy_loss = F.mse_loss(pred_entropy, target_entropy)
    total = soft_loss + hard_weight * hard_loss + entropy_weight * entropy_loss
    return total, soft_loss.detach(), hard_loss.detach(), entropy_loss.detach()


@torch.no_grad()
def validation_loss(model, loader, token_ids, hard_weight, entropy_weight) -> float:
    model.eval()
    values = []
    for batch in loader:
        total, _, _, _ = distribution_loss(model, batch, token_ids, hard_weight, entropy_weight)
        values.append(float(total.cpu()))
    model.train()
    return sum(values) / max(1, len(values))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    ids = label_token_ids(processor)

    collator = DistributionCollator(processor)
    train_loader = DataLoader(
        DistributionDataset(Path(args.train_jsonl), Path(args.image_root)),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        DistributionDataset(Path(args.val_jsonl), Path(args.image_root)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    model = load_model(args)
    token_ids = torch.tensor(ids, dtype=torch.long, device=model.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_updates = max(1, updates_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)

    best_val, best_epoch, stale, global_step = float("inf"), 0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = []
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
            for step, batch in enumerate(progress, start=1):
                total, soft, hard, entropy = distribution_loss(
                    model, batch, token_ids, args.hard_weight, args.entropy_weight
                )
                (total / args.grad_accum).backward()
                running.append(float(total.detach().cpu()))
                if step % args.grad_accum == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                progress.set_postfix(
                    total=f"{running[-1]:.3f}", soft=f"{float(soft):.3f}",
                    hard=f"{float(hard):.3f}", ent=f"{float(entropy):.3f}"
                )
            val = validation_loss(model, val_loader, token_ids, args.hard_weight, args.entropy_weight)
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": sum(running) / max(1, len(running)),
                "val_distribution_objective": val,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(record)
            if val < best_val - args.min_delta:
                best_val, best_epoch, stale = val, epoch, 0
                model.save_pretrained(output_dir / "best_adapter")
                processor.save_pretrained(output_dir / "best_adapter")
            else:
                stale += 1
                if stale >= args.patience:
                    break

    metadata = {
        **vars(args),
        "best_epoch": best_epoch,
        "best_val_distribution_objective": best_val,
        "labels": LABELS,
        "label_forms": LABEL_FORMS,
        "label_token_ids": ids,
        "vision_lora_frozen": True,
        "objective": "soft_cross_entropy + hard_weight*majority_ce + entropy_weight*entropy_mse",
    }
    (output_dir / "training_args.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
