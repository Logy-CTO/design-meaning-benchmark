from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score, f1_score
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info
from train_distribution_aware import LABELS, LABEL_FORMS, SYSTEM_TEXT, classifier_prompt, load_jsonl


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / p.sum(); q = q / q.sum(); midpoint = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def build_messages(row: dict, image_root: Path) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str((image_root / row["image_file"]).resolve())},
                {"type": "text", "text": classifier_prompt(row)},
            ],
        },
    ]


@torch.no_grad()
def collect_logits(model, processor, rows: list[dict], image_root: Path, token_ids: torch.Tensor):
    collected = []
    for row in tqdm(rows, desc="score labels"):
        messages = build_messages(row, image_root)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        output = model(**inputs, use_cache=False)
        position = int(inputs["attention_mask"].long().sum().item() - 1)
        logits = output.logits[0, position, :].index_select(-1, token_ids).float().cpu().numpy()
        collected.append(logits)
    return np.asarray(collected)


def human_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray([[float(row[f"human_p_{label}"]) for label in LABELS] for row in rows])


def choose_temperature(logits: np.ndarray, humans: np.ndarray) -> tuple[float, float]:
    best = (1.0, float("inf"))
    for temperature in np.geomspace(0.35, 5.0, 121):
        probabilities = softmax(logits / temperature, axis=1)
        score = float(np.mean([jsd(p, q) for p, q in zip(humans, probabilities)]))
        if score < best[1]:
            best = (float(temperature), score)
    return best


def summarize(rows: list[dict], probabilities: np.ndarray, temperature: float) -> tuple[list[dict], dict]:
    humans = human_matrix(rows)
    targets = humans.argmax(axis=1)
    predictions = probabilities.argmax(axis=1)
    outputs = []
    for row, human, predicted, target, pred in zip(rows, humans, probabilities, targets, predictions):
        outputs.append({
            "image_id": row["image_id"],
            "category": row["category"],
            "target_design_intent": LABELS[int(target)],
            "pred_design_intent": LABELS[int(pred)],
            "human_distribution": human.tolist(),
            "predicted_distribution": predicted.tolist(),
            "human_probability_of_prediction": float(human[pred]),
            "soft_jsd": jsd(human, predicted),
            "human_entropy": entropy(human),
            "predicted_entropy": entropy(predicted),
        })
    category_rows = defaultdict(list)
    for index, row in enumerate(rows):
        category_rows[row["category"]].append(index)
    category_jsds = {}
    for category, indices in category_rows.items():
        category_jsds[category] = jsd(humans[indices].mean(axis=0), probabilities[indices].mean(axis=0))
    hard_hass = float(np.mean([humans[i, predictions[i]] for i in range(len(rows))]))
    soft_jsds = np.asarray([jsd(p, q) for p, q in zip(humans, probabilities)])
    entropy_errors = np.abs(np.asarray([entropy(p) for p in humans]) - np.asarray([entropy(q) for q in probabilities]))
    brier = float(np.mean(np.sum((probabilities - humans) ** 2, axis=1)))
    summary = {
        "n": len(rows),
        "temperature": temperature,
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=list(range(5)), average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(targets, predictions, labels=list(range(5)))),
        "hass": hard_hass,
        "soft_jsd": float(soft_jsds.mean()),
        "soft_jsd_alignment": float(1.0 - soft_jsds.mean()),
        "entropy_mae": float(entropy_errors.mean()),
        "brier_to_human_distribution": brier,
        "aggregate_distribution_jsd": jsd(humans.mean(axis=0), probabilities.mean(axis=0)),
        "category_distribution_jsd_mean": float(np.mean(list(category_jsds.values()))),
        "category_distribution_jsds": category_jsds,
        "target_distribution": dict(Counter(LABELS[int(x)] for x in targets)),
        "pred_distribution": dict(Counter(LABELS[int(x)] for x in predictions)),
        "human_aggregate_distribution": dict(zip(LABELS, humans.mean(axis=0).tolist())),
        "predicted_aggregate_distribution_soft": dict(zip(LABELS, probabilities.mean(axis=0).tolist())),
    }
    return outputs, summary


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    token_values = []
    for form in LABEL_FORMS:
        encoded = processor.tokenizer.encode(form, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError((form, encoded))
        token_values.append(encoded[0])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    token_ids = torch.tensor(token_values, dtype=torch.long, device=model.device)
    image_root = Path(args.image_root)
    val_rows = load_jsonl(Path(args.val_jsonl)); test_rows = load_jsonl(Path(args.test_jsonl))
    val_logits = collect_logits(model, processor, val_rows, image_root, token_ids)
    temperature, val_jsd = choose_temperature(val_logits, human_matrix(val_rows))
    test_logits = collect_logits(model, processor, test_rows, image_root, token_ids)
    test_probabilities = softmax(test_logits / temperature, axis=1)
    outputs, summary = summarize(test_rows, test_probabilities, temperature)
    summary["validation_soft_jsd_at_selected_temperature"] = val_jsd
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
