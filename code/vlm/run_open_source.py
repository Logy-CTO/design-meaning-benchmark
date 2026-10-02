from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT / "code" / "schema"))
from s2_8plus5 import PROMPT_KEY, PROMPT_TEMPLATE  # noqa: E402

MANIFEST_CSV = PACKAGE_ROOT / "data" / "master_image_manifest.csv"
IMAGE_ROOT = PACKAGE_ROOT / "images"
OUT_DIR = PACKAGE_ROOT / "outputs" / "open_source"
LOG_DIR = PACKAGE_ROOT / "outputs" / "logs"

INTENT_LABELS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
RECOGNIZED_LABELS = ["yes", "partial", "no"]

MODEL_CONFIGS = {
    "qwen25_vl_7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "Qwen2.5-VL",
        "scale": "7B",
        "release_year": "2025",
        "loader": "qwen",
        "precision": "bf16",
    },
    "internvl3_8b": {
        "model_id": "OpenGVLab/InternVL3-8B",
        "family": "InternVL3",
        "scale": "8B",
        "release_year": "2025",
        "loader": "internvl",
        "precision": "bf16",
    },
    "minicpm_v25": {
        "model_id": "openbmb/MiniCPM-Llama3-V-2_5",
        "family": "MiniCPM-V",
        "scale": "8B",
        "release_year": "2024",
        "loader": "minicpm",
        "precision": "fp16",
    },
    "llava15_7b": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "family": "LLaVA",
        "scale": "7B",
        "release_year": "2023",
        "loader": "llava",
        "precision": "fp16",
    },
}

PROMPT = PROMPT_TEMPLATE


EXPECTED_FIELDS = [
    "recognized_as_product",
    "image_quality",
    "target_category_recognizability",
    "confidence",
    "design_intent",
    "function_aesthetic_score",
    "form_function_consistency",
    "ambiguity_reason",
    "function",
    "semantic_meaning",
    "emotion",
    "cognitive_interpretation",
    "cognitive_interpret",
    "style_period",
]


def loose_extract_fields(text: str) -> dict[str, Any]:
    """Recover key-value fields from malformed JSON-like VLM output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{(.*)\}", cleaned, flags=re.DOTALL)
    body = match.group(1) if match else cleaned
    parsed: dict[str, Any] = {}
    for key in EXPECTED_FIELDS:
        pattern = rf'"{re.escape(key)}"\s*:\s*"?(.+?)(?=,?\s*\n\s*"[^"]+"\s*:|\s*\n\s*\}}|\s*\}}|\Z)'
        m = re.search(pattern, body, flags=re.DOTALL)
        if not m:
            continue
        value = m.group(1).strip().rstrip(",").strip().strip('"').strip()
        if key in {
            "image_quality",
            "target_category_recognizability",
            "confidence",
            "function_aesthetic_score",
            "form_function_consistency",
        }:
            nm = re.search(r"-?\d+(?:\.\d+)?", value)
            parsed[key] = float(nm.group(0)) if nm else value
        else:
            parsed[key] = value
    if "cognitive_interpret" in parsed and "cognitive_interpretation" not in parsed:
        parsed["cognitive_interpretation"] = parsed["cognitive_interpret"]
    return parsed


def extract_json(text: str) -> tuple[dict[str, Any], str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    cleaned = cleaned.replace("\\_", "_")
    try:
        return json.loads(cleaned), ""
    except json.JSONDecodeError as exc:
        recovered = loose_extract_fields(text)
        if recovered.get("design_intent") or recovered.get("function") or recovered.get("semantic_meaning"):
            return recovered, f"recovered_loose_json: {exc}"
        return {}, str(exc)


def normalize_intent(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "a": "aesthetics",
        "function": "functionality",
        "f": "functionality",
        "functional": "functionality",
        "aesthetic": "aesthetics",
        "aesthetic/symbolic": "aesthetics",
        "s": "symbolism",
        "symbolic": "symbolism",
        "u": "usability",
        "use": "usability",
        "usable": "usability",
        "ambiguous": "unclear",
        "uncertain": "unclear",
        "unknown": "unclear",
        "n/a": "unclear",
    }
    text = aliases.get(text, text)
    for label in INTENT_LABELS:
        if label == text:
            return label
    for label in INTENT_LABELS:
        if label in text:
            return label
    return "unclear"


def normalize_recognized(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "true": "yes",
        "mostly": "yes",
        "partially": "partial",
        "somewhat": "partial",
        "false": "no",
    }
    text = aliases.get(text, text)
    return text if text in RECOGNIZED_LABELS else "partial"


def clip_rating(value: Any) -> int | None:
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def normalize_eval(parsed: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "recognized_as_product": ["recognized_as_product", "recognized", "product_visible", "is_product"],
        "image_quality": ["image_quality", "quality"],
        "target_category_recognizability": [
            "target_category_recognizability",
            "target_category_recognisability",
            "product_recognizability",
            "category_recognizability",
        ],
        "confidence": ["confidence", "judgment_confidence"],
        "design_intent": ["design_intent", "DesignIntent", "intent"],
        "function_aesthetic_score": ["function_aesthetic_score", "FunctionAestheticScore", "function_aesthetic_direction"],
        "form_function_consistency": ["form_function_consistency", "FormFunctionConsistency", "form_function"],
        "ambiguity_reason": ["ambiguity_reason", "reason", "uncertainty_reason"],
        "function": ["function", "Function"],
        "semantic_meaning": ["semantic_meaning", "SemanticMeaning"],
        "emotion": ["emotion", "Emotion"],
        "cognitive_interpretation": ["cognitive_interpretation", "CognitiveInterpretation"],
        "style_period": ["style_period", "StylePeriod"],
    }
    out: dict[str, Any] = {}
    for target, keys in aliases.items():
        out[target] = next((parsed.get(k) for k in keys if k in parsed), "")
    out["recognized_as_product"] = normalize_recognized(out["recognized_as_product"])
    out["design_intent"] = normalize_intent(out["design_intent"])
    for key in [
        "image_quality",
        "target_category_recognizability",
        "confidence",
        "function_aesthetic_score",
        "form_function_consistency",
    ]:
        out[key] = clip_rating(out[key])
    out["ambiguity_reason"] = str(out.get("ambiguity_reason") or "none").strip() or "none"
    return out


def load_manifest(category: str | None, limit: int | None) -> list[dict[str, Any]]:
    with MANIFEST_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if category:
        rows = [row for row in rows if row["category"] == category]
    rows = sorted(rows, key=lambda r: (r["set"], r["category"], int(r["sample_index"])))
    if limit is not None:
        rows = rows[:limit]
    return rows


def output_paths(model_key: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{model_key}_{PROMPT_KEY}_1600_main4_8b"
    return OUT_DIR / f"{stem}.json", OUT_DIR / f"{stem}.csv"


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["set"], r["category"], int(r["sample_index"])))
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "model_key", "model_id", "family", "scale", "release_year", "precision",
        "image_id", "set", "category", "product", "sample_index", "seed", "prompt",
        "image_file", "image_path", "recognized_as_product", "valid_recognized_as_product",
        "image_quality", "target_category_recognizability", "confidence", "design_intent",
        "valid_design_intent", "function_aesthetic_score", "form_function_consistency",
        "ambiguity_reason", "function", "semantic_meaning", "emotion",
        "cognitive_interpretation", "style_period",
        "raw_response", "parse_error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ev = row.get("evaluation", {})
            out = {key: row.get(key, "") for key in fields}
            for key in [
                "function", "semantic_meaning", "design_intent", "emotion",
                "function_aesthetic_score", "form_function_consistency",
                "cognitive_interpretation", "style_period",
                "recognized_as_product", "image_quality", "target_category_recognizability",
                "confidence", "ambiguity_reason",
            ]:
                out[key] = ev.get(key, "")
            out["valid_recognized_as_product"] = ev.get("recognized_as_product", "") in RECOGNIZED_LABELS
            out["valid_design_intent"] = ev.get("design_intent", "") in INTENT_LABELS
            writer.writerow(out)


class QwenEvaluator:
    def __init__(self, model_id: str):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def __call__(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


class InternVLEvaluator:
    def __init__(self, model_id: str):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
        self.transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def __call__(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
        return self.model.chat(self.tokenizer, pixel_values, prompt, dict(max_new_tokens=max_new_tokens, do_sample=False)).strip()


class MiniCPMEvaluator:
    def __init__(self, model_id: str):
        from transformers import AutoModel, AutoTokenizer

        if not hasattr(torch.nn.Module, "_initialize_weights"):
            torch.nn.Module._initialize_weights = lambda self, module: None  # type: ignore[attr-defined]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        ).eval()

    @torch.inference_mode()
    def __call__(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [image, prompt]}]
        return self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, sampling=False, max_new_tokens=max_new_tokens).strip()


class LlavaEvaluator:
    def __init__(self, model_id: str):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        if not hasattr(self.processor, "patch_size"):
            self.processor.patch_size = getattr(self.model.config.vision_config, "patch_size", 14)
        if not hasattr(self.processor, "vision_feature_select_strategy"):
            self.processor.vision_feature_select_strategy = getattr(self.model.config, "vision_feature_select_strategy", "default")

    @torch.inference_mode()
    def __call__(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        text = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        return decoded.split("ASSISTANT:", 1)[-1].strip() if "ASSISTANT:" in decoded else decoded.strip()


def build_evaluator(config: dict[str, str]):
    loader = config["loader"]
    model_id = config["model_id"]
    if loader == "qwen":
        return QwenEvaluator(model_id)
    if loader == "internvl":
        return InternVLEvaluator(model_id)
    if loader == "minicpm":
        return MiniCPMEvaluator(model_id)
    if loader == "llava":
        return LlavaEvaluator(model_id)
    raise ValueError(loader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config = MODEL_CONFIGS[args.model_key]
    rows = load_manifest(args.category, args.limit)
    json_path, csv_path = output_paths(args.model_key)
    existing = load_existing(json_path)
    done = {r["image_id"] for r in existing}
    output = existing[:]
    evaluator = build_evaluator(config)

    for idx, row in enumerate(rows, start=1):
        if row["image_id"] in done:
            print(f"Skipping existing {args.model_key} {row['image_id']}", flush=True)
            continue
        # Manifest paths were authored on Windows; normalize separators for
        # Linux execution on the 126 server.
        rel_image_path = str(row["image_path"]).replace("\\", "/")
        if rel_image_path.startswith("images/"):
            rel_image_path = rel_image_path[len("images/") :]
        image_path = str(IMAGE_ROOT / Path(rel_image_path))
        prompt = PROMPT.format(product=row["product"])
        print(f"Evaluating {args.model_key} {row['image_id']} ({idx}/{len(rows)})", flush=True)
        raw = evaluator(image_path, prompt, args.max_new_tokens)
        parsed, parse_error = extract_json(raw)
        ev = normalize_eval(parsed) if parsed else {}
        output.append({
            "model_key": args.model_key,
            "model_id": config["model_id"],
            "family": config["family"],
            "scale": config["scale"],
            "release_year": config["release_year"],
            "precision": config["precision"],
            "prompt_key": PROMPT_KEY,
            "image_id": row["image_id"],
            "set": row["set"],
            "category": row["category"],
            "product": row["product"],
            "sample_index": int(row["sample_index"]),
            "seed": row["seed"],
            "prompt": row["prompt"],
            "image_file": row["image_file"],
            "image_path": row["image_path"],
            "evaluation": ev,
            "raw_response": raw,
            "parse_error": parse_error,
        })
        done.add(row["image_id"])
        save_rows(output, json_path, csv_path)

    save_rows(output, json_path, csv_path)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
