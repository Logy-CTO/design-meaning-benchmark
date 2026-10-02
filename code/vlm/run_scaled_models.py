from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CACHE_ROOT = Path(
    os.environ.get(
        "AAAI_CACHE_ROOT",
        Path(__file__).resolve().parents[2] / ".cache",
    )
)
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("TMPDIR", str(CACHE_ROOT / "tmp"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("TRITON_CACHE_DIR", str(CACHE_ROOT / "triton"))
os.environ.setdefault("PIP_CACHE_DIR", str(CACHE_ROOT / "pip"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("CUDA_CACHE_PATH", str(CACHE_ROOT / "cuda"))

IMAGE_ROOT_DIR = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "scaled"
OUT_DIR = WORK_ROOT / "results"
LOG_DIR = WORK_ROOT / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical_eval_utils import (  # noqa: E402
    IMAGE_ROOT,
    PROMPT,
    extract_json,
    load_manifest,
    normalize_eval,
)
import torch  # noqa: E402
from PIL import Image  # noqa: E402

PROMPT_KEY = "s2_descriptive_fields_first_8plus5"
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
EXPECTED_GENERATION_MODEL = "FLUX.2-klein-base-9B"
SOURCE_MANIFEST = str(Path(__file__).resolve().parents[2] / "data" / "master_image_manifest.csv")

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "qwen25_vl_32b_4bit": {
        "model_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "family": "Qwen2.5-VL",
        "scale": "32B",
        "release_year": "2025",
        "loader": "qwen",
        "precision": "bnb4-nf4-bf16",
        "output_stem": "qwen25_vl_32b_4bit_s2_full_1600",
    },
    "qwen25_vl_72b_4bit": {
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "family": "Qwen2.5-VL",
        "scale": "72B",
        "release_year": "2025",
        "loader": "qwen",
        "precision": "bnb4-nf4-bf16",
        "output_stem": "qwen25_vl_72b_4bit_s2_full_1600",
    },
    "internvl3_38b_bf16": {
        "model_id": "OpenGVLab/InternVL3-38B",
        "family": "InternVL3",
        "scale": "38B",
        "release_year": "2025",
        "loader": "internvl",
        "precision": "bf16",
        "output_stem": "internvl3_38b_bf16_s2_full_1600",
    },
}


def bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


class BigQwenEvaluator:
    def __init__(self, model_id: str):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config(),
            low_cpu_mem_usage=True,
            local_files_only=False,
        ).eval()

    @torch.inference_mode()
    def __call__(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


class BigInternVLEvaluator:
    def __init__(self, model_id: str, precision: str):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        self.image_size = 448
        self.max_num = 12
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False, local_files_only=False
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "local_files_only": False,
        }
        if precision != "bf16":
            load_kwargs["quantization_config"] = bnb_config()
        self.model = AutoModel.from_pretrained(model_id, **load_kwargs).eval()
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _dynamic_preprocess(self, image, image_size=448, max_num=12, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = sorted(
            {
                (i, j)
                for n in range(1, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if 1 <= i * j <= max_num
            },
            key=lambda x: x[0] * x[1],
        )
        best_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - (r[0] / r[1])))
        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        resized_img = image.resize((target_width, target_height))
        grid_w = target_width // image_size
        processed_images = []
        for i in range(best_ratio[0] * best_ratio[1]):
            box = (
                (i % grid_w) * image_size,
                (i // grid_w) * image_size,
                ((i % grid_w) + 1) * image_size,
                ((i // grid_w) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    @torch.inference_mode()
    def __call__(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        images = self._dynamic_preprocess(image, image_size=self.image_size, max_num=self.max_num, use_thumbnail=True)
        pixel_values = torch.stack([self.transform(img) for img in images]).to(torch.bfloat16).cuda()
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            "<image>\n" + prompt,
            {"max_new_tokens": max_new_tokens, "do_sample": False},
        ).strip()


def build_evaluator(config: dict[str, str]):
    if config["loader"] == "qwen":
        return BigQwenEvaluator(config["model_id"])
    if config["loader"] == "internvl":
        return BigInternVLEvaluator(config["model_id"], config["precision"])
    raise ValueError(config["loader"])


def output_paths(config: dict[str, str]) -> tuple[Path, Path, Path]:
    stem = config["output_stem"]
    return OUT_DIR / f"{stem}.json", OUT_DIR / f"{stem}.csv", OUT_DIR / f"{stem}.jsonl"


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_outputs(rows: list[dict[str, Any]], json_path: Path, csv_path: Path, jsonl_path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["set"], r["category"], int(r["sample_index"])))
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = [
        "model_key",
        "model_id",
        "family",
        "scale",
        "release_year",
        "precision",
        "prompt_key",
        "prompt_sha256",
        "generation_model",
        "expected_generation_model",
        "image_id",
        "set",
        "category",
        "product",
        "sample_index",
        "seed",
        "prompt",
        "image_file",
        "image_path",
        "design_intent",
        "recognized_as_product",
        "image_quality",
        "target_category_recognizability",
        "confidence",
        "function_aesthetic_score",
        "form_function_consistency",
        "ambiguity_reason",
        "function",
        "semantic_meaning",
        "emotion",
        "cognitive_interpretation",
        "style_period",
        "parse_error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ev = row.get("evaluation", {})
            out = {key: row.get(key, "") for key in fields}
            for key in [
                "design_intent",
                "recognized_as_product",
                "image_quality",
                "target_category_recognizability",
                "confidence",
                "function_aesthetic_score",
                "form_function_consistency",
                "ambiguity_reason",
                "function",
                "semantic_meaning",
                "emotion",
                "cognitive_interpretation",
                "style_period",
            ]:
                out[key] = ev.get(key, "")
            writer.writerow(out)


def run_one(model_key: str, limit: int | None, per_category: int | None, max_new_tokens: int) -> None:
    config = MODEL_CONFIGS[model_key]
    rows = load_manifest(category=None, limit=limit, per_category=per_category)
    json_path, csv_path, jsonl_path = output_paths(config)
    existing = load_existing(json_path)
    done = {row["image_id"] for row in existing}
    output = existing[:]
    print(f"[{datetime.now()}] loading {model_key}: {config['model_id']}", flush=True)
    evaluator = build_evaluator(config)
    print(f"[{datetime.now()}] loaded {model_key}; rows={len(rows)} existing={len(existing)}", flush=True)

    for idx, row in enumerate(rows, start=1):
        if row["image_id"] in done:
            continue
        if str(row.get("generation_model", "")) != EXPECTED_GENERATION_MODEL:
            raise RuntimeError(f"Non-9B image in manifest: {row['image_id']} {row.get('generation_model')}")
        rel_path = str(row["image_path"]).replace("\\", "/")
        if rel_path.startswith("images/"):
            rel_path = rel_path[len("images/") :]
        image_path = str(IMAGE_ROOT / rel_path)
        prompt = PROMPT.format(product=row["product"])
        print(f"[{datetime.now()}] {model_key} {row['image_id']} ({idx}/{len(rows)})", flush=True)
        raw = evaluator(image_path, prompt, max_new_tokens=max_new_tokens)
        parsed, parse_error = extract_json(raw)
        ev = normalize_eval(parsed) if parsed else {}
        output.append(
            {
                "model_key": model_key,
                "model_id": config["model_id"],
                "family": config["family"],
                "scale": config["scale"],
                "release_year": config["release_year"],
                "precision": config["precision"],
                "prompt_key": PROMPT_KEY,
                "prompt_sha256": PROMPT_SHA256,
                "generation_model": row.get("generation_model", EXPECTED_GENERATION_MODEL),
                "expected_generation_model": EXPECTED_GENERATION_MODEL,
                "source_manifest": SOURCE_MANIFEST,
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
            }
        )
        done.add(row["image_id"])
        write_outputs(output, json_path, csv_path, jsonl_path)
    write_outputs(output, json_path, csv_path, jsonl_path)
    print(f"[{datetime.now()}] wrote {json_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()
    keys = [args.model_key] if args.model_key else list(MODEL_CONFIGS)
    if not args.all and not args.model_key:
        raise SystemExit("Use --model-key MODEL or --all")
    for key in keys:
        run_one(key, args.limit, args.per_category, args.max_new_tokens)


if __name__ == "__main__":
    main()
