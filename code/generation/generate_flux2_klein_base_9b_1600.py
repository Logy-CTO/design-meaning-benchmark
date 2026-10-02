from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT
IMAGE_DIR = OUT_DIR / "images"
DATA_DIR = OUT_DIR / "data"
LOG_DIR = OUT_DIR / "logs"

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
GENERATION_MODEL_LABEL = "FLUX.2-klein-base-9B"
BASE_SEED = 20261504
NUM_IMAGES_PER_CATEGORY = 200
WIDTH = 768
HEIGHT = 768
STEPS = 12
GUIDANCE_SCALE = 3.5


@dataclass(frozen=True)
class Category:
    slug: str
    product: str
    prompt: str
    set_name: str
    seed_offset: int


CATEGORIES = [
    Category("early_automobile", "Early automobile", "early automobile", "contrast_automobile_historical", 0),
    Category("sports_car", "Sports car", "sports car", "contrast_automobile_contemporary", 8),
    Category("broom", "Broom", "broom", "contrast_cleaning_manual", 9),
    Category("vacuum_cleaner", "Vacuum cleaner", "vacuum cleaner", "contrast_cleaning_electric", 1),
    Category("electric_kettle", "Electric kettle", "electric kettle", "single_appliance", 2),
    Category("lounge_chair", "Lounge chair", "lounge chair", "single_furniture", 3),
    Category("desk_lamp", "Desk lamp", "desk lamp", "single_lighting", 4),
    Category("sneakers", "Sneakers", "sneakers", "single_wearable", 6),
]


def ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_rows() -> list[dict[str, Any]]:
    path = DATA_DIR / "master_image_manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(rows: list[dict[str, Any]]) -> None:
    order = {cfg.slug: i for i, cfg in enumerate(CATEGORIES)}
    rows = sorted(rows, key=lambda r: (order.get(r["category"], 999), int(r["sample_index"])))
    json_path = DATA_DIR / "master_image_manifest.json"
    csv_path = DATA_DIR / "master_image_manifest.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "image_id",
        "set",
        "category",
        "product",
        "sample_index",
        "seed",
        "prompt",
        "image_file",
        "image_path",
        "generation_model",
        "resolution",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_summary(rows: list[dict[str, Any]]) -> None:
    counts = {cfg.slug: 0 for cfg in CATEGORIES}
    for row in rows:
        if row["category"] in counts:
            counts[row["category"]] += 1
    lines = [
        "# FLUX.2 [klein] Base 9B Product Image Library",
        "",
        f"- Model ID: `{MODEL_ID}`",
        f"- Generator label: `{GENERATION_MODEL_LABEL}`",
        f"- Resolution: `{WIDTH}x{HEIGHT}`",
        f"- Inference steps: `{STEPS}`",
        f"- Guidance scale: `{GUIDANCE_SCALE}`",
        f"- Images per category: `{NUM_IMAGES_PER_CATEGORY}`",
        f"- Total target: `{len(CATEGORIES) * NUM_IMAGES_PER_CATEGORY}`",
        "",
        "## Category Counts",
        "",
        "| Category | Product | Prompt | Set | Count |",
        "|---|---|---|---|---:|",
    ]
    for cfg in CATEGORIES:
        lines.append(f"| `{cfg.slug}` | {cfg.product} | `{cfg.prompt}` | `{cfg.set_name}` | {counts[cfg.slug]} |")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def load_generation_pipeline() -> Flux2KleinPipeline:
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def generate_image(pipe: Flux2KleinPipeline, prompt: str, seed: int) -> Image.Image:
    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        generator=generator,
        guidance_scale=GUIDANCE_SCALE,
    )
    return result.images[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Generate only N missing images for testing/chunking.")
    parser.add_argument("--category", default=None, help="Generate only one category slug.")
    args = parser.parse_args()

    ensure_dirs()
    rows = load_rows()
    done = {row["image_id"] for row in rows}
    made = 0
    pipe = load_generation_pipeline()

    for cfg in CATEGORIES:
        if args.category and cfg.slug != args.category:
            continue
        for sample_idx in range(1, NUM_IMAGES_PER_CATEGORY + 1):
            image_id = f"{cfg.slug}_{sample_idx:03d}"
            if image_id in done and (IMAGE_DIR / f"{image_id}.png").exists():
                print(f"Skipping existing {image_id}", flush=True)
                continue
            if args.limit is not None and made >= args.limit:
                save_rows(rows)
                write_summary(rows)
                print(f"Limit reached. Rows: {len(rows)}", flush=True)
                return
            seed = BASE_SEED + cfg.seed_offset * 1000 + (sample_idx - 1)
            image_file = f"{image_id}.png"
            image_path = IMAGE_DIR / image_file
            print(f"Generating {cfg.slug} {sample_idx}/{NUM_IMAGES_PER_CATEGORY} seed={seed}", flush=True)
            image = generate_image(pipe, cfg.prompt, seed)
            image.save(image_path)
            rows = [row for row in rows if row.get("image_id") != image_id]
            rows.append(
                {
                    "image_id": image_id,
                    "set": cfg.set_name,
                    "category": cfg.slug,
                    "product": cfg.product,
                    "sample_index": sample_idx,
                    "seed": seed,
                    "prompt": cfg.prompt,
                    "image_file": image_file,
                    "image_path": str(Path("images") / image_file),
                    "generation_model": GENERATION_MODEL_LABEL,
                    "resolution": f"{WIDTH}x{HEIGHT}",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "num_inference_steps": STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                }
            )
            done.add(image_id)
            made += 1
            save_rows(rows)
            write_summary(rows)

    save_rows(rows)
    write_summary(rows)
    print(f"Done. Rows: {len(rows)} Output: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
