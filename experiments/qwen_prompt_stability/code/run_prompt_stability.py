from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SEEDS = [11, 23, 37, 47, 59]
CONDITIONS = [
    ("canonical_greedy", "canonical_fields_first.txt", False, None),
    ("label_first_greedy", "label_first.txt", False, None),
] + [(f"sample_seed_{seed}", "canonical_fields_first.txt", True, seed) for seed in SEEDS]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def generation_seed(base_seed: int, image_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{image_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=220)
    args = ap.parse_args()
    manifest = list(csv.DictReader((ROOT / "data" / "master_image_manifest.csv").open(encoding="utf-8-sig", newline="")))
    manifest.sort(key=lambda r: (r["set"], r["category"], int(r["sample_index"])))
    if len(manifest) != 200 or len({r["image_id"] for r in manifest}) != 200:
        raise RuntimeError(f"Expected 200 unique images, found {len(manifest)}")
    prompt_texts = {name: (ROOT / "prompts" / name).read_text(encoding="utf-8") for _, name, _, _ in CONDITIONS}
    prompt_hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in prompt_texts.items()}
    image_root = ROOT / "images" / "human_subset_200"
    image_hashes = {row["image_id"]: sha256(image_root / row["image_file"]) for row in manifest}
    metadata = {
        "experiment": "Qwen2.5-VL-7B prompt-order and stochastic-decoding stability audit",
        "model_id": MODEL_ID, "model_revision": REVISION, "precision": "bfloat16",
        "max_new_tokens": args.max_new_tokens,
        "conditions": [{"name": c, "prompt_file": pf, "do_sample": sample, "seed": seed,
                        "temperature": 0.7 if sample else None, "top_p": 0.9 if sample else None}
                       for c, pf, sample, seed in CONDITIONS],
        "images": len(manifest), "categories": 8,
        "selection": "sample_index 1-25 per category; fixed subset, selection rationale absent in archive",
        "prompt_sha256": prompt_hashes, "image_sha256": image_hashes,
        "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"},
        "sampling_seed_rule": "first 64 bits of SHA-256(base_seed|image_id), modulo 2^63-1, before each image generation",
        "status": "running"
    }
    (ROOT / "outputs" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    out_path = ROOT / "outputs" / "predictions.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["image_id"]))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    start = time.time()
    for condition, prompt_file, do_sample, base_seed in CONDITIONS:
        template = prompt_texts[prompt_file]
        for ix, row in enumerate(manifest, start=1):
            key = (condition, row["image_id"])
            if key in done:
                print(f"SKIP {condition} {row['image_id']}", flush=True)
                continue
            img_path = image_root / row["image_file"]
            image = Image.open(img_path).convert("RGB")
            prompt = template.replace("<TARGET_PRODUCT_CATEGORY>", row["product"])
            message = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            chat = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt").to(model.device)
            kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": do_sample}
            seed = None
            if do_sample:
                seed = generation_seed(base_seed, row["image_id"])
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                kwargs.update({"temperature": 0.7, "top_p": 0.9})
            with torch.inference_mode():
                generated = model.generate(**inputs, **kwargs)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            rec = {
                "condition": condition, "image_id": row["image_id"], "category": row["category"],
                "product": row["product"], "sample_index": int(row["sample_index"]),
                "model_id": MODEL_ID, "model_revision": REVISION, "precision": "bfloat16",
                "do_sample": do_sample, "base_seed": base_seed, "per_image_seed": seed,
                "temperature": 0.7 if do_sample else None, "top_p": 0.9 if do_sample else None,
                "max_new_tokens": args.max_new_tokens, "prompt_file": prompt_file,
                "prompt_sha256": prompt_hashes[prompt_file], "image_sha256": image_hashes[row["image_id"]],
                "raw_response": raw
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            done.add(key)
            print(f"DONE {condition} {row['image_id']} ({ix}/200)", flush=True)
    metadata["status"] = "completed"
    metadata["elapsed_seconds"] = round(time.time() - start, 2)
    metadata["output_rows"] = len(done)
    (ROOT / "outputs" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"COMPLETE rows={len(done)} elapsed={metadata['elapsed_seconds']:.1f}s", flush=True)

if __name__ == "__main__":
    main()
