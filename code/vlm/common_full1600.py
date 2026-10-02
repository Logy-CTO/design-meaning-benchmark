import base64
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path


EXPECTED_GENERATION_MODEL = "FLUX.2-klein-base-9B"
VALID_INTENTS = ["functionality", "aesthetics", "usability", "symbolism", "unclear"]
VALID_RECOGNIZED = ["yes", "partial", "no"]
SCORE_FIELDS = [
    "image_quality",
    "target_category_recognizability",
    "confidence",
    "function_aesthetic_score",
    "form_function_consistency",
]
TEXT_FIELDS = [
    "ambiguity_reason",
    "function",
    "semantic_meaning",
    "emotion",
    "cognitive_interpretation",
    "style_period",
]


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_root() -> Path:
    return package_root() / "outputs" / "frontier"


def images_root() -> Path:
    return package_root() / "images"


def prompt_path() -> Path:
    return package_root() / "code" / "schema" / "s2_8plus5.py"


def manifest_path() -> Path:
    return package_root() / "data" / "master_image_manifest.csv"


def load_prompt_module():
    path = prompt_path()
    spec = importlib.util.spec_from_file_location("canonical_s2_8plus5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prompt module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = module.PROMPT_TEMPLATE
    return module.PROMPT_KEY, template, hashlib.sha256(template.encode("utf-8")).hexdigest()


def read_manifest(check_images: bool = True):
    path = manifest_path()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1600:
        raise SystemExit(f"Refusing to run: manifest row count is {len(rows)}, expected 1600: {path}")
    bad = [row for row in rows if row.get("generation_model") != EXPECTED_GENERATION_MODEL]
    if bad:
        raise SystemExit(
            "Refusing to run: manifest contains non-9B rows. "
            f"expected={EXPECTED_GENERATION_MODEL}, first_bad={bad[0].get('image_id')}"
        )
    if check_images:
        missing = []
        for row in rows:
            image_path = image_path_for(row)
            if not image_path.exists():
                missing.append(str(image_path))
        if missing:
            sample = "\n".join(missing[:5])
            raise SystemExit(f"Refusing to run: missing {len(missing)} image files. First files:\n{sample}")
    return rows


def image_path_for(row):
    rel = str(row["image_path"]).replace("\\", "/")
    if rel.startswith("images/"):
        rel = rel[len("images/") :]
    return images_root() / rel


def image_to_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text.lower()).strip("_")


def extract_json_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return match.group(0) if match else cleaned


def parse_json_text(text: str):
    return json.loads(extract_json_text(text))


def normalize_intent(value):
    text = str(value or "").strip().lower()
    aliases = {
        "a": "aesthetics",
        "aes": "aesthetics",
        "aesthetic": "aesthetics",
        "f": "functionality",
        "func": "functionality",
        "function": "functionality",
        "functional": "functionality",
        "u": "usability",
        "use": "usability",
        "usable": "usability",
        "s": "symbolism",
        "sym": "symbolism",
        "symbolic": "symbolism",
        "ambiguous": "unclear",
        "unknown": "unclear",
        "uncertain": "unclear",
        "not clear": "unclear",
    }
    text = aliases.get(text, text)
    if text in VALID_INTENTS:
        return text
    for label in VALID_INTENTS:
        if label in text:
            return label
    return "unclear"


def normalize_recognized(value):
    text = str(value or "").strip().lower()
    aliases = {
        "true": "yes",
        "y": "yes",
        "partially": "partial",
        "partial/ambiguous": "partial",
        "n": "no",
        "false": "no",
    }
    text = aliases.get(text, text)
    if text in VALID_RECOGNIZED:
        return text
    if "partial" in text or "ambiguous" in text:
        return "partial"
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return "partial"


def normalize_score(value):
    try:
        score = int(round(float(value)))
    except Exception:
        return ""
    return max(1, min(5, score))


def normalize_parsed(parsed):
    parsed = dict(parsed or {})
    parsed["recognized_as_product"] = normalize_recognized(parsed.get("recognized_as_product"))
    parsed["design_intent"] = normalize_intent(parsed.get("design_intent"))
    for field in SCORE_FIELDS:
        parsed[field] = normalize_score(parsed.get(field))
    for field in TEXT_FIELDS:
        parsed[field] = str(parsed.get(field, "")).strip()
    if not parsed["ambiguity_reason"]:
        parsed["ambiguity_reason"] = "none"
    return parsed


def load_completed(jsonl_path: Path):
    completed = {}
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = row.get("parsed") or {}
            meta = row.get("meta") or {}
            if (
                row.get("image_id")
                and not row.get("error")
                and parsed.get("design_intent") in VALID_INTENTS
                and meta.get("generation_model") == EXPECTED_GENERATION_MODEL
            ):
                completed[row["image_id"]] = row
    return completed


def write_jsonl_row(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_outputs(records, json_path: Path, csv_path: Path):
    records = sorted(records, key=lambda r: int((r.get("meta") or {}).get("global_index", 10**9)))
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for record in records:
        meta = record.get("meta") or {}
        parsed = record.get("parsed") or {}
        row = {
            "global_index": meta.get("global_index", ""),
            "image_id": record.get("image_id", ""),
            "category": meta.get("category", ""),
            "product": meta.get("product", ""),
            "image_file": meta.get("image_file", ""),
            "generation_model": meta.get("generation_model", ""),
            "resolution": meta.get("resolution", ""),
            "vlm_provider": record.get("provider", ""),
            "vlm_model": record.get("model", ""),
            "prompt_key": record.get("prompt_key", ""),
            "prompt_sha256": record.get("prompt_sha256", ""),
            "design_intent": parsed.get("design_intent", ""),
            "recognized_as_product": parsed.get("recognized_as_product", ""),
            "image_quality": parsed.get("image_quality", ""),
            "target_category_recognizability": parsed.get("target_category_recognizability", ""),
            "confidence": parsed.get("confidence", ""),
            "function_aesthetic_score": parsed.get("function_aesthetic_score", ""),
            "form_function_consistency": parsed.get("form_function_consistency", ""),
            "ambiguity_reason": parsed.get("ambiguity_reason", ""),
            "function": parsed.get("function", ""),
            "semantic_meaning": parsed.get("semantic_meaning", ""),
            "emotion": parsed.get("emotion", ""),
            "cognitive_interpretation": parsed.get("cognitive_interpretation", ""),
            "style_period": parsed.get("style_period", ""),
            "elapsed_sec": record.get("elapsed_sec", ""),
            "error": record.get("error") or "",
        }
        rows.append(row)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def attach_global_index(rows):
    for idx, row in enumerate(rows, 1):
        row["global_index"] = idx
    return rows


def dry_run_report(model_name: str, limit: int):
    rows = attach_global_index(read_manifest(check_images=False))
    prompt_key, template, prompt_sha = load_prompt_module()
    categories = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    print(f"DRY_RUN_OK model={model_name}")
    print(f"manifest={manifest_path()}")
    print(f"image_root={images_root()}")
    print(f"rows={len(rows)} limit={limit}")
    print(f"generation_model={EXPECTED_GENERATION_MODEL}")
    print(f"prompt_key={prompt_key}")
    print(f"prompt_sha256={prompt_sha}")
    print("categories=" + json.dumps(categories, ensure_ascii=False, sort_keys=True))
    sample = rows[0]
    print("sample_image_id=" + sample["image_id"])
    print("sample_prompt_first_line=" + template.format(product=sample["product"]).splitlines()[0])
    return rows, prompt_key, template, prompt_sha
