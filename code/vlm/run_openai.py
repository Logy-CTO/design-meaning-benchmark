import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from common_full1600 import (
    EXPECTED_GENERATION_MODEL,
    SCORE_FIELDS,
    VALID_INTENTS,
    VALID_RECOGNIZED,
    attach_global_index,
    dry_run_report,
    image_path_for,
    image_to_data_url,
    load_completed,
    load_prompt_module,
    normalize_parsed,
    read_manifest,
    run_root,
    safe_slug,
    write_jsonl_row,
    write_outputs,
)


SYSTEM_PROMPT = (
    "You evaluate generated product images for a design-evaluation task. "
    "Use only visible evidence in the image plus the target category only for recognizability. "
    "Return only valid JSON."
)

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
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
        "style_period",
    ],
    "properties": {
        "recognized_as_product": {"type": "string", "enum": VALID_RECOGNIZED},
        "image_quality": {"type": "integer", "minimum": 1, "maximum": 5},
        "target_category_recognizability": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "design_intent": {"type": "string", "enum": VALID_INTENTS},
        "function_aesthetic_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "form_function_consistency": {"type": "integer", "minimum": 1, "maximum": 5},
        "ambiguity_reason": {"type": "string"},
        "function": {"type": "string"},
        "semantic_meaning": {"type": "string"},
        "emotion": {"type": "string"},
        "cognitive_interpretation": {"type": "string"},
        "style_period": {"type": "string"},
    },
}


def extract_text(resp) -> str:
    if getattr(resp, "output_text", None):
        return resp.output_text
    chunks = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def call_openai(client, model, prompt, image_path, max_attempts):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_to_data_url(image_path)},
                        ],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "design_eval_9b_s2_full1600",
                        "schema": JSON_SCHEMA,
                        "strict": True,
                    }
                },
            )
            text = extract_text(resp)
            return normalize_parsed(json.loads(text)), text, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            lower = last_error.lower()
            if "insufficient_quota" in lower or "exceeded your current quota" in lower:
                return None, "", "FATAL_QUOTA_OR_BILLING: " + last_error
            if "model" in lower and ("not found" in lower or "does not exist" in lower):
                return None, "", "FATAL_MODEL: " + last_error
            if attempt < max_attempts and any(
                marker in lower
                for marker in ["429", "rate", "timeout", "temporarily", "500", "502", "503", "504"]
            ):
                wait = min(90, 4 * attempt * attempt)
                print(f"retry {attempt}/{max_attempts} after {wait}s: {last_error[:180]}", flush=True)
                time.sleep(wait)
                continue
            return None, "", last_error
    return None, "", last_error or "unknown error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("COMMERCIAL_LIMIT", "1600")))
    parser.add_argument("--sleep", type=float, default=float(os.getenv("OPENAI_SLEEP_BETWEEN", "0.1")))
    parser.add_argument("--max-attempts", type=int, default=int(os.getenv("OPENAI_MAX_ATTEMPTS", "5")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_report(args.model, args.limit)
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it in the shell environment before running.")

    rows = attach_global_index(read_manifest())[: args.limit]
    prompt_key, prompt_template, prompt_sha = load_prompt_module()
    model_slug = safe_slug(args.model)
    out_dir = run_root() / "gpt_5_5"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"openai_{model_slug}_full1600_{prompt_key}_raw.jsonl"
    json_path = out_dir / f"openai_{model_slug}_full1600_{prompt_key}_raw.json"
    csv_path = out_dir / f"openai_{model_slug}_full1600_{prompt_key}_summary.csv"

    completed = load_completed(jsonl_path)
    print(
        f"provider=openai model={args.model} target={len(rows)} completed={len(completed)} "
        f"prompt_key={prompt_key} prompt_sha256={prompt_sha} generation_model={EXPECTED_GENERATION_MODEL}",
        flush=True,
    )

    client = OpenAI()
    for idx, meta in enumerate(rows, 1):
        image_id = meta["image_id"]
        if image_id in completed:
            continue
        prompt = prompt_template.format(product=meta["product"])
        image_path = image_path_for(meta)
        started = time.time()
        parsed, raw_text, error = call_openai(client, args.model, prompt, image_path, args.max_attempts)
        elapsed = round(time.time() - started, 2)
        record = {
            "image_id": image_id,
            "provider": "openai",
            "model": args.model,
            "prompt_key": prompt_key,
            "prompt_sha256": prompt_sha,
            "expected_generation_model": EXPECTED_GENERATION_MODEL,
            "meta": meta,
            "parsed": parsed,
            "raw_text": raw_text,
            "error": error,
            "elapsed_sec": elapsed,
        }
        write_jsonl_row(jsonl_path, record)
        if parsed and not error:
            completed[image_id] = record
        intent = (parsed or {}).get("design_intent", "ERROR")
        print(
            f"[{idx}/{len(rows)}] {image_id} {meta['generation_model']} {args.model} "
            f"intent={intent} err={bool(error)} elapsed={elapsed}s",
            flush=True,
        )
        if error.startswith("FATAL_"):
            print("fatal API condition; stopping run so it can be fixed/resumed.", flush=True)
            break
        if args.sleep:
            time.sleep(args.sleep)

    ordered_records = [completed[row["image_id"]] for row in rows if row["image_id"] in completed]
    write_outputs(ordered_records, json_path, csv_path)
    print(f"saved_json={json_path}", flush=True)
    print(f"saved_csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
