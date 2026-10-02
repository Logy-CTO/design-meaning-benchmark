import argparse
import os
import time

from PIL import Image
import google.generativeai as genai

from common_full1600 import (
    EXPECTED_GENERATION_MODEL,
    attach_global_index,
    dry_run_report,
    image_path_for,
    load_completed,
    load_prompt_module,
    normalize_parsed,
    parse_json_text,
    read_manifest,
    run_root,
    safe_slug,
    write_jsonl_row,
    write_outputs,
)


def call_gemini(model, prompt, image_path, max_attempts):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            image = Image.open(image_path).convert("RGB")
            resp = model.generate_content([prompt, image])
            text = getattr(resp, "text", "") or ""
            return normalize_parsed(parse_json_text(text)), text, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            lower = last_error.lower()
            if "generaterequestsperday" in lower or "free_tier_requests" in lower:
                return None, "", "FATAL_DAILY_QUOTA: " + last_error
            if "api key not valid" in lower or "permission" in lower or "billing" in lower:
                return None, "", "FATAL_AUTH_OR_BILLING: " + last_error
            if "model" in lower and ("not found" in lower or "not supported" in lower):
                return None, "", "FATAL_MODEL: " + last_error
            if attempt < max_attempts and any(
                marker in lower
                for marker in ["429", "quota", "rate", "timeout", "temporarily", "resource_exhausted", "500", "502", "503", "504"]
            ):
                wait = min(120, 5 * attempt * attempt)
                print(f"retry {attempt}/{max_attempts} after {wait}s: {last_error[:180]}", flush=True)
                time.sleep(wait)
                continue
            return None, "", last_error
    return None, "", last_error or "unknown error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("COMMERCIAL_LIMIT", "1600")))
    parser.add_argument("--sleep", type=float, default=float(os.getenv("GEMINI_SLEEP_BETWEEN", "0.8")))
    parser.add_argument("--max-attempts", type=int, default=int(os.getenv("GEMINI_MAX_ATTEMPTS", "7")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_report(args.model, args.limit)
        return

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY or GOOGLE_API_KEY is not set. Set it in the shell environment before running.")

    rows = attach_global_index(read_manifest())[: args.limit]
    prompt_key, prompt_template, prompt_sha = load_prompt_module()
    model_slug = safe_slug(args.model)
    out_dir = run_root() / "gemini_2_5_flash"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"gemini_{model_slug}_full1600_{prompt_key}_raw.jsonl"
    json_path = out_dir / f"gemini_{model_slug}_full1600_{prompt_key}_raw.json"
    csv_path = out_dir / f"gemini_{model_slug}_full1600_{prompt_key}_summary.csv"

    completed = load_completed(jsonl_path)
    print(
        f"provider=gemini model={args.model} target={len(rows)} completed={len(completed)} "
        f"prompt_key={prompt_key} prompt_sha256={prompt_sha} generation_model={EXPECTED_GENERATION_MODEL}",
        flush=True,
    )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        args.model,
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )

    for idx, meta in enumerate(rows, 1):
        image_id = meta["image_id"]
        if image_id in completed:
            continue
        prompt = prompt_template.format(product=meta["product"])
        image_path = image_path_for(meta)
        started = time.time()
        parsed, raw_text, error = call_gemini(model, prompt, image_path, args.max_attempts)
        elapsed = round(time.time() - started, 2)
        record = {
            "image_id": image_id,
            "provider": "gemini",
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
