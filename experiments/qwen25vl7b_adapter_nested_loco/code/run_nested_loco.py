from __future__ import annotations
import argparse, json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def log(message, handle):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()

def run(command, env, log_file, handle):
    log("START " + " ".join(command), handle)
    start = time.time()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as output:
        subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    log(f"DONE elapsed_seconds={time.time()-start:.1f}; log={log_file}", handle)

def command_for(python_bin, script, fold_dir, image_root, out, mode, fold, seed, config, validation):
    cmd = [
        python_bin, "-u", str(script), "--fold_dir", str(fold_dir),
        "--image_root", str(image_root), "--output_dir", str(out),
        "--target_mode", mode, "--fold", str(fold), "--seed", str(seed),
        "--epochs", str(config["max_epochs"]), "--patience", "4",
        "--min_delta", "0.0001", "--batch_size", "2", "--grad_accum", "4",
        "--lr", str(config["learning_rate"]), "--weight_decay", "0.01",
        "--lora_r", str(config["rank"]), "--lora_alpha", str(config["alpha"]),
        "--lora_dropout", str(config["dropout"]),
        "--hard_weight", str(config["hard_weight"]),
        "--entropy_weight", str(config["entropy_weight"]),
    ]
    if validation:
        cmd += ["--evaluation_split", "validation"]
    return cmd

def fit_if_missing(cmd, out, pred_name, expected, env, handle):
    summary_path, pred_path = out / "run_summary.json", out / pred_name
    if summary_path.is_file() and pred_path.is_file():
        summary = read_json(summary_path)
        count = summary.get("evaluation_predictions", summary.get("test_predictions"))
        if count == expected:
            log(f"SKIP complete {out}", handle)
            return
        raise RuntimeError(f"Unexpected completed count in {summary_path}: {count}")
    if pred_path.exists() and not summary_path.exists():
        raise RuntimeError(f"Incomplete prior output; inspect manually before resume: {pred_path}")
    run(cmd, env, out / "run.log", handle)
    if not summary_path.is_file() or not pred_path.is_file():
        raise RuntimeError(f"Missing output after fit: {out}")
    summary = read_json(summary_path)
    count = summary.get("evaluation_predictions", summary.get("test_predictions"))
    if count != expected:
        raise RuntimeError(f"Expected {expected} predictions, got {count}: {summary_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_bin", default=__import__("sys").executable)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parents[1]
    experiments, project = experiment.parent, experiment.parent.parent
    source_experiment = experiments / "qwen25vl7b_adapter_leave_category_out"
    fold_root = source_experiment / "data" / "loco_folds"
    image_root = project / "images"
    vendor = experiments / "qwen25vl7b_adapter_target_ablation" / "vendor"
    nested_data, run_root = experiment / "data" / "nested_folds", experiment / "outputs"
    analysis_dir, logs_dir = experiment / "analysis", experiment / "logs"
    train_script = experiment / "code" / "train_nested.py"
    configs_file = experiment / "configs.json"
    for path in (fold_root, image_root, train_script, configs_file):
        if not path.exists():
            raise FileNotFoundError(path)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    used_mib = int(gpu.stdout.strip().splitlines()[0])
    if used_mib > 4096:
        raise RuntimeError(f"GPU 0 is already using {used_mib} MiB; refusing to start")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1",
        "HF_HOME": os.environ.get("HF_HOME", str(Path.home()/".cache/huggingface")),
        "TOKENIZERS_PARALLELISM": "false",
    })
    configs = json.loads(configs_file.read_text(encoding="utf-8"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "orchestrator.log").open("a", encoding="utf-8") as handle:
        log(f"Nested LOCO start; GPU 0 memory used={used_mib} MiB; configs={len(configs)}", handle)
        run(
            [args.python_bin, "-u", str(experiment / "code" / "make_nested_folds.py"),
             "--source_fold_root", str(fold_root), "--output_root", str(nested_data)],
            env, logs_dir / "make_folds.log", handle,
        )
        inner_done = 0
        for outer in range(1, 9):
            for inner_dir in sorted((nested_data / f"outer_{outer:02d}").glob("inner_*")):
                inner = int(inner_dir.name.split("_")[1])
                for config in configs:
                    out = run_root / "inner" / f"outer_{outer:02d}" / f"inner_{inner:02d}" / config["config_id"]
                    cmd = command_for(
                        args.python_bin, train_script, inner_dir, image_root, out, "soft",
                        outer * 100 + inner, 20260925 + 1000000 + 1000 * outer + inner,
                        config, True,
                    )
                    fit_if_missing(cmd, out, "validation_predictions.jsonl", 25, env, handle)
                    inner_done += 1
        if inner_done != 168:
            raise RuntimeError(f"Expected 168 inner fits, checked {inner_done}")
        log("All inner candidates are complete.", handle)
        run(
            [args.python_bin, "-u", str(experiment / "code" / "select_config.py"),
             "--source_fold_root", str(fold_root), "--nested_data_root", str(nested_data),
             "--run_root", str(run_root), "--config_file", str(configs_file),
             "--output_dir", str(analysis_dir)],
            env, logs_dir / "select_config.log", handle,
        )
        selected = read_json(analysis_dir / "selected_configs.json")
        for outer in range(1, 9):
            choice = selected[str(outer)]
            config = {k: choice[k] for k in (
                "rank", "alpha", "learning_rate", "dropout", "max_epochs",
                "hard_weight", "entropy_weight"
            )}
            source_fold = fold_root / f"fold_{outer:02d}"
            for mode, arm in (("soft", "soft_target"), ("majority", "majority_target")):
                out = run_root / f"fold_{outer:02d}" / arm
                cmd = command_for(
                    args.python_bin, train_script, source_fold, image_root, out, mode,
                    outer, 20260925 + 1000 * outer, config, False,
                )
                fit_if_missing(cmd, out, "predictions.jsonl", 25, env, handle)
                metadata = {
                    "selected_config_id": choice["config_id"],
                    "selected_config": config,
                    "inner_mean_soft_cross_entropy_nats": choice["inner_mean_soft_cross_entropy_nats"],
                    "inner_ranking": choice["inner_ranking"],
                    "outer_held_out_category": choice["outer_held_out_category"],
                    "selection_scope": "seven non-test categories; outer test category excluded",
                }
                (out / "nested_tuning_metadata.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        run(
            [args.python_bin, "-u", str(experiment / "code" / "analyze_nested_outer.py"),
             "--fold_root", str(fold_root), "--run_root", str(run_root),
             "--image_root", str(image_root), "--output_dir", str(analysis_dir / "outer_test"),
             "--bootstrap", str(args.bootstrap), "--seed", "8419"],
            env, logs_dir / "outer_analysis.log", handle,
        )
        validation_path = analysis_dir / "outer_test" / "analysis_validation.json"
        validation = read_json(validation_path)
        validation["nested_configuration_selection"] = {
            "configs_per_outer_fold": len(configs),
            "inner_category_folds_per_outer": 7,
            "outer_test_categories_excluded_from_configuration_selection": True,
            "selected_configs": {
                key: {
                    "config_id": value["config_id"],
                    "outer_held_out_category": value["outer_held_out_category"],
                    "inner_mean_soft_cross_entropy_nats": value["inner_mean_soft_cross_entropy_nats"],
                }
                for key, value in selected.items()
            },
        }
        validation["scope"] = (
            "Fixed 200-image human-rated subset; hyperparameter selection nested within each outer "
            "leave-one-category-out training set; one FLUX generator; same 30-rater pool; bootstrap "
            "intervals condition on selected configurations and fitted outer-fold models."
        )
        validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
        log("Nested LOCO and outer analysis complete.", handle)

if __name__ == "__main__":
    main()
