#!/usr/bin/env python3
"""Regenerate raw brain-region spectral rows from completed checkpoint runs.

The script scans completed benchmark runs, derives the target preprocessing
cache for each run, calls spectral_evidence.py for every run, and then
merges the newly generated raw clip-level rows into a configurable output root.

It does not aggregate across clips, patients, channels, bands, models, budgets,
or seeds. The primary output remains one row per
task/direction/model/budget/seed/patient/clip/channel/band.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


DEFAULT_PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path.cwd()))
DEFAULT_DATA_ROOT = Path(os.environ.get('DATA_ROOT', DEFAULT_PROJECT_ROOT / 'data'))
DEFAULT_RESULT_ROOT = DEFAULT_DATA_ROOT / 'results'
DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / 'brain_region_features'
DEFAULT_SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_CROSS_DATASET_PREPROCESS = DEFAULT_DATA_ROOT / 'preprocessed' / 'cross_dataset'
DEFAULT_CROSS_MODAL_PREPROCESS = DEFAULT_DATA_ROOT / 'preprocessed' / 'cross_modal'
DEFAULT_MISSIONS = ["dataset_transfer"]
DEFAULT_WINDOWS = ["12s"]
DEFAULT_TASKS = ["detection"]
DEFAULT_DIRECTIONS = ["TUSZ_to_CHB-MIT", "CHB-MIT_to_TUSZ"]
DEFAULT_SEEDS = [1]
DEFAULT_MODELS = ["CBraMod"]

MODEL_BUDGET_PATTERN = re.compile(
    r"^(?P<model>.+?)_(?P<budget>0|25|50|75|100)(?:%|pct)?_budget$"
)
SEED_PATTERN = re.compile(r"^seed_(?P<seed>\d+)$")

DISPLAY_TO_DATASET = {
    "TUSZ": "tusz",
    "CHB-MIT": "chbmit",
    "Siena": "siena",
    "Epilepsy-iEEG": "epilepsy_ieeg",
}


def parse_model_budget(name: str) -> tuple[str, int] | None:
    match = MODEL_BUDGET_PATTERN.match(name)
    if match is None:
        return None
    return match.group("model"), int(match.group("budget"))


def parse_seed(name: str) -> int | None:
    match = SEED_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group("seed"))


def parse_direction(name: str) -> tuple[str, str] | None:
    if "_to_" not in name:
        return None
    source, target = name.split("_to_", 1)
    if source not in DISPLAY_TO_DATASET or target not in DISPLAY_TO_DATASET:
        return None
    return DISPLAY_TO_DATASET[source], DISPLAY_TO_DATASET[target]


def format_budget_name(model: str, budget: int) -> str:
    return f"{model}_0_budget" if budget == 0 else f"{model}_{budget}%_budget"


def run_has_checkpoint(run_root: Path) -> bool:
    candidates = [
        run_root / "source" / "best.pt",
        run_root / "source" / "best.pth",
        run_root / "source" / "best.pth.tar",
        run_root / "source" / "best.ckpt",
        run_root / "source" / "best.weights.h5",
        run_root / "source" / "best_model.pt",
        run_root / "source" / "checkpoint.pt",
        run_root / "source" / "model.pt",
        run_root / "target" / "best.pt",
        run_root / "target" / "best.pth",
        run_root / "target" / "best.pth.tar",
        run_root / "target" / "best.ckpt",
        run_root / "target" / "best.weights.h5",
        run_root / "target" / "best_model.pt",
        run_root / "target" / "checkpoint.pt",
        run_root / "target" / "model.pt",
        run_root / "best_model.joblib",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return True
    for candidate in run_root.rglob("*"):
        if (
            candidate.is_file()
            and candidate.stat().st_size > 0
            and candidate.suffix.lower() in {".pt", ".pth", ".ckpt", ".joblib", ".h5", ".weights", ".tar"}
        ):
            return True
    return False


def parse_csv_set(value: str | None) -> set[str] | None:
    if value is None or str(value).strip() == "":
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def parse_int_csv_set(value: str | None) -> set[int] | None:
    raw = parse_csv_set(value)
    if raw is None:
        return None
    return {int(item) for item in raw}


def discover_runs(
    result_root: Path,
    missions: set[str] | None,
    windows: set[str] | None,
    tasks: set[str] | None,
    directions: set[str] | None,
    seeds: set[int] | None,
    models: set[str] | None,
    budgets: set[int] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(result_root.rglob("metrics.json")):
        run_root = metrics_path.parent
        if not (run_root / "predictions.csv").is_file():
            continue
        seed = parse_seed(run_root.name)
        if seed is None:
            continue
        model_budget_dir = run_root.parent
        parsed_budget = parse_model_budget(model_budget_dir.name)
        if parsed_budget is None:
            continue
        direction_dir = model_budget_dir.parent
        parsed_direction = parse_direction(direction_dir.name)
        if parsed_direction is None:
            continue
        task_dir = direction_dir.parent
        window_dir = task_dir.parent
        mission_dir = window_dir.parent
        if mission_dir.name not in {"dataset_transfer", "cross_dataset", "eeg_ieeg_transfer", "eeg_ieeg_localization"}:
            continue
        if missions is not None and mission_dir.name not in missions:
            continue
        if windows is not None and window_dir.name not in windows:
            continue
        if tasks is not None and task_dir.name not in tasks:
            continue
        if directions is not None and direction_dir.name not in directions:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if not run_has_checkpoint(run_root):
            continue
        model, budget = parsed_budget
        source_dataset, target_dataset = parsed_direction
        if models is not None and model not in models:
            continue
        if budgets is not None and budget not in budgets:
            continue
        rows.append(
            {
                "mission": mission_dir.name,
                "window": window_dir.name,
                "task": task_dir.name,
                "direction": direction_dir.name,
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "model": model,
                "budget": budget,
                "seed": seed,
                "run_root": run_root,
            }
        )
    return rows


def read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def target_root_from_run(
    run: dict[str, object],
    cross_dataset_preprocess: Path,
    cross_modal_preprocess: Path,
) -> Path | None:
    cache_validation = read_json(Path(run["run_root"]) / "cache_validation.json")
    if cache_validation:
        target = cache_validation.get("target")
        if isinstance(target, dict) and target.get("root"):
            path = Path(str(target["root"]))
            if (path / "manifest.csv").is_file():
                return path
    protocol = read_json(Path(run["run_root"]) / "protocol.json")
    if protocol:
        for key in ("target_root", "target_cache_root"):
            value = protocol.get(key)
            if value:
                path = Path(str(value))
                if (path / "manifest.csv").is_file():
                    return path
        preprocess_root = protocol.get("preprocess_root")
        task = protocol.get("task", run["task"])
        window = protocol.get("window_name", run["window"])
        target_dataset = protocol.get("target_dataset", run["target_dataset"])
        if preprocess_root:
            target_name = {
                "tusz": "TUSZ",
                "chbmit": "CHB-MIT",
                "siena": "Siena",
                "epilepsy_ieeg": "Epilepsy-iEEG",
            }.get(str(target_dataset), str(target_dataset))
            path = Path(str(preprocess_root)) / str(window) / str(task) / target_name
            if (path / "manifest.csv").is_file():
                return path

    target_name = {
        "tusz": "TUSZ",
        "chbmit": "CHB-MIT",
        "siena": "Siena",
        "epilepsy_ieeg": "Epilepsy-iEEG",
    }[str(run["target_dataset"])]
    preprocess_root = (
        cross_dataset_preprocess
        if run["mission"] in {"dataset_transfer", "cross_dataset"}
        else cross_modal_preprocess
    )
    task_folder = "detection" if run["mission"] == "eeg_ieeg_localization" else str(run["task"])
    candidates = [
        preprocess_root / str(run["window"]) / task_folder / target_name,
        DEFAULT_DATA_ROOT / preprocess_root.name / str(run["window"]) / task_folder / target_name,
    ]
    for path in candidates:
        if (path / "manifest.csv").is_file():
            return path
    return None


def spectral_output_root(output_root: Path, run: dict[str, object]) -> Path:
    model_budget = format_budget_name(str(run["model"]), int(run["budget"]))
    return (
        output_root
        / "regenerated_spectral_evidence"
        / str(run["window"])
        / str(run["mission"])
        / str(run["task"])
        / str(run["direction"])
        / model_budget
        / f"seed_{run['seed']}"
    )


def run_export(
    python_exe: str,
    export_script: Path,
    run: dict[str, object],
    target_root: Path,
    output_root: Path,
    patient_count: int,
    clips_per_patient: int,
    selection_seed: int,
    force: bool,
) -> dict[str, object]:
    destination = spectral_output_root(output_root, run)
    done_file = destination / "spectral_evidence_by_clip.csv"
    if done_file.is_file() and done_file.stat().st_size > 0 and not force:
        return {"status": "skipped_existing", "output_root": str(destination)}
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        python_exe,
        str(export_script),
        "--run-root",
        str(run["run_root"]),
        "--target-root",
        str(target_root),
        "--output-root",
        str(destination),
        "--patient-count",
        str(patient_count),
        "--clips-per-patient",
        str(clips_per_patient),
        "--selection-seed",
        str(selection_seed),
    ]
    process = subprocess.run(
        command,
        cwd=str(export_script.parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    status = "complete" if process.returncode == 0 else "failed"
    log_path = destination / "regenerate_stdout.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    return {
        "status": status,
        "returncode": process.returncode,
        "output_root": str(destination),
        "log_path": str(log_path),
    }


def merge_raw_outputs(output_root: Path) -> Path:
    frames = []
    for path in sorted((output_root / "regenerated_spectral_evidence").rglob("spectral_evidence_by_clip.csv")):
        context = parse_regenerated_path(path, output_root)
        frame = pd.read_csv(path, dtype={"patient_id": str, "clip_id": str, "channel_name": str, "band": str})
        for key, value in context.items():
            frame[key] = value
        channels = frame["channel_name"].astype(str).str.strip()
        endpoints = channels.str.split("-", n=1, expand=True)
        frame["brain_region"] = channels
        frame["channel_start"] = endpoints[0].fillna(channels)
        frame["channel_end"] = endpoints[1].fillna("") if endpoints.shape[1] > 1 else ""
        frame["row_granularity"] = "raw_clip_patient_channel_band"
        frame["is_aggregated"] = False
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No regenerated spectral_evidence_by_clip.csv files were produced")
    raw = pd.concat(frames, ignore_index=True)
    columns = [
        "window",
        "mission",
        "task",
        "direction",
        "model",
        "budget",
        "seed",
        "patient_id",
        "clip_id",
        "brain_region",
        "channel_name",
        "channel_start",
        "channel_end",
        "band",
        "sfreq_hz",
        "label",
        "model_score",
        "relative_band_power",
        "positive_score_weighted_power",
        "signed_score_weighted_power",
        "row_granularity",
        "is_aggregated",
        "source_file",
    ]
    for frame_path in sorted((output_root / "regenerated_spectral_evidence").rglob("spectral_evidence_by_clip.csv")):
        pass
    raw["source_file"] = raw.get("source_file", "")
    raw = raw[[column for column in columns if column in raw.columns]]
    out_path = output_root / "raw_brain_region_spectral_features_long.csv"
    raw.to_csv(out_path, index=False)
    return out_path


def parse_regenerated_path(path: Path, output_root: Path) -> dict[str, object]:
    relative = path.relative_to(output_root / "regenerated_spectral_evidence")
    parts = relative.parts
    if len(parts) < 7:
        raise ValueError(f"Cannot parse regenerated output path: {path}")
    window, mission, task, direction, model_budget, seed_name = parts[:6]
    parsed = parse_model_budget(model_budget)
    seed = parse_seed(seed_name)
    if parsed is None or seed is None:
        raise ValueError(f"Cannot parse regenerated output path: {path}")
    model, budget = parsed
    return {
        "window": window,
        "mission": mission,
        "task": task,
        "direction": direction,
        "model": model,
        "budget": budget,
        "seed": seed,
        "source_file": str(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--script-root", default=str(DEFAULT_SCRIPT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--patient-count", type=int, default=999999)
    parser.add_argument("--clips-per-patient", type=int, default=999999)
    parser.add_argument("--selection-seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--missions", default=",".join(DEFAULT_MISSIONS))
    parser.add_argument("--windows", default=",".join(DEFAULT_WINDOWS))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--directions", default=",".join(DEFAULT_DIRECTIONS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--budgets", default="")
    parser.add_argument("--cross-dataset-preprocess", default=str(DEFAULT_CROSS_DATASET_PREPROCESS))
    parser.add_argument("--cross-modal-preprocess", default=str(DEFAULT_CROSS_MODAL_PREPROCESS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    script_root = Path(args.script_root).expanduser().resolve()
    cross_dataset_preprocess = Path(args.cross_dataset_preprocess).expanduser()
    cross_modal_preprocess = Path(args.cross_modal_preprocess).expanduser()
    export_script = script_root / "spectral_evidence.py"
    if not export_script.is_file():
        raise FileNotFoundError(f"Missing export script: {export_script}")
    output_root.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(
        result_root=result_root,
        missions=parse_csv_set(args.missions),
        windows=parse_csv_set(args.windows),
        tasks=parse_csv_set(args.tasks),
        directions=parse_csv_set(args.directions),
        seeds=parse_int_csv_set(args.seeds),
        models=parse_csv_set(args.models),
        budgets=parse_int_csv_set(args.budgets),
    )
    if not runs:
        raise FileNotFoundError(f"No completed checkpoint runs found below {result_root}")
    report = []
    for run in tqdm(runs, desc="Regenerating spectral evidence from ckpts", colour="green", unit="run"):
        target_root = target_root_from_run(
            run,
            cross_dataset_preprocess=cross_dataset_preprocess,
            cross_modal_preprocess=cross_modal_preprocess,
        )
        if target_root is None:
            item = {key: str(value) for key, value in run.items() if key != "run_root"}
            item["run_root"] = str(run["run_root"])
            item["status"] = "missing_target_manifest"
            report.append(item)
            continue
        if args.dry_run:
            status = {"status": "dry_run", "output_root": str(spectral_output_root(output_root, run))}
        else:
            status = run_export(
                python_exe=args.python,
                export_script=export_script,
                run=run,
                target_root=target_root,
                output_root=output_root,
                patient_count=args.patient_count,
                clips_per_patient=args.clips_per_patient,
                selection_seed=args.selection_seed,
                force=bool(args.force),
            )
        item = {key: str(value) for key, value in run.items() if key != "run_root"}
        item["run_root"] = str(run["run_root"])
        item["target_root"] = str(target_root)
        item.update(status)
        report.append(item)
    report_path = output_root / "regeneration_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.dry_run:
        merged_path = merge_raw_outputs(output_root)
        print(f"RAW_BRAIN_REGION_FEATURES_EXPORTED output={merged_path}")
    print(f"REGENERATION_REPORT output={report_path}")


if __name__ == "__main__":
    main()
