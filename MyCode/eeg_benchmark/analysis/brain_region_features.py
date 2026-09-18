#!/usr/bin/env python3
"""Extract raw brain-region spectral feature rows without averaging.

This script is intentionally conservative for figure source-data export:

1. It reads existing spectral audit outputs named spectral_evidence_by_clip.csv.
2. It preserves every clip x patient x channel x band row.
3. It adds model, budget, task, direction, and seed parsed from the run path.
4. It does not average across clips, bands, channels, patients, budgets, or models.

Optional target_channel_profiles.csv files are copied into a separate long table
as recorded by the original run. They are not treated as raw spectral evidence
because those files are already channel-profile summaries produced by the run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


DEFAULT_PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path.cwd()))
DEFAULT_INPUT_ROOTS = [DEFAULT_PROJECT_ROOT / 'spectral_evidence']
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get('BRAIN_REGION_OUTPUT_ROOT', DEFAULT_PROJECT_ROOT / 'brain_region_features')
)

MODEL_BUDGET_PATTERN = re.compile(
    r"^(?P<model>.+?)_(?P<budget>0|25|50|75|100)(?:%|pct)?_budget$"
)
SEED_PATTERN = re.compile(r"^seed_(?P<seed>\d+)$")


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


def parse_run_context(path: Path) -> dict[str, object] | None:
    seed_dir: Path | None = None
    seed: int | None = None
    for parent in path.parents:
        seed = parse_seed(parent.name)
        if seed is not None:
            seed_dir = parent
            break
    if seed_dir is None or seed is None:
        return None

    model_budget_dir: Path | None = None
    parsed: tuple[str, int] | None = None
    for parent in seed_dir.parents:
        parsed = parse_model_budget(parent.name)
        if parsed is not None:
            model_budget_dir = parent
            break
    if model_budget_dir is None or parsed is None:
        return None

    direction_dir = model_budget_dir.parent
    task_dir = direction_dir.parent
    model, budget = parsed
    return {
        "task": task_dir.name,
        "direction": direction_dir.name,
        "model": model,
        "budget": budget,
        "seed": seed,
        "run_root": str(seed_dir),
        "source_file": str(path),
    }


def parse_spectral_path(path: Path) -> dict[str, object] | None:
    return parse_run_context(path)


def parse_profile_path(path: Path) -> dict[str, object] | None:
    return parse_run_context(path)


def normalize_channel_name(value: object) -> str:
    return str(value).strip()


def add_channel_endpoint_columns(frame: pd.DataFrame) -> pd.DataFrame:
    channels = frame["channel_name"].map(normalize_channel_name)
    endpoints = channels.str.split("-", n=1, expand=True)
    frame["brain_region"] = channels
    frame["channel_start"] = endpoints[0].fillna(channels)
    if endpoints.shape[1] > 1:
        frame["channel_end"] = endpoints[1].fillna("")
    else:
        frame["channel_end"] = ""
    return frame


def read_spectral_clip_file(path: Path) -> pd.DataFrame:
    meta = parse_spectral_path(path)
    if meta is None:
        raise ValueError(f"Cannot parse spectral path: {path}")
    frame = pd.read_csv(path, dtype={"patient_id": str, "clip_id": str, "channel_name": str, "band": str})
    required = {
        "patient_id",
        "clip_id",
        "channel_name",
        "band",
        "label",
        "model_score",
        "relative_band_power",
        "positive_score_weighted_power",
        "signed_score_weighted_power",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    for key, value in meta.items():
        frame[key] = value
    frame = add_channel_endpoint_columns(frame)
    frame["row_granularity"] = "raw_clip_patient_channel_band"
    frame["is_aggregated"] = False
    return frame


def read_profile_file(path: Path) -> pd.DataFrame:
    meta = parse_profile_path(path)
    if meta is None:
        raise ValueError(f"Cannot parse profile path: {path}")
    frame = pd.read_csv(path, dtype={"patient_id": str, "channel_name": str})
    required = {"patient_id", "channel_name", "importance"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    for key, value in meta.items():
        frame[key] = value
    frame = add_channel_endpoint_columns(frame)
    frame["row_granularity"] = "recorded_patient_channel_profile"
    frame["is_aggregated"] = True
    return frame


def discover_files(input_roots: list[Path]) -> tuple[list[Path], list[Path]]:
    spectral_files: dict[str, Path] = {}
    profile_files: dict[str, Path] = {}
    for root in input_roots:
        if not root.exists():
            continue
        for path in root.rglob("spectral_evidence_by_clip.csv"):
            spectral_files[str(path.resolve())] = path
        for path in root.rglob("target_channel_profiles.csv"):
            profile_files[str(path.resolve())] = path
    return sorted(spectral_files.values()), sorted(profile_files.values())


def write_manifest(
    output_root: Path,
    input_roots: list[Path],
    spectral_files: list[Path],
    profile_files: list[Path],
    raw_rows: int,
    profile_rows: int,
) -> None:
    manifest = {
        "status": "complete",
        "output_root": str(output_root),
        "input_roots": [str(path) for path in input_roots],
        "raw_spectral_input_files": len(spectral_files),
        "profile_input_files": len(profile_files),
        "raw_spectral_rows": raw_rows,
        "profile_rows": profile_rows,
        "raw_spectral_granularity": "task/direction/model/budget/seed/patient/clip/channel/band",
        "profile_granularity": "task/direction/model/budget/seed/patient/channel as recorded by target_channel_profiles.csv",
        "aggregation_policy": "no averaging, no summation, no groupby aggregation in this extractor",
        "primary_heatmap_source": "raw_brain_region_spectral_features_long.csv",
        "notes": [
            "Use raw_brain_region_spectral_features_long.csv for unaveraged spectral heatmap source data.",
            "Use recorded_channel_profile_importance_long.csv only as a separate audit of existing channel-profile summaries.",
        ],
    }
    (output_root / "extraction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export(args: argparse.Namespace) -> Path:
    input_roots = [Path(value).expanduser().resolve() for value in args.input_root]
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    spectral_files, profile_files = discover_files(input_roots)
    if not spectral_files:
        raise FileNotFoundError(
            "No spectral_evidence_by_clip.csv files were found under the input roots"
        )

    raw_frames = []
    for path in tqdm(spectral_files, desc="Reading raw spectral rows", colour="green", unit="file"):
        raw_frames.append(read_spectral_clip_file(path))
    raw = pd.concat(raw_frames, ignore_index=True)
    raw_columns = [
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
        "run_root",
        "source_file",
    ]
    raw = raw[[column for column in raw_columns if column in raw.columns]]
    raw_path = output_root / "raw_brain_region_spectral_features_long.csv"
    raw.to_csv(raw_path, index=False)

    profile_rows = 0
    if profile_files:
        profile_frames = []
        for path in tqdm(profile_files, desc="Reading recorded channel profiles", colour="cyan", unit="file"):
            try:
                profile_frames.append(read_profile_file(path))
            except ValueError:
                continue
        if profile_frames:
            profiles = pd.concat(profile_frames, ignore_index=True)
            profile_rows = len(profiles)
            profile_path = output_root / "recorded_channel_profile_importance_long.csv"
            profiles.to_csv(profile_path, index=False)

    write_manifest(
        output_root=output_root,
        input_roots=input_roots,
        spectral_files=spectral_files,
        profile_files=profile_files,
        raw_rows=len(raw),
        profile_rows=profile_rows,
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        action="append",
        default=None,
        help="Root to scan. Can be repeated. Defaults to common server roots.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.input_root is None:
        args.input_root = [str(path) for path in DEFAULT_INPUT_ROOTS]
    output = export(args)
    print(f"RAW_BRAIN_REGION_FEATURES_EXPORTED output={output}")


if __name__ == "__main__":
    main()
