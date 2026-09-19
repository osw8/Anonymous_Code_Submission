from __future__ import annotations

# Builds reproducible EEG and iEEG preprocessing caches.
import argparse
import csv
import hashlib
import json
import logging
import math
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from tqdm import tqdm

from eeg_benchmark.engine import (
    ChannelUnionContract,
    MINIMUM_SCALP_BIPOLAR_CHANNELS,
    SCALP_BIPOLAR_CHANNELS,
    align_to_union,
)


LOGGER_NAME = "benchmark.data_preprocess"
SCALP_PREPROCESSING_PROTOCOL = "scalp_common_16_bipolar_eahs"
IEEG_DETECTION_PROTOCOL = "ieeg_detection_native_electrode_eahs"
IEEG_PREDICTION_PROTOCOL = "ieeg_prediction_native_electrode_eahs"
PREPROCESSING_STATISTICS_VERSION = "eeg_ieeg_track_specific_xai_statistics"
OFFICIAL_TRACK_BANDPASS_HZ = {
    "eeg_cross_dataset": (0.5, 45.0),
    "eeg_ieeg_cross_modal": (0.5, 95.0),
}
WINDOW_SECONDS = 60.0
OUTPUT_LAYOUT_POLICY = "split_patient_recording"
TASK_LABELS = {
    "detection": {0: "non_seizure", 1: "seizure"},
    "prediction": {0: "interictal", 1: "preictal"},
}
MANIFEST_COLUMNS = (
    "clip_id",
    "relative_path",
    "dataset",
    "task",
    "patient_id",
    "split",
    "session_id",
    "montage",
    "source_relative_path",
    "source_segments_json",
    "recording_duration_seconds",
    "timeline_component_id",
    "timeline_start_seconds",
    "timeline_clip_start_seconds",
    "timeline_clip_end_seconds",
    "seizure_intervals_json",
    "source_sfreq",
    "sfreq",
    "channel_count",
    "channel_names",
    "channel_types",
    "position_available_count",
    "coordinate_system",
    "coordinate_units",
    "clip_start_seconds",
    "clip_end_seconds",
    "label",
    "class_name",
    "event_id",
    "overlap_seconds",
    "overlap_ratio_of_clip",
    "requested_l_freq",
    "requested_h_freq",
    "effective_h_freq",
    "normalization",
    "flat_channel_fraction",
    "peak_uv",
    "clip_rms_uv",
    "median_channel_mean_uv",
    "median_channel_std_uv",
    "p95_channel_std_uv",
)
DISPLAY_NAMES = {
    "tusz": "TUSZ",
    "siena": "Siena",
    "chbmit": "CHB-MIT",
    "epilepsy_ieeg": "Epilepsy-iEEG",
    "hup_ieeg": "HUP-iEEG",
    "thalamocortical_ieeg": "Thalamocortical-iEEG",
}
SCALP_EEG_DATASETS = {"tusz", "siena", "chbmit"}
IEEG_DATASETS = {"epilepsy_ieeg", "hup_ieeg", "thalamocortical_ieeg"}
DETECTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY = {
    "tusz",
    "siena",
    "chbmit",
    "thalamocortical_ieeg",
}
PREDICTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY = {"tusz", "siena"}
SPLIT_ALIASES = {"train": "train", "dev": "dev", "val": "dev", "eval": "test", "test": "test"}


@dataclass(frozen=True)
class PredictionRule:
    preictal_seconds: float
    preictal_buffer_seconds: float
    postictal_buffer_seconds: float
    interictal_seconds: float | None = None
    interictal_separation_seconds: float | None = None
    require_both_neighbor_seizures: bool = False
    include_leading_interictal: bool = True
    include_no_seizure_components: bool = True


PREDICTION_RULES = {
    "tusz": PredictionRule(300.0, 300.0, 300.0, include_no_seizure_components=False),
    "siena": PredictionRule(
        preictal_seconds=1800.0,
        preictal_buffer_seconds=300.0,
        postictal_buffer_seconds=300.0,
        interictal_separation_seconds=2100.0,
        require_both_neighbor_seizures=False,
        include_leading_interictal=True,
        include_no_seizure_components=False,
    ),
    "chbmit": PredictionRule(
        preictal_seconds=1800.0,
        preictal_buffer_seconds=300.0,
        postictal_buffer_seconds=14400.0,
        interictal_separation_seconds=3600.0,
        require_both_neighbor_seizures=True,
        include_leading_interictal=False,
        include_no_seizure_components=False,
    ),
}


@dataclass(frozen=True)
class ModelInputSpec:
    sfreq: int
    view_seconds: float
    layout: str = "continuous"
    patch_points: int | None = None


MODEL_INPUT_SPECS = {
    "bendr": ModelInputSpec(250, 4.0),
    "biot": ModelInputSpec(200, 10.0),
    "cbramod": ModelInputSpec(200, 10.0, "patch", 200),
    "cst": ModelInputSpec(128, 1.0, "eegnet"),
    "eegnet": ModelInputSpec(128, 1.0, "eegnet"),
    "eegpt": ModelInputSpec(256, 4.0),
    "eegpt_tueg": ModelInputSpec(200, 10.0),
    "evobrain": ModelInputSpec(200, 60.0, "time_channel_patch", 200),
    "labram": ModelInputSpec(200, 8.0, "patch", 200),
    "luna": ModelInputSpec(256, 5.0),
    "steegformer": ModelInputSpec(128, 6.0),
    "river": ModelInputSpec(256, 30.0, "patch", 256),
}

EVOBRAIN_TUSZ_CHANNELS = (
    "EEG FP1",
    "EEG FP2",
    "EEG F3",
    "EEG F4",
    "EEG C3",
    "EEG C4",
    "EEG P3",
    "EEG P4",
    "EEG O1",
    "EEG O2",
    "EEG F7",
    "EEG F8",
    "EEG T3",
    "EEG T4",
    "EEG T5",
    "EEG T6",
    "EEG FZ",
    "EEG CZ",
    "EEG PZ",
)

MODEL_CHANNEL_POLICIES = {
    "bendr": "pretrained_20_channel_projection_after_shared_contract",
    "biot": "native_order_with_model_channel_tokens",
    "cbramod": "native_order_variable_channels",
    "cst": "resizenet_input_alignment_after_shared_contract",
    "eegnet": "native_order_fixed_per_dataset_model",
    "eegpt": "native_order_with_channel_name_ids",
    "eegpt_tueg": "native_order_with_channel_name_ids",
    "evobrain": "native_graph_channels_and_tusz_original_19_channel_subset",
    "labram": "native_order_with_channel_name_ids",
    "luna": "native_order_topology_agnostic_queries",
    "steegformer": "native_order_with_channel_name_ids",
    "river": "native_order_with_dataset_specific_spatial_adapter",
}

IEEG_AUXILIARY_CONTACT = re.compile(
    r"^(?:DC\d+|E|EKG\d*|ECG\d*|EMG\d*|EOG\d*|PULSE|SPO2|ETCO2)$",
    flags=re.IGNORECASE,
)
IEEG_CONTACT_NUMBER = re.compile(r"^(.+?)(\d+)$")


def canonical_ieeg_contact_name(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value).strip())
    name = re.sub(r"^POL\s+", "", name, flags=re.IGNORECASE)
    name = name.replace("$", "")
    return re.sub(r"\s+", "", name).upper()


@dataclass
class SeizureInterval:
    start_seconds: float
    end_seconds: float
    seizure_id: str
    annotation_source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_seconds) or not math.isfinite(self.end_seconds):
            raise ValueError("Seizure interval must be finite")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Seizure end must be greater than seizure onset")


@dataclass
class Recording:
    dataset: str
    path: Path
    source_root: Path
    patient_id: str
    split: str
    session_id: str
    montage: str
    output_subdir: Path
    duration_seconds: float
    source_sfreq: float
    start_timestamp: float | None
    channel_sidecar: Path | None = None
    electrode_sidecar: Path | None = None
    coordsystem_sidecar: Path | None = None
    seizures: list[SeizureInterval] = field(default_factory=list)
    component_id: str = ""
    timeline_start: float = 0.0

    @property
    def timeline_end(self) -> float:
        return self.timeline_start + self.duration_seconds

    @property
    def relative_path(self) -> str:
        return self.path.relative_to(self.source_root).as_posix()


@dataclass(frozen=True)
class ClipCandidate:
    task: str
    recording_path: str
    local_start_seconds: float
    local_end_seconds: float
    label: int
    class_name: str
    event_id: str
    overlap_seconds: float
    source_segments: tuple["ClipSegment", ...] = ()


@dataclass(frozen=True)
class ClipSegment:
    recording_path: str
    local_start_seconds: float
    local_end_seconds: float


class ClipQualityExclusion(ValueError):
    pass


def failures_for_task(
    failures: Sequence[dict[str, Any]],
    task: str,
) -> list[dict[str, Any]]:
    relevant: list[dict[str, Any]] = []
    for failure in failures:
        failure_task = failure.get("task")
        failure_tasks = failure.get("tasks")
        if failure_task is not None:
            applies = str(failure_task) == task
        elif failure_tasks is not None:
            applies = task in {str(value) for value in failure_tasks}
        else:
            applies = True
        if applies:
            relevant.append(failure)
    return relevant


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def configured_window_seconds(config: dict[str, Any]) -> float:
    window_seconds = float(config.get("window_seconds", WINDOW_SECONDS))
    if not math.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("window_seconds must be a finite positive number")
    return window_seconds


def resolved_signal_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    signal = dict(config['signal'])
    dataset_config = config.get('datasets', {}).get(dataset, {})
    overrides = dataset_config.get('signal_overrides', {})
    if overrides:
        raise ValueError(
            f'Dataset-specific signal overrides are forbidden by the frozen track protocol: {dataset}'
        )
    track = str(config.get('protocol_track', ''))
    if track not in OFFICIAL_TRACK_BANDPASS_HZ:
        raise ValueError(f'Unsupported or missing protocol_track: {track}')
    expected_bandpass = OFFICIAL_TRACK_BANDPASS_HZ[track]
    actual_bandpass = (float(signal['l_freq']), float(signal['h_freq']))
    if actual_bandpass != expected_bandpass:
        raise ValueError(
            f'{track} requires bandpass {expected_bandpass}, got {actual_bandpass}'
        )
    policy = str(signal.get('line_noise_policy', ''))
    if policy != 'dataset_native_spectrum_fit':
        raise ValueError(
            f'Frozen track protocol requires dataset_native_spectrum_fit: {policy}'
        )
    if str(signal.get('notch_method', '')) != 'spectrum_fit':
        raise ValueError('Frozen track protocol requires spectrum_fit notch_method')
    line_frequency = float(dataset_config.get('line_frequency_hz', 0.0))
    if line_frequency not in {50.0, 60.0}:
        raise ValueError(
            f'{dataset} requires an explicit native line_frequency_hz of 50 or 60'
        )
    if str(signal.get('resampling_method', '')) != 'fft':
        raise ValueError('Frozen track protocol requires anti-aliased FFT resampling')
    signal['protocol_track'] = track
    signal['line_noise_policy'] = policy
    signal['line_frequency_hz'] = line_frequency
    signal['notch_freqs'] = [line_frequency]
    return signal


def stable_fingerprint(config: dict[str, Any], dataset: str) -> str:
    payload = {
        "preprocessing_protocol": SCALP_PREPROCESSING_PROTOCOL,
        "dataset": dataset,
        "signal": resolved_signal_config(config, dataset),
        "window_seconds": configured_window_seconds(config),
        "detection_positive_operator": ">",
        "detection_min_overlap_seconds": 0.0,
        "detection_seizure_annotated_recordings_only": (
            dataset in DETECTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY
        ),
        "prediction_seizure_annotated_recordings_only": (
            dataset in PREDICTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY
        ),
        "prediction_patient_continuous_recording": dataset == "chbmit",
        "output_layout": OUTPUT_LAYOUT_POLICY,
        "timeline_policy": (
            "patient_file_order_gapless_with_cross_edf_clips"
            if dataset == "chbmit"
            else "independent_edf_sessions"
            if dataset == "siena"
            else "timestamp_contiguous_components"
        ),
        "prediction_rule": asdict(PREDICTION_RULES[dataset]),
        "split": config["split"],
        "qc": config["qc"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def deterministic_patient_split(patient_id: str, config: dict[str, Any]) -> str:
    split = config["split"]
    score = int(hashlib.sha256(patient_id.encode()).hexdigest()[:16], 16) / 16**16
    train_ratio = float(split.get("train_ratio", 0.8))
    dev_ratio = float(split.get("dev_ratio", 0.1))
    if score < train_ratio:
        return "train"
    if score < train_ratio + dev_ratio:
        return "dev"
    return "test"


def patient_split_from_contract(root: Path, patient_id: str, split_config: dict[str, Any]) -> str:
    manifest_path = root / "patient_splits" / "patient_split.csv"
    if manifest_path.exists():
        frame = pd.read_csv(manifest_path, dtype=str)
        required = {"patient_id", "split"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Patient split manifest lacks required columns: {manifest_path}")
        matches = frame[frame["patient_id"] == patient_id]
        if len(matches) != 1:
            raise ValueError(f"Patient split must contain exactly one row for {patient_id}: {manifest_path}")
        split = str(matches.iloc[0]["split"])
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Invalid split {split} for {patient_id}: {manifest_path}")
        return split
    return deterministic_patient_split(patient_id, {"split": split_config})


def open_raw_recording(path: Path, preload: bool) -> Any:
    import mne

    if path.suffix.lower() == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=preload, verbose="ERROR")
    if path.suffix.lower() in {".edf", ".bdf"}:
        return mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported electrophysiology file: {path}")


def raw_header(path: Path) -> tuple[float, float, float | None]:
    raw = open_raw_recording(path, preload=False)
    sfreq = float(raw.info["sfreq"])
    duration = float(raw.n_times / sfreq)
    meas_date = raw.info.get("meas_date")
    timestamp = float(meas_date.timestamp()) if isinstance(meas_date, datetime) else None
    raw.close()
    return sfreq, duration, timestamp


def read_tusz_csv_bi(path: Path, issues: list[dict[str, Any]]) -> list[SeizureInterval]:
    if not path.exists():
        issues.append({"level": "warning", "code": "missing_tusz_csv_bi", "path": str(path)})
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [line for line in stream if not line.lstrip().startswith("#") and line.strip()]
    reader = csv.DictReader(rows)
    seizures = []
    for index, row in enumerate(reader, start=1):
        if str(row.get("label", "")).strip().lower() != "seiz":
            continue
        try:
            seizures.append(
                SeizureInterval(
                    float(row["start_time"]),
                    float(row["stop_time"]),
                    f"{path.stem}:seizure-{index}",
                    path.as_posix(),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                {"level": "error", "code": "invalid_tusz_annotation", "path": str(path), "row": row, "error": str(exc)}
            )
    return merge_duplicate_seizures(seizures)


def scan_tusz(root: Path, split_config: dict[str, Any], issues: list[dict[str, Any]]) -> list[Recording]:
    recordings = []
    for path in sorted(root.rglob("*.edf")):
        relative = path.relative_to(root)
        parts = relative.parts
        split_index = next((i for i, part in enumerate(parts) if part.lower() in SPLIT_ALIASES), None)
        if split_index is None or split_index + 1 >= len(parts):
            issues.append({"level": "error", "code": "unresolved_tusz_patient", "path": str(path)})
            continue
        source_split = parts[split_index].lower()
        patient = parts[split_index + 1]
        session = parts[split_index + 2] if split_index + 2 < len(parts) - 1 else "session_0"
        montage = parts[split_index + 3] if split_index + 3 < len(parts) - 1 else "unknown"
        output_subdir = Path(*parts[split_index + 2 : -1])
        sfreq, duration, timestamp = raw_header(path)
        recordings.append(
            Recording(
                dataset="tusz",
                path=path,
                source_root=root,
                patient_id=patient,
                split=SPLIT_ALIASES[source_split],
                session_id=session,
                montage=montage,
                output_subdir=output_subdir,
                duration_seconds=duration,
                source_sfreq=sfreq,
                start_timestamp=timestamp,
                seizures=read_tusz_csv_bi(path.with_suffix(".csv_bi"), issues),
            )
        )
    return recordings


TIME_PATTERN = re.compile(r"(?<!\d)([0-2]?\d)\s*[.:]\s*([0-5]\d)\s*[.:]\s*([0-5]\d)(?!\d)")


def clock_values(text: str) -> list[float]:
    normalized = re.sub(r"(?<=\d)\s+(?=\d[.:])", "", text)
    values = []
    for hour, minute, second in TIME_PATTERN.findall(normalized):
        values.append(int(hour) * 3600.0 + int(minute) * 60.0 + int(second))
    return values


def normalize_siena_filename(name: str) -> str:
    cleaned = name.strip().replace("PNO", "PN0").replace("pno", "pn0")
    return cleaned


def resolve_siena_filename(name: str, available: dict[str, Path]) -> tuple[Path | None, str | None]:
    normalized = normalize_siena_filename(name)
    exact = available.get(normalized.lower())
    if exact is not None:
        return exact, None
    if normalized.lower().endswith("-.edf"):
        prefix = normalized[:-5].lower()
        matches = [path for key, path in available.items() if key.startswith(prefix)]
        if len(matches) == 1:
            return matches[0], f"corrected {name} to {matches[0].name}"
    if normalized.lower().endswith(".edf"):
        stem = Path(normalized).stem.lower()
        matches = [
            path
            for path in available.values()
            if path.stem.lower().startswith(f"{stem}-")
        ]
        if len(matches) == 1:
            return matches[0], f"corrected {name} to {matches[0].name}"
    stem_key = re.sub(r"[^a-z0-9]", "", Path(normalized).stem.lower())
    matches = [path for key, path in available.items() if re.sub(r"[^a-z0-9]", "", Path(key).stem.lower()) == stem_key]
    if len(matches) == 1:
        return matches[0], f"corrected {name} to {matches[0].name}"
    return None, None


def parse_siena_annotations(patient_dir: Path, issues: list[dict[str, Any]]) -> dict[Path, list[SeizureInterval]]:
    text_files = sorted(patient_dir.glob("Seizures-list-*.txt"))
    if not text_files:
        issues.append({"level": "warning", "code": "missing_siena_annotation", "path": str(patient_dir)})
        return {}
    available = {path.name.lower(): path for path in patient_dir.glob("*.edf")}
    text_path = text_files[0]
    text = text_path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"(?=\bSeizure\s+n\s*\d+)", text, flags=re.IGNORECASE)
    events: dict[Path, list[SeizureInterval]] = defaultdict(list)
    registration_by_file: dict[Path, float] = {}
    current_path: Path | None = None
    header_file_match = re.search(
        r"File\s+name\s*:\s*([^\r\n]+)",
        blocks[0],
        flags=re.IGNORECASE,
    )
    if header_file_match:
        current_path, correction = resolve_siena_filename(
            header_file_match.group(1).strip(),
            available,
        )
        if correction:
            issues.append(
                {
                    "level": "warning",
                    "code": "siena_filename_correction",
                    "path": str(text_path),
                    "detail": correction,
                }
            )
        header_registration_match = re.search(
            r"Registration\s+start\s+time\s*:\s*([^\r\n]+)",
            blocks[0],
            flags=re.IGNORECASE,
        )
        if current_path is not None and header_registration_match:
            registration_values = clock_values(header_registration_match.group(1))
            if registration_values:
                registration_by_file[current_path] = registration_values[0]
    for block_index, block in enumerate(blocks, start=1):
        file_match = re.search(r"File\s+name\s*:\s*([^\r\n]+)", block, flags=re.IGNORECASE)
        onset_match = re.search(
            r"^\s*(?:Seizure\s+)?Start\s+time\s*:\s*([^\r\n]+)",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        end_match = re.search(
            r"^\s*(?:Seizure\s+)?End\s+time\s*:\s*([^\r\n]+)",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not onset_match or not end_match:
            continue
        correction = None
        if file_match:
            declared_name = file_match.group(1).strip()
            edf_path, correction = resolve_siena_filename(declared_name, available)
            current_path = edf_path
        else:
            declared_name = current_path.name if current_path is not None else ""
            edf_path = current_path
        if edf_path is None:
            issues.append(
                {"level": "error", "code": "unresolved_siena_filename", "path": str(text_path), "declared": declared_name}
            )
            continue
        if correction:
            issues.append({"level": "warning", "code": "siena_filename_correction", "path": str(text_path), "detail": correction})
        registration_match = re.search(r"Registration\s+start\s+time\s*:\s*([^\r\n]+)", block, flags=re.IGNORECASE)
        if registration_match:
            registration_values = clock_values(registration_match.group(1))
            if registration_values:
                registration_by_file[edf_path] = registration_values[0]
        registration = registration_by_file.get(edf_path)
        onset_values = clock_values(onset_match.group(1))
        end_values = clock_values(end_match.group(1))
        if registration is None or not onset_values or not end_values:
            issues.append(
                {"level": "error", "code": "invalid_siena_time", "path": str(text_path), "block": block_index}
            )
            continue
        onset_clock = onset_values[-1] if "electric onset" in onset_match.group(1).lower() else onset_values[0]
        if len(onset_values) > 1 or len(end_values) > 1:
            issues.append(
                {
                    "level": "warning",
                    "code": "ambiguous_siena_time",
                    "path": str(text_path),
                    "block": block_index,
                    "onset_text": onset_match.group(1).strip(),
                    "end_text": end_match.group(1).strip(),
                    "selected_onset_clock": onset_clock,
                    "selected_end_clock": end_values[0],
                }
            )
        onset = onset_clock - registration
        while onset < 0:
            onset += 86400.0
        end = end_values[0] - registration
        while end <= onset:
            end += 86400.0
        try:
            events[edf_path].append(
                SeizureInterval(onset, end, f"{patient_dir.name}:seizure-{block_index}", text_path.as_posix())
            )
        except ValueError as exc:
            issues.append({"level": "error", "code": "invalid_siena_interval", "path": str(text_path), "error": str(exc)})
    return {path: merge_duplicate_seizures(values) for path, values in events.items()}


def apply_siena_annotation_overrides(
    patient_dir: Path,
    annotations: dict[Path, list[SeizureInterval]],
    overrides: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    patient_overrides = overrides.get(patient_dir.name, {})
    if not isinstance(patient_overrides, dict):
        issues.append(
            {"level": "error", "code": "invalid_siena_override_patient", "patient_id": patient_dir.name}
        )
        return
    for filename, rows in patient_overrides.items():
        path = patient_dir / filename
        if not path.exists() or not isinstance(rows, list):
            issues.append(
                {"level": "error", "code": "invalid_siena_override_target", "path": str(path)}
            )
            continue
        replacement = []
        try:
            for index, row in enumerate(rows, start=1):
                replacement.append(
                    SeizureInterval(
                        float(row["start_seconds"]),
                        float(row["end_seconds"]),
                        f"{patient_dir.name}:{filename}:override-{index}",
                        "official_track_config",
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                {
                    "level": "error",
                    "code": "invalid_siena_override_interval",
                    "path": str(path),
                    "error": str(exc),
                }
            )
            continue
        annotations[path] = replacement
        issues.append(
            {
                "level": "warning",
                "code": "siena_annotation_override_applied",
                "path": str(path),
                "interval_count": len(replacement),
            }
        )


def scan_siena(
    root: Path,
    split_config: dict[str, Any],
    issues: list[dict[str, Any]],
    annotation_overrides: dict[str, Any] | None = None,
) -> list[Recording]:
    recordings = []
    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.upper().startswith("PN")):
        annotations = parse_siena_annotations(patient_dir, issues)
        apply_siena_annotation_overrides(patient_dir, annotations, annotation_overrides or {}, issues)
        split = patient_split_from_contract(root, patient_dir.name, split_config)
        for path in sorted(patient_dir.glob("*.edf")):
            sfreq, duration, timestamp = raw_header(path)
            seizures = []
            for event in annotations.get(path, []):
                if event.end_seconds <= duration + 1.0:
                    seizures.append(event)
                else:
                    issues.append(
                        {
                            "level": "error",
                            "code": "siena_event_out_of_bounds",
                            "path": str(path),
                            "start_seconds": event.start_seconds,
                            "end_seconds": event.end_seconds,
                            "duration_seconds": duration,
                        }
                    )
            recordings.append(
                Recording(
                    dataset="siena",
                    path=path,
                    source_root=root,
                    patient_id=patient_dir.name,
                    split=split,
                    session_id=path.stem,
                    montage="native_siena",
                    output_subdir=Path(),
                    duration_seconds=duration,
                    source_sfreq=sfreq,
                    start_timestamp=timestamp,
                    seizures=seizures,
                )
            )
    return recordings


def parse_chb_summary(path: Path, issues: list[dict[str, Any]]) -> dict[str, list[SeizureInterval]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"(?=File\s+Name\s*:)", text, flags=re.IGNORECASE)
    events: dict[str, list[SeizureInterval]] = defaultdict(list)
    for block in blocks:
        file_match = re.search(r"File\s+Name\s*:\s*([^\r\n]+)", block, flags=re.IGNORECASE)
        if not file_match:
            continue
        filename = file_match.group(1).strip()
        starts = [float(value) for value in re.findall(r"Seizure\s+\d*\s*Start\s+Time\s*:\s*(\d+(?:\.\d+)?)\s*seconds", block, flags=re.IGNORECASE)]
        ends = [float(value) for value in re.findall(r"Seizure\s+\d*\s*End\s+Time\s*:\s*(\d+(?:\.\d+)?)\s*seconds", block, flags=re.IGNORECASE)]
        if len(starts) != len(ends):
            issues.append(
                {"level": "error", "code": "chb_annotation_count_mismatch", "path": str(path), "file": filename, "starts": starts, "ends": ends}
            )
            continue
        for index, (start, end) in enumerate(zip(starts, ends), start=1):
            try:
                events[filename].append(
                    SeizureInterval(start, end, f"{path.parent.name}:{filename}:seizure-{index}", path.as_posix())
                )
            except ValueError as exc:
                issues.append({"level": "error", "code": "invalid_chb_interval", "path": str(path), "error": str(exc)})
    return dict(events)


def scan_chbmit(root: Path, split_config: dict[str, Any], issues: list[dict[str, Any]]) -> list[Recording]:
    recordings = []
    existing_split_roots = {
        split: root / split
        for split in ("train", "dev", "test")
        if (root / split).is_dir()
    }
    patient_entries: list[tuple[Path, str]] = []
    if existing_split_roots:
        missing_splits = sorted({"train", "dev", "test"} - set(existing_split_roots))
        if missing_splits:
            raise RuntimeError(f"Incomplete CHB-MIT split directories under {root}: {missing_splits}")
        seen_patients: dict[str, str] = {}
        for split in ("train", "dev", "test"):
            for patient_dir in sorted(
                path
                for path in existing_split_roots[split].iterdir()
                if path.is_dir() and path.name.lower().startswith("chb")
            ):
                previous_split = seen_patients.get(patient_dir.name)
                if previous_split is not None:
                    raise RuntimeError(
                        f"CHB-MIT patient appears in multiple splits: {patient_dir.name} in {previous_split} and {split}"
                    )
                seen_patients[patient_dir.name] = split
                patient_entries.append((patient_dir, split))
        if not patient_entries:
            raise RuntimeError(f"No CHB-MIT patient directories found under existing train/dev/test splits: {root}")
    else:
        for patient_dir in sorted(
            path for path in root.iterdir() if path.is_dir() and path.name.lower().startswith("chb")
        ):
            patient_entries.append(
                (patient_dir, patient_split_from_contract(root, patient_dir.name, split_config))
            )

    for patient_dir, split in patient_entries:
        summaries = sorted(patient_dir.glob("*-summary.txt"))
        if not summaries:
            issues.append({"level": "warning", "code": "missing_chb_summary", "path": str(patient_dir)})
            annotation_map = {}
        else:
            annotation_map = parse_chb_summary(summaries[0], issues)
        for path in sorted(patient_dir.glob("*.edf")):
            sfreq, duration, timestamp = raw_header(path)
            recordings.append(
                Recording(
                    dataset="chbmit",
                    path=path,
                    source_root=root,
                    patient_id=patient_dir.name,
                    split=split,
                    session_id=path.stem,
                    montage="native_bipolar",
                    output_subdir=Path(),
                    duration_seconds=duration,
                    source_sfreq=sfreq,
                    start_timestamp=timestamp,
                    seizures=merge_duplicate_seizures(annotation_map.get(path.name, [])),
                )
            )
    return recordings


def bids_partner(path: Path, current_suffix: str, target_suffix: str) -> Path:
    if not path.name.endswith(current_suffix):
        raise ValueError(f"BIDS filename does not end with {current_suffix}: {path}")
    return path.with_name(path.name[: -len(current_suffix)] + target_suffix)


def normalized_event_rows(path: Path) -> list[tuple[float, str]]:
    table = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    if not {"onset", "trial_type"}.issubset(table.columns):
        raise ValueError(f"BIDS events sidecar lacks onset or trial_type: {path}")
    rows = []
    for row in table.itertuples(index=False):
        try:
            onset = float(row.onset)
        except (TypeError, ValueError):
            continue
        rows.append((onset, re.sub(r"\s+", " ", str(row.trial_type).strip().lower())))
    return rows


def parse_epilepsy_ieeg_interval(
    events_path: Path,
    duration_seconds: float,
    issues: list[dict[str, Any]],
) -> SeizureInterval | None:
    rows = normalized_event_rows(events_path)
    onset_pattern = re.compile(
        r"1st change|electrographic ons|elelctrographic on|ictal onset|eeg onset|eeg sz start|"
        r"definite onset|sz onset|poss .*onset|ictal build|sz event|\bonset\b|^start$|^sz$|"
        r"^seizure #\d+$|onset tt|ad1-3 onset|seizure #\d+ onset|sz #\d+ onset|sz onset #\d+"
    )
    onset_candidates = [time for time, label in rows if onset_pattern.search(label) and "clinical onset" not in label]
    if not onset_candidates:
        onset_candidates = [time for time, label in rows if "clinical onset" in label]
    explicit_offset_pattern = re.compile(
        r"offset|eeg sz end|electrographic end|sz end|seizure off|definite off|generalized off"
    )
    fallback_offset_pattern = re.compile(
        r"^end$|end fast|seizure over|event over|clinical end|(^| )z? ?over($| on)|"
        r"z ending|z stopping|dissipating|devolves|devolution|post.?ictal depress"
    )
    offset_candidates = [time for time, label in rows if explicit_offset_pattern.search(label)]
    used_fallback = False
    if not offset_candidates:
        offset_candidates = [time for time, label in rows if fallback_offset_pattern.search(label)]
        used_fallback = bool(offset_candidates)
    if not onset_candidates or not offset_candidates:
        issues.append(
            {
                "level": "warning",
                "code": "excluded_epilepsy_ieeg_missing_boundary",
                "path": str(events_path),
                "onset_found": bool(onset_candidates),
                "offset_found": bool(offset_candidates),
            }
        )
        return None
    onset = min(onset_candidates)
    offset = max(offset_candidates)
    if used_fallback:
        issues.append(
            {
                "level": "warning",
                "code": "epilepsy_ieeg_fallback_offset_marker",
                "path": str(events_path),
                "selected_offset_seconds": offset,
            }
        )
    if not 0.0 <= onset < offset <= duration_seconds + 1e-3:
        issues.append(
            {
                "level": "warning",
                "code": "excluded_epilepsy_ieeg_boundary_out_of_range",
                "path": str(events_path),
                "onset_seconds": onset,
                "offset_seconds": offset,
                "duration_seconds": duration_seconds,
            }
        )
        return None
    return SeizureInterval(onset, offset, f"{events_path.stem}:seizure-1", events_path.as_posix())


def parse_exact_bids_interval(
    dataset: str,
    events_path: Path,
    duration_seconds: float,
    issues: list[dict[str, Any]],
) -> SeizureInterval | None:
    expected = {
        "hup_ieeg": ("sz onset", "sz offset"),
        "thalamocortical_ieeg": ("seizure_onset", "seizure_offset"),
    }
    onset_label, offset_label = expected[dataset]
    rows = normalized_event_rows(events_path)
    onsets = [time for time, label in rows if label == onset_label]
    offsets = [time for time, label in rows if label == offset_label]
    if not onsets or not offsets:
        issues.append(
            {
                "level": "warning",
                "code": f"excluded_{dataset}_missing_boundary",
                "path": str(events_path),
            }
        )
        return None
    onset = min(onsets)
    offset = max(offsets)
    if not 0.0 <= onset < offset <= duration_seconds + 1e-3:
        issues.append(
            {
                "level": "warning",
                "code": f"excluded_{dataset}_boundary_out_of_range",
                "path": str(events_path),
                "onset_seconds": onset,
                "offset_seconds": offset,
                "duration_seconds": duration_seconds,
            }
        )
        return None
    return SeizureInterval(onset, offset, f"{events_path.stem}:seizure-1", events_path.as_posix())


def scan_ieeg_dataset(
    dataset: str,
    root: Path,
    split_config: dict[str, Any],
    issues: list[dict[str, Any]],
) -> list[Recording]:
    recordings = []
    for json_path in sorted(root.rglob("*_ieeg.json")):
        metadata = json.loads(json_path.read_text(encoding="utf-8-sig"))
        task_name = str(metadata.get("TaskName", "")).strip().lower()
        signal_suffix = "_ieeg.vhdr" if dataset == "epilepsy_ieeg" else "_ieeg.edf"
        signal_path = bids_partner(json_path, "_ieeg.json", signal_suffix)
        if not signal_path.exists():
            issues.append(
                {
                    "level": "warning",
                    "code": f"excluded_{dataset}_missing_signal",
                    "path": str(signal_path),
                }
            )
            continue
        events_path = bids_partner(json_path, "_ieeg.json", "_events.tsv")
        channels_path = bids_partner(json_path, "_ieeg.json", "_channels.tsv")
        if not channels_path.exists():
            issues.append(
                {
                    "level": "warning",
                    "code": f"excluded_{dataset}_missing_channels",
                    "path": str(channels_path),
                }
            )
            continue
        sidecar_duration = float(metadata["RecordingDuration"])
        sidecar_sfreq = float(metadata["SamplingFrequency"])
        try:
            source_sfreq, duration, _ = raw_header(signal_path)
        except Exception as exc:
            issues.append(
                {
                    "level": "warning",
                    "code": f"excluded_{dataset}_unreadable_signal_header",
                    "path": str(signal_path),
                    "error": str(exc),
                }
            )
            continue
        if abs(duration - sidecar_duration) > 1.0 or not math.isclose(
            source_sfreq,
            sidecar_sfreq,
            rel_tol=0.01,
            abs_tol=5.0,
        ):
            issues.append(
                {
                    "level": "warning",
                    "code": f"excluded_{dataset}_signal_sidecar_mismatch",
                    "path": str(signal_path),
                    "raw_duration_seconds": duration,
                    "sidecar_duration_seconds": sidecar_duration,
                    "raw_sfreq": source_sfreq,
                    "sidecar_sfreq": sidecar_sfreq,
                }
            )
            continue
        seizures = []
        is_interictal = task_name == "interictal"
        is_ictal = task_name in {"ictal", "seizure"}
        if is_ictal:
            if not events_path.exists():
                issues.append(
                    {
                        "level": "warning",
                        "code": f"excluded_{dataset}_missing_events",
                        "path": str(events_path),
                    }
                )
                continue
            if dataset == "epilepsy_ieeg":
                event = parse_epilepsy_ieeg_interval(events_path, duration, issues)
            else:
                event = parse_exact_bids_interval(dataset, events_path, duration, issues)
            if event is None:
                continue
            seizures = [event]
        elif not is_interictal:
            issues.append(
                {
                    "level": "warning",
                    "code": f"excluded_{dataset}_unsupported_task",
                    "path": str(json_path),
                    "task_name": task_name,
                }
            )
            continue
        relative = signal_path.relative_to(root)
        patient = next((part for part in relative.parts if part.startswith("sub-")), None)
        session = next((part for part in relative.parts if part.startswith("ses-")), "ses-unknown")
        if patient is None:
            issues.append(
                {"level": "warning", "code": f"excluded_{dataset}_missing_patient", "path": str(signal_path)}
            )
            continue
        acquisition_match = re.search(r"_acq-([^_]+)", signal_path.name)
        acquisition = acquisition_match.group(1) if acquisition_match else "ieeg"
        electrode_candidates = sorted(signal_path.parent.glob("*_electrodes.tsv"))
        electrode_sidecar = electrode_candidates[0] if len(electrode_candidates) == 1 else None
        coordsystem_sidecar = None
        if electrode_sidecar is not None:
            candidate = electrode_sidecar.with_name(
                electrode_sidecar.name.replace("_electrodes.tsv", "_coordsystem.json")
            )
            if candidate.exists():
                coordsystem_sidecar = candidate
        recording = Recording(
            dataset=dataset,
            path=signal_path,
            source_root=root,
            patient_id=patient,
            split=patient_split_from_contract(root, patient, split_config),
            session_id=session,
            montage=f"native_bids_{acquisition}",
            output_subdir=Path(session, "ieeg"),
            duration_seconds=duration,
            source_sfreq=source_sfreq,
            start_timestamp=None,
            channel_sidecar=channels_path,
            electrode_sidecar=electrode_sidecar,
            coordsystem_sidecar=coordsystem_sidecar,
            seizures=seizures,
            component_id=f"{patient}:{session}:{signal_path.stem}",
            timeline_start=0.0,
        )
        recordings.append(recording)
    return recordings


def merge_duplicate_seizures(seizures: Iterable[SeizureInterval]) -> list[SeizureInterval]:
    ordered = sorted(seizures, key=lambda event: (event.start_seconds, event.end_seconds))
    result = []
    for event in ordered:
        if result and abs(result[-1].start_seconds - event.start_seconds) < 1e-6 and abs(result[-1].end_seconds - event.end_seconds) < 1e-6:
            continue
        result.append(event)
    return result


def partition_seizure_annotated_recordings(
    recordings: Sequence[Recording],
    dataset: str,
) -> tuple[list[Recording], list[Recording]]:
    if dataset == "chbmit":
        patient_has_seizure = {
            recording.patient_id
            for recording in recordings
            if recording.seizures
        }
        eligible = [
            recording
            for recording in recordings
            if recording.patient_id in patient_has_seizure
        ]
        excluded = [
            recording
            for recording in recordings
            if recording.patient_id not in patient_has_seizure
        ]
        return eligible, excluded
    if dataset not in PREDICTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY:
        return list(recordings), []
    return partition_recordings_by_seizure_annotation(recordings)


def partition_detection_recordings(
    recordings: Sequence[Recording],
    dataset: str,
) -> tuple[list[Recording], list[Recording]]:
    if dataset not in DETECTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY:
        return list(recordings), []
    return partition_recordings_by_seizure_annotation(recordings)


def partition_recordings_by_seizure_annotation(
    recordings: Sequence[Recording],
) -> tuple[list[Recording], list[Recording]]:
    eligible = [recording for recording in recordings if recording.seizures]
    excluded = [recording for recording in recordings if not recording.seizures]
    return eligible, excluded


def scan_dataset(dataset: str, root: Path, config: dict[str, Any], issues: list[dict[str, Any]]) -> list[Recording]:
    if dataset == "siena":
        overrides = config["datasets"]["siena"].get("annotation_overrides", {})
        return scan_siena(root, config["split"], issues, overrides)
    scanners = {"tusz": scan_tusz, "chbmit": scan_chbmit}
    if dataset not in scanners:
        raise KeyError(f"Unsupported EEG dataset: {dataset}")
    return scanners[dataset](root, config["split"], issues)


def assign_timeline_components(recordings: list[Recording], max_gap_seconds: float, issues: list[dict[str, Any]]) -> None:
    by_patient: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        by_patient[recording.patient_id].append(recording)
    for patient, patient_recordings in by_patient.items():
        if patient_recordings and all(
            recording.dataset == "siena" for recording in patient_recordings
        ):
            for recording in patient_recordings:
                recording.component_id = f"{patient}:{recording.session_id}"
                recording.timeline_start = 0.0
            continue
        if patient_recordings and all(
            recording.dataset == "chbmit" for recording in patient_recordings
        ):
            timeline_start = 0.0
            ordered = sorted(
                patient_recordings,
                key=lambda item: item.relative_path,
            )
            for recording in ordered:
                recording.component_id = f"{patient}:continuous-recording"
                recording.timeline_start = timeline_start
                timeline_start += recording.duration_seconds
            issues.append(
                {
                    "level": "info",
                    "code": "chb_patient_edf_timeline_stitched",
                    "patient_id": patient,
                    "recording_count": len(ordered),
                    "timeline_duration_seconds": timeline_start,
                    "policy": "file_order_gapless",
                }
            )
            continue
        valid = sorted((recording for recording in patient_recordings if recording.start_timestamp is not None), key=lambda item: float(item.start_timestamp))
        missing = sorted((recording for recording in patient_recordings if recording.start_timestamp is None), key=lambda item: item.relative_path)
        component_index = 0
        previous: Recording | None = None
        for recording in valid:
            if previous is None:
                component_index += 1
            else:
                gap = float(recording.start_timestamp) - (float(previous.start_timestamp) + previous.duration_seconds)
                if gap > max_gap_seconds or gap < -1.0:
                    component_index += 1
            recording.component_id = f"{patient}:component-{component_index:04d}"
            recording.timeline_start = float(recording.start_timestamp)
            previous = recording
        for recording in missing:
            component_index += 1
            recording.component_id = f"{patient}:component-{component_index:04d}"
            recording.timeline_start = 0.0
            issues.append({"level": "warning", "code": "missing_recording_timestamp", "path": str(recording.path)})


def detection_candidates(
    recording: Recording,
    window_seconds: float = WINDOW_SECONDS,
) -> list[ClipCandidate]:
    candidates = []
    count = int(recording.duration_seconds // window_seconds)
    for index in range(count):
        start = index * window_seconds
        end = start + window_seconds
        overlaps = [max(0.0, min(end, event.end_seconds) - max(start, event.start_seconds)) for event in recording.seizures]
        best_overlap = max(overlaps, default=0.0)
        label = int(best_overlap > 0.0)
        event_id = ""
        if overlaps and best_overlap > 0:
            event_id = recording.seizures[int(np.argmax(overlaps))].seizure_id
        candidates.append(
            ClipCandidate("detection", str(recording.path), start, end, label, TASK_LABELS["detection"][label], event_id, best_overlap)
        )
    return candidates


def first_aligned_start(
    lower: float,
    origin: float,
    window_seconds: float = WINDOW_SECONDS,
) -> float:
    if lower <= origin:
        return origin
    return origin + math.ceil((lower - origin) / window_seconds - 1e-12) * window_seconds


def add_interval_candidates(
    output: list[ClipCandidate],
    recordings: Sequence[Recording],
    interval_start: float,
    interval_end: float,
    label: int,
    class_name: str,
    event_id: str,
    task_name: str = "prediction",
    window_seconds: float = WINDOW_SECONDS,
) -> None:
    if interval_end - interval_start < window_seconds:
        return
    if not recordings or not all(
        recording.dataset == "chbmit" for recording in recordings
    ):
        for recording in recordings:
            lower = max(interval_start, recording.timeline_start)
            upper = min(interval_end, recording.timeline_end)
            start = first_aligned_start(lower, interval_start, window_seconds)
            while start + window_seconds <= upper + 1e-7:
                local_start = start - recording.timeline_start
                output.append(
                    ClipCandidate(
                        task_name,
                        str(recording.path),
                        local_start,
                        local_start + window_seconds,
                        label,
                        class_name,
                        event_id,
                        0.0,
                    )
                )
                start += window_seconds
        return
    ordered = sorted(recordings, key=lambda item: item.timeline_start)
    start = first_aligned_start(interval_start, interval_start, window_seconds)
    while start + window_seconds <= interval_end + 1e-7:
        end = start + window_seconds
        segments = []
        covered_seconds = 0.0
        for recording in ordered:
            overlap_start = max(start, recording.timeline_start)
            overlap_end = min(end, recording.timeline_end)
            if overlap_end <= overlap_start:
                continue
            local_start = overlap_start - recording.timeline_start
            local_end = overlap_end - recording.timeline_start
            segments.append(
                ClipSegment(
                    str(recording.path),
                    local_start,
                    local_end,
                )
            )
            covered_seconds += overlap_end - overlap_start
        if segments and math.isclose(
            covered_seconds,
            window_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            first_segment = segments[0]
            output.append(
                ClipCandidate(
                    task_name,
                    first_segment.recording_path,
                    first_segment.local_start_seconds,
                    first_segment.local_start_seconds + window_seconds,
                    label,
                    class_name,
                    event_id,
                    0.0,
                    tuple(segments),
                )
            )
        start += window_seconds


def prediction_candidates_with_rule(
    recordings: list[Recording],
    rule: PredictionRule,
    task_name: str = "prediction",
    window_seconds: float = WINDOW_SECONDS,
) -> list[ClipCandidate]:
    by_component: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        by_component[recording.component_id].append(recording)
    output: list[ClipCandidate] = []
    for component_recordings in by_component.values():
        component_recordings.sort(key=lambda item: item.timeline_start)
        component_start = min(item.timeline_start for item in component_recordings)
        component_end = max(item.timeline_end for item in component_recordings)
        seizures = []
        for recording in component_recordings:
            for event in recording.seizures:
                seizures.append(
                    (
                        recording.timeline_start + event.start_seconds,
                        recording.timeline_start + event.end_seconds,
                        event.seizure_id,
                    )
                )
        seizures.sort()
        if not seizures:
            if rule.include_no_seizure_components and not rule.require_both_neighbor_seizures:
                add_interval_candidates(
                    output,
                    component_recordings,
                    component_start,
                    component_end,
                    0,
                    "interictal",
                    "no-seizure-component",
                    task_name,
                    window_seconds,
                )
            continue
        preictal_starts = []
        for index, (onset, offset, event_id) in enumerate(seizures):
            start = onset - rule.preictal_buffer_seconds - rule.preictal_seconds
            end = onset - rule.preictal_buffer_seconds
            if index > 0:
                previous_offset = seizures[index - 1][1]
                if start < previous_offset:
                    start = previous_offset
            preictal_starts.append(onset - rule.preictal_buffer_seconds - rule.preictal_seconds)
            add_interval_candidates(
                output,
                component_recordings,
                start,
                end,
                1,
                "preictal",
                event_id,
                task_name,
                window_seconds,
            )
        if rule.require_both_neighbor_seizures:
            separation = float(rule.interictal_separation_seconds or 0.0)
            for previous, following in zip(seizures[:-1], seizures[1:]):
                add_interval_candidates(
                    output,
                    component_recordings,
                    previous[1] + rule.postictal_buffer_seconds,
                    following[0] - separation,
                    0,
                    "interictal",
                    f"between:{previous[2]}:{following[2]}",
                    task_name,
                    window_seconds,
                )
        else:
            first_index = 0 if rule.include_leading_interictal else 1
            for index in range(first_index, len(seizures) + 1):
                lower = component_start if index == 0 else seizures[index - 1][1] + rule.postictal_buffer_seconds
                if index == len(seizures):
                    upper = component_end
                elif index == 0 or rule.interictal_separation_seconds is None:
                    upper = preictal_starts[index]
                else:
                    upper = seizures[index][0] - rule.interictal_separation_seconds
                if rule.interictal_seconds is not None:
                    upper = min(upper, lower + rule.interictal_seconds)
                left_id = "boundary" if index == 0 else seizures[index - 1][2]
                right_id = "boundary" if index == len(seizures) else seizures[index][2]
                add_interval_candidates(
                    output,
                    component_recordings,
                    lower,
                    upper,
                    0,
                    "interictal",
                    f"between:{left_id}:{right_id}",
                    task_name,
                    window_seconds,
                )
    deduplicated = {}
    for candidate in output:
        key = (candidate.recording_path, round(candidate.local_start_seconds, 6), candidate.label)
        deduplicated[key] = candidate
    return sorted(deduplicated.values(), key=lambda item: (item.recording_path, item.local_start_seconds, item.label))


def prediction_candidates(
    recordings: list[Recording],
    dataset: str,
    window_seconds: float = WINDOW_SECONDS,
) -> list[ClipCandidate]:
    eligible_recordings, _ = partition_seizure_annotated_recordings(recordings, dataset)
    return prediction_candidates_with_rule(
        eligible_recordings,
        PREDICTION_RULES[dataset],
        window_seconds=window_seconds,
    )


AUXILIARY_CHANNEL_PATTERN = re.compile(
    r"(^|[\s_-])(ECG\d*|EKG\d*|EMG\d*|EOG\d*|SPO2|PLET|PULSE|PHOTIC|TRIG|TRIGGER|STATUS|ANNOTATION|IBI|HEART|HR|BURSTS?|SUPPR(?:ESSION)?)([\s_-]|$)",
    flags=re.IGNORECASE,
)
PLACEHOLDER_CHANNEL_PATTERN = re.compile(r"^--\d*$")


def select_eeg_channel_names(raw: Any, dataset: str | None = None) -> list[str]:
    names = []
    channel_types = raw.get_channel_types()
    has_explicit_eeg = any(channel_type == "eeg" for channel_type in channel_types)
    for name, channel_type in zip(raw.ch_names, channel_types):
        stripped = name.strip()
        if (
            not stripped
            or stripped == "-"
            or PLACEHOLDER_CHANNEL_PATTERN.fullmatch(stripped)
            or AUXILIARY_CHANNEL_PATTERN.search(stripped)
        ):
            continue
        if dataset == "siena" and not stripped.upper().startswith("EEG "):
            continue
        if has_explicit_eeg and channel_type != "eeg":
            continue
        names.append(name)
    if not names:
        raise ValueError("No EEG channels remain after auxiliary-channel exclusion")
    return names


def _ieeg_alias_priority(name: str, raw_index: int) -> tuple[int, int, int, str]:
    value = str(name).strip()
    return (
        int('$' in value),
        int(value.upper().startswith('POL ')),
        int(raw_index),
        value,
    )


def select_bids_ieeg_channel_names(
    raw: Any,
    sidecar: Path,
    return_audit: bool = False,
) -> list[str] | tuple[list[str], dict[str, Any]]:
    table = pd.read_csv(sidecar, sep="\t", encoding="utf-8-sig")
    required = {"name", "type", "status"}
    if not required.issubset(table.columns):
        raise ValueError(f"BIDS channel sidecar lacks required columns: {sidecar}")
    raw_by_canonical: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for raw_index, raw_name in enumerate(raw.ch_names):
        raw_by_canonical[canonical_ieeg_contact_name(raw_name)].append(
            (raw_index, str(raw_name))
        )
    eligible_by_canonical: dict[str, list[dict[str, str]]] = defaultdict(list)
    excluded_sidecar_rows: list[dict[str, str]] = []
    for row in table.itertuples(index=False):
        name = str(row.name)
        channel_type = str(row.type).strip().upper()
        status = str(row.status).strip().lower()
        if channel_type not in {"ECOG", "SEEG"} or status == "bad":
            excluded_sidecar_rows.append({
                'name': name,
                'type': channel_type,
                'status': status,
                'reason': 'non_ieeg_type_or_bad_status',
            })
            continue
        eligible_by_canonical[canonical_ieeg_contact_name(name)].append({
            'name': name,
            'type': channel_type,
            'status': status,
        })

    selected_records: list[tuple[int, str, str]] = []
    excluded_aliases: list[dict[str, Any]] = []
    unmatched_sidecar_channels: list[str] = []
    for canonical_name, sidecar_rows in eligible_by_canonical.items():
        raw_candidates = raw_by_canonical.get(canonical_name, [])
        if not raw_candidates:
            unmatched_sidecar_channels.extend(row['name'] for row in sidecar_rows)
            continue
        eligible_exact_names = {row['name'] for row in sidecar_rows}
        exact_candidates = [
            item for item in raw_candidates if item[1] in eligible_exact_names
        ]
        selection_pool = exact_candidates if exact_candidates else raw_candidates
        selected_index, selected_name = min(
            selection_pool,
            key=lambda item: _ieeg_alias_priority(item[1], item[0]),
        )
        selected_sidecar = min(
            sidecar_rows,
            key=lambda row: (
                int('$' in row['name']),
                int(row['name'] != selected_name),
                row['name'],
            ),
        )
        selected_records.append(
            (selected_index, selected_name, selected_sidecar['type'])
        )
        for raw_index, raw_name in raw_candidates:
            if raw_index == selected_index:
                continue
            excluded_aliases.append({
                'canonical_name': canonical_name,
                'selected_name': selected_name,
                'excluded_name': raw_name,
                'reason': (
                    'sidecar_non_ieeg_alias_excluded_before_canonical_matching'
                    if raw_name not in eligible_exact_names and exact_candidates
                    else 'duplicate_canonical_alias_prefer_non_dollar'
                ),
            })
    names = [item[1] for item in selected_records]
    if not names:
        raise ValueError(f"No good ECoG or SEEG channels match the raw recording: {sidecar}")
    audit = {
        'policy': 'bids_good_ieeg_then_canonical_alias_deduplication',
        'selected_channel_count': len(names),
        'selected_channel_names': names,
        'selected_channel_types': [item[2] for item in selected_records],
        'excluded_aliases': excluded_aliases,
        'excluded_sidecar_rows': excluded_sidecar_rows,
        'unmatched_sidecar_channels': unmatched_sidecar_channels,
    }
    return (names, audit) if return_audit else names


def bids_ieeg_channel_metadata(
    channel_sidecar: Path,
    selected_names: Sequence[str],
    electrode_sidecar: Path | None,
    coordsystem_sidecar: Path | None,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    channel_table = pd.read_csv(channel_sidecar, sep="\t", encoding="utf-8-sig")
    channel_type_candidates: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for row_index, row in enumerate(channel_table.itertuples(index=False)):
        name = str(row.name)
        channel_type = str(row.type).strip().upper()
        status = str(row.status).strip().lower()
        if channel_type not in {'ECOG', 'SEEG'} or status == 'bad':
            continue
        channel_type_candidates[canonical_ieeg_contact_name(name)].append(
            (row_index, name, channel_type)
        )
    channel_types = [
        min(
            channel_type_candidates[canonical_ieeg_contact_name(name)],
            key=lambda item: (
                int('$' in item[1]),
                int(item[1] != name),
                item[0],
            ),
        )[2]
        for name in selected_names
    ]
    positions = np.full((len(selected_names), 3), np.nan, dtype=np.float32)
    coordinate_metadata: dict[str, Any] = {
        "coordinate_system": None,
        "coordinate_units": None,
        "electrode_sidecar": None,
        "coordsystem_sidecar": None,
    }
    if electrode_sidecar is not None and electrode_sidecar.exists():
        electrode_table = pd.read_csv(electrode_sidecar, sep="\t", encoding="utf-8-sig")
        required = {"name", "x", "y", "z"}
        if required.issubset(electrode_table.columns):
            rows = {
                canonical_ieeg_contact_name(str(row.name)): row
                for row in electrode_table.itertuples(index=False)
            }
            for index, name in enumerate(selected_names):
                canonical_name = canonical_ieeg_contact_name(name)
                row = rows.get(canonical_name)
                values = None
                if row is not None:
                    try:
                        values = np.asarray([float(row.x), float(row.y), float(row.z)], dtype=np.float32)
                    except (TypeError, ValueError):
                        values = None
                elif "-" in canonical_name:
                    first_name, second_name = canonical_name.rsplit("-", 1)
                    endpoints = [rows.get(first_name), rows.get(second_name)]
                    if all(endpoint is not None for endpoint in endpoints):
                        try:
                            values = np.asarray(
                                [
                                    [float(endpoint.x), float(endpoint.y), float(endpoint.z)]
                                    for endpoint in endpoints
                                ],
                                dtype=np.float32,
                            ).mean(axis=0)
                        except (TypeError, ValueError):
                            values = None
                if values is None:
                    continue
                if np.isfinite(values).all():
                    positions[index] = values
            coordinate_metadata["electrode_sidecar"] = electrode_sidecar.name
    if coordsystem_sidecar is not None and coordsystem_sidecar.exists():
        coordinate = json.loads(coordsystem_sidecar.read_text(encoding="utf-8-sig"))
        coordinate_metadata.update(
            {
                "coordinate_system": coordinate.get("iEEGCoordinateSystem"),
                "coordinate_units": coordinate.get("iEEGCoordinateUnits"),
                "coordsystem_sidecar": coordsystem_sidecar.name,
            }
        )
    return channel_types, positions, coordinate_metadata


def thalamocortical_bipolar_montage(
    signal: np.ndarray,
    channel_names: Sequence[str],
    channel_types: Sequence[str],
    channel_positions: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, dict[str, Any]]:
    """Create an annotation-independent adjacent bipolar SEEG montage.

    Existing bipolar channels are retained. Monopolar contacts are grouped by
    their alphanumeric shaft prefix and differenced only when their contact
    indices are consecutive. Clinical SOZ, onset, propagation, and anatomical
    labels are intentionally not read by this function.
    """
    values = np.asarray(signal, dtype=np.float32)
    positions = np.asarray(channel_positions, dtype=np.float32)
    normalized = [canonical_ieeg_contact_name(name) for name in channel_names]
    outputs: list[np.ndarray] = []
    names: list[str] = []
    types: list[str] = []
    output_positions: list[np.ndarray] = []
    seen: set[str] = set()
    monopolar: dict[tuple[str, int], int] = {}

    for index, name in enumerate(normalized):
        if IEEG_AUXILIARY_CONTACT.fullmatch(name):
            continue
        parts = name.rsplit("-", 1)
        if len(parts) == 2 and all(IEEG_CONTACT_NUMBER.fullmatch(part) for part in parts):
            if name in seen:
                continue
            seen.add(name)
            outputs.append(values[index])
            names.append(name)
            types.append(str(channel_types[index]).upper())
            first_position = positions[index]
            output_positions.append(first_position.copy())
            continue
        matched = IEEG_CONTACT_NUMBER.fullmatch(name)
        if matched is None:
            continue
        prefix, number_text = matched.groups()
        monopolar.setdefault((prefix, int(number_text)), index)

    shafts: dict[str, list[int]] = {}
    for prefix, number in monopolar:
        shafts.setdefault(prefix, []).append(number)
    for prefix in sorted(shafts):
        numbers = sorted(set(shafts[prefix]))
        for first_number, second_number in zip(numbers, numbers[1:]):
            if second_number != first_number + 1:
                continue
            first_index = monopolar[(prefix, first_number)]
            second_index = monopolar[(prefix, second_number)]
            name = f"{prefix}{first_number}-{prefix}{second_number}"
            if name in seen:
                continue
            seen.add(name)
            outputs.append(values[first_index] - values[second_index])
            names.append(name)
            types.append("SEEG")
            endpoints = positions[[first_index, second_index]]
            if np.isfinite(endpoints).all():
                output_positions.append(endpoints.mean(axis=0).astype(np.float32))
            else:
                output_positions.append(np.full(3, np.nan, dtype=np.float32))

    if not outputs:
        raise ValueError("Thalamocortical recording has no valid adjacent bipolar SEEG channels")
    output = np.stack(outputs).astype(np.float32, copy=False)
    position_array = np.stack(output_positions).astype(np.float32, copy=False)
    return output, names, types, position_array, {
        "policy": "annotation_independent_adjacent_bipolar_seeg",
        "input_channel_count": int(values.shape[0]),
        "output_channel_count": int(output.shape[0]),
        "coordinate_available_count": int(np.isfinite(position_array).all(axis=1).sum()),
    }


def scalp_common_bipolar_montage(
    signal: np.ndarray,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, dict[str, Any]]:
    contract = ChannelUnionContract(
        source_dataset='scalp',
        target_dataset='scalp',
        channel_keys=SCALP_BIPOLAR_CHANNELS,
        source_manifest='',
        target_manifest='',
        policy='standard_1020_bipolar_semantic_bridge',
    )
    output, mask = align_to_union(signal, channel_names, contract)
    available = int(mask.sum())
    if available != MINIMUM_SCALP_BIPOLAR_CHANNELS:
        missing = [
            name for name, present in zip(SCALP_BIPOLAR_CHANNELS, mask)
            if not present
        ]
        raise ValueError(
            f'Scalp recording exposes {available} of '
            f'{MINIMUM_SCALP_BIPOLAR_CHANNELS} required common bipolar channels; '
            f'missing {missing}'
        )
    positions = np.full((available, 3), np.nan, dtype=np.float32)
    return (
        output,
        list(SCALP_BIPOLAR_CHANNELS),
        ['EEG'] * available,
        positions,
        {
            'policy': 'standard_1020_common_16_bipolar_semantic_bridge',
            'input_channel_count': int(signal.shape[0]),
            'output_channel_count': available,
            'missing_channels': [],
        },
    )


def continuous_bad_channel_qc(
    raw: Any,
    minimum_std_uv: float,
    maximum_seconds: float,
) -> tuple[list[str], dict[str, Any]]:
    sfreq = float(raw.info['sfreq'])
    total_points = int(raw.n_times)
    target_points = max(1, min(total_points, int(round(maximum_seconds * sfreq))))
    segment_count = min(12, max(1, int(math.ceil(target_points / max(sfreq * 10.0, 1.0)))))
    segment_points = max(1, target_points // segment_count)
    starts = np.linspace(
        0,
        max(0, total_points - segment_points),
        num=segment_count,
        dtype=np.int64,
    )
    samples = [
        raw.get_data(
            start=int(start),
            stop=min(total_points, int(start) + segment_points),
            units='uV',
        )
        for start in starts
    ]
    sample = np.concatenate(samples, axis=1)
    finite = np.isfinite(sample).all(axis=1)
    standard_deviation = np.nanstd(sample, axis=1)
    bad_mask = (~finite) | (standard_deviation < float(minimum_std_uv))
    bad_names = [
        str(name) for name, is_bad in zip(raw.ch_names, bad_mask) if bool(is_bad)
    ]
    return bad_names, {
        'policy': 'continuous_distributed_flat_and_nonfinite_qc',
        'sampled_seconds': float(sample.shape[1] / sfreq),
        'sampled_segment_count': int(segment_count),
        'minimum_channel_std_uv': float(minimum_std_uv),
        'bad_channel_names': bad_names,
        'bad_channel_count': len(bad_names),
        'amplitude_rejection_enabled': False,
    }


def preprocess_recording(
    recording: Recording,
    signal_config: dict[str, Any],
    qc_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = open_raw_recording(recording.path, preload=True)
    channel_selection_audit: dict[str, Any] = {
        'policy': 'native_eeg_auxiliary_exclusion',
        'excluded_aliases': [],
    }
    if recording.channel_sidecar is None:
        selected_names = select_eeg_channel_names(raw, recording.dataset)
        channel_types = ["EEG"] * len(selected_names)
        channel_positions = np.full((len(selected_names), 3), np.nan, dtype=np.float32)
        coordinate_metadata = {
            "coordinate_system": None,
            "coordinate_units": None,
            "electrode_sidecar": None,
            "coordsystem_sidecar": None,
        }
    else:
        selected_names, channel_selection_audit = select_bids_ieeg_channel_names(
            raw,
            recording.channel_sidecar,
            return_audit=True,
        )
        channel_types, channel_positions, coordinate_metadata = bids_ieeg_channel_metadata(
            recording.channel_sidecar,
            selected_names,
            recording.electrode_sidecar,
            recording.coordsystem_sidecar,
        )
    marked_bad = set(str(value) for value in raw.info.get('bads', []))
    marked_bad_selected = [name for name in selected_names if name in marked_bad]
    if marked_bad_selected and recording.dataset in SCALP_EEG_DATASETS:
        raise ValueError(
            f'Required scalp channels are marked bad: {marked_bad_selected}'
        )
    if marked_bad_selected:
        keep_indices = [
            index for index, name in enumerate(selected_names) if name not in marked_bad
        ]
        selected_names = [selected_names[index] for index in keep_indices]
        channel_types = [channel_types[index] for index in keep_indices]
        channel_positions = channel_positions[keep_indices]
    raw.pick(selected_names)
    detected_bad, continuous_qc = continuous_bad_channel_qc(
        raw,
        minimum_std_uv=float(qc_config.get('min_channel_std_uv', 0.1)),
        maximum_seconds=float(qc_config.get('continuous_bad_channel_max_seconds', 120.0)),
    )
    continuous_qc['marked_bad_channel_names'] = marked_bad_selected
    if detected_bad and recording.dataset in SCALP_EEG_DATASETS:
        raise ValueError(
            f'Required scalp channels failed continuous bad-channel QC: {detected_bad}'
        )
    if detected_bad:
        keep_indices = [
            index for index, name in enumerate(selected_names) if name not in set(detected_bad)
        ]
        selected_names = [selected_names[index] for index in keep_indices]
        channel_types = [channel_types[index] for index in keep_indices]
        channel_positions = channel_positions[keep_indices]
        raw.drop_channels(detected_bad)
    if len(selected_names) < 2:
        raise ValueError('Fewer than two channels remain after continuous bad-channel QC')
    source_sfreq = float(raw.info["sfreq"])
    requested_high = float(signal_config.get("h_freq", 128.0))
    nyquist_margin = float(signal_config.get("nyquist_margin_hz", 0.5))
    effective_high = min(requested_high, source_sfreq / 2.0 - nyquist_margin)
    low = float(signal_config.get("l_freq", 0.1))
    if effective_high <= low:
        raise ValueError(f"Invalid effective bandpass: {low} to {effective_high}")
    requested_notches = [float(value) for value in signal_config.get("notch_freqs", [])]
    notch_frequencies = [
        value for value in requested_notches
        if 0.0 < value < source_sfreq / 2.0 - nyquist_margin
    ]
    if notch_frequencies:
        notch_method = str(signal_config.get('notch_method', 'fir'))
        notch_kwargs: dict[str, Any] = {
            'freqs': notch_frequencies,
            'picks': 'all',
            'method': notch_method,
            'verbose': 'ERROR',
        }
        if notch_method == 'spectrum_fit':
            notch_kwargs['filter_length'] = str(
                signal_config.get('notch_filter_length', '10s')
            )
            if signal_config.get('notch_mt_bandwidth') is not None:
                notch_kwargs['mt_bandwidth'] = float(signal_config['notch_mt_bandwidth'])
        raw.notch_filter(**notch_kwargs)
    raw.filter(
        l_freq=low,
        h_freq=effective_high,
        picks='all',
        method=str(signal_config.get('bandpass_method', 'fir')),
        phase=str(signal_config.get('bandpass_phase', 'zero')),
        fir_window=str(signal_config.get('fir_window', 'hamming')),
        fir_design=str(signal_config.get('fir_design', 'firwin')),
        verbose='ERROR',
    )
    target_sfreq = int(signal_config.get("target_sfreq", 256))
    if not math.isclose(source_sfreq, target_sfreq, rel_tol=0.0, abs_tol=1e-6):
        raw.resample(
            target_sfreq,
            npad='auto',
            method=str(signal_config.get('resampling_method', 'fft')),
            verbose='ERROR',
        )
    signal_uv = raw.get_data(units="uV").astype(np.float32, copy=False)
    raw.close()
    montage_metadata: dict[str, Any] = {"policy": "native_bids_good_ieeg_channels"}
    if recording.dataset in SCALP_EEG_DATASETS:
        signal_uv, selected_names, channel_types, channel_positions, montage_metadata = (
            scalp_common_bipolar_montage(
                signal_uv,
                selected_names,
            )
        )
    elif recording.dataset == "thalamocortical_ieeg":
        signal_uv, selected_names, channel_types, channel_positions, montage_metadata = (
            thalamocortical_bipolar_montage(
                signal_uv,
                selected_names,
                channel_types,
                channel_positions,
            )
        )
    clip_uv = signal_config.get("clip_uv")
    if clip_uv is not None:
        signal_uv = np.clip(signal_uv, -float(clip_uv), float(clip_uv)).astype(np.float32, copy=False)
    if not np.isfinite(signal_uv).all():
        raise ValueError("Non-finite values remain after preprocessing")
    return signal_uv, {
        "source_sfreq": source_sfreq,
        "target_sfreq": target_sfreq,
        "requested_l_freq": low,
        "requested_h_freq": requested_high,
        "effective_h_freq": effective_high,
        "requested_notch_freqs": requested_notches,
        "effective_notch_freqs": notch_frequencies,
        "line_noise_policy": str(signal_config.get('line_noise_policy', 'dataset_native_spectrum_fit')),
        "notch_method": str(signal_config.get('notch_method', 'fir')),
        "line_frequency_hz": float(signal_config['line_frequency_hz']),
        "bandpass_method": str(signal_config.get('bandpass_method', 'fir')),
        "bandpass_phase": str(signal_config.get('bandpass_phase', 'zero')),
        "fir_window": str(signal_config.get('fir_window', 'hamming')),
        "fir_design": str(signal_config.get('fir_design', 'firwin')),
        "resampling_method": str(signal_config.get('resampling_method', 'fft')),
        "channel_names": selected_names,
        "channel_types": channel_types,
        "channel_positions": channel_positions,
        "coordinate_metadata": coordinate_metadata,
        "position_available_count": int(np.isfinite(channel_positions).all(axis=1).sum()),
        "channel_count": len(selected_names),
        "montage_adaptation": montage_metadata,
        "channel_selection_audit": channel_selection_audit,
        "continuous_bad_channel_qc": continuous_qc,
        "normalization": signal_config.get("normalization", "per_clip_per_channel_zscore"),
        "clip_uv": clip_uv,
    }


def extract_clip(
    signal_uv: np.ndarray,
    candidate: ClipCandidate,
    sfreq: int,
    qc_config: dict[str, Any],
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    start = int(round(candidate.local_start_seconds * sfreq))
    clip_seconds = candidate.local_end_seconds - candidate.local_start_seconds
    if not math.isfinite(clip_seconds) or clip_seconds <= 0.0:
        raise ValueError(f"Invalid clip duration: {clip_seconds}")
    stop = start + int(round(clip_seconds * sfreq))
    if start < 0 or stop > signal_uv.shape[1]:
        raise ValueError(f"Clip indices out of bounds: {start}:{stop} for {signal_uv.shape[1]}")
    clip = np.asarray(signal_uv[:, start:stop], dtype=np.float32)
    return finalize_clip(clip, qc_config, normalization)


def finalize_clip(
    raw_clip: np.ndarray,
    qc_config: dict[str, Any],
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    clip = np.asarray(raw_clip, dtype=np.float32)
    if clip.ndim != 2 or clip.shape[1] == 0:
        raise ValueError(f"Invalid clip shape: {clip.shape}")
    means = clip.mean(axis=1, dtype=np.float64).astype(np.float32)
    stds = clip.std(axis=1, dtype=np.float64).astype(np.float32)
    min_std = float(qc_config.get("min_channel_std_uv", 0.1))
    flat_fraction = float(np.mean(stds < min_std))
    if flat_fraction > float(qc_config.get("max_flat_channel_fraction", 0.2)):
        raise ClipQualityExclusion(
            f"Flat-channel fraction {flat_fraction:.6f} exceeds threshold"
        )
    if normalization == "per_clip_per_channel_zscore":
        clip = ((clip - means[:, None]) / np.maximum(stds[:, None], 1e-8)).astype(np.float32)
    elif normalization != "none":
        raise ValueError(f"Unknown normalization: {normalization}")
    return clip, means, stds, {
        "flat_channel_fraction": flat_fraction,
        "peak_uv": float(np.max(np.abs(raw_clip))),
        "clip_rms_uv": float(
            np.sqrt(np.mean(np.square(raw_clip.astype(np.float64, copy=False))))
        ),
        "median_channel_mean_uv": float(np.median(means)),
        "median_channel_std_uv": float(np.median(stds)),
        "p95_channel_std_uv": float(np.quantile(stds, 0.95)),
    }


def extract_stitched_clip(
    preprocessed_by_path: dict[str, tuple[np.ndarray, dict[str, Any]]],
    candidate: ClipCandidate,
    qc_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, Any]]:
    segments = candidate.source_segments or (
        ClipSegment(
            candidate.recording_path,
            candidate.local_start_seconds,
            candidate.local_end_seconds,
        ),
    )
    pieces = []
    reference: dict[str, Any] | None = None
    for segment in segments:
        signal_uv, preprocessing = preprocessed_by_path[segment.recording_path]
        if reference is None:
            reference = preprocessing
        else:
            for key in ("target_sfreq", "channel_names", "channel_types"):
                if preprocessing[key] != reference[key]:
                    raise ValueError(
                        f"Cross-EDF preprocessing mismatch for {key}: "
                        f"{preprocessing[key]} != {reference[key]}"
                    )
        sfreq = int(preprocessing["target_sfreq"])
        start = int(round(segment.local_start_seconds * sfreq))
        stop = int(round(segment.local_end_seconds * sfreq))
        if start < 0 or stop > signal_uv.shape[1] or stop <= start:
            raise ValueError(
                f"Cross-EDF segment indices out of bounds: {start}:{stop} "
                f"for {signal_uv.shape[1]}"
            )
        pieces.append(np.asarray(signal_uv[:, start:stop], dtype=np.float32))
    if reference is None:
        raise ValueError("Clip has no source segments")
    expected_points = int(
        round(
            (candidate.local_end_seconds - candidate.local_start_seconds)
            * int(reference["target_sfreq"])
        )
    )
    raw_clip = np.concatenate(pieces, axis=1)
    if raw_clip.shape[1] != expected_points:
        raise ValueError(
            f"Cross-EDF clip length mismatch: {raw_clip.shape[1]} != {expected_points}"
        )
    clip, means, stds, qc = finalize_clip(
        raw_clip,
        qc_config,
        str(reference["normalization"]),
    )
    return clip, means, stds, qc, reference


def safe_clip_filename(recording: Recording, candidate: ClipCandidate) -> str:
    start_ms = int(round(candidate.local_start_seconds * 1000.0))
    end_ms = int(round(candidate.local_end_seconds * 1000.0))
    return f"{recording.path.stem}__{candidate.task}__{start_ms:012d}-{end_ms:012d}ms__label-{candidate.label}.npz"


def recording_clip_directory(task_root: Path, recording: Recording) -> Path:
    if recording.split not in {"train", "dev", "test"}:
        raise ValueError(f"Unsupported normalized split: {recording.split}")
    if recording.output_subdir.is_absolute() or ".." in recording.output_subdir.parts:
        raise ValueError(f"Unsafe recording output subdirectory: {recording.output_subdir}")
    return task_root / recording.split / recording.patient_id / recording.output_subdir


def save_clip(
    destination: Path,
    clip: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    metadata: dict[str, Any],
    compress: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compress else np.savez
    writer(
        destination,
        eeg=clip,
        label=np.asarray(metadata["label"], dtype=np.int64),
        sfreq=np.asarray(metadata["sfreq"], dtype=np.int64),
        channel_names=np.asarray(metadata["channel_names"], dtype=np.str_),
        channel_types=np.asarray(metadata["channel_types"], dtype=np.str_),
        channel_positions=np.asarray(metadata["channel_positions"], dtype=np.float32),
        channel_mean_uv=means,
        channel_std_uv=stds,
        metadata_json=np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False),
            dtype=np.str_,
        ),
    )


def json_safe_channel_positions(positions: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if math.isfinite(float(value)) else None for value in row]
        for row in np.asarray(positions)
    ]


def setup_logger(dataset: str, task_roots: dict[str, Path]) -> logging.Logger:
    logger = logging.getLogger(f"{LOGGER_NAME}.{dataset}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    for root in task_roots.values():
        root.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(root / "preprocess.log", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def prepare_task_roots(
    output_root: Path,
    dataset: str,
    tasks: Sequence[str],
    fingerprint: str,
    overwrite: bool,
) -> dict[str, Path]:
    roots = {task: output_root / task / DISPLAY_NAMES[dataset] for task in tasks}
    for task, root in roots.items():
        contract_path = root / "dataset_contract.json"
        if overwrite and root.exists():
            shutil.rmtree(root)
        elif contract_path.exists():
            old = json.loads(contract_path.read_text(encoding="utf-8"))
            if old.get("fingerprint") != fingerprint:
                raise RuntimeError(f"Existing {task} output uses a different contract. Use --overwrite explicitly: {root}")
        root.mkdir(parents=True, exist_ok=True)
    return roots


def manifest_row(
    task_root: Path,
    destination: Path,
    recording: Recording,
    candidate: ClipCandidate,
    preprocessing: dict[str, Any],
    qc: dict[str, float],
) -> dict[str, Any]:
    source_segments = candidate.source_segments or (
        ClipSegment(
            candidate.recording_path,
            candidate.local_start_seconds,
            candidate.local_end_seconds,
        ),
    )
    timeline_clip_start = recording.timeline_start + source_segments[0].local_start_seconds
    timeline_clip_end = timeline_clip_start + (
        candidate.local_end_seconds - candidate.local_start_seconds
    )
    return {
        "clip_id": destination.stem,
        "relative_path": destination.relative_to(task_root).as_posix(),
        "dataset": DISPLAY_NAMES[recording.dataset],
        "task": candidate.task,
        "patient_id": recording.patient_id,
        "split": recording.split,
        "session_id": recording.session_id,
        "montage": recording.montage,
        "source_relative_path": recording.relative_path,
        "source_segments_json": json.dumps(
            [
                {
                    "source_relative_path": Path(segment.recording_path)
                    .relative_to(recording.source_root)
                    .as_posix(),
                    "local_start_seconds": segment.local_start_seconds,
                    "local_end_seconds": segment.local_end_seconds,
                }
                for segment in source_segments
            ],
            ensure_ascii=False,
        ),
        "recording_duration_seconds": recording.duration_seconds,
        "timeline_component_id": recording.component_id,
        "timeline_start_seconds": recording.timeline_start,
        "timeline_clip_start_seconds": timeline_clip_start,
        "timeline_clip_end_seconds": timeline_clip_end,
        "seizure_intervals_json": json.dumps(
            [
                {
                    "start_seconds": event.start_seconds,
                    "end_seconds": event.end_seconds,
                    "seizure_id": event.seizure_id,
                }
                for event in recording.seizures
            ],
            ensure_ascii=False,
        ),
        "source_sfreq": preprocessing["source_sfreq"],
        "sfreq": preprocessing["target_sfreq"],
        "channel_count": preprocessing["channel_count"],
        "channel_names": json.dumps(preprocessing["channel_names"], ensure_ascii=False),
        "channel_types": json.dumps(preprocessing["channel_types"], ensure_ascii=False),
        "position_available_count": preprocessing["position_available_count"],
        "coordinate_system": preprocessing["coordinate_metadata"]["coordinate_system"],
        "coordinate_units": preprocessing["coordinate_metadata"]["coordinate_units"],
        "clip_start_seconds": candidate.local_start_seconds,
        "clip_end_seconds": candidate.local_end_seconds,
        "label": candidate.label,
        "class_name": candidate.class_name,
        "event_id": candidate.event_id,
        "overlap_seconds": candidate.overlap_seconds,
        "overlap_ratio_of_clip": candidate.overlap_seconds
        / (candidate.local_end_seconds - candidate.local_start_seconds),
        "requested_l_freq": preprocessing["requested_l_freq"],
        "requested_h_freq": preprocessing["requested_h_freq"],
        "effective_h_freq": preprocessing["effective_h_freq"],
        "normalization": preprocessing["normalization"],
        "flat_channel_fraction": qc["flat_channel_fraction"],
        "peak_uv": qc["peak_uv"],
        "clip_rms_uv": qc["clip_rms_uv"],
        "median_channel_mean_uv": qc["median_channel_mean_uv"],
        "median_channel_std_uv": qc["median_channel_std_uv"],
        "p95_channel_std_uv": qc["p95_channel_std_uv"],
    }


def manifest_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["patient_id", "source_relative_path", "clip_start_seconds", "label"]
        ).reset_index(drop=True)
    return frame


def _json_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def write_preprocessing_statistics(
    task_root: Path,
    frame: pd.DataFrame,
    dataset: str,
    task: str,
    config: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    numeric_columns = (
        "channel_count",
        "position_available_count",
        "flat_channel_fraction",
        "peak_uv",
        "clip_rms_uv",
        "median_channel_mean_uv",
        "median_channel_std_uv",
        "p95_channel_std_uv",
    )
    rows: list[dict[str, Any]] = []
    if not frame.empty:
        enriched = frame.copy()
        enriched["coordinate_available_fraction"] = (
            enriched["position_available_count"].astype(float)
            / enriched["channel_count"].astype(float).clip(lower=1.0)
        )
        for (split, label), group in enriched.groupby(["split", "label"], sort=True):
            row: dict[str, Any] = {
                "dataset": DISPLAY_NAMES[dataset],
                "task": task,
                "split": str(split),
                "label": int(label),
                "clip_count": int(len(group)),
                "patient_count": int(group["patient_id"].astype(str).nunique()),
                "recording_count": int(group["source_relative_path"].astype(str).nunique()),
            }
            for column in (*numeric_columns, "coordinate_available_fraction"):
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                row[f"{column}_median"] = float(values.median()) if not values.empty else None
                row[f"{column}_q25"] = float(values.quantile(0.25)) if not values.empty else None
                row[f"{column}_q75"] = float(values.quantile(0.75)) if not values.empty else None
            rows.append(row)
    grouped = pd.DataFrame(rows)
    grouped.to_csv(task_root / "preprocessing_statistics.csv", index=False)
    channel_type_clip_counts: dict[str, int] = defaultdict(int)
    for value in frame.get("channel_types", pd.Series(dtype=str)).dropna():
        for channel_type in sorted(set(_json_list(value))):
            channel_type_clip_counts[channel_type.upper()] += 1
    signal_config = resolved_signal_config(config, dataset)
    payload = {
        "version": PREPROCESSING_STATISTICS_VERSION,
        "fingerprint": fingerprint,
        "dataset": DISPLAY_NAMES[dataset],
        "task": task,
        "statistical_unit": "patient",
        "clip_count": int(len(frame)),
        "patient_count": int(frame["patient_id"].astype(str).nunique()) if not frame.empty else 0,
        "recording_count": int(frame["source_relative_path"].astype(str).nunique()) if not frame.empty else 0,
        "channel_type_clip_counts": dict(sorted(channel_type_clip_counts.items())),
        "signal_contract": {
            "protocol_track": str(signal_config['protocol_track']),
            "sampling_frequency_hz": int(signal_config["target_sfreq"]),
            "bandpass_hz": [
                float(signal_config["l_freq"]),
                float(signal_config["h_freq"]),
            ],
            "notch_frequencies_hz": [
                float(value) for value in signal_config.get("notch_freqs", [])
            ],
            "line_noise_policy": str(signal_config['line_noise_policy']),
            "notch_method": str(signal_config.get('notch_method', 'fir')),
            "line_frequency_hz": float(signal_config['line_frequency_hz']),
            "notch_filter_length": str(signal_config['notch_filter_length']),
            "notch_mt_bandwidth": float(signal_config['notch_mt_bandwidth']),
            "bandpass_method": str(signal_config['bandpass_method']),
            "bandpass_phase": str(signal_config['bandpass_phase']),
            "fir_window": str(signal_config['fir_window']),
            "fir_design": str(signal_config['fir_design']),
            "resampling_method": str(signal_config['resampling_method']),
            "normalization": str(signal_config["normalization"]),
            "amplitude_statistics_space": "microvolts_before_normalization",
        },
        "grouping": "split_and_label",
        "grouped_statistics_file": "preprocessing_statistics.csv",
    }
    (task_root / "preprocessing_statistics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def process_dataset(
    dataset: str,
    config: dict[str, Any],
    tasks: Sequence[str],
    overwrite: bool = False,
) -> dict[str, Path]:
    dataset_config = config["datasets"][dataset]
    source_root = Path(dataset_config["source_root"]).expanduser().resolve()
    output_root = Path(config["output_root"]).expanduser().resolve()
    window_seconds = configured_window_seconds(config)
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset source root does not exist: {source_root}")
    signal_config = resolved_signal_config(config, dataset)
    fingerprint = stable_fingerprint(config, dataset)
    task_roots = prepare_task_roots(output_root, dataset, tasks, fingerprint, overwrite)
    logger = setup_logger(dataset, task_roots)
    if "prediction" in tasks and dataset in {"epilepsy_ieeg", "hup_ieeg"}:
        logger.info("Prediction is disabled by contract for %s", DISPLAY_NAMES[dataset])
    issues: list[dict[str, Any]] = []
    logger.info("Scanning %s from %s", DISPLAY_NAMES[dataset], source_root)
    recordings = scan_dataset(dataset, source_root, config, issues)
    if not recordings:
        raise RuntimeError(f"No EDF recordings found for {dataset}")
    detection_recordings, detection_excluded_no_seizure_recordings = partition_detection_recordings(
        recordings,
        dataset,
    )
    prediction_recordings, prediction_excluded_no_seizure_recordings = partition_seizure_annotated_recordings(
        recordings,
        dataset,
    )
    if "detection" in tasks and detection_excluded_no_seizure_recordings:
        logger.info(
            "Excluded %d recordings without seizure annotations from detection",
            len(detection_excluded_no_seizure_recordings),
        )
    if "prediction" in tasks and prediction_excluded_no_seizure_recordings:
        logger.info(
            "Excluded %d recordings without seizure annotations from prediction",
            len(prediction_excluded_no_seizure_recordings),
        )
    if "detection" in tasks and not detection_recordings:
        raise RuntimeError(
            f"No detection-eligible EDF recordings remain for {DISPLAY_NAMES[dataset]}"
        )
    if "prediction" in tasks and not prediction_recordings:
        raise RuntimeError(
            f"No prediction-eligible EDF recordings remain for {DISPLAY_NAMES[dataset]}"
        )
    assign_timeline_components(recordings, float(config["timeline"].get("max_contiguous_gap_seconds", 60.0)), issues)
    candidates = []
    if "detection" in tasks:
        for recording in detection_recordings:
            candidates.extend(detection_candidates(recording, window_seconds))
    if "prediction" in tasks:
        candidates.extend(prediction_candidates(recordings, dataset, window_seconds))
    by_path: dict[str, list[ClipCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_path[candidate.recording_path].append(candidate)
    recording_by_path = {str(recording.path): recording for recording in recordings}
    manifests: dict[str, list[dict[str, Any]]] = {task: [] for task in tasks}
    qc_exclusions: dict[str, list[dict[str, Any]]] = {task: [] for task in tasks}
    failures: list[dict[str, Any]] = []
    preprocessed_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    def load_preprocessed(path_key: str) -> tuple[np.ndarray, dict[str, Any]]:
        if path_key not in preprocessed_cache:
            source_recording = recording_by_path[path_key]
            preprocessed_cache[path_key] = preprocess_recording(
                source_recording,
                signal_config,
                config["qc"],
            )
        return preprocessed_cache[path_key]

    progress = tqdm(sorted(by_path), desc=f"Preprocessing {DISPLAY_NAMES[dataset]}", unit="recording", colour="green")
    for path_key in progress:
        recording = recording_by_path[path_key]
        try:
            load_preprocessed(path_key)
        except Exception as exc:
            failure = {
                "stage": "recording_preprocess",
                "path": path_key,
                "tasks": sorted({candidate.task for candidate in by_path[path_key]}),
                "error": str(exc),
            }
            failures.append(failure)
            logger.error("Recording failed: %s | %s", path_key, exc)
            continue
        for candidate in by_path[path_key]:
            task_root = task_roots[candidate.task]
            destination = recording_clip_directory(task_root, recording) / safe_clip_filename(recording, candidate)
            try:
                source_segments = candidate.source_segments or (
                    ClipSegment(
                        candidate.recording_path,
                        candidate.local_start_seconds,
                        candidate.local_end_seconds,
                    ),
                )
                candidate_sources = {
                    segment.recording_path: load_preprocessed(segment.recording_path)
                    for segment in source_segments
                }
                clip, means, stds, qc, preprocessing = extract_stitched_clip(
                    candidate_sources,
                    candidate,
                    config["qc"],
                )
                metadata = {
                    "dataset": DISPLAY_NAMES[dataset],
                    "task": candidate.task,
                    "patient_id": recording.patient_id,
                    "split": recording.split,
                    "session_id": recording.session_id,
                    "montage": recording.montage,
                    "source_relative_path": recording.relative_path,
                    "source_segments": [
                        {
                            "source_relative_path": recording_by_path[segment.recording_path].relative_path,
                            "local_start_seconds": segment.local_start_seconds,
                            "local_end_seconds": segment.local_end_seconds,
                        }
                        for segment in source_segments
                    ],
                    "clip_start_seconds": candidate.local_start_seconds,
                    "clip_end_seconds": candidate.local_end_seconds,
                    "label": candidate.label,
                    "class_name": candidate.class_name,
                    "event_id": candidate.event_id,
                    "overlap_seconds": candidate.overlap_seconds,
                    "overlap_ratio_of_clip": candidate.overlap_seconds
                    / (candidate.local_end_seconds - candidate.local_start_seconds),
                    "sfreq": int(preprocessing["target_sfreq"]),
                    "channel_names": preprocessing["channel_names"],
                    "channel_types": preprocessing["channel_types"],
                    "channel_positions": json_safe_channel_positions(preprocessing["channel_positions"]),
                    "coordinate_metadata": preprocessing["coordinate_metadata"],
                    "requested_l_freq": preprocessing["requested_l_freq"],
                    "requested_h_freq": preprocessing["requested_h_freq"],
                    "effective_h_freq": preprocessing["effective_h_freq"],
                    "normalization": preprocessing["normalization"],
                    "qc": qc,
                    "contract_fingerprint": fingerprint,
                }
                if not destination.exists():
                    save_clip(destination, clip, means, stds, metadata, bool(config["storage"].get("compress_npz", True)))
                manifests[candidate.task].append(manifest_row(task_root, destination, recording, candidate, preprocessing, qc))
            except ClipQualityExclusion as exc:
                exclusion = {
                    "stage": "clip_qc",
                    "path": path_key,
                    "task": candidate.task,
                    "start_seconds": candidate.local_start_seconds,
                    "end_seconds": candidate.local_end_seconds,
                    "label": candidate.label,
                    "reason": str(exc),
                }
                qc_exclusions[candidate.task].append(exclusion)
                logger.warning(
                    "Clip excluded by QC: %s | %.3f | %s",
                    path_key,
                    candidate.local_start_seconds,
                    exc,
                )
            except Exception as exc:
                failure = {
                    "stage": "clip_export",
                    "path": path_key,
                    "task": candidate.task,
                    "start_seconds": candidate.local_start_seconds,
                    "error": str(exc),
                }
                failures.append(failure)
                logger.error("Clip failed: %s | %.3f | %s", path_key, candidate.local_start_seconds, exc)
        preprocessed_cache.pop(path_key, None)
    for task, task_root in task_roots.items():
        frame = manifest_frame(manifests[task])
        frame.to_csv(task_root / "manifest.csv", index=False)
        preprocessing_statistics = write_preprocessing_statistics(
            task_root, frame, dataset, task, config, fingerprint
        )
        task_failures = failures_for_task(failures, task)
        blocking_issue_count = sum(issue.get("level") == "error" for issue in issues)
        ready_for_training = not task_failures and blocking_issue_count == 0 and not frame.empty
        result_status = "complete" if ready_for_training else "complete_with_errors"
        task_excluded_no_seizure_recordings = (
            detection_excluded_no_seizure_recordings
            if task == "detection"
            else prediction_excluded_no_seizure_recordings
        )
        task_seizure_annotated_only = (
            dataset in DETECTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY
            if task == "detection"
            else dataset in PREDICTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY
        )
        contract = {
            "status": result_status,
            "ready_for_training": ready_for_training,
            "preprocessing_protocol": SCALP_PREPROCESSING_PROTOCOL,
            "fingerprint": fingerprint,
            "dataset": DISPLAY_NAMES[dataset],
            "source_root": str(source_root.resolve()),
            "task": task,
            "window_seconds": window_seconds,
            "target_sfreq": int(signal_config["target_sfreq"]),
            "protocol_track": str(signal_config['protocol_track']),
            "requested_bandpass_hz": [float(signal_config["l_freq"]), float(signal_config["h_freq"])],
            "requested_notch_freqs_hz": [float(value) for value in signal_config.get("notch_freqs", [])],
            "line_noise_policy": str(signal_config['line_noise_policy']),
            "notch_method": str(signal_config.get('notch_method', 'fir')),
            "line_frequency_hz": float(signal_config['line_frequency_hz']),
            "notch_filter_length": str(signal_config['notch_filter_length']),
            "notch_mt_bandwidth": float(signal_config['notch_mt_bandwidth']),
            "bandpass_method": str(signal_config['bandpass_method']),
            "bandpass_phase": str(signal_config['bandpass_phase']),
            "fir_window": str(signal_config['fir_window']),
            "fir_design": str(signal_config['fir_design']),
            "resampling_method": str(signal_config['resampling_method']),
            "channel_policy": "standard_1020_common_16_bipolar_semantic_bridge",
            "normalization": signal_config["normalization"],
            "detection_rule": (
                f"a {window_seconds:g}-second clip is positive when it has any "
                "positive-duration overlap with a seizure"
            ),
            "prediction_rule": asdict(PREDICTION_RULES[dataset]),
            "timeline_policy": (
                "patient_file_order_gapless_with_cross_edf_clips"
                if dataset == "chbmit" and task == "prediction"
                else "independent_edf_sessions"
                if dataset == "siena" and task == "prediction"
                else "timestamp_contiguous_components"
            ),
            "patient_directory_required": True,
            "output_layout": "dataset/split/patient_id/recording_subdirectories/clip",
            "sample_count": len(frame),
            "patient_count": int(frame["patient_id"].nunique()) if not frame.empty else 0,
            "seizure_annotated_recordings_only": task_seizure_annotated_only,
            "excluded_no_seizure_recording_count": len(task_excluded_no_seizure_recordings),
            "class_counts": {str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()} if not frame.empty else {},
            "qc_excluded_clip_count": len(qc_exclusions[task]),
            "qc_policy": "continuous flat and nonfinite channel QC before filtering, then clip-level flat-channel exclusion with audit records",
            "preprocessing_statistics": preprocessing_statistics,
            "model_input_specs": {name: asdict(spec) for name, spec in MODEL_INPUT_SPECS.items()},
            "model_channel_policies": MODEL_CHANNEL_POLICIES,
        }
        (task_root / "dataset_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
        (task_root / "annotation_issues.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
        (task_root / "excluded_recordings.json").write_text(
            json.dumps(
                [
                    {
                        "path": recording.relative_path,
                        "patient_id": recording.patient_id,
                        "split": recording.split,
                        "reason": "no_seizure_annotation_in_recording",
                    }
                    for recording in task_excluded_no_seizure_recordings
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (task_root / "qc_exclusions.json").write_text(
            json.dumps(qc_exclusions[task], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (task_root / "failures.json").write_text(json.dumps(task_failures, ensure_ascii=False, indent=2), encoding="utf-8")
        (task_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (task_root / "status.json").write_text(
            json.dumps(
                {
                    "status": result_status,
                    "ready_for_training": ready_for_training,
                    "clips": len(frame),
                    "qc_excluded_clips": len(qc_exclusions[task]),
                    "failures": len(task_failures),
                    "annotation_issues": len(issues),
                    "blocking_annotation_errors": blocking_issue_count,
                    "excluded_no_seizure_recordings": len(task_excluded_no_seizure_recordings),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Completed %s %s with %d clips, %d QC exclusions, and %d failures",
            DISPLAY_NAMES[dataset],
            task,
            len(frame),
            len(qc_exclusions[task]),
            len(task_failures),
        )
    return task_roots


def thalamocortical_prediction_rule() -> PredictionRule:
    return PredictionRule(
        preictal_seconds=1800.0,
        preictal_buffer_seconds=300.0,
        postictal_buffer_seconds=300.0,
        include_leading_interictal=True,
        include_no_seizure_components=False,
    )


def stable_ieeg_fingerprint(
    config: dict[str, Any],
    dataset: str,
    task: str,
    prediction_rule: PredictionRule | None,
) -> str:
    payload = {
        "preprocessing_protocol": (
            IEEG_DETECTION_PROTOCOL if task == "detection" else IEEG_PREDICTION_PROTOCOL
        ),
        "dataset": dataset,
        "task": task,
        "signal": resolved_signal_config(config, dataset),
        "window_seconds": configured_window_seconds(config),
        "detection_positive_operator": ">",
        "detection_min_overlap_seconds": 0.0,
        "prediction_rule": asdict(prediction_rule) if prediction_rule is not None else None,
        "channel_policy": (
            "annotation independent adjacent bipolar SEEG montage"
            if dataset == "thalamocortical_ieeg"
            else "preserve good ECoG or SEEG channels in BIDS sidecar order"
        ),
        "channel_selection_policy": "bids_good_ieeg_then_canonical_alias_deduplication",
        "output_layout": OUTPUT_LAYOUT_POLICY,
        "split": config["split"],
        "qc": config["qc"],
    }
    if task == "detection":
        payload["detection_seizure_annotated_recordings_only"] = (
            dataset in DETECTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def prepare_ieeg_task_roots(
    output_root: Path,
    dataset: str,
    task_rules: dict[str, PredictionRule | None],
    config: dict[str, Any],
    overwrite: bool,
) -> tuple[dict[str, Path], dict[str, str]]:
    roots = {}
    fingerprints = {}
    for task, rule in task_rules.items():
        public_task = "prediction" if task.startswith("prediction") else task
        root = output_root / public_task / DISPLAY_NAMES[dataset]
        fingerprint = stable_ieeg_fingerprint(config, dataset, task, rule)
        contract_path = root / "dataset_contract.json"
        if overwrite and root.exists():
            shutil.rmtree(root)
        elif contract_path.exists():
            old = json.loads(contract_path.read_text(encoding="utf-8"))
            if old.get("fingerprint") != fingerprint:
                raise RuntimeError(f"Existing {task} output uses a different contract. Use --overwrite explicitly: {root}")
        root.mkdir(parents=True, exist_ok=True)
        roots[task] = root
        fingerprints[task] = fingerprint
    return roots, fingerprints


def process_ieeg_dataset(
    dataset: str,
    config: dict[str, Any],
    tasks: Sequence[str],
    overwrite: bool = False,
) -> dict[str, Path]:
    dataset_config = config["datasets"][dataset]
    source_root = Path(dataset_config["source_root"]).expanduser().resolve()
    signal_config = resolved_signal_config(config, dataset)
    output_root = Path(config["output_root"]).expanduser().resolve()
    window_seconds = configured_window_seconds(config)
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset source root does not exist: {source_root}")
    if "prediction" in tasks and dataset in {"epilepsy_ieeg", "hup_ieeg"}:
        disabled_prediction_root = output_root / "prediction" / DISPLAY_NAMES[dataset]
        if disabled_prediction_root.exists():
            if overwrite:
                shutil.rmtree(disabled_prediction_root)
            else:
                raise RuntimeError(
                    f"Prediction is disabled for {DISPLAY_NAMES[dataset]}, but stale output exists. "
                    f"Use --overwrite to remove it: {disabled_prediction_root}"
                )
    task_rules: dict[str, PredictionRule | None] = {}
    if "detection" in tasks:
        task_rules["detection"] = None
    if "prediction" in tasks and dataset == "thalamocortical_ieeg":
        task_rules["prediction_30min"] = thalamocortical_prediction_rule()
    if not task_rules:
        return {}
    task_roots, fingerprints = prepare_ieeg_task_roots(
        output_root,
        dataset,
        task_rules,
        config,
        overwrite,
    )
    logger = setup_logger(dataset, task_roots)
    if "prediction" in tasks and dataset in {"epilepsy_ieeg", "hup_ieeg"}:
        logger.info("Prediction is disabled by contract for %s", DISPLAY_NAMES[dataset])
    issues: list[dict[str, Any]] = []
    logger.info("Scanning %s from %s", DISPLAY_NAMES[dataset], source_root)
    recordings = scan_ieeg_dataset(dataset, source_root, config["split"], issues)
    if not recordings:
        raise RuntimeError(f"No valid iEEG recordings found for {dataset}")
    detection_recordings, detection_excluded_no_seizure_recordings = partition_detection_recordings(
        recordings,
        dataset,
    )
    if "detection" in task_rules and detection_excluded_no_seizure_recordings:
        logger.info(
            "Excluded %d recordings without seizure annotations from detection",
            len(detection_excluded_no_seizure_recordings),
        )
    if "detection" in task_rules and not detection_recordings:
        raise RuntimeError(
            f"No seizure-annotated iEEG recordings remain for {DISPLAY_NAMES[dataset]} detection"
        )
    candidates = []
    if "detection" in task_rules:
        for recording in detection_recordings:
            candidates.extend(detection_candidates(recording, window_seconds))
    for task, rule in task_rules.items():
        if rule is not None:
            candidates.extend(
                prediction_candidates_with_rule(
                    recordings,
                    rule,
                    task,
                    window_seconds,
                )
            )
    by_path: dict[str, list[ClipCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_path[candidate.recording_path].append(candidate)
    recording_by_path = {str(recording.path): recording for recording in recordings}
    manifests: dict[str, list[dict[str, Any]]] = {task: [] for task in task_rules}
    qc_exclusions: dict[str, list[dict[str, Any]]] = {task: [] for task in task_rules}
    failures: list[dict[str, Any]] = []
    channel_selection_audits: list[dict[str, Any]] = []
    progress = tqdm(
        sorted(by_path),
        desc=f"Preprocessing {DISPLAY_NAMES[dataset]}",
        unit="recording",
        colour="green",
    )
    for path_key in progress:
        recording = recording_by_path[path_key]
        try:
            signal_uv, preprocessing = preprocess_recording(
                recording, signal_config, config['qc']
            )
        except Exception as exc:
            failure = {
                "stage": "recording_preprocess",
                "path": path_key,
                "tasks": sorted({candidate.task for candidate in by_path[path_key]}),
                "error": str(exc),
            }
            failures.append(failure)
            logger.error("Recording failed: %s | %s", path_key, exc)
            continue
        selection_audit = dict(preprocessing.get('channel_selection_audit', {}))
        selection_audit.update({
            'path': recording.relative_path,
            'patient_id': recording.patient_id,
            'split': recording.split,
        })
        channel_selection_audits.append(selection_audit)
        for candidate in by_path[path_key]:
            task_root = task_roots[candidate.task]
            destination = recording_clip_directory(task_root, recording) / safe_clip_filename(recording, candidate)
            try:
                clip, means, stds, qc = extract_clip(
                    signal_uv,
                    candidate,
                    int(preprocessing["target_sfreq"]),
                    config["qc"],
                    str(preprocessing["normalization"]),
                )
                metadata = {
                    "dataset": DISPLAY_NAMES[dataset],
                    "task": candidate.task,
                    "patient_id": recording.patient_id,
                    "split": recording.split,
                    "session_id": recording.session_id,
                    "montage": recording.montage,
                    "source_relative_path": recording.relative_path,
                    "clip_start_seconds": candidate.local_start_seconds,
                    "clip_end_seconds": candidate.local_end_seconds,
                    "label": candidate.label,
                    "class_name": candidate.class_name,
                    "event_id": candidate.event_id,
                    "overlap_seconds": candidate.overlap_seconds,
                    "overlap_ratio_of_clip": candidate.overlap_seconds
                    / (candidate.local_end_seconds - candidate.local_start_seconds),
                    "sfreq": int(preprocessing["target_sfreq"]),
                    "channel_names": preprocessing["channel_names"],
                    "channel_types": preprocessing["channel_types"],
                    "channel_positions": json_safe_channel_positions(preprocessing["channel_positions"]),
                    "coordinate_metadata": preprocessing["coordinate_metadata"],
                    "requested_l_freq": preprocessing["requested_l_freq"],
                    "requested_h_freq": preprocessing["requested_h_freq"],
                    "effective_h_freq": preprocessing["effective_h_freq"],
                    "normalization": preprocessing["normalization"],
                    "qc": qc,
                    "contract_fingerprint": fingerprints[candidate.task],
                }
                if not destination.exists():
                    save_clip(
                        destination,
                        clip,
                        means,
                        stds,
                        metadata,
                        bool(config["storage"].get("compress_npz", True)),
                    )
                manifests[candidate.task].append(
                    manifest_row(task_root, destination, recording, candidate, preprocessing, qc)
                )
            except ClipQualityExclusion as exc:
                exclusion = {
                    "stage": "clip_qc",
                    "path": path_key,
                    "task": candidate.task,
                    "start_seconds": candidate.local_start_seconds,
                    "end_seconds": candidate.local_end_seconds,
                    "label": candidate.label,
                    "reason": str(exc),
                }
                qc_exclusions[candidate.task].append(exclusion)
                logger.warning(
                    "Clip excluded by QC: %s | %.3f | %s",
                    path_key,
                    candidate.local_start_seconds,
                    exc,
                )
            except Exception as exc:
                failure = {
                    "stage": "clip_export",
                    "path": path_key,
                    "task": candidate.task,
                    "start_seconds": candidate.local_start_seconds,
                    "error": str(exc),
                }
                failures.append(failure)
                logger.error("Clip failed: %s | %.3f | %s", path_key, candidate.local_start_seconds, exc)
    scanner_excluded_recordings = sum(
        str(issue.get("code", "")).startswith("excluded_") for issue in issues
    )
    blocking_issue_count = sum(issue.get("level") == "error" for issue in issues)
    for task, task_root in task_roots.items():
        frame = manifest_frame(manifests[task])
        frame.to_csv(task_root / "manifest.csv", index=False)
        preprocessing_statistics = write_preprocessing_statistics(
            task_root, frame, dataset, task, config, fingerprints[task]
        )
        task_failures = failures_for_task(failures, task)
        ready_for_training = not task_failures and blocking_issue_count == 0 and not frame.empty
        result_status = "complete" if ready_for_training else "complete_with_errors"
        rule = task_rules[task]
        task_no_seizure_exclusions = (
            detection_excluded_no_seizure_recordings if task == "detection" else []
        )
        task_valid_recording_count = (
            len(detection_recordings) if task == "detection" else len(recordings)
        )
        task_excluded_recording_count = (
            scanner_excluded_recordings + len(task_no_seizure_exclusions)
        )
        contract = {
            "status": result_status,
            "ready_for_training": ready_for_training,
            "preprocessing_protocol": (
                IEEG_DETECTION_PROTOCOL if task == "detection" else IEEG_PREDICTION_PROTOCOL
            ),
            "fingerprint": fingerprints[task],
            "dataset": DISPLAY_NAMES[dataset],
            "source_root": str(source_root.resolve()),
            "task": task,
            "window_seconds": window_seconds,
            "target_sfreq": int(signal_config["target_sfreq"]),
            "protocol_track": str(signal_config['protocol_track']),
            "requested_bandpass_hz": [float(signal_config["l_freq"]), float(signal_config["h_freq"])],
            "requested_notch_freqs_hz": [float(value) for value in signal_config.get("notch_freqs", [])],
            "line_noise_policy": str(signal_config['line_noise_policy']),
            "notch_method": str(signal_config.get('notch_method', 'fir')),
            "line_frequency_hz": float(signal_config['line_frequency_hz']),
            "notch_filter_length": str(signal_config['notch_filter_length']),
            "notch_mt_bandwidth": float(signal_config['notch_mt_bandwidth']),
            "bandpass_method": str(signal_config['bandpass_method']),
            "bandpass_phase": str(signal_config['bandpass_phase']),
            "fir_window": str(signal_config['fir_window']),
            "fir_design": str(signal_config['fir_design']),
            "resampling_method": str(signal_config['resampling_method']),
            "channel_policy": (
                "annotation-independent adjacent bipolar SEEG montage with midpoint coordinates"
                if dataset == "thalamocortical_ieeg"
                else "preserve good ECoG or SEEG channels in BIDS sidecar order"
            ),
            "channel_selection_policy": "bids_good_ieeg_then_canonical_alias_deduplication",
            "normalization": signal_config["normalization"],
            "detection_rule": (
                f"a {window_seconds:g}-second clip is positive when it has any "
                "positive-duration overlap with a seizure"
            ),
            "detection_seizure_annotated_recordings_only": (
                task == "detection"
                and dataset in DETECTION_SEIZURE_ANNOTATED_RECORDINGS_ONLY
            ),
            "prediction_rule": asdict(rule) if rule is not None else None,
            "patient_directory_required": True,
            "output_layout": "dataset/split/patient_id/recording_subdirectories/clip",
            "sample_count": len(frame),
            "patient_count": int(frame["patient_id"].nunique()) if not frame.empty else 0,
            "class_counts": {str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()} if not frame.empty else {},
            "valid_recording_count": task_valid_recording_count,
            "excluded_recording_count": task_excluded_recording_count,
            "excluded_no_seizure_recording_count": len(task_no_seizure_exclusions),
            "qc_excluded_clip_count": len(qc_exclusions[task]),
            "qc_policy": "continuous flat and nonfinite channel QC before filtering, then clip-level flat-channel exclusion with audit records",
            "preprocessing_statistics": preprocessing_statistics,
            "model_input_specs": {name: asdict(spec) for name, spec in MODEL_INPUT_SPECS.items()},
            "model_channel_policies": MODEL_CHANNEL_POLICIES,
        }
        (task_root / "dataset_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
        (task_root / "annotation_issues.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
        (task_root / "excluded_recordings.json").write_text(
            json.dumps(
                [
                    {
                        "path": recording.relative_path,
                        "patient_id": recording.patient_id,
                        "split": recording.split,
                        "reason": "no_seizure_annotation_in_recording",
                    }
                    for recording in task_no_seizure_exclusions
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (task_root / "qc_exclusions.json").write_text(
            json.dumps(qc_exclusions[task], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (task_root / "channel_selection_audit.json").write_text(
            json.dumps(channel_selection_audits, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (task_root / "failures.json").write_text(json.dumps(task_failures, ensure_ascii=False, indent=2), encoding="utf-8")
        (task_root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (task_root / "status.json").write_text(
            json.dumps(
                {
                    "status": result_status,
                    "ready_for_training": ready_for_training,
                    "clips": len(frame),
                    "qc_excluded_clips": len(qc_exclusions[task]),
                    "failures": len(task_failures),
                    "annotation_issues": len(issues),
                    "excluded_recordings": task_excluded_recording_count,
                    "excluded_no_seizure_recordings": len(task_no_seizure_exclusions),
                    "blocking_annotation_errors": blocking_issue_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "Completed %s %s with %d clips, %d QC exclusions, %d excluded recordings, and %d failures",
            DISPLAY_NAMES[dataset],
            task,
            len(frame),
            len(qc_exclusions[task]),
            task_excluded_recording_count,
            len(task_failures),
        )
    return task_roots


def resample_clip(signal: np.ndarray, source_sfreq: int, target_sfreq: int) -> np.ndarray:
    if source_sfreq == target_sfreq:
        return signal.astype(np.float32, copy=False)
    divisor = math.gcd(source_sfreq, target_sfreq)
    return resample_poly(signal, target_sfreq // divisor, source_sfreq // divisor, axis=-1).astype(np.float32, copy=False)


def split_native_views(signal: np.ndarray, sfreq: int, view_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    points = int(round(view_seconds * sfreq))
    total = signal.shape[-1]
    if points <= 0 or points > total:
        raise ValueError(f"Invalid native view length: {points} for {total}")
    starts = list(range(0, total - points + 1, points))
    tail_start = total - points
    if starts[-1] != tail_start:
        starts.append(tail_start)
    views = np.stack([signal[:, start : start + points] for start in starts]).astype(np.float32, copy=False)
    return views, np.asarray(starts, dtype=np.int64)


def format_model_views(views: np.ndarray, spec: ModelInputSpec) -> np.ndarray:
    if spec.layout == "continuous":
        return views
    if spec.layout == "eegnet":
        return views[..., None]
    if spec.patch_points is None:
        raise ValueError(f"Layout {spec.layout} requires patch_points")
    view_count, channel_count, point_count = views.shape
    if point_count % spec.patch_points:
        raise ValueError(f"View length {point_count} is not divisible by patch size {spec.patch_points}")
    patched = views.reshape(view_count, channel_count, point_count // spec.patch_points, spec.patch_points)
    if spec.layout == "patch":
        return patched
    if spec.layout == "time_channel_patch":
        return patched.transpose(0, 2, 1, 3)
    raise ValueError(f"Unknown model layout: {spec.layout}")


def canonical_eeg_channel_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", str(name).strip().upper())
    normalized = normalized.split("-")[0].strip()
    normalized = re.sub(r"^(EEG\s+)?", "EEG ", normalized)
    return normalized


def adapt_model_channels(
    signal: np.ndarray,
    channel_names: Sequence[str],
    model_name: str,
    dataset_name: str,
) -> tuple[np.ndarray, list[str], str]:
    model_name = model_name.lower()
    names = [str(value) for value in channel_names]
    policy = MODEL_CHANNEL_POLICIES[model_name]
    if model_name != "evobrain" or dataset_name.upper() != "TUSZ":
        return signal, names, policy
    available: dict[str, int] = {}
    for index, name in enumerate(names):
        available.setdefault(canonical_eeg_channel_name(name), index)
    missing = [name for name in EVOBRAIN_TUSZ_CHANNELS if name not in available]
    if missing:
        raise ValueError(f"EvoBrain TUSZ input lacks original required channels: {missing}")
    indices = [available[name] for name in EVOBRAIN_TUSZ_CHANNELS]
    return signal[indices], [names[index] for index in indices], policy


class BaselineClipDataset(Dataset):
    def __init__(self, task_root: str | Path, model_name: str, split: str | None = None) -> None:
        self.task_root = Path(task_root)
        self.model_name = model_name.lower()
        if self.model_name not in MODEL_INPUT_SPECS:
            raise KeyError(f"Unknown model input specification: {model_name}")
        self.spec = MODEL_INPUT_SPECS[self.model_name]
        self.manifest = pd.read_csv(self.task_root / "manifest.csv", dtype={"patient_id": str, "clip_id": str})
        if split is not None:
            self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if self.manifest.empty:
            raise ValueError("No clips match the requested dataset selection")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        with np.load(self.task_root / row.relative_path, allow_pickle=False) as archive:
            signal = np.asarray(archive["eeg"], dtype=np.float32)
            source_sfreq = int(archive["sfreq"])
            channel_names = [str(value) for value in archive["channel_names"]]
            channel_types = (
                [str(value) for value in archive["channel_types"]]
                if "channel_types" in archive.files
                else ["UNKNOWN"] * len(channel_names)
            )
            channel_positions = (
                np.asarray(archive["channel_positions"], dtype=np.float32)
                if "channel_positions" in archive.files
                else np.full((len(channel_names), 3), np.nan, dtype=np.float32)
            )
            label = int(archive["label"])
        dataset_name = str(getattr(row, "dataset", self.task_root.name.split("_")[0]))
        original_names = list(channel_names)
        signal, channel_names, channel_policy = adapt_model_channels(
            signal,
            channel_names,
            self.model_name,
            dataset_name,
        )
        selected_indices = [original_names.index(name) for name in channel_names]
        channel_types = [channel_types[index] for index in selected_indices]
        channel_positions = channel_positions[selected_indices]
        resampled = resample_clip(signal, source_sfreq, self.spec.sfreq)
        views, starts = split_native_views(resampled, self.spec.sfreq, self.spec.view_seconds)
        formatted = format_model_views(views, self.spec)
        return {
            "eeg": torch.from_numpy(formatted.copy()),
            "label": torch.tensor(label, dtype=torch.long),
            "clip_id": str(row.clip_id),
            "patient_id": str(row.patient_id),
            "dataset": dataset_name,
            "modality": "ieeg" if "IEEG" in dataset_name.upper() else "eeg",
            "montage": str(getattr(row, "montage", "native_unknown")),
            "model_name": self.model_name,
            "channel_policy": channel_policy,
            "channel_names": channel_names,
            "channel_types": channel_types,
            "channel_positions": torch.from_numpy(channel_positions.copy()),
            "channel_count": len(channel_names),
            "sfreq": self.spec.sfreq,
            "view_start_samples": torch.from_numpy(starts),
            "view_seconds": self.spec.view_seconds,
            "channel_dimension": 2 if self.spec.layout == "time_channel_patch" else 1,
        }


def variable_channel_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_channels = max(int(item["channel_count"]) for item in batch)
    channel_dimension = int(batch[0]["channel_dimension"])
    if any(int(item["channel_dimension"]) != channel_dimension for item in batch):
        raise ValueError("A batch cannot mix model input layouts")
    model_name = str(batch[0]["model_name"])
    if any(str(item["model_name"]) != model_name for item in batch):
        raise ValueError("A batch cannot mix baseline models")
    padded = []
    masks = []
    for item in batch:
        tensor = item["eeg"]
        channel_count = int(item["channel_count"])
        shape = list(tensor.shape)
        shape[channel_dimension] = max_channels
        destination = torch.zeros(shape, dtype=tensor.dtype)
        slices = [slice(None)] * tensor.ndim
        slices[channel_dimension] = slice(0, channel_count)
        destination[tuple(slices)] = tensor
        padded.append(destination)
        mask = torch.zeros(max_channels, dtype=torch.bool)
        mask[:channel_count] = True
        masks.append(mask)
    return {
        "eeg": torch.stack(padded),
        "label": torch.stack([item["label"] for item in batch]),
        "channel_mask": torch.stack(masks),
        "clip_id": [item["clip_id"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "modality": [item["modality"] for item in batch],
        "montage": [item["montage"] for item in batch],
        "model_name": model_name,
        "channel_policy": [item["channel_policy"] for item in batch],
        "channel_names": [item["channel_names"] for item in batch],
        "channel_types": [item["channel_types"] for item in batch],
        "channel_positions": [item["channel_positions"] for item in batch],
        "sfreq": batch[0]["sfreq"],
        "view_seconds": batch[0]["view_seconds"],
    }


def prepare_native_model_batch(batch: dict[str, Any]) -> dict[str, Any]:
    eeg = batch["eeg"]
    labels = batch["label"]
    channel_mask = batch["channel_mask"]
    model_name = str(batch["model_name"])
    clip_count = int(eeg.shape[0])
    if model_name == "evobrain":
        if eeg.ndim != 5 or eeg.shape[1] != 1:
            raise ValueError(f"Unexpected EvoBrain batch shape: {tuple(eeg.shape)}")
        model_input = eeg[:, 0]
        clip_index = torch.arange(clip_count, dtype=torch.long)
        native_labels = labels
        native_channel_mask = channel_mask
    else:
        if eeg.ndim < 4:
            raise ValueError(f"Expected a view dimension for {model_name}: {tuple(eeg.shape)}")
        view_count = int(eeg.shape[1])
        model_input = eeg.reshape(clip_count * view_count, *eeg.shape[2:])
        native_labels = labels.repeat_interleave(view_count)
        native_channel_mask = channel_mask.repeat_interleave(view_count, dim=0)
        clip_index = torch.arange(clip_count, dtype=torch.long).repeat_interleave(view_count)
    return {
        "input": model_input,
        "target": native_labels,
        "clip_index": clip_index,
        "clip_count": clip_count,
        "channel_mask": native_channel_mask,
        "channel_names": batch["channel_names"],
        "channel_types": batch["channel_types"],
        "channel_positions": batch["channel_positions"],
        "dataset": batch["dataset"],
        "modality": batch["modality"],
        "montage": batch["montage"],
        "channel_policy": batch["channel_policy"],
    }


def aggregate_view_logits(logits: torch.Tensor, clip_index: torch.Tensor, clip_count: int) -> torch.Tensor:
    if logits.shape[0] != clip_index.shape[0]:
        raise ValueError("Logits and clip indices must have the same leading dimension")
    flattened = logits.reshape(logits.shape[0], -1)
    output = torch.zeros((clip_count, flattened.shape[1]), dtype=flattened.dtype, device=flattened.device)
    counts = torch.zeros((clip_count, 1), dtype=flattened.dtype, device=flattened.device)
    output.index_add_(0, clip_index.to(flattened.device), flattened)
    counts.index_add_(
        0,
        clip_index.to(flattened.device),
        torch.ones((flattened.shape[0], 1), dtype=flattened.dtype, device=flattened.device),
    )
    averaged = output / counts.clamp_min(1.0)
    return averaged[:, 0] if logits.ndim == 1 else averaged.reshape(clip_count, *logits.shape[1:])


def print_contract_table(config: dict[str, Any], datasets: Sequence[str], tasks: Sequence[str]) -> None:
    print("| Dataset | Tasks | Source | Output | Target rate | Bandpass | Notch | Window |")
    print("|:--|:--|:--|:--|--:|:--|:--|--:|")
    for dataset in datasets:
        signal_config = resolved_signal_config(config, dataset)
        effective_tasks = list(tasks)
        if dataset in {"epilepsy_ieeg", "hup_ieeg"}:
            effective_tasks = [task for task in effective_tasks if task != "prediction"]
        print(
            f"| {DISPLAY_NAMES[dataset]} | {','.join(effective_tasks)} | {config['datasets'][dataset]['source_root']} | "
            f"{config['output_root']} | {signal_config['target_sfreq']}Hz | "
            f"{signal_config['l_freq']}-{signal_config['h_freq']}Hz | "
            f"{signal_config.get('notch_freqs', [])} | "
            f"{configured_window_seconds(config):g}s |"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="data_preprocess for EEG and iEEG detection and prediction")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DISPLAY_NAMES), required=True)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_LABELS), default=["detection", "prediction"])
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.window_seconds is not None:
        config["window_seconds"] = args.window_seconds
    else:
        config.setdefault("window_seconds", WINDOW_SECONDS)
    configured_window_seconds(config)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root.expanduser())
    datasets = list(dict.fromkeys(args.datasets))
    tasks = list(dict.fromkeys(args.tasks))
    print_contract_table(config, datasets, tasks)
    for dataset in datasets:
        if dataset in SCALP_EEG_DATASETS:
            process_dataset(dataset, config, tasks, overwrite=args.overwrite)
        else:
            process_ieeg_dataset(
                dataset,
                config,
                tasks,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
