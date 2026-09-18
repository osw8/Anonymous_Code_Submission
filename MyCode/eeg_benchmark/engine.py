
# Provides shared data, training, evaluation, and artifact utilities.
from __future__ import annotations




import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REFERENCE_SUFFIX = re.compile(
    r"(?:[-_\s](?:REF|LE|AVG|AR|CS2))$",
    flags=re.IGNORECASE,
)
IEEG_DATASETS = frozenset({"epilepsy_ieeg", "hup_ieeg", "thalamocortical_ieeg"})
SCALP_EEG_DATASETS = frozenset({
    "tusz", "siena", "chbmit", "chbmit_historical", "chbmit_future",
})
SCALP_BIPOLAR_DERIVATIONS = (
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
)
SCALP_BIPOLAR_CHANNELS = tuple(
    f"{first}-{second}" for first, second in SCALP_BIPOLAR_DERIVATIONS
)
MINIMUM_SCALP_BIPOLAR_CHANNELS = len(SCALP_BIPOLAR_DERIVATIONS)
LEGACY_ELECTRODE_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "01": "O1",
}
CROSS_MODAL_STATISTIC_KEYS = (
    "NATIVE_MEAN",
    "NATIVE_STD",
    "NATIVE_MEDIAN",
    "NATIVE_Q05",
    "NATIVE_Q10",
    "NATIVE_Q25",
    "NATIVE_Q75",
    "NATIVE_Q90",
    "NATIVE_Q95",
    "NATIVE_RMS",
    "NATIVE_MEAN_ABS",
    "NATIVE_IQR",
)
CROSS_MODAL_LATENT_KEYS = tuple(
    f"LATENT_ELECTRODE_SET_{index:02d}" for index in range(1, 19)
)


def canonical_channel_name(name: str) -> str:
    value = re.sub(r"\s+", " ", str(name).strip().upper())
    value = re.sub(r"^EEG\s+", "", value)
    value = REFERENCE_SUFFIX.sub("", value)
    if value.count("-") >= 2:
        value = re.sub(r"-(?:0|1|2)$", "", value)
    return value.strip()


def canonical_electrode_name(name: str) -> str:
    value = canonical_channel_name(name)
    return LEGACY_ELECTRODE_ALIASES.get(value, value)


def channel_derivation(name: str) -> tuple[str, str] | None:
    value = canonical_channel_name(name)
    parts = value.split("-")
    if len(parts) != 2:
        return None
    first = canonical_electrode_name(parts[0])
    second = canonical_electrode_name(parts[1])
    if not first or not second:
        return None
    return first, second


def available_scalp_derivations(channel_keys: Sequence[str]) -> set[str]:
    unipolar = {
        canonical_electrode_name(name)
        for name in channel_keys
        if channel_derivation(name) is None
    }
    bipolar = {
        channel_derivation(name)
        for name in channel_keys
        if channel_derivation(name) is not None
    }
    available: set[str] = set()
    for first, second in SCALP_BIPOLAR_DERIVATIONS:
        if (first, second) in bipolar or (second, first) in bipolar:
            available.add(f"{first}-{second}")
        elif first in unipolar and second in unipolar:
            available.add(f"{first}-{second}")
    return available


def parse_channel_names(value: object) -> list[str]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return [str(item) for item in parsed]


@dataclass(frozen=True)
class ChannelUnionContract:
    source_dataset: str
    target_dataset: str
    channel_keys: tuple[str, ...]
    source_manifest: str
    target_manifest: str
    source_channel_keys: tuple[str, ...] = ()
    target_channel_keys: tuple[str, ...] = ()
    policy: str = "pairwise_name_union_zero_fill_no_interpolation"
    source_native_channel_keys: tuple[str, ...] = ()
    target_native_channel_keys: tuple[str, ...] = ()
    native_channel_capacity: int = 0

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_channel_keys(manifest_path: str | Path) -> set[str]:
    frame = pd.read_csv(manifest_path, usecols=["channel_names"])
    keys: set[str] = set()
    for value in frame["channel_names"].dropna().unique():
        keys.update(canonical_channel_name(name) for name in parse_channel_names(value))
    keys.discard("")
    return keys


def manifest_max_channel_count(manifest_path: str | Path) -> int:
    frame = pd.read_csv(manifest_path, usecols=["channel_names"])
    values = frame["channel_names"].dropna()
    if values.empty:
        return 0
    return max(len(parse_channel_names(value)) for value in values)


def build_channel_union(
    source_manifest: str | Path,
    target_manifest: str | Path,
    source_dataset: str,
    target_dataset: str,
    cross_modal_policy: str = "native_electrode_set_attention",
) -> ChannelUnionContract:
    source_path = Path(source_manifest).resolve()
    target_path = Path(target_manifest).resolve()
    source_keys = manifest_channel_keys(source_path)
    target_keys = manifest_channel_keys(target_path)
    cross_modal = source_dataset in SCALP_EEG_DATASETS and target_dataset in IEEG_DATASETS
    ieeg_in_domain = (
        source_dataset in IEEG_DATASETS
        and target_dataset in IEEG_DATASETS
        and source_dataset == target_dataset
    )
    scalp_to_scalp = source_dataset in SCALP_EEG_DATASETS and target_dataset in SCALP_EEG_DATASETS
    if cross_modal or ieeg_in_domain:
        if cross_modal_policy == "native_electrode_set_attention":
            keys = list(CROSS_MODAL_LATENT_KEYS)
        elif cross_modal_policy == "inductive_permutation_invariant_native_channel_statistics":
            keys = list(CROSS_MODAL_STATISTIC_KEYS)
        else:
            raise ValueError(f"Unsupported cross-modal policy: {cross_modal_policy}")
        source_contract_keys = tuple(keys)
        target_contract_keys = tuple(keys)
        policy = cross_modal_policy
    elif scalp_to_scalp:
        keys = list(SCALP_BIPOLAR_CHANNELS)
        source_contract_keys = tuple(sorted(available_scalp_derivations(source_keys)))
        target_contract_keys = tuple(sorted(available_scalp_derivations(target_keys)))
        if len(source_contract_keys) < MINIMUM_SCALP_BIPOLAR_CHANNELS:
            raise ValueError(
                f"Source dataset exposes only {len(source_contract_keys)} standard bipolar derivations"
            )
        if len(target_contract_keys) < MINIMUM_SCALP_BIPOLAR_CHANNELS:
            raise ValueError(
                f"Target dataset exposes only {len(target_contract_keys)} standard bipolar derivations"
            )
        policy = "standard_1020_bipolar_semantic_bridge"
    else:
        keys = sorted(source_keys | target_keys)
        source_contract_keys = tuple(sorted(source_keys))
        target_contract_keys = tuple(sorted(target_keys))
        policy = "pairwise_name_union_zero_fill_no_interpolation"
    if not keys:
        raise ValueError("No channels were discovered in the source and target manifests")
    return ChannelUnionContract(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        channel_keys=tuple(keys),
        source_manifest=str(source_path),
        target_manifest=str(target_path),
        source_channel_keys=source_contract_keys,
        target_channel_keys=target_contract_keys,
        policy=policy,
        source_native_channel_keys=tuple(sorted(source_keys)),
        target_native_channel_keys=tuple(sorted(target_keys)),
        native_channel_capacity=(
            max(
                manifest_max_channel_count(source_path),
                manifest_max_channel_count(target_path),
                len(SCALP_BIPOLAR_CHANNELS),
            )
            if cross_modal or ieeg_in_domain else len(keys)
        ),
    )


def native_channel_statistics(signal: np.ndarray) -> np.ndarray:
    if signal.ndim != 2 or signal.shape[0] == 0:
        raise ValueError(f"Expected non-empty channel by time signal, got {signal.shape}")
    values = np.asarray(signal, dtype=np.float32)
    quantiles = np.quantile(
        values,
        [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        axis=0,
    )
    projected = np.stack(
        [
            values.mean(axis=0),
            values.std(axis=0),
            quantiles[3],
            quantiles[0],
            quantiles[1],
            quantiles[2],
            quantiles[4],
            quantiles[5],
            quantiles[6],
            np.sqrt(np.mean(np.square(values), axis=0)),
            np.mean(np.abs(values), axis=0),
            quantiles[4] - quantiles[2],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    if not np.isfinite(projected).all():
        raise ValueError("Cross-modal channel statistics contain non-finite values")
    return projected


def align_to_union(
    signal: np.ndarray,
    channel_names: Sequence[str],
    contract: ChannelUnionContract,
) -> tuple[np.ndarray, np.ndarray]:
    if signal.ndim != 2:
        raise ValueError(f"Expected channel by time signal, got {signal.shape}")
    if signal.shape[0] != len(channel_names):
        raise ValueError("Signal channel count and channel name count differ")
    if contract.policy == "native_electrode_set_attention":
        return np.asarray(signal, dtype=np.float32), np.ones(signal.shape[0], dtype=np.bool_)
    if contract.policy == "inductive_permutation_invariant_native_channel_statistics":
        projected = native_channel_statistics(signal)
        return projected, np.ones(projected.shape[0], dtype=np.bool_)
    if contract.policy == "standard_1020_bipolar_semantic_bridge":
        destination = np.zeros((len(contract.channel_keys), signal.shape[1]), dtype=np.float32)
        mask = np.zeros(len(contract.channel_keys), dtype=np.bool_)
        unipolar: dict[str, np.ndarray] = {}
        bipolar: dict[tuple[str, str], np.ndarray] = {}
        for source_index, raw_name in enumerate(channel_names):
            derivation = channel_derivation(raw_name)
            if derivation is None:
                key = canonical_electrode_name(raw_name)
                if key and key not in unipolar:
                    unipolar[key] = signal[source_index]
            elif derivation not in bipolar:
                bipolar[derivation] = signal[source_index]
        for index, (first, second) in enumerate(SCALP_BIPOLAR_DERIVATIONS):
            if (first, second) in bipolar:
                destination[index] = bipolar[(first, second)]
                mask[index] = True
            elif (second, first) in bipolar:
                destination[index] = -bipolar[(second, first)]
                mask[index] = True
            elif first in unipolar and second in unipolar:
                destination[index] = unipolar[first] - unipolar[second]
                mask[index] = True
        return destination, mask
    if contract.policy != "pairwise_name_union_zero_fill_no_interpolation":
        raise ValueError(f"Unknown channel contract policy: {contract.policy}")
    destination = np.zeros((len(contract.channel_keys), signal.shape[1]), dtype=np.float32)
    mask = np.zeros(len(contract.channel_keys), dtype=np.bool_)
    union_index = {name: index for index, name in enumerate(contract.channel_keys)}
    seen: set[str] = set()
    for source_index, raw_name in enumerate(channel_names):
        key = canonical_channel_name(raw_name)
        if key in seen:
            raise ValueError(f"Duplicate canonical channel in one recording: {key}")
        seen.add(key)
        target_index = union_index.get(key)
        if target_index is None:
            continue
        destination[target_index] = signal[source_index]
        mask[target_index] = True
    return destination, mask


def align_preprocessed_clip(
    signal: np.ndarray,
    channel_names: Sequence[str],
    contract: ChannelUnionContract,
    channel_mean_uv: np.ndarray | None = None,
    channel_std_uv: np.ndarray | None = None,
    normalization: str | None = None,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float32)
    if (
        contract.policy == "native_electrode_set_attention"
        and contract.source_dataset in SCALP_EEG_DATASETS
        and dataset_name is not None
        and str(dataset_name).strip().lower() in {
            contract.source_dataset.lower(),
            contract.source_dataset.replace('_', '-').lower(),
            {'tusz': 'tusz', 'siena': 'siena', 'chbmit': 'chb-mit'}[contract.source_dataset],
        }
        and len(available_scalp_derivations(channel_names)) >= MINIMUM_SCALP_BIPOLAR_CHANNELS
    ):
        if normalization == "per_clip_per_channel_zscore":
            if channel_mean_uv is None or channel_std_uv is None:
                raise ValueError("Scalp source bridge requires saved channel scale metadata")
            means = np.asarray(channel_mean_uv, dtype=np.float32).reshape(-1)
            stds = np.asarray(channel_std_uv, dtype=np.float32).reshape(-1)
            values = values * stds[:, None] + means[:, None]
        elif normalization not in {None, "none"}:
            raise ValueError(f"Unknown cached normalization: {normalization}")
        scalp_contract = ChannelUnionContract(
            source_dataset=contract.source_dataset,
            target_dataset=contract.target_dataset,
            channel_keys=SCALP_BIPOLAR_CHANNELS,
            source_manifest=contract.source_manifest,
            target_manifest=contract.target_manifest,
            policy="standard_1020_bipolar_semantic_bridge",
        )
        aligned, mask = align_to_union(values, channel_names, scalp_contract)
        selected = aligned[mask]
        means = selected.mean(axis=1, dtype=np.float64).astype(np.float32)
        stds = selected.std(axis=1, dtype=np.float64).astype(np.float32)
        aligned[mask] = (
            (selected - means[:, None]) / np.maximum(stds[:, None], 1e-8)
        ).astype(np.float32)
        return aligned, mask
    if contract.policy != "standard_1020_bipolar_semantic_bridge":
        return align_to_union(values, channel_names, contract)
    if normalization == "per_clip_per_channel_zscore":
        if channel_mean_uv is None or channel_std_uv is None:
            raise ValueError("Scalp semantic bridge requires saved channel scale metadata")
        means = np.asarray(channel_mean_uv, dtype=np.float32).reshape(-1)
        stds = np.asarray(channel_std_uv, dtype=np.float32).reshape(-1)
        if means.size != values.shape[0] or stds.size != values.shape[0]:
            raise ValueError("Saved channel scale metadata does not match signal channels")
        values = values * stds[:, None] + means[:, None]
    elif normalization not in {None, "none"}:
        raise ValueError(f"Unknown cached normalization: {normalization}")
    aligned, mask = align_to_union(values, channel_names, contract)
    if int(mask.sum()) < MINIMUM_SCALP_BIPOLAR_CHANNELS:
        raise ValueError(
            f"Recording exposes only {int(mask.sum())} standard bipolar derivations; "
            f"at least {MINIMUM_SCALP_BIPOLAR_CHANNELS} are required"
        )
    if mask.any():
        selected = aligned[mask]
        means = selected.mean(axis=1, dtype=np.float64).astype(np.float32)
        stds = selected.std(axis=1, dtype=np.float64).astype(np.float32)
        aligned[mask] = (
            (selected - means[:, None]) / np.maximum(stds[:, None], 1e-8)
        ).astype(np.float32)
    return aligned, mask




import os
import random
from typing import Any

import numpy as np


def configure_reproducibility(seed: int, deterministic: bool = True) -> dict[str, Any]:
    numeric_mode = os.environ.get("BENCHMARK_NUMERIC_MODE", "fp32").strip().lower()
    if numeric_mode not in {"fp32", "tf32"}:
        raise ValueError("BENCHMARK_NUMERIC_MODE must be fp32 or tf32")
    fast_algorithms = bool(int(os.environ.get("BENCHMARK_FAST_ALGORITHMS", "0")))
    allow_tf32 = numeric_mode == "tf32"
    strict_deterministic_algorithms = bool(deterministic and not fast_algorithms and not allow_tf32)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if strict_deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    result: dict[str, Any] = {
        "seed": seed,
        "deterministic": deterministic,
        "strict_deterministic_algorithms": strict_deterministic_algorithms,
        "numeric_mode": numeric_mode,
        "fast_algorithms": fast_algorithms,
        "cudnn_benchmark": bool(fast_algorithms),
        "tf32": allow_tf32,
    }
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = bool(fast_algorithms)
        torch.backends.cudnn.deterministic = bool(strict_deterministic_algorithms)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.use_deterministic_algorithms(strict_deterministic_algorithms, warn_only=False)
    except ImportError:
        result["torch"] = "not_available"
    return result




import json
import logging
import sys
from pathlib import Path
from typing import Any

def setup_run_logger(output_dir: str | Path, name: str) -> logging.Logger:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(root / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def count_parameters(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def enable_full_finetuning(module: Any) -> list[str]:
    non_differentiable_parameters = []
    for name, parameter in module.named_parameters():
        differentiable = parameter.is_floating_point() or parameter.is_complex()
        parameter.requires_grad_(differentiable)
        if not differentiable:
            non_differentiable_parameters.append(name)
    return non_differentiable_parameters


def print_model_information(rows: dict[str, Any]) -> None:
    print("| Field | Value |")
    print("|:--|:--|")
    for key, value in rows.items():
        print(f"| {key} | {value} |")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")




import math

import numpy as np
from scipy.signal import resample_poly


def resample_clip(
    signal: np.ndarray,
    source_sfreq: int,
    target_sfreq: int,
) -> np.ndarray:
    if source_sfreq == target_sfreq:
        return signal.astype(np.float32, copy=False)
    divisor = math.gcd(source_sfreq, target_sfreq)
    return resample_poly(
        signal,
        target_sfreq // divisor,
        source_sfreq // divisor,
        axis=-1,
    ).astype(np.float32, copy=False)


def split_native_views(
    signal: np.ndarray,
    sfreq: int,
    view_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = int(round(view_seconds * sfreq))
    total = signal.shape[-1]
    if points <= 0 or points > total:
        raise ValueError(f'Invalid native view length: {points} for {total}')
    starts = list(range(0, total - points + 1, points))
    tail_start = total - points
    if starts[-1] != tail_start:
        starts.append(tail_start)
    views = np.stack([
        signal[:, start:start + points] for start in starts
    ]).astype(np.float32, copy=False)
    return views, np.asarray(starts, dtype=np.int64)




import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from tqdm.auto import tqdm


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.end <= self.start:
            raise ValueError(f"Invalid interval: {self.start}, {self.end}")


def select_f1_threshold(labels: Iterable[int], scores: Iterable[float]) -> float:
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_score = np.asarray(list(scores), dtype=np.float64)
    if y_true.size == 0 or y_true.size != y_score.size:
        raise ValueError("Labels and scores must be non-empty and have equal length")
    if np.unique(y_true).size < 2:
        raise ValueError("F1 threshold selection requires both classes in validation data")
    candidates = np.unique(np.concatenate(([0.0], y_score, [1.0])))
    order = np.argsort(y_score, kind='mergesort')
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    positive_prefix = np.concatenate(([0], np.cumsum(sorted_labels, dtype=np.int64)))
    first_predicted = np.searchsorted(sorted_scores, candidates, side='left')
    predicted_count = y_true.size - first_predicted
    true_positive = positive_prefix[-1] - positive_prefix[first_predicted]
    false_positive = predicted_count - true_positive
    false_negative = positive_prefix[-1] - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1_values = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )
    best = np.flatnonzero(np.isclose(f1_values, f1_values.max(), rtol=0.0, atol=1e-12))
    return float(candidates[best[-1]])


def clip_level_metrics(labels: Iterable[int], scores: Iterable[float], threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_score = np.asarray(list(scores), dtype=np.float64)
    if y_true.size == 0 or y_true.size != y_score.size:
        raise ValueError("Labels and scores must be non-empty and have equal length")
    if np.unique(y_true).size < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
    y_pred = (y_score >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
        if math.isfinite(sensitivity) and math.isfinite(specificity)
        else float("nan")
    )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "clip_count": int(y_true.size),
    }


def _percentile_interval(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {'lower': float('nan'), 'upper': float('nan'), 'valid_resamples': 0}
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {
        'lower': float(lower),
        'upper': float(upper),
        'valid_resamples': int(finite.size),
    }


def patient_cluster_bootstrap(
    predictions: pd.DataFrame,
    threshold: float,
    seed: int,
    resamples: int = 2000,
    event_by_patient: pd.DataFrame | None = None,
    workers: int = 8,
) -> dict[str, object]:
    if 'patient_id' not in predictions.columns:
        raise ValueError('Patient-cluster bootstrap requires patient_id')
    if resamples <= 0:
        raise ValueError('Bootstrap resamples must be positive')
    if workers <= 0:
        raise ValueError('Bootstrap workers must be positive')
    patient_ids = sorted(predictions['patient_id'].astype(str).unique())
    if not patient_ids:
        raise ValueError('Patient-cluster bootstrap requires at least one patient')
    groups = {
        patient_id: np.flatnonzero(
            predictions['patient_id'].astype(str).to_numpy() == patient_id
        )
        for patient_id in patient_ids
    }
    labels = predictions['label'].astype(int).to_numpy()
    scores = predictions['score'].astype(float).to_numpy()
    generator = np.random.default_rng(int(seed))
    patient_count = len(patient_ids)
    sampled_patient_indices = generator.integers(
        0,
        patient_count,
        size=(int(resamples), patient_count),
        dtype=np.int64,
    )
    aurocs = np.full(int(resamples), np.nan, dtype=np.float64)
    auprcs = np.full(int(resamples), np.nan, dtype=np.float64)
    f1_values = np.full(int(resamples), np.nan, dtype=np.float64)
    event_sensitivities = np.full(int(resamples), np.nan, dtype=np.float64)
    event_precisions = np.full(int(resamples), np.nan, dtype=np.float64)
    event_f1_values = np.full(int(resamples), np.nan, dtype=np.float64)
    false_alarms_per_hour = np.full(int(resamples), np.nan, dtype=np.float64)
    event_lookup = None
    event_sensitivity_by_patient = None
    event_precision_by_patient = None
    event_f1_by_patient = None
    fa_per_hour_by_patient = None
    if event_by_patient is not None and not event_by_patient.empty:
        event_lookup = event_by_patient.set_index(
            event_by_patient['patient_id'].astype(str), drop=False
        )
        event_sensitivity_by_patient = event_lookup.loc[patient_ids, 'event_sensitivity'].to_numpy(
            dtype=np.float64
        )
        event_precision_by_patient = event_lookup.loc[patient_ids, 'event_precision'].to_numpy(
            dtype=np.float64
        )
        event_f1_by_patient = event_lookup.loc[patient_ids, 'event_f1'].to_numpy(
            dtype=np.float64
        )
        fa_per_hour_by_patient = event_lookup.loc[patient_ids, 'fa_per_hour'].to_numpy(
            dtype=np.float64
        )

    group_indices = [groups[patient_id] for patient_id in patient_ids]

    def calculate_resample(resample_index: int) -> tuple[int, float, float, float, float, float, float, float]:
        sampled_group_indices = sampled_patient_indices[resample_index]
        indices = np.concatenate([group_indices[index] for index in sampled_group_indices])
        sampled_labels = labels[indices]
        sampled_scores = scores[indices]
        auroc = (
            float(roc_auc_score(sampled_labels, sampled_scores))
            if np.unique(sampled_labels).size == 2
            else float('nan')
        )
        auprc = (
            float(average_precision_score(sampled_labels, sampled_scores))
            if np.unique(sampled_labels).size == 2
            else float('nan')
        )
        sampled_predictions = sampled_scores >= threshold
        true_positive = np.count_nonzero(sampled_predictions & (sampled_labels == 1))
        false_positive = np.count_nonzero(sampled_predictions & (sampled_labels == 0))
        false_negative = np.count_nonzero(~sampled_predictions & (sampled_labels == 1))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_value = float(2 * true_positive / denominator) if denominator else 0.0
        event_sensitivity = float('nan')
        event_precision = float('nan')
        event_f1 = float('nan')
        fa_per_hour = float('nan')
        if (
            event_sensitivity_by_patient is not None
            and event_precision_by_patient is not None
            and event_f1_by_patient is not None
            and fa_per_hour_by_patient is not None
        ):
            with np.errstate(invalid='ignore'):
                event_sensitivity = float(np.nanmean(event_sensitivity_by_patient[sampled_group_indices]))
                event_precision = float(np.nanmean(event_precision_by_patient[sampled_group_indices]))
                event_f1 = float(np.nanmean(event_f1_by_patient[sampled_group_indices]))
            fa_per_hour = float(fa_per_hour_by_patient[sampled_group_indices].mean())
        return resample_index, auroc, auprc, f1_value, event_sensitivity, event_precision, event_f1, fa_per_hour

    maximum_workers = min(int(workers), int(resamples))
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = [executor.submit(calculate_resample, index) for index in range(int(resamples))]
        completed = tqdm(
            as_completed(futures),
            total=int(resamples),
            desc='Patient bootstrap',
            unit='resample',
            colour='magenta',
            dynamic_ncols=True,
        )
        for future in completed:
            index, auroc, auprc, f1_value, event_sensitivity, event_precision, event_f1, fa_per_hour = future.result()
            aurocs[index] = auroc
            auprcs[index] = auprc
            f1_values[index] = f1_value
            event_sensitivities[index] = event_sensitivity
            event_precisions[index] = event_precision
            event_f1_values[index] = event_f1
            false_alarms_per_hour[index] = fa_per_hour
    result: dict[str, object] = {
        'method': 'patient_cluster_percentile_bootstrap',
        'confidence_level': 0.95,
        'seed': int(seed),
        'requested_resamples': int(resamples),
        'patient_count': patient_count,
        'cpu_workers': maximum_workers,
        'auroc': _percentile_interval(aurocs.tolist()),
        'auprc': _percentile_interval(auprcs.tolist()),
        'f1': _percentile_interval(f1_values.tolist()),
    }
    if event_lookup is not None:
        result['event_sensitivity'] = _percentile_interval(event_sensitivities.tolist())
        result['event_precision'] = _percentile_interval(event_precisions.tolist())
        result['event_f1'] = _percentile_interval(event_f1_values.tolist())
        result['fa_per_hour'] = _percentile_interval(false_alarms_per_hour.tolist())
        result['fa_per_24h'] = {
            **_percentile_interval((false_alarms_per_hour * 24.0).tolist()),
        }
    return result


def merge_intervals(intervals: Iterable[Interval], maximum_gap_seconds: float) -> list[Interval]:
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    merged: list[Interval] = []
    for interval in ordered:
        if merged and interval.start - merged[-1].end < maximum_gap_seconds:
            merged[-1] = Interval(merged[-1].start, max(merged[-1].end, interval.end))
        else:
            merged.append(interval)
    return merged


def split_long_intervals(intervals: Iterable[Interval], maximum_duration_seconds: float) -> list[Interval]:
    output: list[Interval] = []
    for interval in intervals:
        start = interval.start
        while interval.end - start > maximum_duration_seconds:
            output.append(Interval(start, start + maximum_duration_seconds))
            start += maximum_duration_seconds
        output.append(Interval(start, interval.end))
    return output


def interval_union_seconds(intervals: Iterable[Interval]) -> float:
    return float(sum(item.end - item.start for item in merge_intervals(intervals, 0.0)))


def overlaps(first: Interval, second: Interval) -> bool:
    return min(first.end, second.end) > max(first.start, second.start)


def parse_reference_intervals(value: object) -> list[Interval]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    records = json.loads(str(value))
    return [Interval(float(item["start_seconds"]), float(item["end_seconds"])) for item in records]


def match_events(predicted: list[Interval], reference: list[Interval]) -> tuple[int, int, int]:
    true_positive = sum(any(overlaps(prediction, target) for prediction in predicted) for target in reference)
    false_positive = sum(not any(overlaps(prediction, target) for target in reference) for prediction in predicted)
    return true_positive, false_positive, len(reference) - true_positive


def detection_event_metrics(
    predictions: pd.DataFrame,
    threshold: float,
    merge_gap_seconds: float = 90.0,
    maximum_event_seconds: float = 300.0,
    preictal_tolerance_seconds: float = 30.0,
    postictal_tolerance_seconds: float = 60.0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    required = {
        "patient_id",
        "source_relative_path",
        "clip_start_seconds",
        "clip_end_seconds",
        "score",
        "seizure_intervals_json",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Detection event scoring is missing columns: {sorted(missing)}")
    patient_rows = []
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_hours = 0.0
    for patient_id, patient_frame in predictions.groupby("patient_id", sort=True):
        patient_tp = 0
        patient_fp = 0
        patient_fn = 0
        patient_seconds = 0.0
        for _, record_frame in patient_frame.groupby("source_relative_path", sort=True):
            record_frame = record_frame.sort_values("clip_start_seconds")
            positive = record_frame[record_frame["score"] >= threshold]
            predicted = [
                Interval(float(row.clip_start_seconds), float(row.clip_end_seconds))
                for row in positive.itertuples(index=False)
            ]
            predicted = split_long_intervals(
                merge_intervals(predicted, merge_gap_seconds),
                maximum_event_seconds,
            )
            reference_raw = parse_reference_intervals(record_frame.iloc[0]["seizure_intervals_json"])
            reference_raw = split_long_intervals(
                merge_intervals(reference_raw, merge_gap_seconds),
                maximum_event_seconds,
            )
            reference = [
                Interval(
                    max(0.0, item.start - preictal_tolerance_seconds),
                    item.end + postictal_tolerance_seconds,
                )
                for item in reference_raw
            ]
            tp, fp, fn = match_events(predicted, reference)
            patient_tp += tp
            patient_fp += fp
            patient_fn += fn
            evaluated = [
                Interval(float(row.clip_start_seconds), float(row.clip_end_seconds))
                for row in record_frame.itertuples(index=False)
            ]
            patient_seconds += interval_union_seconds(evaluated)
        hours = patient_seconds / 3600.0
        sensitivity = patient_tp / (patient_tp + patient_fn) if patient_tp + patient_fn else float("nan")
        precision = patient_tp / (patient_tp + patient_fp) if patient_tp + patient_fp else float("nan")
        f1_denominator = 2 * patient_tp + patient_fp + patient_fn
        event_f1 = 2 * patient_tp / f1_denominator if f1_denominator else float("nan")
        fa_per_hour = patient_fp / hours if hours > 0.0 else float("nan")
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "event_tp": patient_tp,
                "event_fp": patient_fp,
                "event_fn": patient_fn,
                "event_sensitivity": sensitivity,
                "event_precision": precision,
                "event_f1": event_f1,
                "fa_per_hour": fa_per_hour,
                "fa_per_24h": fa_per_hour * 24.0,
                "evaluated_hours": hours,
            }
        )
        global_tp += patient_tp
        global_fp += patient_fp
        global_fn += patient_fn
        global_hours += hours
    per_patient = pd.DataFrame(patient_rows)
    patient_macro_sensitivity = float(per_patient["event_sensitivity"].mean()) if not per_patient.empty else float("nan")
    patient_macro_precision = float(per_patient["event_precision"].mean()) if not per_patient.empty else float("nan")
    patient_macro_f1 = float(per_patient["event_f1"].mean()) if not per_patient.empty else float("nan")
    patient_macro_fa_per_hour = float(per_patient["fa_per_hour"].mean()) if not per_patient.empty else float("nan")
    global_metrics = {
        "event_sensitivity": patient_macro_sensitivity,
        "event_precision": patient_macro_precision,
        "event_f1": patient_macro_f1,
        "fa_per_hour": patient_macro_fa_per_hour,
        "fa_per_24h": patient_macro_fa_per_hour * 24.0,
        "event_tp": global_tp,
        "event_fp": global_fp,
        "event_fn": global_fn,
        "evaluated_hours": global_hours,
        "pooled_event_sensitivity": global_tp / (global_tp + global_fn) if global_tp + global_fn else float("nan"),
        "pooled_event_precision": global_tp / (global_tp + global_fp) if global_tp + global_fp else float("nan"),
        "pooled_event_f1": (
            2 * global_tp / (2 * global_tp + global_fp + global_fn)
            if 2 * global_tp + global_fp + global_fn
            else float("nan")
        ),
        "pooled_fa_per_hour": global_fp / global_hours if global_hours > 0.0 else float("nan"),
        "pooled_fa_per_24h": global_fp / global_hours * 24.0 if global_hours > 0.0 else float("nan"),
        "aggregation": "patient_macro_primary_with_pooled_audit_values",
        "evaluation_scope": "preprocessed_test_clip_time_only",
    }
    return global_metrics, per_patient


def save_confusion_matrix_svg(metrics: dict[str, float | int], destination: str | Path) -> None:
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    palette = LinearSegmentedColormap.from_list(
        "benchmark_blue_orange", ["#FDDBC7", "#56B4E9", "#0072B2", "#003366"]
    )
    image = axis.imshow(matrix, cmap=palette)
    axis.set_xticks([0, 1], labels=["Negative", "Positive"])
    axis.set_yticks([0, 1], labels=["Negative", "Positive"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg", bbox_inches="tight")
    plt.close(figure)


def evaluate_predictions(
    predictions: pd.DataFrame,
    task: str,
    threshold: float,
    output_dir: str | Path | None = None,
    bootstrap_seed: int = 1,
    bootstrap_resamples: int = 2000,
    bootstrap_workers: int = 8,
) -> dict[str, object]:
    clip_metrics = clip_level_metrics(predictions["label"], predictions["score"], threshold)
    result: dict[str, object] = {"clip_level": clip_metrics}
    per_patient = None
    if task == "detection":
        event_metrics, per_patient = detection_event_metrics(predictions, threshold)
        result["event_level"] = event_metrics
    elif task != "prediction":
        raise ValueError(f"Unsupported task: {task}")
    result['uncertainty'] = patient_cluster_bootstrap(
        predictions,
        threshold,
        bootstrap_seed,
        bootstrap_resamples,
        event_by_patient=per_patient,
        workers=bootstrap_workers,
    )
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(root / "predictions.csv", index=False)
        (root / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
        save_confusion_matrix_svg(clip_metrics, root / "confusion_matrix.svg")
        from eeg_benchmark.tasks.cross_modal import plot_prediction_explanations

        plot_prediction_explanations(predictions, root)
        if per_patient is not None:
            per_patient.to_csv(root / "event_metrics_by_patient.csv", index=False)
    return result




import json
import os
from pathlib import Path
from typing import Any
from collections.abc import Collection

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import ConcatDataset, Dataset, Sampler
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = None

    class Dataset:
        pass

    class Sampler:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

    class ConcatDataset:
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                'PyTorch is required for PyTorch dataset execution'
            )

from eeg_benchmark.tasks.cross_dataset import (
    DETECTION_UNDERSAMPLE_DATASETS,
    DYNAMIC_NEGATIVE_TO_POSITIVE_RATIO,
    balance_prediction_training_frame,
    undersample_detection_training_frame,
)
from eeg_benchmark.tasks.cross_dataset import (
    SOURCE_REHEARSAL_FRACTION,
    event_balanced_rehearsal_indices,
)


def mission_training_kwargs(spec: Any, stage: str, split: str) -> dict[str, Any]:
    if split != 'train':
        return {}
    dataset_key = spec.source_dataset if stage == 'source' else spec.target_dataset
    return {
        'dataset_key': dataset_key,
        'task': spec.task,
        'undersample_seed': spec.undersample_seed,
        'sampling_audit_root': spec.output_dir / 'sampling' / f'{stage}_train',
    }


def align_channel_metadata(
    channel_names: list[str],
    channel_types: list[str],
    channel_positions: np.ndarray,
    channel_contract: ChannelUnionContract,
    dataset_name: str,
    aligned_channel_count: int,
) -> tuple[list[str], list[str], np.ndarray]:
    positions = np.asarray(channel_positions, dtype=np.float32)
    if positions.shape != (len(channel_names), 3):
        raise ValueError(
            'Channel position shape does not match the native channel metadata: '
            f'{positions.shape} versus {(len(channel_names), 3)}'
        )
    if len(channel_types) != len(channel_names):
        raise ValueError('Channel type count does not match the native channel names')

    is_cross_modal_scalp_bridge = bool(
        channel_contract.policy == 'native_electrode_set_attention'
        and aligned_channel_count == len(SCALP_BIPOLAR_CHANNELS)
        and str(dataset_name).upper() in {'TUSZ', 'SIENA', 'CHB-MIT'}
    )
    if (
        channel_contract.policy == 'standard_1020_bipolar_semantic_bridge'
        or is_cross_modal_scalp_bridge
    ):
        names = list(SCALP_BIPOLAR_CHANNELS)
        return (
            names,
            ['EEG'] * len(names),
            np.full((len(names), 3), np.nan, dtype=np.float32),
        )

    if channel_contract.policy == 'pairwise_name_union_zero_fill_no_interpolation':
        union_index = {
            name: index for index, name in enumerate(channel_contract.channel_keys)
        }
        output_types = ['UNKNOWN'] * len(channel_contract.channel_keys)
        output_positions = np.full(
            (len(channel_contract.channel_keys), 3), np.nan, dtype=np.float32
        )
        for source_index, raw_name in enumerate(channel_names):
            target_index = union_index.get(canonical_channel_name(raw_name))
            if target_index is None:
                continue
            output_types[target_index] = channel_types[source_index]
            output_positions[target_index] = positions[source_index]
        return list(channel_contract.channel_keys), output_types, output_positions

    if channel_contract.policy == 'inductive_permutation_invariant_native_channel_statistics':
        names = list(channel_contract.channel_keys)
        return (
            names,
            ['STATISTIC'] * len(names),
            np.full((len(names), 3), np.nan, dtype=np.float32),
        )

    return list(channel_names), list(channel_types), positions


class UnionClipDataset(Dataset):
    def __init__(
        self,
        task_root: str | Path,
        split: str,
        model_name: str,
        channel_contract: ChannelUnionContract,
        patient_ids: Collection[str] | None = None,
        clip_ids: Collection[str] | None = None,
        event_budget_training: bool = False,
        view_seconds_override: float | None = None,
        dataset_key: str | None = None,
        task: str | None = None,
        undersample_seed: int = 1,
        sampling_audit_root: str | Path | None = None,
    ) -> None:
        from preprocessing.pipeline import MODEL_INPUT_SPECS

        self.task_root = Path(task_root)
        self.model_name = model_name.lower()
        self.task = task
        if self.model_name not in MODEL_INPUT_SPECS:
            raise ValueError(f"Unsupported model input format: {model_name}")
        self.spec = MODEL_INPUT_SPECS[self.model_name]
        manifest = pd.read_csv(self.task_root / "manifest.csv", dtype={"patient_id": str, "clip_id": str})
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        if patient_ids is not None:
            allowed = {str(patient_id) for patient_id in patient_ids}
            self.manifest = self.manifest[
                self.manifest["patient_id"].astype(str).isin(allowed)
            ].reset_index(drop=True)
        if clip_ids is not None:
            allowed_clips = {str(clip_id) for clip_id in clip_ids}
            self.manifest = self.manifest[
                self.manifest['clip_id'].astype(str).isin(allowed_clips)
            ].reset_index(drop=True)
        self.dynamic_detection_sampling = bool(
            split == 'train'
            and (task == 'detection' or event_budget_training)
            and dataset_key in DETECTION_UNDERSAMPLE_DATASETS
        )
        self.dynamic_prediction_sampling = bool(
            split == 'train'
            and task == 'prediction'
        )
        self.prediction_negative_to_positive_ratio = (
            1.0 if task == 'prediction' else 2.0
        )
        self.sampling_dataset_key = dataset_key
        self.undersample_seed = int(undersample_seed)
        self.sampling_audit_root = (
            Path(sampling_audit_root) if sampling_audit_root is not None else None
        )
        if self.manifest.empty:
            raise ValueError(f"No {split} clips in {self.task_root}")
        self.channel_contract = channel_contract
        if view_seconds_override is not None and view_seconds_override <= 0.0:
            raise ValueError("view_seconds_override must be positive")
        self.view_seconds = (
            float(view_seconds_override)
            if view_seconds_override is not None
            else self.spec.view_seconds
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        with np.load(self.task_root / row.relative_path, allow_pickle=False) as archive:
            signal = np.asarray(archive["eeg"], dtype=np.float32)
            channel_names = [str(value) for value in archive["channel_names"]]
            source_sfreq = int(archive["sfreq"])
            label = int(archive["label"])
            channel_mean_uv = np.asarray(archive['channel_mean_uv'], dtype=np.float32) if 'channel_mean_uv' in archive else None
            channel_std_uv = np.asarray(archive['channel_std_uv'], dtype=np.float32) if 'channel_std_uv' in archive else None
            metadata = json.loads(str(archive['metadata_json'].item())) if 'metadata_json' in archive else {}
            channel_types = (
                [str(value) for value in archive['channel_types']]
                if 'channel_types' in archive else ['EEG'] * len(channel_names)
            )
            channel_positions = (
                np.asarray(archive['channel_positions'], dtype=np.float32)
                if 'channel_positions' in archive
                else np.full((len(channel_names), 3), np.nan, dtype=np.float32)
            )
        signal, channel_mask = align_preprocessed_clip(
            signal,
            channel_names,
            self.channel_contract,
            channel_mean_uv,
            channel_std_uv,
            metadata.get('normalization'),
            str(row.dataset),
        )
        channel_names, channel_types, channel_positions = align_channel_metadata(
            channel_names,
            channel_types,
            channel_positions,
            self.channel_contract,
            str(row.dataset),
            int(signal.shape[0]),
        )
        if not (
            signal.shape[0]
            == len(channel_mask)
            == len(channel_names)
            == len(channel_types)
            == channel_positions.shape[0]
        ):
            raise ValueError(
                'Aligned signal and channel metadata counts differ: '
                f'signal={signal.shape[0]},mask={len(channel_mask)},'
                f'names={len(channel_names)},types={len(channel_types)},'
                f'positions={channel_positions.shape[0]}'
            )
        from preprocessing.pipeline import (
            format_model_views,
            resample_clip as preprocess_resample_clip,
            split_native_views as preprocess_split_native_views,
        )

        signal = preprocess_resample_clip(signal, source_sfreq, self.spec.sfreq)
        views, starts = preprocess_split_native_views(
            signal, self.spec.sfreq, self.view_seconds
        )
        formatted = format_model_views(views, self.spec)
        return {
            "eeg": torch.from_numpy(formatted.copy()),
            "channel_mask": torch.from_numpy(channel_mask.copy()),
            "channel_names": channel_names,
            "channel_types": channel_types,
            "channel_positions": torch.from_numpy(channel_positions.copy()),
            "layout": self.spec.layout,
            "label": torch.tensor(label, dtype=torch.long),
            "clip_id": str(row.clip_id),
            "patient_id": str(row.patient_id),
            "dataset": str(row.dataset),
            "montage": str(row.montage),
            "source_relative_path": str(row.source_relative_path),
            "clip_start_seconds": float(row.clip_start_seconds),
            "clip_end_seconds": float(row.clip_end_seconds),
            "seizure_intervals_json": str(getattr(row, "seizure_intervals_json", "[]")),
            "view_start_samples": torch.from_numpy(starts),
        }


class DynamicDetectionSampler(Sampler[int]):
    def __init__(self, dataset: UnionClipDataset) -> None:
        if not dataset.dynamic_detection_sampling:
            raise ValueError('DynamicDetectionSampler requires an eligible detection train dataset')
        self.dataset = dataset
        self.epoch = 0
        labels = dataset.manifest['label'].astype(int)
        self.positive_count = int((labels == 1).sum())
        self.negative_count = int((labels == 0).sum())
        if self.positive_count == 0 or self.negative_count == 0:
            raise ValueError('Dynamic detection sampling requires both classes')

    def __len__(self) -> int:
        negative_count = min(
            self.negative_count,
            int(self.positive_count * DYNAMIC_NEGATIVE_TO_POSITIVE_RATIO),
        )
        return self.positive_count + negative_count

    def __iter__(self):
        epoch = self.epoch
        indexed = self.dataset.manifest.assign(
            _dataset_index=np.arange(len(self.dataset.manifest), dtype=np.int64)
        )
        selected, _ = undersample_detection_training_frame(
            indexed,
            dataset=str(self.dataset.sampling_dataset_key),
            split='train',
            seed=self.dataset.undersample_seed,
            epoch=epoch,
            audit_root=self.dataset.sampling_audit_root,
        )
        indices = selected['_dataset_index'].to_numpy(dtype=np.int64, copy=True)
        rng = np.random.default_rng(self.dataset.undersample_seed + epoch)
        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices.tolist())


class DynamicPredictionSampler(Sampler[int]):
    def __init__(self, dataset: UnionClipDataset) -> None:
        if not dataset.dynamic_prediction_sampling:
            raise ValueError(
                'DynamicPredictionSampler requires cross-dataset prediction train data'
            )
        self.dataset = dataset
        self.epoch = 0
        labels = dataset.manifest['label'].astype(int)
        self.positive_count = int((labels == 1).sum())
        self.negative_count = int((labels == 0).sum())
        if self.positive_count == 0 or self.negative_count == 0:
            raise ValueError('Dynamic prediction sampling requires both classes')

    def __len__(self) -> int:
        anchor_count = (
            self.negative_count
            if str(self.dataset.sampling_dataset_key).strip().lower().replace('-', '').replace('_', '') == 'siena'
            else self.positive_count
        )
        return 2 * anchor_count

    def __iter__(self):
        indexed = self.dataset.manifest.assign(
            _dataset_index=np.arange(len(self.dataset.manifest), dtype=np.int64)
        )
        selected, _ = balance_prediction_training_frame(
            indexed,
            dataset=str(self.dataset.sampling_dataset_key),
            split='train',
            seed=self.dataset.undersample_seed,
            epoch=self.epoch,
            audit_root=self.dataset.sampling_audit_root,
        )
        indices = selected['_dataset_index'].to_numpy(dtype=np.int64, copy=True)
        self.epoch += 1
        return iter(indices.tolist())


def dynamic_training_sampler(
    dataset: UnionClipDataset,
) -> DynamicDetectionSampler | DynamicPredictionSampler | None:
    if dataset.dynamic_detection_sampling:
        return DynamicDetectionSampler(dataset)
    if dataset.dynamic_prediction_sampling:
        return DynamicPredictionSampler(dataset)
    return None


class EventBalancedRehearsalSampler(Sampler[int]):
    def __init__(
        self,
        target_dataset: UnionClipDataset,
        source_dataset: UnionClipDataset,
        seed: int,
        rehearsal_fraction: float = SOURCE_REHEARSAL_FRACTION,
    ) -> None:
        self.target_dataset = target_dataset
        self.source_dataset = source_dataset
        self.seed = int(seed)
        self.rehearsal_fraction = float(rehearsal_fraction)
        self.epoch = 0
        self.dataset = ConcatDataset([target_dataset, source_dataset])
        indices, _ = event_balanced_rehearsal_indices(
            target_dataset.manifest, source_dataset.manifest, self.seed, 0,
            rehearsal_fraction=self.rehearsal_fraction,
            target_dataset=str(target_dataset.sampling_dataset_key),
            source_dataset=str(source_dataset.sampling_dataset_key),
            task=target_dataset.task,
            prediction_negative_to_positive_ratio=(
                target_dataset.prediction_negative_to_positive_ratio
            ),
        )
        self.epoch_size = len(indices)

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self):
        indices, report = event_balanced_rehearsal_indices(
            self.target_dataset.manifest,
            self.source_dataset.manifest,
            self.seed,
            self.epoch,
            rehearsal_fraction=self.rehearsal_fraction,
            target_dataset=str(self.target_dataset.sampling_dataset_key),
            source_dataset=str(self.source_dataset.sampling_dataset_key),
            task=self.target_dataset.task,
            prediction_negative_to_positive_ratio=(
                self.target_dataset.prediction_negative_to_positive_ratio
            ),
        )
        root = self.target_dataset.sampling_audit_root
        if root is not None and os.environ.get('SAMPLING_AUDIT_MODE', 'compact') != 'none':
            root.mkdir(parents=True, exist_ok=True)
            (root / f'rehearsal_epoch_{self.epoch:04d}.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
            )
        self.epoch += 1
        return iter(indices.tolist())


class PatientBalancedClassSampler(Sampler[int]):
    def __init__(self, dataset: UnionClipDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        frame = dataset.manifest
        self.patient_ids = sorted(frame['patient_id'].astype(str).unique())
        if not self.patient_ids:
            raise ValueError('Patient-balanced sampling requires at least one patient')
        self.indices: dict[str, dict[int, np.ndarray]] = {}
        for patient_id in self.patient_ids:
            patient_rows = frame['patient_id'].astype(str).to_numpy() == patient_id
            labels = frame['label'].astype(int).to_numpy()
            by_class = {
                label: np.flatnonzero(patient_rows & (labels == label))
                for label in (0, 1)
            }
            if any(values.size == 0 for values in by_class.values()):
                raise ValueError(
                    'Patient-balanced class sampling requires both classes for every patient: '
                    f'{patient_id}'
                )
            self.indices[patient_id] = by_class
        per_patient = int(np.ceil(len(frame) / len(self.patient_ids)))
        self.samples_per_class = int(np.ceil(per_patient / 2))
        self.samples_per_patient = 2 * self.samples_per_class

    def __len__(self) -> int:
        return len(self.patient_ids) * self.samples_per_patient

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = []
        for patient_id in self.patient_ids:
            for label in (0, 1):
                candidates = self.indices[patient_id][label]
                selected.extend(
                    rng.choice(
                        candidates,
                        size=self.samples_per_class,
                        replace=candidates.size < self.samples_per_class,
                    ).tolist()
                )
        rng.shuffle(selected)
        self.epoch += 1
        return iter(selected)


def union_clip_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    channel_counts = [int(item['channel_mask'].numel()) for item in batch]
    maximum_channels = max(channel_counts)
    padded_eeg = []
    padded_masks = []
    padded_positions = []
    for item, channel_count in zip(batch, channel_counts):
        tensor = item['eeg']
        channel_axis = 2 if item['layout'] == 'time_channel_patch' else 1
        shape = list(tensor.shape)
        shape[channel_axis] = maximum_channels
        destination = torch.zeros(shape, dtype=tensor.dtype)
        slices = [slice(None)] * tensor.ndim
        slices[channel_axis] = slice(0, channel_count)
        destination[tuple(slices)] = tensor
        padded_eeg.append(destination)
        mask = torch.zeros(maximum_channels, dtype=torch.bool)
        mask[:channel_count] = item['channel_mask']
        padded_masks.append(mask)
        item_positions = item['channel_positions']
        if tuple(item_positions.shape) != (channel_count, 3):
            raise ValueError(
                'Collate received channel positions inconsistent with the aligned mask: '
                f'{tuple(item_positions.shape)} versus {(channel_count, 3)}'
            )
        position = torch.full((maximum_channels, 3), float('nan'), dtype=torch.float32)
        position[:channel_count] = item_positions
        padded_positions.append(position)
    return {
        "eeg": torch.stack(padded_eeg),
        "channel_mask": torch.stack(padded_masks),
        "channel_names": [item['channel_names'] for item in batch],
        "channel_types": [item['channel_types'] for item in batch],
        "channel_positions": torch.stack(padded_positions),
        "label": torch.stack([item["label"] for item in batch]),
        "clip_id": [item["clip_id"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "montage": [item["montage"] for item in batch],
        "source_relative_path": [item["source_relative_path"] for item in batch],
        "clip_start_seconds": torch.tensor([item["clip_start_seconds"] for item in batch], dtype=torch.float64),
        "clip_end_seconds": torch.tensor([item["clip_end_seconds"] for item in batch], dtype=torch.float64),
        "seizure_intervals_json": [item["seizure_intervals_json"] for item in batch],
    }


def prediction_frame(batch: dict[str, Any], scores: torch.Tensor) -> pd.DataFrame:
    values = scores.detach().cpu().reshape(-1).numpy()
    labels = batch["label"].detach().cpu().reshape(-1).numpy()
    return pd.DataFrame(
        {
            "clip_id": batch["clip_id"],
            "patient_id": batch["patient_id"],
            "dataset": batch["dataset"],
            "montage": batch["montage"],
            "source_relative_path": batch["source_relative_path"],
            "clip_start_seconds": batch["clip_start_seconds"].numpy(),
            "clip_end_seconds": batch["clip_end_seconds"].numpy(),
            "seizure_intervals_json": batch["seizure_intervals_json"],
            "label": labels.astype(np.int64),
            "score": values.astype(np.float64),
        }
    )




import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

if torch is not None:
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
else:
    F = None
    DataLoader = Any


ClipForward = Callable[[Any, Any, Any], Any]
TrainingObjective = Callable[
    [Any, dict[str, Any], Any, Any, Any, str],
    tuple[Any, Any, dict[str, Any]],
]
EvaluationDiagnostics = Callable[
    [Any, Any, Any, Path, logging.Logger], dict[str, Any]
]


def disable_stochastic_layers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, 'drop_prob'):
            module.drop_prob = 0.0


def load_shape_compatible_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    prefixes: tuple[str, ...] = ('module.', 'model.'),
) -> dict[str, object]:
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        key = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        normalized[key] = value
    current = model.state_dict()
    compatible = {
        key: value for key, value in normalized.items()
        if key in current and current[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    skipped = sorted(key for key in normalized if key not in compatible)
    return {
        'loaded_tensor_count': len(compatible),
        'skipped_keys': skipped,
        'missing_keys': list(result.missing_keys),
        'unexpected_keys': list(result.unexpected_keys),
    }


def fit_clip_stage(
    model: torch.nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    forward_clip: ClipForward,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    patience: int,
    min_delta: float,
    output_dir: Path,
    stage: str,
    logger: logging.Logger,
    max_grad_norm: float,
    unfreeze_epoch: int | None = None,
    unfreeze_parameters: tuple[torch.nn.Parameter, ...] = (),
    training_objective: TrainingObjective | None = None,
) -> Path:
    stage_root = output_dir / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    best_path = stage_root / 'best.pt'
    last_path = stage_root / 'last.pt'
    history_path = stage_root / 'epoch_metrics.csv'
    best_auroc = -np.inf
    early_stop_reference = -np.inf
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []
    model.to(device)
    progress_update_interval = max(
        1, int(getattr(model, 'progress_update_interval', 1))
    )
    for epoch in range(1, epochs + 1):
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            for parameter in unfreeze_parameters:
                parameter.requires_grad = True
            logger.info(
                'stage=%s phase=head_plus_last_feature_block unfreeze_epoch=%d',
                stage, epoch,
            )
        model.train()
        running_loss = 0.0
        running_terms: dict[str, float] = {}
        seen = 0
        pending_statistics: list[
            tuple[torch.Tensor, dict[str, Any], int]
        ] = []
        progress = tqdm(
            train_loader,
            desc=f'{stage} epoch {epoch}/{epochs}',
            colour='green',
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress, start=1):
            eeg = batch['eeg'].to(device, non_blocking=True)
            mask = batch['channel_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            if training_objective is None:
                logits = forward_clip(model, eeg, mask).reshape(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss_terms = {'classification_loss': loss.detach()}
            else:
                logits, loss, loss_terms = training_objective(
                    model, batch, eeg, mask, labels, stage
                )
                logits = logits.reshape(-1)
                if logits.shape != labels.shape:
                    raise ValueError(
                        f'{stage} training objective returned incompatible logits'
                    )
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError(
                        f'{stage} training objective returned an invalid scalar loss'
                    )
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            batch_size = labels.numel()
            detached_terms = {
                name: value.detach() if torch.is_tensor(value) else float(value)
                for name, value in loss_terms.items()
            }
            pending_statistics.append((loss.detach(), detached_terms, batch_size))
            should_refresh = (
                batch_index % progress_update_interval == 0
                or batch_index == len(train_loader)
            )
            if should_refresh:
                loss_values = torch.stack([
                    value for value, _, _ in pending_statistics
                ]).cpu().tolist()
                term_names = sorted({
                    name
                    for _, terms, _ in pending_statistics
                    for name in terms
                })
                term_values = {
                    name: torch.stack([
                        (
                            terms[name].detach()
                            if torch.is_tensor(terms[name])
                            else loss.new_tensor(float(terms[name]))
                        )
                        for loss, terms, _ in pending_statistics
                    ]).cpu().tolist()
                    for name in term_names
                }
                for pending_index, (_, _, pending_size) in enumerate(
                    pending_statistics
                ):
                    running_loss += float(loss_values[pending_index]) * pending_size
                    for name in term_names:
                        scalar = float(term_values[name][pending_index])
                        if not math.isfinite(scalar):
                            raise ValueError(
                                f'{stage} training objective term is not finite: {name}'
                            )
                        running_terms[name] = (
                            running_terms.get(name, 0.0) + scalar * pending_size
                        )
                    seen += pending_size
                pending_statistics.clear()
                postfix = {'loss': f'{running_loss / max(seen, 1):.6f}'}
                if 'slot_alignment_loss' in running_terms:
                    postfix['align'] = (
                        f'{running_terms["slot_alignment_loss"] / max(seen, 1):.6f}'
                    )
                progress.set_postfix(**postfix)

        dev_labels, dev_scores = collect_validation_scores(
            model, dev_loader, device, forward_clip
        )
        if np.unique(dev_labels).size < 2:
            raise ValueError(f'{stage} dev split must contain both classes')
        val_auroc = float(roc_auc_score(dev_labels, dev_scores))
        train_loss = running_loss / max(seen, 1)
        learning_rate = float(optimizer.param_groups[0]['lr'])
        epoch_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_auroc': val_auroc,
            'learning_rate': learning_rate,
        }
        averaged_terms = {
            name: value / max(seen, 1)
            for name, value in sorted(running_terms.items())
        }
        epoch_row.update(averaged_terms)
        rows.append(epoch_row)
        pd.DataFrame(rows).to_csv(history_path, index=False)
        logger.info(
            'stage=%s epoch=%d train_loss=%.8f val_auroc=%.8f lr=%.8g',
            stage,
            epoch,
            train_loss,
            val_auroc,
            learning_rate,
        )
        if averaged_terms:
            logger.info(
                'stage=%s epoch=%d objective_terms=%s',
                stage,
                epoch,
                json.dumps(averaged_terms, sort_keys=True),
            )
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_auroc': val_auroc}, last_path)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_auroc': val_auroc}, best_path)
        early_stopping_active = unfreeze_epoch is None or epoch >= unfreeze_epoch
        if early_stopping_active:
            if val_auroc > early_stop_reference + min_delta:
                early_stop_reference = val_auroc
                stale_epochs = 0
            else:
                stale_epochs += 1
        if scheduler is not None:
            scheduler.step()
        if early_stopping_active and stale_epochs >= patience:
            logger.info(
                'stage=%s early_stop_epoch=%d patience=%d min_delta=%.8f',
                stage, epoch, patience, min_delta,
            )
            break

    if not best_path.exists():
        raise RuntimeError(f'{stage} did not create a best checkpoint')
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'], strict=True)
    return best_path


def resolve_source_reference_checkpoint(reference_output_dir: Path) -> Path:
    candidates = [
        reference_output_dir / 'source' / 'best.ckpt',
        reference_output_dir / 'source' / 'best.pt',
        reference_output_dir / 'source' / 'best.pth',
        reference_output_dir / 'source' / 'best.pth.tar',
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        'No reference checkpoint found in source directory: '
        f'{[str(path) for path in candidates]}'
    )


def collect_clip_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_clip: ClipForward,
) -> pd.DataFrame:
    model.to(device).eval()
    frames: list[pd.DataFrame] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc='evaluation', colour='green', dynamic_ncols=True):
            eeg = batch['eeg'].to(device, non_blocking=True)
            mask = batch['channel_mask'].to(device, non_blocking=True)
            scores = torch.sigmoid(forward_clip(model, eeg, mask).reshape(-1))
            frames.append(prediction_frame(batch, scores))
    if not frames:
        raise ValueError('Evaluation loader is empty')
    return pd.concat(frames, ignore_index=True)


def collect_validation_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_clip: ClipForward,
) -> tuple[np.ndarray, np.ndarray]:
    model.to(device).eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in tqdm(
            loader, desc='validation', colour='green', dynamic_ncols=True
        ):
            eeg = batch['eeg'].to(device, non_blocking=True)
            mask = batch['channel_mask'].to(device, non_blocking=True)
            batch_scores = torch.sigmoid(
                forward_clip(model, eeg, mask).reshape(-1)
            )
            labels.append(batch['label'].numpy().astype(np.int64, copy=False))
            scores.append(batch_scores.cpu().numpy().astype(np.float64, copy=False))
    if not labels:
        raise ValueError('Validation loader is empty')
    return np.concatenate(labels), np.concatenate(scores)




import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from eeg_benchmark.tasks.cross_dataset import mission_sampling_summary
from eeg_benchmark.tasks.cross_dataset import (
    add_mission_arguments,
    build_spec,
    prepare_mission,
    resolve_in_domain_source_checkpoint,
    validate_zero_shot_reference,
    validate_mode_args,
)


ModelFactory = Callable[[ChannelUnionContract], Any]
PretrainedLoader = Callable[[Any, Path, ChannelUnionContract], dict[str, object]]
TransferConfigurator = Callable[[Any, str], None]
BudgetModuleResolver = Callable[[Any], tuple[Any, Any]]
TargetInputAdapter = Callable[[Any, ChannelUnionContract], None]
ClipForward = Callable[[Any, Any, Any], Any]


def make_loader(dataset, args: argparse.Namespace, shuffle: bool, patient_balanced: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    sampler = (
        PatientBalancedClassSampler(dataset, seed=args.seed)
        if shuffle and patient_balanced
        else dynamic_training_sampler(dataset) if shuffle else None
    )
    loader_options = {}
    if args.num_workers > 0:
        loader_options['prefetch_factor'] = int(getattr(args, 'prefetch_factor', 2))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=union_clip_collate,
        generator=generator,
        **loader_options,
    )


def make_budget_loader(target_dataset, source_dataset, args) -> DataLoader:
    sampler = EventBalancedRehearsalSampler(
        target_dataset, source_dataset, args.undersample_seed,
        rehearsal_fraction=args.source_rehearsal_fraction,
    )
    loader_options = {}
    if args.num_workers > 0:
        loader_options['prefetch_factor'] = int(getattr(args, 'prefetch_factor', 2))
    return DataLoader(
        sampler.dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=union_clip_collate,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )


def run_torch_transfer(
    args: argparse.Namespace,
    model_name: str,
    input_spec_name: str,
    model_factory: ModelFactory,
    pretrained_loader: PretrainedLoader,
    transfer_configurator: TransferConfigurator,
    forward_clip: ClipForward,
    target_input_adapter: TargetInputAdapter | None = None,
    budget_module_resolver: BudgetModuleResolver | None = None,
    handles_native_electrodes: bool = False,
    use_full_clip: bool = False,
    model_arguments: dict[str, object] | None = None,
    training_objective: TrainingObjective | None = None,
    evaluation_diagnostics: EvaluationDiagnostics | None = None,
    runtime_arguments: dict[str, object] | None = None,
) -> None:
    from eeg_benchmark.tasks.cross_modal import (
        CROSS_MODAL_ADAPTER_POLICY,
        NativeElectrodeAdapterContract,
        apply_native_electrode_adapter,
        attach_native_electrode_adapter,
        collect_native_contact_evidence,
        finalize_standard_torch_interpretability,
        run_epilepsy_localization_task,
        safe_refresh_budget_interpretability,
    )

    validate_mode_args(args)
    spec = build_spec(args, model_name)
    reproducibility = configure_reproducibility(spec.seed, spec.deterministic)
    if int(getattr(args, 'prefetch_factor', 2)) <= 0:
        raise ValueError('prefetch_factor must be positive')
    torch.cuda.set_device(spec.gpu)
    device = torch.device(f'cuda:{spec.gpu}')
    logger = setup_run_logger(spec.output_dir, f'benchmark.task1.{model_name.lower()}')
    source_root, target_root, budget_selection = prepare_mission(spec)
    contract = build_channel_union(
        source_root / 'manifest.csv',
        target_root / 'manifest.csv',
        spec.source_dataset,
        spec.target_dataset,
    )
    contract.save(spec.output_dir / 'channel_union.json')
    split_specs = {
        'source_train': (source_root, 'train'),
        'source_dev': (source_root, 'dev'),
        'target_test': (target_root, 'test'),
    }
    if spec.budget_percent > 0.0:
        split_specs.update({
            'target_train': (target_root, 'train'),
            'target_dev': (target_root, 'dev'),
        })
    datasets = {}
    for name, (root, split) in split_specs.items():
        training_kwargs = mission_training_kwargs(
            spec,
            'target' if name.startswith('target') else 'source',
            split,
        )
        if spec.task == 'localization' and name.startswith('source'):
            training_kwargs['task'] = 'detection'
        datasets[name] = UnionClipDataset(
            root,
            split,
            input_spec_name,
            contract,
            clip_ids=(
                budget_selection.clip_ids(split)
                if budget_selection is not None
                and name in {'target_train', 'target_dev'}
                else None
            ),
            event_budget_training=name == 'target_train' and budget_selection is not None,
            view_seconds_override=(spec.window_seconds if use_full_clip else None),
            **training_kwargs,
        )
    loaders = {
        name: make_loader(
            dataset, args, name.endswith('train'),
            patient_balanced=False,
        )
        for name, dataset in datasets.items()
    }
    if spec.budget_percent > 0.0:
        loaders['target_train'] = make_budget_loader(
            datasets['target_train'], datasets['source_train'], args
        )
    model = model_factory(contract)
    model.progress_update_interval = max(
        1, int(getattr(args, 'progress_update_interval', 1))
    )
    if contract.policy == CROSS_MODAL_ADAPTER_POLICY and not handles_native_electrodes:
        adapter_contract = attach_native_electrode_adapter(
            model, contract.policy, datasets['source_train'].spec.layout,
            NativeElectrodeAdapterContract(
                sampling_frequency=datasets['source_train'].spec.sfreq
            ),
        )
        save_json(spec.output_dir / 'cross_modal_adapter.json', adapter_contract)
        original_forward_clip = forward_clip

        def forward_clip(selected_model, eeg, channel_mask):
            adapted, adapted_mask = apply_native_electrode_adapter(
                selected_model, eeg, channel_mask
            )
            return original_forward_clip(selected_model, adapted, adapted_mask)
    elif contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        contract_resolver = getattr(model, 'cross_modal_adapter_contract', None)
        if not callable(contract_resolver):
            raise ValueError(
                f'{model_name} declares native electrode handling without an adapter contract'
            )
        adapter_contract = contract_resolver()
        if adapter_contract.get('policy') != CROSS_MODAL_ADAPTER_POLICY:
            raise ValueError(
                f'{model_name} native electrode policy does not match the mission contract'
            )
        save_json(spec.output_dir / 'cross_modal_adapter.json', adapter_contract)
    pretrained_report = None
    if spec.use_pretrained:
        pretrained_path = spec.pretrained_path()
        if pretrained_path is None or not pretrained_path.exists():
            raise FileNotFoundError(f'Pretrained weight does not exist: {pretrained_path}')
        pretrained_report = pretrained_loader(model, pretrained_path, contract)
        save_json(spec.output_dir / 'pretrained_load_report.json', pretrained_report)
    total, trainable = count_parameters(model)
    model_information_resolver = getattr(model, 'model_information', None)
    model_information = (
        model_information_resolver()
        if callable(model_information_resolver)
        else {}
    )
    print_model_information({
        'Model': model_name,
        'Mission': spec.mission_type,
        'In-domain source policy': (
            'reuse canonical completed source checkpoint'
            if spec.is_in_domain else 'train source model'
        ),
        'Parameters': total,
        'Trainable parameters': trainable,
        'Mode': spec.mode,
        'Budget percent': spec.budget_percent,
        'Budget unit': spec.to_dict()['budget_unit'],
        'Budget seed': spec.budget_seed,
        'Window seconds': spec.window_seconds,
        'Task': spec.task,
        'Source': spec.source_dataset,
        'Target': spec.target_dataset,
        'Use pretrained': spec.use_pretrained,
        'Channel policy': contract.policy,
        'Contract channels': len(contract.channel_keys),
        'Source available channels': len(contract.source_channel_keys),
        'Target available channels': len(contract.target_channel_keys),
        'Epochs': spec.epochs,
        'Patience': spec.patience,
        'Budget epochs': args.budget_epochs,
        'Budget patience': args.budget_patience,
        'Budget fine-tuning': 'linear_probe_then_full_model_discriminative_lr',
        'Budget head-only epochs': args.budget_head_only_epochs,
        'Early stopping min delta': spec.min_delta,
        'Seed': spec.seed,
        'GPU': spec.physical_gpu,
        'Logical GPU after visibility mask': spec.gpu,
        'Batch size': spec.batch_size,
        'DataLoader workers': spec.num_workers,
        'DataLoader prefetch factor': int(getattr(args, 'prefetch_factor', 2)),
        'Statistics CPU workers': spec.stats_num_workers,
        'Bootstrap resamples': spec.bootstrap_resamples,
        'Training sampling': mission_sampling_summary(spec),
        'Source rehearsal fraction': spec.source_rehearsal_fraction,
        'Undersample seed': spec.undersample_seed,
        'Interpretability': spec.generate_interpretability,
        'Output': spec.output_dir,
        **model_information,
    })
    native_arguments = {
        'learning_rate': args.lr,
        'target_learning_rate': args.target_lr,
        'target_backbone_learning_rate': args.target_backbone_lr,
        'weight_decay': args.weight_decay,
        'maximum_gradient_norm': args.max_grad_norm,
        **(model_arguments or {}),
    }
    save_json(spec.output_dir / 'args.json', {
        **spec.to_dict(),
        **native_arguments,
        'runtime_arguments': {
            **(runtime_arguments or {}),
            'prefetch_factor': int(getattr(args, 'prefetch_factor', 2)),
        },
        'budget_epochs': args.budget_epochs,
        'budget_patience': args.budget_patience,
        'budget_head_only_epochs': args.budget_head_only_epochs,
        'budget_finetune_strategy': 'linear_probe_then_full_model_discriminative_lr',
        'reproducibility': reproducibility,
        'pretrained_report': pretrained_report,
        'reuse_incomplete_source_checkpoint': bool(args.reuse_incomplete_source_checkpoint),
    })
    if runtime_arguments and 'parameter_audit' in runtime_arguments:
        save_json(
            spec.output_dir / 'parameter_audit.json',
            runtime_arguments['parameter_audit'],
        )

    if spec.task == 'localization':
        reference_output_dir = spec.source_reference_output_dir
        source_checkpoint = spec.source_reference_checkpoint_path
        if not source_checkpoint.is_file():
            source_checkpoint = resolve_source_reference_checkpoint(reference_output_dir)
        validate_zero_shot_reference(
            spec,
            source_checkpoint,
            native_arguments,
            contract,
            reference_output_dir=reference_output_dir,
            reference_task='detection',
            reference_mission_type='eeg_ieeg_transfer',
        )
        checkpoint = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        save_json(spec.output_dir / 'reference_checkpoint.json', {
            'path': str(source_checkpoint),
            'reference_output_dir': str(reference_output_dir),
            'policy': 'reuse_cross_modal_detection_source_stage_checkpoint',
            'epoch': checkpoint.get('epoch'),
            'validation_auroc': checkpoint.get('val_auroc'),
        })
        configured_localization_batch_size = int(
            os.environ.get('LOCALIZATION_BATCH_SIZE', '0') or '0'
        )
        evidence_batch_size = (
            configured_localization_batch_size
            if configured_localization_batch_size > 0
            else 1
            if model_name == 'EEGPT'
            else None
        )
        logger.info(
            'localization_evidence_batch_size=%s model=%s',
            (
                evidence_batch_size
                if evidence_batch_size is not None
                else loaders['target_test'].batch_size
            ),
            model_name,
        )
        evidence, cohort = collect_native_contact_evidence(
            model,
            loaders['target_test'],
            device,
            forward_clip,
            dataset=spec.target_dataset,
            task=spec.task,
            maximum_per_patient_class=max(1, min(32, int(spec.interpretability_max_clips))),
            seed=spec.budget_seed,
            cohort_root=(
                Path(spec.result_root) / spec.mission_type / 'xai_cohorts'
                / spec.window_name
            ),
            evidence_batch_size=evidence_batch_size,
        )
        root = spec.output_dir / 'localization'
        root.mkdir(parents=True, exist_ok=True)
        evidence.to_csv(spec.output_dir / 'predictions.csv', index=False)
        evidence.to_csv(root / 'native_contact_evidence.csv', index=False)
        (root / 'fixed_cohort.json').write_text(json.dumps(cohort, indent=2), encoding='utf-8')
        target_root_contract = json.loads((target_root / 'dataset_contract.json').read_text(encoding='utf-8'))
        raw_root = target_root_contract.get('source_root')
        if not raw_root:
            raise ValueError('Localization requires source_root in target dataset contract')
        summary = run_epilepsy_localization_task(evidence, raw_root, spec.output_dir)
        save_json(spec.output_dir / 'metrics.json', summary)
        save_json(spec.output_dir / 'run_summary.json', {
            'status': 'complete',
            'task': spec.task,
            'localization': summary,
            'reference_checkpoint': str(source_checkpoint),
            'reference_output_dir': str(reference_output_dir),
            'cohort_summary': {
                'cohort_sha256': cohort['cohort_sha256'],
                'clip_count': cohort['clip_count'],
                'patient_count': cohort['patient_count'],
                'maximum_clips_per_patient_per_class': cohort['maximum_clips_per_patient_per_class'],
                'sampling_seed': cohort['sampling_seed'],
            },
        })
        logger.info('completed localization task model=%s output=%s', model_name, spec.output_dir)
        return

    if spec.budget_percent == 0.0:
        recoverable_checkpoint = spec.output_dir / 'source' / 'best.pt'
        if spec.is_in_domain:
            source_checkpoint, source_reference = resolve_in_domain_source_checkpoint(
                spec,
                'best.pt',
                native_arguments,
                contract,
            )
            checkpoint = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
            save_json(spec.output_dir / 'source_checkpoint_reference.json', source_reference)
            logger.info('In-domain source checkpoint loaded without source retraining: %s', source_checkpoint)
        elif args.reuse_incomplete_source_checkpoint:
            if not recoverable_checkpoint.is_file():
                raise FileNotFoundError(
                    f'Requested incomplete-run recovery checkpoint is missing: {recoverable_checkpoint}'
                )
            checkpoint = torch.load(
                recoverable_checkpoint, map_location='cpu', weights_only=False
            )
            model.load_state_dict(checkpoint['model'], strict=True)
            save_json(spec.output_dir / 'source_checkpoint_recovery.json', {
                'path': str(recoverable_checkpoint),
                'epoch': checkpoint.get('epoch'),
                'validation_auroc': checkpoint.get('val_auroc'),
                'policy': 'reuse_trained_source_weights_after_post_training_failure',
            })
            logger.info(
                'recovered incomplete source run checkpoint=%s epoch=%s val_auroc=%s',
                recoverable_checkpoint,
                checkpoint.get('epoch'),
                checkpoint.get('val_auroc'),
            )
        else:
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs, 1),
            )
            fit_clip_stage(
                model,
                loaders['source_train'],
                loaders['source_dev'],
                device,
                forward_clip,
                optimizer,
                scheduler,
                args.epochs,
                args.patience,
                args.min_delta,
                spec.output_dir,
                'source',
                logger,
                args.max_grad_norm,
                training_objective=training_objective,
            )
    else:
        source_checkpoint = spec.zero_shot_output_dir / 'source' / 'best.pt'
        validate_zero_shot_reference(
            spec, source_checkpoint, native_arguments, contract
        )
        checkpoint = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        save_json(spec.output_dir / 'source_checkpoint_reference.json', {
            'path': str(source_checkpoint),
            'epoch': checkpoint.get('epoch'),
            'validation_auroc': checkpoint.get('val_auroc'),
        })
    if target_input_adapter is not None:
        target_input_adapter(model, contract)

    if spec.budget_percent == 0.0:
        selection_loader = loaders['source_dev']
        threshold_source = 'source_dev_max_f1'
    else:
        transfer_configurator(model, spec.mode)
        for parameter in model.parameters():
            parameter.requires_grad = True
        trainable_parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if not trainable_parameters:
            raise ValueError(f'{model_name} full-model fine-tuning has no trainable parameters')
        target_total, target_trainable = count_parameters(model)
        logger.info(
            'target_stage strategy=full_model mode=%s total_parameters=%d trainable_parameters=%d',
            spec.mode,
            target_total,
            target_trainable,
        )
        if budget_module_resolver is None:
            raise ValueError(
                f'{model_name} requires a classifier-head resolver for '
                'discriminative full-model fine-tuning'
            )
        head_module, _ = budget_module_resolver(model)
        head_parameter_ids = {
            id(parameter) for parameter in head_module.parameters()
        }
        head_parameters = [
            parameter for parameter in trainable_parameters
            if id(parameter) in head_parameter_ids
        ]
        backbone_parameters = [
            parameter for parameter in trainable_parameters
            if id(parameter) not in head_parameter_ids
        ]
        if not head_parameters or not backbone_parameters:
            raise ValueError(
                f'{model_name} could not separate classifier and backbone parameters'
            )
        if args.budget_head_only_epochs > 0:
            for parameter in backbone_parameters:
                parameter.requires_grad = False
            head_optimizer = torch.optim.AdamW(
                head_parameters,
                lr=args.target_lr,
                weight_decay=args.weight_decay,
            )
            head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                head_optimizer,
                T_max=max(args.budget_head_only_epochs, 1),
            )
            logger.info(
                'target_stage phase=linear_probe epochs=%d head_lr=%g '
                'head_parameters=%d backbone_trainable=0',
                args.budget_head_only_epochs,
                args.target_lr,
                sum(parameter.numel() for parameter in head_parameters),
            )
            fit_clip_stage(
                model,
                loaders['target_train'],
                loaders['target_dev'],
                device,
                forward_clip,
                head_optimizer,
                head_scheduler,
                args.budget_head_only_epochs,
                args.budget_head_only_epochs + 1,
                args.min_delta,
                spec.output_dir,
                'target_linear_probe',
                logger,
                args.max_grad_norm,
                training_objective=training_objective,
            )
        for parameter in model.parameters():
            parameter.requires_grad = True
        target_optimizer = torch.optim.AdamW(
            [
                {'params': backbone_parameters, 'lr': args.target_backbone_lr},
                {'params': head_parameters, 'lr': args.target_lr},
            ],
            weight_decay=args.weight_decay,
        )
        logger.info(
            'target_stage phase=full_model strategy=discriminative_lr '
            'head_lr=%g backbone_lr=%g head_parameters=%d backbone_parameters=%d',
            args.target_lr,
            args.target_backbone_lr,
            sum(parameter.numel() for parameter in head_parameters),
            sum(parameter.numel() for parameter in backbone_parameters),
        )
        target_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            target_optimizer,
            T_max=max(args.budget_epochs, 1),
        )
        fit_clip_stage(
            model,
            loaders['target_train'],
            loaders['target_dev'],
            device,
            forward_clip,
            target_optimizer,
            target_scheduler,
            args.budget_epochs,
            args.budget_patience,
            args.min_delta,
            spec.output_dir,
            'target',
            logger,
            args.max_grad_norm,
            training_objective=training_objective,
        )
        selection_loader = loaders['target_dev']
        threshold_source = 'target_dev_max_f1'

    dev_predictions = collect_clip_predictions(model, selection_loader, device, forward_clip)
    dev_predictions.to_csv(spec.output_dir / 'selection_predictions.csv', index=False)
    validation_auroc = float(roc_auc_score(
        dev_predictions['label'], dev_predictions['score']
    ))
    threshold = select_f1_threshold(dev_predictions['label'], dev_predictions['score'])
    target_predictions = collect_clip_predictions(model, loaders['target_test'], device, forward_clip)
    target_predictions['predicted_label'] = (target_predictions['score'] >= threshold).astype(np.int64)
    diagnostic_status = (
        evaluation_diagnostics(
            model,
            loaders['target_test'],
            device,
            spec.output_dir,
            logger,
        )
        if evaluation_diagnostics is not None
        else {'status': 'not_requested'}
    )
    metrics = evaluate_predictions(
        target_predictions,
        spec.task,
        threshold,
        spec.output_dir,
        bootstrap_seed=spec.seed,
        bootstrap_resamples=spec.bootstrap_resamples,
        bootstrap_workers=spec.stats_num_workers,
    )
    interpretability_status = finalize_standard_torch_interpretability(
        spec, model, loaders['target_test'], device, forward_clip,
        source_root, target_root, target_predictions,
    )
    run_summary = {
        'status': 'complete',
        'selection_metric': 'validation_auroc',
        'threshold_source': threshold_source,
        'validation_auroc': validation_auroc,
        'metrics': metrics,
        'model_diagnostics': diagnostic_status,
        'interpretability': interpretability_status,
    }
    save_json(spec.output_dir / 'run_summary.json', run_summary)
    run_summary['budget_interpretability'] = safe_refresh_budget_interpretability(spec)
    save_json(spec.output_dir / 'run_summary.json', run_summary)
    logger.info('completed model=%s output=%s', model_name, spec.output_dir)


def add_common_transfer_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_mission_arguments(parser)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--target-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument(
        '--localization-reference-window-seconds',
        type=float,
        default=None,
        help='Window length of completed detection checkpoints used by localization.',
    )
    parser.add_argument(
        '--reuse-incomplete-source-checkpoint', type=int, choices=[0, 1], default=0
    )
    return parser
