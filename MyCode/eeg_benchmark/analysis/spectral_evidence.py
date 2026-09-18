#!/usr/bin/env python3
"""Export reproducible score-weighted spectral evidence from completed runs.

This module deliberately does not retrain or alter a checkpoint. It joins the
saved test predictions to the immutable preprocessing manifest, computes
relative band power for each channel, weights it by the saved model score, and
records the checkpoint hash for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (12.0, 30.0),
    'low_gamma': (30.0, 50.0),
    'high_gamma': (50.0, 95.0),
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parse_bands(value: str) -> dict[str, tuple[float, float]]:
    if not value:
        return dict(DEFAULT_BANDS)
    bands: dict[str, tuple[float, float]] = {}
    for item in value.split(','):
        name, low, high = item.split(':')
        low_value = float(low)
        high_value = float(high)
        if not (0.0 <= low_value < high_value):
            raise ValueError(f'Invalid band: {item}')
        bands[name] = (low_value, high_value)
    return bands


def _find_checkpoint(run_root: Path) -> Path:
    candidates = [
        run_root / 'source' / 'best.pt',
        run_root / 'source' / 'best.pth',
        run_root / 'source' / 'best.pth.tar',
        run_root / 'source' / 'best.ckpt',
        run_root / 'source' / 'best.weights.h5',
        run_root / 'source' / 'best_model.pt',
        run_root / 'source' / 'checkpoint.pt',
        run_root / 'source' / 'checkpoint.pth',
        run_root / 'source' / 'model.pt',
        run_root / 'source' / 'model.pth',
        run_root / 'target' / 'best.pt',
        run_root / 'target' / 'best.pth',
        run_root / 'target' / 'best.pth.tar',
        run_root / 'target' / 'best.ckpt',
        run_root / 'target' / 'best.weights.h5',
        run_root / 'target' / 'best_model.pt',
        run_root / 'target' / 'checkpoint.pt',
        run_root / 'target' / 'checkpoint.pth',
        run_root / 'target' / 'model.pt',
        run_root / 'target' / 'model.pth',
        run_root / 'best_model.joblib',
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    discovered = sorted(
        candidate
        for candidate in run_root.rglob('*')
        if candidate.is_file()
        and candidate.stat().st_size > 0
        and candidate.suffix.lower() in {'.pt', '.pth', '.ckpt', '.joblib', '.h5', '.weights', '.tar'}
    )
    if discovered:
        return discovered[0]
    raise FileNotFoundError(f'No non-empty checkpoint found below {run_root}')


def _validate_run(run_root: Path) -> tuple[pd.DataFrame, Path]:
    metrics_path = run_root / 'metrics.json'
    predictions_path = run_root / 'predictions.csv'
    if not metrics_path.is_file():
        raise FileNotFoundError(f'Missing metrics.json: {metrics_path}')
    if not predictions_path.is_file():
        raise FileNotFoundError(f'Missing predictions.csv: {predictions_path}')
    checkpoint = _find_checkpoint(run_root)
    predictions = pd.read_csv(predictions_path, dtype={'clip_id': str, 'patient_id': str})
    required = {'clip_id', 'patient_id', 'label', 'score'}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f'Predictions lack required columns: {sorted(missing)}')
    predictions['clip_id'] = predictions['clip_id'].astype(str)
    predictions['patient_id'] = predictions['patient_id'].astype(str)
    predictions['label'] = predictions['label'].astype(int)
    predictions['score'] = pd.to_numeric(predictions['score'], errors='coerce')
    predictions = predictions.dropna(subset=['score']).reset_index(drop=True)
    return predictions, checkpoint


def _load_test_manifest(target_root: Path) -> pd.DataFrame:
    manifest_path = target_root / 'manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing target manifest.csv: {manifest_path}')
    manifest = pd.read_csv(
        manifest_path,
        dtype={'clip_id': str, 'patient_id': str},
    )
    required = {'clip_id', 'patient_id', 'split'}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f'Manifest lacks required columns: {sorted(missing)}')
    if 'relative_path' not in manifest.columns:
        path_columns = [
            'relative_path',
            'path',
            'file_path',
            'archive_path',
            'npz_path',
            'filepath',
        ]
        for column in path_columns:
            if column in manifest.columns:
                manifest['relative_path'] = manifest[column].astype(str)
                break
    if 'relative_path' not in manifest.columns:
        archives = {
            path.stem: path.relative_to(target_root).as_posix()
            for path in sorted(target_root.rglob('*.npz'))
            if path.is_file()
        }
        manifest['relative_path'] = manifest['clip_id'].astype(str).map(archives)
    manifest = manifest.loc[manifest['split'].astype(str).eq('test')].copy()
    manifest['clip_id'] = manifest['clip_id'].astype(str)
    manifest['patient_id'] = manifest['patient_id'].astype(str)
    if 'relative_path' not in manifest.columns or manifest['relative_path'].isna().any():
        relative_path = manifest['relative_path'] if 'relative_path' in manifest.columns else pd.Series(index=manifest.index, dtype=object)
        missing_paths = manifest.loc[relative_path.isna(), 'clip_id'].astype(str).head(10).tolist()
        raise ValueError(
            f'Manifest lacks usable relative_path and npz inference failed: target={target_root}, examples={missing_paths}'
        )
    manifest['relative_path'] = manifest['relative_path'].astype(str)
    if manifest.empty:
        raise ValueError(f'Target test manifest is empty: {manifest_path}')
    return manifest


def _select_patients(
    merged: pd.DataFrame,
    patient_count: int,
    selection_seed: int,
) -> list[str]:
    counts = (
        merged.groupby('patient_id', as_index=False)
        .agg(test_clip_count=('clip_id', 'count'), positive_clip_count=('label', 'sum'))
    )
    if counts.empty:
        return []
    rng = np.random.default_rng(selection_seed)
    counts['_tie'] = rng.random(len(counts))
    counts = counts.sort_values(
        ['positive_clip_count', 'test_clip_count', '_tie', 'patient_id'],
        ascending=[False, False, True, True],
        kind='mergesort',
    )
    return counts.head(patient_count)['patient_id'].astype(str).tolist()


def _normalize_merged_relative_path(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    if 'relative_path' in out.columns:
        out['relative_path'] = out['relative_path'].astype(str)
        return out
    candidates = [
        'relative_path_y',
        'target_relative_path',
        'path_y',
        'file_path_y',
        'archive_path_y',
        'npz_path_y',
        'filepath_y',
        'relative_path_x',
        'path_x',
        'file_path_x',
        'archive_path_x',
        'npz_path_x',
        'filepath_x',
    ]
    for column in candidates:
        if column in out.columns:
            out['relative_path'] = out[column].astype(str)
            return out
    raise ValueError(f'Merged predictions lack usable relative_path column: columns={list(out.columns)}')


def _band_power(
    signal: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    if signal.ndim != 2:
        raise ValueError(f'Expected channel x time signal, got shape={signal.shape}')
    finite = np.nan_to_num(signal.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = finite.shape[-1]
    if n_samples < 4 or not math.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError(f'Invalid signal metadata: samples={n_samples}, sfreq={sfreq}')
    demeaned = finite - finite.mean(axis=-1, keepdims=True)
    window = np.hanning(n_samples)
    spectrum = np.fft.rfft(demeaned * window[None, :], axis=-1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    total_mask = (frequencies >= 1.0) & (frequencies <= min(95.0, sfreq / 2.0))
    total = power[:, total_mask].sum(axis=-1)
    total = np.maximum(total, np.finfo(np.float64).eps)
    result: dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < min(high, sfreq / 2.0 + 1e-9))
        result[name] = power[:, mask].sum(axis=-1) / total
    return result


def _iter_rows(
    merged: pd.DataFrame,
    target_root: Path,
    selected_patients: Iterable[str],
    clips_per_patient: int,
    bands: dict[str, tuple[float, float]],
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_set = set(str(value) for value in selected_patients)
    subset = merged.loc[merged['patient_id'].isin(selected_set)].copy()
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for patient_id, group in subset.groupby('patient_id', sort=True):
        indices = group.index.to_numpy()
        order = rng.permutation(indices)
        selected_indices.extend(order[:clips_per_patient].tolist())
    clip_records: list[dict[str, object]] = []
    for index in tqdm(
        selected_indices,
        desc='Exporting spectral evidence',
        colour='magenta',
        unit='clip',
    ):
        row = merged.loc[index]
        archive_path = target_root / str(row['relative_path'])
        with np.load(archive_path, allow_pickle=False) as archive:
            signal = np.asarray(archive['eeg'], dtype=np.float32)
            channel_names = [str(value) for value in archive['channel_names']]
            sfreq = float(np.asarray(archive['sfreq']).reshape(-1)[0])
        if signal.shape[0] != len(channel_names):
            raise ValueError(f'Channel count mismatch in {archive_path}')
        relative_power = _band_power(signal, sfreq, bands)
        score = float(row['score'])
        signed_weight = 2.0 * score - 1.0
        positive_weight = max(score, 0.0)
        for channel_index, channel_name in enumerate(channel_names):
            for band_name in bands:
                value = float(relative_power[band_name][channel_index])
                clip_records.append({
                    'patient_id': str(row['patient_id']),
                    'clip_id': str(row['clip_id']),
                    'channel_name': channel_name,
                    'band': band_name,
                    'sfreq_hz': sfreq,
                    'label': int(row['label']),
                    'model_score': score,
                    'relative_band_power': value,
                    'positive_score_weighted_power': value * positive_weight,
                    'signed_score_weighted_power': value * signed_weight,
                })
    frame = pd.DataFrame(clip_records)
    if frame.empty:
        return [], []
    patient_summary = (
        frame.groupby(['patient_id', 'channel_name', 'band'], as_index=False)
        .agg(
            clip_count=('clip_id', 'nunique'),
            mean_model_score=('model_score', 'mean'),
            mean_relative_band_power=('relative_band_power', 'mean'),
            mean_positive_score_weighted_power=(
                'positive_score_weighted_power',
                'mean',
            ),
            mean_signed_score_weighted_power=(
                'signed_score_weighted_power',
                'mean',
            ),
        )
    )
    return frame.to_dict('records'), patient_summary.to_dict('records')


def export(args: argparse.Namespace) -> Path:
    run_root = Path(args.run_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions, checkpoint = _validate_run(run_root)
    manifest = _load_test_manifest(target_root)
    merged = predictions.merge(
        manifest[['clip_id', 'patient_id', 'relative_path']],
        on=['clip_id', 'patient_id'],
        how='inner',
        validate='one_to_one',
    )
    if merged.empty:
        raise ValueError(
            f'No prediction rows matched target test manifest: run={run_root}, target={target_root}'
        )
    merged = _normalize_merged_relative_path(merged)
    bands = _parse_bands(args.bands)
    selected_patients = _select_patients(
        merged,
        patient_count=args.patient_count,
        selection_seed=args.selection_seed,
    )
    clip_records, patient_records = _iter_rows(
        merged,
        target_root,
        selected_patients,
        clips_per_patient=args.clips_per_patient,
        bands=bands,
        seed=args.selection_seed,
    )
    if not clip_records:
        raise ValueError('No spectral evidence rows were exported')
    clip_frame = pd.DataFrame(clip_records)
    patient_frame = pd.DataFrame(patient_records)
    clip_frame.to_csv(output_root / 'spectral_evidence_by_clip.csv', index=False)
    patient_frame.to_csv(output_root / 'spectral_evidence_by_patient.csv', index=False)
    metadata = {
        'status': 'complete',
        'evidence_type': 'prediction_score_weighted_signal_spectral_evidence',
        'attribution_warning': (
            'This is output-conditioned spectral evidence, not gradient attribution. '
            'It is deterministic and uses saved model scores without retraining.'
        ),
        'run_root': str(run_root),
        'target_root': str(target_root),
        'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': _sha256(checkpoint),
        'checkpoint_size_bytes': checkpoint.stat().st_size,
        'patient_selection_seed': int(args.selection_seed),
        'selected_patients': selected_patients,
        'clips_per_patient': int(args.clips_per_patient),
        'bands_hz': {name: list(values) for name, values in bands.items()},
        'matched_test_clips': int(len(merged)),
        'exported_test_clips': int(clip_frame['clip_id'].nunique()),
        'exported_channels': int(clip_frame['channel_name'].nunique()),
        'exported_rows': int(len(clip_frame)),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
    }
    (output_root / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--target-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--patient-count', type=int, default=3)
    parser.add_argument('--clips-per-patient', type=int, default=128)
    parser.add_argument('--selection-seed', type=int, default=1)
    parser.add_argument('--bands', default='')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = export(args)
    print(f'SPECTRAL_EXPORT_COMPLETED output={output}')


if __name__ == '__main__':
    main()
