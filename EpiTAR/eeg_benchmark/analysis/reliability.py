from __future__ import annotations

# Computes probability quality and reliability analyses.
import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm.auto import tqdm


DEFAULT_MODELS = (
    'BIOT',
    'CBraMod',
    'EEGPT',
    'LaBraM',
    'STEEGFormer',
    'EEGNet',
    'EvoBrain',
    'SVM',
    'BENDR',
    'CST',
)
DEFAULT_BUDGETS = (0.0, 25.0, 50.0, 75.0, 100.0)


@dataclass(frozen=True)
class RunRecord:
    track: str
    mission: str
    task: str
    direction: str
    model: str
    budget_percent: float
    window_seconds: float
    seed: int
    run_root: Path
    metrics_path: Path
    predictions_path: Path
    protocol_path: Path
    summary_path: Path | None


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def format_budget(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f'{value:g}'


def score_column(frame: pd.DataFrame) -> pd.Series:
    if 'score' in frame.columns:
        return pd.to_numeric(frame['score'], errors='coerce')
    if 'probability' in frame.columns:
        return pd.to_numeric(frame['probability'], errors='coerce')
    if 'pred_prob' in frame.columns:
        return pd.to_numeric(frame['pred_prob'], errors='coerce')
    raise ValueError('prediction file lacks score probability column')


def labels_and_scores(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if 'label' not in frame.columns:
        raise ValueError('prediction file lacks label column')
    labels = pd.to_numeric(frame['label'], errors='coerce').to_numpy(dtype=float)
    scores = score_column(frame).to_numpy(dtype=float)
    mask = np.isfinite(labels) & np.isfinite(scores)
    labels = labels[mask].astype(np.int64)
    scores = np.clip(scores[mask].astype(np.float64), 1e-7, 1.0 - 1e-7)
    if labels.size == 0:
        raise ValueError('prediction file has no finite labels and scores')
    return labels, scores


def expected_calibration_error(
    labels: np.ndarray,
    scores: np.ndarray,
    bins: int,
    strategy: str,
) -> float:
    confidence = np.maximum(scores, 1.0 - scores)
    correct = ((scores >= 0.5).astype(np.int64) == labels).astype(np.float64)
    if strategy == 'equal_width':
        edges = np.linspace(0.5, 1.0, int(bins) + 1)
    elif strategy == 'equal_mass':
        quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
        edges = np.quantile(confidence, quantiles)
        edges[0] = 0.5
        edges[-1] = 1.0
        edges = np.unique(edges)
        if edges.size < 2:
            return 0.0
    else:
        raise ValueError(f'Unsupported ECE strategy: {strategy}')
    total = float(labels.size)
    ece = 0.0
    for index in range(edges.size - 1):
        lower = edges[index]
        upper = edges[index + 1]
        if index == edges.size - 2:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += float(mask.sum()) / total * abs(accuracy - mean_confidence)
    return float(ece)


def brier_score(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((scores - labels) ** 2))


def negative_log_likelihood(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)))


def risk_coverage_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    steps: int = 101,
) -> pd.DataFrame:
    uncertainty = 1.0 - np.maximum(scores, 1.0 - scores)
    order = np.argsort(uncertainty, kind='mergesort')
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    rows: list[dict[str, float]] = []
    for coverage in np.linspace(0.01, 1.0, int(steps)):
        count = max(1, int(math.ceil(labels.size * float(coverage))))
        accepted_labels = sorted_labels[:count]
        accepted_scores = sorted_scores[:count]
        predictions = (accepted_scores >= 0.5).astype(np.int64)
        if np.unique(accepted_labels).size == 2:
            risk = 1.0 - float(balanced_accuracy_score(accepted_labels, predictions))
        else:
            risk = 1.0 - float((predictions == accepted_labels).mean())
        rows.append({'coverage': float(count / labels.size), 'risk': float(risk)})
    return pd.DataFrame(rows).drop_duplicates('coverage')


def area_under_curve(frame: pd.DataFrame, x: str, y: str) -> float:
    if frame.empty:
        return float('nan')
    ordered = frame.sort_values(x)
    return trapezoid_area(
        ordered[y].to_numpy(dtype=float),
        ordered[x].to_numpy(dtype=float),
    )


def trapezoid_area(y_values: np.ndarray, x_values: np.ndarray) -> float:
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y_values, x_values))
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    if x_array.size != y_array.size:
        raise ValueError('x and y arrays must have equal length')
    if x_array.size < 2:
        return 0.0
    return float(np.sum((x_array[1:] - x_array[:-1]) * (y_array[1:] + y_array[:-1]) * 0.5))


def reliability_bins(labels: np.ndarray, scores: np.ndarray, bins: int) -> pd.DataFrame:
    confidence = np.maximum(scores, 1.0 - scores)
    correct = ((scores >= 0.5).astype(np.int64) == labels).astype(np.float64)
    edges = np.linspace(0.5, 1.0, int(bins) + 1)
    rows: list[dict[str, float | int]] = []
    for index in range(int(bins)):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (confidence >= lower) & (confidence <= upper) if index == int(bins) - 1 else (confidence >= lower) & (confidence < upper)
        rows.append({
            'bin_index': index,
            'confidence_low': float(lower),
            'confidence_high': float(upper),
            'sample_count': int(mask.sum()),
            'mean_confidence': float(confidence[mask].mean()) if mask.any() else float('nan'),
            'empirical_accuracy': float(correct[mask].mean()) if mask.any() else float('nan'),
        })
    return pd.DataFrame(rows)


def discover_runs(
    roots: Iterable[tuple[str, Path]],
    models: set[str],
    tasks: set[str],
    directions: set[str] | None,
    windows: set[float],
    budgets: set[float],
    seeds: set[int],
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for track, result_root in roots:
        mission_name = 'dataset_transfer' if track == 'cross_dataset' else 'eeg_ieeg_transfer'
        mission_root = result_root / mission_name
        if not mission_root.is_dir():
            continue
        for metrics_path in sorted(mission_root.glob('*/*/*/*/seed_*/metrics.json')):
            run_root = metrics_path.parent
            protocol_path = run_root / 'protocol.json'
            predictions_path = run_root / 'predictions.csv'
            if not protocol_path.is_file() or not predictions_path.is_file():
                continue
            protocol = read_json(protocol_path)
            model = str(protocol.get('model', ''))
            task = str(protocol.get('task', ''))
            source = str(protocol.get('source_dataset', ''))
            target = str(protocol.get('target_dataset', ''))
            direction = f'{source}:{target}'
            window_seconds = float(protocol.get('window_seconds', float('nan')))
            budget_percent = float(protocol.get('budget_percent', float('nan')))
            seed = int(protocol.get('seed', -1))
            if models and model not in models:
                continue
            if tasks and task not in tasks:
                continue
            if directions is not None and direction not in directions:
                continue
            if windows and window_seconds not in windows:
                continue
            if budgets and budget_percent not in budgets:
                continue
            if seeds and seed not in seeds:
                continue
            records.append(
                RunRecord(
                    track=track,
                    mission=mission_name,
                    task=task,
                    direction=direction,
                    model=model,
                    budget_percent=budget_percent,
                    window_seconds=window_seconds,
                    seed=seed,
                    run_root=run_root,
                    metrics_path=metrics_path,
                    predictions_path=predictions_path,
                    protocol_path=protocol_path,
                    summary_path=run_root / 'run_summary.json'
                    if (run_root / 'run_summary.json').is_file()
                    else None,
                )
            )
    return records


def run_reliability(record: RunRecord, args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(record.predictions_path)
    labels, scores = labels_and_scores(predictions)
    predictions = predictions.copy()
    predictions['score'] = score_column(predictions)
    risk_coverage = risk_coverage_curve(labels, scores)
    bins = reliability_bins(labels, scores, args.ece_bins)
    try:
        auroc = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float('nan')
    except Exception:
        auroc = float('nan')
    row = {
        'track': record.track,
        'mission': record.mission,
        'task': record.task,
        'direction': record.direction,
        'model': record.model,
        'budget_percent': record.budget_percent,
        'window_seconds': record.window_seconds,
        'seed': record.seed,
        'run_root': str(record.run_root),
        'sample_count': int(labels.size),
        'patient_count': int(predictions['patient_id'].astype(str).nunique()) if 'patient_id' in predictions.columns else 0,
        'auroc': auroc,
        'ece': expected_calibration_error(labels, scores, args.ece_bins, 'equal_width'),
        'ece_equal_mass': expected_calibration_error(labels, scores, args.ece_bins, 'equal_mass'),
        'brier': brier_score(labels, scores),
        'nll': negative_log_likelihood(labels, scores),
        'aurc': area_under_curve(risk_coverage, 'coverage', 'risk'),
    }
    return row, bins, risk_coverage


def budget_recovery(reliability: pd.DataFrame) -> pd.DataFrame:
    if reliability.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ['track', 'mission', 'task', 'direction', 'model', 'window_seconds', 'seed']
    for keys, group in reliability.groupby(group_columns, sort=True, dropna=False):
        group = group.sort_values('budget_percent')
        row = dict(zip(group_columns, keys))
        budgets = group['budget_percent'].to_numpy(dtype=float)
        for metric in ('auroc', 'ece', 'brier', 'nll', 'aurc'):
            values = group[metric].to_numpy(dtype=float)
            if budgets.size >= 2 and np.isfinite(values).any():
                row[f'aubc_{metric}'] = float(trapezoid_area(values, budgets) / 100.0)
            else:
                row[f'aubc_{metric}'] = float('nan')
            lookup = {float(budget): float(value) for budget, value in zip(budgets, values)}
            zero = lookup.get(0.0, float('nan'))
            full = lookup.get(100.0, float('nan'))
            for budget in (25.0, 50.0, 75.0):
                current = lookup.get(budget, float('nan'))
                denominator = zero - full
                if metric == 'auroc':
                    denominator = full - zero
                    ratio = (current - zero) / denominator if denominator and math.isfinite(denominator) else float('nan')
                else:
                    ratio = (zero - current) / denominator if denominator and math.isfinite(denominator) else float('nan')
                row[f'recovery_{metric}_{format_budget(budget)}'] = float(ratio)
        rows.append(row)
    return pd.DataFrame(rows)


def save_reliability_diagram(path: Path, bins: pd.DataFrame, title: str) -> None:
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    axis.plot([0.5, 1.0], [0.5, 1.0], color='#666666', linewidth=1.0, linestyle=':')
    finite = bins.dropna(subset=['mean_confidence', 'empirical_accuracy'])
    axis.plot(finite['mean_confidence'], finite['empirical_accuracy'], marker='o', color='#0072B2', linewidth=1.5)
    axis.set_xlim(0.5, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel('Predicted confidence')
    axis.set_ylabel('Empirical accuracy')
    axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format='svg', bbox_inches='tight')
    plt.close(figure)


def save_risk_coverage(path: Path, frame: pd.DataFrame, title: str) -> None:
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    axis.plot(frame['coverage'], frame['risk'], color='#D55E00', linewidth=1.5)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, max(1.0, float(frame['risk'].max()) if not frame.empty else 1.0))
    axis.set_xlabel('Coverage')
    axis.set_ylabel('Risk')
    axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format='svg', bbox_inches='tight')
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / 'epishift_reliability.log'
    roots = [
        ('cross_dataset', args.cross_dataset_result_root),
        ('cross_modal', args.cross_modal_result_root),
    ]
    directions = None if args.directions == ['all'] else set(args.directions)
    records = discover_runs(
        roots,
        models=set(args.models),
        tasks=set(args.tasks),
        directions=directions,
        windows=set(float(value) for value in args.window_seconds),
        budgets=set(float(value) for value in args.budget_percent),
        seeds=set(int(value) for value in args.seeds),
    )
    rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    with log_path.open('w', encoding='utf-8', buffering=1) as log:
        log.write('EpiShift reliability audit started\n')
        log.write(f'run_count={len(records)}\n')
        iterator = tqdm(records, desc='EpiShift audit', unit='run', colour='green')
        for record in iterator:
            run_id = (
                f'{record.track}/{record.task}/{record.direction}/'
                f'{record.model}/{format_budget(record.budget_percent)}/seed_{record.seed}'
            )
            log.write(f'RUN {run_id}\n')
            try:
                row, bins, risk = run_reliability(record, args)
                rows.append(row)
                detail_root = (
                    output_root / 'per_run' / record.track / record.task
                    / record.direction.replace(':', '_to_') / record.model
                    / f'{format_budget(record.budget_percent)}_budget'
                    / f'seed_{record.seed}'
                )
                detail_root.mkdir(parents=True, exist_ok=True)
                bins.to_csv(detail_root / 'reliability_bins.csv', index=False)
                risk.to_csv(detail_root / 'risk_coverage.csv', index=False)
                if args.save_figures:
                    save_reliability_diagram(
                        detail_root / 'reliability_diagram.svg',
                        bins,
                        f'{record.model} {record.task} {format_budget(record.budget_percent)}%',
                    )
                    save_risk_coverage(
                        detail_root / 'risk_coverage.svg',
                        risk,
                        f'{record.model} {record.task} {format_budget(record.budget_percent)}%',
                    )
                log.write(f'OK {run_id}\n')
            except Exception as exc:
                failure_rows.append({
                    'track': record.track,
                    'task': record.task,
                    'direction': record.direction,
                    'model': record.model,
                    'budget_percent': record.budget_percent,
                    'window_seconds': record.window_seconds,
                    'seed': record.seed,
                    'run_root': str(record.run_root),
                    'status': 'failed',
                    'error': f'{type(exc).__name__}: {exc}',
                })
                log.write(f'FAILED {run_id} {type(exc).__name__}: {exc}\n')
                if args.stop_on_error:
                    raise
    reliability = pd.DataFrame(rows)
    failures = pd.DataFrame(failure_rows)
    recovery = budget_recovery(reliability)
    reliability.to_csv(output_root / 'reliability_results.csv', index=False)
    failures.to_csv(output_root / 'failed_runs.csv', index=False)
    recovery.to_csv(output_root / 'budget_reliability_recovery.csv', index=False)
    summary = {
        'status': 'complete',
        'task_family': 'epishift_reliability_audit',
        'run_count': len(records),
        'completed_reliability_rows': int(len(reliability)),
        'failed_rows': int(len(failures)),
        'output_root': str(output_root),
        'experiments': [
            'reliability_ece_brier_nll',
            'risk_coverage_aurc',
            'budget_reliability_recovery',
        ],
        'scope': 'read_only_reliability_metrics_without_leakage_audit_or_clinical_alarm',
    }
    (output_root / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding='utf-8',
    )
    print('|Output|Path|')
    print('|:--|:--|')
    for name in (
        'reliability_results.csv',
        'budget_reliability_recovery.csv',
        'failed_runs.csv',
        'summary.json',
    ):
        print(f'|{name}|{output_root / name}|')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='EpiShift clinical alarm and reliability audit')
    project_root = Path(
        os.environ.get(
            'BENCHMARK_ROOT',
            str(Path(__file__).resolve().parents[2]),
        )
    )
    data_root = Path(os.environ.get('DATA_ROOT', str(project_root / 'data')))
    parser.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    parser.add_argument('--tasks', nargs='+', choices=['detection', 'prediction'], default=['detection', 'prediction'])
    parser.add_argument('--directions', nargs='+', default=['all'])
    parser.add_argument('--window-seconds', nargs='+', type=float, default=[12.0])
    parser.add_argument('--budget-percent', nargs='+', type=float, default=list(DEFAULT_BUDGETS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[1])
    parser.add_argument('--cross-dataset-result-root', type=Path, default=data_root / 'results/cross_dataset')
    parser.add_argument('--cross-modal-result-root', type=Path, default=data_root / 'results/cross_modal')
    parser.add_argument('--output-dir', type=Path, default=data_root / 'results/reliability')
    parser.add_argument('--ece-bins', type=int, default=15)
    parser.add_argument('--save-figures', type=int, choices=[0, 1], default=1)
    parser.add_argument('--stop-on-error', type=int, choices=[0, 1], default=0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
