from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


DISPLAY_NAMES = {
    'chbmit': 'CHB-MIT',
    'chbmit_historical': 'CHBMIT-Historical',
    'chbmit_future': 'CHBMIT-Future',
}
BUDGET_GRID = (25.0, 50.0, 75.0, 100.0)
FUTURE_EVENT_FRACTION = 0.30
DEV_FRACTION_OF_EARLIER = 0.20
ADAPT_FRACTION_OF_EARLIER = 0.30


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f'{value:g}'


def event_key(row: pd.Series) -> str:
    event_id = str(row.get('event_id', '')).strip()
    if not event_id:
        raise ValueError(f'Positive clip lacks event_id: {row.get("clip_id")}')
    return f'{row.patient_id}::{event_id}'


def allocate_event_blocks(event_count: int) -> tuple[int, int, int, int]:
    if event_count < 4:
        raise ValueError('Cross-time requires at least four positive events')
    future = max(1, round(event_count * FUTURE_EVENT_FRACTION))
    if event_count - future < 3:
        future = event_count - 3
    earlier = event_count - future
    dev = max(1, round(earlier * DEV_FRACTION_OF_EARLIER))
    adapt = max(1, round(earlier * ADAPT_FRACTION_OF_EARLIER))
    historical = earlier - dev - adapt
    adjustable = {'adapt': adapt, 'dev': dev}
    while historical < 1:
        key = max(adjustable, key=lambda item: adjustable[item])
        if adjustable[key] <= 1:
            raise ValueError('Cannot allocate temporal blocks')
        adjustable[key] -= 1
        historical += 1
    dev = adjustable['dev']
    adapt = adjustable['adapt']
    while historical > 1 and (dev + adapt + historical) < earlier:
        adapt += 1
        historical -= 1
    return historical, dev, adapt, future


def minimum_budget_percent(rank: int, total: int) -> float:
    for percent in BUDGET_GRID:
        if rank <= max(1, math.ceil(total * percent / 100.0)):
            return percent
    return 100.0


def ordered_positive_events(frame: pd.DataFrame) -> pd.DataFrame:
    positive = frame.loc[frame['label'].astype(int) == 1].copy()
    positive['_event_key'] = positive.apply(event_key, axis=1)
    start_column = (
        'timeline_clip_start_seconds'
        if 'timeline_clip_start_seconds' in positive.columns
        else 'clip_start_seconds'
    )
    events = (
        positive.groupby(['patient_id', '_event_key'], as_index=False)
        .agg(
            event_id=('event_id', 'first'),
            event_start_seconds=(start_column, 'min'),
            positive_clip_count=('clip_id', 'size'),
        )
        .sort_values(['patient_id', 'event_start_seconds', '_event_key'], kind='mergesort')
        .reset_index(drop=True)
    )
    return events


def clip_crosses_boundary(
    clip_start: float,
    clip_end: float,
    boundaries: tuple[float, float, float],
) -> bool:
    return any(clip_start < boundary < clip_end for boundary in boundaries)


def temporal_block_from_time(
    clip_start: float,
    boundaries: tuple[float, float, float],
) -> str:
    dev_start, adapt_start, future_start = boundaries
    if clip_start < dev_start:
        return 'historical'
    if clip_start < adapt_start:
        return 'dev'
    if clip_start < future_start:
        return 'adapt'
    return 'future'


def assign_temporal_blocks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    if frame.empty:
        raise ValueError('Manifest is empty')
    start_column = (
        'timeline_clip_start_seconds'
        if 'timeline_clip_start_seconds' in frame.columns
        else 'clip_start_seconds'
    )
    end_column = (
        'timeline_clip_end_seconds'
        if 'timeline_clip_end_seconds' in frame.columns
        else 'clip_end_seconds'
    )
    events = ordered_positive_events(frame)
    audit_rows = []
    block_by_event: dict[str, str] = {}
    budget_by_event: dict[str, float] = {}
    boundaries_by_patient: dict[str, tuple[float, float, float]] = {}
    for patient_id, patient_events in events.groupby('patient_id', sort=False):
        patient_events = patient_events.reset_index(drop=True)
        try:
            historical_n, dev_n, adapt_n, future_n = allocate_event_blocks(
                len(patient_events)
            )
        except ValueError:
            audit_rows.append({
                'patient_id': str(patient_id),
                'eligible': False,
                'event_count': int(len(patient_events)),
                'reason': 'fewer_than_four_positive_events',
            })
            continue
        limits = {
            'historical': historical_n,
            'dev': historical_n + dev_n,
            'adapt': historical_n + dev_n + adapt_n,
            'future': historical_n + dev_n + adapt_n + future_n,
        }
        adapt_events = []
        event_order_by_key: dict[str, int] = {}
        for index, event in patient_events.iterrows():
            ordinal = int(index) + 1
            event_key_value = str(event['_event_key'])
            event_order_by_key[event_key_value] = ordinal
            if ordinal <= limits['historical']:
                block = 'historical'
            elif ordinal <= limits['dev']:
                block = 'dev'
            elif ordinal <= limits['adapt']:
                block = 'adapt'
                adapt_events.append(event_key_value)
            else:
                block = 'future'
            block_by_event[event_key_value] = block
        for rank, event_key_value in enumerate(adapt_events, start=1):
            budget_by_event[event_key_value] = minimum_budget_percent(
                rank, len(adapt_events)
            )
        dev_start = float(patient_events.iloc[historical_n]['event_start_seconds'])
        adapt_start = float(
            patient_events.iloc[historical_n + dev_n]['event_start_seconds']
        )
        future_start = float(
            patient_events.iloc[historical_n + dev_n + adapt_n]['event_start_seconds']
        )
        boundaries_by_patient[str(patient_id)] = (dev_start, adapt_start, future_start)
        audit_rows.append({
            'patient_id': str(patient_id),
            'eligible': True,
            'event_count': int(len(patient_events)),
            'historical_events': int(historical_n),
            'dev_events': int(dev_n),
            'adapt_events': int(adapt_n),
            'future_events': int(future_n),
            'future_event_fraction_requested': float(FUTURE_EVENT_FRACTION),
            'future_event_fraction_actual': float(future_n / len(patient_events)),
            'dev_start_seconds': dev_start,
            'adapt_start_seconds': adapt_start,
            'future_start_seconds': future_start,
            'temporal_order_policy': 'latest_30pct_future_then_earlier_historical_dev_adapt',
        })
    eligible_patients = {
        str(row['patient_id']) for row in audit_rows if bool(row.get('eligible'))
    }
    frame = frame.loc[frame['patient_id'].astype(str).isin(eligible_patients)].copy()
    if frame.empty:
        raise ValueError('No patient is eligible for cross-time splitting')

    temporal_blocks = []
    temporal_event_orders = []
    temporal_budget_percents = []
    excluded_cross_boundary = []
    order_by_event = {
        str(event['_event_key']): int(index) + 1
        for index, event in events.iterrows()
    }
    for row in frame.itertuples(index=False):
        patient_id = str(row.patient_id)
        boundaries = boundaries_by_patient[patient_id]
        clip_start = float(getattr(row, start_column))
        clip_end = float(getattr(row, end_column))
        crosses_boundary = clip_crosses_boundary(clip_start, clip_end, boundaries)
        label = int(row.label)
        if crosses_boundary:
            excluded_cross_boundary.append({
                'patient_id': patient_id,
                'clip_id': str(row.clip_id),
                'label': label,
                'event_id': str(getattr(row, 'event_id', '')),
                'clip_start_seconds': clip_start,
                'clip_end_seconds': clip_end,
                'dev_start_seconds': boundaries[0],
                'adapt_start_seconds': boundaries[1],
                'future_start_seconds': boundaries[2],
                'reason': 'clip_crosses_temporal_boundary',
            })
            temporal_blocks.append('excluded_cross_boundary')
            temporal_budget_percents.append(0.0)
            temporal_event_orders.append(0)
            continue
        if label == 1:
            key = f'{patient_id}::{str(row.event_id).strip()}'
            block = block_by_event[key]
            temporal_blocks.append(block)
            temporal_budget_percents.append(float(budget_by_event.get(key, 0.0)))
            temporal_event_orders.append(order_by_event.get(key, 0))
            continue
        block = temporal_block_from_time(clip_start, boundaries)
        temporal_blocks.append(block)
        temporal_budget_percents.append(0.0)
        temporal_event_orders.append(0)
    frame['temporal_block'] = temporal_blocks
    frame['temporal_budget_percent'] = temporal_budget_percents
    frame['temporal_event_order'] = temporal_event_orders
    frame['temporal_patient_eligible'] = True
    excluded = frame.loc[
        frame['temporal_block'].astype(str) == 'excluded_cross_boundary'
    ].copy()
    frame = frame.loc[
        frame['temporal_block'].astype(str) != 'excluded_cross_boundary'
    ].copy()
    block_counts = (
        frame.groupby(['patient_id', 'temporal_block', 'label'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in (0, 1):
        if column not in block_counts.columns:
            block_counts[column] = 0
    block_counts = block_counts.rename(
        columns={0: 'negative_clip_count', 1: 'positive_clip_count'}
    )
    audit = pd.DataFrame(audit_rows)
    if not block_counts.empty:
        evaluable = (
            block_counts
            .assign(
                has_both_classes=lambda item: (
                    (item['negative_clip_count'].astype(int) > 0)
                    & (item['positive_clip_count'].astype(int) > 0)
                )
            )
        )
        block_summary = (
            evaluable.pivot_table(
                index='patient_id',
                columns='temporal_block',
                values='has_both_classes',
                aggfunc='first',
                fill_value=False,
            )
            .reset_index()
        )
        for block in ('historical', 'dev', 'adapt', 'future'):
            if block not in block_summary.columns:
                block_summary[block] = False
        block_summary = block_summary.rename(columns={
            'historical': 'historical_has_both_classes',
            'dev': 'dev_has_both_classes',
            'adapt': 'adapt_has_both_classes',
            'future': 'future_has_both_classes',
        })
        audit = audit.merge(block_summary, on='patient_id', how='left')
    excluded_counts = (
        pd.DataFrame(excluded_cross_boundary)
        .groupby('patient_id')
        .size()
        .rename('excluded_cross_boundary_clip_count')
        .reset_index()
        if excluded_cross_boundary
        else pd.DataFrame(columns=['patient_id', 'excluded_cross_boundary_clip_count'])
    )
    audit = audit.merge(excluded_counts, on='patient_id', how='left')
    audit['excluded_cross_boundary_clip_count'] = (
        audit['excluded_cross_boundary_clip_count'].fillna(0).astype(int)
    )
    frame.attrs['excluded_cross_boundary'] = excluded_cross_boundary
    frame.attrs['block_counts'] = block_counts.to_dict('records')
    return frame, audit


def prepare_manifest(frame: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    prepared = frame.copy()
    prepared['dataset'] = DISPLAY_NAMES[dataset_key]
    if dataset_key == 'chbmit_historical':
        split_map = {
            'historical': 'train',
            'dev': 'dev',
            'adapt': 'unused',
            'future': 'test',
        }
    else:
        split_map = {
            'historical': 'unused',
            'dev': 'dev',
            'adapt': 'train',
            'future': 'test',
        }
    prepared['split'] = prepared['temporal_block'].map(split_map)
    prepared = prepared.loc[prepared['split'].isin({'train', 'dev', 'test'})].copy()
    for split in ('train', 'dev', 'test'):
        labels = set(prepared.loc[prepared['split'] == split, 'label'].astype(int))
        if labels != {0, 1}:
            raise ValueError(
                f'{DISPLAY_NAMES[dataset_key]} {split} lacks both classes: {labels}'
            )
    return prepared


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == 'copy':
        shutil.copy2(source, destination)
        return
    if mode == 'symlink':
        os.symlink(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_clips(
    source_task_root: Path,
    target_task_root: Path,
    manifest: pd.DataFrame,
    mode: str,
) -> None:
    for relative_path in sorted(set(manifest['relative_path'].astype(str))):
        source = source_task_root / relative_path
        destination = target_task_root / relative_path
        if not source.exists():
            raise FileNotFoundError(f'Missing source clip: {source}')
        link_or_copy(source, destination, mode)


def write_contract(
    source_task_root: Path,
    target_task_root: Path,
    dataset_key: str,
    task: str,
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    source_contract = json.loads(
        (source_task_root / 'dataset_contract.json').read_text(encoding='utf-8')
    )
    payload = dict(source_contract)
    payload.update({
        'dataset': DISPLAY_NAMES[dataset_key],
        'source_dataset': 'CHB-MIT',
        'task': task,
        'cross_time_track': 'chbmit_patient_chronological_generalization',
        'temporal_blocks': ['historical', 'dev', 'adapt', 'future'],
        'source_role': dataset_key,
        'budget_policy': 'nested_chronological_adapt_prefix',
        'test_policy': 'fixed_latest_30pct_future_block',
        'future_event_fraction': FUTURE_EVENT_FRACTION,
        'earlier_dev_fraction': DEV_FRACTION_OF_EARLIER,
        'earlier_adapt_fraction': ADAPT_FRACTION_OF_EARLIER,
        'cross_boundary_clip_policy': 'exclude_any_clip_crossing_dev_adapt_or_future_boundary',
        'patient_count': int(manifest['patient_id'].nunique()),
        'clip_count': int(len(manifest)),
        'eligible_patient_count': int(audit['eligible'].astype(bool).sum()),
    })
    fingerprint_payload = {
        'source_fingerprint': source_contract.get('fingerprint'),
        'dataset': dataset_key,
        'task': task,
        'clip_ids': sorted(manifest['clip_id'].astype(str).tolist()),
        'splits': manifest[['clip_id', 'split', 'temporal_block']].to_dict('records'),
    }
    payload['fingerprint'] = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(',', ':')
        ).encode('utf-8')
    ).hexdigest()[:16]
    (target_task_root / 'dataset_contract.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (target_task_root / 'status.json').write_text(
        json.dumps({
            'status': 'complete',
            'ready_for_training': True,
            'clips': int(len(manifest)),
            'cross_time_track': True,
        }, indent=2),
        encoding='utf-8',
    )


def build_task_cache(
    source_preprocess_root: Path,
    output_root: Path,
    window_seconds: float,
    task: str,
    overwrite: bool,
    link_mode: str,
) -> None:
    window_name = f'{format_number(window_seconds)}s'
    source_task_root = source_preprocess_root / window_name / task / DISPLAY_NAMES['chbmit']
    if not source_task_root.exists():
        raise FileNotFoundError(f'Base CHB-MIT cache does not exist: {source_task_root}')
    manifest = pd.read_csv(
        source_task_root / 'manifest.csv',
        dtype={'patient_id': str, 'clip_id': str, 'event_id': str},
    )
    assigned, audit = assign_temporal_blocks(manifest)
    audit_root = output_root / window_name / task / 'temporal_audit'
    audit_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_root / 'patient_temporal_blocks.csv', index=False)
    assigned.to_csv(audit_root / 'clip_temporal_blocks.csv', index=False)
    pd.DataFrame(
        assigned.attrs.get('excluded_cross_boundary', [])
    ).to_csv(audit_root / 'excluded_cross_boundary_clips.csv', index=False)
    pd.DataFrame(
        assigned.attrs.get('block_counts', [])
    ).to_csv(audit_root / 'temporal_block_label_counts.csv', index=False)
    for dataset_key in ('chbmit_historical', 'chbmit_future'):
        target_task_root = output_root / window_name / task / DISPLAY_NAMES[dataset_key]
        if target_task_root.exists():
            if not overwrite:
                raise FileExistsError(
                    f'Output cache exists, use --overwrite: {target_task_root}'
                )
            shutil.rmtree(target_task_root)
        target_task_root.mkdir(parents=True, exist_ok=True)
        prepared = prepare_manifest(assigned, dataset_key)
        materialize_clips(source_task_root, target_task_root, prepared, link_mode)
        prepared.to_csv(target_task_root / 'manifest.csv', index=False)
        write_contract(
            source_task_root,
            target_task_root,
            dataset_key,
            task,
            prepared,
            audit,
        )
        (target_task_root / 'temporal_protocol.json').write_text(
            json.dumps({
                'track': 'chbmit_cross_time_temporal_generalization',
                'task': task,
                'dataset_key': dataset_key,
                'base_cache': str(source_task_root),
                'link_mode': link_mode,
                'split_before_training': True,
                'future_event_fraction': FUTURE_EVENT_FRACTION,
                'earlier_dev_fraction': DEV_FRACTION_OF_EARLIER,
                'earlier_adapt_fraction': ADAPT_FRACTION_OF_EARLIER,
                'budget_policy': 'nested chronological Adapt prefix',
                'future_test_fixed': True,
                'cross_boundary_clip_policy': 'exclude',
            }, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build CHB-MIT cross-time temporal generalization cache'
    )
    parser.add_argument(
        '--source-preprocess-root',
        type=Path,
        default=Path('data/preprocessed/cross_dataset'),
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('data/preprocessed/cross_time'),
    )
    parser.add_argument('--window-seconds', type=float, nargs='+', required=True)
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=['detection', 'prediction'],
        default=['detection', 'prediction'],
    )
    parser.add_argument(
        '--link-mode',
        choices=['hardlink', 'symlink', 'copy'],
        default='hardlink',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for window_seconds in args.window_seconds:
        for task in args.tasks:
            build_task_cache(
                args.source_preprocess_root,
                args.output_root,
                float(window_seconds),
                task,
                bool(args.overwrite),
                args.link_mode,
            )
            print(
                f'CROSSTIME_CACHE_READY task={task} window={format_number(float(window_seconds))}s '
                f'output={args.output_root}'
            )


if __name__ == '__main__':
    main()
