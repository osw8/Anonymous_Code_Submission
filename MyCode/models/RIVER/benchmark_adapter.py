from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, MODEL_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from model import EpiTransOp, EpiTransOpConfig
from trainer import install_river_training_runtime
from eeg_benchmark.engine import ChannelUnionContract
from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer, save_json


def build_config(args: argparse.Namespace) -> EpiTransOpConfig:
    model_task = 'detection' if args.task == 'localization' else args.task
    return EpiTransOpConfig(
        sampling_frequency=256,
        patch_points=256,
        descriptor_dim=16,
        descriptor_hidden_dim=args.river_descriptor_hidden_dim,
        descriptor_window_seconds=args.river_descriptor_window_seconds,
        descriptor_slow_seconds=args.river_descriptor_slow_seconds,
        descriptor_channel_chunk_size=args.river_descriptor_channel_chunk_size,
        latent_electrodes=args.river_latent_electrodes,
        embed_dim=args.river_embed_dim,
        depth=args.river_depth,
        heads=args.river_heads,
        ff_dim=args.river_ff_dim,
        fourier_modes=args.river_fourier_modes,
        fourier_rank=args.river_fourier_rank,
        local_kernel_size=args.river_local_kernel_size,
        slot_temperature=args.river_slot_temperature,
        sinkhorn_iterations=args.river_sinkhorn_iterations,
        transport_epsilon=args.river_transport_epsilon,
        electrode_mass_relaxation=args.river_electrode_mass_relaxation,
        electrode_mass_temperature=args.river_electrode_mass_temperature,
        pooling_temperature=args.river_pooling_temperature,
        evidence_smoothing_kernel=args.river_evidence_smoothing_kernel,
        dropout=args.river_dropout,
        adapter_mode=args.river_transport_mode,
        temporal_mixer=args.river_temporal_mixer,
        task=model_task,
        mass_aware_transport=bool(args.river_mass_aware_transport),
        residual_bottleneck_dim=args.river_residual_bottleneck_dim,
    )


def build_model(
    contract: ChannelUnionContract,
    config: EpiTransOpConfig,
) -> EpiTransOp:
    del contract
    return EpiTransOp(config)


def load_pretrained(
    model: EpiTransOp,
    path: Path,
    contract: ChannelUnionContract,
) -> dict[str, object]:
    del model, path, contract
    raise ValueError('EpiTransOp pretraining is disabled until its dataset is approved')


def configure_transfer(model: EpiTransOp, mode: str) -> None:
    del mode
    for parameter in model.parameters():
        parameter.requires_grad = True


def budget_modules(model: EpiTransOp) -> tuple[torch.nn.Module, torch.nn.Module]:
    return model.head, model.blocks[-1]


def forward_clip(
    model: EpiTransOp,
    eeg: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    return model(eeg, channel_mask)


def normalized_entropy(probability: np.ndarray) -> float:
    probability = probability.astype(np.float64, copy=False)
    probability = probability / max(float(probability.sum()), 1e-12)
    positive = probability[probability > 0.0]
    if probability.size <= 1 or positive.size == 0:
        return 0.0
    return float(
        -(positive * np.log(positive)).sum() / np.log(float(probability.size))
    )


def drop_valid_recording_sites(
    channel_mask: torch.Tensor,
    drop_ratio: float,
) -> torch.Tensor:
    if not 0.0 <= drop_ratio < 1.0:
        raise ValueError('Montage channel drop ratio must be in [0, 1)')
    mask = channel_mask.to(dtype=torch.bool)
    perturbed = mask.clone()
    if drop_ratio == 0.0:
        return perturbed
    valid_count = mask.sum(dim=1)
    drop_count = torch.round(valid_count.float() * drop_ratio).long().clamp_min(1)
    drop_count = torch.minimum(drop_count, (valid_count - 1).clamp_min(0))
    random_score = torch.rand(mask.shape, device=mask.device)
    random_score = random_score.masked_fill(~mask, 2.0)
    rank = random_score.argsort(dim=1).argsort(dim=1)
    dropped = rank < drop_count[:, None]
    return mask & ~dropped


def _channel_names_for_sample(
    batch: dict[str, object],
    index: int,
    channel_count: int,
) -> list[str]:
    values = batch.get('channel_names')
    if isinstance(values, (list, tuple)) and len(values) > index:
        sample_names = values[index]
        if isinstance(sample_names, (list, tuple, np.ndarray)):
            selected = [str(value) for value in sample_names]
            if len(selected) == channel_count:
                return selected
    return [f'channel_{channel_index}' for channel_index in range(channel_count)]


def _stable_random_ranking(clip_id: str, valid_indices: np.ndarray) -> np.ndarray:
    digest = hashlib.sha256(f'EpiTransOp:{clip_id}'.encode('utf-8')).hexdigest()
    generator = np.random.default_rng(int(digest[:16], 16))
    return generator.permutation(valid_indices)


def collect_transport_diagnostics(
    model: EpiTransOp,
    loader,
    device: torch.device,
    output_dir: Path,
    logger,
) -> dict[str, object]:
    model.to(device).eval()
    rows = []
    site_rows = []
    intervention_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / 'transport_details.jsonl.gz'
    with torch.no_grad(), gzip.open(detail_path, 'wt', encoding='utf-8') as detail_file:
        for batch in tqdm(
            loader,
            desc='transport diagnostics',
            colour='green',
            dynamic_ncols=True,
        ):
            eeg = batch['eeg'].to(device, non_blocking=True)
            mask = batch['channel_mask'].to(device, non_blocking=True)
            logits, details = model(eeg, mask, return_details=True)
            base_probability = torch.sigmoid(logits).detach().cpu().numpy()
            assignment = details['transport'].detach().cpu().numpy()
            temporal_assignment = details['temporal_transport'].detach().cpu().numpy()
            column_mass = details['transport_column_mass'].detach().cpu().numpy()
            temporal_column_mass = details[
                'temporal_transport_column_mass'
            ].detach().cpu().numpy()
            prior_mass = details['electrode_prior_mass'].detach().cpu().numpy()
            temporal_prior_mass = details[
                'temporal_electrode_prior_mass'
            ].detach().cpu().numpy()
            cost = details['transport_cost'].detach().cpu().numpy()
            slot_features = details['transport_slot_features'].detach().cpu().numpy()
            valid_mask = batch['channel_mask'].cpu().numpy().astype(bool, copy=False)
            batch_rankings = []
            for index in range(assignment.shape[0]):
                valid = valid_mask[index]
                valid_indices = np.flatnonzero(valid)
                channel_names = _channel_names_for_sample(
                    batch, index, valid_mask.shape[1]
                )
                valid_names = [channel_names[value] for value in valid_indices]
                matrix = assignment[index][:, valid]
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                singular_probability = singular_values / max(
                    float(singular_values.sum()), 1e-12
                )
                positive = singular_probability[singular_probability > 0.0]
                effective_rank = float(
                    np.exp(-(positive * np.log(positive)).sum())
                )
                mass = column_mass[index, valid]
                prior = prior_mass[index, valid]
                latent_entropies = [
                    normalized_entropy(row) for row in matrix
                ]
                transport_temporal_variation = (
                    float(np.abs(np.diff(
                        temporal_assignment[index][:, :, valid], axis=0
                    )).mean())
                    if temporal_assignment.shape[1] > 1 else 0.0
                )
                mass_temporal_variation = (
                    float(np.abs(np.diff(
                        temporal_column_mass[index][:, valid], axis=0
                    )).mean())
                    if temporal_column_mass.shape[1] > 1 else 0.0
                )
                rows.append({
                    'clip_id': str(batch['clip_id'][index]),
                    'patient_id': str(batch['patient_id'][index]),
                    'dataset': str(batch['dataset'][index]),
                    'label': int(batch['label'][index]),
                    'valid_electrodes': int(valid.sum()),
                    'transport_effective_rank': effective_rank,
                    'electrode_mass_entropy': normalized_entropy(mass),
                    'maximum_electrode_mass': float(mass.max()),
                    'prior_transport_l1': float(np.abs(mass - prior).sum()),
                    'mean_latent_assignment_entropy': float(np.mean(latent_entropies)),
                    'mean_transport_cost': float(
                        (matrix * cost[index][:, valid]).sum()
                        / matrix.shape[0]
                    ),
                    'transport_temporal_variation': transport_temporal_variation,
                    'electrode_mass_temporal_variation': mass_temporal_variation,
                })
                ranking = {
                    'top': valid_indices[np.argsort(-prior, kind='stable')],
                    'bottom': valid_indices[np.argsort(prior, kind='stable')],
                    'random': _stable_random_ranking(
                        str(batch['clip_id'][index]), valid_indices
                    ),
                }
                batch_rankings.append(ranking)
                for rank, site_index in enumerate(ranking['top'], start=1):
                    site_rows.append({
                        'clip_id': str(batch['clip_id'][index]),
                        'patient_id': str(batch['patient_id'][index]),
                        'dataset': str(batch['dataset'][index]),
                        'label': int(batch['label'][index]),
                        'channel_index': int(site_index),
                        'channel_name': channel_names[int(site_index)],
                        'evidence_rank': rank,
                        'electrode_prior_mass': float(prior_mass[index, site_index]),
                        'transport_column_mass': float(column_mass[index, site_index]),
                    })
                detail_file.write(json.dumps({
                    'clip_id': str(batch['clip_id'][index]),
                    'patient_id': str(batch['patient_id'][index]),
                    'dataset': str(batch['dataset'][index]),
                    'label': int(batch['label'][index]),
                    'base_probability': float(base_probability[index]),
                    'valid_channel_indices': valid_indices.tolist(),
                    'valid_channel_names': valid_names,
                    'electrode_prior_mass': prior_mass[index, valid].tolist(),
                    'transport_column_mass': column_mass[index, valid].tolist(),
                    'transport_assignment': matrix.tolist(),
                    'temporal_electrode_prior_mass': (
                        temporal_prior_mass[index][:, valid].tolist()
                    ),
                    'temporal_transport_assignment': (
                        temporal_assignment[index][:, :, valid].tolist()
                    ),
                    'transport_slot_features': slot_features[index].tolist(),
                }, separators=(',', ':')) + '\n')

            for removal_count in (1, 2, 4):
                for strategy in ('top', 'bottom', 'random'):
                    intervention_mask = mask.clone()
                    removed_by_sample = []
                    for index, ranking in enumerate(batch_rankings):
                        maximum = max(int(valid_mask[index].sum()) - 1, 0)
                        effective_count = min(removal_count, maximum)
                        selected = ranking[strategy][:effective_count]
                        if effective_count:
                            selected_tensor = torch.as_tensor(
                                selected, device=device, dtype=torch.long
                            )
                            intervention_mask[index, selected_tensor] = False
                        removed_by_sample.append(selected)
                    perturbed_logits = model(eeg, intervention_mask)
                    perturbed_probability = torch.sigmoid(
                        perturbed_logits
                    ).detach().cpu().numpy()
                    for index, selected in enumerate(removed_by_sample):
                        channel_names = _channel_names_for_sample(
                            batch, index, valid_mask.shape[1]
                        )
                        intervention_rows.append({
                            'clip_id': str(batch['clip_id'][index]),
                            'patient_id': str(batch['patient_id'][index]),
                            'dataset': str(batch['dataset'][index]),
                            'label': int(batch['label'][index]),
                            'strategy': strategy,
                            'requested_removal_count': removal_count,
                            'effective_removal_count': len(selected),
                            'removed_channel_indices': json.dumps(
                                [int(value) for value in selected], separators=(',', ':')
                            ),
                            'removed_channel_names': json.dumps(
                                [channel_names[int(value)] for value in selected],
                                separators=(',', ':'),
                            ),
                            'base_probability': float(base_probability[index]),
                            'perturbed_probability': float(perturbed_probability[index]),
                            'probability_drop': float(
                                base_probability[index] - perturbed_probability[index]
                            ),
                        })
    if not rows:
        raise ValueError('Transport diagnostic loader is empty')
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / 'transport_diagnostics.csv', index=False)
    pd.DataFrame(site_rows).to_csv(
        output_dir / 'transport_site_masses.csv', index=False
    )
    intervention_frame = pd.DataFrame(intervention_rows)
    intervention_frame.to_csv(
        output_dir / 'causal_removal_interventions.csv', index=False
    )
    grouped = intervention_frame.groupby(
        ['strategy', 'requested_removal_count'], sort=True
    )['probability_drop']
    removal_summary = {
        f'{strategy}_k{int(removal_count)}': {
            'mean_probability_drop': float(values.mean()),
            'standard_deviation': float(values.std(ddof=0)),
            'sample_count': int(values.size),
        }
        for (strategy, removal_count), values in grouped
    }
    save_json(output_dir / 'causal_removal_summary.json', removal_summary)
    metric_columns = [
        'transport_effective_rank',
        'electrode_mass_entropy',
        'maximum_electrode_mass',
        'prior_transport_l1',
        'mean_latent_assignment_entropy',
        'mean_transport_cost',
        'transport_temporal_variation',
        'electrode_mass_temporal_variation',
    ]
    summary = {
        'status': 'complete',
        'clip_count': len(frame),
        'causal_removal': removal_summary,
        'metrics': {
            name: {
                'mean': float(frame[name].mean()),
                'standard_deviation': float(frame[name].std(ddof=0)),
                'minimum': float(frame[name].min()),
                'maximum': float(frame[name].max()),
            }
            for name in metric_columns
        },
    }
    save_json(output_dir / 'transport_diagnostics_summary.json', summary)
    logger.info(
        'transport_diagnostics_complete clips=%d effective_rank_mean=%.8f '
        'electrode_mass_entropy_mean=%.8f',
        len(frame),
        summary['metrics']['transport_effective_rank']['mean'],
        summary['metrics']['electrode_mass_entropy']['mean'],
    )
    return summary


def canonical_dataset_name(value: object) -> str:
    return ''.join(character for character in str(value).lower() if character.isalnum())


def build_training_objective(args: argparse.Namespace):
    source_key = canonical_dataset_name(args.source_dataset)
    target_key = canonical_dataset_name(args.target_dataset)

    def training_objective(
        model: EpiTransOp,
        batch: dict[str, object],
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
        labels: torch.Tensor,
        stage: str,
    ):
        if stage == 'target_linear_probe':
            logits, details = model(eeg, channel_mask, return_details=True)
            task_loss = F.binary_cross_entropy_with_logits(
                logits.reshape(-1), labels
            )
            zero = task_loss.detach() * 0.0
            return logits, task_loss, {
                'task_loss': task_loss.detach(),
                'original_task_loss': task_loss.detach(),
                'augmentation_task_loss': zero,
                'transfer_alignment_loss': zero,
            }

        datasets = batch.get('dataset')
        if not isinstance(datasets, (list, tuple)):
            raise ValueError('EpiTransOp auxiliary objective requires dataset metadata')
        dataset_keys = [canonical_dataset_name(value) for value in datasets]
        source_mask = torch.tensor(
            [value == source_key for value in dataset_keys],
            device=labels.device,
            dtype=torch.bool,
        )
        target_mask = torch.tensor(
            [value == target_key for value in dataset_keys],
            device=labels.device,
            dtype=torch.bool,
        )
        if stage == 'source' and not torch.all(source_mask):
            raise ValueError('Source training batch contains a non-source dataset')

        progress = float(getattr(model, 'training_progress', 1.0))
        drop_ratio = (
            args.river_channel_drop_start_ratio
            + progress * (
                args.river_channel_drop_ratio
                - args.river_channel_drop_start_ratio
            )
        )
        perturbed_channel_mask = None
        if (
            stage == 'source'
            and bool(args.river_channel_drop_enabled)
            and torch.any(source_mask)
        ):
            candidate = drop_valid_recording_sites(channel_mask[source_mask], drop_ratio)
            if torch.any(candidate != channel_mask[source_mask]):
                perturbed_channel_mask = candidate

        batch_size = eeg.shape[0]
        if perturbed_channel_mask is not None and bool(
            getattr(args, 'river_fuse_augmentation_views', 1)
        ):
            combined_eeg = torch.cat((eeg, eeg[source_mask]), dim=0)
            combined_mask = torch.cat((channel_mask, perturbed_channel_mask), dim=0)
            combined_logits, combined_details = model(
                combined_eeg, combined_mask, return_details=True
            )
            logits = combined_logits[:batch_size]
            details = {
                name: (
                    value[:batch_size]
                    if torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == combined_eeg.shape[0]
                    else value
                )
                for name, value in combined_details.items()
            }
            perturbed_logits = combined_logits[batch_size:]
        elif perturbed_channel_mask is not None:
            logits, details = model(eeg, channel_mask, return_details=True)
            perturbed_logits, _ = model(
                eeg[source_mask], perturbed_channel_mask, return_details=True
            )
        else:
            logits, details = model(eeg, channel_mask, return_details=True)
            perturbed_logits = None

        original_task_loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1), labels
        )
        zero = original_task_loss.detach() * 0.0
        augmentation_task_loss = zero
        if perturbed_logits is not None:
            augmentation_task_loss = F.binary_cross_entropy_with_logits(
                perturbed_logits.reshape(-1), labels[source_mask]
            )
            task_loss = 0.5 * (original_task_loss + augmentation_task_loss)
        else:
            task_loss = original_task_loss
        model.update_source_slot_prototypes(
            details['slot_features'],
            labels,
            source_mask,
            args.river_prototype_momentum,
        )
        alignment_loss = details['slot_features'].sum() * 0.0
        if stage == 'target' and args.river_alignment_weight > 0.0:
            alignment_loss = model.class_conditional_slot_alignment_loss(
                details['slot_features'], labels, target_mask
            )
        alignment_start = float(args.river_alignment_start_ratio)
        alignment_ramp = min(
            max((progress - alignment_start) / max(1.0 - alignment_start, 1e-8), 0.0),
            1.0,
        )
        total_loss = task_loss + (
            alignment_ramp * args.river_alignment_weight * alignment_loss
        )
        return logits, total_loss, {
            'task_loss': task_loss.detach(),
            'original_task_loss': original_task_loss.detach(),
            'augmentation_task_loss': augmentation_task_loss.detach(),
            'transfer_alignment_loss': alignment_loss.detach(),
            'alignment_ramp': alignment_ramp,
            'channel_drop_ratio': drop_ratio if perturbed_logits is not None else 0.0,
        }

    return training_objective


def model_arguments(
    config: EpiTransOpConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        'river_version': 'RIVER',
        'river_architecture_version': 'RIVER_Final',
        'river_sampling_frequency': config.sampling_frequency,
        'river_patch_points': config.patch_points,
        'river_descriptor_dim': config.descriptor_dim,
        'river_use_full_clip': True,
        'river_descriptor_hidden_dim': config.descriptor_hidden_dim,
        'river_descriptor_window_seconds': config.descriptor_window_seconds,
        'river_descriptor_slow_seconds': config.descriptor_slow_seconds,
        'river_descriptor_channel_chunk_size': config.descriptor_channel_chunk_size,
        'river_latent_electrodes': config.latent_electrodes,
        'river_embed_dim': config.embed_dim,
        'river_depth': config.depth,
        'river_heads': config.heads,
        'river_ff_dim': config.ff_dim,
        'river_fourier_modes': config.fourier_modes,
        'river_fourier_rank': config.fourier_rank,
        'river_local_kernel_size': config.local_kernel_size,
        'river_sinkhorn_iterations': config.sinkhorn_iterations,
        'river_transport_epsilon': config.transport_epsilon,
        'river_electrode_mass_relaxation': config.electrode_mass_relaxation,
        'river_electrode_mass_temperature': config.electrode_mass_temperature,
        'river_pooling_temperature': config.pooling_temperature,
        'river_transport_mode': config.adapter_mode,
        'river_mass_aware_transport': config.mass_aware_transport,
        'river_residual_bottleneck_dim': config.residual_bottleneck_dim,
        'river_channel_drop_enabled': bool(args.river_channel_drop_enabled),
        'river_fuse_augmentation_views': bool(
            args.river_fuse_augmentation_views
        ),
        'river_slot_temperature': config.slot_temperature,
        'river_evidence_smoothing_kernel': config.evidence_smoothing_kernel,
        'river_temporal_mixer': config.temporal_mixer,
        'river_task_conditioning': config.task,
        'river_alignment_weight': args.river_alignment_weight,
        'river_alignment_start_ratio': args.river_alignment_start_ratio,
        'river_channel_drop_start_ratio': (
            args.river_channel_drop_start_ratio
        ),
        'river_channel_drop_ratio': args.river_channel_drop_ratio,
        'river_prototype_momentum': args.river_prototype_momentum,
        'river_ema_decay': args.river_ema_decay,
        'river_precision': args.river_precision,
        'river_warmup_ratio': args.river_warmup_ratio,
        'river_min_lr_ratio': args.river_min_lr_ratio,
        'river_dropout': config.dropout,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='EpiTransOp EEG transfer and EEG to iEEG transfer entrypoint'
    )
    parser.add_argument('--model', choices=['RIVER'], default='RIVER')
    parser.add_argument('--river-descriptor-hidden-dim', type=int, default=64)
    parser.add_argument('--river-descriptor-window-seconds', type=float, default=4.0)
    parser.add_argument(
        '--river-descriptor-slow-seconds',
        type=float,
        default=float(os.environ.get('RIVER_DESCRIPTOR_SLOW_SECONDS', 16.0)),
    )
    parser.add_argument('--river-descriptor-channel-chunk-size', type=int, default=16384)
    parser.add_argument('--river-latent-electrodes', type=int, default=12)
    parser.add_argument('--river-embed-dim', type=int, default=128)
    parser.add_argument('--river-depth', type=int, default=6)
    parser.add_argument('--river-heads', type=int, default=8)
    parser.add_argument('--river-ff-dim', type=int, default=384)
    parser.add_argument('--river-fourier-modes', type=int, default=32)
    parser.add_argument('--river-fourier-rank', type=int, default=8)
    parser.add_argument('--river-local-kernel-size', type=int, default=5)
    parser.add_argument(
        '--river-slot-temperature',
        type=float,
        default=float(os.environ.get('RIVER_SLOT_TEMPERATURE', 0.50)),
    )
    parser.add_argument(
        '--river-evidence-smoothing-kernel',
        type=int,
        default=int(os.environ.get('RIVER_EVIDENCE_SMOOTHING_KERNEL', 3)),
    )
    parser.add_argument(
        '--river-temporal-mixer',
        choices=['spectral_local', 'axial_attention'],
        default='spectral_local',
    )
    parser.add_argument('--river-alignment-weight', type=float, default=0.05)
    parser.add_argument(
        '--river-alignment-start-ratio',
        type=float,
        default=float(os.environ.get('RIVER_ALIGNMENT_START_RATIO', 0.15)),
    )
    parser.add_argument(
        '--river-channel-drop-start-ratio',
        type=float,
        default=float(os.environ.get('RIVER_CHANNEL_DROP_START_RATIO', 0.05)),
    )
    parser.add_argument('--river-channel-drop-ratio', type=float, default=0.15)
    parser.add_argument(
        '--river-channel-drop-enabled', type=int, choices=[0, 1],
        default=int(os.environ.get('RIVER_CHANNEL_DROP_ENABLED', 1)),
    )
    parser.add_argument(
        '--river-mass-aware-transport', type=int, choices=[0, 1],
        default=int(os.environ.get('RIVER_MASS_AWARE_TRANSPORT', 1)),
    )
    parser.add_argument(
        '--river-residual-bottleneck-dim', type=int,
        default=int(os.environ.get('RIVER_RESIDUAL_BOTTLENECK_DIM', 32)),
    )
    parser.add_argument(
        '--river-fuse-augmentation-views', type=int, choices=[0, 1],
        default=int(os.environ.get('RIVER_FUSE_AUGMENTATION_VIEWS', 1)),
    )
    parser.add_argument('--river-prototype-momentum', type=float, default=0.95)
    parser.add_argument(
        '--river-ema-decay',
        type=float,
        default=float(os.environ.get('RIVER_EMA_DECAY', 0.999)),
    )
    parser.add_argument('--river-sinkhorn-iterations', type=int, default=4)
    parser.add_argument('--river-transport-epsilon', type=float, default=0.20)
    parser.add_argument('--river-electrode-mass-relaxation', type=float, default=0.80)
    parser.add_argument('--river-electrode-mass-temperature', type=float, default=0.50)
    parser.add_argument('--river-pooling-temperature', type=float, default=0.50)
    parser.add_argument(
        '--river-transport-mode',
        choices=['temporal_uot', 'static_uot', 'query_attention'],
        default='temporal_uot',
    )
    parser.add_argument(
        '--river-precision',
        choices=['fp32', 'bf16'],
        default=os.environ.get('RIVER_PRECISION', 'fp32'),
    )
    parser.add_argument(
        '--river-warmup-ratio',
        type=float,
        default=float(os.environ.get('RIVER_WARMUP_RATIO', 0.05)),
    )
    parser.add_argument(
        '--river-min-lr-ratio',
        type=float,
        default=float(os.environ.get('RIVER_MIN_LR_RATIO', 0.01)),
    )
    parser.add_argument('--river-dropout', type=float, default=0.0)
    return add_common_transfer_arguments(parser)


def main() -> None:
    args = build_parser().parse_args()
    config = build_config(args)
    if args.river_alignment_weight < 0.0:
        raise ValueError('EpiTransOp alignment loss weight must be non-negative')
    if not 0.0 <= args.river_channel_drop_ratio < 1.0:
        raise ValueError('EpiTransOp channel drop ratio must be in [0, 1)')
    if not 0.0 <= args.river_channel_drop_start_ratio <= args.river_channel_drop_ratio:
        raise ValueError('EpiTransOp channel drop curriculum is invalid')
    if not 0.0 <= args.river_alignment_start_ratio < 1.0:
        raise ValueError('EpiTransOp alignment start ratio must be in [0, 1)')
    if not 0.0 <= args.river_warmup_ratio < 1.0:
        raise ValueError('EpiTransOp warmup ratio must be in [0, 1)')
    if not 0.0 < args.river_min_lr_ratio <= 1.0:
        raise ValueError('EpiTransOp minimum LR ratio must be in (0, 1]')
    if not 0.0 <= args.river_prototype_momentum < 1.0:
        raise ValueError('EpiTransOp prototype momentum must be in [0, 1)')
    if not 0.0 <= args.river_ema_decay < 1.0:
        raise ValueError('EpiTransOp EMA decay must be in [0, 1)')
    def factory_with_training_recipe(contract):
        model = build_model(contract, config)
        model.training_ema_decay = args.river_ema_decay
        model.training_precision = args.river_precision
        model.training_warmup_ratio = args.river_warmup_ratio
        model.training_min_lr_ratio = args.river_min_lr_ratio
        return model
    install_river_training_runtime()
    diagnostics_callback = (
        collect_transport_diagnostics
        if bool(getattr(args, 'generate_interpretability', 0))
        else None
    )
    run_torch_transfer(
        args,
        model_name=args.model,
        input_spec_name='river',
        model_factory=factory_with_training_recipe,
        pretrained_loader=load_pretrained,
        transfer_configurator=configure_transfer,
        forward_clip=forward_clip,
        budget_module_resolver=budget_modules,
        handles_native_electrodes=True,
        use_full_clip=True,
        model_arguments=model_arguments(config, args),
        training_objective=build_training_objective(args),
        evaluation_diagnostics=diagnostics_callback,
    )


if __name__ == '__main__':
    main()
