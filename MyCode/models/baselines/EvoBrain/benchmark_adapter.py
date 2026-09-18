from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
EVOBRAIN_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, EVOBRAIN_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import utils
from main import train
from model.EvoBrain import EvoBrain_classification
from eeg_benchmark.engine import build_channel_union
from eeg_benchmark.engine import (
    EventBalancedRehearsalSampler,
    PatientBalancedClassSampler,
    UnionClipDataset,
    dynamic_training_sampler,
    mission_training_kwargs,
    prediction_frame,
)
from eeg_benchmark.engine import evaluate_predictions, select_f1_threshold
from eeg_benchmark.engine import configure_reproducibility
from eeg_benchmark.engine import count_parameters, print_model_information, save_json, setup_run_logger
from eeg_benchmark.tasks.cross_dataset import add_mission_arguments, build_spec, prepare_mission, resolve_in_domain_source_checkpoint, validate_mode_args, validate_zero_shot_reference
from eeg_benchmark.tasks.cross_modal import (
    LastLinearInputCapture,
    _aggregate_captured_features,
    collect_native_contact_evidence,
    finalize_precomputed_interpretability,
    run_epilepsy_localization_task,
    safe_refresh_budget_interpretability,
)
from eeg_benchmark.tasks.cross_dataset import mission_sampling_summary
from eeg_benchmark.tasks.cross_modal import (
    CROSS_MODAL_ADAPTER_POLICY,
    apply_native_electrode_adapter,
    attach_native_electrode_adapter,
    NativeElectrodeAdapterContract,
)
from eeg_benchmark.tasks.cross_modal import (
    run_i1_epilepsy_soz,
    run_i4_epilepsy_localization,
    run_i2_thalamocortical_trajectory,
)
from eeg_benchmark.tasks.cross_modal import build_fixed_xai_cohort
from eeg_benchmark.tasks.cross_modal import canonical_contact_name


class EvoBrainDataset(Dataset):
    def __init__(self, dataset: UnionClipDataset, top_k: int) -> None:
        self.dataset = dataset
        self.top_k = top_k
        self.cross_modal_mask_only = (
            getattr(dataset.channel_contract, "policy", None)
            == CROSS_MODAL_ADAPTER_POLICY
        )
        self.metadata_by_clip = {
            str(row.clip_id): {
                "clip_id": str(row.clip_id),
                "patient_id": str(row.patient_id),
                "dataset": str(row.dataset),
                "montage": str(row.montage),
                "source_relative_path": str(row.source_relative_path),
                "clip_start_seconds": float(row.clip_start_seconds),
                "clip_end_seconds": float(row.clip_end_seconds),
                "seizure_intervals_json": str(getattr(row, "seizure_intervals_json", "[]")),
            }
            for row in dataset.manifest.itertuples(index=False)
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        x = item["eeg"][0]
        channel_mask = item["channel_mask"].bool()
        sequence_length = torch.tensor(x.shape[0], dtype=torch.long)
        graph_steps = 1 if self.cross_modal_mask_only else x.shape[0]
        adjacency = torch.diag(channel_mask.float()).unsqueeze(0).repeat(graph_steps, 1, 1)
        return (
            x,
            item["label"].float(),
            sequence_length,
            torch.empty(0, dtype=x.dtype),
            adjacency,
            item["clip_id"],
        )


def collate(batch):
    x, y, lengths, supports, adjacency, metadata = zip(*batch)
    maximum = max(item.shape[1] for item in x)
    padded = []
    padded_adjacency = []
    for tensor, graph in zip(x, adjacency):
        destination = torch.zeros(
            (tensor.shape[0], maximum, tensor.shape[2]), dtype=tensor.dtype
        )
        destination[:, :tensor.shape[1]] = tensor
        padded.append(destination)
        graph_destination = torch.zeros(
            (graph.shape[0], maximum, maximum), dtype=graph.dtype
        )
        graph_destination[:, :graph.shape[1], :graph.shape[2]] = graph
        padded_adjacency.append(graph_destination)
    return (
        torch.stack(padded),
        torch.stack(y),
        torch.stack(lengths),
        torch.empty((len(batch), 0), dtype=torch.float32),
        torch.stack(padded_adjacency),
        list(metadata),
    )


class NativeAdapterEvoBrain(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, top_k: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.top_k = int(top_k)
        attach_native_electrode_adapter(
            self, CROSS_MODAL_ADAPTER_POLICY, 'time_channel_patch',
            NativeElectrodeAdapterContract(sampling_frequency=200.0),
        )

    def forward(self, x, seq_lengths, raw_adjacency):
        native_mask = raw_adjacency[:, 0].diagonal(dim1=-2, dim2=-1) > 0
        adapted, adapted_mask = apply_native_electrode_adapter(
            self, x[:, None], native_mask
        )
        signal = adapted[:, 0]
        centered = signal - signal.mean(dim=-1, keepdim=True)
        normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        adjacency = torch.einsum('btcp,btdp->btcd', normalized, normalized)
        adjacency = (
            adjacency
            * adapted_mask[:, None, :, None]
            * adapted_mask[:, None, None, :]
        )
        if 0 < self.top_k < adjacency.shape[-1]:
            _, indices = torch.topk(adjacency.abs(), self.top_k, dim=-1)
            keep = torch.zeros_like(adjacency, dtype=torch.bool)
            keep.scatter_(-1, indices, True)
            adjacency = adjacency * keep
        return self.backbone(signal, seq_lengths, adjacency)


def make_loader(
    dataset,
    args,
    shuffle: bool,
    patient_class_balanced: bool = False,
) -> DataLoader:
    wrapped = EvoBrainDataset(dataset, args.top_k)
    if shuffle and patient_class_balanced:
        sampler = PatientBalancedClassSampler(dataset, args.seed)
    else:
        sampler = dynamic_training_sampler(dataset) if shuffle else None
    return DataLoader(
        wrapped,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )


def make_budget_loader(target_dataset, source_dataset, args) -> DataLoader:
    sampler = EventBalancedRehearsalSampler(
        target_dataset, source_dataset, args.undersample_seed,
        rehearsal_fraction=args.source_rehearsal_fraction,
    )
    wrapped = torch.utils.data.ConcatDataset([
        EvoBrainDataset(target_dataset, args.top_k),
        EvoBrainDataset(source_dataset, args.top_k),
    ])
    return DataLoader(
        wrapped,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )


def collect_predictions(
    model,
    loader,
    device,
    description: str = 'Inference',
    capture_embeddings: bool = False,
):
    model.eval()
    frames = []
    capture = LastLinearInputCapture(model) if capture_embeddings else None
    feature_parts = []
    try:
        metadata_lookup = loader.dataset.metadata_by_clip
        with torch.no_grad(), tqdm(
            loader,
            total=len(loader),
            desc=description,
            unit='batch',
            colour='green',
            dynamic_ncols=True,
        ) as batches:
            for x, y, lengths, supports, adjacency, clip_ids in batches:
                if capture is not None:
                    capture.values.clear()
                logits, _ = model(
                    x.to(device), lengths.to(device), adjacency.to(device)
                )
                scores = torch.sigmoid(logits.reshape(-1))
                if capture is not None:
                    feature_parts.append(
                        _aggregate_captured_features(capture.values, len(clip_ids))
                    )
                metadata = [metadata_lookup[clip_id] for clip_id in clip_ids]
                pseudo_batch = {
                    "label": y.long(),
                    "clip_id": [item["clip_id"] for item in metadata],
                    "patient_id": [item["patient_id"] for item in metadata],
                    "dataset": [item["dataset"] for item in metadata],
                    "montage": [item["montage"] for item in metadata],
                    "source_relative_path": [item["source_relative_path"] for item in metadata],
                    "clip_start_seconds": torch.tensor([item["clip_start_seconds"] for item in metadata], dtype=torch.float64),
                    "clip_end_seconds": torch.tensor([item["clip_end_seconds"] for item in metadata], dtype=torch.float64),
                    "seizure_intervals_json": [item["seizure_intervals_json"] for item in metadata],
                }
                frames.append(prediction_frame(pseudo_batch, scores))
    finally:
        if capture is not None:
            capture.close()
    predictions = pd.concat(frames, ignore_index=True)
    if capture is None:
        return predictions
    features = np.concatenate(feature_parts, axis=0)
    embedding_metadata = predictions[['clip_id', 'patient_id', 'label', 'score']].copy()
    return predictions, features, embedding_metadata


def collect_embeddings(model, loader, device) -> tuple[np.ndarray, pd.DataFrame]:
    capture = LastLinearInputCapture(model)
    feature_parts = []
    rows = []
    try:
        model.eval()
        metadata_lookup = loader.dataset.metadata_by_clip
        with torch.no_grad():
            for x, y, lengths, supports, adjacency, clip_ids in loader:
                capture.values.clear()
                logits, _ = model(
                    x.to(device), lengths.to(device), adjacency.to(device)
                )
                scores = torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
                feature_parts.append(_aggregate_captured_features(capture.values, len(clip_ids)))
                for index, clip_id in enumerate(clip_ids):
                    metadata = metadata_lookup[clip_id]
                    rows.append({
                        'clip_id': str(clip_id),
                        'patient_id': str(metadata['patient_id']),
                        'label': int(y[index]),
                        'score': float(scores[index]),
                    })
    finally:
        capture.close()
    return np.concatenate(feature_parts, axis=0), pd.DataFrame(rows)


def collect_evobrain_native_evidence(model, loader, device, spec):
    manifest = loader.dataset.dataset.manifest
    cohort = build_fixed_xai_cohort(
        manifest,
        dataset=spec.target_dataset,
        task=spec.task,
        seed=spec.budget_seed,
        maximum_clips_per_patient_per_class=max(
            1, min(32, int(spec.interpretability_max_clips))
        ),
        require_both_classes_per_patient=not (
            spec.target_dataset == 'thalamocortical_ieeg' and spec.task == 'prediction'
        ),
    )
    shared_root = (
        Path(spec.result_root) / spec.mission_type / 'xai_cohorts' / spec.window_name
        / spec.target_dataset / spec.task
    )
    existing = shared_root / 'xai_cohort.json'
    if existing.is_file():
        payload = json.loads(existing.read_text(encoding='utf-8'))
        if payload.get('sha256') != cohort.sha256:
            shared_root.mkdir(parents=True, exist_ok=True)
            (shared_root / 'xai_cohort_conflict.json').write_text(
                json.dumps(
                    {
                        'status': 'rebuilt_shared_xai_cohort',
                        'reason': 'existing_shared_cohort_differs_from_current_contract',
                        'existing_sha256': payload.get('sha256'),
                        'current_sha256': cohort.sha256,
                        'existing_sampling_seed': payload.get('sampling_seed'),
                        'current_sampling_seed': cohort.sampling_seed,
                        'policy': 'continue_with_current_fixed_budget_seed_cohort',
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
            cohort.save(shared_root)
    else:
        cohort.save(shared_root)
    selected_ids = set(cohort.frame['clip_id'].astype(str))
    rows = []
    model.eval()
    metadata_lookup = loader.dataset.metadata_by_clip
    for x, y, lengths, supports, adjacency, clip_ids in loader:
        keep = [index for index, clip_id in enumerate(clip_ids) if clip_id in selected_ids]
        if not keep:
            continue
        selected = torch.as_tensor(keep, dtype=torch.long)
        values = x[selected].to(device).detach().requires_grad_(True)
        selected_lengths = lengths[selected].to(device)
        selected_adjacency = adjacency[selected].to(device)
        model.zero_grad(set_to_none=True)
        logits, _ = model(values, selected_lengths, selected_adjacency)
        gradient = torch.autograd.grad(logits.reshape(-1).sum(), values)[0]
        evidence = (values * gradient).abs().sum(dim=(1, 3))
        evidence = evidence / evidence.sum(dim=1, keepdim=True).clamp_min(1e-12)
        scores = evidence.detach().cpu().numpy()
        selected_ids_batch = [clip_ids[index] for index in keep]
        for sample_index, clip_id in enumerate(selected_ids_batch):
            dataset_index = int(manifest.index[manifest['clip_id'].astype(str) == str(clip_id)][0])
            item = loader.dataset.dataset[dataset_index]
            metadata = metadata_lookup[clip_id]
            positions = item['channel_positions'].numpy()
            for channel_index, name in enumerate(item['channel_names']):
                position = positions[channel_index]
                rows.append({
                    'dataset': spec.target_dataset,
                    'task': spec.task,
                    'patient_id': metadata['patient_id'],
                    'clip_id': clip_id,
                    'label': int(y[keep[sample_index]]),
                    'source_relative_path': metadata['source_relative_path'],
                    'clip_start_seconds': metadata['clip_start_seconds'],
                    'clip_end_seconds': metadata['clip_end_seconds'],
                    'seizure_intervals_json': metadata['seizure_intervals_json'],
                    'contact_name': name,
                    'contact_key': canonical_contact_name(name),
                    'evidence': float(scores[sample_index, channel_index]),
                    'x': float(position[0]), 'y': float(position[1]), 'z': float(position[2]),
                })
    return pd.DataFrame(rows), {
        'cohort_sha256': cohort.sha256,
        'clip_count': int(len(selected_ids)),
        'patient_count': int(cohort.frame['patient_id'].nunique()),
        'maximum_clips_per_patient_per_class': max(
            1, min(32, int(spec.interpretability_max_clips))
        ),
        'sampling_seed': int(spec.seed),
    }


def finalize_evobrain_clinical_interpretability(spec, model, loader, device, target_root):
    contract = json.loads((Path(target_root) / 'dataset_contract.json').read_text(encoding='utf-8'))
    raw_root = contract.get('source_root')
    if not raw_root:
        return {'status': 'skipped_missing_source_root_contract'}
    evidence, cohort = collect_evobrain_native_evidence(model, loader, device, spec)
    root = Path(spec.output_dir) / 'interpretability'
    root.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(root / 'native_contact_evidence.csv', index=False)
    (root / 'fixed_cohort.json').write_text(json.dumps(cohort, indent=2), encoding='utf-8')
    if spec.target_dataset == 'epilepsy_ieeg':
        result = {'I1': run_i1_epilepsy_soz(evidence, raw_root, spec.output_dir)}
        if spec.mission_type == 'eeg_ieeg_localization':
            result['I4'] = run_i4_epilepsy_localization(evidence, raw_root, spec.output_dir)
        result['cohort'] = cohort
        return result
    return {'I2': run_i2_thalamocortical_trajectory(evidence, raw_root, spec.output_dir), 'cohort': cohort}


def native_args(args, spec, channel_count: int, save_dir: Path):
    return argparse.Namespace(
        task=spec.task,
        model_name="evobrain",
        num_classes=1,
        num_nodes=channel_count,
        input_dim=200,
        rnn_units=args.rnn_units,
        agg=args.agg,
        dropout=0.0,
        num_rnn_layers=2,
        max_seq_len=int(round(spec.window_seconds)),
        lr_init=args.lr,
        l2_wd=args.weight_decay,
        num_epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        min_delta=spec.min_delta,
        eval_every=1,
        metric_name="auroc",
        maximize_metric=True,
        stop=False,
        save_dir=str(save_dir),
    )


def configure_budget_training(native, args, model) -> None:
    if args.budget_epochs <= 0:
        raise ValueError('Budget epochs must be positive')
    if args.budget_patience <= 0:
        raise ValueError('Budget patience must be positive')
    if args.target_lr <= 0.0:
        raise ValueError('Target learning rate must be positive')
    native.num_epochs = args.budget_epochs + args.budget_head_only_epochs
    native.patience = args.budget_patience
    native.lr_init = args.target_lr
    native.budget_schedule = False
    native.budget_discriminative_full_model = True
    native.budget_head_only_epochs = args.budget_head_only_epochs
    native.budget_backbone_lr = args.target_backbone_lr
    native.budget_finetune_strategy = 'linear_probe_then_full_model_discriminative_lr'


def run(args: argparse.Namespace) -> None:
    validate_mode_args(args)
    if args.use_pretrained:
        raise ValueError("EvoBrain has no official pretrained weight")
    spec = build_spec(args, "EvoBrain")
    reproducibility = configure_reproducibility(spec.seed, spec.deterministic)
    torch.cuda.set_device(spec.gpu)
    logger = setup_run_logger(spec.output_dir, "benchmark.task1.evobrain")
    source_root, target_root, budget_selection = prepare_mission(spec)
    contract = build_channel_union(source_root / "manifest.csv", target_root / "manifest.csv", spec.source_dataset, spec.target_dataset)
    contract.save(spec.output_dir / "channel_union.json")
    datasets = {
        "source_train": UnionClipDataset(
            source_root, "train", "evobrain", contract,
            view_seconds_override=spec.window_seconds,
            **mission_training_kwargs(spec, 'source', 'train'),
        ),
        "source_dev": UnionClipDataset(source_root, "dev", "evobrain", contract, view_seconds_override=spec.window_seconds),
        "target_test": UnionClipDataset(target_root, "test", "evobrain", contract, view_seconds_override=spec.window_seconds),
    }
    if spec.budget_percent > 0.0:
        datasets.update({
            "target_train": UnionClipDataset(
                target_root, "train", "evobrain", contract,
                clip_ids=budget_selection.clip_ids('train'),
                event_budget_training=True,
                view_seconds_override=spec.window_seconds,
                **mission_training_kwargs(spec, 'target', 'train'),
            ),
            "target_dev": UnionClipDataset(
                target_root, "dev", "evobrain", contract,
                clip_ids=budget_selection.clip_ids('dev'),
                view_seconds_override=spec.window_seconds,
            ),
        })
    source_root_out = spec.output_dir / "source"
    native = native_args(args, spec, len(contract.channel_keys), source_root_out)
    source_loaders = {
        "train": make_loader(datasets["source_train"], args, True),
        "dev": make_loader(datasets["source_dev"], args, False),
    }
    test_loader = make_loader(datasets["target_test"], args, False)
    device = torch.device(f"cuda:{spec.gpu}")
    backbone = EvoBrain_classification(args=native, num_classes=1, device=device)
    model = backbone
    if contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        model = NativeAdapterEvoBrain(backbone, args.top_k)
        adapter_report = model.native_electrode_adapter.contract.to_dict()
        save_json(spec.output_dir / 'cross_modal_adapter.json', adapter_report)
    model = model.to(device)
    total, trainable = count_parameters(model)
    print_model_information(
        {
            "Model": "EvoBrain",
            'Mission': spec.mission_type,
            'In-domain source policy': (
                'reuse canonical completed source checkpoint'
                if spec.is_in_domain else 'train source model'
            ),
            "Parameters": total,
            "Trainable parameters": trainable,
            "Mode": spec.mode,
            "Budget percent": spec.budget_percent,
            "Budget unit": spec.to_dict()['budget_unit'],
            "Budget seed": spec.budget_seed,
            "Window seconds": spec.window_seconds,
            "Task": spec.task,
            "Source": spec.source_dataset,
            "Target": spec.target_dataset,
            "Use pretrained": False,
            "Channel policy": contract.policy,
            "Contract channels": len(contract.channel_keys),
            "Source available channels": len(contract.source_channel_keys),
            "Target available channels": len(contract.target_channel_keys),
            'Laplacian PE width': backbone.evobrain.num_eigenvectors,
            'Laplacian PE policy': 'nontrivial_eigenvectors_zero_pad_to_fixed_width',
            'Laplacian PE batching': 'single_decomposition_for_identical_batch_graphs',
            "Epochs": spec.epochs,
            "Patience": spec.patience,
            'Budget epochs': args.budget_epochs,
            'Budget patience': args.budget_patience,
            'Budget fine-tuning': 'linear_probe_then_full_model_discriminative_lr',
            'Budget head-only epochs': args.budget_head_only_epochs,
            'Early stopping min delta': spec.min_delta,
            "Seed": spec.seed,
            "GPU": spec.physical_gpu,
            "Logical GPU after visibility mask": spec.gpu,
            "Batch size": spec.batch_size,
            'DataLoader workers': spec.num_workers,
            'Statistics CPU workers': spec.stats_num_workers,
            'Bootstrap resamples': spec.bootstrap_resamples,
            'Training sampling': mission_sampling_summary(spec),
            'Source rehearsal fraction': spec.source_rehearsal_fraction,
            'Undersample seed': spec.undersample_seed,
            'Interpretability': spec.generate_interpretability,
            "Output": spec.output_dir,
        }
    )
    save_json(spec.output_dir / "args.json", {
        **spec.to_dict(),
        "learning_rate": args.lr,
        "target_learning_rate": args.target_lr,
        'target_backbone_learning_rate': args.target_backbone_lr,
        'budget_epochs': args.budget_epochs,
        'budget_patience': args.budget_patience,
        'budget_head_only_epochs': args.budget_head_only_epochs,
        'budget_finetune_strategy': 'linear_probe_then_full_model_discriminative_lr',
        "weight_decay": args.weight_decay,
        "maximum_gradient_norm": args.max_grad_norm,
        "rnn_units": args.rnn_units,
        "aggregation": args.agg,
        "top_k": args.top_k,
        'laplacian_pe_width': backbone.evobrain.num_eigenvectors,
        'laplacian_pe_policy': 'nontrivial_eigenvectors_zero_pad_to_fixed_width',
        'laplacian_pe_batching': 'single_decomposition_for_identical_batch_graphs',
        "reproducibility": reproducibility,
        "native_args": vars(native),
    })
    if spec.task == 'localization':
        reference_checkpoint = spec.source_reference_checkpoint_path
        if not reference_checkpoint.is_file():
            raise FileNotFoundError(
                f'Localization requires a completed detection reference checkpoint: {reference_checkpoint}'
            )
        logger.info('Loading localization reference checkpoint from %s', reference_checkpoint)
        model = utils.load_model_checkpoint(reference_checkpoint, model).to(device)
        save_json(spec.output_dir / 'reference_checkpoint.json', {
            'path': str(reference_checkpoint),
            'reference_output_dir': str(spec.source_reference_output_dir),
            'policy': 'reuse_completed_detection_checkpoint_for_soz_localization',
        })
        evidence, cohort = collect_evobrain_native_evidence(
            model,
            test_loader,
            device,
            spec,
        )
        root = Path(spec.output_dir) / 'localization'
        root.mkdir(parents=True, exist_ok=True)
        evidence.to_csv(spec.output_dir / 'predictions.csv', index=False)
        evidence.to_csv(root / 'native_contact_evidence.csv', index=False)
        (root / 'fixed_cohort.json').write_text(json.dumps(cohort, indent=2), encoding='utf-8')
        target_contract = json.loads((target_root / 'dataset_contract.json').read_text(encoding='utf-8'))
        raw_root = target_contract.get('source_root')
        if not raw_root:
            raise ValueError('Localization requires source_root in target dataset contract')
        summary = run_epilepsy_localization_task(evidence, raw_root, spec.output_dir)
        save_json(spec.output_dir / 'metrics.json', summary)
        save_json(spec.output_dir / 'run_summary.json', {
            'status': 'complete',
            'task': spec.task,
            'localization': summary,
            'reference_checkpoint': str(reference_checkpoint),
            'reference_output_dir': str(spec.source_reference_output_dir),
            'cohort_summary': {
                'cohort_sha256': cohort['cohort_sha256'],
                'clip_count': cohort['clip_count'],
                'patient_count': cohort['patient_count'],
                'maximum_clips_per_patient_per_class': cohort['maximum_clips_per_patient_per_class'],
                'sampling_seed': cohort['sampling_seed'],
            },
        })
        logger.info('Completed EvoBrain localization run at %s', spec.output_dir)
        return
    if spec.budget_percent == 0.0:
        if spec.is_in_domain:
            source_checkpoint, source_reference = resolve_in_domain_source_checkpoint(
                spec,
                'best.pth.tar',
                {
                    'learning_rate': args.lr,
                    'weight_decay': args.weight_decay,
                    'maximum_gradient_norm': args.max_grad_norm,
                    'rnn_units': args.rnn_units,
                    'aggregation': args.agg,
                    'top_k': args.top_k,
                },
                contract,
            )
            logger.info('Loading compatible in-domain source checkpoint from %s', source_checkpoint)
            model = utils.load_model_checkpoint(source_checkpoint, model).to(device)
            save_json(spec.output_dir / 'source_checkpoint_reference.json', source_reference)
            logger.info('In-domain source checkpoint loaded without source retraining')
        else:
            source_root_out.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(source_root_out)
            train(model, source_loaders, native, device, str(source_root_out), logger, writer)
            writer.close()
            logger.info('Loading best source checkpoint from %s', source_root_out / 'best.pth.tar')
            model = utils.load_model_checkpoint(source_root_out / "best.pth.tar", model).to(device)
            logger.info('Best source checkpoint loaded')
        selection_loader = source_loaders["dev"]
        threshold_source = "source_dev_max_f1"
    else:
        source_checkpoint = spec.zero_shot_output_dir / "source" / "best.pth.tar"
        validate_zero_shot_reference(spec, source_checkpoint, {
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "maximum_gradient_norm": args.max_grad_norm,
            "rnn_units": args.rnn_units,
            "aggregation": args.agg,
            "top_k": args.top_k,
        }, contract)
        logger.info('Loading zero-shot source checkpoint from %s', source_checkpoint)
        model = utils.load_model_checkpoint(source_checkpoint, model).to(device)
        logger.info('Zero-shot source checkpoint loaded')
        target_root_out = spec.output_dir / "target"
        target_root_out.mkdir(parents=True, exist_ok=True)
        native.save_dir = str(target_root_out)
        configure_budget_training(native, args, model)
        target_loaders = {
            "train": make_budget_loader(
                datasets["target_train"], datasets['source_train'], args,
            ),
            "dev": make_loader(datasets["target_dev"], args, False),
        }
        writer = SummaryWriter(target_root_out)
        train(model, target_loaders, native, device, str(target_root_out), logger, writer)
        writer.close()
        logger.info('Loading best target checkpoint from %s', target_root_out / 'best.pth.tar')
        model = utils.load_model_checkpoint(target_root_out / "best.pth.tar", model).to(device)
        logger.info('Best target checkpoint loaded')
        save_json(spec.output_dir / "source_checkpoint_reference.json", {"path": str(source_checkpoint)})
        selection_loader = target_loaders["dev"]
        threshold_source = "target_dev_max_f1"
    dev_predictions = collect_predictions(
        model,
        selection_loader,
        device,
        description='Best checkpoint dev inference',
    )
    dev_predictions.to_csv(spec.output_dir / 'selection_predictions.csv', index=False)
    from sklearn.metrics import roc_auc_score
    validation_auroc = float(roc_auc_score(
        dev_predictions['label'], dev_predictions['score']
    ))
    threshold = select_f1_threshold(dev_predictions["label"], dev_predictions["score"])
    features = None
    embedding_metadata = None
    if spec.generate_interpretability:
        try:
            predictions, features, embedding_metadata = collect_predictions(
                model,
                test_loader,
                device,
                description='Target test and embedding inference',
                capture_embeddings=True,
            )
        except Exception as exc:
            logger.warning('Combined embedding capture failed: %s', exc)
            predictions = collect_predictions(
                model,
                test_loader,
                device,
                description='Target test inference fallback',
            )
    else:
        predictions = collect_predictions(
            model,
            test_loader,
            device,
            description='Target test inference',
        )
    predictions["predicted_label"] = (predictions["score"] >= threshold).astype(np.int64)
    metrics = evaluate_predictions(
        predictions,
        spec.task,
        threshold,
        spec.output_dir,
        bootstrap_seed=spec.seed,
        bootstrap_resamples=spec.bootstrap_resamples,
        bootstrap_workers=spec.stats_num_workers,
    )
    interpretability = finalize_precomputed_interpretability(
        spec, features, embedding_metadata, source_root, target_root, predictions
    )
    if spec.generate_interpretability and contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        try:
            interpretability['clinical_tasks'] = finalize_evobrain_clinical_interpretability(
                spec, model, test_loader, device, target_root
            )
        except Exception as exc:
            logger.warning('Clinical interpretability failed: %s', exc)
            interpretability['clinical_tasks'] = f'skipped: {type(exc).__name__}: {exc}'
    run_summary = {
        "status": "complete", "selection_metric": "validation_auroc",
        "threshold_source": threshold_source, 'validation_auroc': validation_auroc,
        "metrics": metrics, 'interpretability': interpretability,
    }
    save_json(spec.output_dir / "run_summary.json", run_summary)
    run_summary['budget_interpretability'] = safe_refresh_budget_interpretability(spec)
    save_json(spec.output_dir / "run_summary.json", run_summary)
    logger.info("Completed EvoBrain Task1 run at %s", spec.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvoBrain native trainer for Task1")
    parser.add_argument("--model", choices=["EvoBrain"], default="EvoBrain")
    add_mission_arguments(parser)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--rnn-units", type=int, default=64)
    parser.add_argument("--agg", choices=["max", "mean", "sum", "concat"], default="max")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    torch.cuda.set_device(arguments.gpu)
    run(arguments)
