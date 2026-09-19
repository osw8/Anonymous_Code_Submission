from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
BIOT_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, BIOT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from model.biot import BIOTClassifier
from eeg_benchmark.engine import build_channel_union
from eeg_benchmark.engine import EventBalancedRehearsalSampler, PatientBalancedClassSampler, UnionClipDataset, dynamic_training_sampler, mission_training_kwargs, prediction_frame, union_clip_collate
from eeg_benchmark.engine import evaluate_predictions, select_f1_threshold
from eeg_benchmark.engine import configure_reproducibility
from eeg_benchmark.engine import (
    count_parameters,
    enable_full_finetuning,
    print_model_information,
    save_json,
    setup_run_logger,
)
from eeg_benchmark.tasks.cross_dataset import add_mission_arguments, build_spec, prepare_mission, resolve_in_domain_source_checkpoint, validate_mode_args, validate_zero_shot_reference
from eeg_benchmark.tasks.cross_modal import (
    collect_native_contact_evidence,
    finalize_standard_torch_interpretability,
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


def disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


class BIOTTaskModule(pl.LightningModule):
    def __init__(
        self,
        model: BIOTClassifier,
        learning_rate: float,
        weight_decay: float,
        backbone_learning_rate: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.backbone_learning_rate = backbone_learning_rate
        self.validation_outputs: list[tuple[np.ndarray, np.ndarray]] = []

    def clip_logits(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        eeg, channel_mask = apply_native_electrode_adapter(
            self.model, eeg, channel_mask
        )
        if channel_mask.shape[0] > 0 and torch.equal(
            channel_mask,
            channel_mask[:1].expand_as(channel_mask),
        ):
            indices = torch.nonzero(
                channel_mask[0], as_tuple=False
            ).reshape(-1).to(eeg.device)
            if indices.numel() == 0:
                raise ValueError('BIOT batch has no available channels')
            batch, views, _, points = eeg.shape
            selected = eeg[:, :, indices].reshape(
                batch * views, indices.numel(), points
            )
            logits = self.model(
                selected, channel_indices=indices
            ).reshape(batch, views, -1)
            return logits.mean(dim=1).reshape(batch)
        outputs = []
        for sample, mask in zip(eeg, channel_mask):
            indices = torch.nonzero(mask, as_tuple=False).reshape(-1).to(sample.device)
            if indices.numel() == 0:
                raise ValueError("BIOT sample has no available channels")
            selected = sample[:, indices]
            logits = self.model(selected, channel_indices=indices).reshape(sample.shape[0], -1)
            outputs.append(logits.mean(dim=0))
        return torch.stack(outputs).reshape(eeg.shape[0])

    def training_step(self, batch, batch_index):
        logits = self.clip_logits(batch["eeg"], batch["channel_mask"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"].float())
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=int(batch["label"].shape[0]),
        )
        return loss

    def validation_step(self, batch, batch_index):
        scores = torch.sigmoid(self.clip_logits(batch["eeg"], batch["channel_mask"]))
        labels = batch["label"]
        self.validation_outputs.append((scores.detach().cpu().numpy(), labels.detach().cpu().numpy()))

    def on_validation_epoch_end(self) -> None:
        if not self.validation_outputs:
            return
        scores = np.concatenate([item[0] for item in self.validation_outputs])
        labels = np.concatenate([item[1] for item in self.validation_outputs])
        from sklearn.metrics import roc_auc_score

        value = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else 0.0
        self.log(
            "val_auroc",
            value,
            prog_bar=True,
            sync_dist=False,
            on_step=False,
            on_epoch=True,
        )
        self.validation_outputs.clear()

    def configure_optimizers(self):
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if self.backbone_learning_rate is None:
            parameter_groups = trainable_parameters
        else:
            head_ids = {
                id(parameter) for parameter in self.model.classifier.parameters()
            }
            head_parameters = [
                parameter for parameter in trainable_parameters
                if id(parameter) in head_ids
            ]
            backbone_parameters = [
                parameter for parameter in trainable_parameters
                if id(parameter) not in head_ids
            ]
            if not head_parameters or not backbone_parameters:
                raise ValueError(
                    'BIOT could not separate classifier and backbone parameters'
                )
            parameter_groups = [
                {
                    'params': backbone_parameters,
                    'lr': self.backbone_learning_rate,
                },
                {'params': head_parameters, 'lr': self.learning_rate},
            ]
        return torch.optim.Adam(
            parameter_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class GreenProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        progress = super().init_train_tqdm()
        progress.colour = "green"
        return progress

    def init_validation_tqdm(self):
        progress = super().init_validation_tqdm()
        progress.colour = "green"
        return progress


class EnglishEpochLogger(Callback):
    def __init__(self, logger) -> None:
        self.run_logger = logger

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_auroc = metrics.get("val_auroc")
        self.run_logger.info(
            "epoch=%d train_loss=%s val_auroc=%s",
            trainer.current_epoch + 1,
            "nan" if train_loss is None else f"{float(train_loss):.8f}",
            "nan" if val_auroc is None else f"{float(val_auroc):.8f}",
        )


class BudgetEarlyStoppingCallback(Callback):
    def __init__(self, patience: int, min_delta: float, logger) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.logger = logger
        self.reference = -np.inf
        self.stale = 0

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get('val_auroc')
        if value is None:
            return
        score = float(value)
        if score > self.reference + self.min_delta:
            self.reference = score
            self.stale = 0
        else:
            self.stale += 1
        if self.stale >= self.patience:
            self.logger.info(
                'early_stop_epoch=%d patience=%d min_delta=%.8f',
                trainer.current_epoch + 1, self.patience, self.min_delta,
            )
            trainer.should_stop = True


def load_pretrained_encoder(model: BIOTClassifier, path: Path, channel_keys: tuple[str, ...]) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    current = model.biot.state_dict()
    compatible = {}
    skipped = []
    pretrained_channel_names = (
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "C3-A2", "C4-A1",
    )
    channel_weight = checkpoint.get("channel_tokens.weight")
    if channel_weight is not None:
        initialized = channel_weight.mean(dim=0, keepdim=True).repeat(len(channel_keys), 1)
        source_index = {name: index for index, name in enumerate(pretrained_channel_names)}
        for target_index, name in enumerate(channel_keys):
            if name in source_index:
                initialized[target_index] = channel_weight[source_index[name]]
        compatible["channel_tokens.weight"] = initialized
    for key, value in checkpoint.items():
        if key == "channel_tokens.weight":
            continue
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    result = model.biot.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensor_count": len(compatible),
        "skipped_keys": skipped,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
            "channel_embedding_policy": "semantic_copy_for_18_BIOT_channels_mean_embedding_for_unseen_channels",
    }


def make_loader(dataset, args, shuffle: bool, patient_balanced: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    sampler = (
        PatientBalancedClassSampler(dataset, seed=args.seed)
        if shuffle and patient_balanced
        else dynamic_training_sampler(dataset) if shuffle else None
    )
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
    )


def make_budget_loader(target_dataset, source_dataset, args) -> DataLoader:
    sampler = EventBalancedRehearsalSampler(
        target_dataset, source_dataset, args.undersample_seed,
        rehearsal_fraction=args.source_rehearsal_fraction,
    )
    return DataLoader(
        sampler.dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=union_clip_collate,
        generator=torch.Generator().manual_seed(args.seed),
    )


def collect_predictions(module: BIOTTaskModule, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    module.to(device).eval()
    frames = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            scores = torch.sigmoid(module.clip_logits(eeg, batch["channel_mask"].to(device)))
            frames.append(prediction_frame(batch, scores))
    return pd.concat(frames, ignore_index=True)


def fit_stage(module, train_loader, dev_loader, output_dir: Path, args, stage: str) -> Path:
    stage_root = output_dir / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    if len(dev_loader) == 0:
        raise ValueError(f"Validation loader is empty for stage={stage}")
    args.run_logger.info(
        'stage=%s train_batches=%d val_batches=%d',
        stage,
        len(train_loader),
        len(dev_loader),
    )
    checkpoint = ModelCheckpoint(
        dirpath=stage_root,
        filename="best",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
    )
    if stage == 'target':
        early_stop = BudgetEarlyStoppingCallback(
            args.budget_patience, args.min_delta, args.run_logger,
        )
        max_epochs = args.budget_epochs
    elif stage == 'target_linear_probe':
        early_stop = BudgetEarlyStoppingCallback(
            args.budget_head_only_epochs + 1,
            args.min_delta,
            args.run_logger,
        )
        max_epochs = args.budget_head_only_epochs
    else:
        early_stop = EarlyStopping(
            monitor='val_auroc', patience=args.patience,
            min_delta=args.min_delta, mode='max',
            check_on_train_epoch_end=False,
        )
        max_epochs = args.epochs
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=max_epochs,
        callbacks=[checkpoint, early_stop, GreenProgressBar(), EnglishEpochLogger(args.run_logger)],
        logger=False,
        deterministic=bool(args.deterministic),
        benchmark=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        default_root_dir=stage_root,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    if checkpoint.best_model_path:
        return Path(checkpoint.best_model_path)
    if checkpoint.last_model_path:
        last_path = Path(checkpoint.last_model_path)
        fallback_path = stage_root / "best.ckpt"
        shutil.copy2(last_path, fallback_path)
        args.run_logger.warning(
            'stage=%s monitor_checkpoint_missing_fallback_to_last=%s',
            stage,
            last_path,
        )
        return fallback_path
    raise RuntimeError(f"No checkpoint was created for {stage}")


def run(args: argparse.Namespace) -> None:
    validate_mode_args(args)
    spec = build_spec(args, "BIOT")
    reproducibility = configure_reproducibility(spec.seed, spec.deterministic)
    logger = setup_run_logger(spec.output_dir, "benchmark.task1.biot")
    args.run_logger = logger
    source_root, target_root, budget_selection = prepare_mission(spec)
    contract = build_channel_union(
        source_root / "manifest.csv",
        target_root / "manifest.csv",
        spec.source_dataset,
        spec.target_dataset,
    )
    contract.save(spec.output_dir / "channel_union.json")
    split_specs = {
        "source_train": UnionClipDataset(
            source_root, "train", "biot", contract,
            **mission_training_kwargs(spec, 'source', 'train'),
        ),
        "source_dev": UnionClipDataset(source_root, "dev", "biot", contract),
        "target_test": UnionClipDataset(target_root, "test", "biot", contract),
    }
    if spec.budget_percent > 0.0:
        split_specs.update({
            "target_train": UnionClipDataset(
                target_root, "train", "biot", contract,
                clip_ids=budget_selection.clip_ids('train'),
                patient_budget_training=True,
                **mission_training_kwargs(spec, 'target', 'train'),
            ),
            "target_dev": UnionClipDataset(
                target_root, "dev", "biot", contract,
                clip_ids=budget_selection.clip_ids('dev'),
            ),
        })
    datasets = split_specs
    model = BIOTClassifier(
        n_classes=1,
        n_channels=len(contract.channel_keys),
        n_fft=args.token_size,
        hop_length=args.hop_length,
    )
    if contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        adapter_report = attach_native_electrode_adapter(
            model, contract.policy, 'continuous',
            NativeElectrodeAdapterContract(sampling_frequency=200.0),
        )
        save_json(spec.output_dir / 'cross_modal_adapter.json', adapter_report)
    disable_dropout(model)
    pretrained_report = None
    if spec.use_pretrained:
        pretrained_path = spec.pretrained_path()
        if pretrained_path is None or not pretrained_path.exists():
            raise FileNotFoundError(f"BIOT pretrained weight does not exist: {pretrained_path}")
        pretrained_report = load_pretrained_encoder(model, pretrained_path, contract.channel_keys)
        save_json(spec.output_dir / "pretrained_load_report.json", pretrained_report)
    total, trainable = count_parameters(model)
    print_model_information(
        {
            "Model": "BIOT",
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
            "Use pretrained": spec.use_pretrained,
            "Channel policy": contract.policy,
            "Contract channels": len(contract.channel_keys),
            "Source available channels": len(contract.source_channel_keys),
            "Target available channels": len(contract.target_channel_keys),
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
        "token_size": args.token_size,
        "hop_length": args.hop_length,
        "reproducibility": reproducibility,
        "pretrained_report": pretrained_report,
    })
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
    source_module = BIOTTaskModule(model, args.lr, args.weight_decay)
    if spec.task == 'localization':
        reference_checkpoint = spec.source_reference_checkpoint_path
        if not reference_checkpoint.is_file():
            raise FileNotFoundError(
                f'Localization requires a completed detection reference checkpoint: {reference_checkpoint}'
            )
        source_module = BIOTTaskModule.load_from_checkpoint(
            reference_checkpoint,
            map_location='cpu',
            model=model,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            backbone_learning_rate=args.target_backbone_lr,
        ).to(torch.device(f'cuda:{spec.gpu}'))
        save_json(spec.output_dir / 'reference_checkpoint.json', {
            'path': str(reference_checkpoint),
            'reference_output_dir': str(spec.reference_output_dir),
            'policy': 'reuse_completed_detection_checkpoint_for_soz_localization',
        })
        evidence, cohort = collect_native_contact_evidence(
            source_module,
            loaders['target_test'],
            torch.device(f'cuda:{spec.gpu}'),
            lambda module, eeg, mask: module.clip_logits(eeg, mask),
            dataset=spec.target_dataset,
            task=spec.task,
            maximum_per_patient_class=max(1, min(32, int(spec.interpretability_max_clips))),
            seed=spec.budget_seed,
            cohort_root=(Path(spec.result_root) / spec.mission_type / 'xai_cohorts' / spec.window_name),
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
            'cohort_summary': {
                'cohort_sha256': cohort['cohort_sha256'],
                'clip_count': cohort['clip_count'],
                'patient_count': cohort['patient_count'],
                'maximum_clips_per_patient_per_class': cohort['maximum_clips_per_patient_per_class'],
                'sampling_seed': cohort['sampling_seed'],
            },
        })
        logger.info('Completed BIOT localization run at %s', spec.output_dir)
        return
    if spec.budget_percent == 0.0:
        if spec.is_in_domain:
            source_best, source_reference = resolve_in_domain_source_checkpoint(
                spec,
                'best.ckpt',
                {
                    'learning_rate': args.lr,
                    'weight_decay': args.weight_decay,
                    'token_size': args.token_size,
                    'hop_length': args.hop_length,
                },
                contract,
            )
            save_json(spec.output_dir / 'source_checkpoint_reference.json', source_reference)
            logger.info('In-domain source checkpoint loaded without source retraining: %s', source_best)
        else:
            source_best = fit_stage(
                source_module, loaders["source_train"], loaders["source_dev"],
                spec.output_dir, args, "source"
            )
        source_module = BIOTTaskModule.load_from_checkpoint(
            source_best,
            map_location='cpu',
            model=model,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        )
        selection_module = source_module
        dev_predictions = collect_predictions(selection_module, loaders["source_dev"], torch.device(f"cuda:{spec.gpu}"))
        threshold_source = "source_dev_max_f1"
    else:
        source_best = spec.zero_shot_output_dir / "source" / "best.ckpt"
        validate_zero_shot_reference(spec, source_best, {
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "token_size": args.token_size,
            "hop_length": args.hop_length,
        }, contract)
        source_module = BIOTTaskModule.load_from_checkpoint(
            source_best,
            map_location='cpu',
            model=model,
            learning_rate=args.target_lr,
            weight_decay=args.weight_decay,
            backbone_learning_rate=args.target_backbone_lr,
        )
        if args.budget_head_only_epochs > 0:
            for parameter in source_module.parameters():
                parameter.requires_grad = False
            for parameter in source_module.model.classifier.parameters():
                parameter.requires_grad = True
            source_module.learning_rate = args.target_lr
            source_module.backbone_learning_rate = None
            logger.info(
                'target_stage phase=linear_probe epochs=%d head_lr=%g',
                args.budget_head_only_epochs,
                args.target_lr,
            )
            head_best = fit_stage(
                source_module,
                loaders['target_train'],
                loaders['target_dev'],
                spec.output_dir,
                args,
                'target_linear_probe',
            )
            source_module = BIOTTaskModule.load_from_checkpoint(
                head_best,
                map_location='cpu',
                model=source_module.model,
                learning_rate=args.target_lr,
                weight_decay=args.weight_decay,
                backbone_learning_rate=args.target_backbone_lr,
            )
        non_differentiable_parameters = enable_full_finetuning(source_module)
        source_module.learning_rate = args.target_lr
        source_module.backbone_learning_rate = args.target_backbone_lr
        logger.info(
            'target_stage phase=full_model strategy=discriminative_lr '
            'head_lr=%g backbone_lr=%g non_differentiable_frozen=%s',
            args.target_lr,
            args.target_backbone_lr,
            ','.join(non_differentiable_parameters) or 'none',
        )
        save_json(spec.output_dir / "source_checkpoint_reference.json", {"path": str(source_best)})
        target_best = fit_stage(
            source_module,
            loaders["target_train"],
            loaders["target_dev"],
            spec.output_dir,
            args,
            "target",
        )
        selection_module = BIOTTaskModule.load_from_checkpoint(
            target_best,
            map_location='cpu',
            model=source_module.model,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            backbone_learning_rate=None,
        )
        dev_predictions = collect_predictions(selection_module, loaders["target_dev"], torch.device(f"cuda:{spec.gpu}"))
        threshold_source = "target_dev_max_f1"
    threshold = select_f1_threshold(dev_predictions["label"], dev_predictions["score"])
    dev_predictions.to_csv(spec.output_dir / 'selection_predictions.csv', index=False)
    from sklearn.metrics import roc_auc_score
    validation_auroc = float(roc_auc_score(
        dev_predictions['label'], dev_predictions['score']
    ))
    target_predictions = collect_predictions(selection_module, loaders["target_test"], torch.device(f"cuda:{spec.gpu}"))
    target_predictions["predicted_label"] = (target_predictions["score"] >= threshold).astype(np.int64)
    metrics = evaluate_predictions(
        target_predictions,
        spec.task,
        threshold,
        spec.output_dir,
        bootstrap_seed=spec.seed,
        bootstrap_resamples=spec.bootstrap_resamples,
        bootstrap_workers=spec.stats_num_workers,
    )
    interpretability = finalize_standard_torch_interpretability(
        spec,
        selection_module,
        loaders['target_test'],
        torch.device(f'cuda:{spec.gpu}'),
        lambda module, eeg, mask: module.clip_logits(eeg, mask),
        source_root,
        target_root,
        target_predictions,
    )
    run_summary = {
            "status": "complete",
            "selection_metric": "validation_auroc",
            "threshold_source": threshold_source,
            'validation_auroc': validation_auroc,
            "metrics": metrics,
            'interpretability': interpretability,
    }
    save_json(spec.output_dir / "run_summary.json", run_summary)
    run_summary['budget_interpretability'] = safe_refresh_budget_interpretability(spec)
    save_json(spec.output_dir / "run_summary.json", run_summary)
    logger.info("Completed BIOT Task1 run at %s", spec.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BIOT native Lightning trainer for Task1")
    parser.add_argument("--model", choices=["BIOT"], default="BIOT")
    add_mission_arguments(parser)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--token-size", type=int, default=200)
    parser.add_argument("--hop-length", type=int, default=100)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    torch.cuda.set_device(arguments.gpu)
    run(arguments)
