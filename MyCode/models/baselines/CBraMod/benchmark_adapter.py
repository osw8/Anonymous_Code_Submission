from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
CBRAMOD_ROOT = Path(__file__).resolve().parent
for value in (CBRAMOD_ROOT, BENCHMARK_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from models.baselines.CBraMod.finetune_trainer import Trainer
from models.baselines.CBraMod.models.model_for_task1 import Model
from eeg_benchmark.engine import build_channel_union
from eeg_benchmark.engine import EventBalancedRehearsalSampler, PatientBalancedClassSampler, UnionClipDataset, dynamic_training_sampler, mission_training_kwargs, prediction_frame, union_clip_collate
from eeg_benchmark.engine import evaluate_predictions, select_f1_threshold
from eeg_benchmark.engine import configure_reproducibility
from eeg_benchmark.engine import count_parameters, print_model_information, save_json, setup_run_logger
from eeg_benchmark.tasks.cross_dataset import add_mission_arguments, build_spec, prepare_mission, resolve_in_domain_source_checkpoint, validate_mode_args, validate_zero_shot_reference
from eeg_benchmark.tasks.cross_modal import collect_native_contact_evidence, finalize_standard_torch_interpretability, run_epilepsy_localization_task, safe_refresh_budget_interpretability
from eeg_benchmark.tasks.cross_dataset import mission_sampling_summary
from eeg_benchmark.tasks.cross_modal import (
    CROSS_MODAL_ADAPTER_POLICY,
    apply_native_electrode_adapter,
    attach_native_electrode_adapter,
    NativeElectrodeAdapterContract,
)

CBRAMOD_TASK1_RUNTIME_VERSION = 'mask_grouped_forward_v2'


class NativeTask1Trainer(Trainer):
    def __init__(self, params, loaders, model, output_dir: Path, logger) -> None:
        native_loaders = {"train": loaders["train"], "val": loaders["dev"], "test": loaders["dev"]}
        super().__init__(params, native_loaders, model)
        self.loaders = loaders
        self.output_dir = output_dir
        self.logger = logger
        self.criterion = BCEWithLogitsLoss().cuda()

    def configure_linear_probe_budget(
        self,
        head_learning_rate: float,
        total_epochs: int,
    ) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        head_parameters = tuple(self.model.classifier.parameters())
        for parameter in head_parameters:
            parameter.requires_grad = True
        if not head_parameters:
            raise ValueError('CBraMod linear probe has no classifier parameters')
        self.params.epochs = total_epochs
        if self.params.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                head_parameters,
                lr=head_learning_rate,
                weight_decay=self.params.weight_decay,
            )
        else:
            self.optimizer = torch.optim.SGD(
                head_parameters,
                lr=head_learning_rate,
                momentum=0.9,
                weight_decay=self.params.weight_decay,
            )
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs * self.data_length, 1),
            eta_min=1e-6,
        )

    def configure_full_model_budget(
        self,
        head_learning_rate: float,
        backbone_learning_rate: float,
        total_epochs: int,
    ) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        head_parameters = tuple(self.model.classifier.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        backbone_parameters = tuple(
            parameter for parameter in self.model.parameters()
            if id(parameter) not in head_ids
        )
        if not head_parameters or not backbone_parameters:
            raise ValueError('CBraMod full-model fine-tuning has no parameters')
        parameter_groups = [
            {'params': backbone_parameters, 'lr': backbone_learning_rate},
            {'params': head_parameters, 'lr': head_learning_rate},
        ]
        self.params.epochs = total_epochs
        if self.params.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                parameter_groups,
                lr=head_learning_rate,
                weight_decay=self.params.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                parameter_groups, lr=head_learning_rate, momentum=0.9,
                weight_decay=self.params.weight_decay,
            )
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_epochs * self.data_length, 1), eta_min=1e-6
        )

    @staticmethod
    def _forward(model, batch):
        eeg = batch["eeg"].cuda(non_blocking=True)
        mask = batch["channel_mask"].cuda(non_blocking=True)
        eeg, mask = apply_native_electrode_adapter(model, eeg, mask)
        return model(eeg, mask)

    def predict(self, loader) -> pd.DataFrame:
        self.model.eval()
        frames = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", unit="batch", colour="green"):
                scores = torch.sigmoid(self._forward(self.model, batch))
                frames.append(prediction_frame(batch, scores))
        return pd.concat(frames, ignore_index=True)

    def validation_auroc(self) -> float:
        self.model.eval()
        labels = []
        scores = []
        with torch.no_grad():
            for batch in tqdm(
                self.loaders['dev'],
                desc='Validating',
                unit='batch',
                colour='green',
            ):
                values = torch.sigmoid(self._forward(self.model, batch))
                labels.append(batch['label'].numpy().astype(np.int64, copy=False))
                scores.append(values.detach().cpu().numpy().astype(np.float64, copy=False))
        if not labels:
            raise ValueError('CBraMod validation loader is empty')
        merged_labels = np.concatenate(labels)
        if np.unique(merged_labels).size < 2:
            raise ValueError('CBraMod validation requires both classes')
        return float(roc_auc_score(merged_labels, np.concatenate(scores)))

    def fit_task1(self, patience: int, min_delta: float) -> None:
        best_auroc = -np.inf
        early_stop_reference = -np.inf
        stale_epochs = 0
        best_state = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(self.params.epochs):
            self.model.train()
            losses = []
            progress = tqdm(self.loaders["train"], desc=f"Epoch {epoch + 1}", unit="batch", colour="green")
            for batch_index, batch in enumerate(progress):
                if batch_index == 0:
                    unique_mask_groups = int(torch.unique(
                        batch['channel_mask'],
                        dim=0,
                    ).shape[0])
                    self.logger.info(
                        'epoch=%d first_batch_size=%d unique_channel_mask_groups=%d runtime=%s',
                        epoch + 1,
                        int(batch['channel_mask'].shape[0]),
                        unique_mask_groups,
                        CBRAMOD_TASK1_RUNTIME_VERSION,
                    )
                self.optimizer.zero_grad(set_to_none=True)
                logits = self._forward(self.model, batch)
                labels = batch["label"].float().cuda(non_blocking=True)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(float(loss.detach().cpu()))
                progress.set_postfix(loss=f"{losses[-1]:.5f}")
            dev_auroc = self.validation_auroc()
            self.logger.info(
                "Epoch %d train_loss=%.8f validation_auroc=%.8f",
                epoch + 1,
                float(np.mean(losses)),
                dev_auroc,
            )
            torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": epoch}, self.output_dir / "last.pth")
            if dev_auroc > best_auroc:
                best_auroc = dev_auroc
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save({"model": best_state, "epoch": epoch, "validation_auroc": best_auroc}, self.output_dir / "best.pth")
            if dev_auroc > early_stop_reference + min_delta:
                early_stop_reference = dev_auroc
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                self.logger.info(
                    "Early stopping at epoch %d patience=%d min_delta=%.8f",
                    epoch + 1, patience, min_delta,
                )
                break
        if best_state is None:
            raise RuntimeError("CBraMod did not create a best checkpoint")
        self.model.load_state_dict(best_state)


def make_loader(dataset, args, shuffle: bool, patient_balanced: bool = False) -> DataLoader:
    sampler = (
        PatientBalancedClassSampler(dataset, seed=args.seed)
        if shuffle and patient_balanced
        else dynamic_training_sampler(dataset) if shuffle else None
    )
    loader_options = {}
    if args.num_workers > 0:
        loader_options['prefetch_factor'] = args.prefetch_factor
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
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )


def make_budget_loader(target_dataset, source_dataset, args) -> DataLoader:
    sampler = EventBalancedRehearsalSampler(
        target_dataset, source_dataset, args.undersample_seed,
        rehearsal_fraction=args.source_rehearsal_fraction,
    )
    loader_options = {'prefetch_factor': args.prefetch_factor} if args.num_workers > 0 else {}
    return DataLoader(
        sampler.dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        collate_fn=union_clip_collate,
        generator=torch.Generator().manual_seed(args.seed), **loader_options,
    )


def disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def run(args: argparse.Namespace) -> None:
    if args.prefetch_factor <= 0:
        raise ValueError('CBraMod prefetch factor must be positive')
    validate_mode_args(args)
    spec = build_spec(args, "CBraMod")
    reproducibility = configure_reproducibility(spec.seed, spec.deterministic)
    logger = setup_run_logger(spec.output_dir, "benchmark.task1.cbramod")
    source_root, target_root, budget_selection = prepare_mission(spec)
    contract = build_channel_union(source_root / "manifest.csv", target_root / "manifest.csv", spec.source_dataset, spec.target_dataset)
    contract.save(spec.output_dir / "channel_union.json")
    datasets = {
        "source_train": UnionClipDataset(
            source_root, "train", "cbramod", contract,
            **mission_training_kwargs(spec, 'source', 'train'),
        ),
        "source_dev": UnionClipDataset(source_root, "dev", "cbramod", contract),
        "target_test": UnionClipDataset(target_root, "test", "cbramod", contract),
    }
    if spec.budget_percent > 0.0:
        datasets.update({
            "target_train": UnionClipDataset(
                target_root, "train", "cbramod", contract,
                clip_ids=budget_selection.clip_ids('train'),
                patient_budget_training=True,
                **mission_training_kwargs(spec, 'target', 'train'),
            ),
            "target_dev": UnionClipDataset(
                target_root, "dev", "cbramod", contract,
                clip_ids=budget_selection.clip_ids('dev'),
            ),
        })
    params = argparse.Namespace(
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip_value=args.clip_value,
        multi_lr=bool(args.multi_lr),
        frozen=False,
        downstream_dataset="TUAB",
        label_smoothing=0.0,
        use_pretrained_weights=bool(args.use_pretrained),
        foundation_dir=str(spec.pretrained_path()),
    )
    model = Model(params, len(contract.channel_keys))
    if contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        adapter_report = attach_native_electrode_adapter(
            model, contract.policy, 'patch',
            NativeElectrodeAdapterContract(sampling_frequency=200.0),
        )
        save_json(spec.output_dir / 'cross_modal_adapter.json', adapter_report)
    disable_dropout(model)
    total, trainable = count_parameters(model)
    print_model_information(
        {
            "Model": "CBraMod",
            'Runtime version': CBRAMOD_TASK1_RUNTIME_VERSION,
            'Forward batching': 'group samples by identical channel mask',
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
            'DataLoader prefetch factor': args.prefetch_factor,
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
        'runtime_version': CBRAMOD_TASK1_RUNTIME_VERSION,
        "learning_rate": args.lr,
        "target_learning_rate": args.target_lr,
        'target_backbone_learning_rate': args.target_backbone_lr,
        'budget_epochs': args.budget_epochs,
        'budget_patience': args.budget_patience,
        'budget_head_only_epochs': args.budget_head_only_epochs,
        'budget_finetune_strategy': 'linear_probe_then_full_model_discriminative_lr',
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "clip_value": args.clip_value,
        "multi_lr": args.multi_lr,
        'prefetch_factor': args.prefetch_factor,
        "reproducibility": reproducibility,
        "native_optimizer": vars(params),
    })
    source_loaders = {
        "train": make_loader(datasets["source_train"], args, True),
        "dev": make_loader(datasets["source_dev"], args, False),
    }
    target_loaders = {"test": make_loader(datasets["target_test"], args, False)}
    if spec.budget_percent > 0.0:
        target_loaders.update({
            "train": make_budget_loader(
                datasets['target_train'], datasets['source_train'], args,
            ),
            "dev": make_loader(datasets["target_dev"], args, False),
        })
    if spec.task == 'localization':
        reference_checkpoint = spec.source_reference_checkpoint_path
        if not reference_checkpoint.is_file():
            raise FileNotFoundError(
                f'Localization requires a completed detection reference checkpoint: {reference_checkpoint}'
            )
        checkpoint = torch.load(reference_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        save_json(spec.output_dir / 'source_checkpoint_reference.json', {
            'path': str(reference_checkpoint),
            'reference_output_dir': str(spec.source_reference_output_dir),
            'policy': 'reuse_completed_detection_checkpoint_for_soz_localization',
        })
        logger.info(
            'Localization evaluation-only mode: skipping clip prediction metrics '
            'and threshold selection'
        )
        evidence, cohort = collect_native_contact_evidence(
            model,
            target_loaders['test'],
            torch.device(f'cuda:{spec.gpu}'),
            lambda selected_model, eeg, mask: selected_model(
                *apply_native_electrode_adapter(selected_model, eeg, mask)
            ),
            dataset=spec.target_dataset,
            task=spec.task,
            maximum_per_patient_class=max(1, min(32, int(spec.interpretability_max_clips))),
            seed=spec.budget_seed,
        )
        root = spec.output_dir / 'localization'
        root.mkdir(parents=True, exist_ok=True)
        evidence.to_csv(spec.output_dir / 'predictions.csv', index=False)
        evidence.to_csv(root / 'native_contact_evidence.csv', index=False)
        (root / 'fixed_cohort.json').write_text(json.dumps(cohort, indent=2), encoding='utf-8')
        dataset_contract = json.loads((target_root / 'dataset_contract.json').read_text(encoding='utf-8'))
        raw_root = dataset_contract.get('source_root')
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
        })
        logger.info("Completed CBraMod localization run at %s", spec.output_dir)
        return
    if spec.budget_percent == 0.0:
        source_trainer = NativeTask1Trainer(params, source_loaders, model, spec.output_dir / "source", logger)
        if spec.is_in_domain:
            source_checkpoint, source_reference = resolve_in_domain_source_checkpoint(
                spec,
                'best.pth',
                {
                    'learning_rate': args.lr,
                    'weight_decay': args.weight_decay,
                    'optimizer': args.optimizer,
                    'clip_value': args.clip_value,
                    'multi_lr': args.multi_lr,
                },
                contract,
            )
            checkpoint = torch.load(source_checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
            save_json(spec.output_dir / 'source_checkpoint_reference.json', source_reference)
            logger.info('In-domain source checkpoint loaded without source retraining: %s', source_checkpoint)
        else:
            source_trainer.fit_task1(spec.patience, spec.min_delta)
        final_trainer = source_trainer
        dev_predictions = source_trainer.predict(source_loaders["dev"])
        threshold_source = "source_dev_max_f1"
    else:
        source_checkpoint = spec.zero_shot_output_dir / "source" / "best.pth"
        validate_zero_shot_reference(spec, source_checkpoint, {
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "clip_value": args.clip_value,
            "multi_lr": args.multi_lr,
        }, contract)
        checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        params.lr = args.target_lr
        params.frozen = False
        save_json(spec.output_dir / "source_checkpoint_reference.json", {"path": str(source_checkpoint)})
        if args.budget_head_only_epochs > 0:
            head_trainer = NativeTask1Trainer(
                params,
                target_loaders,
                model,
                spec.output_dir / 'target_linear_probe',
                logger,
            )
            head_trainer.configure_linear_probe_budget(
                args.target_lr,
                args.budget_head_only_epochs,
            )
            logger.info(
                'target_stage phase=linear_probe epochs=%d head_lr=%g',
                args.budget_head_only_epochs,
                args.target_lr,
            )
            head_trainer.fit_task1(
                args.budget_head_only_epochs + 1,
                spec.min_delta,
            )
        final_trainer = NativeTask1Trainer(params, target_loaders, model, spec.output_dir / "target", logger)
        final_trainer.configure_full_model_budget(
            args.target_lr, args.target_backbone_lr, args.budget_epochs,
        )
        logger.info(
            'target_stage phase=full_model strategy=discriminative_lr '
            'head_lr=%g backbone_lr=%g',
            args.target_lr, args.target_backbone_lr,
        )
        final_trainer.fit_task1(args.budget_patience, spec.min_delta)
        dev_predictions = final_trainer.predict(target_loaders["dev"])
        threshold_source = "target_dev_max_f1"
    threshold = select_f1_threshold(dev_predictions["label"], dev_predictions["score"])
    dev_predictions.to_csv(spec.output_dir / 'selection_predictions.csv', index=False)
    validation_auroc = float(roc_auc_score(
        dev_predictions['label'], dev_predictions['score']
    ))
    predictions = final_trainer.predict(target_loaders["test"])
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
    interpretability = finalize_standard_torch_interpretability(
        spec,
        final_trainer.model,
        target_loaders['test'],
        torch.device(f'cuda:{spec.gpu}'),
        lambda selected_model, eeg, mask: selected_model(
            *apply_native_electrode_adapter(selected_model, eeg, mask)
        ),
        source_root,
        target_root,
        predictions,
    )
    run_summary = {
        "status": "complete", "selection_metric": "validation_auroc",
        "threshold_source": threshold_source, 'validation_auroc': validation_auroc,
        "metrics": metrics, 'interpretability': interpretability,
    }
    save_json(spec.output_dir / "run_summary.json", run_summary)
    run_summary['budget_interpretability'] = safe_refresh_budget_interpretability(spec)
    save_json(spec.output_dir / "run_summary.json", run_summary)
    logger.info("Completed CBraMod Task1 run at %s", spec.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CBraMod native trainer for Task1")
    parser.add_argument("--model", choices=["CBraMod"], default="CBraMod")
    add_mission_arguments(parser)
    parser.add_argument("--optimizer", choices=["AdamW", "SGD"], default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--clip-value", type=float, default=1.0)
    parser.add_argument("--multi-lr", type=int, choices=[0, 1], default=1)
    parser.add_argument('--prefetch-factor', type=int, default=4)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    torch.cuda.set_device(arguments.gpu)
    run(arguments)
