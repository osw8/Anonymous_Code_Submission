from __future__ import annotations

# Implements RIVER optimization and training objectives.
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


ClipForward = Callable[[Any, Any, Any], Any]
TrainingObjective = Callable[
    [Any, dict[str, Any], Any, Any, Any, str],
    tuple[Any, Any, dict[str, Any]],
]


def _loss_failure_context(
    stage: str,
    epoch: int,
    epochs: int,
    batch_index: int,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_terms: dict[str, Any],
) -> str:
    finite_logits = logits.detach().float()[torch.isfinite(logits.detach().float())]
    if finite_logits.numel():
        logit_range = (
            float(finite_logits.min().cpu()),
            float(finite_logits.max().cpu()),
        )
    else:
        logit_range = ('none', 'none')
    terms = {}
    for name, value in loss_terms.items():
        tensor = value.detach().float() if torch.is_tensor(value) else torch.tensor(float(value))
        terms[name] = float(tensor.cpu()) if torch.isfinite(tensor).all() else 'non_finite'
    return (
        f'stage={stage} epoch={epoch}/{epochs} batch={batch_index} '
        f'logit_range={logit_range} label_mean={float(labels.float().mean().cpu()):.6f} '
        f'loss_terms={terms}'
    )


class _EpochProgress:
    def __init__(self, total: int, description: str) -> None:
        self.total = max(int(total), 1)
        self.description = description
        self.started_at = time.perf_counter()
        self.last_completed = 0
        self.last_loss: float | None = None
        self.update(0)

    def update(self, completed: int, loss: float | None = None) -> None:
        completed = min(max(int(completed), 0), self.total)
        self.last_completed = completed
        if loss is not None:
            self.last_loss = loss
        elapsed = max(time.perf_counter() - self.started_at, 1e-8)
        rate = completed / elapsed
        remaining = (self.total - completed) / rate if rate > 0.0 else math.inf
        terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        reserved = len(self.description) + 76
        bar_width = min(40, max(10, terminal_width - reserved))
        filled = int(round(bar_width * completed / self.total))
        if completed > 0:
            filled = max(filled, 1)
        bar = (
            f'\033[42m{" " * filled}\033[0m'
            + ' ' * (bar_width - filled)
        )
        percent = 100.0 * completed / self.total
        suffix = '' if loss is None else f', loss={loss:.5f}'
        line = (
            f'{self.description}: {percent:3.0f}%|{bar}| '
            f'{completed}/{self.total} '
            f'[{_format_duration(elapsed)}<{_format_duration(remaining)}, '
            f'{rate:.2f}batch/s{suffix}]'
        )
        sys.stdout.write(f'\r\033[2K{line}')
        sys.stdout.flush()

    def close(self) -> None:
        if self.last_completed < self.total:
            self.update(self.total, self.last_loss)
        sys.stdout.write('\n')
        sys.stdout.flush()


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return '?'
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


def _write_training_summary(
    completed: int,
    total: int,
    elapsed: float,
    train_loss: float,
    val_auroc: float,
) -> None:
    total = max(int(total), 1)
    completed = min(max(int(completed), 0), total)
    seconds_per_epoch = elapsed / max(completed, 1)
    remaining = seconds_per_epoch * (total - completed)
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    reserved = 91
    bar_width = min(40, max(10, terminal_width - reserved))
    filled = int(round(bar_width * completed / total))
    if completed > 0:
        filled = max(filled, 1)
    bar = (
        f'\033[42m{" " * filled}\033[0m'
        + ' ' * (bar_width - filled)
    )
    percent = 100.0 * completed / total
    line = (
        f'Training: {percent:3.0f}%|{bar}| '
        f'{completed}/{total} '
        f'[{_format_duration(elapsed)}<{_format_duration(remaining)}, '
        f'{seconds_per_epoch:.2f}s/epoch, loss={train_loss:.5f}, '
        f'val_auroc={val_auroc:.6f}]'
    )
    sys.stdout.write(f'\r\033[2K{line}\n')
    sys.stdout.flush()


def _log_file_only(
    logger: logging.Logger,
    message: str,
    *args: object,
) -> None:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        args,
        None,
    )
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def _clone_state(
    model: torch.nn.Module,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device=device).clone()
        for name, value in model.state_dict().items()
    }


def _sanitize_nonfinite_gradients(model: torch.nn.Module) -> list[str]:
    repaired: list[str] = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if torch.isfinite(parameter.grad).all():
                continue
            parameter.grad.copy_(
                torch.nan_to_num(parameter.grad, nan=0.0, posinf=0.0, neginf=0.0)
            )
            repaired.append(name)
    return repaired


def _sanitize_nonfinite_parameters(model: torch.nn.Module) -> list[str]:
    repaired: list[str] = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if torch.isfinite(parameter).all():
                continue
            parameter.copy_(
                torch.nan_to_num(parameter, nan=0.0, posinf=1.0, neginf=-1.0)
            )
            repaired.append(name)
    return repaired


@torch.inference_mode()
def _collect_validation_scores(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    forward_clip: ClipForward,
    description: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.to(device).eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    running_loss = 0.0
    seen = 0
    progress = _EpochProgress(len(loader), description)
    use_bf16 = (
        str(getattr(model, 'training_precision', 'fp32')) == 'bf16'
        and device.type == 'cuda'
    )
    for batch_index, batch in enumerate(loader, start=1):
        eeg = batch['eeg'].to(device, non_blocking=True)
        mask = batch['channel_mask'].to(device, non_blocking=True)
        batch_labels = batch['label'].to(device, non_blocking=True).float()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits = forward_clip(model, eeg, mask).reshape(-1)
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(min=-20.0, max=20.0)
        batch_scores = torch.sigmoid(logits)
        batch_size = batch_labels.numel()
        running_loss += float(
            F.binary_cross_entropy_with_logits(
                logits, batch_labels, reduction='sum'
            ).cpu()
        )
        seen += batch_size
        labels.append(batch_labels.long().cpu().numpy().astype(np.int64, copy=False))
        scores.append(batch_scores.cpu().numpy().astype(np.float64, copy=False))
        progress.update(batch_index, running_loss / max(seen, 1))
    progress.close()
    if not labels:
        raise ValueError('Validation loader is empty')
    return np.concatenate(labels), np.concatenate(scores), running_loss / seen


def fit_clip_stage(
    model: torch.nn.Module,
    train_loader,
    dev_loader,
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
    if hasattr(model, 'head') and hasattr(model.head, 'residual_enabled'):
        model.head.residual_enabled = stage != 'source'

    ema_decay = float(getattr(model, 'training_ema_decay', 0.0))
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError('Training EMA decay must be in [0, 1)')
    parameter_names = set(dict(model.named_parameters()))
    ema_state = _clone_state(model) if ema_decay > 0.0 else None
    ema_parameter_names = (
        [
            name for name in parameter_names
            if ema_state is not None and ema_state[name].is_floating_point()
        ]
        if ema_state is not None else []
    )
    ema_buffer_names = (
        [name for name in ema_state if name not in parameter_names]
        if ema_state is not None else []
    )
    ema_updates = 0
    logger.info(
        'stage=%s epoch_progress=enabled ema_enabled=%s ema_decay=%.8f',
        stage,
        ema_state is not None,
        ema_decay,
    )

    batches_per_epoch = len(train_loader)
    if batches_per_epoch <= 0:
        raise ValueError('Training loader is empty')
    total_steps = epochs * batches_per_epoch
    warmup_ratio = float(getattr(model, 'training_warmup_ratio', 0.0))
    minimum_lr_ratio = float(getattr(model, 'training_min_lr_ratio', 1.0))
    base_learning_rates = [float(group['lr']) for group in optimizer.param_groups]
    precision = str(getattr(model, 'training_precision', 'fp32'))
    use_bf16 = precision == 'bf16' and device.type == 'cuda'
    progress_update_interval = max(
        1, int(getattr(model, 'progress_update_interval', 1))
    )
    global_step = 0
    stage_started_at = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.training_progress = (epoch - 1) / max(epochs, 1)
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            for parameter in unfreeze_parameters:
                parameter.requires_grad = True
            logger.info(
                'stage=%s phase=head_plus_last_feature_block unfreeze_epoch=%d',
                stage,
                epoch,
            )
        model.train()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_train_started_at = time.perf_counter()
        running_loss_tensor = torch.zeros((), device=device, dtype=torch.float32)
        running_term_tensors: dict[str, torch.Tensor] = {}
        seen = 0
        train_progress = _EpochProgress(
            batches_per_epoch,
            f'Train epoch {epoch}/{epochs}',
        )
        skipped_nonfinite_batches = 0
        maximum_nonfinite_batches = max(3, batches_per_epoch // 10)
        for batch_index, batch in enumerate(train_loader, start=1):
            model.training_progress = global_step / max(total_steps - 1, 1)
            if warmup_ratio > 0.0 or minimum_lr_ratio < 1.0:
                step_fraction = global_step / max(total_steps - 1, 1)
                if step_fraction < warmup_ratio:
                    factor = (global_step + 1) / max(
                        int(math.ceil(total_steps * warmup_ratio)), 1
                    )
                else:
                    cosine_progress = (
                        (step_fraction - warmup_ratio)
                        / max(1.0 - warmup_ratio, 1e-8)
                    )
                    factor = minimum_lr_ratio + (
                        1.0 - minimum_lr_ratio
                    ) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
                for group, base_lr in zip(optimizer.param_groups, base_learning_rates):
                    group['lr'] = base_lr * factor
            eeg = batch['eeg'].to(device, non_blocking=True)
            mask = batch['channel_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
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
                skipped_nonfinite_batches += 1
                context = _loss_failure_context(
                    stage, epoch, epochs, batch_index, logits, labels, loss_terms
                )
                logger.warning('%s nonfinite_loss_skipped %s', stage, context)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if skipped_nonfinite_batches > maximum_nonfinite_batches:
                    raise ValueError(
                        f'{stage} exceeded nonfinite loss skip limit: {context}'
                    )
                if (
                    batch_index % progress_update_interval == 0
                    or batch_index == batches_per_epoch
                ):
                    train_progress.update(
                        batch_index,
                        loss=float(running_loss_tensor.cpu()) / max(seen, 1)
                        if seen > 0 else None,
                    )
                continue
            loss.backward()
            repaired_gradients = _sanitize_nonfinite_gradients(model)
            if repaired_gradients:
                logger.warning(
                    '%s nonfinite_gradients_repaired count=%d first=%s',
                    stage,
                    len(repaired_gradients),
                    repaired_gradients[:8],
                )
            if max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm, error_if_nonfinite=False
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    logger.warning(
                        '%s nonfinite_grad_norm_step_skipped epoch=%d batch=%d',
                        stage,
                        epoch,
                        batch_index,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue
            optimizer.step()
            repaired_parameters = _sanitize_nonfinite_parameters(model)
            if repaired_parameters:
                logger.warning(
                    '%s nonfinite_parameters_repaired count=%d first=%s',
                    stage,
                    len(repaired_parameters),
                    repaired_parameters[:8],
                )

            if ema_state is not None:
                with torch.no_grad():
                    ema_updates += 1
                    effective_decay = min(
                        ema_decay,
                        (1.0 + ema_updates) / (10.0 + ema_updates),
                    )
                    live_state = model.state_dict()
                    torch._foreach_lerp_(
                        [ema_state[name] for name in ema_parameter_names],
                        [live_state[name].detach() for name in ema_parameter_names],
                        1.0 - effective_decay,
                    )
                    for name in ema_buffer_names:
                        ema_state[name].copy_(live_state[name].detach())

            batch_size = labels.numel()
            running_loss_tensor.add_(loss.detach().float() * batch_size)
            for name, value in loss_terms.items():
                scalar_tensor = (
                    value.detach().float()
                    if torch.is_tensor(value)
                    else loss.new_tensor(float(value), dtype=torch.float32)
                )
                if name not in running_term_tensors:
                    running_term_tensors[name] = torch.zeros(
                        (), device=device, dtype=torch.float32
                    )
                if not torch.isfinite(scalar_tensor):
                    logger.warning(
                        '%s nonfinite_objective_term_ignored epoch=%d batch=%d term=%s',
                        stage,
                        epoch,
                        batch_index,
                        name,
                    )
                    continue
                running_term_tensors[name].add_(scalar_tensor * batch_size)
            seen += batch_size
            global_step += 1
            if batch_index % progress_update_interval == 0 or batch_index == batches_per_epoch:
                train_progress.update(
                    batch_index,
                    loss=float(running_loss_tensor.cpu()) / max(seen, 1),
                )
        train_progress.close()
        if seen <= 0:
            raise ValueError(f'{stage} epoch {epoch} did not complete a finite batch')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_train_started_at
        train_batches_per_second = batches_per_epoch / max(train_seconds, 1e-8)
        gpu_peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == 'cuda'
            else 0.0
        )

        raw_state = None
        if ema_state is not None:
            raw_state = _clone_state(model)
            model.load_state_dict(ema_state, strict=True)
        try:
            dev_labels, dev_scores, val_loss = _collect_validation_scores(
                model,
                dev_loader,
                device,
                forward_clip,
                f'Validation epoch {epoch}/{epochs}',
            )
        finally:
            if raw_state is not None:
                model.load_state_dict(raw_state, strict=True)
        if np.unique(dev_labels).size < 2:
            raise ValueError(f'{stage} dev split must contain both classes')

        val_auroc = float(roc_auc_score(dev_labels, dev_scores))
        running_loss = float(running_loss_tensor.cpu())
        running_terms = {
            name: float(value.cpu())
            for name, value in running_term_tensors.items()
        }
        non_finite_terms = [
            name for name, value in running_terms.items() if not math.isfinite(value)
        ]
        if non_finite_terms:
            raise ValueError(
                f'{stage} training objective terms are not finite: {non_finite_terms}'
            )
        train_loss = running_loss / max(seen, 1)
        learning_rate = float(optimizer.param_groups[0]['lr'])
        averaged_terms = {
            name: value / max(seen, 1)
            for name, value in sorted(running_terms.items())
        }
        epoch_row: dict[str, float | int] = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auroc': val_auroc,
            'learning_rate': learning_rate,
            'train_seconds': train_seconds,
            'train_batches_per_second': train_batches_per_second,
            'gpu_peak_memory_gb': gpu_peak_memory_gb,
        }
        epoch_row.update(averaged_terms)
        rows.append(epoch_row)
        pd.DataFrame(rows).to_csv(history_path, index=False)
        metric_line = (
            f'{stage} epoch {epoch}/{epochs} metrics '
            f'train_loss={train_loss:.6f} val_loss={val_loss:.6f} '
            f'val_auroc={val_auroc:.6f} lr={learning_rate:.8g} '
            f'train_batch_per_s={train_batches_per_second:.3f} '
            f'gpu_peak_gb={gpu_peak_memory_gb:.3f}'
        )
        print(metric_line, flush=True)
        _write_training_summary(
            epoch,
            epochs,
            time.perf_counter() - stage_started_at,
            train_loss,
            val_auroc,
        )
        _log_file_only(logger, metric_line)
        if averaged_terms:
            _log_file_only(
                logger,
                'stage=%s epoch=%d objective_terms=%s',
                stage,
                epoch,
                json.dumps(averaged_terms, sort_keys=True),
            )

        selected_state = ema_state if ema_state is not None else model.state_dict()
        checkpoint = {
            'model': {
                name: value.detach().cpu().clone()
                for name, value in selected_state.items()
            },
            'raw_model': _clone_state(model, torch.device('cpu')),
            'epoch': epoch,
            'val_auroc': val_auroc,
            'ema_decay': ema_decay,
            'ema_updates': ema_updates,
            'validation_weights': 'ema' if ema_state is not None else 'raw',
        }
        torch.save(checkpoint, last_path)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(checkpoint, best_path)

        early_stopping_active = unfreeze_epoch is None or epoch >= unfreeze_epoch
        if early_stopping_active:
            if val_auroc > early_stop_reference + min_delta:
                early_stop_reference = val_auroc
                stale_epochs = 0
            else:
                stale_epochs += 1
        if scheduler is not None and warmup_ratio == 0.0 and minimum_lr_ratio == 1.0:
            scheduler.step()
        if early_stopping_active and stale_epochs >= patience:
            break

    if not best_path.exists():
        raise RuntimeError(f'{stage} did not create a best checkpoint')
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(best['model'], strict=True)
    if stale_epochs >= patience:
        logger.info(
            'stage=%s early_stop_epoch=%d patience=%d min_delta=%.8f',
            stage,
            int(rows[-1]['epoch']),
            patience,
            min_delta,
        )
    return best_path


def install_river_training_runtime() -> None:
    import eeg_benchmark.engine as runtime

    runtime.fit_clip_stage = fit_clip_stage
