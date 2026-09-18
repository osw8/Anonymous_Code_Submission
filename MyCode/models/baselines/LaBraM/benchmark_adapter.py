from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import sys
from pathlib import Path

import torch


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
LABRAM_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, LABRAM_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import modeling_finetune
from eeg_benchmark.engine import ChannelUnionContract
from eeg_benchmark.engine import disable_stochastic_layers, load_shape_compatible_state
from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer


LABRAM_STANDARD_1020 = (
    'FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', 'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9H', 'TTP7H', 'TPP9H', 'FTT10H', 'TPP8H', 'TPP10H',
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
)


def channel_position_indices(contract: ChannelUnionContract) -> tuple[torch.Tensor, torch.Tensor]:
    standard_index = {
        str(name).upper(): index + 1
        for index, name in enumerate(LABRAM_STANDARD_1020)
        if index + 1 <= 128
    }
    used = {0}
    output = [0]
    semantic = []
    unknown = []
    for name in contract.channel_keys:
        index = standard_index.get(name.upper())
        if index is None or index in used:
            output.append(-1)
            unknown.append(len(output) - 1)
            semantic.append(False)
        else:
            output.append(index)
            used.add(index)
            semantic.append(True)
    available = [index for index in range(1, 129) if index not in used]
    if len(unknown) > len(available):
        raise ValueError('LaBraM positional table cannot represent this channel union')
    for position, index in zip(unknown, available):
        output[position] = index
    return torch.tensor(output, dtype=torch.long), torch.tensor(semantic, dtype=torch.bool)


class TaskLaBraM(torch.nn.Module):
    def __init__(self, contract: ChannelUnionContract) -> None:
        super().__init__()
        self.backbone = modeling_finetune.labram_base_patch200_200(
            num_classes=1,
            EEG_size=1600,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            use_mean_pooling=True,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            init_scale=0.001,
            init_values=0.1,
            qkv_bias=False,
        )
        input_chans, semantic_mask = channel_position_indices(contract)
        self.register_buffer('input_chans', input_chans, persistent=True)
        self.register_buffer('semantic_channel_mask', semantic_mask, persistent=True)

    @property
    def head(self):
        return self.backbone.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, input_chans=self.input_chans)


def build_model(contract: ChannelUnionContract) -> TaskLaBraM:
    model = TaskLaBraM(contract)
    disable_stochastic_layers(model)
    return model


def load_pretrained(model: TaskLaBraM, path: Path, contract: ChannelUnionContract) -> dict[str, object]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = {
        f'backbone.{key[8:]}': value
        for key, value in checkpoint['model'].items()
        if key.startswith('student.')
    }
    report = load_shape_compatible_state(model, state, prefixes=('module.',))
    unknown = torch.nonzero(~model.semantic_channel_mask, as_tuple=False).reshape(-1)
    if unknown.numel() > 0:
        unipolar_index = {
            str(name).upper(): index + 1
            for index, name in enumerate(LABRAM_STANDARD_1020)
            if '-' not in str(name) and index + 1 <= 128
        }
        with torch.no_grad():
            mean_position = model.backbone.pos_embed[:, 1:].mean(dim=1, keepdim=True)
            for channel_index in unknown.tolist():
                slot = int(model.input_chans[channel_index + 1])
                endpoints = str(contract.channel_keys[channel_index]).upper().split('-')
                endpoint_slots = [unipolar_index[name] for name in endpoints if name in unipolar_index]
                if len(endpoint_slots) == 2:
                    source_slots = torch.tensor(
                        endpoint_slots,
                        dtype=torch.long,
                        device=model.backbone.pos_embed.device,
                    )
                    replacement = model.backbone.pos_embed.index_select(1, source_slots).mean(
                        dim=1, keepdim=True
                    )
                else:
                    replacement = mean_position
                model.backbone.pos_embed[:, slot:slot + 1].copy_(replacement)
    report['channel_position_policy'] = (
        'official_bipolar_slots_when_representable_other_bipolar_slots_use_'
        'pretrained_endpoint_position_mean'
    )
    return report


def configure_transfer(model: TaskLaBraM, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def budget_modules(model: TaskLaBraM) -> tuple[torch.nn.Module, torch.nn.Module]:
    return model.head, model.backbone.blocks[-1]


def prepare_target_input(model: TaskLaBraM, contract: ChannelUnionContract) -> None:
    union_index = {name: index for index, name in enumerate(contract.channel_keys)}
    source_indices = [union_index[name] for name in contract.source_channel_keys]
    source_set = set(contract.source_channel_keys)
    target_only_unknown = [
        union_index[name]
        for name in contract.target_channel_keys
        if name not in source_set and not bool(model.semantic_channel_mask[union_index[name]])
    ]
    if not source_indices or not target_only_unknown:
        return
    source_slots = model.input_chans[torch.tensor(source_indices, dtype=torch.long, device=model.input_chans.device) + 1]
    target_slots = model.input_chans[torch.tensor(target_only_unknown, dtype=torch.long, device=model.input_chans.device) + 1]
    with torch.no_grad():
        source_positions = model.backbone.pos_embed.index_select(1, source_slots)
        source_mean = source_positions.mean(dim=1, keepdim=True)
        replacement = source_mean.expand(-1, target_slots.numel(), -1).contiguous()
        model.backbone.pos_embed.index_copy_(1, target_slots, replacement)


def forward_clip(model: TaskLaBraM, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    if channel_mask.shape[0] > 0 and torch.equal(
        channel_mask,
        channel_mask[:1].expand_as(channel_mask),
    ):
        real = torch.nonzero(channel_mask[0], as_tuple=False).reshape(-1)
        if real.numel() == 0:
            raise ValueError('LaBraM batch has no available channels')
        batch, views = eeg.shape[:2]
        selected = eeg[:, :, real].reshape(batch * views, real.numel(), *eeg.shape[3:])
        input_chans = torch.cat((model.input_chans[:1], model.input_chans[real + 1]))
        logits = model.backbone(selected, input_chans=input_chans).reshape(batch, views, -1)
        return logits.mean(dim=1).reshape(batch)
    outputs = []
    for sample, mask in zip(eeg, channel_mask):
        real = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if real.numel() == 0:
            raise ValueError('LaBraM sample has no available channels')
        views = sample[:, real]
        input_chans = torch.cat((model.input_chans[:1], model.input_chans[real + 1]))
        logits = model.backbone(views, input_chans=input_chans).reshape(views.shape[0], -1)
        outputs.append(logits.mean())
    return torch.stack(outputs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='LaBraM Task1 60s clip transfer')
    parser.add_argument('--model', choices=['LaBraM'], default='LaBraM')
    return add_common_transfer_arguments(parser)


if __name__ == '__main__':
    run_torch_transfer(
        build_parser().parse_args(),
        model_name='LaBraM',
        input_spec_name='labram',
        model_factory=build_model,
        pretrained_loader=load_pretrained,
        transfer_configurator=configure_transfer,
        forward_clip=forward_clip,
        target_input_adapter=prepare_target_input,
        budget_module_resolver=budget_modules,
    )
