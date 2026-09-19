from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import pickle
import sys
from pathlib import Path

import torch


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
STEEG_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, STEEG_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from eeg_foundation_2025.utils.models_vit_eeg import vit_small_patch16
from eeg_benchmark.engine import ChannelUnionContract
from eeg_benchmark.engine import disable_stochastic_layers, load_shape_compatible_state
from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer


STEEG_CHANNEL_VOCABULARY = STEEG_ROOT / 'pretrain' / 'senloc_file' / 'sen_chan_idx.pkl'
STEEG_ADAPTER_SLOT_START = 202
FUNCTIONAL_CHANNEL_POLICIES = {
    'native_electrode_set_attention',
    'inductive_permutation_invariant_native_channel_statistics',
}


def bipolar_channel_adapter(contract: ChannelUnionContract) -> tuple[torch.Tensor, dict[int, tuple[int, int]]]:
    with STEEG_CHANNEL_VOCABULARY.open('rb') as handle:
        raw_mapping = pickle.load(handle)['channels_mapping']
    electrode_index = {str(name).upper(): int(index) for name, index in raw_mapping.items()}
    channel_indices = []
    endpoint_pairs: dict[int, tuple[int, int]] = {}
    for offset, channel in enumerate(contract.channel_keys):
        adapter_slot = STEEG_ADAPTER_SLOT_START + offset
        if adapter_slot >= 256:
            raise ValueError('STEEGFormer channel adapter exceeds its 256 channel slots')
        if contract.policy in FUNCTIONAL_CHANNEL_POLICIES:
            channel_indices.append(adapter_slot)
            continue
        endpoints = str(channel).upper().split('-')
        if len(endpoints) != 2 or any(endpoint not in electrode_index for endpoint in endpoints):
            raise ValueError(f'STEEGFormer cannot map bipolar channel endpoints: {channel}')
        channel_indices.append(adapter_slot)
        endpoint_pairs[adapter_slot] = (
            electrode_index[endpoints[0]], electrode_index[endpoints[1]]
        )
    return torch.tensor(channel_indices, dtype=torch.long), endpoint_pairs


class TaskSTEEGFormer(torch.nn.Module):
    def __init__(self, contract: ChannelUnionContract) -> None:
        super().__init__()
        self.backbone = vit_small_patch16(
            num_classes=1,
            global_pool='avg',
            head_drop_out=0.0,
            num_tasks=1,
            num_tokens=None,
            img_size=224,
            in_chans=3,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )
        self.backbone.input_sfreq = 128.0
        channel_indices, endpoint_pairs = bipolar_channel_adapter(contract)
        self.backbone.default_chan_idx = channel_indices
        self.bipolar_endpoint_pairs = endpoint_pairs

    @property
    def head(self):
        return self.backbone.cls_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(contract: ChannelUnionContract) -> TaskSTEEGFormer:
    if len(contract.channel_keys) > 256:
        raise ValueError('STEEGFormer supports at most 256 union channels')
    model = TaskSTEEGFormer(contract)
    disable_stochastic_layers(model)
    return model


def load_pretrained(model: TaskSTEEGFormer, path: Path, contract: ChannelUnionContract) -> dict[str, object]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = {f'backbone.{key}': value for key, value in checkpoint['model'].items()}
    report = load_shape_compatible_state(model, state, prefixes=('module.',))
    embedding = model.backbone.enc_channel_emd.channel_transformation.weight
    with torch.no_grad():
        for adapter_slot, endpoint_slots in model.bipolar_endpoint_pairs.items():
            source_slots = torch.tensor(
                endpoint_slots,
                dtype=torch.long,
                device=embedding.device,
            )
            embedding[adapter_slot].copy_(embedding.index_select(0, source_slots).mean(dim=0))
    if contract.policy in FUNCTIONAL_CHANNEL_POLICIES:
        with STEEG_CHANNEL_VOCABULARY.open('rb') as handle:
            raw_mapping = pickle.load(handle)['channels_mapping']
        pretrained_embedding = checkpoint['model'][
            'enc_channel_emd.channel_transformation.weight'
        ].to(device=embedding.device, dtype=embedding.dtype)
        pretrained_slots = torch.tensor(
            sorted({int(index) for index in raw_mapping.values()}),
            dtype=torch.long,
            device=embedding.device,
        )
        if int(pretrained_slots.max()) >= pretrained_embedding.shape[0]:
            raise ValueError(
                'STEEGFormer pretrained channel vocabulary exceeds its embedding table'
            )
        adapter_slots = model.backbone.default_chan_idx.to(embedding.device)
        with torch.no_grad():
            pretrained_mean = pretrained_embedding.index_select(
                0, pretrained_slots
            ).mean(dim=0, keepdim=True)
            embedding.index_copy_(
                0,
                adapter_slots,
                pretrained_mean.expand(adapter_slots.numel(), -1),
            )
        report['channel_position_policy'] = (
            'dedicated_functional_slots_initialized_from_pretrained_channel_mean_'
            'then_source_training'
        )
    else:
        report['channel_position_policy'] = (
            'dedicated_bipolar_slots_initialized_from_pretrained_endpoint_mean'
        )
    return report


def configure_transfer(model: TaskSTEEGFormer, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def budget_modules(model: TaskSTEEGFormer) -> tuple[torch.nn.Module, torch.nn.Module]:
    return model.head, model.backbone.blocks[-1]


def prepare_target_input(model: TaskSTEEGFormer, contract: ChannelUnionContract) -> None:
    union_index = {name: index for index, name in enumerate(contract.channel_keys)}
    source_indices = [union_index[name] for name in contract.source_channel_keys]
    target_only = [
        union_index[name]
        for name in contract.target_channel_keys
        if name not in set(contract.source_channel_keys)
    ]
    if not source_indices or not target_only:
        return
    embedding = model.backbone.enc_channel_emd.channel_transformation.weight
    with torch.no_grad():
        source_mean = embedding[source_indices].mean(dim=0, keepdim=True)
        embedding[target_only] = source_mean


def forward_clip(model: TaskSTEEGFormer, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    if channel_mask.shape[0] > 0 and torch.equal(
        channel_mask,
        channel_mask[:1].expand_as(channel_mask),
    ):
        real = torch.nonzero(channel_mask[0], as_tuple=False).reshape(-1)
        if real.numel() == 0:
            raise ValueError('STEEGFormer batch has no available channels')
        batch, views = eeg.shape[:2]
        original_indices = model.backbone.default_chan_idx
        model.backbone.default_chan_idx = original_indices[real]
        selected = eeg[:, :, real].reshape(batch * views, real.numel(), eeg.shape[-1])
        logits = model(selected).reshape(batch, views, -1)
        model.backbone.default_chan_idx = original_indices
        return logits.mean(dim=1).reshape(batch)
    outputs = []
    original_indices = model.backbone.default_chan_idx
    for sample, mask in zip(eeg, channel_mask):
        real = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if real.numel() == 0:
            raise ValueError('STEEGFormer sample has no available channels')
        model.backbone.default_chan_idx = original_indices[real]
        logits = model(sample[:, real]).reshape(sample.shape[0], -1)
        outputs.append(logits.mean())
    model.backbone.default_chan_idx = original_indices
    return torch.stack(outputs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='STEEGFormer Task1 60s clip transfer')
    parser.add_argument('--model', choices=['STEEGFormer'], default='STEEGFormer')
    return add_common_transfer_arguments(parser)


if __name__ == '__main__':
    run_torch_transfer(
        build_parser().parse_args(),
        model_name='STEEGFormer',
        input_spec_name='steegformer',
        model_factory=build_model,
        pretrained_loader=load_pretrained,
        transfer_configurator=configure_transfer,
        forward_clip=forward_clip,
        target_input_adapter=prepare_target_input,
        budget_module_resolver=budget_modules,
    )
