from __future__ import annotations

import torch
import torch.nn as nn

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, params, channel_count: int) -> None:
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=10,
            n_layer=12,
            nhead=8,
        )
        if params.use_pretrained_weights:
            checkpoint = torch.load(params.foundation_dir, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(checkpoint, strict=True)
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Linear(200, 1)
        self.channel_count = channel_count

    def reset_head(self) -> None:
        self.classifier.reset_parameters()

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected batch,view,channel,patch,point input, got {tuple(x.shape)}")
        batch, views, channels, patches, points = x.shape
        if channel_mask is None:
            common_indices = torch.arange(channels, device=x.device)
        elif channel_mask.shape[0] > 0 and torch.equal(
            channel_mask,
            channel_mask[:1].expand_as(channel_mask),
        ):
            common_indices = torch.nonzero(
                channel_mask[0], as_tuple=False
            ).reshape(-1)
        else:
            common_indices = None
        if common_indices is not None:
            if common_indices.numel() == 0:
                raise ValueError('CBraMod batch has no available channels')
            selected = x[:, :, common_indices].reshape(
                batch * views, common_indices.numel(), patches, points
            )
            features = self.backbone(selected).reshape(
                batch, views, common_indices.numel(), patches, -1
            )
            pooled = features.mean(dim=(1, 2, 3))
            return self.classifier(pooled).reshape(batch)
        if channel_mask is None:
            raise RuntimeError('CBraMod reached an invalid missing-mask fallback')

        unique_masks, group_ids = torch.unique(
            channel_mask,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        grouped_indices = []
        grouped_logits = []
        for group_index, group_mask in enumerate(unique_masks):
            sample_indices = torch.nonzero(
                group_ids == group_index,
                as_tuple=False,
            ).reshape(-1)
            channel_indices = torch.nonzero(
                group_mask,
                as_tuple=False,
            ).reshape(-1)
            if channel_indices.numel() == 0:
                raise ValueError('CBraMod sample group has no available channels')
            selected = x.index_select(0, sample_indices).index_select(
                2,
                channel_indices,
            )
            group_size = int(sample_indices.numel())
            features = self.backbone(
                selected.reshape(
                    group_size * views,
                    channel_indices.numel(),
                    patches,
                    points,
                )
            ).reshape(
                group_size,
                views,
                channel_indices.numel(),
                patches,
                -1,
            )
            pooled = features.mean(dim=(1, 2, 3))
            grouped_indices.append(sample_indices)
            grouped_logits.append(self.classifier(pooled).reshape(group_size))

        concatenated_indices = torch.cat(grouped_indices)
        concatenated_logits = torch.cat(grouped_logits)
        restore_order = torch.argsort(concatenated_indices)
        return concatenated_logits.index_select(0, restore_order)
