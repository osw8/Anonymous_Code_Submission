










from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
import timm.models.vision_transformer
import math
import torch.nn.functional as F

try:
    
    from torch.amp import autocast as _autocast_new
    autocast_amp = partial(_autocast_new, device_type="cuda")  
    _HAS_TORCH_AMP = True
except Exception:
    
    from torch.cuda.amp import autocast as autocast_amp
    _HAS_TORCH_AMP = False
    
HBN_MODEL_CHANIDX = [142,39,13,54,143,14,144,145,60,146,25,147,18,112,148,100,149,150,42,151,152,6,86,71,153,37,72,49,70,0,154,155,133,122,156,130,85,45,157,158,20,84,159,134,65,111,51,160,161,162,90,74,163,164,119,165,135,41,166,99,167,1,168,24,114,169,102,170,171,95,172,63,173,174,5,175,58,176,177,178,179,180,103,181,117,182,46,183,184,129,185,116,62,186,29,21,23,52,187,137,188,16,127,2,10,189,190,68,191,192,75,34,193,136,194,22,19,195,196,197,87,118,3,11,198,199,200,201,110]

class EEGClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, mode="token",
                 num_tokens=None, dropout=0.1,
                 bn_eps=1e-5, bn_momentum=0.1,
                 num_special_tokens: int = 1):  
        super().__init__()
        assert mode in {"token", "avg", "all_cnn", "all_simple"}
        print(f"USING: {mode} strategy for classification")
        self.mode = mode
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_special_tokens = int(num_special_tokens)  
        print(f"special tokens: {num_special_tokens}")
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        print(f"head drop out = {dropout}")
        if mode in {"token", "avg"}:
            self.norm = nn.LayerNorm(embed_dim)
            self.final = nn.Linear(embed_dim, num_classes)

        elif mode == "all_cnn":
            assert num_tokens is not None and num_tokens > 0, \
                "num_tokens must be provided for mode='all_cnn'."
            self.num_tokens = int(num_tokens)
            self.per_token_simple = nn.Sequential(
                nn.Conv1d(self.num_tokens+self.num_special_tokens, 256, kernel_size=1),
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Conv1d(256, 128, kernel_size=1),
            )
            self.final_simple = nn.Linear(128 * 512, num_classes)

        elif mode == "all_simple":
            assert num_tokens is not None and num_tokens > 0, \
                "num_tokens must be provided for mode='all_simple'."
            self.num_tokens = int(num_tokens)
            self.per_token_simple = nn.Linear(embed_dim, 64)
            self.final_simple = nn.Linear(self.num_tokens * 64, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L, D = tokens.shape

        if self.mode == "token":
            
            x = self.norm(tokens[:, 0, :])
            x = self.dropout(x)
            return self.final(x)

        
        start = self.num_special_tokens

        if self.mode == "avg":
            x = self.norm(tokens[:, start:, :]).mean(dim=1)
            x = self.dropout(x)
            return self.final(x)

        if self.mode == "all_cnn":
            x = tokens[:, :, :]          
            x = self.per_token_simple(x)      
            x = x.flatten(1)
            x = self.dropout(x)
            return self.final_simple(x)

        if self.mode == "all_simple":
            x = tokens[:, start:, :]          
            x = self.per_token_simple(x)      
            x = x.flatten(1)
            x = self.dropout(x)
            return self.final_simple(x)


        
class PatchEmbedEEG(nn.Module):
    def __init__(self, patch_size=32, embed_dim=256):
        super().__init__()
        self.p = patch_size
        self.embed_dim = embed_dim
        self.unfold = torch.nn.Unfold(kernel_size=(1,patch_size), stride=int(patch_size))
        self.proj = nn.Linear(self.p, self.embed_dim) 
        
    def forward(self, x):
        output = self.patchify_eeg(x)
        embd = self.proj(output)
        return embd

    def patchify_eeg(self,x):
        
        bs, c, L = x.shape
        x = x.unsqueeze(2)
        unfolded = self.unfold(x)
        bs, _, seq = unfolded.shape
        
        unfolded = torch.reshape(unfolded,(bs, c, self.p, seq))
        
        
        output = unfolded.permute(0, 3, 1, 2) 
        return output

class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim):
        super(ChannelPositionalEmbed, self).__init__()
        self.channel_transformation = nn.Embedding(256, embedding_dim)
        init.zeros_(self.channel_transformation.weight)
    def forward(self, channel_indices):
        channel_embeddings = self.channel_transformation(channel_indices)
        return channel_embeddings

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float())
        pe = torch.zeros(1, max_len, d_model)
        pe[0,:, 0::2] = torch.sin(position.float() * div_term)
        pe[0,:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer('pe', pe)
    
    def get_cls_token(self):
        return self.pe[0,0,:]
    
    def forward(self, seq_indices):
        batch_size, seq_len = seq_indices.shape
        pe_embeddings = self.pe[0, seq_indices.view(-1)].view(batch_size, seq_len, -1)
        return pe_embeddings


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self,
        global_pool: str = "avg",
        head_drop_out = 0.0,
        num_tokens: int | None = None,   
        num_tasks: int | None = None,    
        **kwargs
    ):
        super(VisionTransformer, self).__init__(**kwargs)

        
        if hasattr(self, "pos_embed") and isinstance(self.pos_embed, torch.nn.Parameter):
            self.pos_embed.requires_grad_(False)

        self.global_pool = global_pool
        self.input_sfreq = 100.0
        embed_dim = kwargs["embed_dim"]

        
        if hasattr(self, "head"):
            delattr(self, "head")
        if hasattr(self, "norm"):
            delattr(self, "norm")

        
        self.patch_embed = PatchEmbedEEG(
            patch_size=kwargs["patch_size"],
            embed_dim=embed_dim
        )
        self.enc_channel_emd = ChannelPositionalEmbed(embed_dim)
        self.enc_temporal_emd = TemporalPositionalEncoding(embed_dim, 512)

        
        self.register_buffer(
            "default_chan_idx",
            torch.tensor(HBN_MODEL_CHANIDX, dtype=torch.long),
            persistent=False
        )

        
        self.num_tasks = int(num_tasks) if num_tasks is not None else None
        if self.num_tasks is not None and self.num_tasks > 1:
            self.task_token_embed = nn.Embedding(self.num_tasks, embed_dim)
            
            nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
            self.num_special_tokens = 2  
        else:
            self.task_token_embed = None
            self.num_special_tokens = 1  

        
        self.cls_head = EEGClassificationHead(
            embed_dim=embed_dim,
            num_classes=self.num_classes,
            mode=self.global_pool,
            num_tokens=num_tokens,
            dropout=head_drop_out,
            num_special_tokens=self.num_special_tokens,  
        )
        
    def upsample_eeg_linear(self, x: torch.Tensor, fs_out: float, fs_in: float = 100.0):
        if x.dim() == 2:
            C, T = x.shape
            x_in = x.unsqueeze(0)
            squeeze_back = True
        elif x.dim() == 3:
            B, C, T = x.shape
            x_in = x
            squeeze_back = False
        else:
            raise ValueError(f"upsample_eeg_linear expects 2D or 3D tensor, got {x.shape}")

        if not x_in.is_floating_point():
            x_in = x_in.float()
        if not x_in.is_contiguous():
            x_in = x_in.contiguous()

        scale = float(fs_out) / float(fs_in)
        if scale <= 0:
            raise ValueError(f"Invalid fs_out/fs_in ratio: {fs_out}/{fs_in}")
        T_out = max(1, int(round(T * scale)))
        if T_out == T:
            return x_in.squeeze(0) if squeeze_back else x_in
        y = F.interpolate(x_in, size=T_out, mode="linear", align_corners=False)
        return y.squeeze(0) if squeeze_back else y
    
    def _forward_tokens(self, eeg: torch.Tensor, task_index: torch.Tensor | None) -> torch.Tensor:
        B, C, _ = eeg.shape
        if self.default_chan_idx.numel() != C:
            raise ValueError(
                f"Channel count mismatch: EEG has {C} channels but "
                f"HBN_MODEL_CHANIDX has {self.default_chan_idx.numel()} entries."
            )

        x = self.patch_embed(eeg)             
        B, Seq, Ch, D = x.shape
        N = Seq * Ch
        x = x.view(B, N, D)                   

        
        chan_idx = self.default_chan_idx.to(eeg.device)
        eeg_chan_indices = chan_idx.unsqueeze(0).unsqueeze(1).repeat(B, Seq, 1).view(B, N)
        seq_tensor = torch.arange(1, Seq + 1, device=eeg.device)
        eeg_seq_indices = seq_tensor.unsqueeze(0).unsqueeze(-1).repeat(B, 1, Ch).view(B, N)
        x = x + self.enc_temporal_emd(eeg_seq_indices) + self.enc_channel_emd(eeg_chan_indices)

        
        cls_token = self.cls_token + self.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(B, -1, -1)     

        if (self.task_token_embed is not None) and (task_index is not None):
            
            if task_index.dim() != 1 or task_index.shape[0] != B:
                raise ValueError(f"task_index must be shape [B], got {tuple(task_index.shape)}")
            task_tok = self.task_token_embed(task_index.to(eeg.device)).unsqueeze(1)  
            x = torch.cat((cls_tokens, task_tok, x), dim=1)  
        else:
            x = torch.cat((cls_tokens, x), dim=1)            

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x=None, task_index=None, **kwargs):
    
        if x is None and "input_ids" in kwargs:
            x = kwargs.pop("input_ids")
        
        eeg = self.upsample_eeg_linear(x, 128, self.input_sfreq)
        with autocast_amp(enabled=False):
            eegf = eeg.float()
            mean = eegf.mean(dim=2, keepdim=True)
            std  = eegf.std(dim=2, keepdim=True).clamp_min(1e-6)
            eegz = (eegf - mean) / std

        tokens = self._forward_tokens(eegz, task_index)   
        return self.cls_head(tokens)
        


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
