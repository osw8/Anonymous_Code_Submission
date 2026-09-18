







import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn

import math

import torch

from logging import getLogger

logger = getLogger()


CHANNEL_DICT = {k.upper():v for v,k in enumerate(
                     [      'FP1', 'FPZ', 'FP2', 
                        "AF7", 'AF3', 'AF4', "AF8", 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
                               'O1', 'OZ', 'O2', ])}



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    
    
    def norm_cdf(x):
        
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        
        
        
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        
        
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        
        
        tensor.erfinv_()

        
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_mask(mask, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """    
    B, N, C, D = x.shape
    if len(mask.shape)==2:
        mN, mC = mask.shape
        
        mask_keep = mask.reshape((1,mN*mC,1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
        masked_x = masked_x.contiguous().view((B,mN,mC,D))
    else:
        mN = mask.shape[0]
        
        mask_keep = mask.reshape((1,mN,1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
    return masked_x

def apply_mask_t(mask_t, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """    
    B, N, D = x.shape
    mN = mask_t.shape[0]
    
    mask_keep = mask_t.reshape((1,mN,1)).repeat((B, 1, D))
    masked_x = torch.gather(x, dim=1, index=mask_keep)
    return masked_x

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x


def exists(val):
    return val is not None



def rotate_half(x):
    
    
    x = x.reshape((*x.shape[:-1],x.shape[-1]//2, 2))
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    
    return x.flatten(-2)

def apply_rotary_emb(freqs, t, start_index = 0, scale = 1.):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim = -1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        theta = 10000,
        learned_freq = False,
        interpolate_factor = 1.
    ):
        super().__init__()
        
        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)), 
            requires_grad = learned_freq)
        
        

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        self.register_buffer('scale', None)
        
    def prepare_freqs(self, num_patches = (1, 8), device='cuda', dtype=torch.float, offset = 0):
        
        C, N = num_patches
        cache_key = f'freqs:{num_patches}'
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        seq_pos = torch.arange(N, device = device, dtype = dtype)
        seq_pos = seq_pos.repeat_interleave(repeats=C, dim=0) 
        seq_pos = (seq_pos + offset) / self.interpolate_factor
        
        freqs = self.freqs
        freqs = torch.outer(seq_pos.type(freqs.dtype), freqs) 
        freqs = freqs.repeat_interleave(repeats=2, dim=-1)    

        self.cache[cache_key] = freqs

        return freqs
    
    




class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.return_attention= return_attention

    def forward(self, x, freqs=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        if self.use_rope:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
                attn_maak = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_maak.masked_fill(torch.logical_not(attn_mask), -float('inf'))
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            else:
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            return attn_weight
        
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal)
        x = y.transpose(1, 2).contiguous().view(B, T, C) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        
        self.return_attention= return_attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_causal=is_causal, use_rope=use_rope, return_attention = return_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs=None):
        y = self.attn(self.norm1(x), freqs)
        if self.return_attention: return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1,patch_size), 
                              stride=(1, patch_size if patch_stride is None else patch_stride))
        
    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = self.proj(x).transpose(1,3) 
        return x
class PatchNormEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        
        assert img_size[1] % patch_size==0
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.unfold = torch.nn.Unfold(kernel_size=(1, patch_size), stride = (1, patch_stride if patch_stride is not None else patch_size))

        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        
        B,C,T = x.shape
        x = x.unsqueeze(1) 
        
        x = self.unfold(x)
        
        x = x.transpose(-1,-2)
        
        x = x.view(B, C, -1, self.patch_size).contiguous()
        x = x.transpose(1,2)
        
        
        
        x = torch.layer_norm(x, (self.patch_size,))
        
        
        
        x = self.proj(x) 
        
        return x
class EEGTransformerReconstructor(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        num_patches,
        patch_size=64,
        embed_num=1,
        use_pos_embed = False,
        use_inp_embed = True,
        embed_dim=768,
        reconstructor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_inp_embed = use_inp_embed
        self.use_pos_embed = use_pos_embed
        self.num_patches = num_patches
        
        if use_inp_embed:
            self.reconstructor_embed = nn.Linear(embed_dim, reconstructor_embed_dim, bias=True)
        
        if use_pos_embed:
            self.pos_embed           = nn.Parameter(torch.zeros(1, 1, embed_num, reconstructor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)
        
        self.mask_token          = nn.Parameter(torch.zeros(1, 1, reconstructor_embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        
        self.time_embed_dim = (reconstructor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), reconstructor_embed_dim)
        
        self.reconstructor_blocks = nn.ModuleList([
            Block(
                dim=reconstructor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.reconstructor_norm = norm_layer(reconstructor_embed_dim)
        self.reconstructor_proj = nn.Linear(reconstructor_embed_dim, patch_size, bias=True)
        
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.reconstructor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x, chan_ids=None, mask_x=None, mask_y=None):
        
        
        
        chan_ids = chan_ids.to(x).long()
        
        
        if self.use_inp_embed:
            x = self.reconstructor_embed(x)

        C, N        = self.num_patches
        B, mN, eN, D= x.shape
        
        
        
        chan_embed = self.chan_embed(chan_ids).unsqueeze(0) 
        
        
        if mask_x is not None:
            mask_x       = mask_x.to(x.device)
            mask_x       = torch.floor(mask_x[:,0] / C).long().to(x.device)    
            
            freqs_x      = self.time_embed.prepare_freqs((1, N), x.device, x.dtype)
            freqs_x      = freqs_x.contiguous().view((1,N,self.time_embed_dim))
            freqs_x      = apply_mask_t(mask_x, freqs_x)                                       
            freqs_x      = freqs_x.contiguous().view((mask_x.shape[0], 1, self.time_embed_dim)) 
            freqs_x      = freqs_x.repeat((1, eN, 1)).flatten(0,1)
            
        else:
            freqs_x      = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype) 
            
        
        if mask_y is not None:
            mask_y       = mask_y.to(x.device)
            
            
            N_y          = mask_y.shape[0]
            chan_embed   = chan_embed.repeat((1,N,1,1))
            chan_embed   = apply_mask(mask_y, chan_embed)
            
            freqs        = self.time_embed.prepare_freqs((C, N), x.device, x.dtype) 
            freqs_y      = freqs.contiguous().view((1, N, C, self.time_embed_dim))
            freqs_y      = apply_mask(mask_y, freqs_y)        
            freqs_y      = freqs_y.contiguous().view((N_y, self.time_embed_dim))
            
            y = self.mask_token.repeat((B, N_y, 1)) + chan_embed
            
            
            if self.use_pos_embed:
                x        = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)
                
            
            x           = x.flatten(1,2) 
            x           = torch.cat([x,y], dim=1)
            freqs_x     = torch.cat([freqs_x, freqs_y], dim=0).to(x)
            
            
            
            for blk in self.reconstructor_blocks:
                x = blk(x, freqs_x) 
                if blk.return_attention==True: return x
            
            
            x = x[:,-N_y:,:]      
            
            x = self.reconstructor_norm(x) 
                
            x = self.reconstructor_proj(x)
            
            return x
        
class EEGTransformerPredictor(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        embed_num=1,
        use_pos_embed = False,
        use_inp_embed = True,
        use_part_pred = False,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_part_pred = use_part_pred
        self.use_pos_embed = use_pos_embed
        self.use_inp_embed = use_inp_embed
        self.num_patches = num_patches
        self.embed_num = embed_num
        
        if use_inp_embed:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        if use_pos_embed:
            self.pos_embed   = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)
        
        self.mask_token      = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        
        self.time_embed_dim = (predictor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        
        
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, mask_x=None, mask_t=None):
        
        
        
        
        if self.use_part_pred:
            inp_x = x
            
        if self.use_inp_embed:
            x = self.predictor_embed(x)

        C, N        = self.num_patches
        B, mN, eN, D    = x.shape
        
        
        
        freqs = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype) 
        
        if mask_x is not None:
            mask_x       = mask_x
            mask_x       = torch.floor(mask_x[:,0] / C).long()
            
            if mask_t is None:
                mask_t       = torch.tensor(list(set(list(range(0,N))) - set(mask_x.tolist()))).long()
            
            N_y              = mask_t.shape[0]
            y                = self.mask_token.repeat((B, N_y, 1, 1))
            x                = torch.cat([x,y], dim=1)
            
            
            mask_id          = torch.concat([mask_x.to(x.device), mask_t.to(x.device)], dim=0)            
            x                = torch.index_select(x, dim=1, index=torch.argsort(mask_id))    
            
        if self.use_pos_embed:
            x                = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)
            
        B, N, eN, D    = x.shape
        x = x.flatten(1,2)
        
        
        for blk in self.predictor_blocks:
            x = blk(x, freqs) 
            if blk.return_attention==True: return x
        
        
        x = x.reshape((B, N, eN, D))
        
        x = self.predictor_norm(x) 
            
        x = self.predictor_proj(x)
        
        if self.use_part_pred and mask_x is not None:
            cmb_x = torch.index_select(x, dim=1, index=mask_t.to(x.device)) 
            cmb_x = torch.concat([inp_x, cmb_x], dim=1)
            cmb_x = torch.index_select(cmb_x, dim=1, index=torch.argsort(mask_id)) 
            return x, cmb_x
        return x

class EEGTransformer(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        img_size=(64,1000),
        patch_size=64,
        patch_stride=None,
        embed_dim=768,
        embed_num=1,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=PatchEmbed,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.embed_num = embed_num
        
        self.num_heads = num_heads
        
        
        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                is_causal=False, use_rope= False, return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.init_std = init_std
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
            
        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def prepare_chan_ids(self, channels):
        chan_ids = []
        for ch in channels:
            ch = ch.upper().strip('.')
            assert ch in CHANNEL_DICT
            chan_ids.append(CHANNEL_DICT[ch])
        return torch.tensor(chan_ids).unsqueeze_(0).long()
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        
        
        
        
        
        x = self.patch_embed(x) 
        B, N, C, D = x.shape
        
        assert N==self.num_patches[1] and C==self.num_patches[0], f"{N}=={self.num_patches[1]} and {C}=={self.num_patches[0]}"
        
        if chan_ids is None:
            chan_ids = torch.arange(0,C)     
        chan_ids = chan_ids.to(x)
        
        
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0) 
        
        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)
            B, N, C, D = x.shape
            
        
        x = x.flatten(0, 1) 
        
        
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x,summary_token], dim=1)  
        
        
        for i, blk in enumerate(self.blocks):
            x = blk(x) 
            if blk.return_attention==True: return x

        x = x[:, -summary_token.shape[1]:, :]
        
        if self.norm is not None:
            x = self.norm(x) 

        
        x = x.flatten(-2)
        x = x.reshape((B, N, -1))
        
            
        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)
        
        x = x.reshape((B, N, self.embed_num, -1))
        
        return x
    


if __name__=="__main__":
    m = PatchNormEmbed((3,100), 10, 2)
    print(m(1000*torch.randn((1,3,100))))
    exit()
    VIT_EMBED_DIMS = {
        'vit_little': 32*3,
        'vit_tiny': 192,
        'vit_small': 384,
        'vit_base': 768,
        'vit_large': 1024,
        'vit_huge': 1280,
        'vit_giant': 1408,
    }

    import random
    import os
    def seed_torch(seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    seed_torch(7)
    model = EEGTransformer(
        img_size=[2, 12],
        patch_size=2,
        embed_dim=4,
        depth=2,
        num_heads=1,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    x = torch.rand(1,2,12)
    out = model(x, mask_x=torch.tensor([[4,5],[0,1],[2,3]]))
    
    print(out.shape) 
    
    
    
    model2 = EEGTransformerPredictor(
        num_patches=model.num_patches,
        embed_dim=4,
        predictor_embed_dim=4,
        depth=2,
        num_heads=1,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    
    
    
    
    
    a = model2(out, mask_x=torch.tensor([[4,5],[0,1],[2,3]]), mask_t=None)
    
    model3 = EEGTransformerReconstructor(
        num_patches=model.num_patches,
        patch_size=2,
        embed_dim=4,
        reconstructor_embed_dim=4,
        depth=2,
        num_heads=1,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    b = model3(a, mask_y=torch.tensor([6,7,8,9]))
    print(b) 