










from functools import partial

import torch
import torch.nn as nn
import torch.nn.init as init
from timm.models.vision_transformer import PatchEmbed, Block
import math

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
        self.channel_transformation = nn.Embedding(145, embedding_dim)
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
        
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=32, max_eeg_chans=134,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.patch_len = patch_size
        
        
        self.patch_embed = PatchEmbedEEG(patch_size=patch_size, embed_dim=embed_dim) 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        
        self.enc_channel_emd = ChannelPositionalEmbed(embed_dim)
        self.enc_temporal_emd = TemporalPositionalEncoding(embed_dim,512)
        
        self.dec_channel_emd = ChannelPositionalEmbed(decoder_embed_dim)
        self.dec_temporal_emd = TemporalPositionalEncoding(decoder_embed_dim,512)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        

        
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True) 
        

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        
        
        
        

        
        

        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  
        
        
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_masking_demo(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  
        
        
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_keep, ids_restore
    
    def mask_use_ids_keep(self, x, ids_keep):
        N, L, D = x.shape  
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked
    
    def forward_encoder_demo(self, eeg, chan_idx, mask_ratio, ids_keep=None, mask=None, ids_restore=None):
        
        x = self.patch_embed(eeg)
        B, Seq, Ch_all, Dmodel = x.shape
        Seq_total = Seq*Ch_all

        x = x.view(B,Seq_total,Dmodel)
        
        
        eeg_chan_indices = chan_idx.unsqueeze(1).repeat(1,Seq,1)
        eeg_chan_indices = eeg_chan_indices.view(B,Seq_total)
        
        
        seq_tensor = torch.arange(1, Seq+1, device=eeg.device)
        eeg_seq_indices = seq_tensor.unsqueeze(0).unsqueeze(-1).repeat(B,1,Ch_all)
        eeg_seq_indices = eeg_seq_indices.view(B,Seq_total)
        
        
        tp_embd = self.enc_temporal_emd(eeg_seq_indices)
        
        ch_embd = self.enc_channel_emd(eeg_chan_indices)
        
        x = x + tp_embd + ch_embd
        if ids_keep is None:
            
            x, mask, ids_keep, ids_restore = self.random_masking_demo(x, mask_ratio)
        else:
            x = self.mask_use_ids_keep(x,ids_keep)

        
        cls_token = self.cls_token + self.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
 
        return x, mask, ids_restore, (B, Seq, Ch_all, Dmodel), ids_keep

        
    def forward_encoder(self, eeg, chan_idx, mask_ratio):
        
        x = self.patch_embed(eeg)
        B, Seq, Ch_all, Dmodel = x.shape
        Seq_total = Seq * Ch_all

        
        x = x.view(B, Seq_total, Dmodel)  

        
        
        if chan_idx.dim() == 1:
            chan_idx = chan_idx.unsqueeze(0).expand(B, -1)  

        
        ch_emb_small = self.enc_channel_emd(chan_idx)      

        
        
        
        
        ch_emb = (
            ch_emb_small
              .unsqueeze(1)                
              .repeat_interleave(Seq, dim=1)  
              .view(B, Seq_total, Dmodel)     
        )

        
        
        temp_idx = torch.arange(Seq, device=x.device, dtype=torch.long).unsqueeze(0)  
        temp_emb_small_2d = self.enc_temporal_emd(temp_idx)             
        temp_emb_small = temp_emb_small_2d.squeeze(0)                   

        
        
        
        
        temp_emb_flat = (
            temp_emb_small
              .unsqueeze(1)               
              .repeat_interleave(Ch_all, dim=1)  
              .view(Seq_total, Dmodel)         
        )

        
        tp_emb = temp_emb_flat.unsqueeze(0).expand(B, -1, -1)  

        
        
        
        
        x = x + tp_emb + ch_emb

        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        
        cls_token = self.cls_token + self.enc_temporal_emd.get_cls_token()
        
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  

        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        
        return x, mask, ids_restore, (B, Seq, Ch_all, Dmodel)

    def forward_decoder(self, x, chan_idx, eeg_patch_shape, ids_restore):
        B, Seq, Ch_all, DmodEnc = eeg_patch_shape
        Seq_total = Seq*Ch_all
        
        x = self.decoder_embed(x)

        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  
        
        
        
        
        eeg_chan_indices = chan_idx.unsqueeze(1).repeat(1,Seq,1)
        eeg_chan_indices = eeg_chan_indices.view(B,Seq_total)
        
        
        seq_tensor = torch.arange(1, Seq+1, device=x.device)
        eeg_seq_indices = seq_tensor.unsqueeze(0).unsqueeze(-1).repeat(B,1,Ch_all)
        eeg_seq_indices = eeg_seq_indices.view(B,Seq_total)
        
        
        
        tp_embd = self.dec_temporal_emd(eeg_seq_indices)
        
        ch_embd = self.dec_channel_emd(eeg_chan_indices)
        
        x_ = x_ + tp_embd + ch_embd
        cls_ = x[:, :1, :] + self.dec_temporal_emd.get_cls_token()
        
        x = torch.cat([cls_, x_], dim=1)  
        
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        
        x = self.decoder_pred(x)

        
        x = x[:, 1:, :]
        
        return x

    def forward_loss(self, eeg, pred, mask):
        """
        eeg: [N, Ch, DL]
        pred: [N, L, DL]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patch_embed.patchify_eeg(eeg)
        target = target.reshape(pred.shape)
        
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  

        loss = (loss * mask).sum() / mask.sum()  
        return loss
    
    def forward(self, eeg, chan_idx, mask_ratio=0.75):
        latent, mask, ids_restore, eeg_patch_shape = self.forward_encoder(eeg, chan_idx, mask_ratio)
        pred = self.forward_decoder(latent, chan_idx, eeg_patch_shape, ids_restore)  
        loss = self.forward_loss(eeg, pred, mask)
        return loss, pred, mask
    
    def forward_demo(self, eeg, chan_idx, mask_ratio=0.75, ids_keep=None, mask=None, ids_restore=None):
        latent, mask, ids_restore, eeg_patch_shape, ids_keep = self.forward_encoder_demo(eeg, chan_idx, mask_ratio, ids_keep, mask, ids_restore)
        pred = self.forward_decoder(latent, chan_idx, eeg_patch_shape, ids_restore)  
        loss = self.forward_loss(eeg, pred, mask)
        return loss, pred, mask, ids_keep, ids_restore

    def unpatchify_eeg(self, x, original_length, Seq, Ch):
        
        bs = x.shape[0]
        x = x.view(bs,Seq,Ch,16)
        fold = torch.nn.Fold(output_size=(1, original_length), kernel_size=(1, self.patch_len), stride=self.patch_len)
        output_permuted = x.permute(0, 2, 3, 1).reshape(bs, Ch*self.patch_len, Seq)  
        reconstructed = fold(output_permuted)  
        reconstructed = reconstructed.squeeze(2)  
        return reconstructed

def mae_vit_small_patch16_dec256d4b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=512, depth=8, num_heads=8,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



mae_vit_small_patch16 = mae_vit_small_patch16_dec256d4b 
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  
