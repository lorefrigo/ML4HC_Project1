import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer using Pre-LN and a SwiGLU FeedForward Network.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # SwiGLU components
        self.w1 = nn.Linear(d_model, dim_feedforward)
        self.w2 = nn.Linear(d_model, dim_feedforward)
        self.w3 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        # Pre-LN Self-Attention
        src_norm = self.norm1(src)
        attn_out, _ = self.self_attn(src_norm, src_norm, src_norm, key_padding_mask=src_key_padding_mask, need_weights=False)
        src = src + self.dropout(attn_out)
        
        # Pre-LN SwiGLU FFN
        src_norm = self.norm2(src)
        gate = F.silu(self.w1(src_norm))
        ffn_inner = gate * self.w2(src_norm)
        ffn_out = self.w3(self.dropout(ffn_inner))
        
        src = src + self.dropout(ffn_out)
        return src