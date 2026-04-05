import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # 1. Calculate probabilities
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos

        # 2. Asymmetric Shifting (The "Margin")
        # This mutes easy negatives by pushing their probability towards 1 
        # (making the loss zero when clipped)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 3. Basic Binary Cross Entropy
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        
        # 4. Asymmetric Focusing
        # We apply different gamma exponents to the positive and negative parts
        if self.gamma_pos > 0:
            loss_pos *= (1 - xs_pos) ** self.gamma_pos
        if self.gamma_neg > 0:
            loss_neg *= (1 - xs_neg) ** self.gamma_neg

        return -(loss_pos + loss_neg).mean()


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


class SimpleTransformer(nn.Module):
    """
    Transformer classifier for tabular / aggregated time-series inputs.

    Takes a plain (B, T, F) feature tensor — no triplet encoding or
    sinusoidal time embedding — and otherwise mirrors TripletTransformer:

        Linear projection  →  Pre-LN SwiGLU TransformerEncoder  →
        Masked Global Average Pooling  →  Linear classifier head

    Args:
        num_features:    Number of input features per time-step F.
        d_model:         Transformer hidden dimension.
        nhead:           Number of attention heads.
        num_layers:      Number of SwiGLUEncoderLayer blocks.
        dim_feedforward: Inner dim of the SwiGLU FFN (default 128).
        dropout:         Dropout probability (default 0.1).
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        swiglu_layer: bool = False,
        loss: str = "BCEWithLogitsLoss",
        gamma_plus: float = 0.0,
        gamma_minus: float = 4.0,
        m: float = 0.05,
    ) -> None:
        super().__init__()

        # (B, T, F) -> (B, T, d_model)
        self.input_projection = nn.Linear(num_features, d_model)

        if swiglu_layer:
            encoder_layer = SwiGLUEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation='gelu'
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, 1)

        if loss == "AsymFocwithAPS":
            self.loss_fn = AsymmetricLoss(gamma_neg=gamma_minus, gamma_pos=gamma_plus, clip=m)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, F)  — batch of padded feature sequences.
            mask: (B, T) bool — True marks padding positions.

        Returns:
            Logits of shape (B, 1).
        """
        x = self.input_projection(x)                                     # (B, T, d_model)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)       # (B, T, d_model)
        x = self.final_norm(x)                                          # (B, T, d_model)

        # Masked Global Average Pooling
        inv_mask = (torch.ones_like(mask).float().unsqueeze(-1) - mask.unsqueeze(-1).float()) # (B, T, 1)
        sum_x    = torch.sum(x.to(torch.float32) * inv_mask, dim=1)     # (B, d_model)
        count_x  = torch.clamp(torch.sum(inv_mask, dim=1), min=1e-9)    # (B, 1)
        pooled   = (sum_x / count_x).to(x.dtype)                        # (B, d_model)

        return self.classifier(pooled)                                   # (B, 1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_time: int, T: float = 2880.0, tau: float = 100.0) -> None:
        super().__init__()
        self.d_time = d_time
        self.T = T
        self.tau = tau

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        I_{2k}(t) := sin( t / (T^(2k / tau)) ) [cite: 124]
        I_{2k+1}(t) := cos( t / (T^(2k / tau)) ) [cite: 124]
        """
        device = t.device
        half_dim = self.d_time // 2
        k = torch.arange(half_dim, device=device).float()
        denominators = torch.pow(self.T, (2 * k) / self.tau)
        
        args = t.unsqueeze(-1) / denominators.view(1, 1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TripletEmbedding(nn.Module):
    def __init__(self, d_model: int, d_time: int, num_categories: int = 41) -> None:
        super().__init__()
        self.time_enc = SinusoidalTimeEmbedding(d_time)
        # Input: d_time + 41 (one-hot) + 1 (value) -> Output: d_model
        self.num_categories = num_categories
        self.projection = nn.Linear(d_time + num_categories + 1, d_model)

    def forward(self, t: torch.Tensor, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_enc(t)
        z_onehot = F.one_hot(z, num_classes=self.num_categories).float()
        v_val = v.unsqueeze(-1)
        
        combined = torch.cat([t_emb, z_onehot, v_val], dim=-1)
        return self.projection(combined)


class TripletTransformer(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        num_layers: int, 
        d_time_emb: int, 
        dim_feedforward: int = 128, 
        dropout: float = 0.1,
        swiglu_layer: bool = False,
        loss: str = "BCEWithLogitsLoss",
        gamma_plus: float = 0.0,
        gamma_minus: float = 4.0,
        m: float = 0.05,
    ) -> None:
        super().__init__()
        self.embedding = TripletEmbedding(d_model, d_time_emb)
        
        if swiglu_layer:
            encoder_layer = SwiGLUEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation='gelu'
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        
        self.classifier = nn.Linear(d_model, 1)
        
        if loss == "AsymFocwithAPS":
            self.loss_fn = AsymmetricLoss(gamma_neg=gamma_minus, gamma_pos=gamma_plus, clip=m)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, t: torch.Tensor, z: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Generate initial embeddings from (t, z, v) triplets
        x = self.embedding(t, z, v)
        
        # Process sequence with self-attention 
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.final_norm(x)
        
        # Masked Global Average Pooling
        mask_expanded = mask.unsqueeze(-1).float() 
        inv_mask = 1.0 - mask_expanded 
        
        sum_x = torch.sum(x.to(torch.float32) * inv_mask, dim=1)
        count_x = torch.clamp(torch.sum(inv_mask, dim=1), min=1e-9)
        pooled_x = (sum_x / count_x).to(x.dtype)
        
        return self.classifier(pooled_x)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class SwiGLU(nn.Module):
    """
    SwiGLU Activation: (xW + b) * silu(xV + c)
    Standard implementation that chunks the input tensor into two halves.
    """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class MLPAggregator(nn.Module):
    def __init__(self, num_channels=37, embed_dim=768, n_hidden=1, hidden_dim=512, dropout=0.1):
        """
        A 'smarter' channel aggregator using multiple hidden layers and SwiGLU.
        
        Args:
            num_channels: 37 dynamic variables.
            embed_dim: 768 (for Chronos Bolt Small).
            n_hidden: Number of SwiGLU blocks.
            hidden_dim: The internal dimensionality.
            dropout: Dropout rate for regularization.
        """
        super(MLPAggregator, self).__init__()
        
        self.flatten = nn.Flatten()
        
        layers = []
        input_size = num_channels * embed_dim
        
        for i in range(n_hidden):
            # Project to 2x hidden_dim for SwiGLU 
            layers.append(nn.Linear(input_size, hidden_dim * 2))
            layers.append(SwiGLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            input_size = hidden_dim 
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # Final projection to a single logit for binary classification
        self.output_head = nn.Linear(input_size, 1)
        
    def forward(self, x, mask=None):
        # x shape: (Batch, 37, 768)
        x = self.flatten(x) # (Batch, 37 * 768)
        x = self.hidden_layers(x) # (Batch, hidden_dim)
        return self.output_head(x) # (Batch, 1)

    def compute_loss(self, logits, targets):
        """
        Computes Binary Cross Entropy with Logits.
        """
        return F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1))