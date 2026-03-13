import torch
import torch.nn as nn
import math

class Transformer_Reg_Attn(nn.Module):
    def __init__(self, input_size=8, hidden_size=256, nhead=8, dim_feedforward=1024, num_layers=2, num_out=1, dropout=0.3):
        super().__init__()
        d_model = hidden_size
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_out)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Input x: (batch_size, 450, 8)
        
        # Input Projection
        x = self.input_projection(x)  # (batch_size, 450, 256)
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(x)  # (batch_size, 450, 256)
        
        # Take the representation of the first token
        context = transformer_out[:, 0, :]  # (batch_size, 256)
        
        # Layer normalization
        normed = self.layer_norm(context)  # (batch_size, 256)
        
        # Dropout
        dropped = self.dropout(normed)  # (batch_size, 256)
        
        # Final fully connected layer
        return self.fc(dropped)  # (batch_size, 2)

# Example usage:
# model = Transformer_Reg_Attn(input_size=8, d_model=256, nhead=8, num_layers=2, num_out=2, dropout=0.3)
# x = torch.randn(32, 450, 8)  # (batch_size, 450, 8)
# output = model(x)  # (batch_size, 2)

# Rotary Position Embedding (RoPE) Implementation

def apply_rotary_pos_emb(q, k, sin, cos):
    # Applies rotary position embedding (RoPE) to q and k
    # Shapes: q, k, sin, cos -> (batch, seq_len, nhead, head_dim)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def rotate_half(x):
    # Split last dim into two halves and rotate
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def get_rotary_emb(seq_len, dim, device):
    # Generate sin/cos position encodings for RoPE
    position = torch.arange(seq_len, device=device, dtype=torch.float32)
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    angles = torch.einsum('i,j->ij', position, freq)  # (seq_len, dim/2)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    sin = torch.stack((sin, sin), dim=-1).reshape(seq_len, dim)
    cos = torch.stack((cos, cos), dim=-1).reshape(seq_len, dim)
    return sin, cos


class MultiheadAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "Head dimension must be even for RoPE"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # Get RoPE embeddings
        sin, cos = get_rotary_emb(L, self.head_dim, x.device)
        sin, cos = sin.unsqueeze(0).unsqueeze(2), cos.unsqueeze(0).unsqueeze(2)  # (1, L, 1, head_dim)

        q, k = apply_rotary_pos_emb(q, k, sin, cos)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)  # (B, L, nhead, head_dim)
        attn_out = attn_out.reshape(B, L, D)
        return self.out_proj(attn_out)


class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttentionRoPE(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Transformer_Reg_Attn_RoPE(nn.Module):
    def __init__(self, input_size=8, hidden_size=256, nhead=8, num_layers=2, num_out=2, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerRoPE(d_model=hidden_size, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, num_out)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)

        context = x[:, 0, :]  # first token representation
        normed = self.layer_norm(context)
        dropped = self.dropout(normed)
        return self.fc(dropped)

# Example usage:
# model = Transformer_Reg_Attn_RoPE(input_size=8, hidden_size=256, nhead=8, num_layers=2, num_out=2, dropout=0.3)
# x = torch.randn(32, 450, 8)
# y = model(x)
# print(y.shape)  # (32, 2)