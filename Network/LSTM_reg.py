import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Reg(nn.Module):
    def __init__(self, input_size=13, hidden_size=256, num_layers=2, num_out=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_output = out[:, -1, :]
        normed = self.layer_norm(last_output)
        dropped = self.dropout(normed)
        return self.fc(dropped)


class LSTM_Reg_Attn(nn.Module):
    def __init__(self, input_size=13, hidden_size=256, num_layers=2, num_out=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        
        # Attention parameters
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, x):
        # Initial hidden & cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # LSTM outputs: (batch, seq_len, hidden)
        out, _ = self.lstm(x, (h0, c0))
        
        # --- Attention mechanism ---
        # Energy scores: (batch, seq_len, 1)
        attn_energy = torch.tanh(self.attn(out))
        attn_scores = self.v(attn_energy).squeeze(-1)  # (batch, seq_len)
        
        # Normalize to probabilities
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(out * attn_weights, dim=1)  # (batch, hidden)
        # --- End attention ---
        
        normed = self.layer_norm(context)
        dropped = self.dropout(normed)
        return self.fc(dropped)

# class LSTM_Reg(nn.Module):
#     def __init__(self, input_size=13, hidden_size=256, num_layers=2, num_out=2, dropout_ratio=0.2, fc_size=32):
#         super(LSTM_Reg, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # LSTM encoder
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout_ratio,
#             batch_first=True
#         )
        
#         self.out = nn.Linear(hidden_size, num_out)

#     def forward(self, x):
#         # x: [batch_size, seq_len, features] = [B, 2048, 13]
#         batch_size = x.size(0)
        
#         # Initialize hidden states
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

#         # LSTM output
#         out, _ = self.lstm(x, (h0, c0))
        
#         # Take the last timestep representation
#         last_output = out[:, -1, :]  # [B, hidden_size*2]

#         out = self.out(last_output)

#         return out


# Example Usage
if __name__ == "__main__":
    batch_size = 8
    seq_length = 2048
    input_features = 13

    model = LSTM_Reg(input_size=input_features, hidden_size=256, num_layers=2)
    x = torch.randn(batch_size, seq_length, input_features)
    out = model(x)
    
    print("Output shape:", out.shape)  # [B, num_classes]