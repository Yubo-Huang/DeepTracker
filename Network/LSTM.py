import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_MultiTask(nn.Module):
    def __init__(self, input_size=13, hidden_size=256, num_layers=2, num_classes1=19, num_classes2=6, dropout_ratio=0.2, fc_size=32):
        super(LSTM_MultiTask, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=dropout_ratio,
            batch_first=True
        )
        
        # Separate heads for classification and regression
        # self.classifier_head1 = nn.Linear(hidden_size, num_classes1)
        
        self.classifier_head2 = nn.Linear(hidden_size, num_classes2)

    def forward(self, x):
        # x: [batch_size, seq_len, features] = [B, 2048, 13]
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM output
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last timestep representation
        last_output = out[:, -1, :]  # [B, hidden_size*2]

        # Two parallel heads
        # class_out1 = self.classifier_head1(last_output)
        class_out2 = self.classifier_head2(last_output)

        return class_out2


# Example Usage
if __name__ == "__main__":
    batch_size = 8
    seq_length = 2048
    input_features = 13

    model = LSTM_MultiTask(input_size=input_features, hidden_size=256, num_layers=2)
    x = torch.randn(batch_size, seq_length, input_features)
    class_pred1, class_pred2 = model(x)
    
    print("Classification output shape:", class_pred1.shape)  # [B, num_classes]
    print("Regression output shape:", class_pred2.shape)        # [B, 1]