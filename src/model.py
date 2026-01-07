import torch
import torch.nn as nn

class CryptoLSTM(nn.Module):
    """
    Simple LSTM model for crypto price movement prediction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last time step
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
