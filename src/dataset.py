import torch
from torch.utils.data import Dataset

class CryptoDataset(Dataset):
    """
    PyTorch Dataset for LSTM: sequences of features and labels.
    """
    def __init__(self, data, feature_cols, target_col, seq_len=240):
        self.features = data[feature_cols].values
        self.labels = data[target_col].values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.labels[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
