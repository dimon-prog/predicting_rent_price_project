import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(-1)
