import torch
import torch.nn as nn


class PricePredictor(nn.Module):
    def __init__(self, input_features: int):
        super(PricePredictor, self).__init__()
        self.network = nn.Sequential( nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
