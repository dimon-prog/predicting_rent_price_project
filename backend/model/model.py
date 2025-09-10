import torch
import torch.nn as nn


class PricePredictor(nn.Module):
    def __init__(self, input_features: int):
        super(PricePredictor, self).__init__()
        self.network = nn.Sequential( nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
