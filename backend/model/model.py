import torch
import torch.nn as nn


class PricePredictor(nn.Module):
    def __init__(self, input_features: int):
        super(PricePredictor, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_features, 128),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),

                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),

                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),

                                     nn.Linear(32, 1),
                                     nn.ReLU(),
                                     nn.Dropout(0.3)
                                     )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
