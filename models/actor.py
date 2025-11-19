import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Policy network for discrete actions.
    Outputs action probabilities (softmax) for the environment.
    """

    def __init__(self, obs_dim, action_size, hidden_size=128, num_layers=2, activation="ReLU"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_size = action_size

        # Build hidden layers
        layers = []
        input_size = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "elu":
                layers.append(nn.ELU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            input_size = hidden_size

        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.net(x)
        logits = self.output(x)
        probs = F.softmax(logits, dim=-1)
        return probs
