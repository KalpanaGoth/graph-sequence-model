# optimizer.py
import torch

def get_optimizer(model, lr=0.001):
    """
    Returns the optimizer for the given model.
    Args:
    - model: The model to optimize.
    - lr (float): Learning rate for the optimizer.

    Returns:
    - optimizer: Optimizer for training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer
