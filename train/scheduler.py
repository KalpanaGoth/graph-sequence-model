# scheduler.py
import torch

def get_scheduler(optimizer, step_size=10, gamma=0.1):
    """
    Returns a learning rate scheduler.
    Args:
    - optimizer: The optimizer to which the scheduler is applied.
    - step_size (int): Number of epochs after which to decrease the learning rate.
    - gamma (float): Multiplicative factor of learning rate decay.

    Returns:
    - scheduler: A scheduler that adjusts the learning rate.
    """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler
