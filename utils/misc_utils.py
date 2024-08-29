# misc_utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Sets the seed for reproducibility.
    Args:
    - seed (int): Seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_directory(dir_path):
    """
    Creates a directory if it does not exist.
    Args:
    - dir_path (str): Path of the directory to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_model(model, path):
    """
    Saves the model to the specified path.
    Args:
    - model: The model to save.
    - path (str): The path to save the model to.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    Loads the model from the specified path.
    Args:
    - model: The model object to load the state into.
    - path (str): The path from which to load the model.

    Returns:
    - model: The model with loaded weights.
    """
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model
