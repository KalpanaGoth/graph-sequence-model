import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        """
        Initialize the dataset by loading data from a CSV file.

        Args:
            data_path (str): Path to the CSV file.
        """
        self.data = pd.read_csv(data_path)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (nodes, edges, label) where nodes and edges are input features as tensors
            and label is the target value as a tensor.
        """
        # Ensure it extracts 10 features for nodes
        nodes = torch.tensor(self.data.iloc[idx, :-2].values, dtype=torch.float32)  # First 10 columns for nodes
        edges = torch.tensor(self.data.iloc[idx, -2], dtype=torch.float32)  # Second-to-last column for edges
        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)  # Last column for the label
        return nodes, edges, label

def get_data_loader(data_path, batch_size):
    """
    Create a DataLoader for the given dataset.

    Args:
        data_path (str): Path to the data file.
        batch_size (int): Batch size for data loading.

    Returns:
        DataLoader: DataLoader object for the dataset.
    """
    dataset = CustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader
