import yaml
import os
import torch
from train.train import train_model
from models.graph_model import GraphBasedModel
from train.optimizer import get_optimizer
from train.scheduler import get_scheduler
from utils.data_utils import get_data_loader
from evaluate.evaluate import evaluate_model
from torch.utils.data import DataLoader
from utils.data_utils import CustomDataset
from models.graph_model import GraphBasedModel

def main():
    # Load configuration and initialize model
    config = load_config()
    model = GraphBasedModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        use_attention=config['model']['use_attention']
    )
    # Initialize DataLoader, Optimizer, Scheduler, etc.
    # Train the model
    train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=num_epochs, patience=config['training']['patience'])
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

def load_config(config_path='configs/config.yaml'):
    """
    Loads the configuration file in YAML format.

    Args:
        config_path (str): Path to the configuration YAML file. Default is 'configs/config.yaml'.

    Returns:
        dict: Dictionary containing the configuration parameters loaded from the YAML file.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML file.
    """
    # Get the absolute path to the config file
    absolute_path = os.path.join(os.path.dirname(__file__), config_path)
    
    try:
        # Open and read the configuration file
        with open(absolute_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    except FileNotFoundError:
        print(f"Error: The configuration file at '{absolute_path}' was not found.")
        raise

    except yaml.YAMLError as e:
        print(f"Error: An error occurred while parsing the YAML file at '{absolute_path}'.\nDetails: {e}")
        raise

def main():
    """
    Main function to load the configuration and run experiments.
    """
    # Load the configuration
    config = load_config()
    print("Configuration loaded successfully:", config)

    # Extract values from the loaded configuration
    train_data_path = config['data']['train_data_path']
    val_data_path = config['data']['val_data_path']
    test_data_path = config['data']['test_data_path']
    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    
    # Initialize model, optimizer, scheduler, and dataloaders
    model = GraphBasedModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        use_attention=config['model']['use_attention']
    )

    # Set up optimizer, scheduler, and loss criterion
    optimizer = get_optimizer(model, lr=learning_rate)
    scheduler = get_scheduler(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])
    criterion = torch.nn.MSELoss()

    # Load training data using the paths from the configuration
    # dataloader = get_data_loader(train_data_path, batch_size=config['data']['batch_size'])
    dataloader = get_data_loader(train_data_path, batch_size=config['data']['batch_size'])

    # Train the model
    train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=num_epochs, patience=config['training']['patience'])

    # Evaluate the model
    metrics = evaluate_model(model, dataloader, criterion)
    print(f"Final Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    main()
