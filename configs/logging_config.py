import yaml
import os

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
