import pandas as pd

def load_data(file_path, file_type="csv", sheet_name=None):
    """
    Loads data from a given file path based on the file type.

    Args:
        file_path (str): Path to the file.
        file_type (str): Type of the file ('csv', 'json', 'excel'). Default is 'csv'.
        sheet_name (str): Sheet name for Excel files. Default is None.

    Returns:
        DataFrame: Loaded data in a pandas DataFrame.
    """
    if file_type == "csv":
        return pd.read_csv(file_path)
    elif file_type == "json":
        return pd.read_json(file_path)
    elif file_type == "excel":
        return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

if __name__ == "__main__":
    # Replace hardcoded paths with dynamic paths or parameters
    csv_data = load_data("sample_datasets/train_data.csv")
    json_data = load_data("data/sample.json", file_type="json")
    excel_data = load_data("data/sample.xlsx", file_type="excel", sheet_name="Sheet1")
