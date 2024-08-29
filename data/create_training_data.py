import pandas as pd

# Sample data for training
train_data = {
    'feature_1': [0.12, 0.34, 0.89, 0.45, 0.23, 0.56, 0.12, 0.98, 0.67, 0.34],
    'feature_2': [0.45, 0.67, 0.21, 0.12, 0.56, 0.43, 0.65, 0.34, 0.23, 0.87],
    'feature_3': [0.78, 0.56, 0.35, 0.98, 0.67, 0.32, 0.87, 0.12, 0.45, 0.56],
    'feature_4': [0.65, 0.12, 0.55, 0.34, 0.89, 0.78, 0.21, 0.56, 0.89, 0.21],
    'feature_5': [0.23, 0.78, 0.47, 0.68, 0.12, 0.45, 0.34, 0.78, 0.12, 0.45],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

# Sample data for validation
val_data = {
    'feature_1': [0.55, 0.32, 0.21, 0.12, 0.45],
    'feature_2': [0.67, 0.78, 0.65, 0.34, 0.23],
    'feature_3': [0.34, 0.56, 0.87, 0.67, 0.56],
    'feature_4': [0.23, 0.45, 0.43, 0.89, 0.78],
    'feature_5': [0.78, 0.12, 0.34, 0.45, 0.67],
    'label': [1, 0, 1, 0, 1]
}

# Sample data for testing
test_data = {
    'feature_1': [0.78, 0.23, 0.34, 0.67, 0.45],
    'feature_2': [0.12, 0.87, 0.65, 0.43, 0.34],
    'feature_3': [0.45, 0.21, 0.89, 0.56, 0.23],
    'feature_4': [0.34, 0.56, 0.12, 0.21, 0.65],
    'feature_5': [0.56, 0.34, 0.45, 0.78, 0.12],
    'label': [1, 0, 1, 0, 1]
}

# Convert to DataFrames
df_train = pd.DataFrame(train_data)
df_val = pd.DataFrame(val_data)
df_test = pd.DataFrame(test_data)

# Save as CSV files
df_train.to_csv('sample_datasets/train_data.csv', index=False)
df_val.to_csv('sample_datasets/val_data.csv', index=False)
df_test.to_csv('sample_datasets/test_data.csv', index=False)
