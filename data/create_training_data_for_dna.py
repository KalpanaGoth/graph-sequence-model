import pandas as pd

# Sample data for training
train_data = {
    'sequence': [
        'ATCGGCTAAGCTTAGGCTTAAGGCTTAGCCTGA',
        'TGCATGCAATGCCGATCGGATCGATTGCTAGCT',
        'CGTACGATGCTAGCTAGCATGCCGATCGGATAG',
        'GGATCCGGAATTCGGCCAATTGCGCGCTAGCAG',
        'ATGCGTACGTTAGCATGCTTAGGCATCGTAGCT',
        'CGATGCTAGCATCGGATCGTACGCTAGCGGCTA',
        'GGCATCGTAGGCTTAGCTAGCTAGGCGATGCAT',
        'TATGCTAGCTGCTAGCATGCTAGGATCGTACGA',
        'ACTGCGATAGCGGCTAGCTAGATCGATGCGCTA',
        'TAGCTAGCTAGGCTAGCGTAGCTAGATCGCGTA'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Sample labels for training data
}

# Sample data for validation
val_data = {
    'sequence': [
        'GGTACGTCAGTAGCTAGCGATCGTACGATCGTA',
        'ATCGTAGCTAGCTAGCTAGGCGATCGTACGTAG',
        'CGTAGCTAGCTGATCGTAGCTAGCATCGTAGCT',
        'TGCATGCGTAGCTAGCGTAGCTAGTGCATGCTA',
        'CGTACGTAGCTGACTGATCGTACGATCGTAGCT'
    ],
    'label': [1, 0, 1, 0, 1]  # Sample labels for validation data
}

# Sample data for testing
test_data = {
    'sequence': [
        'TGCAGCTAGCTAGCATCGTAGCTAGCTAGCTAG',
        'GCTAGCTAGCATCGATCGATCGTAGCTAGCTGA',
        'CTAGCTAGCTAGGATCGTAGCTAGCTGACTGAT',
        'TAGCTAGCATCGATCGTAGCTAGCTAGCTACGA',
        'CGTAGCTAGCATCGATGCTAGCTAGCTAGGCTA'
    ],
    'label': [1, 0, 1, 0, 1]  # Sample labels for testing data
}

# Convert to DataFrames
df_train = pd.DataFrame(train_data)
df_val = pd.DataFrame(val_data)
df_test = pd.DataFrame(test_data)

# Save as CSV files
df_train.to_csv('sample_datasets/train_data.csv', index=False)
df_val.to_csv('sample_datasets/val_data.csv', index=False)
df_test.to_csv('sample_datasets/test_data.csv', index=False)
