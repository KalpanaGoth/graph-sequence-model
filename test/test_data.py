# test_data.py
import unittest
from utils.data_utils import CustomDataset, get_data_loader
import os

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Setup before each test
        self.test_data_path = "sample_datasets/sample_texts.txt"
        self.batch_size = 2

    def test_custom_dataset(self):
        # Test that the CustomDataset loads correctly
        dataset = CustomDataset(self.test_data_path)
        self.assertGreater(len(dataset), 0)  # Check that the dataset is not empty

    def test_data_loader(self):
        # Test that DataLoader can load data in batches
        dataloader = get_data_loader(self.test_data_path, batch_size=self.batch_size)
        for batch in dataloader:
            self.assertEqual(len(batch), self.batch_size)  # Check if the batch size is correct
            break

if __name__ == "__main__":
    unittest.main()
