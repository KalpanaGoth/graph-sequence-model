# test_model.py
import unittest
import torch
from models.graph_model import GraphBasedModel

class TestGraphModel(unittest.TestCase):
    def setUp(self):
        # Setup before each test
        self.model = GraphBasedModel(input_dim=10, hidden_dim=64, output_dim=1)
        self.sample_input = torch.randn(5, 10)  # 5 samples with 10 features each

    def test_model_output_shape(self):
        # Test that the model output has the correct shape
        output = self.model(self.sample_input, None)  # Dummy edge input
        self.assertEqual(output.shape, (5, 1))

    def test_model_forward_pass(self):
        # Test that the forward pass does not throw an error
        try:
            self.model(self.sample_input, None)
        except Exception as e:
            self.fail(f"Model forward pass failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()
