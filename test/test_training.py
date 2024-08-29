# test_training.py
import unittest
import torch
from train.train import train_model
from models.graph_model import GraphBasedModel
from train.optimizer import get_optimizer
from train.scheduler import get_scheduler

class TestTraining(unittest.TestCase):
    def setUp(self):
        # Setup before each test
        self.model = GraphBasedModel(input_dim=10, hidden_dim=64, output_dim=1)
        self.optimizer = get_optimizer(self.model, lr=0.001)
        self.scheduler = get_scheduler(self.optimizer)
        self.dummy_data = [(torch.randn(5, 10), torch.randint(0, 2, (5, 1))) for _ in range(10)]
    
    def test_optimizer_initialization(self):
        # Test that the optimizer initializes correctly
        self.assertIsNotNone(self.optimizer)

    def test_training_loop(self):
        # Test that the training loop runs without errors
        try:
            train_model(self.model, self.dummy_data, torch.nn.MSELoss(), self.optimizer, self.scheduler, num_epochs=1, patience=2)
        except Exception as e:
            self.fail(f"Training loop failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()
