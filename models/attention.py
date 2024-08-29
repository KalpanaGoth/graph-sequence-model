# attention.py
import torch
import torch.nn as nn

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, nodes):
        """
        Computes attention scores for each node and applies them.
        """
        scores = self.attention_weights(nodes)
        attention_scores = torch.softmax(scores, dim=0)
        
        # Apply attention scores to nodes
        attended_nodes = nodes * attention_scores
        return attended_nodes
