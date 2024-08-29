# message_passing.py
import torch
import torch.nn as nn

class MessagePassing(nn.Module):
    def __init__(self, hidden_dim):
        super(MessagePassing, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, nodes, edge_weights):
        """
        Implements the message passing mechanism by updating each node's state 
        based on its neighbors' states and the edge weights.
        """
        # Compute messages from neighbors
        messages = torch.matmul(edge_weights, nodes)
        
        # Update node states
        updated_nodes = nodes + messages
        
        return updated_nodes
