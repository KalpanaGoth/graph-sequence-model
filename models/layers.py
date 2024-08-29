# layers.py
import torch
import torch.nn as nn

class GraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, nodes, edge_weights):
        """
        Forward pass for a graph layer, applying linear transformation and activation.
        """
        transformed_nodes = self.linear(nodes)
        activated_nodes = self.activation(transformed_nodes)
        return activated_nodes

class GraphNormalization(nn.Module):
    def __init__(self):
        super(GraphNormalization, self).__init__()

    def forward(self, nodes):
        """
        Applies normalization to node features.
        """
        normed_nodes = (nodes - nodes.mean(dim=0)) / nodes.std(dim=0)
        return normed_nodes
