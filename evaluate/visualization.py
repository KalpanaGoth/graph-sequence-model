# visualization.py
import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(graph_data):
    """
    Plots a graph using the networkx library.
    Args:
    - graph_data: A tuple (nodes, edges) representing the graph structure.

    Returns:
    - None
    """
    G = nx.Graph()
    nodes, edges = graph_data
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

def plot_attention_distribution(attention_scores, labels):
    """
    Plots the attention distribution of the model.
    Args:
    - attention_scores: Attention scores from the model.
    - labels: Labels corresponding to the attention scores.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    plt.bar(labels, attention_scores)
    plt.xlabel('Tokens')
    plt.ylabel('Attention Score')
    plt.title('Attention Distribution')
    plt.show()

def plot_metrics(metrics_history):
    """
    Plots the training and validation metrics over epochs.
    Args:
    - metrics_history: A dictionary containing metric values for each epoch.

    Returns:
    - None
    """
    epochs = range(1, len(metrics_history['Loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_history['Loss'], label='Loss')
    plt.plot(epochs, metrics_history['Accuracy'], label='Accuracy')
    plt.plot(epochs, metrics_history['F1 Score'], label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training and Validation Metrics Over Epochs')
    plt.show()
