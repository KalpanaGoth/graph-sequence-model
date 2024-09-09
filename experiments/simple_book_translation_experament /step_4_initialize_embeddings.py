import networkx as nx
import pickle
from node2vec import Node2Vec

# Step 4 Load the graph Create a Node2Vec model and fit model and generate embeddings then assign embeddings to nodes

def initialize_node_embeddings(graph_path):
    # Load the graph
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    # Create a Node2Vec model
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

    # Fit model and generate embeddings
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Assign embeddings to nodes
    embeddings = {node: model.wv[node] for node in G.nodes()}

    return embeddings

# Example usage
graph_path = 'output/book_translation_graph.gpickle'
embeddings = initialize_node_embeddings(graph_path)

print(f"Initialized {len(embeddings)} node embeddings.")
