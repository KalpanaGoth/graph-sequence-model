# Import necessary libraries
import numpy as np
import networkx as nx
import time
from tqdm import tqdm
import pickle
from node2vec import Node2Vec

# Function to initialize node embeddings using Node2Vec
def initialize_node_embeddings(graph):
    # Create a Node2Vec model
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    
    # Fit model and generate embeddings
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Assign embeddings to nodes
    embeddings = {node: model.wv[node] for node in graph.nodes()}
    
    return embeddings

# Function to perform graph-based translation
def graph_based_translation(graph, embeddings, source_node, top_k=1):
    # Initialize node scores with 0, except the source node
    scores = {node: 0.0 for node in graph.nodes}
    scores[source_node] = 1.0  # Start with source node

    # Start timer
    start_time = time.time()

    # Propagate scores using the graph's edges and weights with a progress bar
    propagation_steps = 3
    for _ in tqdm(range(propagation_steps), desc="Propagation Steps"):
        new_scores = scores.copy()
        for node in graph.nodes:
            for neighbor in graph.neighbors(node):
                # Propagate score using edge weight
                new_scores[neighbor] += scores[node] * graph[node][neighbor].get('weight', 1.0)
        scores = new_scores

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Message Passing Completed in {elapsed_time:.2f} seconds.")

    # Extract Spanish nodes
    spa_nodes = [node for node in graph.nodes if graph.nodes[node]['lang'] == 'SPA']
    
    # Rank Spanish nodes by score
    ranked_spa_nodes = sorted(spa_nodes, key=lambda n: scores[n], reverse=True)
    
    # Select top-k translations
    top_translations = ranked_spa_nodes[:top_k]
    
    return top_translations

# Load the graph from the pickle file
graph_path = 'output/book_translation_graph.gpickle'
with open(graph_path, 'rb') as f:
    G = pickle.load(f)

# Initialize node embeddings
embeddings = initialize_node_embeddings(G)

# Translate from a specific English node (e.g., 'Go.')
translations = graph_based_translation(G, embeddings, 'Go.', top_k=3)

# Display the translations
print(f"Translations for 'Go.': {translations}")
