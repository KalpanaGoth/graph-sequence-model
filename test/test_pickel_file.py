import networkx as nx
import pickle
import os

# Example: Creating a simple graph as G
G = nx.DiGraph()

# Adding nodes and edges to G
G.add_node("Go.", lang='ENG')
G.add_node("Váyase.", lang='SPA')
G.add_edge("Go.", "Váyase.", weight=1.0)

# Ensure the output directory exists
os.makedirs('output', exist_ok=True)

# Save the graph to the pickle file
graph_path = 'output/book_translation_graph.gpickle'
with open(graph_path, 'wb') as f:
    pickle.dump(G, f)
