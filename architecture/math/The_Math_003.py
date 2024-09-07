import matplotlib.pyplot as plt
import networkx as nx

# Create a sample dynamic graph
G = nx.DiGraph()
nodes = ['h1', 'h2', 'h3', 'h4']
edges = [('h1', 'h2'), ('h2', 'h3'), ('h3', 'h4'), ('h1', 'h4')]

# Add nodes and edges to the graph
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Assign initial weights to edges
weights = [2, 1, 3, 0.5]  # Example edge weights

# Draw the graph with weights
plt.figure(figsize=(8, 5))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'w={w}' for (u, v), w in zip(edges, weights)})

plt.title('Dynamic Graph Learning: Node Embeddings and Edge Weights')
plt.show()
