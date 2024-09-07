import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Build the Graph
G = nx.DiGraph()
edges = [
    ("Go.", "Váyase."), ("Go.", "Ve."), 
    ("Go.", "Vete."), ("Go.", "Vaya.")
]
G.add_edges_from(edges)

# Set positions for visualization
pos = {
    "Go.": (0, 0), "Váyase.": (2, 2), 
    "Ve.": (2, 1), "Vete.": (2, 0), 
    "Vaya.": (2, -1)
}

# Step 2: Initialize Embeddings (Position nodes on a map)
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10, font_weight='bold')

# Step 3: Message Passing (Visualize connections and propagation)
for edge in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], width=2, alpha=0.6, edge_color='green')

# Step 4: Output Computation
plt.title('Graph-Based Translation Flow', fontsize=16)
plt.show()
