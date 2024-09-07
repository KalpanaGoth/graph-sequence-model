import matplotlib.pyplot as plt
import networkx as nx

# Prepare a figure with multiple subplots for each step
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

# Step 1: Create Nodes for English and Spanish Phrases
G1 = nx.DiGraph()
G1.add_edges_from([("Go.", "Váyase."), ("Go.", "Ve."), ("Go.", "Vete."), ("Go.", "Vaya.")])
nx.draw(G1, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, ax=axes[0])
axes[0].set_title("Step 1: Create Nodes for English and Spanish Phrases")

# Step 2: Add Frequency-Based Weights
G2 = nx.DiGraph()
G2.add_edges_from([("Go.", "Váyase.", {'weight': 2}), ("Go.", "Ve.", {'weight': 1}), 
                   ("Go.", "Vete.", {'weight': 3}), ("Go.", "Vaya.", {'weight': 1})])
pos = nx.spring_layout(G2)
nx.draw(G2, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, ax=axes[1])
labels = nx.get_edge_attributes(G2, 'weight')
nx.draw_networkx_edge_labels(G2, pos, edge_labels=labels)
axes[1].set_title("Step 2: Add Frequency-Based Weights")

# Step 3: Load Existing Graph
nx.draw(G2, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, ax=axes[2])
axes[2].set_title("Step 3: Load Existing Graph")

# Step 4: Fit Model and Generate Embeddings
nx.draw(G2, pos, with_labels=True, node_color='lightgreen', node_size=2000, font_size=10, ax=axes[3])
axes[3].set_title("Step 4: Fit Model and Generate Embeddings")

# Step 5: Propagate Scores and Update Embeddings
G3 = G2.copy()
nx.draw(G3, pos, with_labels=True, node_color='orange', node_size=2000, font_size=10, ax=axes[4])
axes[4].set_title("Step 5: Propagate Scores and Update Embeddings")

# Step 6: Contextual Refinement of Graph State
G4 = G3.copy()
nx.draw(G4, pos, with_labels=True, node_color='pink', node_size=2000, font_size=10, ax=axes[5])
axes[5].set_title("Step 6: Contextual Refinement of Graph State")

plt.tight_layout()
plt.show()
