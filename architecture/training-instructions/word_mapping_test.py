import matplotlib.pyplot as plt
import networkx as nx

# Define the graph with the nodes and edges you have
G = nx.DiGraph()

# Adding nodes and edges based on your updated data
edges = [
    ("Go.", "Ve."), ("Go.", "Vete."), ("Go.", "Vaya."),
    ("Run!", "¡Corre!"), ("Run!", "¡Corran!"), ("Run!", "¡Corra!"),
    ("Hello!", "Hola."), ("Hello!", "Ve."), ("Hello!", "Vete.")
]
G.add_edges_from(edges)

# Add missing node explicitly
G.add_node('Stop.')

# Set positions for nodes
pos = nx.spring_layout(G)

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold', edge_color='green')

# Highlight nodes that were not found in the graph
missing_nodes = ["Stop."]
nx.draw_networkx_nodes(G, pos, nodelist=missing_nodes, node_color='red', node_size=2000)

plt.title('Graph-Based Translation Visualization')
plt.show()
