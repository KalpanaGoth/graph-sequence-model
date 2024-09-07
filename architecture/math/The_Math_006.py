import matplotlib.pyplot as plt
import networkx as nx

# Define nodes (words) and initial connections
words = ["I", "Love", "Machine", "Learning", "in", "2024", 
         "because", "the", "world", "is", "changing", 
         "at", "the", "speed", "of", "AI"]
edges = [("I", "Love"), ("Love", "Machine"), ("Machine", "Learning"), 
         ("Learning", "in"), ("in", "2024"), ("2024", "because"),
         ("because", "the"), ("the", "world"), ("world", "is"),
         ("is", "changing"), ("changing", "at"), ("at", "the"),
         ("the", "speed"), ("speed", "of"), ("of", "AI")]

# Define weights (simulating message importance)
weights = [0.8, 1.2, 1.5, 0.7, 1.0, 0.6, 0.9, 1.1, 0.5, 1.3, 1.4, 0.6, 1.1, 1.3, 1.0]

# Create graph
G = nx.DiGraph()
G.add_nodes_from(words)
G.add_edges_from(edges)

# Convert node sizes to match node count
node_sizes = [1500 if word in ["Love", "Machine", "Learning", "AI"] else 500 for word in words]
node_sizes = [node_sizes[words.index(node)] for node in G.nodes()]

# Draw graph with edge labels to show message passing weights
plt.figure(figsize=(14, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', 
        edge_color='gray', node_size=node_sizes, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{w:.1f}' for (u, v), w in zip(edges, weights)})

plt.title('Message Passing Visualization with Words and Weights')
plt.show()
