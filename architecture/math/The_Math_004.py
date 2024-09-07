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

# Create graph
G = nx.DiGraph()
G.add_nodes_from(words)
G.add_edges_from(edges)

# Draw graph
plt.figure(figsize=(14, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', 
        edge_color='gray', node_size=2000, font_size=10)
plt.title('Dynamic Graph Visualization with Words')
plt.show()
