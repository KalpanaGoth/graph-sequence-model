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

# Simulate message passing (example: update node size based on "importance")
node_sizes = [1500 if word in ["Love", "Machine", "Learning", "AI"] else 500 for word in words]

# Convert node_sizes to match node count
node_sizes = [node_sizes[words.index(node)] for node in G.nodes()]

# Draw graph
plt.figure(figsize=(14, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', 
        edge_color='gray', node_size=node_sizes, font_size=10)
plt.title('Message Passing Visualization with Words')
plt.show()
