import time
import networkx as nx
import pickle
import matplotlib.pyplot as plt

# Paths
graph_path = 'output/book_translation_graph.gpickle'

# Load the graph and measure load time
start_time = time.time()
with open(graph_path, 'rb') as f:
    G = pickle.load(f)
load_time = time.time() - start_time

# Test phrases
test_sentences = ["Go.", "Run!", "Stop.", "Hello!"]

def graph_based_translation(G, source_node, top_k=1):
    # Initialize node scores
    scores = {node: 0.0 for node in G.nodes}
    scores[source_node] = 1.0

    # Propagation
    for _ in range(3):
        new_scores = scores.copy()
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                new_scores[neighbor] += scores[node] * G[node][neighbor].get('weight', 1.0)
        scores = new_scores

    # Extract Spanish nodes and rank by score
    spa_nodes = [n for n in G.nodes if G.nodes[n].get('lang') == 'SPA']
    ranked_spa_nodes = sorted(spa_nodes, key=lambda n: scores[n], reverse=True)

    return ranked_spa_nodes[:top_k]

# Evaluate translations and measure processing time
processing_times = []
for sentence in test_sentences:
    start_time = time.time()
    if sentence in G.nodes:
        translations = graph_based_translation(G, sentence, top_k=3)
        processing_times.append(time.time() - start_time)
        print(f"Translations for '{sentence}': {translations}")
    else:
        processing_times.append(time.time() - start_time)
        print(f"'{sentence}' not found in graph.")

# Comparison with traditional models (mock data)
traditional_times = [0.01, 0.02, 0.015, 0.01]  # Mock times for comparison

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(["Graph Load Time"], [load_time], color='blue', label='Graph Load Time')
plt.bar([f"Phrase {i+1}" for i in range(len(processing_times))], processing_times, color='green', label='Graph-Based Model')
plt.bar([f"Phrase {i+1}" for i in range(len(traditional_times))], traditional_times, color='red', label='Traditional Model')
plt.xlabel('Steps')
plt.ylabel('Time (seconds)')
plt.title('Performance Analysis of Translation Models')
plt.legend()
plt.show()
