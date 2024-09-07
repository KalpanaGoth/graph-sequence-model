import networkx as nx
import pickle
from tqdm import tqdm

#Step 7: Evaluation if new model
# Test English sentences using the graph-based model and compare the results with 
# traditional translation models for performance assessment.

# Load the trained graph with embeddings
graph_path = 'output/book_translation_graph.gpickle'
with open(graph_path, 'rb') as f:
    G = pickle.load(f)

# Sample English sentences for evaluation
test_sentences = ["Go.", "Run!", "Stop.", "Hello!"]

def graph_based_translation(G, source_node, top_k=1):
    # Assuming 'source_node' is in the graph
    scores = {node: 0.0 for node in G.nodes}
    scores[source_node] = 1.0  # Starting point

    # Propagate scores using the graph's edges and weights with a progress bar
    propagation_steps = 3  # Number of propagation steps
    for step in tqdm(range(propagation_steps), desc="Propagation Steps", unit="step"):
        new_scores = scores.copy()
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                new_scores[neighbor] += scores[node] * G[node][neighbor].get('weight', 1.0)
        
        # Update scores
        scores = new_scores

        # Print progress as a percentage
        percent_complete = ((step + 1) / propagation_steps) * 100
        print(f"Propagation Step {step + 1}/{propagation_steps} complete: {percent_complete:.0f}% done")

    # Extract Spanish nodes
    spa_nodes = [n for n in G.nodes if G.nodes[n].get('lang') == 'SPA']

    # Rank Spanish nodes by score
    ranked_spa_nodes = sorted(spa_nodes, key=lambda n: scores[n], reverse=True)

    # Select top-k translations
    top_translations = ranked_spa_nodes[:top_k]
    return top_translations

# Evaluate translations
for sentence in test_sentences:
    if sentence in G.nodes:
        translations = graph_based_translation(G, sentence, top_k=3)
        print(f"Translations for '{sentence}': {translations}")
    else:
        print(f"'{sentence}' not found in graph.")

# Placeholder for comparison with traditional models
# (Implement traditional translation model evaluation here for comparison)
