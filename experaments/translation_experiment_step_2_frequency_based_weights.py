import networkx as nx
import pickle
import os
from collections import Counter

# In Step 2, we add Frequency-Based Weights: Edge weights are assigned based on how often each 
# translation pair appears in the data. Graph Output: The graph with weights is saved in 
# output/translation_graph_with_weights.gpickle, and the translations with weights are saved 
# in output/translated_sentences_with_weights.csv.

def build_translation_graph_with_weights(file_path):
    G = nx.DiGraph()
    translations = []
    translation_counts = Counter()

    # Read the file and parse lines
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split('\t')
            if len(parts) >= 2:
                eng = parts[0].strip()
                spa = parts[1].strip()
                G.add_node(eng, lang='ENG')
                G.add_node(spa, lang='SPA')
                G.add_edge(eng, spa, label='translation')
                translations.append((eng, spa))
                translation_counts[(eng, spa)] += 1  # Count frequency

    # Assign edge weights based on frequency
    for (eng, spa), count in translation_counts.items():
        G[eng][spa]['weight'] = count  # Set weight as frequency

    return G, translations

file_path = '../graph-sequence-model/data/language/eng-spa.txt'
translation_graph, translations = build_translation_graph_with_weights(file_path)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save the graph to a file using pickle
with open('output/translation_graph_with_weights.gpickle', 'wb') as f:
    pickle.dump(translation_graph, f)

# Save translations to CSV with weights in the output folder
with open('output/translated_sentences_with_weights.csv', 'w', encoding='utf-8') as f:
    f.write("English,Spanish,Weight\n")
    for eng, spa in translations:
        weight = translation_graph[eng][spa]['weight']
        f.write(f"{eng},{spa},{weight}\n")

print(f"Step 2 complete: Graph has {translation_graph.number_of_nodes()} nodes and {translation_graph.number_of_edges()} edges with weights.")
