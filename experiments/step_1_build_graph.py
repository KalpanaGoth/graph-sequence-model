import networkx as nx
import pickle

# In Step 1 we build translation graph, prepare the file for create the Nodes to Represent English and Spanish phrases.
# Edges: Represent translation relationships between the nodes.
# Graph Structure: Uses a directed graph to model the translation pairs, making it easy to expand 
# and visualize. build_translation_graph

def build_translation_graph(file_path):
    G = nx.DiGraph()
    translations = []

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

    return G, translations

file_path = '../graph-sequence-model/data/language/eng-spa.txt'
translation_graph, translations = build_translation_graph(file_path)

# Save the graph to a file using pickle
with open('output/translation_graph.gpickle', 'wb') as f:
    pickle.dump(translation_graph, f)

# Save translations to CSV
with open('output/translated_sentences.csv', 'w', encoding='utf-8') as f:
    for eng, spa in translations:
        f.write(f"{eng},{spa}\n")

print(f"Graph has {translation_graph.number_of_nodes()} nodes and {translation_graph.number_of_edges()} edges.")
