import networkx as nx
import pickle
import os
from chardet import detect

# Step 3 Start Book translation using the mapped graphed English to spanish dictionary. 
# Detect the encoding of the file, Load existing graph,  Read and process book text with detected encoding, Check if sentences are defined correctly

def detect_encoding(file_path):
    # Detect the encoding of the file
    with open(file_path, 'rb') as f:
        result = detect(f.read())
    return result['encoding']

def integrate_book_to_graph(graph_path, book_path):
    # Load existing graph
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    # Detect encoding of the book file
    encoding = detect_encoding(book_path)
    print(f"Detected encoding: {encoding}")

    # Read and process book text with detected encoding
    with open(book_path, 'r', encoding=encoding) as book:
        book_text = book.read()
        sentences = book_text.split('.')  # Define sentences correctly here
    
    # Check if sentences are defined correctly
    if not sentences:
        print("No sentences found in the book text.")
        return G

    # Process each sentence and map to graph
    for sentence in sentences:
        words = sentence.split()  # Simple split, can use more advanced NLP for tokenizing
        for word in words:
            # Check if the word exists in the graph, or if it needs to be added
            if word in G.nodes:
                # Increase weight of existing connections based on context in the book
                for neighbor in G.neighbors(word):
                    if 'weight' in G[word][neighbor]:
                        G[word][neighbor]['weight'] += 1

    # Save updated graph
    os.makedirs('output', exist_ok=True)
    with open('output/book_translation_graph.gpickle', 'wb') as f:
        pickle.dump(G, f)

    return G

# Example usage
graph_path = 'output/translation_graph_with_weights.gpickle'
book_path = '../graph-sequence-model/data/books/Smoke_Blood_and_Time.txt'  # Replace with your book file path
updated_graph = integrate_book_to_graph(graph_path, book_path)

print(f"Updated graph has {updated_graph.number_of_nodes()} nodes and {updated_graph.number_of_edges()} edges with updated weights.")
