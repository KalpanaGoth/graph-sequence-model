import csv
import networkx as nx
import pickle

def load_csv_data(csv_path):
    """Load translation pairs from a CSV file."""
    translations = []
    try:
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                eng, spa = row[0], row[1]
                translations.append((eng, spa))
        if not translations:
            print("Step 1: CSV data is empty. Please check the file contents.")
        else:
            print("Step 1: CSV data loaded successfully.")
    except FileNotFoundError:
        print("Step 1: CSV file not found. Please ensure the path is correct and re-run Step 1.")
    return translations

def load_graph(graph_path):
    """Load the graph from a pickle file."""
    try:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        print("Step 2: Graph loaded successfully from pickle file.")
    except FileNotFoundError:
        print("Step 2: Graph file not found. Please ensure the path is correct and re-run Step 2.")
        G = None
    return G

def check_model_readiness(translations, G):
    """Check if all nodes from translations are in the graph and calculate readiness."""
    if not G:
        print("Graph not loaded, skipping readiness check.")
        return
    
    total_phrases = len(translations)
    if total_phrases == 0:
        print("Step 3: No phrases to check readiness. Please re-run Step 1 with a valid CSV file.")
        return

    missing_nodes = []
    
    for eng, _ in translations:
        if eng not in G.nodes:
            missing_nodes.append(eng)
    
    found_percentage = (total_phrases - len(missing_nodes)) / total_phrases * 100
    
    if missing_nodes:
        print(f"Step 3: Model Readiness Check - {found_percentage:.2f}% of nodes found.")
        print("Issues found: Missing words in the graph.")
        print("Suggested Actions: Re-run Step 1 or 2 to address missing nodes.")
        for node in missing_nodes[:5]:  # Display first 5 missing nodes for brevity
            print(f"- Missing node: {node}")
    else:
        print("Step 3: Model Readiness Check - 100% of nodes are present in the graph. Model is ready.")

def main():
    # File paths (update with your actual paths)
    csv_path = '../graph-sequence-model/data/translated_sentences.csv'
    graph_path = '../graph-sequence-model/output/translation_graph.gpickle'

    # Step 1: Load CSV data
    translations = load_csv_data(csv_path)
    
    # Step 2: Load the graph
    G = load_graph(graph_path)

    # Step 3: Check model readiness
    check_model_readiness(translations, G)

if __name__ == '__main__':
    main()
