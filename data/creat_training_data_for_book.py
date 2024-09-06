from docx import Document
import networkx as nx

# Load the document
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return text

# Convert the text into graph structure
def text_to_graph(text_data):
    G = nx.DiGraph()
    previous_node = None
    
    for i, line in enumerate(text_data):
        node = f"sentence_{i}"
        G.add_node(node, text=line)
        
        if previous_node:
            # Create an edge between the previous sentence and the current one
            G.add_edge(previous_node, node)
        
        # Update previous node to the current one
        previous_node = node
        
    return G

# Extract text and convert to graph
file_path = 'data/Books/Smoke_Blood_and_Time.docx'
text_data = extract_text_from_docx(file_path)
book_graph = text_to_graph(text_data)

# Save the graph to a file for model training
nx.write_gpickle(book_graph, "book_graph.gpickle")
