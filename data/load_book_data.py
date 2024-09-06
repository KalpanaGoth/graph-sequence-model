import networkx as nx
from jfs_model_library import GraphSequenceModel, train_model

# Load the graph data
graph_data = nx.read_gpickle("book_graph.gpickle")

# Initialize your graph-sequence model
model = GraphSequenceModel(input_dim=128, hidden_dim=256, output_dim=128)  # Adjust dimensions as necessary

# Prepare data for training (convert graph to suitable format for your model)
# This step will vary depending on your model's requirements
train_data, train_labels = prepare_data(graph_data)

# Train the model
train_model(model, train_data, train_labels, epochs=10, batch_size=32)

# Save the trained model
model.save("trained_graph_sequence_model.pth")
