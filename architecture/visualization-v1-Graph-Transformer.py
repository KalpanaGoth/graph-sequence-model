from graphviz import Digraph

# Create the Graph-Based Sequence Processing Architecture Diagram
graph_sequence_model = Digraph('Graph-Based Sequence Processing Architecture', 
                               node_attr={'shape': 'box', 'style': 'filled', 'color': 'black', 'fontcolor': 'white'}, 
                               format='png')

# Define the nodes with black background and white text
graph_sequence_model.node('data', 'Input Sequence Data')
graph_sequence_model.node('embedding', 'Embedding Layer')
graph_sequence_model.node('graph', 'Graph Representation')
graph_sequence_model.node('edge_weight', 'Edge Weight Calculation')
graph_sequence_model.node('message_passing', 'Message Passing')
graph_sequence_model.node('output', 'Final Output')

# Define the edges
graph_sequence_model.edge('data', 'embedding')
graph_sequence_model.edge('embedding', 'graph')
graph_sequence_model.edge('graph', 'edge_weight')
graph_sequence_model.edge('edge_weight', 'message_passing')
graph_sequence_model.edge('message_passing', 'output')

# Render and save the diagram
graph_sequence_model.render('graph_sequence_model')
