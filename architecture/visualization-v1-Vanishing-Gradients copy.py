from graphviz import Digraph

# Create the Graph-Based Sequence Processing Architecture Diagram
vanishing_gradients_graph = Digraph('Vanishing Gradients in RNN/LSTM', 
                                    node_attr={'shape': 'circle', 'style': 'filled', 'color': 'lightblue', 'fontcolor': 'black'}, 
                                    format='png')

# Define the input nodes
vanishing_gradients_graph.node('Input0', 'Input 0')
vanishing_gradients_graph.node('Input1', 'Input 1')
vanishing_gradients_graph.node('Input2', 'Input 2')
vanishing_gradients_graph.node('Input3', 'Input 3')
vanishing_gradients_graph.node('InputN', 'Input N')

# Define the hidden state nodes
vanishing_gradients_graph.node('Hidden0', 'H0', color='lightcoral')
vanishing_gradients_graph.node('Hidden1', 'H1', color='lightcoral')
vanishing_gradients_graph.node('Hidden2', 'H2', color='lightcoral')
vanishing_gradients_graph.node('Hidden3', 'H3', color='lightcoral')
vanishing_gradients_graph.node('HiddenN', 'HN', color='lightcoral')

# Define the output nodes
vanishing_gradients_graph.node('Output0', 'Output 0')
vanishing_gradients_graph.node('Output1', 'Output 1')
vanishing_gradients_graph.node('Output2', 'Output 2')
vanishing_gradients_graph.node('Output3', 'Output 3')
vanishing_gradients_graph.node('OutputN', 'Output N')

# Add edges between input and hidden states
vanishing_gradients_graph.edge('Input0', 'Hidden0')
vanishing_gradients_graph.edge('Input1', 'Hidden1')
vanishing_gradients_graph.edge('Input2', 'Hidden2')
vanishing_gradients_graph.edge('Input3', 'Hidden3')
vanishing_gradients_graph.edge('InputN', 'HiddenN')

# Add edges between hidden states (showing sequential processing)
vanishing_gradients_graph.edge('Hidden0', 'Hidden1')
vanishing_gradients_graph.edge('Hidden1', 'Hidden2')
vanishing_gradients_graph.edge('Hidden2', 'Hidden3')
vanishing_gradients_graph.edge('Hidden3', 'HiddenN')

# Add edges between hidden states and outputs
vanishing_gradients_graph.edge('Hidden0', 'Output0')
vanishing_gradients_graph.edge('Hidden1', 'Output1')
vanishing_gradients_graph.edge('Hidden2', 'Output2')
vanishing_gradients_graph.edge('Hidden3', 'Output3')
vanishing_gradients_graph.edge('HiddenN', 'OutputN')

# Add diminishing gradient edges
vanishing_gradients_graph.edge('HiddenN', 'Hidden3', label='Diminishing Gradient', color='red', style='dashed')
vanishing_gradients_graph.edge('Hidden3', 'Hidden2', color='red', style='dashed')
vanishing_gradients_graph.edge('Hidden2', 'Hidden1', color='red', style='dashed')
vanishing_gradients_graph.edge('Hidden1', 'Hidden0', color='red', style='dashed')

# Render the graph (save as a file if needed)
vanishing_gradients_graph.render('vanishing_gradients_diagram', view=True)
