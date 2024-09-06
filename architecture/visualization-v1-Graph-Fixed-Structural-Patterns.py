from graphviz import Digraph

# Create the Graph-Based Diagram for Fixed Structural Patterns in Current Training Methodologies
fixed_structural_patterns_graph = Digraph('Fixed Structural Patterns in Current Training Methodologies',
                                          node_attr={'shape': 'circle', 'style': 'filled', 'color': 'lightblue', 'fontcolor': 'black'},
                                          format='png')

# Set overall graph attributes
fixed_structural_patterns_graph.attr(rankdir='LR', splines='line', ranksep='1.0', size="7,5", fontname="Helvetica-Oblique", fontsize='12')

# Define nodes for each layer (input, hidden layers, output)
layers = {
    'Input': ['I1', 'I2', 'I3'],
    'Hidden1': ['H1_1', 'H1_2', 'H1_3'],
    'Hidden2': ['H2_1', 'H2_2', 'H2_3'],
    'Output': ['O1', 'O2', 'O3']
}

# Add nodes to graph
for layer_name, nodes in layers.items():
    with fixed_structural_patterns_graph.subgraph() as s:
        s.attr(rank='same')
        for node in nodes:
            s.node(node, f'{node}')

# Add edges to represent fixed connections (fixed paths) between layers
for input_node in layers['Input']:
    for hidden_node in layers['Hidden1']:
        fixed_structural_patterns_graph.edge(input_node, hidden_node)

for hidden_node in layers['Hidden1']:
    for hidden_node_2 in layers['Hidden2']:
        fixed_structural_patterns_graph.edge(hidden_node, hidden_node_2)

for hidden_node_2 in layers['Hidden2']:
    for output_node in layers['Output']:
        fixed_structural_patterns_graph.edge(hidden_node_2, output_node)

# Add labels for clarity
fixed_structural_patterns_graph.attr('node', shape='plaintext')
fixed_structural_patterns_graph.node('label1', 'Fixed Structural Patterns in Transformer Models', fontcolor='black', fontsize='14')
fixed_structural_patterns_graph.node('label2', 'Predefined paths and fixed layer connections', fontcolor='black', fontsize='10')
fixed_structural_patterns_graph.edge('label1', 'I1', style='invis')
fixed_structural_patterns_graph.edge('label2', 'H1_1', style='invis')

# Render the graph
fixed_structural_patterns_graph.render('fixed_structural_patterns_visualization', view=False)
