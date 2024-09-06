from graphviz import Digraph

# Create the Graph-Based Diagram for Dynamically Learned Graph Structures
dynamic_graph_structure = Digraph('Dynamically Learned Graph Structures',
                                  node_attr={'shape': 'circle', 'style': 'filled', 'color': 'lightblue', 'fontcolor': 'black'},
                                  format='png')

# Set overall graph attributes for a flexible layout
dynamic_graph_structure.attr(rankdir='TB', splines='true', ranksep='1.0', size="5,8", fontname="Helvetica-Oblique", fontsize='12')

# Define nodes representing elements in the dynamically learned graph
nodes = ['Node1', 'Node2', 'Node3', 'Node4', 'Node5', 'Node6']
for node in nodes:
    dynamic_graph_structure.node(node, f'{node}')

# Add dynamic edges to represent adaptable, learnable relationships
dynamic_graph_structure.edge('Node1', 'Node3', label='adaptive', color='black', style='solid')
dynamic_graph_structure.edge('Node1', 'Node5', label='hierarchical', color='black', style='dotted')
dynamic_graph_structure.edge('Node2', 'Node4', label='non-local', color='black', style='dashed')
dynamic_graph_structure.edge('Node2', 'Node6', label='sparse', color='black', style='dotted')
dynamic_graph_structure.edge('Node3', 'Node6', label='adaptive', color='black', style='solid')
dynamic_graph_structure.edge('Node4', 'Node5', label='non-local', color='black', style='dashed')

# Add labels for clarity at the top and bottom
dynamic_graph_structure.attr('node', shape='plaintext')
dynamic_graph_structure.node('label_top', 'Dynamically Learned Graph Structures', fontcolor='black', fontsize='12')
dynamic_graph_structure.node('label_bottom', 'Adaptable edges represent non-local, hierarchical, or sparse dependencies', fontcolor='black', fontsize='10')
dynamic_graph_structure.edge('label_top', 'Node1', style='invis')
dynamic_graph_structure.edge('Node6', 'label_bottom', style='invis')

# Render the graph
dynamic_graph_structure.render('dynamic_graph_structure_sequence_processing', view=False)
