from graphviz import Digraph

# Create the Graph-Based Diagram for Dynamic Graph Learning Approach
graph_based_approach = Digraph('Graph-Based Approach to Sequence Processing',
                               node_attr={'shape': 'circle', 'style': 'filled', 'color': 'lightblue', 'fontcolor': 'black'},
                               format='png')

# Set overall graph attributes for a vertical layout
graph_based_approach.attr(rankdir='TB', splines='true', ranksep='1.0', size="5,8", fontname="Helvetica-Oblique", fontsize='12')

# Define nodes representing elements in the sequence
nodes = ['Node1', 'Node2', 'Node3', 'Node4', 'Node5']
for node in nodes:
    graph_based_approach.node(node, f'{node}')

# Add dynamic edges to represent learnable relationships
graph_based_approach.edge('Node1', 'Node2', label='learnable', color='black')
graph_based_approach.edge('Node1', 'Node3', label='learnable', color='black', style='dashed')
graph_based_approach.edge('Node1', 'Node4', label='learnable', color='black', style='dotted')
graph_based_approach.edge('Node2', 'Node4', label='learnable', color='black')
graph_based_approach.edge('Node2', 'Node5', label='learnable', color='black', style='dashed')
graph_based_approach.edge('Node3', 'Node5', label='learnable', color='black')
graph_based_approach.edge('Node4', 'Node5', label='learnable', color='black', style='dotted')

# Add labels for clarity at the top and bottom
graph_based_approach.attr('node', shape='plaintext')
graph_based_approach.node('label_top', 'Graph-Based Approach to Sequence Processing', fontcolor='black', fontsize='12')
graph_based_approach.node('label_bottom', 'Nodes = Sequence Elements, Edges = Learnable Relationships', fontcolor='black', fontsize='10')
graph_based_approach.edge('label_top', 'Node1', style='invis')
graph_based_approach.edge('Node5', 'label_bottom', style='invis')

# Render the graph
graph_based_approach.render('graph_based_approach_sequence_processing_vertical', view=False)
