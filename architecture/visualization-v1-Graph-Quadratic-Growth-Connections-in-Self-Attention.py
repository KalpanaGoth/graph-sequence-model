from graphviz import Digraph

# Create the Graph-Based Diagram for Quadratic Computational Complexity in Self-Attention
quadratic_complexity_graph = Digraph('Quadratic Complexity in Self-Attention',
                                     node_attr={'shape': 'circle', 'style': 'filled', 'color': 'lightblue', 'fontcolor': 'black'},
                                     format='png')

# Set overall graph attributes
quadratic_complexity_graph.attr(rankdir='LR', splines='true', ranksep='1.0', size="8,5", fontname="Helvetica-Oblique", fontsize='12')

# Define sequence elements (nodes) to illustrate a small and a large sequence
small_sequence = ['A', 'B', 'C']
large_sequence = ['D', 'E', 'F', 'G', 'H']

# Add nodes for small sequence
for node in small_sequence:
    quadratic_complexity_graph.node(node, f'{node}')

# Add nodes for large sequence
for node in large_sequence:
    quadratic_complexity_graph.node(node, f'{node}')

# Add pairwise connections for small sequence (showing quadratic growth)
for i in range(len(small_sequence)):
    for j in range(len(small_sequence)):
        if i != j:
            quadratic_complexity_graph.edge(small_sequence[i], small_sequence[j])

# Add pairwise connections for large sequence (showing increased complexity)
for i in range(len(large_sequence)):
    for j in range(len(large_sequence)):
        if i != j:
            quadratic_complexity_graph.edge(large_sequence[i], large_sequence[j])

# Add labels for clarity
quadratic_complexity_graph.attr('node', shape='plaintext')
quadratic_complexity_graph.node('label1', 'Small Sequence: Quadratic Growth of Connections', fontcolor='black', fontsize='10')
quadratic_complexity_graph.node('label2', 'Large Sequence: Exponential Increase in Connections', fontcolor='black', fontsize='10')
quadratic_complexity_graph.edge('label1', 'A', style='invis')
quadratic_complexity_graph.edge('label2', 'D', style='invis')

# Render the graph
quadratic_complexity_graph.render('quadratic_complexity_self_attention', view=False)
