from graphviz import Digraph

# Create the Graph-Based Diagram for Pairwise Querying in Self-Attention
pairwise_querying_graph = Digraph('Pairwise Querying in Self-Attention',
                                  node_attr={'shape': 'circle', 'style': 'filled', 'color': 'lightblue', 'fontcolor': 'black'},
                                  format='png')

# Set overall graph attributes
pairwise_querying_graph.attr(rankdir='TB', splines='true', ranksep='1.0', size="5,5", fontname="Helvetica-Oblique", fontsize='12')

# Define nodes representing sequence elements
nodes = ['A', 'B', 'C', 'D', 'E']
for node in nodes:
    pairwise_querying_graph.node(node, f'{node}')

# Define pairwise edges showing querying relationships
for i in range(len(nodes)):
    for j in range(len(nodes)):
        if i != j:
            pairwise_querying_graph.edge(nodes[i], nodes[j])

# Add labels for clarity
pairwise_querying_graph.attr('node', shape='plaintext')
pairwise_querying_graph.node('label1', 'Pairwise Querying in Self-Attention', fontcolor='black', fontsize='14')
pairwise_querying_graph.node('label2', 'Each element queries every other element', fontcolor='black', fontsize='10')
pairwise_querying_graph.edge('label1', 'A', style='invis')
pairwise_querying_graph.edge('label2', 'B', style='invis')

# Render the graph
pairwise_querying_graph.render('pairwise_querying_self_attention_improved', view=False)
