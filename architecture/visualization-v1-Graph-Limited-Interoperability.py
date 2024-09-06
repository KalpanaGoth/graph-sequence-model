from graphviz import Digraph

# Create the Graph-Based Diagram for Limited Interpretability in Self-Attention
limited_interpretability_graph = Digraph('Limited Interpretability in Self-Attention',
                                         node_attr={'shape': 'circle', 'style': 'filled', 'color': 'lightblue', 'fontcolor': 'black'},
                                         format='png')

# Set overall graph attributes
limited_interpretability_graph.attr(rankdir='LR', splines='true', ranksep='1.0', size="8,5", fontname="Helvetica-Oblique", fontsize='12')

# Define nodes representing data elements with attention scores
nodes = ['A', 'B', 'C', 'D', 'E']
for node in nodes:
    limited_interpretability_graph.node(node, f'{node}')

# Add edges representing attention weights (with varying line thickness to simulate attention scores)
limited_interpretability_graph.edge('A', 'B', label='0.2', penwidth='0.5')
limited_interpretability_graph.edge('A', 'C', label='0.5', penwidth='1.5')
limited_interpretability_graph.edge('A', 'D', label='0.3', penwidth='1.0')
limited_interpretability_graph.edge('A', 'E', label='0.1', penwidth='0.3')

limited_interpretability_graph.edge('B', 'A', label='0.4', penwidth='1.0')
limited_interpretability_graph.edge('B', 'C', label='0.3', penwidth='0.8')
limited_interpretability_graph.edge('B', 'D', label='0.2', penwidth='0.5')
limited_interpretability_graph.edge('B', 'E', label='0.1', penwidth='0.3')

# Add more connections to show complex attention patterns without clear structure
limited_interpretability_graph.edge('C', 'A', label='0.3', penwidth='1.0')
limited_interpretability_graph.edge('C', 'B', label='0.1', penwidth='0.3')
limited_interpretability_graph.edge('C', 'D', label='0.4', penwidth='1.2')
limited_interpretability_graph.edge('C', 'E', label='0.2', penwidth='0.5')

# Illustrating the lack of clear hierarchy or structure
limited_interpretability_graph.edge('D', 'A', label='0.2', penwidth='0.5')
limited_interpretability_graph.edge('D', 'B', label='0.5', penwidth='1.5')
limited_interpretability_graph.edge('D', 'C', label='0.3', penwidth='1.0')
limited_interpretability_graph.edge('D', 'E', label='0.1', penwidth='0.3')

limited_interpretability_graph.edge('E', 'A', label='0.1', penwidth='0.3')
limited_interpretability_graph.edge('E', 'B', label='0.3', penwidth='0.8')
limited_interpretability_graph.edge('E', 'C', label='0.2', penwidth='0.5')
limited_interpretability_graph.edge('E', 'D', label='0.4', penwidth='1.2')

# Add labels for clarity
limited_interpretability_graph.attr('node', shape='plaintext')
limited_interpretability_graph.node('label1', 'Attention Distributions in Self-Attention', fontcolor='black', fontsize='12')
limited_interpretability_graph.node('label2', 'Complex Attention Scores with No Clear Structure', fontcolor='black', fontsize='10')
limited_interpretability_graph.edge('label1', 'A', style='invis')
limited_interpretability_graph.edge('label2', 'B', style='invis')

# Render the graph
limited_interpretability_graph.render('limited_interpretability_self_attention', view=False)
