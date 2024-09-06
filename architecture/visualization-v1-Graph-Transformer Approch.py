from graphviz import Digraph

graph_based_flow = Digraph('Graph-Based Flow', node_attr={'shape': 'ellipse', 'style': 'filled', 'color': 'black', 'fontcolor': 'white'}, format='png')

# Define the nodes
graph_based_flow.node('I', 'I [0.1, 0.2, 0.3]')
graph_based_flow.node('love', 'love [0.4, 0.5, 0.6]')
graph_based_flow.node('machine', 'machine [0.7, 0.8, 0.9]')
graph_based_flow.node('learning', 'learning [1.0, 1.1, 1.2]')

# Define the edges representing graph-based relations
graph_based_flow.edge('I', 'love', label='w_{1,2}')
graph_based_flow.edge('love', 'machine', label='w_{2,3}')
graph_based_flow.edge('machine', 'learning', label='w_{3,4}')
graph_based_flow.edge('I', 'learning', label='w_{1,4}')
graph_based_flow.edge('love', 'learning', label='w_{2,4}')
graph_based_flow.edge('I', 'machine', label='w_{1,3}')

# Render the diagram
graph_based_flow.render('graph_based_flow')
