from graphviz import Digraph

self_attention_flow = Digraph('Self-Attention Flow', node_attr={'shape': 'box', 'style': 'filled', 'color': 'black', 'fontcolor': 'white'}, format='png')

# Define the nodes
self_attention_flow.node('I', 'I [0.1, 0.2, 0.3]')
self_attention_flow.node('love', 'love [0.4, 0.5, 0.6]')
self_attention_flow.node('machine', 'machine [0.7, 0.8, 0.9]')
self_attention_flow.node('learning', 'learning [1.0, 1.1, 1.2]')

# Define the edges representing attention flow
self_attention_flow.edge('I', 'love', label='w_{I, love}')
self_attention_flow.edge('I', 'machine', label='w_{I, machine}')
self_attention_flow.edge('I', 'learning', label='w_{I, learning}')

self_attention_flow.edge('love', 'I', label='w_{love, I}')
self_attention_flow.edge('love', 'machine', label='w_{love, machine}')
self_attention_flow.edge('love', 'learning', label='w_{love, learning}')

self_attention_flow.edge('machine', 'I', label='w_{machine, I}')
self_attention_flow.edge('machine', 'love', label='w_{machine, love}')
self_attention_flow.edge('machine', 'learning', label='w_{machine, learning}')

self_attention_flow.edge('learning', 'I', label='w_{learning, I}')
self_attention_flow.edge('learning', 'love', label='w_{learning, love}')
self_attention_flow.edge('learning', 'machine', label='w_{learning, machine}')

# Render the diagram
self_attention_flow.render('self_attention_flow')
