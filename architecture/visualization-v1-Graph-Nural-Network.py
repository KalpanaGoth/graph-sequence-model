from graphviz import Digraph

# Create the Graph-Based Neural Network Architecture Diagram
neural_network_graph = Digraph('Neural Network Architecture',
                               node_attr={'shape': 'circle', 'style': 'filled'},
                               format='png')

# Set overall graph attributes
neural_network_graph.attr(rankdir='LR', splines='false', ranksep='1.4', size="7.75,10.25",
                          fontname="Helvetica-Oblique", fontsize='12')

# Define input layer nodes
neural_network_graph.attr('node', color='chartreuse', fillcolor='chartreuse')
neural_network_graph.node('x1', '<x1>')
neural_network_graph.node('x2', '<x2>')

# Define hidden layer 1 nodes
neural_network_graph.attr('node', color='dodgerblue', fillcolor='dodgerblue')
neural_network_graph.node('a12', '<a<sub>1</sub><sup>(2)</sup>>')
neural_network_graph.node('a22', '<a<sub>2</sub><sup>(2)</sup>>')
neural_network_graph.node('a32', '<a<sub>3</sub><sup>(2)</sup>>')
neural_network_graph.node('a42', '<a<sub>4</sub><sup>(2)</sup>>')
neural_network_graph.node('a52', '<a<sub>5</sub><sup>(2)</sup>>')

# Define hidden layer 2 nodes
neural_network_graph.node('a13', '<a<sub>1</sub><sup>(3)</sup>>')
neural_network_graph.node('a23', '<a<sub>2</sub><sup>(3)</sup>>')
neural_network_graph.node('a33', '<a<sub>3</sub><sup>(3)</sup>>')
neural_network_graph.node('a43', '<a<sub>4</sub><sup>(3)</sup>>')
neural_network_graph.node('a53', '<a<sub>5</sub><sup>(3)</sup>>')

# Define output layer nodes
neural_network_graph.attr('node', color='coral1', fillcolor='coral1')
neural_network_graph.node('O1', '<y1>')
neural_network_graph.node('O2', '<y2>')
neural_network_graph.node('O3', '<y3>')

# Define plaintext labels for layers
neural_network_graph.attr('node', shape='plaintext')
neural_network_graph.node('l0', 'layer 1 (input layer)')
neural_network_graph.node('l1', 'layer 2 (hidden layer)')
neural_network_graph.node('l2', 'layer 3 (hidden layer)')
neural_network_graph.node('l3', 'layer 4 (output layer)')

# Connect layer labels to nodes
neural_network_graph.edge('l0', 'x1')
neural_network_graph.edge('l1', 'a12')
neural_network_graph.edge('l2', 'a13')
neural_network_graph.edge('l3', 'O1')

# Ensure nodes of the same layer are ranked equally
neural_network_graph.attr(rank='same')
neural_network_graph.edge('x1', 'x2')
neural_network_graph.edge('a12', 'a22')
neural_network_graph.edge('a22', 'a32')
neural_network_graph.edge('a32', 'a42')
neural_network_graph.edge('a42', 'a52')
neural_network_graph.edge('a13', 'a23')
neural_network_graph.edge('a23', 'a33')
neural_network_graph.edge('a33', 'a43')
neural_network_graph.edge('a43', 'a53')
neural_network_graph.edge('O1', 'O2')
neural_network_graph.edge('O2', 'O3')

# Define edges between layers
neural_network_graph.attr('edge', style='solid', tailport='e', headport='w')
neural_network_graph.edges([('x1', 'a12'), ('x1', 'a22'), ('x1', 'a32'), ('x1', 'a42'), ('x1', 'a52'),
                            ('x2', 'a12'), ('x2', 'a22'), ('x2', 'a32'), ('x2', 'a42'), ('x2', 'a52')])

neural_network_graph.edges([('a12', 'a13'), ('a22', 'a23'), ('a32', 'a33'), ('a42', 'a43'), ('a52', 'a53')])

neural_network_graph.edges([('a13', 'O1'), ('a23', 'O2'), ('a33', 'O3'),
                            ('a43', 'O1'), ('a53', 'O2')])

# Render the graph
neural_network_graph.render('neural_network_01', view=False)
