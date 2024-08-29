# graph_utils.py
import networkx as nx

def create_graph_from_edges(nodes, edges):
    """
    Creates a graph from a list of nodes and edges.
    Args:
    - nodes (list): List of nodes in the graph.
    - edges (list): List of edges in the graph (tuples of (node1, node2)).

    Returns:
    - G (networkx.Graph): A graph object created from the nodes and edges.
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def compute_graph_degree_centrality(G):
    """
    Computes the degree centrality of each node in the graph.
    Args:
    - G (networkx.Graph): The input graph.

    Returns:
    - dict: A dictionary with nodes as keys and their degree centrality as values.
    """
    return nx.degree_centrality(G)

def get_neighbors(node, G):
    """
    Returns the neighbors of a given node in the graph.
    Args:
    - node: The node for which neighbors are to be found.
    - G (networkx.Graph): The input graph.

    Returns:
    - list: A list of neighbors for the given node.
    """
    return list(G.neighbors(node))
