"""
Utility functions for graph manipulation
"""
import networkx as nx


def compute_topological_order(graph):
    """
    Compute a topological ordering of nodes for the forward pass
    
    Args:
        graph (dict): Graph representation
        
    Returns:
        list: Topologically sorted list of node IDs
    """
    # Create a directed graph
    G = nx.DiGraph()
    for node in graph['nodes']:
        G.add_node(str(node['id']))
        
    for edge in graph['edges']:
        G.add_edge(str(edge['src']), str(edge['dst']))
        
    # Get topological order
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # If graph has cycles, use a fallback ordering
        # For simplicity, we'll just sort by node id
        return sorted([str(node['id']) for node in graph['nodes']])


def count_network_size(graph):
    """
    Count the number of hidden nodes and edges in a graph
    
    Args:
        graph (dict): Graph representation
        
    Returns:
        tuple: (num_hidden_nodes, num_edges)
    """
    num_hidden = len([n for n in graph['nodes'] if n['type'] == 'hidden'])
    num_edges = len(graph['edges'])
    return num_hidden, num_edges


def get_node_ids_by_type(graph):
    """
    Get the node IDs organized by type
    
    Args:
        graph (dict): Graph representation
        
    Returns:
        tuple: (input_ids, hidden_ids, output_ids)
    """
    input_ids = [node['id'] for node in graph['nodes'] if node['type'] == 'input']
    hidden_ids = [node['id'] for node in graph['nodes'] if node['type'] == 'hidden']
    output_ids = [node['id'] for node in graph['nodes'] if node['type'] == 'output']
    
    return input_ids, hidden_ids, output_ids