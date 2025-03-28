"""
Utility functions for graph manipulation with layer support
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
        # If graph has cycles, use a fallback ordering by layer
        # Sort nodes by layer, and then by ID within each layer
        sorted_nodes = sorted(graph['nodes'], key=lambda n: (n.get('layer', 0), n['id']))
        return [str(node['id']) for node in sorted_nodes]


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


def get_nodes_by_layer(graph):
    """
    Get the node IDs organized by layer
    
    Args:
        graph (dict): Graph representation
        
    Returns:
        list: List of lists containing node IDs for each layer
    """
    # Determine the maximum layer number
    max_layer = max([node.get('layer', 0) for node in graph['nodes']])
    
    # Initialize lists for each layer
    layers = [[] for _ in range(max_layer + 1)]
    
    # Populate layers
    for node in graph['nodes']:
        layer = node.get('layer', 0)
        layers[layer].append(node['id'])
    
    return layers


def analyze_layer_connectivity(graph):
    """
    Analyze the connectivity between layers
    
    Args:
        graph (dict): Graph representation
        
    Returns:
        dict: Statistics about layer connectivity
    """
    # Get nodes by layer
    layers = get_nodes_by_layer(graph)
    
    # Initialize statistics
    stats = {
        'layer_sizes': [len(layer) for layer in layers],
        'skip_connections': 0,
        'adjacent_connections': 0,
        'layer_connections': {}
    }
    
    # Count connections between layers
    for edge in graph['edges']:
        src_id = edge['src']
        dst_id = edge['dst']
        
        # Find the layers for src and dst
        src_layer = None
        dst_layer = None
        
        for i, layer in enumerate(layers):
            if src_id in layer:
                src_layer = i
            if dst_id in layer:
                dst_layer = i
        
        # If we found both layers
        if src_layer is not None and dst_layer is not None:
            layer_pair = (src_layer, dst_layer)
            
            # Update statistics
            if layer_pair not in stats['layer_connections']:
                stats['layer_connections'][layer_pair] = 0
            stats['layer_connections'][layer_pair] += 1
            
            # Check if this is a skip connection
            if dst_layer - src_layer > 1:
                stats['skip_connections'] += 1
            else:
                stats['adjacent_connections'] += 1
    
    return stats