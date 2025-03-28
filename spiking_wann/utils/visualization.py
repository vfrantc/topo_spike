"""
Enhanced visualization utilities for Spiking WANNs with layer support
"""
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ..model.graph_utils import get_nodes_by_layer


def visualize_graph(graph, filename='best_wann_graph.png'):
    """
    Visualize the network graph with layer information
    
    Args:
        graph (dict): Graph representation
        filename (str): Output filename
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph['nodes']:
        node_id = node['id']
        node_type = node['type']
        layer = node.get('layer', 0)
        G.add_node(node_id, type=node_type, layer=layer)
    
    # Add edges
    edge_attributes = {}
    for edge in graph['edges']:
        src = edge['src']
        dst = edge['dst']
        sign = edge['sign']
        G.add_edge(src, dst)
        edge_attributes[(src, dst)] = {'sign': sign}
    
    nx.set_edge_attributes(G, edge_attributes)
    
    # Get nodes by layer for positioning
    layers = get_nodes_by_layer(graph)
    max_layer = len(layers) - 1
    
    # Position nodes using layer information
    pos = {}
    
    for layer_idx, layer_nodes in enumerate(layers):
        # Calculate the vertical spread based on number of nodes in this layer
        y_positions = np.linspace(-0.9, 0.9, len(layer_nodes)) if layer_nodes else []
        
        # Normalized x-coordinate for this layer
        x_coord = layer_idx / max(1, max_layer)
        
        # Assign positions
        for i, node_id in enumerate(layer_nodes):
            pos[node_id] = (x_coord, y_positions[i] if y_positions else 0)
    
    # Create node colors based on type
    node_colors = []
    for node in G.nodes:
        node_type = G.nodes[node]['type']
        if node_type == 'input':
            node_colors.append('lightblue')
        elif node_type == 'hidden':
            node_colors.append('lightgreen')
        else:  # output
            node_colors.append('salmon')
    
    # Create edge colors based on sign
    edge_colors = []
    edge_styles = []
    
    for u, v in G.edges:
        sign = G[u][v].get('sign', 1)
        # Color based on sign
        if sign > 0:
            edge_colors.append('green')
        else:
            edge_colors.append('red')
            
        # Determine if this is a skip connection
        src_layer = G.nodes[u].get('layer', 0)
        dst_layer = G.nodes[v].get('layer', 0)
        
        if dst_layer - src_layer > 1:
            # This is a skip connection
            edge_styles.append('dashed')
        else:
            edge_styles.append('solid')
            
    # Set up figure
    plt.figure(figsize=(12, 10))
    
    # Draw the network
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
           node_size=500, arrowsize=10, font_size=8)
    
    # Draw skip connections with dashed style
    skip_edges = [e for i, e in enumerate(G.edges) if edge_styles[i] == 'dashed']
    if skip_edges:
        nx.draw_networkx_edges(G, pos, edgelist=skip_edges, style='dashed', 
                               edge_color=[edge_colors[i] for i, e in enumerate(G.edges) if edge_styles[i] == 'dashed'])
    
    # Add layer labels
    layer_labels = {i: f"Layer {i}" for i in range(max_layer + 1)}
    x_positions = [i / max(1, max_layer) for i in range(max_layer + 1)]
    for i, x in enumerate(x_positions):
        plt.text(x, 1.05, layer_labels[i], ha='center', fontsize=12)
    
    # Add a title showing network statistics
    num_hidden = len([n for n in graph['nodes'] if n['type'] == 'hidden'])
    num_edges = len(graph['edges'])
    
    # Count skip connections
    skip_connections = 0
    for u, v in G.edges:
        src_layer = G.nodes[u].get('layer', 0)
        dst_layer = G.nodes[v].get('layer', 0)
        if dst_layer - src_layer > 1:
            skip_connections += 1
    
    plt.title(f"Spiking Neural Network: {len(G.nodes)} nodes ({num_hidden} hidden), "
              f"{num_edges} edges ({skip_connections} skip connections)")
    
    plt.savefig(filename)
    plt.close()


def visualize_fitness_history(fitness_history, filename='fitness_history.png'):
    """
    Visualize the fitness history
    
    Args:
        fitness_history (list): List of (mean_fitness, max_fitness, mean_accuracy, max_accuracy) tuples
        filename (str): Output filename
    """
    means_fitness, maxes_fitness, means_acc, maxes_acc = zip(*fitness_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Fitness plot
    ax1.plot(means_fitness, label='Mean Fitness')
    ax1.plot(maxes_fitness, label='Max Fitness')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness History')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(means_acc, label='Mean Accuracy')
    ax2.plot(maxes_acc, label='Max Accuracy')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy History')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_layer_connectivity(graph, filename='layer_connectivity.png'):
    """
    Visualize the connectivity between layers
    
    Args:
        graph (dict): Graph representation
        filename (str): Output filename
    """
    # Get nodes by layer
    layers = get_nodes_by_layer(graph)
    
    # Count connections between layers
    layer_connections = {}
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
            if layer_pair not in layer_connections:
                layer_connections[layer_pair] = 0
            layer_connections[layer_pair] += 1
    
    # Create a matrix visualization of layer connectivity
    num_layers = len(layers)
    connection_matrix = np.zeros((num_layers, num_layers))
    
    for (src, dst), count in layer_connections.items():
        connection_matrix[src, dst] = count
    
    plt.figure(figsize=(10, 8))
    plt.imshow(connection_matrix, cmap='Blues')
    plt.colorbar(label='Number of Connections')
    
    # Add labels
    plt.xlabel('Destination Layer')
    plt.ylabel('Source Layer')
    plt.title('Layer Connectivity Matrix')
    
    # Add layer labels
    layer_labels = [f"Layer {i}" for i in range(num_layers)]
    plt.xticks(range(num_layers), layer_labels)
    plt.yticks(range(num_layers), layer_labels)
    
    # Add connection counts as text
    for i in range(num_layers):
        for j in range(num_layers):
            if connection_matrix[i, j] > 0:
                plt.text(j, i, int(connection_matrix[i, j]), 
                         ha="center", va="center", color="black" if connection_matrix[i, j] < 10 else "white")
    
    plt.savefig(filename)
    plt.close()