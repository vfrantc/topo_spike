"""
Visualization utilities for Spiking WANNs
"""
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(graph, filename='best_wann_graph.png'):
    """
    Visualize the network graph
    
    Args:
        graph (dict): Graph representation
        filename (str): Output filename
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph['nodes']:
        node_id = node['id']
        node_type = node['type']
        G.add_node(node_id, type=node_type)
    
    # Add edges
    for edge in graph['edges']:
        src = edge['src']
        dst = edge['dst']
        sign = edge['sign']
        G.add_edge(src, dst, sign=sign)
    
    # Position nodes
    pos = {}
    
    # Get node types
    input_nodes = [n['id'] for n in graph['nodes'] if n['type'] == 'input']
    hidden_nodes = [n['id'] for n in graph['nodes'] if n['type'] == 'hidden']
    output_nodes = [n['id'] for n in graph['nodes'] if n['type'] == 'output']
    
    # Place input nodes in a row on the left
    for i, node_id in enumerate(input_nodes):
        pos[node_id] = (0, i - len(input_nodes)/2)
    
    # Place hidden nodes in the middle
    for i, node_id in enumerate(hidden_nodes):
        pos[node_id] = (0.5, i - len(hidden_nodes)/2)
    
    # Place output nodes in a row on the right
    for i, node_id in enumerate(output_nodes):
        pos[node_id] = (1, i - len(output_nodes)/2)
    
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
    for u, v in G.edges:
        sign = G[u][v]['sign']
        if sign > 0:
            edge_colors.append('green')
        else:
            edge_colors.append('red')
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
           node_size=500, arrowsize=10, font_size=8)
    
    plt.title("Weight-Agnostic Spiking Neural Network Architecture")
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