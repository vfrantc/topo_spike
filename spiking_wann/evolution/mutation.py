"""
Mutation operations for graph evolution with support for multiple layers
"""
import random
import copy

from ..model.graph_utils import get_node_ids_by_type, get_nodes_by_layer
from ..config import (
    MUTATION_RATE, MAX_HIDDEN_NEURONS, MAX_EDGES, 
    ADD_EDGE_PROB, NUM_HIDDEN_LAYERS, LAYER_SKIP_PROB
)


def mutate_graph(graph, mutation_rate=MUTATION_RATE, max_hidden=MAX_HIDDEN_NEURONS, 
                max_edges=MAX_EDGES, add_edge_prob=ADD_EDGE_PROB):
    """
    Mutate a graph architecture with layer support
    
    Args:
        graph (dict): Graph representation
        mutation_rate (float): Probability of mutation
        max_hidden (int): Maximum number of hidden neurons
        max_edges (int): Maximum number of edges
        add_edge_prob (float): Probability of adding an edge vs other mutations
    
    Returns:
        dict: Mutated graph
    """
    # Make a deep copy to avoid modifying the original
    mutated = copy.deepcopy(graph)
    
    # Extract graph components
    nodes = mutated['nodes']
    edges = mutated['edges']
    
    # Get node types and ids
    input_ids, hidden_ids, output_ids = get_node_ids_by_type(mutated)
    
    # Get nodes by layer
    layer_nodes = get_nodes_by_layer(mutated)
    
    # Find max node id
    max_id = max([node['id'] for node in nodes])
    
    # Current edge connections as tuples for quick lookup
    current_edges = set((edge['src'], edge['dst']) for edge in edges)
    
    # For the first few generations, prioritize adding edges to increase connectivity
    if random.random() < add_edge_prob and len(edges) < max_edges:
        # Create potential new edges (ones that don't already exist)
        possible_new_edges = []
        
        # Generate possible edges between layers (including skip connections)
        for layer_idx in range(len(layer_nodes) - 1):
            src_layer = layer_nodes[layer_idx]
            
            # Connect to all subsequent layers
            for target_layer_idx in range(layer_idx + 1, len(layer_nodes)):
                # Skip probability decreases with layer distance
                if layer_idx + 1 < target_layer_idx and random.random() > LAYER_SKIP_PROB:
                    continue
                    
                dst_layer = layer_nodes[target_layer_idx]
                
                for src in src_layer:
                    for dst in dst_layer:
                        if (src, dst) not in current_edges:
                            possible_new_edges.append((src, dst))
        
        # Add a new edge if possible
        if possible_new_edges:
            src, dst = random.choice(possible_new_edges)
            edges.append({
                "src": src, 
                "dst": dst,
                "sign": random.choice([-1, 1])
            })
    else:
        # Regular mutations
        # 1. Possibly add a hidden neuron
        if random.random() < mutation_rate and len(hidden_ids) < max_hidden:
            new_id = max_id + 1
            # Select a random hidden layer
            layer = random.randint(1, NUM_HIDDEN_LAYERS)
            nodes.append({"id": new_id, "type": "hidden", "layer": layer})
            hidden_ids.append(new_id)
            
            # Connect the new neuron to the network
            # Connect from previous layer
            prev_layer_nodes = layer_nodes[layer-1] if layer-1 >= 0 else []
            if prev_layer_nodes:
                num_inputs_to_connect = random.randint(1, min(3, len(prev_layer_nodes)))
                for _ in range(num_inputs_to_connect):
                    src = random.choice(prev_layer_nodes)
                    edges.append({
                        "src": src,
                        "dst": new_id,
                        "sign": random.choice([-1, 1])
                    })
            
            # Connect to next layer
            next_layer_nodes = layer_nodes[layer+1] if layer+1 < len(layer_nodes) else []
            if next_layer_nodes:
                num_outputs_to_connect = random.randint(1, min(3, len(next_layer_nodes)))
                for _ in range(num_outputs_to_connect):
                    dst = random.choice(next_layer_nodes)
                    edges.append({
                        "src": new_id,
                        "dst": dst,
                        "sign": random.choice([-1, 1])
                    })
                    
            # Possibly add a skip connection
            if random.random() < LAYER_SKIP_PROB:
                # Select a random distant layer
                skip_options = list(range(layer+2, len(layer_nodes)))
                if skip_options and layer_nodes[skip_options[0]]:
                    skip_layer = random.choice(skip_options)
                    dst = random.choice(layer_nodes[skip_layer])
                    edges.append({
                        "src": new_id,
                        "dst": dst,
                        "sign": random.choice([-1, 1])
                    })
        
        # 2. Possibly remove a hidden neuron
        if random.random() < mutation_rate and len(hidden_ids) > NUM_HIDDEN_LAYERS:  # Ensure at least one neuron per layer
            remove_id = random.choice(hidden_ids)
            nodes = [node for node in nodes if node['id'] != remove_id]
            edges = [edge for edge in edges if edge['src'] != remove_id and edge['dst'] != remove_id]
        
        # 3. Possibly add a skip connection
        if random.random() < mutation_rate * LAYER_SKIP_PROB and len(edges) < max_edges:
            # Find layers with distance > 1
            possible_skip_connections = []
            for layer_idx in range(len(layer_nodes) - 2):  # Skip immediate next layer
                for target_layer_idx in range(layer_idx + 2, len(layer_nodes)):
                    for src in layer_nodes[layer_idx]:
                        for dst in layer_nodes[target_layer_idx]:
                            if (src, dst) not in current_edges:
                                possible_skip_connections.append((src, dst))
            
            if possible_skip_connections:
                src, dst = random.choice(possible_skip_connections)
                edges.append({
                    "src": src,
                    "dst": dst,
                    "sign": random.choice([-1, 1])
                })
        
        # 4. Possibly add a regular edge (lower priority)
        elif random.random() < mutation_rate/2 and len(edges) < max_edges:
            # Create a new edge between adjacent layers
            possible_new_edges = []
            for layer_idx in range(len(layer_nodes) - 1):
                src_layer = layer_nodes[layer_idx]
                dst_layer = layer_nodes[layer_idx + 1]
                
                for src in src_layer:
                    for dst in dst_layer:
                        if (src, dst) not in current_edges:
                            possible_new_edges.append((src, dst))
            
            if possible_new_edges:
                src, dst = random.choice(possible_new_edges)
                edges.append({
                    "src": src,
                    "dst": dst,
                    "sign": random.choice([-1, 1])
                })
        
        # 5. Possibly remove an edge
        if random.random() < mutation_rate and len(edges) > NUM_HIDDEN_LAYERS * 3:  # Keep minimum connectivity
            remove_idx = random.randrange(len(edges))
            edges.pop(remove_idx)
        
        # 6. Possibly flip an edge sign
        if random.random() < mutation_rate and edges:
            flip_idx = random.randrange(len(edges))
            edges[flip_idx]['sign'] *= -1
    
    mutated['nodes'] = nodes
    mutated['edges'] = edges
    
    return mutated


def mutate_individual(args):
    """
    Worker function to mutate an individual in parallel
    
    Args:
        args: Tuple containing (parent, mutation_rate, max_hidden, max_edges, add_edge_prob)
        
    Returns:
        dict: Mutated graph
    """
    parent, mutation_rate, max_hidden, max_edges, add_edge_prob = args
    return mutate_graph(parent, mutation_rate, max_hidden, max_edges, add_edge_prob)