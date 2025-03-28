"""
Mutation operations for graph evolution
"""
import random
import copy

from ..model.graph_utils import get_node_ids_by_type
from ..config import MUTATION_RATE, MAX_HIDDEN_NEURONS, MAX_EDGES, ADD_EDGE_PROB


def mutate_graph(graph, mutation_rate=MUTATION_RATE, max_hidden=MAX_HIDDEN_NEURONS, 
                max_edges=MAX_EDGES, add_edge_prob=ADD_EDGE_PROB):
    """
    Mutate a graph architecture
    
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
    
    # Find max node id
    max_id = max([node['id'] for node in nodes])
    
    # Current edge connections as tuples for quick lookup
    current_edges = set((edge['src'], edge['dst']) for edge in edges)
    
    # For the first few generations, prioritize adding edges to increase connectivity
    if random.random() < add_edge_prob and len(edges) < max_edges:
        # Create potential new edges (ones that don't already exist)
        possible_new_edges = []
        
        # Input to hidden connections
        for src in input_ids:
            for dst in hidden_ids:
                if (src, dst) not in current_edges:
                    possible_new_edges.append((src, dst))
        
        # Hidden to hidden connections
        for src in hidden_ids:
            for dst in hidden_ids:
                if src != dst and (src, dst) not in current_edges:
                    possible_new_edges.append((src, dst))
        
        # Hidden to output connections
        for src in hidden_ids:
            for dst in output_ids:
                if (src, dst) not in current_edges:
                    possible_new_edges.append((src, dst))
                    
        # Direct input to output connections
        for src in input_ids:
            for dst in output_ids:
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
            nodes.append({"id": new_id, "type": "hidden"})
            hidden_ids.append(new_id)
            
            # Connect the new neuron to the network
            # Connect to inputs
            num_inputs_to_connect = random.randint(1, min(5, len(input_ids)))
            for _ in range(num_inputs_to_connect):
                src = random.choice(input_ids)
                edges.append({
                    "src": src,
                    "dst": new_id,
                    "sign": random.choice([-1, 1])
                })
                
            # Connect to outputs
            num_outputs_to_connect = random.randint(1, min(3, len(output_ids)))
            for _ in range(num_outputs_to_connect):
                dst = random.choice(output_ids)
                edges.append({
                    "src": new_id,
                    "dst": dst,
                    "sign": random.choice([-1, 1])
                })
        
        # 2. Possibly remove a hidden neuron
        if random.random() < mutation_rate and len(hidden_ids) > 1:
            remove_id = random.choice(hidden_ids)
            nodes = [node for node in nodes if node['id'] != remove_id]
            edges = [edge for edge in edges if edge['src'] != remove_id and edge['dst'] != remove_id]
        
        # 3. Possibly add an edge (lower priority than the special case above)
        if random.random() < mutation_rate/2 and len(edges) < max_edges:
            # Create a new edge
            all_srcs = input_ids + hidden_ids
            all_dsts = hidden_ids + output_ids
            
            # Try to find a new connection that doesn't already exist
            for _ in range(10):  # Try a few times
                src = random.choice(all_srcs)
                dst = random.choice(all_dsts)
                
                # Check if this edge already exists
                edge_exists = (src, dst) in current_edges
                
                if not edge_exists:
                    edges.append({
                        "src": src, 
                        "dst": dst,
                        "sign": random.choice([-1, 1])
                    })
                    break
        
        # 4. Possibly remove an edge
        if random.random() < mutation_rate and len(edges) > 5:  # Keep at least 5 edges
            remove_idx = random.randrange(len(edges))
            edges.pop(remove_idx)
        
        # 5. Possibly flip an edge sign
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