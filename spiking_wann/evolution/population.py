"""
Population initialization and management for evolutionary algorithm with support for multiple layers
"""
import torch
import random
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..config import (
    POPULATION_SIZE, TOP_K, MAX_HIDDEN_NEURONS, MAX_EDGES, 
    INITIAL_HIDDEN, INITIAL_EDGES, NUM_WORKERS, MUTATION_RATE,
    ADD_EDGE_PROB, NUM_HIDDEN_LAYERS
)
from .mutation import mutate_graph, mutate_individual


def initialize_population(pop_size, num_inputs, num_outputs, max_hidden=MAX_HIDDEN_NEURONS, 
                         initial_edges=INITIAL_EDGES, initial_hidden=INITIAL_HIDDEN):
    """
    Initialize a population of random graph architectures with multiple layers
    
    Args:
        pop_size (int): Population size
        num_inputs (int): Number of input nodes
        num_outputs (int): Number of output nodes
        max_hidden (int): Maximum number of hidden neurons
        initial_edges (int): Initial number of edges per graph
        initial_hidden (int): Initial number of hidden neurons
    
    Returns:
        list: List of graph dictionaries
    """
    population = []
    
    for _ in range(pop_size):
        # Create nodes
        nodes = []
        
        # Input nodes
        for i in range(num_inputs):
            nodes.append({"id": i, "type": "input", "layer": 0})
        
        # Hidden nodes distributed across layers
        hidden_per_layer = initial_hidden // NUM_HIDDEN_LAYERS
        remaining_hidden = initial_hidden % NUM_HIDDEN_LAYERS
        
        hidden_id_start = num_inputs
        for layer in range(1, NUM_HIDDEN_LAYERS + 1):
            layer_neurons = hidden_per_layer
            if layer == 1:
                layer_neurons += remaining_hidden  # Add remaining neurons to first hidden layer
                
            for i in range(layer_neurons):
                nodes.append({
                    "id": hidden_id_start + i, 
                    "type": "hidden", 
                    "layer": layer
                })
            hidden_id_start += layer_neurons
        
        # Output nodes
        for i in range(num_outputs):
            nodes.append({
                "id": hidden_id_start + i, 
                "type": "output", 
                "layer": NUM_HIDDEN_LAYERS + 1
            })
        
        # Create edges
        edges = []
        
        # Helper function to get nodes by layer
        def get_layer_nodes(layer_num):
            return [node["id"] for node in nodes if node["layer"] == layer_num]
        
        # Connect layers in sequence (including skipping connections)
        for layer in range(NUM_HIDDEN_LAYERS + 1):
            # Source nodes from current layer
            src_layer_nodes = get_layer_nodes(layer)
            
            # Connect to all subsequent layers (allows skipping)
            for target_layer in range(layer + 1, NUM_HIDDEN_LAYERS + 2):
                dst_layer_nodes = get_layer_nodes(target_layer)
                
                # Create possible connections between these layers
                possible_connections = []
                for src in src_layer_nodes:
                    for dst in dst_layer_nodes:
                        possible_connections.append((src, dst))
                
                # For first-to-last layer connections (input to output), limit to a few connections
                if layer == 0 and target_layer == NUM_HIDDEN_LAYERS + 1:
                    # Direct input-to-output connections (limited)
                    num_direct_connections = min(len(possible_connections), num_outputs * 3)
                    if possible_connections:
                        selected = random.sample(possible_connections, num_direct_connections)
                        for src, dst in selected:
                            edges.append({
                                "src": src,
                                "dst": dst,
                                "sign": random.choice([-1, 1])
                            })
                else:
                    # For other layer connections, connect more densely
                    # But reduce connection probability for layers that skip multiple layers
                    connection_prob = 0.5 / (target_layer - layer)  # Probability decreases with layer distance
                    
                    for src, dst in possible_connections:
                        if random.random() < connection_prob:
                            edges.append({
                                "src": src,
                                "dst": dst,
                                "sign": random.choice([-1, 1])
                            })
        
        # Ensure we don't exceed the initial edge count
        if len(edges) > initial_edges:
            edges = random.sample(edges, initial_edges)
        
        # Create graph
        graph = {
            "nodes": nodes,
            "edges": edges
        }
        
        population.append(graph)
    
    return population


def select_and_mutate_parallel(evaluated_pop, top_k=TOP_K, mutation_rate=MUTATION_RATE, population_size=POPULATION_SIZE):
    """
    Select top individuals and create a new population through mutation in parallel
    
    Args:
        evaluated_pop (list): List of (graph, fitness, accuracy) tuples
        top_k (int): Number of top individuals to select
        mutation_rate (float): Mutation rate
        population_size (int): Size of the population to create
    
    Returns:
        list: New population
    """
    # Sort by fitness
    evaluated_pop.sort(key=lambda x: x[1], reverse=True)
    
    # Select top k individuals
    top_individuals = [x[0] for x in evaluated_pop[:top_k]]
    
    # Create new population
    new_population = []
    
    # Always keep the best unchanged
    new_population.append(copy.deepcopy(top_individuals[0]))
    
    # Prepare args for parallel mutation
    args_list = []
    
    # Fill the rest with mutations
    while len(new_population) + len(args_list) < population_size:
        # Select a parent (weighted by fitness rank)
        weights = [top_k - i for i in range(top_k)]
        parent_idx = random.choices(range(top_k), weights=weights, k=1)[0]
        parent = top_individuals[parent_idx]
        
        args_list.append((parent, mutation_rate, MAX_HIDDEN_NEURONS, MAX_EDGES, ADD_EDGE_PROB))
    
    # Use ThreadPoolExecutor for parallel mutation
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        mutated_individuals = list(executor.map(mutate_individual, args_list))
    
    new_population.extend(mutated_individuals)
    
    return new_population


def worker_init_fn(worker_id):
    """
    Initialize worker with different random seed
    """
    seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)