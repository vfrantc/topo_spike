"""
Population initialization and management for evolutionary algorithm
"""
import torch
import random
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..config import (
    POPULATION_SIZE, TOP_K, MAX_HIDDEN_NEURONS, MAX_EDGES, 
    INITIAL_HIDDEN, INITIAL_EDGES, NUM_WORKERS, MUTATION_RATE,
    ADD_EDGE_PROB
)
from .mutation import mutate_graph, mutate_individual


def initialize_population(pop_size, num_inputs, num_outputs, max_hidden=MAX_HIDDEN_NEURONS, 
                         initial_edges=INITIAL_EDGES, initial_hidden=INITIAL_HIDDEN):
    """
    Initialize a population of random graph architectures
    
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
        # Use a fixed number of hidden neurons to start
        num_hidden = initial_hidden
        
        # Create nodes
        nodes = []
        # Input nodes
        for i in range(num_inputs):
            nodes.append({"id": i, "type": "input"})
        
        # Hidden nodes
        for i in range(num_hidden):
            nodes.append({"id": num_inputs + i, "type": "hidden"})
        
        # Output nodes
        for i in range(num_outputs):
            nodes.append({"id": num_inputs + num_hidden + i, "type": "output"})
        
        # Create random edges
        edges = []
        
        # Directly connect some inputs to outputs for initial signal flow
        for i in range(num_outputs):
            output_id = num_inputs + num_hidden + i
            # Connect a few random inputs to each output
            for _ in range(3):  # Connect 3 random inputs to each output
                input_id = random.randint(0, num_inputs - 1)
                edges.append({
                    "src": input_id,
                    "dst": output_id,
                    "sign": random.choice([-1, 1])
                })
                
        # Now add random connections
        possible_connections = []
        
        # Input to hidden connections
        for src in range(num_inputs):
            for dst in range(num_inputs, num_inputs + num_hidden):
                possible_connections.append((src, dst))
        
        # Hidden to hidden connections
        for src in range(num_inputs, num_inputs + num_hidden):
            for dst in range(num_inputs, num_inputs + num_hidden):
                if src != dst:  # No self-connections
                    possible_connections.append((src, dst))
        
        # Hidden to output connections
        for src in range(num_inputs, num_inputs + num_hidden):
            for dst in range(num_inputs + num_hidden, num_inputs + num_hidden + num_outputs):
                possible_connections.append((src, dst))
        
        # Randomly select initial edges (ensuring we don't exceed the number of possible connections)
        num_edges = min(initial_edges, len(possible_connections))
        if len(possible_connections) > 0:
            selected_connections = random.sample(possible_connections, num_edges)
            
            # Create edge dictionaries
            for src, dst in selected_connections:
                edges.append({
                    "src": src,
                    "dst": dst,
                    "sign": random.choice([-1, 1])
                })
        
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
    # (Process pool not needed here since mutation is not compute-intensive)
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