"""
Main script for training a Spiking Weight-Agnostic Neural Network with multiple layers
"""
import torch
import random
import numpy as np
import copy
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from spiking_wann.config import (
    DEVICE, POPULATION_SIZE, NUM_GENERATIONS, 
    NUM_INPUTS, NUM_OUTPUTS, BATCH_SIZE, NUM_HIDDEN_LAYERS
)
from spiking_wann.model import SpikingWANN
from spiking_wann.evolution import initialize_population, select_and_mutate_parallel
from spiking_wann.training import evaluate_population_parallel, evaluate_best_model, setup_parallel_environment
from spiking_wann.utils import visualize_graph, visualize_fitness_history, visualize_layer_connectivity
from spiking_wann.model.graph_utils import analyze_layer_connectivity


def main():
    # Set up parallel environment
    setup_parallel_environment()
    
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Using device: {DEVICE}")
    print(f"Using {NUM_HIDDEN_LAYERS} hidden layers")
    
    # Parameters for training
    num_generations = NUM_GENERATIONS
    population_size = POPULATION_SIZE
    weight_value = 1.0
    subset_size = 5000
    
    # Number of top networks to save per generation
    top_networks_to_save = 3
    
    # Create output directory
    output_dir = 'out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use a subset for faster evaluation during evolution
    indices = torch.randperm(len(train_dataset))[:subset_size]
    subset_dataset = Subset(train_dataset, indices)
    
    # Create test loader for final evaluation
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize population
    print("Initializing population...")
    population = initialize_population(population_size, NUM_INPUTS, NUM_OUTPUTS)

    # Fitness history
    fitness_history = []
    best_graph = None
    best_fitness = -float('inf')
    best_accuracy = 0

    for generation in range(num_generations):
        # Create generation-specific directory
        gen_dir = os.path.join(output_dir, f'generation_{generation+1}')
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
            
        print(f"\nGeneration {generation+1}/{num_generations}")
        
        # Evaluate population in parallel
        print("Evaluating population...")
        evaluated_pop = evaluate_population_parallel(population, weight_value, subset_dataset)
        
        # Update best individual
        if evaluated_pop[0][1] > best_fitness:
            best_graph = copy.deepcopy(evaluated_pop[0][0])
            best_fitness = evaluated_pop[0][1]
            best_accuracy = evaluated_pop[0][2]
        
        # Record fitness
        gen_fitness = [f for _, f, _ in evaluated_pop]
        gen_accuracy = [a for _, _, a in evaluated_pop]
        fitness_history.append((np.mean(gen_fitness), np.max(gen_fitness), np.mean(gen_accuracy), np.max(gen_accuracy)))
        
        # Print statistics
        print(f"Best fitness: {evaluated_pop[0][1]:.4f}")
        print(f"Mean fitness: {np.mean(gen_fitness):.4f}")
        best_hidden = len([n for n in best_graph['nodes'] if n['type'] == 'hidden'])
        print(f"Best network: {len(best_graph['nodes'])} nodes ({best_hidden} hidden), {len(best_graph['edges'])} edges")
        print(f"Best accuracy: {best_accuracy:.4f}")
        
        # Analyze layer connectivity for the best network
        layer_stats = analyze_layer_connectivity(best_graph)
        print(f"Skip connections: {layer_stats['skip_connections']}")
        print(f"Layer sizes: {layer_stats['layer_sizes']}")
        
        # Save top networks for this generation
        print(f"Saving top {top_networks_to_save} networks...")
        for i in range(min(top_networks_to_save, len(evaluated_pop))):
            network_graph = evaluated_pop[i][0]
            fitness_value = evaluated_pop[i][1]
            accuracy_value = evaluated_pop[i][2]
            
            # Generate informative filename
            hidden_count = len([n for n in network_graph['nodes'] if n['type'] == 'hidden'])
            edge_count = len(network_graph['edges'])
            filename = f"rank_{i+1}_fit_{fitness_value:.4f}_acc_{accuracy_value:.4f}_nodes_{len(network_graph['nodes'])}_hidden_{hidden_count}_edges_{edge_count}"
            
            # Save network visualization
            network_file = os.path.join(gen_dir, f"{filename}.png")
            visualize_graph(network_graph, filename=network_file)
            
            # For the best network, also create layer connectivity visualization
            if i == 0:
                connectivity_file = os.path.join(gen_dir, f"{filename}_layer_connectivity.png")
                visualize_layer_connectivity(network_graph, filename=connectivity_file)
            
            # Save current generation fitness history
            fitness_file = os.path.join(gen_dir, "fitness_so_far.png")
            visualize_fitness_history(fitness_history, filename=fitness_file)
        
        # Create next generation in parallel
        if generation < num_generations - 1:  # Skip for the last generation
            print("Creating next generation...")
            population = select_and_mutate_parallel(evaluated_pop)

    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    test_accuracy = evaluate_best_model(best_graph, weight_value)
    print(f"Best model test accuracy: {test_accuracy:.4f}")

    # Save final results
    print("Generating final visualizations...")
    visualize_graph(best_graph, filename=os.path.join(output_dir, "best_wann_graph.png"))
    visualize_fitness_history(fitness_history, filename=os.path.join(output_dir, "fitness_history.png"))
    visualize_layer_connectivity(best_graph, filename=os.path.join(output_dir, "layer_connectivity.png"))
    
    # Save the best network with test accuracy info
    best_hidden = len([n for n in best_graph['nodes'] if n['type'] == 'hidden'])
    layer_stats = analyze_layer_connectivity(best_graph)
    skip_connections = layer_stats['skip_connections']
    
    final_best_file = os.path.join(
        output_dir, 
        f"final_best_acc_{test_accuracy:.4f}_nodes_{len(best_graph['nodes'])}_hidden_{best_hidden}_edges_{len(best_graph['edges'])}_skips_{skip_connections}.png"
    )
    visualize_graph(best_graph, filename=final_best_file)
    
    print("Done!")


if __name__ == '__main__':
    main()