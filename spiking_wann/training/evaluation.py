"""
Evaluation functions for Spiking WANNs with layer support
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from spikingjelly.activation_based import functional
from concurrent.futures import ProcessPoolExecutor
from torchvision import datasets, transforms

from ..config import BATCH_SIZE, T, ALPHA, NUM_WORKERS, DEVICE, NUM_INPUTS, NUM_OUTPUTS
from ..model.network import SpikingWANN
from ..model.graph_utils import count_network_size, analyze_layer_connectivity


def evaluate_individual(args):
    """
    Worker function to evaluate a single individual in parallel
    
    Args:
        args: Tuple containing (graph, weight_value, dataset, device_id, batch_size, num_steps, alpha)
        
    Returns:
        tuple: (graph, fitness, accuracy)
    """
    graph, weight_value, dataset, device_id, batch_size, num_steps, alpha = args
    
    # Set device for this worker
    if torch.cuda.is_available() and device_id >= 0:
        # For multi-GPU systems, assign different GPUs to different workers
        device = torch.device(f'cuda:{device_id % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
    
    # Create data loader for this worker
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Build model
    model = SpikingWANN(graph, weight_value).to(device)
    
    # Evaluate accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.view(-1, inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).to(device)
            targets = targets.to(device)
            
            # Reset network states
            functional.reset_net(model)
            
            # Forward pass
            output_spikes = model(inputs, num_steps)
            
            # Get predictions
            predictions = output_spikes.argmax(dim=1)
            
            # Calculate accuracy
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total if total > 0 else 0
    
    # Count nodes and edges for complexity penalty
    num_hidden, num_edges = count_network_size(graph)
    
    # Get layer connectivity stats
    layer_stats = analyze_layer_connectivity(graph)
    skip_connections = layer_stats['skip_connections']
    
    # Compute fitness with a small bonus for skip connections
    # This encourages exploration of layer-skipping architectures
    skip_bonus = 0.001 * skip_connections
    fitness = 100 * accuracy - alpha * (num_hidden + num_edges) + skip_bonus
    
    return (graph, fitness, accuracy)


def evaluate_population_parallel(population, weight_value, dataset, batch_size=BATCH_SIZE, num_steps=T, alpha=ALPHA):
    """
    Evaluate fitness for all individuals in the population in parallel
    
    Args:
        population (list): List of graph dictionaries
        weight_value (float): Weight magnitude
        dataset: PyTorch dataset
        batch_size (int): Batch size for evaluation
        num_steps (int): Number of simulation timesteps
        alpha (float): Complexity penalty factor
    
    Returns:
        list: List of (graph, fitness, accuracy) tuples
    """
    # Prepare arguments for parallel processing
    args_list = []
    
    for i, graph in enumerate(population):
        # Assign different device IDs in a round-robin fashion if multiple GPUs
        device_id = i % max(1, torch.cuda.device_count()) if torch.cuda.is_available() else -1
        args_list.append((graph, weight_value, dataset, device_id, batch_size, num_steps, alpha))
    
    # Use ProcessPoolExecutor for CPU parallelism
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(evaluate_individual, args_list), total=len(args_list)))
    
    # Sort by fitness (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


def evaluate_best_model(best_graph, weight_value=1.0):
    """
    Evaluate the best model on the full MNIST test set
    
    Args:
        best_graph (dict): The best graph architecture
        weight_value (float): Weight magnitude
    
    Returns:
        float: Test accuracy
    """    
    # Load MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build model from best graph
    model = SpikingWANN(best_graph, weight_value).to(DEVICE)
    
    # Evaluate
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.view(-1, NUM_INPUTS).to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Reset network states
            functional.reset_net(model)
            
            # Forward pass
            output_spikes = model(inputs)
            
            # Get predictions
            predictions = output_spikes.argmax(dim=1)
            
            # Calculate accuracy
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    return correct / total