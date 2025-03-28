"""
Configuration parameters for the Spiking Weight-Agnostic Neural Network with multilayer support
"""
import torch
import multiprocessing as mp

# Constants for the network architecture
NUM_INPUTS = 784  # 28x28 MNIST images
NUM_OUTPUTS = 10  # 10 digits
T = 100  # Number of timesteps for simulation
BATCH_SIZE = 64

# New parameter for multiple hidden layers
NUM_HIDDEN_LAYERS = 3  # Number of hidden layers

# Parameters for the spiking neurons
BETA = 0.9  # Decay factor
THRESHOLD = 1.0  # Firing threshold
V_RESET = 0.0  # Reset potential

# Parameters for the evolutionary algorithm
POPULATION_SIZE = 16
NUM_GENERATIONS = 100
TOP_K = 5
MUTATION_RATE = 0.3
WEIGHT_VALUES = [1.0]  # Could be expanded to [0.5, 1.0, 2.0]
ALPHA = 0.0001  # Complexity penalty factor - MUCH smaller to prioritize accuracy
ADD_EDGE_PROB = 0.6  # Higher probability to add edges during mutation
LAYER_SKIP_PROB = 0.3  # Probability of creating skip-layer connections during mutation

# Maximum network size constraints
MAX_HIDDEN_NEURONS = 60  # Increased for more neurons across multiple layers
INITIAL_HIDDEN = 30  # Start with more hidden neurons (distributed across layers)
MAX_EDGES = 600  # Increased maximum edges for more complex connectivity
INITIAL_EDGES = 250  # Start with more edges

# Get number of available cores for parallel processing
NUM_CORES = mp.cpu_count()
NUM_WORKERS = NUM_CORES

# CUDA settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')