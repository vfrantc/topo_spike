"""
Utilities for parallel processing
"""
import torch
import torch.multiprocessing as torch_mp
import random
import numpy as np


def setup_parallel_environment():
    """
    Set up the environment for parallel processing
    """
    try:
        torch_mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method is already set
        pass


def worker_init_fn(worker_id):
    """
    Initialize worker with different random seed
    """
    seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)