"""
SpikingWANN model definition with support for multiple layers and skip connections
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import spikingjelly.activation_based as snn
from spikingjelly.activation_based import functional, surrogate

from ..config import T, BETA, THRESHOLD, V_RESET, NUM_OUTPUTS


class SpikingWANN(nn.Module):
    def __init__(self, graph, weight_value=1.0):
        """
        Build a spiking neural network from a graph representation with layer support
        
        Args:
            graph (dict): A dictionary with 'nodes' and 'edges' lists
            weight_value (float): The uniform weight magnitude to use
        """
        super(SpikingWANN, self).__init__()
        
        self.graph = graph
        self.weight_value = weight_value
        
        # Get node types and layer information
        self.node_types = {}
        self.node_layers = {}
        for node in graph['nodes']:
            self.node_types[node['id']] = node['type']
            self.node_layers[node['id']] = node.get('layer', 0)
        
        # Create LIF neurons for hidden and output nodes
        self.lif_nodes = nn.ModuleDict()
        for node in graph['nodes']:
            if node['type'] in ['hidden', 'output']:
                neuron_id = str(node['id'])
                # Create different neurons parameters based on layer if needed
                # For now, all neurons use the same parameters
                self.lif_nodes[neuron_id] = snn.neuron.LIFNode(
                    tau=1.0 / (1.0 - BETA),
                    v_threshold=THRESHOLD,
                    v_reset=V_RESET,
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True
                )
        
        # Create incoming connection dictionaries
        self.connections = {}
        
        # Organize edges by destination node
        for edge in graph['edges']:
            dst = str(edge['dst'])
            src = str(edge['src'])
            sign = edge['sign']
            
            if dst not in self.connections:
                self.connections[dst] = {'srcs': [], 'weights': []}
                
            self.connections[dst]['srcs'].append(src)
            self.connections[dst]['weights'].append(weight_value * sign)
            
        # Topological ordering for forward pass
        self.node_order = self._compute_topological_order()
        
        # Organize nodes by layer for more efficient processing
        self._organize_nodes_by_layer()
    
    def _compute_topological_order(self):
        """Compute a topological ordering of nodes for the forward pass"""
        # Create a directed graph
        G = nx.DiGraph()
        for node in self.graph['nodes']:
            G.add_node(str(node['id']))
            
        for edge in self.graph['edges']:
            G.add_edge(str(edge['src']), str(edge['dst']))
            
        # Get topological order
        try:
            return list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # If graph has cycles, use a fallback ordering by layer
            sorted_nodes = sorted(
                [str(n['id']) for n in self.graph['nodes']], 
                key=lambda nid: (self.node_layers[int(nid)], int(nid))
            )
            return sorted_nodes
    
    def _organize_nodes_by_layer(self):
        """Organize nodes by layer for more efficient processing"""
        # Find the maximum layer
        max_layer = max(self.node_layers.values())
        
        # Create a list of lists, where each inner list contains nodes for a layer
        self.nodes_by_layer = [[] for _ in range(max_layer + 1)]
        
        # Populate the lists
        for node_id, layer in self.node_layers.items():
            self.nodes_by_layer[layer].append(str(node_id))
        
        # Sort nodes within each layer by ID for consistency
        for layer in range(len(self.nodes_by_layer)):
            self.nodes_by_layer[layer].sort(key=lambda x: int(x))
        
        # Also get input and output nodes
        self.input_nodes = [str(node['id']) for node in self.graph['nodes'] if node['type'] == 'input']
        self.output_nodes = [str(node['id']) for node in self.graph['nodes'] if node['type'] == 'output']
    
    def reset_states(self):
        """Reset all neuron states"""
        functional.reset_net(self)
            
    def forward(self, x, num_steps=T):
        """
        Forward pass through the spiking network
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_inputs]
            num_steps (int): Number of timesteps to simulate
        
        Returns:
            torch.Tensor: Output spike counts for each class
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Explicitly reset all neuron states at the beginning of each forward pass
        functional.reset_net(self)
        
        # Initialize storage for output spikes
        output_spikes = torch.zeros(batch_size, NUM_OUTPUTS, device=device)
        
        # Create Poisson spike encodings of the input
        # We'll use the pixel intensities to set firing probabilities
        encoded_x = []
        for t in range(num_steps):
            # Each pixel intensity determines the probability of a spike
            spike_prob = x.clone()  # Use pixel values (0-1) as probabilities
            # Generate spikes using Bernoulli trials
            spikes = torch.bernoulli(spike_prob)
            encoded_x.append(spikes)
        
        # Stack over time dimension
        encoded_x = torch.stack(encoded_x)  # Shape: [T, batch_size, num_inputs]
        
        # Simulation loop
        for t in range(num_steps):
            # Set input spikes for this timestep
            current_inputs = {}
            for i, node_id in enumerate(self.input_nodes):
                if i < x.shape[1]:  # Ensure we don't exceed input dimensions
                    current_inputs[node_id] = encoded_x[t, :, i].float()
                else:
                    current_inputs[node_id] = torch.zeros(batch_size, device=device)
            
            # Process nodes by layer to handle skip connections properly
            for layer_idx in range(1, len(self.nodes_by_layer)):  # Skip input layer (0)
                layer_nodes = self.nodes_by_layer[layer_idx]
                
                # Process all nodes in this layer
                for node_id in layer_nodes:
                    # Initialize aggregated input for this node
                    agg_input = torch.zeros(batch_size, device=device)
                    
                    # Get incoming connections
                    if node_id in self.connections:
                        incoming = self.connections[node_id]
                        
                        # Aggregate inputs from source nodes
                        for i, src_id in enumerate(incoming['srcs']):
                            weight = incoming['weights'][i]
                            if src_id in current_inputs:
                                agg_input += weight * current_inputs[src_id]
                    
                    # Update LIF neuron
                    spike_out = self.lif_nodes[node_id](agg_input)
                    current_inputs[node_id] = spike_out
                    
                    # For output nodes, accumulate spikes
                    if node_id in self.output_nodes:
                        idx = self.output_nodes.index(node_id)
                        if idx < NUM_OUTPUTS:
                            output_spikes[:, idx] += spike_out
        
        return output_spikes