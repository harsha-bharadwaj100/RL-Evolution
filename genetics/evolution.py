import torch
import random
import copy
from typing import List
from .genome import TribalGenome

def mutate_weights(genome: TribalGenome, mutation_rate: float = 0.1, mutation_scale: float = 0.1) -> TribalGenome:
    """
    Applies random Gaussian noise to the PyTorch weights.
    This mutates their logical processing and their "latent language" translation.
    
    Args:
        genome: The TribalGenome to mutate.
        mutation_rate: Probability of each parameter element being mutated.
        mutation_scale: Standard deviation of the Gaussian noise added.
        
    Returns:
        The mutated TribalGenome (modified in-place, but returned for convenience).
    """
    with torch.no_grad():
        for param in genome.parameters():
            # Create a boolean mask for which weights to mutate
            mask = torch.rand_like(param) < mutation_rate
            # Generate Gaussian noise
            noise = torch.randn_like(param) * mutation_scale
            # Apply noise
            param.add_(mask * noise)
    return genome

def mutate_tribe(genome: TribalGenome, possible_tribes: List[str], probability: float = 0.01) -> TribalGenome:
    """
    Flips the self.tribe_id to a different tribe with a low probability,
    simulating a cultural crossover or "spy" event.
    
    Args:
        genome: The TribalGenome.
        possible_tribes: List of all valid tribe string IDs.
        probability: The chance of flipping the tribe ID.
        
    Returns:
        The updated TribalGenome.
    """
    if random.random() < probability:
        other_tribes = [t for t in possible_tribes if t != genome.tribe_id]
        if other_tribes:
            genome.tribe_id = random.choice(other_tribes)
    return genome

def crossover(parent1: TribalGenome, parent2: TribalGenome) -> TribalGenome:
    """
    Takes two parent TribalGenome models and randomly mixes their weight tensors
    to produce a child genome via uniform crossover.
    
    Args:
        parent1: The first parent genome.
        parent2: The second parent genome.
        
    Returns:
        A new child TribalGenome.
    """
    # Create a child genome as a copy of parent1 to inherit structure
    child = copy.deepcopy(parent1)
    
    with torch.no_grad():
        for child_param, p1_param, p2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            # Uniform crossover: randomly pick from p1 or p2 for each weight element
            mask = torch.rand_like(child_param) < 0.5
            child_param.copy_(torch.where(mask, p1_param, p2_param))
            
    # For tribe_id, randomly inherit from one of the parents
    child.tribe_id = random.choice([parent1.tribe_id, parent2.tribe_id])
    
    return child
