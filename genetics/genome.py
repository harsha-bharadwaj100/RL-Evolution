import torch
import torch.nn as nn
from typing import Tuple, Optional

class TribalGenome(nn.Module):
    """
    The biological brain of the Tribal RAG agent.
    Optimized for fast, multi-agent CPU inference.
    """
    def __init__(
        self, 
        state_dim: int, 
        latent_dim: int, 
        context_dim: int, 
        action_dim: int = 4, 
        tribe_id: str = "Tribe_A"
    ):
        super().__init__()
        self.tribe_id = tribe_id
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        
        # The Translator (Latent Projection):
        # A lightweight layer that takes raw environment state and compresses it.
        # This output acts as the raw embedding for ChromaDB.
        self.translator = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.Tanh() # Using Tanh to keep the latent vector bounded between -1 and 1, standard for embeddings
        )
        
        # The Query Gate:
        # Decides if the agent should execute a Query Culture action based on the latent state.
        self.query_gate = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid()
        )
        
        # The Action Layer:
        # Final layer that outputs movement actions (Up, Down, Left, Right).
        self.action_layer = nn.Linear(latent_dim + context_dim, action_dim)

    def forward(self, state: torch.Tensor, cultural_action_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-stage forward pass for the TribalGenome.
        
        Args:
            state: The raw environment state tensor [batch_size, state_dim] or [state_dim].
            cultural_action_context: The retrieved action context from ChromaDB (if query triggered).
                                     If None, it assumes no query was made.
                                     
        Returns:
            latent_vector: The compressed representation (sent to ChromaDB).
            query_prob: The probability of executing a Query Culture action.
            action_logits: The logits for the movement actions.
        """
        # Pass state through Latent Projection
        latent_vector = self.translator(state)
        
        # Pass latent vector through Query Gate
        query_prob = self.query_gate(latent_vector)
        
        # Check if we have a cultural action context (e.g., query triggered in environment loop)
        if cultural_action_context is not None:
            # Concatenate cultural_action_context to latent_vector
            combined_input = torch.cat([latent_vector, cultural_action_context], dim=-1)
        else:
            # If no query, concatenate zeros to maintain dimension for the Action Layer
            device = latent_vector.device
            # Ensure zero context matches the batch dimensions of latent_vector
            zero_context = torch.zeros(*latent_vector.shape[:-1], self.context_dim, device=device)
            combined_input = torch.cat([latent_vector, zero_context], dim=-1)
            
        # Action Layer
        action_logits = self.action_layer(combined_input)
        
        return latent_vector, query_prob, action_logits
