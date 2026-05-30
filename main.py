import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from env.grid_world import TribalGridWorld
from genetics.genome import TribalGenome
from genetics.evolution import mutate_weights, mutate_tribe, crossover
from culture.chroma_manager import ChromaManager

def main():
    print("Initializing Tribal RAG Framework...")
    
    # Configuration
    NUM_GENERATIONS = 300
    LIFESPAN_STEPS = 50
    POPULATION_SIZE = 64
    GRID_SIZE = 10
    FOV_SIZE = 5
    STATE_DIM = FOV_SIZE * FOV_SIZE
    LATENT_DIM = 8
    CONTEXT_DIM = 4 # Action one-hot size
    ACTION_DIM = 4  # Movement options
    MAX_CULTURE_CAPACITY = 500
    NUM_TRIBES = 8
    
    TRIBES = [f"Tribe_{i}" for i in range(1, NUM_TRIBES + 1)]
    
    # TensorBoard Initialization
    writer = SummaryWriter("./runs/tribal_rag_experiment")
    
    # 1. Initialize ChromaDB
    chroma_manager = ChromaManager(persist_directory="./.chroma_db")
    # Clean databases for fresh run
    for tribe in TRIBES:
        chroma_manager.clear_tribe_collection(tribe)
    
    # 2. Initialize Population (evenly distributed)
    population = []
    for i in range(POPULATION_SIZE):
        tribe = TRIBES[i % NUM_TRIBES]
        genome = TribalGenome(
            state_dim=STATE_DIM,
            latent_dim=LATENT_DIM,
            context_dim=CONTEXT_DIM,
            action_dim=ACTION_DIM,
            tribe_id=tribe
        )
        population.append(genome)
        
    # 3. Initialize Environment
    env = TribalGridWorld(num_agents=POPULATION_SIZE, grid_size=GRID_SIZE, fov_size=FOV_SIZE)
    
    for gen in range(NUM_GENERATIONS):
        # Paradigm Shift!
        if gen == 150:
            env.invert_rules = True
            print("\n" + "="*50)
            print("*** PARADIGM SHIFT: ENVIRONMENT RULES INVERTED ***")
            print("="*50 + "\n")
            
        print(f"\n--- Generation {gen + 1} ---")
        
        observations, _ = env.reset()
        agent_names = env.agents
        agent_genomes = {name: genome for name, genome in zip(agent_names, population)}
        agent_rewards = {name: 0.0 for name in agent_names}
        
        total_queries = 0
        total_actions = 0
        
        # The Inner Loop (Lifespan)
        for step in range(LIFESPAN_STEPS):
            actions = {}
            step_latents = {}
            step_genomes = {}
            
            # Action Selection for all agents
            for agent_name, obs_array in observations.items():
                genome = agent_genomes[agent_name]
                # Cast NumPy observation to PyTorch tensor
                state_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    latent, query_prob, action_logits = genome(state_tensor)
                    step_latents[agent_name] = latent.squeeze(0).tolist()
                    step_genomes[agent_name] = genome
                    
                    # The RAG Integration
                    total_actions += 1
                    if query_prob.item() > 0.5:
                        total_queries += 1
                        query_results = chroma_manager.query_culture(
                            tribe_id=genome.tribe_id,
                            latent_state_vector=step_latents[agent_name],
                            n_results=1,
                            min_reward_threshold=0.1 # Look for strictly positive outcomes
                        )
                        
                        # If we found a matching cultural memory
                        if query_results and len(query_results["ids"][0]) > 0:
                            retrieved_action = query_results["metadatas"][0][0]["action"]
                            
                            # Convert retrieved discrete action into one-hot context tensor
                            context_tensor = F.one_hot(
                                torch.tensor(retrieved_action), 
                                num_classes=CONTEXT_DIM
                            ).float().unsqueeze(0)
                            
                            # Re-run action layer with cultural context
                            _, _, action_logits = genome(state_tensor, cultural_action_context=context_tensor)
                    
                    # Choose final action based on highest logit
                    final_action = torch.argmax(action_logits, dim=-1).item()
                    actions[agent_name] = final_action
            
            # Step Environment
            next_obs, rewards, _, _, _ = env.step(actions)
            
            # Execution & Memory Update
            for agent_name, reward in rewards.items():
                agent_rewards[agent_name] += reward
                
                # If reward is positive, embed the experience in the tribe's culture database
                if reward > 0.0:
                    genome = step_genomes[agent_name]
                    experience_id = f"gen{gen}_step{step}_{agent_name}"
                    chroma_manager.write_experience(
                        tribe_id=genome.tribe_id,
                        experience_id=experience_id,
                        latent_state_vector=step_latents[agent_name],
                        action=actions[agent_name],
                        reward=reward
                    )
                    
            observations = next_obs
            
        # Logging & Pruning (Dynamic for 8 tribes)
        avg_fitnesses = []
        for tribe in TRIBES:
            # Fitness
            tribe_rewards = [r for n, r in agent_rewards.items() if agent_genomes[n].tribe_id == tribe]
            avg_r = sum(tribe_rewards) / len(tribe_rewards) if tribe_rewards else 0.0
            avg_fitnesses.append(f"{tribe}: {avg_r:.2f}")
            writer.add_scalar(f'Fitness/{tribe}', avg_r, gen)
            
            # DB Size
            try:
                db_size = chroma_manager.get_or_create_tribe_collection(tribe).count()
            except Exception:
                db_size = 0
            writer.add_scalar(f'Culture/Database_Size/{tribe}', db_size, gen)
            
            # Pruning
            chroma_manager.prune_database(tribe, MAX_CULTURE_CAPACITY)
            
        print("Average Fitness -> " + " | ".join(avg_fitnesses))
        
        query_frequency = total_queries / max(1, total_actions)
        writer.add_scalar('Culture/Query_Frequency', query_frequency, gen)
        
        # Culling & Breeding (Evolution)
        sorted_agents = sorted(agent_names, key=lambda n: agent_rewards[n], reverse=True)
        survivors = [agent_genomes[n] for n in sorted_agents[:POPULATION_SIZE // 2]]
        
        new_population = list(survivors) # Retain top 50%
        
        # Breed new 50%
        for _ in range(POPULATION_SIZE // 2):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = crossover(parent1, parent2)
            child = mutate_weights(child)
            child = mutate_tribe(child, TRIBES)
            new_population.append(child)
            
        population = new_population
        
    writer.close()

if __name__ == "__main__":
    main()
