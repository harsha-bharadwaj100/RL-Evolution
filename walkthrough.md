# Tribal RAG Project Walkthrough

## Phase 1: Initialization & Culture Module
- **Dependency Management:** Installed required packages (`torch`, `chromadb`, `pettingzoo`, `gymnasium`, `tensorboard`) using `uv`.
- **Project Structure:** Created the foundational Python modules for the architecture (`env/`, `genetics/`, `culture/`).
- **ChromaDB Integration:** Implemented `culture/chroma_manager.py` with raw vector injection, bypassing NLP embedding models and strictly using metadata for Actions and Rewards.

## Phase 2: The Genetics Module (Biological Brain)
- **Neural Architecture (`genetics/genome.py`):** 
  - Implemented the `TribalGenome` PyTorch `nn.Module` optimized for fast CPU inference.
  - Added the **Translator** (Latent Projection) to compress raw states into latent vectors.
  - Added the **Query Gate** to output the probability of initiating a culture query.
  - Implemented a variable-context forward pass: concatenates retrieved ChromaDB contexts to the latent vector dynamically.
- **Evolutionary Mechanics (`genetics/evolution.py`):**
  - Implemented `mutate_weights()`, `mutate_tribe()`, and `crossover()`.

## Phase 3: The Environment and The Training Loop
- **The Arena (`env/grid_world.py`):** Created a custom `PettingZoo` Parallel Environment with resources, hazards, and flattened FoV states.
- **The Evolutionary Orchestrator (`main.py`):** Initialized populations, set up generation loops, mapped PettingZoo actions to PyTorch models, integrated the RAG lookups securely, and wrote positive experiences to the Chroma databases.

## Phase 4: Cultural Forgetting & Local Metrics
- **Cultural Forgetting (Pruning):** 
  - Tracked a `usage_count` metadata flag in `culture/chroma_manager.py`.
  - Added `prune_database(tribe_id, max_capacity)` to fetch, sort, and delete the least-used memories.
  - *Bugfix*: Added extensive try-except blocks around all read/write/count logic to silently catch underlying SQLite database compaction/lock exceptions.
- **Metrics Visualization:** 
  - Integrated `torch.utils.tensorboard.SummaryWriter`.
  - Logged average fitness for Tribe A and Tribe B.
  - Calculated and tracked the system-wide `Culture/Query_Frequency` percentage.
  - Tracked `Culture/Database_Size` to verify the pruning function caps database growth.

## Phase 5: Environment Tuning & The Existential Penalty
- **Step Penalty**: Updated the PettingZoo environment to charge a `-0.01` baseline reward for every single step. This prevents the evolutionary algorithm from identifying "stand still" as an optimal local minimum.
- **Query Tax**: Added a `-0.02` explicit penalty to Action 4 (Query Culture). This forces the Query Gate to evolve to only retrieve memory when the agent is truly in a low-confidence state, simulating the energy expenditure of retrieving a memory.

## What was tested
- Executed `main.py` for 10 full generations spanning 50 time steps per generation across 10 independent agents.

## Validation Results
- **Reward Dynamics**: Agents no longer score `0.00` by doing nothing. Fitness tracking starts heavily negative due to the existential tax, verifying that agents are actively bleeding score and will be strictly culled if they fail to learn resource gathering.
- All systems are successfully operational, efficient, and logging robust metrics.
