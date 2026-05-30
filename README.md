# 🧬 Tribal RAG: Evolutionary Reinforcement Learning

> **What happens when Artificial Intelligence develops a culture?** > 
> Tribal RAG is a custom Artificial Life framework that bridges **Darwinian genetic evolution** with **Lamarckian cultural memory**. By combining PyTorch neuroevolution with a ChromaDB vector database, this project proves that AI agents who can dynamically read and write to a shared "tribal history" can survive apocalyptic environmental changes vastly faster than agents relying purely on biological genetics.

---

## 🧠 Core Architecture: The Split-Brain

At the heart of the framework is the `EvaluativeBrain`, a custom PyTorch Actor-Critic neural network designed to separate instinct from wisdom.

* **The Actor (Instinct):** Evolves purely through Darwinian mutation. It dictates physical movement in the grid world.
* **The Critic (Wisdom):** Learns via Backpropagation during the agent's lifetime and is passed down *perfectly intact* to offspring (Lamarckian inheritance). It evaluates the grid and predicts the future reward.
* **The Instinct Veto (RAG Gate):** If the Critic predicts a negative or low-confidence outcome (Score < 0.2), the agent suppresses its biological instinct and queries the **ChromaDB Cultural Vector Database** for ancestral advice.

## ⚙️ Key System Mechanics

* **Dynamic Elite Publishing (A2C Baselines):** To prevent database bloat, the system calculates the fitness distribution of every generation. Only agents in the **90th Percentile** are granted permission to write their short-term memories to the permanent cultural database.
* **Cultural Forgetting (Pruning):** When the database exceeds 2,000 memories, the tribe mathematically sorts and "forgets" the worst 500 strategies (lowest reward + lowest usage count).
* **The Paradigm Shift (Gen 50 Apocalypse):** At Generation 50, the environment rules invert. Food becomes poisonous, and hazards become nutritious. This proves the hypothesis: *Cultural adaptation outpaces genetic adaptation.*
* **I/O Hash Caching:** A localized Short-Term Memoization Cache intercepts O(log N) ChromaDB queries and turns them into O(1) dictionary lookups, eliminating Disk I/O bottlenecks and boosting GPU efficiency by >80%.

---

## 💀 The Evolutionary Journey (Documented Failures)

This architecture was built organically. Along the way, the agents discovered several highly complex, unintended emergent behaviors that required systemic engineering to fix:

1.  **Mode Collapse (The Cowardice Optima):** Early generations realized that exploring yielded negative hazard penalties. They optimized for a flat `0.00` by standing completely still. *Fix: Implemented an Existential Step Tax (-0.01 per step) to force foraging.*
2.  **The Tax Evasion Suicide Exploit:** To avoid paying the new Existential Tax, agents evolved to immediately walk out-of-bounds on Step 1, crashing their instance to secure a `0.00` rather than slowly starving to a `-1.50`. *Fix: Environment bounds enforcement.*
3.  **Organ Rejection:** Initially, the Critic and Actor shared a core neural layer. Passing down learned wisdom (Lamarckian) while mutating the physical body (Darwinian) resulted in "Neural Schizophrenia"—agents possessed the predictive wisdom of an elite runner but the mutated body of a toddler, destroying their fitness. *Fix: Decoupled the lobes into the Split-Brain architecture.*
4.  **The Genesis Cult:** Setting the initial elite publishing threshold too high (10.0) meant no agents wrote to the database. Confused agents queried an empty library 10,000 times a generation, receiving only the hardcoded dummy vector: "GO UP." *Fix: Implemented the Dynamic 90th Percentile threshold.*

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.9+
* CUDA-compatible GPU (Highly recommended but not required)

### Setup
```bash
# Clone the repository
git clone [https://github.com/harsha-bharadwaj100/tribal-rag-framework.git](https://github.com/harsha-bharadwaj100/tribal-rag-framework.git)
cd tribal-rag-framework

# Install dependencies
pip install torch numpy pandas matplotlib chromadb