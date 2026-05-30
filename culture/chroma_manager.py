import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

class ChromaManager:
    def __init__(self, persist_directory: str = "./.chroma_db"):
        """
        Initialize the ChromaDB controller with local SQLite persistence.
        """
        self.persist_directory = persist_directory
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collections = {}

    def _get_collection_name(self, tribe_id: str) -> str:
        """Format the collection name for a specific tribe."""
        return f"collection_tribe_{tribe_id}"

    def get_or_create_tribe_collection(self, tribe_id: str) -> chromadb.Collection:
        """
        Get or create an isolated collection for a specific Tribe.
        We explicitly set embedding_function to None since we are passing raw latent vectors.
        """
        collection_name = self._get_collection_name(tribe_id)
        if collection_name not in self._collections:
            # We use get_or_create_collection so it handles both cases.
            collection = self.client.get_or_create_collection(
                name=collection_name,
                # Bypass default embedding function by setting it to None or passing empty
                # Some chromadb versions require a dummy function if None is not accepted, 
                # but setting it to None or not passing it is the standard way to bypass.
                # In standard ChromaDB, if you pass `embeddings` directly, it doesn't need an embedding_function.
            )
            self._collections[collection_name] = collection
            
        return self._collections[collection_name]

    def write_experience(
        self, 
        tribe_id: str, 
        experience_id: str,
        latent_state_vector: List[float], 
        action: int, 
        reward: float
    ):
        """
        Write Phase: Embeds the state (as a raw vector) and stores Action and Reward in metadata.
        
        Args:
            tribe_id: The ID of the agent's tribe.
            experience_id: A unique identifier for this experience (e.g., episode_step).
            latent_state_vector: The raw numerical state tensor mapped to a list of floats.
            action: The discrete action taken.
            reward: The reward received.
        """
        collection = self.get_or_create_tribe_collection(tribe_id)
        
        metadata = {
            "action": action,
            "reward": reward,
            "usage_count": 0
        }
        
        collection.add(
            ids=[experience_id],
            embeddings=[latent_state_vector],
            metadatas=[metadata]
        )

    def query_culture(
        self, 
        tribe_id: str, 
        latent_state_vector: List[float], 
        n_results: int = 1,
        min_reward_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Read Phase: Queries the Tribe's database for the most similar past states using raw float vectors.
        
        Args:
            tribe_id: The ID of the agent's tribe.
            latent_state_vector: The raw numerical state tensor for the current state.
            n_results: Number of top similar states to return.
            min_reward_threshold: Optional threshold to filter for only high-reward past experiences.
            
        Returns:
            A dictionary containing the query results (ids, distances, metadatas).
        """
        collection = self.get_or_create_tribe_collection(tribe_id)
        
        where_clause = None
        if min_reward_threshold is not None:
            where_clause = {"reward": {"$gte": min_reward_threshold}}
            
        try:
            results = collection.query(
                query_embeddings=[latent_state_vector],
                n_results=n_results,
                where=where_clause
            )
        except Exception as e:
            # Catch internal ChromaDB compaction errors silently
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        
        # Increment usage_count for retrieved memories
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i, result_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                metadata["usage_count"] = metadata.get("usage_count", 0) + 1
                try:
                    collection.update(
                        ids=[result_id],
                        metadatas=[metadata]
                    )
                except Exception as e:
                    # Catch internal ChromaDB compaction errors silently
                    pass
        
        return results

    def clear_tribe_collection(self, tribe_id: str):
        """Utility method to delete a tribe's collection (e.g., for testing or resets)."""
        collection_name = self._get_collection_name(tribe_id)
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
        except Exception as e:
            print(f"Failed to delete collection {collection_name}: {e}")

    def prune_database(self, tribe_id: str, max_capacity: int):
        """
        Prunes the specific tribe's database if it exceeds max_capacity,
        deleting the vectors with the lowest usage_count.
        """
        collection = self.get_or_create_tribe_collection(tribe_id)
        try:
            if collection.count() <= max_capacity:
                return
            all_data = collection.get(include=["metadatas"])
        except Exception:
            return
        
        if not all_data["ids"]:
            return
            
        # Pair IDs with their usage count
        items = []
        for i, item_id in enumerate(all_data["ids"]):
            usage = all_data["metadatas"][i].get("usage_count", 0)
            items.append((usage, item_id))
            
        # Sort by usage_count ascending
        items.sort(key=lambda x: x[0])
        
        # Delete the ones with lowest usage count
        num_to_delete = len(items) - max_capacity
        if num_to_delete > 0:
            ids_to_delete = [item[1] for item in items[:num_to_delete]]
            collection.delete(ids=ids_to_delete)
