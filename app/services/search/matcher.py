# app/services/search/matcher.py
import numpy as np
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity

class VectorMatcher:
    def __init__(self, match_threshold: float = 0.68):
        """
        Initializes the similarity matcher.
        A threshold of ~0.68 is generally optimal for ArcFace Cosine Similarity,
        but you should fine-tune this during your evaluation phase.
        """
        self.match_threshold = match_threshold

    def verify_identity(self, source_embedding: list[float], target_embedding: list[float]) -> Tuple[bool, float]:
        """
        Compares two embeddings to determine if they belong to the same person.
        
        Args:
            source_embedding: The embedding captured live during attendance.
            target_embedding: The registered embedding fetched from the database.
            
        Returns:
            Tuple[bool, float]: (Is_Match, Similarity_Score)
        """
        # Convert lists to 2D numpy arrays required by scikit-learn
        emb1 = np.array(source_embedding).reshape(1, -1)
        emb2 = np.array(target_embedding).reshape(1, -1)
        
        # Calculate Cosine Similarity (-1 to 1)
        similarity_score = cosine_similarity(emb1, emb2)[0][0]
        
        # Determine match based on threshold
        is_match = bool(similarity_score >= self.match_threshold)
        
        return is_match, float(similarity_score)