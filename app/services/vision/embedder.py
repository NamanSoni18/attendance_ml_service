import numpy as np
import logging
from typing import List
from deepface import DeepFace
from app.core.exceptions import EmbeddingGenerationError

logger = logging.getLogger(__name__)

class ArcFaceEmbedder:
    def __init__(self, model_name: str = "ArcFace"):
        """
        Initializes the DeepFace embedding extractor.
        """
        self.model_name = model_name
        self._warmup_model()

    def _warmup_model(self):
        """
        Machine learning models often experience high latency on the first inference.
        We run a dummy matrix through the model on startup to warm it up.
        """
        logger.info(f"Warming up {self.model_name} model...")
        dummy_image = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            self.get_embedding(dummy_image)
            logger.info("Model warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup failed, but will proceed: {e}")

    def get_embedding(self, face_image: np.ndarray) -> List[float]:
        """
        Extracts the 512-D embedding from the cropped face.
        
        Args:
            face_image (np.ndarray): Cropped face image.
            
        Returns:
            List[float]: 512-dimensional vector.
        """
        try:
            # We enforce_detection=False because YOLO already cropped the face
            embedding_objs = DeepFace.represent(
                img_path=face_image, 
                model_name=self.model_name, 
                enforce_detection=False
            )
            
            if not embedding_objs or "embedding" not in embedding_objs[0]:
                raise EmbeddingGenerationError("Failed to extract embedding vector.")
                
            embedding = embedding_objs[0]["embedding"]
            return embedding
            
        except Exception as e:
            raise EmbeddingGenerationError(f"Error during ArcFace representation: {str(e)}")