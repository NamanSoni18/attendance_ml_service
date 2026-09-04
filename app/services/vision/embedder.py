import base64
import logging
from typing import List, Union

import cv2
import numpy as np
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

    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        if not isinstance(base64_str, str):
            raise ValueError("Expected a base64 string or image array.")

        cleaned = base64_str.split(",", 1)[-1] if "," in base64_str else base64_str
        image_bytes = base64.b64decode(cleaned, validate=True)
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode Base64 image.")
        return image

    def get_embedding(self, face_image: Union[np.ndarray, str]) -> List[float]:
        """
        Extracts the 512-D embedding from the cropped face.

        Args:
            face_image (np.ndarray | str): Cropped face image or base64-encoded image.

        Returns:
            List[float]: 512-dimensional vector.
        """
        try:
            if isinstance(face_image, str):
                face_image = self._decode_base64_image(face_image)

            embedding_objs = DeepFace.represent(
                img_path=face_image,
                model_name=self.model_name,
                enforce_detection=False,
            )

            if not embedding_objs or "embedding" not in embedding_objs[0]:
                raise EmbeddingGenerationError("Failed to extract embedding vector.")

            embedding = embedding_objs[0]["embedding"]
            return embedding

        except Exception as e:
            raise EmbeddingGenerationError(f"Error during ArcFace representation: {str(e)}") from e


FaceEmbedder = ArcFaceEmbedder
__all__ = ["ArcFaceEmbedder", "FaceEmbedder"]