import cv2
import numpy as np
import base64
from typing import Tuple
from app.services.vision.detector import YOLOFaceDetector
from app.services.vision.aligner import FaceAligner
from app.services.vision.embedder import ArcFaceEmbedder
from app.services.search.matcher import VectorMatcher

class FaceRecognitionPipeline:
    def __init__(
        self, 
        detector: YOLOFaceDetector, 
        aligner: FaceAligner,
        embedder: ArcFaceEmbedder, 
        matcher: VectorMatcher
    ):
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.matcher = matcher

    def _decode_image(self, base64_str: str) -> np.ndarray:
        img_data = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode Base64 image.")
        return img

    def generate_embedding(self, base64_image: str) -> list[float]:
        img = self._decode_image(base64_image)
        
        # 1. Detect
        bbox, keypoints = self.detector.detect(img)
        
        # 2. Align (using keypoints if available, otherwise fallback to unaligned crop)
        if len(keypoints) >= 2:
            processed_img = self.aligner.align(img, keypoints)
            # Crop the aligned image using the original bounding box dimensions
            x1, y1, x2, y2 = bbox
            # Recalculate crop bounds safely
            h, w = processed_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            final_face = processed_img[y1:y2, x1:x2]
        else:
            # Fallback if no landmarks are detected
            x1, y1, x2, y2 = bbox
            final_face = img[y1:y2, x1:x2]
            
        # 3. Embed
        return self.embedder.get_embedding(final_face)

    def verify_attendance(self, live_base64: str, stored_embedding: list[float]) -> Tuple[bool, float]:
        live_embedding = self.generate_embedding(live_base64)
        return self.matcher.verify_identity(live_embedding, stored_embedding)