import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List
from app.core.exceptions import FaceNotFoundError, MultipleFacesDetectedError

class YOLOFaceDetector:
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.65):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> Tuple[list[float], List[Tuple[float, float]]]:
        """
        Detects a face and returns its bounding box and landmarks.
        
        Returns:
            Tuple containing:
            - bbox: [x1, y1, x2, y2]
            - keypoints: List of (x, y) tuples for eyes, nose, mouth.
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            raise FaceNotFoundError("No face detected in the frame.")
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 1:
            raise MultipleFacesDetectedError("Multiple faces detected. Proxy risk.")
            
        # Extract bounding box
        x1, y1, x2, y2 = map(int, boxes[0])
        
        # Extract keypoints (landmarks) if the model supports it
        keypoints = []
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy[0].cpu().numpy() # Get keypoints for the first face
            keypoints = [(int(pt[0]), int(pt[1])) for pt in kpts]
            
        return [x1, y1, x2, y2], keypoints