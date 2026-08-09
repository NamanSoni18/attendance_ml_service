import cv2
import numpy as np
import math
import logging
from app.core.exceptions import FaceAlignmentError

logger = logging.getLogger(__name__)

class FaceAligner:
    def __init__(self, desired_face_width: int = 112, desired_face_height: int = 112):
        """
        Initializes the aligner. ArcFace expects 112x112 input images.
        """
        self.desired_width = desired_face_width
        self.desired_height = desired_face_height

    def align(self, image: np.ndarray, keypoints: list) -> np.ndarray:
        """
        Aligns the face based on eye coordinates.
        YOLO face keypoints generally follow: [left_eye, right_eye, nose, left_mouth, right_mouth]
        
        Args:
            image (np.ndarray): The original uncropped BGR image.
            keypoints (list): A list of (x, y) coordinates for facial landmarks.
            
        Returns:
            np.ndarray: The aligned and cropped face image.
        """
        try:
            if len(keypoints) < 2:
                raise FaceAlignmentError("Insufficient keypoints for alignment.")

            # Extract eye coordinates (assuming first two keypoints are left and right eyes)
            left_eye = keypoints[0]
            right_eye = keypoints[1]

            # Calculate the angle between the eyes
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = math.degrees(math.atan2(dy, dx))

            # Calculate the center point between the eyes
            eyes_center = (
                (left_eye[0] + right_eye[0]) // 2,
                (left_eye[1] + right_eye[1]) // 2
            )

            # Get the rotation matrix for a 2D affine transform
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

            # Apply the affine transformation to the entire image
            aligned_img = cv2.warpAffine(
                image, 
                M, 
                (image.shape[1], image.shape[0]), 
                flags=cv2.INTER_CUBIC
            )

            return aligned_img

        except Exception as e:
            logger.error(f"Alignment failed: {str(e)}")
            raise FaceAlignmentError("Failed to align face geometry.")