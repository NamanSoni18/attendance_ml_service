class FaceNotFoundError(Exception):
    """Raised when YOLO cannot detect any face in the image."""
    pass

class MultipleFacesDetectedError(Exception):
    """Raised when YOLO detects more than one face, posing a proxy risk."""
    pass

class FaceAlignmentError(Exception):
    """Raised when facial landmarks cannot be found or alignment fails."""
    pass

class EmbeddingGenerationError(Exception):
    """Raised when DeepFace/ArcFace fails to extract a 512-D embedding."""
    pass