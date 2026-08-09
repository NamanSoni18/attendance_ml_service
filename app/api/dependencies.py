from app.core.config import settings
from app.services.vision.aligner import FaceAligner
from app.services.vision.detector import YOLOFaceDetector
from app.services.vision.embedder import ArcFaceEmbedder
from app.services.search.matcher import VectorMatcher
from app.services.pipeline import FaceRecognitionPipeline
from fastapi import Request

# Singleton instances loaded at startup
detector = YOLOFaceDetector(conf_threshold=settings.FACE_CONFIDENCE_THRESHOLD)
embedder = ArcFaceEmbedder()
matcher = VectorMatcher(match_threshold=settings.COSINE_SIMILARITY_THRESHOLD)
aligner = FaceAligner()
pipeline = FaceRecognitionPipeline(detector, aligner, embedder, matcher)


def get_pipeline(request: Request) -> FaceRecognitionPipeline:
    return request.app.state.pipeline