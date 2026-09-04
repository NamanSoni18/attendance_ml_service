import os

# STRICT REQUIREMENT: Force TensorFlow to use Keras 2.x API for DeepFace compatibility.
# This MUST be declared at the very top before ANY other imports trigger TensorFlow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.state.mongodb = None

from app.api.v1.endpoints.attendance import verify_attendance as verify_attendance_compat


@app.post("/api/verify")
async def compatibility_verify_attendance(request):
    return await verify_attendance_compat(request)

# Allow requests from your Node.js backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to your Node.js server IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}

@app.on_event("startup")
async def load_models():
    from motor.motor_asyncio import AsyncIOMotorClient

    from app.core.config import settings
    from app.services.vision.detector import YOLOFaceDetector
    from app.services.vision.embedder import ArcFaceEmbedder
    from app.services.search.matcher import VectorMatcher
    from app.services.vision.aligner import FaceAligner
    from app.services.pipeline import FaceRecognitionPipeline

    client = AsyncIOMotorClient(settings.MONGODB_URI)
    app.state.mongodb = client[settings.MONGODB_DB_NAME]

    app.state.detector = YOLOFaceDetector(conf_threshold=settings.FACE_CONFIDENCE_THRESHOLD)
    app.state.embedder = ArcFaceEmbedder()
    app.state.matcher = VectorMatcher(match_threshold=settings.COSINE_SIMILARITY_THRESHOLD)
    app.state.aligner = FaceAligner()
    app.state.pipeline = FaceRecognitionPipeline(app.state.detector, app.state.aligner, app.state.embedder, app.state.matcher)