from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Standard health check endpoint to ensure the ML microservice is running.
    """
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "message": "Computer Vision ML microservice is active and waiting for inference requests."
    }