from fastapi import APIRouter
from app.api.v1.endpoints import attendance, enrollment, health

api_router = APIRouter()
api_router.include_router(attendance.router, prefix="/attendance", tags=["Attendance"])
api_router.include_router(enrollment.router, prefix="/enrollment", tags=["Enrollment"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])