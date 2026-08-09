from fastapi import APIRouter, Depends, HTTPException
from app.schemas.requests import EnrollmentRequest
from app.schemas.responses import EnrollmentResponse
from app.services.pipeline import FaceRecognitionPipeline
from app.api.dependencies import get_pipeline
from app.core.exceptions import FaceNotFoundError

router = APIRouter()

@router.post("/generate-embedding", response_model=EnrollmentResponse)
async def enroll_student(
    request: EnrollmentRequest, 
    pipeline: FaceRecognitionPipeline = Depends(get_pipeline)
):
    try:
        embedding = pipeline.generate_embedding(request.image_base64)
        return EnrollmentResponse(
            status="success",
            embedding=embedding,
            message=f"Embedding generated for student {request.student_id}."
        )
    except FaceNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))