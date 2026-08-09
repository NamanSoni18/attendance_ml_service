from fastapi import APIRouter, Depends, HTTPException
from app.schemas.requests import AttendanceVerificationRequest
from app.schemas.responses import VerificationResponse
from app.services.pipeline import FaceRecognitionPipeline
from app.api.dependencies import get_pipeline
from app.core.exceptions import FaceNotFoundError, MultipleFacesDetectedError, EmbeddingGenerationError

router = APIRouter()

@router.post("/verify", response_model=VerificationResponse)
async def verify_attendance(
    request: AttendanceVerificationRequest, 
    pipeline: FaceRecognitionPipeline = Depends(get_pipeline)
):
    try:
        is_match, confidence = pipeline.verify_attendance(
            request.image_base64, 
            request.stored_embedding
        )
        return VerificationResponse(
            status="success" if is_match else "rejected",
            is_match=is_match,
            confidence=confidence,
            message="Identity verified." if is_match else "Face does not match RFID owner."
        )
    except (FaceNotFoundError, MultipleFacesDetectedError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")