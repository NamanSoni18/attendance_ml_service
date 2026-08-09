from pydantic import BaseModel

class VerificationResponse(BaseModel):
    status: str
    is_match: bool
    confidence: float
    message: str

class EnrollmentResponse(BaseModel):
    status: str
    embedding: list[float] | None = None
    message: str