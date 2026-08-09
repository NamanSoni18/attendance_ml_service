from pydantic import BaseModel, Field

class AttendanceVerificationRequest(BaseModel):
    rfid_uid: str = Field(..., description="The RFID UID scanned by the student")
    image_base64: str = Field(..., description="Base64 encoded image from ESP32-CAM")
    stored_embedding: list[float] = Field(..., description="The 512-D registered embedding fetched by Node.js")

class EnrollmentRequest(BaseModel):
    student_id: str = Field(..., description="The database ID of the student")
    image_base64: str = Field(..., description="Base64 encoded image to generate an embedding for")