from fastapi import APIRouter, HTTPException, Request

from app.services.search.matcher import face_matcher
from app.services.vision.embedder import FaceEmbedder

router = APIRouter()

face_embedder = FaceEmbedder()


@router.post("/verify")
async def verify_attendance(request: Request):
    data = await request.json()
    uid = data.get("rfid_uid")
    image_base64 = data.get("image_base64")

    if not uid or not image_base64:
        raise HTTPException(status_code=400, detail="Missing rfid_uid or image_base64")

    try:
        target_embedding = face_embedder.get_embedding(image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}") from e

    db_client = getattr(request.app.state, "mongodb", None)
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database is not configured for attendance verification.")

    if isinstance(db_client, dict):
        db_collection = db_client.get("students")
    elif hasattr(db_client, "students"):
        db_collection = db_client.students
    else:
        db_collection = db_client

    if db_collection is None or not hasattr(db_collection, "find_one"):
        raise HTTPException(status_code=503, detail="Student collection is not available for attendance verification.")

    result = await face_matcher.verify_face(uid, target_embedding, db_collection)

    return result