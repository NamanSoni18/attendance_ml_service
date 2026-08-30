from fastapi import APIRouter, Request, HTTPException
from app.services.vision.embedder import face_embedder # Adjust import based on your exact file name
from app.services.search.matcher import face_matcher

router = APIRouter()

@router.post("/verify")
async def verify_attendance(request: Request):
    data = await request.json()
    uid = data.get("rfid_uid")
    image_base64 = data.get("image_base64")

    if not uid or not image_base64:
        raise HTTPException(status_code=400, detail="Missing rfid_uid or image_base64")

    # 1. Convert the incoming base64 image to a 512D embedding
    try:
        target_embedding = await face_embedder.get_embedding(image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face extraction failed: {str(e)}")

    # 2. Reference your MongoDB collection (Assuming 'students' is your collection name)
    db_collection = request.app.mongodb["students"]

    # 3. Pass both the uid and the live embedding to the matcher
    result = await face_matcher.verify_face(uid, target_embedding, db_collection)

    return result