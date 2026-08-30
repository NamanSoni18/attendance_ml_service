import numpy as np
from numpy.linalg import norm

class FaceMatcher:
    def __init__(self, threshold=0.60):
        # 60% similarity threshold; adjust if needed
        self.threshold = threshold

    def _cosine_similarity(self, vec1, vec2):
        """Calculates the cosine similarity between two 512D vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    async def verify_face(self, uid, target_embedding, db_collection):
        """
        Fetches the student by UID and checks the live camera embedding 
        against ALL stored embeddings for that student.
        """
        # 1. Fetch the specific student using the RFID UID
        student = await db_collection.find_one({"uid": uid})
        
        if not student:
            return {"status": "Failed", "match": False, "message": "RFID UID not found."}
            
        if "face_embeddings" not in student or not student["face_embeddings"]:
            return {"status": "Failed", "match": False, "message": "No face data enrolled."}

        highest_similarity = -1.0
        
        # 2. Iterate through all saved embeddings for this specific user
        for db_embedding in student["face_embeddings"]:
            sim_score = self._cosine_similarity(target_embedding, db_embedding)
            
            # Track the best score
            if sim_score > highest_similarity:
                highest_similarity = sim_score

        # 3. Check if the best match exceeds your security threshold
        if highest_similarity >= self.threshold:
            return {
                "status": "Success",
                "match": True,
                "confidence": float(highest_similarity),
                "student": {
                    "uid": student.get('uid'),
                    "name": student.get('name'),
                    "rollNo": student.get('rollNo'),
                    "department": student.get('department')
                }
            }
        
        # Deny access if no saved image was a close enough match
        return {
            "status": "Failed",
            "match": False, 
            "message": f"Face mismatch. Max confidence: {highest_similarity:.2f}"
        }

# Instantiate the matcher for your endpoints
face_matcher = FaceMatcher(threshold=0.60)