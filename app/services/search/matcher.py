import numpy as np
from numpy.linalg import norm


class VectorMatcher:
    def __init__(self, match_threshold: float = 0.68):
        self.threshold = match_threshold

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Calculates the cosine similarity between two embedding vectors."""
        vec1 = np.asarray(vec1, dtype=np.float32)
        vec2 = np.asarray(vec2, dtype=np.float32)
        denom = norm(vec1) * norm(vec2)
        if np.isclose(denom, 0.0):
            return 0.0
        return float(np.dot(vec1, vec2) / denom)

    def verify_identity(self, live_embedding, stored_embedding):
        score = self._cosine_similarity(live_embedding, stored_embedding)
        return score >= self.threshold, float(score)

    async def verify_face(self, uid, target_embedding, db_collection):
        """
        Fetches the student by UID and checks the live camera embedding
        against ALL stored embeddings for that student.
        """
        student = await db_collection.find_one({"uid": uid})

        if not student:
            return {"status": "Failed", "match": False, "is_match": False, "confidence": 0.0, "message": "RFID UID not found."}

        if "face_embeddings" not in student or not student["face_embeddings"]:
            return {"status": "Failed", "match": False, "is_match": False, "confidence": 0.0, "message": "No face data enrolled."}

        highest_similarity = -1.0

        for db_embedding in student["face_embeddings"]:
            sim_score = self._cosine_similarity(target_embedding, db_embedding)
            if sim_score > highest_similarity:
                highest_similarity = sim_score

        if highest_similarity >= self.threshold:
            return {
                "status": "Success",
                "match": True,
                "is_match": True,
                "confidence": float(highest_similarity),
                "student": {
                    "uid": student.get("uid"),
                    "name": student.get("name"),
                    "rollNo": student.get("rollNo"),
                    "department": student.get("department"),
                },
            }

        return {
            "status": "Failed",
            "match": False,
            "is_match": False,
            "confidence": float(highest_similarity),
            "message": f"Face mismatch. Max confidence: {highest_similarity:.2f}",
        }


FaceMatcher = VectorMatcher

# Instantiate the matcher for your endpoints
face_matcher = VectorMatcher(match_threshold=0.68)