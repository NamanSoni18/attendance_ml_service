from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Attendance ML Service"
    API_V1_STR: str = "/api/v1"

    # Model Thresholds
    FACE_CONFIDENCE_THRESHOLD: float = 0.65
    COSINE_SIMILARITY_THRESHOLD: float = 0.68

    # Database
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "attendance"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()