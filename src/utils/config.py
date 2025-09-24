from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "SPORTSDATA_BASE_URL",
            "https://api.sportsdata.io/v3/tennis/scores/json"  # v3 default
        )
    )
    api_key: str = Field(default_factory=lambda: os.getenv("SPORTSDATA_API_KEY", ""))

settings = Settings()
