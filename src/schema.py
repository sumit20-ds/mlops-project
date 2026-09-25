"""Pydantic schemas shared by the FastAPI app and the SageMaker handler."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src import config


class StudentFeatures(BaseModel):
    """One student's lifestyle features, as required by the trained pipeline."""

    Study_Hours: float = Field(..., ge=0, le=15, description="Daily study hours")
    Age: int = Field(..., ge=10, le=100)
    Avg_Daily_Usage_Hours: float = Field(..., ge=0, le=24, description="Daily social media usage")
    Daily_Unlocks: int = Field(..., ge=0, le=1000, description="Phone unlocks per day")
    Physical_Activity_Hours: float = Field(..., ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(..., ge=0, le=24)
    Stress_Level: Literal["Low", "Medium", "High", "Very High"]
    Gender: Literal["Female", "Male"]
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform: Literal[
        "Facebook", "Instagram", "KakaoTalk", "LINE", "LinkedIn", "Snapchat",
        "TikTok", "Twitter", "VKontakte", "WeChat", "WhatsApp", "YouTube",
    ]
    Purpose_Of_Use: Literal["Education", "Entertainment", "Networking", "News"]
    Country: str = Field(..., description="Raw country name; auto-bucketed server-side")

    @field_validator("Country")
    @classmethod
    def _non_empty_country(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Country must not be empty")
        return v

    def to_model_row(self) -> dict:
        """Build the exact feature dict the sklearn pipeline expects,
        including the Grouped_country derived feature."""
        row = self.model_dump(exclude={"Country"})
        row["Grouped_country"] = config.group_country(self.Country)
        return row

    model_config = {
        "json_schema_extra": {
            "example": {
                "Study_Hours": 4.5,
                "Age": 21,
                "Avg_Daily_Usage_Hours": 4.0,
                "Daily_Unlocks": 134,
                "Physical_Activity_Hours": 2.2,
                "Sleep_Hours_Per_Night": 6.7,
                "Stress_Level": "Medium",
                "Gender": "Male",
                "Academic_Level": "Undergraduate",
                "Most_Used_Platform": "Facebook",
                "Purpose_Of_Use": "Networking",
                "Country": "India",
            }
        }
    }


class PredictionResponse(BaseModel):
    mental_health_score: float
    model_version: str
    request_id: str


class BatchPredictionRequest(BaseModel):
    instances: list[StudentFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
