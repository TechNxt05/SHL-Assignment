"""
SHL Assessment Recommender - Catalog Schemas
Defines the Pydantic models for assessment catalog entries.
"""
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, field_validator


class Assessment(BaseModel):
    """Represents a single SHL Individual Test Solution from the catalog."""

    name: str
    url: str  # Raw string URL from catalog — validated as reachable during scrape
    description: str
    test_type: str  # A=Ability, B=Biodata, C=Competency, K=Knowledge, P=Personality, S=Simulation
    category: str
    duration_minutes: Optional[int] = None
    remote_testing: bool = False
    adaptive: bool = False
    job_levels: List[str] = []
    languages: List[str] = []
    skills_measured: List[str] = []
    competencies: List[str] = []
    keywords: List[str] = []  # Derived from all text fields for BM25

    @field_validator("name", "description", "category", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("test_type", mode="before")
    @classmethod
    def normalize_test_type(cls, v: str) -> str:
        """Normalize test type to uppercase single letter or abbreviation."""
        mapping = {
            "ability": "A",
            "biodata": "B",
            "competency": "C",
            "knowledge": "K",
            "personality": "P",
            "simulation": "S",
            "behavioral": "B",
            "cognitive": "A",
            "skills": "K",
        }
        v_lower = v.strip().lower()
        return mapping.get(v_lower, v.strip().upper())
