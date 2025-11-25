from typing import List
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """
    Schema for the user search request.
    """
    query: str = Field(
        ..., 
        min_length=3, 
        description="The natural language job search query (e.g., 'Senior Python Engineer in New York')."
    )

class JobCitation(BaseModel):
    """
    Schema for a specific job listing source cited in the response.
    """
    id: str = Field(..., description="Unique Job ID (e.g., LF0001).")
    title: str = Field(..., description="Job title.")
    company: str = Field(..., description="Hiring company name.")
    location: str = Field(..., description="Job location.")
    score: float = Field(..., description="Relevance score from the Reranker (0.0 to 1.0).")

class QueryResponse(BaseModel):
    """
    Schema for the API response containing the generated answer and sources.
    """
    answer: str = Field(..., description="The AI-generated answer to the user's query.")
    citations: List[JobCitation] = Field(..., description="List of job sources used to generate the answer.")
    processing_time: float = Field(..., description="Time taken to process the request in seconds.")