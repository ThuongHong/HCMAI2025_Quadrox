from pydantic import BaseModel, Field
from typing import List, Optional


class AgentQueryVariant(BaseModel):
    query: str = Field(..., description="A reformulated English query preserving original intent")
    score: Optional[float] = Field(default=None, description="0-10 heuristic effectiveness score from LLM")
    rationale: Optional[str] = Field(default=None, description="Short reason why this variant may work")


class AgentResponse(BaseModel):
    refined_query: str = Field(..., description="Primary refined query in English")
    list_of_objects: List[str] = Field(default_factory=list, description="Relevant COCO objects if explicitly mentioned/implied")
    query_variants: List[AgentQueryVariant] = Field(default_factory=list, description="3-5 semantic variants")


class AgentQueryRequest(BaseModel):
    """Request model for agent queries"""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)


class AgentQueryResponse(BaseModel):
    """Response model for agent queries"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")


class QueryRefineResponse(BaseModel):
    translated_query: str = Field(..., description="Input translated to English (or original if already English)")
    enhanced_query: str = Field(..., description="Optimized English query for retrieval")

