from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ClaimVerification(BaseModel):
    claim: str
    is_hallucinated: bool
    score: float
    confidence: float
    verdict: str
    reasoning: str
    ungrounded_entities: List[str] = Field(default_factory=list)

class VerificationRequest(BaseModel):
    context: str = Field(..., description="Source knowledge document / context")
    answer: str = Field(..., description="LLM generated answer to verify")
    question: Optional[str] = Field(default="", description="Original user prompt or question")
    threshold: float = Field(default=0.82, ge=0.0, le=1.0, description="Hallucination threshold (below = hallucinated)")

class BatchVerificationRequest(BaseModel):
    items: List[VerificationRequest]

class VerificationResponse(BaseModel):
    is_hallucinated: bool
    verdict: str
    overall_score: float
    confidence: float
    summary: str
    claims: List[ClaimVerification]
    latency_ms: float
    metrics: Dict[str, float] = Field(default_factory=dict)
