import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import time

from src.detector.models import (
    VerificationRequest,
    VerificationResponse,
    BatchVerificationRequest
)
from src.detector.hybrid_detector import HybridHallucinationDetector

app = FastAPI(
    title="LLM Hallucination Detection API",
    description="Enterprise-grade verification service for detecting factual hallucinations in LLM outputs.",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector singleton
detector = HybridHallucinationDetector(use_bertscore=False)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "llm-hallucination-detector",
        "version": "2.0.0",
        "models_loaded": ["sentence-transformers/all-MiniLM-L6-v2", "NER-Regex-Rule-Engine"]
    }

@app.post("/v1/verify", response_model=VerificationResponse)
def verify_claim(request: VerificationRequest):
    """
    Verify whether an LLM answer is factually grounded in the given context document.
    Returns granular claim-by-claim breakdown, ungrounded entities, and overall verdict.
    """
    if not request.context.strip():
        raise HTTPException(status_code=400, detail="Context document cannot be empty.")
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")
        
    return detector.verify(request)

@app.post("/v1/verify/batch", response_model=List[VerificationResponse])
def verify_batch(batch_request: BatchVerificationRequest):
    """
    High-throughput batch verification for evaluating datasets or pipeline logs.
    """
    results = []
    for item in batch_request.items:
        results.append(detector.verify(item))
    return results

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
