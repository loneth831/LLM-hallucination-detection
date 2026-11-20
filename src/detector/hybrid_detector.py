import time
from typing import List, Dict, Any, Optional
from src.detector.models import VerificationRequest, VerificationResponse, ClaimVerification
from src.detector.sentence_splitter import split_into_claims
from src.detector.entity_checker import check_entity_grounding
from src.detector.embedding_detector import FastEmbeddingDetector

class HybridHallucinationDetector:
    """
    Production-grade hybrid hallucination detector combining:
    1. Granular sentence/claim decomposition
    2. Deep semantic representation (MiniLM dense embeddings)
    3. Factual entity & numerical verification (NER consistency)
    4. Token-level precision grounding
    """
    def __init__(self, use_bertscore: bool = False):
        self.fast_embedder = FastEmbeddingDetector()
        self.use_bertscore = use_bertscore
        self._bertscore = None

    def _get_bertscore(self):
        if self._bertscore is None:
            from src.detector.bertscore_detector import BERTScoreDetector
            self._bertscore = BERTScoreDetector()
        return self._bertscore

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        start_time = time.perf_counter()
        context = request.context.strip()
        answer = request.answer.strip()
        threshold = request.threshold

        claims = split_into_claims(answer)
        if not claims:
            return VerificationResponse(
                is_hallucinated=True,
                verdict="EMPTY_RESPONSE",
                overall_score=0.0,
                confidence=1.0,
                summary="Provided answer was empty.",
                claims=[],
                latency_ms=0.0
            )

        claim_results: List[ClaimVerification] = []
        claim_scores: List[float] = []

        for claim in claims:
            # 1. Semantic Embedding Similarity (0.0 to 1.0)
            semantic_score = self.fast_embedder.compute_similarity(context, claim)
            
            # 2. Entity & Numerical Verification (0.0 to 1.0)
            entity_score, ungrounded_entities = check_entity_grounding(context, claim)
            
            # 3. Optional BERTScore
            if self.use_bertscore:
                bert_score_val = self._get_bertscore().compute_similarity(context, claim)
                # Weighted blend
                composite_score = (0.45 * semantic_score) + (0.25 * bert_score_val) + (0.30 * entity_score)
            else:
                # High-speed blend
                composite_score = (0.60 * semantic_score) + (0.40 * entity_score)

            # Heavy penalty if key entities or numbers are invented
            if ungrounded_entities:
                composite_score = min(composite_score, 0.65)
                
            is_claim_hallucinated = composite_score < threshold
            confidence = abs(composite_score - threshold) + 0.5
            confidence = min(1.0, max(0.5, confidence))

            if ungrounded_entities:
                reasoning = f"Mismatched entities or numbers detected: {', '.join(ungrounded_entities)}"
                verdict = "FABRICATED_ENTITY"
            elif composite_score >= threshold:
                reasoning = f"Strong factual alignment with source context (score: {composite_score:.2f})"
                verdict = "GROUNDED"
            else:
                reasoning = f"Low semantic support in source text (score: {composite_score:.2f})"
                verdict = "UNSUPPORTED"

            claim_results.append(ClaimVerification(
                claim=claim,
                is_hallucinated=is_claim_hallucinated,
                score=round(composite_score, 3),
                confidence=round(confidence, 3),
                verdict=verdict,
                reasoning=reasoning,
                ungrounded_entities=ungrounded_entities
            ))
            claim_scores.append(composite_score)

        # Overall summary
        overall_score = sum(claim_scores) / len(claim_scores)
        hallucinated_claims = [c for c in claim_results if c.is_hallucinated]
        is_overall_hallucinated = len(hallucinated_claims) > 0 or (overall_score < threshold)

        if not is_overall_hallucinated:
            overall_verdict = "FACTUALLY_GROUNDED"
            summary = "The response is fully grounded in the provided context."
        elif len(hallucinated_claims) == len(claim_results):
            overall_verdict = "TOTAL_HALLUCINATION"
            summary = "The response is completely unsupported or contradicted by the context."
        else:
            overall_verdict = "PARTIAL_HALLUCINATION"
            summary = f"Detected factual discrepancies in {len(hallucinated_claims)} out of {len(claim_results)} claims."

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return VerificationResponse(
            is_hallucinated=is_overall_hallucinated,
            verdict=overall_verdict,
            overall_score=round(overall_score, 3),
            confidence=round(min(1.0, max(0.5, abs(overall_score - threshold) + 0.5)), 3),
            summary=summary,
            claims=claim_results,
            latency_ms=latency_ms,
            metrics={
                "total_claims": len(claims),
                "hallucinated_claims_count": len(hallucinated_claims),
                "threshold_applied": threshold
            }
        )
