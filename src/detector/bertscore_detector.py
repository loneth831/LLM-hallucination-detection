from bert_score import BERTScorer
from typing import Optional

class BERTScoreDetector:
    """
    Precision contextual embedding detector using RoBERTa-large via BERTScore.
    Utilizes singleton pattern to prevent reloading 1.4GB model weights.
    """
    _instance: Optional['BERTScoreDetector'] = None
    _scorer: Optional[BERTScorer] = None

    def __new__(cls, lang: str = "en"):
        if cls._instance is None:
            cls._instance = super(BERTScoreDetector, cls).__new__(cls)
            cls._scorer = BERTScorer(lang=lang)
        return cls._instance

    def compute_similarity(self, context: str, answer: str) -> float:
        if self._scorer is None:
            self._scorer = BERTScorer(lang="en")
        P, R, F1 = self._scorer.score([answer], [context], verbose=False)
        return float(F1.item())
