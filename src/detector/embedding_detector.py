import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Union

class FastEmbeddingDetector:
    """
    Lightweight, high-speed semantic similarity detector using sentence-transformers/all-MiniLM-L6-v2.
    Runs in <5ms per query on CPU.
    """
    _instance = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(FastEmbeddingDetector, cls).__new__(cls)
            cls._instance._init_model(model_name)
        return cls._instance

    def _init_model(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1)

    def compute_similarity(self, context: str, claim: str) -> float:
        embeddings = self.encode([claim, context])
        sim = torch.dot(embeddings[0], embeddings[1]).item()
        # Scale from [-1, 1] to [0, 1]
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))
