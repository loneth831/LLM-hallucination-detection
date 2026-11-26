# 🔍 Enterprise LLM Hallucination Detection & Grounding Audit

An enterprise-grade system for detecting, verifying, and explaining factual hallucinations in Large Language Model (LLM) responses against source reference documents.

```
                           +----------------------+
                           |   Context Document   |
                           +----------+-----------+
                                      |
+---------------------+               v               +------------------------+
|    User Question    | ---> [ Claim Decomposition ] <--- |   LLM Generated Text   |
+---------------------+               |               +------------------------+
                                      v
                    +------------------------------------+
                    |  Hybrid Verification Pipeline      |
                    |  * Dense Embeddings (MiniLM)       |
                    |  * Contextual Alignment (RoBERTa)  |
                    |  * Named Entity & Number Auditing  |
                    +-----------------+------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
        [ REST API (FastAPI) ]            [ Interactive UI (Gradio) ]
        * /v1/verify (Single)             * Sentence Color Highlighting
        * /v1/verify/batch (Batch)        * Entity Breakdown Cards
        * /health (Diagnostics)           * Confidence & Latency Meters
```

---

## 🚀 Key Features

* **🔬 Claim-by-Claim Decomposition:** Automatically splits multi-sentence LLM responses into atomic assertions for individual verification.
* **⚡ High-Speed Inference (<200ms):** Uses dense transformer embeddings (`all-MiniLM-L6-v2`) for real-time validation.
* **🏷️ Factual Entity & Number Auditing:** Flags fabricated names, numbers, years, percentages, and entities not present in the reference source.
* **🌐 Production-Ready REST API:** Fully asynchronous FastAPI service with automatic OpenAPI / Swagger documentation (`/docs`).
* **🎨 Interactive Visual Dashboard:** Explainable Gradio interface with color-coded sentence highlighting (Green = Supported, Red = Hallucinated).
* **📊 Quantitative Benchmarking:** Built-in evaluation harness computing Precision, Recall, F1-Score, and Latency metrics.

---

## 🛠️ Quickstart

### Option 1: Double-Click Batch Files (Windows)
* **Launch UI Dashboard:** Double-click `run_dashboard.bat` (opens `http://127.0.0.1:7860`)
* **Launch REST API Server:** Double-click `run_api.bat` (Swagger docs at `http://127.0.0.1:8000/docs`)
* **Run Benchmark:** Double-click `run_benchmark.bat`

### Option 2: Command Line (PowerShell)
```powershell
# 1. Run the interactive Gradio dashboard
.\venv\Scripts\python.exe src\demo_advanced.py

# 2. Start the FastAPI server
.\venv\Scripts\python.exe src\api.py

# 3. Run the evaluation benchmark
.\venv\Scripts\python.exe src\evaluate_benchmark.py
```

---

## 📡 REST API Documentation

### Endpoint: `POST /v1/verify`
Audits an LLM response against a source reference document.

**Request Body:**
```json
{
  "context": "The Eiffel Tower is located in Paris, France. It was built in 1889 and is 330 meters tall.",
  "question": "Where is the Eiffel Tower?",
  "answer": "The Eiffel Tower is located in London. It was built in 1945.",
  "threshold": 0.82
}
```

**Response Body:**
```json
{
  "is_hallucinated": true,
  "verdict": "TOTAL_HALLUCINATION",
  "overall_score": 0.582,
  "confidence": 0.738,
  "summary": "The response is completely unsupported or contradicted by the context.",
  "claims": [
    {
      "claim": "The Eiffel Tower is located in London.",
      "is_hallucinated": true,
      "score": 0.62,
      "confidence": 0.70,
      "verdict": "FABRICATED_ENTITY",
      "reasoning": "Mismatched entities or numbers detected: london",
      "ungrounded_entities": ["london"]
    },
    {
      "claim": "It was built in 1945.",
      "is_hallucinated": true,
      "score": 0.54,
      "confidence": 0.78,
      "verdict": "FABRICATED_ENTITY",
      "reasoning": "Mismatched entities or numbers detected: 1945",
      "ungrounded_entities": ["1945"]
    }
  ],
  "latency_ms": 14.8
}
```

---

## 📁 Project Structure

```text
llm-hallucination-detector/
├── README.md                      # Comprehensive project documentation
├── run_dashboard.bat              # 1-click launcher for Gradio UI
├── run_api.bat                    # 1-click launcher for FastAPI
├── run_benchmark.bat              # 1-click benchmark evaluation
├── data/
│   └── qa_data.json               # 10,000 paired question-answer evaluation dataset
└── src/
    ├── api.py                     # Production FastAPI server (/v1/verify)
    ├── demo_advanced.py           # Explainable claim-level visual dashboard
    ├── evaluate_benchmark.py      # Precision / Recall / F1 benchmark suite
    ├── bertscore_simple.py        # Baseline BERTScore detector
    ├── fast_bertscore.py          # Batch threshold search baseline
    └── detector/                  # Core modular package
        ├── __init__.py
        ├── models.py              # Pydantic schemas (Request/Response)
        ├── sentence_splitter.py   # Atomic claim segmentation
        ├── entity_checker.py      # NER & numerical consistency verification
        ├── embedding_detector.py  # MiniLM semantic similarity (<5ms)
        ├── bertscore_detector.py  # RoBERTa-large alignment
        └── hybrid_detector.py     # Ensemble verification engine
```
