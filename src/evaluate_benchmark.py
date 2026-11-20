import sys
import json
import time
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.detector.models import VerificationRequest
from src.detector.hybrid_detector import HybridHallucinationDetector

def run_benchmark(sample_count: int = 50):
    print("="*65)
    print(f"🏆 LLM Hallucination Detection Benchmark ({sample_count} Samples)")
    print("="*65)

    data_path = Path(__file__).resolve().parent.parent / "data" / "qa_data.json"
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        return

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Filter valid samples
    valid_samples = [
        s for s in data 
        if 'knowledge' in s and 'right_answer' in s and 'hallucinated_answer' in s
    ][:sample_count]

    print(f"Loaded {len(valid_samples)} paired samples ({len(valid_samples)*2} total predictions).")
    print("Initializing Hybrid Detector...")
    detector = HybridHallucinationDetector(use_bertscore=False)
    print("✅ Ready! Running evaluation...\n")

    tp = 0  # True Positive: Flagged hallucination correctly
    fp = 0  # False Positive: Flagged factual answer wrongly
    tn = 0  # True Negative: Accepted factual answer correctly
    fn = 0  # False Negative: Missed a hallucination

    latencies = []
    threshold = 0.80

    start_eval = time.perf_counter()

    for idx, s in enumerate(valid_samples):
        context = s['knowledge']
        hall_answer = s['hallucinated_answer']
        right_answer = s['right_answer']

        # 1. Test Hallucinated Answer (Should be flagged)
        t0 = time.perf_counter()
        res_hall = detector.verify(VerificationRequest(context=context, answer=hall_answer, threshold=threshold))
        latencies.append((time.perf_counter() - t0) * 1000)

        if res_hall.is_hallucinated:
            tp += 1
        else:
            fn += 1

        # 2. Test Factual Answer (Should NOT be flagged)
        t0 = time.perf_counter()
        res_right = detector.verify(VerificationRequest(context=context, answer=right_answer, threshold=threshold))
        latencies.append((time.perf_counter() - t0) * 1000)

        if not res_right.is_hallucinated:
            tn += 1
        else:
            fp += 1

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(valid_samples)} samples...")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "="*65)
    print("📊 BENCHMARK EVALUATION RESULTS")
    print("="*65)
    print(f"Total Evaluated Pairs : {total}")
    print(f"Overall Accuracy      : {accuracy:.1%}")
    print(f"Precision             : {precision:.1%}")
    print(f"Recall                : {recall:.1%}")
    print(f"F1-Score              : {f1:.3f}")
    print(f"Average Latency       : {avg_latency:.2f} ms / query")
    print("\nConfusion Matrix:")
    print(f"  True Positives  (Hallucinations caught): {tp}")
    print(f"  False Negatives (Hallucinations missed): {fn}")
    print(f"  True Negatives  (Factual approved)     : {tn}")
    print(f"  False Positives (False alarms)         : {fp}")
    print("="*65)

if __name__ == "__main__":
    run_benchmark(sample_count=50)
