from bert_score import score
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

print("="*50)
print("Fast BERTScore Hallucination Detector")
print("="*50)

def get_similarity_batch(contexts, answers):
    """Get BERTScore similarities for multiple samples at once (MUCH faster)"""
    P, R, F1 = score(answers, contexts, lang="en", verbose=False)
    return F1.tolist()

# Load dataset (resolves correctly from root or src/)
data_path = Path(__file__).resolve().parent.parent / "data" / "qa_data.json"
if not data_path.exists():
    data_path = Path("data/qa_data.json")
data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"✅ Loaded {len(data)} samples")

# Process in batches for speed
batch_size = 50
hallucinated_sims = []
correct_sims = []

print("Computing similarities on first 100 samples (this will take 2-3 minutes)...")

# Process hallucinated answers in batches
for i in tqdm(range(0, 100, batch_size), desc="Processing"):
    batch = data[i:i+batch_size]
    contexts = [s['knowledge'] for s in batch]
    answers = [s['hallucinated_answer'] for s in batch]
    sims = get_similarity_batch(contexts, answers)
    hallucinated_sims.extend(sims)
    
    # Process correct answers for same batch
    answers_correct = [s['right_answer'] for s in batch]
    sims_correct = get_similarity_batch(contexts, answers_correct)
    correct_sims.extend(sims_correct)

hallucinated_sims = np.array(hallucinated_sims[:100])
correct_sims = np.array(correct_sims[:100])

print(f"\n📊 Similarity Statistics:")
print(f"Hallucinated answers - Mean: {hallucinated_sims.mean():.3f}, Std: {hallucinated_sims.std():.3f}")
print(f"Correct answers - Mean: {correct_sims.mean():.3f}, Std: {correct_sims.std():.3f}")

# Find best threshold
best_threshold = 0.5
best_accuracy = 0
print("\nTesting thresholds:")
for threshold in np.arange(0.5, 0.9, 0.05):
    correct = sum(hallucinated_sims < threshold) + sum(correct_sims >= threshold)
    accuracy = correct / (len(hallucinated_sims) + len(correct_sims))
    print(f"  Threshold {threshold:.2f}: Accuracy {accuracy:.1%}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"\n✅ Best threshold: {best_threshold:.2f} (Accuracy: {best_accuracy:.1%})")

# Test on next 50 samples
print("\n" + "="*50)
print(f"Testing on next 50 samples...")
print("="*50)

test_samples = data[100:150]
contexts = [s['knowledge'] for s in test_samples]
hall_answers = [s['hallucinated_answer'] for s in test_samples]
correct_answers = [s['right_answer'] for s in test_samples]

print("Computing similarities for test set...")
hall_sims = get_similarity_batch(contexts, hall_answers)
correct_sims_test = get_similarity_batch(contexts, correct_answers)

correct = 0
total = 0

for sim_hall, sim_correct in zip(hall_sims, correct_sims_test):
    if sim_hall < best_threshold:
        correct += 1
    if sim_correct >= best_threshold:
        correct += 1
    total += 2

print(f"\n{'='*50}")
print(f"✅ FINAL ACCURACY: {correct/total*100:.1f}% ({correct}/{total} correct)")
print(f"{'='*50}")

accuracy_pct = (correct / total) * 100
if accuracy_pct > 70:
    print("🎉 Good! Your detector is working well!")
elif accuracy_pct > 60:
    print("📈 Moderate accuracy. Try the ensemble method with NLI next.")
else:
    print("⚠️  Low accuracy. The dataset has challenging hallucinations. Let's try the NLI ensemble method.")

