from bert_score import BERTScorer
import json
from pathlib import Path

print("="*50)
print("BERTScore Hallucination Detector")
print("="*50)

print("Loading BERTScore model (RoBERTa-large)...")
scorer = BERTScorer(lang="en")
print("✅ Model loaded successfully!\n")

def detect_hallucination_bertscore(context, answer, threshold=0.5):
    P, R, F1 = scorer.score([answer], [context], verbose=False)
    similarity = F1.item()
    is_hallucinated = similarity < threshold
    confidence = 1 - similarity
    return {
        "is_hallucinated": is_hallucinated,
        "confidence": confidence,
        "similarity": similarity
    }

# Robust path resolution: works whether run from project root or src/
data_path = Path(__file__).resolve().parent.parent / "data" / "qa_data.json"
if not data_path.exists():
    data_path = Path("data/qa_data.json")

if not data_path.exists():
    print(f"❌ Error: {data_path} not found!")
    exit(1)

# Read JSON Lines format
data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  # Skip empty lines
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping invalid line: {e}")
                continue

print(f"✅ Loaded {len(data)} samples from {data_path}")

# Check if we have the right structure
if len(data) == 0:
    print("❌ No valid data found!")
    exit(1)

# Display first sample structure
print(f"\n📋 First sample keys: {list(data[0].keys())}")

# Count samples with both right and hallucinated answers
valid_samples = 0
for sample in data:
    if 'right_answer' in sample and 'hallucinated_answer' in sample and 'knowledge' in sample:
        valid_samples += 1

print(f"✅ Found {valid_samples} samples with both correct and hallucinated answers")

if valid_samples == 0:
    print("\n❌ The dataset doesn't have 'right_answer' and 'hallucinated_answer' fields.")
    print("Available fields in first sample:", list(data[0].keys()))
    exit(1)

print("\n" + "="*50)
print("Testing on first 10 valid samples...")
print("="*50)

correct = 0
total = 0
samples_tested = 0

for i, sample in enumerate(data):
    if samples_tested >= 10:
        break
    
    # Check if sample has required fields
    if 'knowledge' not in sample or 'right_answer' not in sample or 'hallucinated_answer' not in sample:
        continue
    
    context = sample['knowledge']
    question = sample.get('question', 'No question provided')
    
    # Test hallucinated answer (should be detected as hallucinated)
    result_hall = detect_hallucination_bertscore(context, sample['hallucinated_answer'])
    # Test correct answer (should NOT be detected as hallucinated)
    result_correct = detect_hallucination_bertscore(context, sample['right_answer'])
    
    if result_hall['is_hallucinated']:
        correct += 1
    if not result_correct['is_hallucinated']:
        correct += 1
    total += 2
    
    # Show first 3 examples in detail
    if samples_tested < 3:
        print(f"\n📝 Sample {samples_tested+1}: {question[:60]}...")
        print(f"   Hallucinated answer similarity: {result_hall['similarity']:.3f} -> {'✓ DETECTED' if result_hall['is_hallucinated'] else '✗ MISSED'}")
        print(f"   Correct answer similarity: {result_correct['similarity']:.3f} -> {'✓ CORRECT' if not result_correct['is_hallucinated'] else '✗ FALSE ALARM'}")
    
    samples_tested += 1

if total > 0:
    accuracy = (correct / total) * 100
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
    
    if accuracy >= 70:
        print("✅ Good! Your detector is working well!")
    elif accuracy >= 50:
        print("📊 Moderate accuracy. We can improve with NLI detection.")
    else:
        print("⚠️  Low accuracy. We need to adjust the threshold or use NLI.")
    
    print(f"\n💡 Tip: Try adjusting the threshold (currently 0.5)")
else:
    print("❌ No valid samples were tested.")