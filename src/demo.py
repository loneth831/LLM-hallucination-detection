from bert_score import score
import gradio as gr
import json
from pathlib import Path

print("Loading BERTScore model...")

def detect_hallucination(context, answer, threshold=0.5):
    """Simple BERTScore-based hallucination detection"""
    P, R, F1 = score([answer], [context], lang="en", verbose=False)
    similarity = F1.item()
    is_hallucinated = similarity < threshold
    confidence = abs(similarity - 0.5) * 2
    
    return {
        "is_hallucinated": is_hallucinated,
        "similarity": similarity,
        "confidence": confidence
    }

def analyze(context, question, answer):
    """Gradio interface function"""
    if not context or not answer:
        return "⚠️ Please provide both context and answer", None
    
    result = detect_hallucination(context, answer)
    
    if result["is_hallucinated"]:
        verdict = "❌ HALLUCINATION DETECTED"
        color = "red"
        explanation = f"The answer has low semantic similarity ({result['similarity']:.2f}) with the context, suggesting hallucination."
    else:
        verdict = "✅ FACTUALLY GROUNDED"
        color = "green"
        explanation = f"The answer shows good semantic similarity ({result['similarity']:.2f}) with the context."
    
    html = f"""
    <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 4px solid {color};">
        <h2 style="color: {color}; margin: 0 0 10px 0;">{verdict}</h2>
        <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
        <p><strong>Similarity Score:</strong> {result['similarity']:.3f} (0=unrelated, 1=identical)</p>
        <hr>
        <p><strong>Explanation:</strong> {explanation}</p>
        <p><small>Threshold: 0.5 (answers below this are flagged as hallucinations)</small></p>
    </div>
    """
    return html, result["similarity"]

# Create Gradio interface
demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(label="📖 Context / Source Document", lines=6, 
                   placeholder="Paste the source text here..."),
        gr.Textbox(label="❓ Question", lines=2,
                   placeholder="What was asked to the LLM?"),
        gr.Textbox(label="🤖 LLM Answer", lines=4,
                   placeholder="Paste the LLM's response here...")
    ],
    outputs=[
        gr.HTML(label="Detection Result"),
        gr.Number(label="Similarity Score", visible=False)
    ],
    title="🔍 LLM Hallucination Detection System",
    description="Detect if an LLM's answer is hallucinated based on the provided context.",
    examples=[
        ["The Eiffel Tower is located in Paris, France. It was built in 1889 and is 330 meters tall.", 
         "Where is the Eiffel Tower?", 
         "The Eiffel Tower is in Paris."],
        ["The Eiffel Tower is located in Paris, France. It was built in 1889 and is 330 meters tall.", 
         "Where is the Eiffel Tower?", 
         "The Eiffel Tower is in London, England."],
        ["Albert Einstein won the Nobel Prize in Physics in 1921 for the photoelectric effect.", 
         "What did Einstein win the Nobel Prize for?", 
         "Einstein won the Nobel Prize for his theory of relativity."]
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=True)
    print("\n✅ Demo running! Click the URL to open in your browser")