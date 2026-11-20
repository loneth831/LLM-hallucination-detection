import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import gradio as gr
import json
from src.detector.models import VerificationRequest
from src.detector.hybrid_detector import HybridHallucinationDetector

print("Initializing Hybrid Hallucination Detector for Dashboard...")
detector = HybridHallucinationDetector(use_bertscore=False)
print("✅ Detector ready!")

def run_audit(context, question, answer, threshold):
    if not context or not answer:
        return "⚠️ <b>Error:</b> Please provide both a Context Document and an LLM Answer.", "", ""

    req = VerificationRequest(
        context=context,
        question=question,
        answer=answer,
        threshold=threshold
    )
    res = detector.verify(req)

    # 1. Main Verdict Banner
    if not res.is_hallucinated:
        verdict_color = "#10b981"  # Emerald Green
        verdict_icon = "✅"
        badge_text = "FACTUALLY GROUNDED"
    else:
        verdict_color = "#ef4444"  # Red
        verdict_icon = "❌"
        badge_text = "HALLUCINATION DETECTED"

    header_html = f"""
    <div style="background: {verdict_color}15; border-left: 6px solid {verdict_color}; padding: 18px 22px; border-radius: 8px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="color: {verdict_color}; margin: 0; font-size: 22px;">{verdict_icon} {badge_text}</h2>
            <span style="font-size: 13px; color: #64748b; background: white; padding: 4px 10px; border-radius: 12px; border: 1px solid #e2e8f0;">
                Latency: <b>{res.latency_ms:.1f}ms</b>
            </span>
        </div>
        <p style="margin: 8px 0 0 0; color: #334155; font-size: 15px;">{res.summary}</p>
        <div style="display: flex; gap: 24px; margin-top: 14px; font-size: 14px;">
            <span>Grounding Score: <b style="color: {verdict_color};">{res.overall_score:.1%}</b></span>
            <span>Confidence: <b>{res.confidence:.1%}</b></span>
            <span>Threshold: <b>{threshold:.2f}</b></span>
        </div>
    </div>
    """

    # 2. Granular Sentence-by-Sentence Breakdown
    claims_html = '<h3 style="margin-top: 15px; color: #1e293b;">🔬 Claim-by-Claim Attribution & Verification</h3>'
    claims_html += '<div style="display: flex; flex-direction: column; gap: 12px;">'

    for i, c in enumerate(res.claims, 1):
        c_color = "#10b981" if not c.is_hallucinated else "#ef4444"
        c_bg = "#f0fdf4" if not c.is_hallucinated else "#fef2f2"
        c_status = "SUPPORTED" if not c.is_hallucinated else "HALLUCINATED"
        
        entities_badge = ""
        if c.ungrounded_entities:
            badges = " ".join([f'<span style="background: #fee2e2; color: #991b1b; padding: 2px 7px; border-radius: 4px; font-size: 12px; border: 1px solid #fca5a5;">⚠️ {e}</span>' for e in c.ungrounded_entities])
            entities_badge = f'<div style="margin-top: 8px;"><b>Fabricated / Missing Entities:</b> {badges}</div>'

        claims_html += f"""
        <div style="background: {c_bg}; border: 1px solid {c_color}40; border-left: 4px solid {c_color}; padding: 14px 18px; border-radius: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 12px; font-weight: bold; color: {c_color}; text-transform: uppercase;">
                    Claim {i} • {c_status} ({c.score:.1%})
                </span>
                <span style="font-size: 12px; color: #64748b;">Confidence: {c.confidence:.1%}</span>
            </div>
            <div style="font-size: 15px; color: #0f172a; font-weight: 500; margin-bottom: 6px;">
                "{c.claim}"
            </div>
            <div style="font-size: 13px; color: #475569;">
                {c.reasoning}
            </div>
            {entities_badge}
        </div>
        """
    claims_html += '</div>'

    # 3. JSON Payload output
    json_output = res.model_dump_json(indent=2)

    return header_html, claims_html, json_output

# Build Gradio Blocks Dashboard
custom_css = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
"""

with gr.Blocks(title="Enterprise Hallucination Detector", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Enterprise LLM Hallucination Detection & Grounding Audit
    Audit LLM outputs against source documents using hybrid **semantic embeddings**, **claim-level decomposition**, and **entity consistency verification**.
    """)

    with gr.Row():
        with gr.Column(scale=5):
            context_input = gr.Textbox(
                label="📖 Source Knowledge / Ground Truth Document",
                lines=7,
                placeholder="Paste the verified reference document or retrieval context here..."
            )
            question_input = gr.Textbox(
                label="❓ User Prompt / Question (Optional)",
                lines=1,
                placeholder="What did the user ask the LLM?"
            )
            answer_input = gr.Textbox(
                label="🤖 LLM Generated Response to Audit",
                lines=4,
                placeholder="Paste the model's generated paragraph here..."
            )
            threshold_slider = gr.Slider(
                label="Sensitivity Threshold",
                minimum=0.50,
                maximum=0.95,
                value=0.82,
                step=0.01,
                info="Scores below this threshold are flagged as hallucinations (Default: 0.82)."
            )
            audit_btn = gr.Button("🚀 Run Hallucination Audit", variant="primary", size="lg")

        with gr.Column(scale=6):
            verdict_panel = gr.HTML(label="Audit Verdict")
            claims_panel = gr.HTML(label="Sentence Breakdown")
            with gr.Accordion("📋 Raw JSON Response", open=False):
                json_panel = gr.Code(language="json")

    # Wire event
    audit_btn.click(
        fn=run_audit,
        inputs=[context_input, question_input, answer_input, threshold_slider],
        outputs=[verdict_panel, claims_panel, json_panel]
    )

    # Preset Examples
    gr.Examples(
        examples=[
            [
                "The Eiffel Tower is located in Paris, France. It was built in 1889 for the Exposition Universelle and is 330 meters tall.",
                "Tell me about the Eiffel Tower.",
                "The Eiffel Tower is located in Paris. It was completed in 1889.",
                0.82
            ],
            [
                "The Eiffel Tower is located in Paris, France. It was built in 1889 for the Exposition Universelle and is 330 meters tall.",
                "Where is the Eiffel Tower and when was it built?",
                "The Eiffel Tower is in London, England. It was built in 1945.",
                0.82
            ],
            [
                "Albert Einstein won the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.",
                "What Nobel Prize did Einstein win?",
                "Albert Einstein received the Nobel Prize in Physics for his theory of relativity in 1921.",
                0.82
            ]
        ],
        inputs=[context_input, question_input, answer_input, threshold_slider]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
