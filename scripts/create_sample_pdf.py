"""
create_sample_pdf.py - 테스트용 기술 PDF 문서 생성 스크립트
"""

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PDF_PATH = Path("docs/data/gemma_technical_overview.pdf")
PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

c = canvas.Canvas(str(PDF_PATH), pagesize=letter)
width, height = letter

# Page 1: Introduction & Architecture
c.setFont("Helvetica-Bold", 18)
c.drawString(50, height - 60, "Technical Overview: Google Gemma Architecture")

c.setFont("Helvetica-Bold", 12)
c.drawString(50, height - 100, "1. Model Family and Specifications")
c.setFont("Helvetica", 10)
text_p1 = [
    "Google Gemma is a family of lightweight, state-of-the-art open models built from",
    "the same research and technology used to create the Gemini models.",
    "Gemma models are available in two primary parameter sizes:",
    " - Gemma 2B: Designed for resource-constrained environments and on-device execution.",
    " - Gemma 9B/7B: Delivers industry-leading performance for general reasoning and coding.",
    "",
    "Key Architectural Highlights:",
    " - Multi-Query Attention (MQA) / Grouped-Query Attention (GQA) for faster inference.",
    " - RoPE (Rotary Position Embeddings) for effective context preservation.",
    " - GeGLU non-linear activations replacing standard ReLU/GELU.",
]
y = height - 120
for line in text_p1:
    c.drawString(50, y, line)
    y -= 18

c.showPage()

# Page 2: Local Deployment & RAG Integration
c.setFont("Helvetica-Bold", 18)
c.drawString(50, height - 60, "Technical Overview: Gemma in RAG Systems")

c.setFont("Helvetica-Bold", 12)
c.drawString(50, height - 100, "2. Efficient Local Deployment via Ollama")
c.setFont("Helvetica", 10)
text_p2 = [
    "Running Gemma locally provides complete data confidentiality and zero API costs.",
    "Using Ollama, Gemma can be executed with 4-bit and 8-bit quantization (GGUF).",
    "Command to launch Gemma locally: 'ollama run gemma2:2b'",
    "",
    "3. Grounding and Context Window Guidelines:",
    " - Keep the prompt structured with explicit delimiters: [Context] and [Question].",
    " - Set temperature between 0.0 and 0.2 to minimize creative hallucinations.",
    " - Include negative constraint: Instruct the model to say 'I do not know' if information is missing.",
]
y = height - 120
for line in text_p2:
    c.drawString(50, y, line)
    y -= 18

c.save()
print(f"✅ Generated sample PDF: {PDF_PATH} (2 pages)")
