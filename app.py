import os
import re
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# === Config ===
PDF_PATH_AR = os.getenv("PDF_PATH_AR", "saudi_labor_law_arabic.pdf")
PDF_PATH_EN = os.getenv("PDF_PATH_EN", "saudi_labor_law_english.pdf")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in environment variables.")
client = OpenAI(api_key=OPENAI_API_KEY)

# === PDF Loader ===
def read_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        # Normalize Arabic presentation forms and remove diacritics
        page_text = re.sub(r"[\u064B-\u065F\u0670]", "", page_text)
        text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 200:
            chunks.append(chunk)
    return chunks

print("🔹 Loading Arabic and English Labor Law PDFs...")
text_ar = read_pdf_text(PDF_PATH_AR)
text_en = read_pdf_text(PDF_PATH_EN)

chunks_ar = chunk_text(text_ar)
chunks_en = chunk_text(text_en)
print(f"✅ Loaded {len(chunks_ar)} Arabic chunks and {len(chunks_en)} English chunks.")

# === Build TF-IDF Indexes ===
vectorizer_ar = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), min_df=2)
vectorizer_en = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
tfidf_ar = vectorizer_ar.fit_transform(chunks_ar)
tfidf_en = vectorizer_en.fit_transform(chunks_en)
print("✅ Arabic and English TF-IDF indexes ready.")

# === Language Detection ===
def detect_language(text):
    arabic_ratio = len(re.findall(r"[\u0600-\u06FF]", text)) / max(len(text), 1)
    if 0.2 < arabic_ratio < 0.8:
        return "mixed"
    return "ar" if arabic_ratio >= 0.8 else "en"

# === Context Retrieval ===
def retrieve_context(question, lang="en", top_k=5):
    if lang == "ar":
        vec = vectorizer_ar.transform([question])
        sims = cosine_similarity(tfidf_ar, vec).ravel()
        idx = np.argsort(-sims)[:top_k]
        return "\n\n".join(chunks_ar[i] for i in idx)
    elif lang == "mixed":
        vec_ar = vectorizer_ar.transform([question])
        vec_en = vectorizer_en.transform([question])
        sims_ar = cosine_similarity(tfidf_ar, vec_ar).ravel()
        sims_en = cosine_similarity(tfidf_en, vec_en).ravel()
        idx_ar = np.argsort(-sims_ar)[:top_k // 2]
        idx_en = np.argsort(-sims_en)[:top_k // 2]
        return "\n\n".join(chunks_ar[i] for i in idx_ar) + "\n\n" + "\n\n".join(chunks_en[i] for i in idx_en)
    else:
        vec = vectorizer_en.transform([question])
        sims = cosine_similarity(tfidf_en, vec).ravel()
        idx = np.argsort(-sims)[:top_k]
        return "\n\n".join(chunks_en[i] for i in idx)

# === Endpoint ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    q = data.get("question", "").strip()
    if not q:
        return jsonify({"error": "No question provided"}), 400

    lang = detect_language(q)
    context = retrieve_context(q, lang)
    if not context.strip():
        context = "No relevant articles found in the Saudi Labor Law documents."

    # === Strong grounding prompt ===
    system_prompt = (
    "You are the Saudi Labor Law Assistant — an expert bilingual advisor specialized "
    "in the Saudi Labor Law. You have access to both the Arabic and English versions of the law. "
    "Your role is to explain, clarify, and provide guidance strictly based on the information found "
    "within these official Saudi Labor Law documents. Do not use or refer to any external standards, "
    "such as ISO, or unrelated legal frameworks. "
    "\n\n"
    "Always detect the language of the user's question: if it is asked in Arabic, answer in Arabic; "
    "if in English, answer in English. If the question mixes both, use the dominant language. "
    "Use natural, professional tone and clear formatting to make your explanation easy to read. "
    "\n\n"
    "When explaining, you may summarize or rephrase legal content in simpler terms, "
    "but ensure accuracy and compliance with the Saudi Labor Law. "
    "Whenever applicable, mention relevant article numbers or sections that support your explanation. "
    "If the law does not clearly address a question, respond with: "
    "'I could not find a specific reference in the Saudi Labor Law for that question.' "
    "Avoid speculation or advice outside the provided law context."
)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question:\n{q}\n\n---\nCONTEXT:\n{context}\n---"}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=600
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "message": "POST /ask {question:'...'}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
