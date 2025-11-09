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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ Warning: Missing OPENAI_API_KEY. App will not function properly.")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

client = OpenAI(api_key=OPENAI_API_KEY)

# === PDF Loader ===
def read_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 200:  # skip very small fragments
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
    arabic_chars = re.findall(r"[\u0600-\u06FF]", text)
    return "ar" if len(arabic_chars) / max(len(text), 1) > 0.2 else "en"

# === Context Retrieval ===
def retrieve_context(question, lang="en", top_k=5):
    if lang == "ar":
        vec = vectorizer_ar.transform([question])
        sims = cosine_similarity(tfidf_ar, vec).ravel()
        idx = np.argsort(-sims)[:top_k]
        return "\n\n".join(chunks_ar[i] for i in idx)
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

    if lang == "ar":
        system_prompt = (
            "أنت مساعد قانوني متخصص في نظام العمل السعودي. "
            "تستند إجاباتك إلى نظام العمل السعودي بنسختيه العربية والإنجليزية، "
            "وتلتزم بالصياغة الرسمية والواضحة باللغة العربية الفصحى. "
            "عند الاقتضاء، أشر إلى المادة ذات الصلة مثل (المادة ٨٠). "
            "أجب بإيجاز ووضوح دون استخدام زخارف أو رموز أو تنسيق خاص."
        )
    else:
        system_prompt = (
            "You are a legal assistant specializing in the Saudi Labor Law. "
            "Base your answers on both the Arabic and English versions of the law. "
            "When applicable, refer to the relevant article (e.g., Article 80). "
            "Respond in clear, professional English prose with no markdown, symbols, or formatting."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {q}\n\nRelevant Text:\n{context}"}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=500
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
