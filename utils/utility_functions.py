# utils/utility_functions.py
import os
import json
import re
import requests
from bs4 import BeautifulSoup
import docx2txt
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Job description fetching -------------------------------------------------
def fetch_job_description_from_url(url: str) -> str:
    """
    Fetch main text from a job posting URL.
    Returns a text blob (trimmed to 20k chars) or empty string on failure.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # try common containers: article, main, role description blocks, paragraphs
        content = ""
        article = soup.find("article")
        if article:
            content = article.get_text(separator=" ", strip=True)
        else:
            main = soup.find("main")
            if main:
                content = main.get_text(separator=" ", strip=True)
            else:
                # fall back to concatenated paragraphs
                paragraphs = soup.find_all("p")
                content = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

        # simple cleanup
        content = re.sub(r"\s+", " ", content).strip()
        return content[:20000]
    except Exception:
        return ""

# --- File text extraction ---------------------------------------------------
def extract_text_from_file(path: str) -> str:
    """
    Extract text from .pdf or .docx file.
    Returns empty string on error.
    """
    _, ext = os.path.splitext(path.lower())
    try:
        if ext == ".docx":
            # docx2txt returns plain text
            return docx2txt.process(path) or ""
        elif ext == ".pdf":
            text_parts = []
            with fitz.open(path) as doc:
                for page in doc:
                    text_parts.append(page.get_text())
            return "\n".join(text_parts)
        else:
            return ""
    except Exception:
        return ""

# --- Keyword extraction -----------------------------------------------------
def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Return top_n TF-IDF keywords (unigrams and bigrams) from a single text.
    If extraction fails, returns an empty list.
    """
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=4000)
        tfidf = vectorizer.fit_transform([text])
        scores = tfidf.toarray().flatten()
        features = vectorizer.get_feature_names_out()
        ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
        keywords = [w for w, s in ranked[:top_n] if s > 0]
        return keywords
    except Exception:
        return []

# --- Simple summary ----------------------------------------------------------
def generate_summary(text: str, max_sentences: int = 3) -> str:
    """
    Heuristic summary: picks the first few long sentences.
    Not an ML summariser, but works for many resumes.
    """
    if not text:
        return ""
    # split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    meaningful = [s.strip() for s in sentences if len(s.strip()) > 30]
    return " ".join(meaningful[:max_sentences])

# --- Basic feedback rules ---------------------------------------------------
def generate_feedback(text: str) -> str:
    """
    Return short feedback based on simple rules.
    Extend this with your own heuristics.
    """
    if not text:
        return "No content found."

    feedback = []
    word_count = len(text.split())
    if word_count < 120:
        feedback.append("Resume is short — add more detail about results.")
    if not any(x in text.lower() for x in ["%", "increased", "reduced", "improved", "growth", "decreased", "saved"]):
        feedback.append("Add metrics where possible (%, numbers).")
    if any(word in text.lower() for word in ["managed", "led", "supervised"]):
        feedback.append("Good leadership language — quantify team size or impact.")

    return " ".join(feedback) if feedback else "Resume looks concise and achievement-focused."

# --- Skill gap analysis -----------------------------------------------------
def analyze_skill_gap(resume_text: str, job_description: str) -> list:
    """
    Compare top keywords in JD vs resume and return missing keywords (max 12).
    """
    if not job_description or not resume_text:
        return []

    jd_keywords = set(extract_keywords(job_description, top_n=40))
    resume_keywords = set(extract_keywords(resume_text, top_n=40))
    missing = list(jd_keywords - resume_keywords)
    # Keep ordering stable and limit results
    return missing[:12]

# --- Persistence helpers ----------------------------------------------------
def save_ranked_data(data, storage_path: str = "storage/ranked_data.json"):
    """
    Save ranked data to a JSON file. Creates folders if necessary.
    """
    try:
        dirpath = os.path.dirname(storage_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(storage_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        # fail silently (app handles errors elsewhere)
        pass

def load_ranked_data(storage_path: str = "storage/ranked_data.json"):
    """
    Load ranked data from JSON. Returns empty list if file missing or unreadable.
    """
    try:
        if not os.path.exists(storage_path):
            return []
        with open(storage_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []

# --- Job suggestions --------------------------------------------------------
def suggest_jobs(keywords: list) -> list:
    """
    Create a couple of job search URLs using the provided keyword list.
    Keep it simple — returns Google search links.
    """
    if not keywords:
        return []
    base = "https://www.google.com/search?q="
    q1 = "+".join(keywords[:4] + ["jobs"])
    q2 = "+".join(keywords[:3] + ["remote", "jobs"])
    return [base + q1, base + q2]
