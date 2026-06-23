import os
import uuid
import json
from typing import Optional
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
    current_app,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.utility_functions import (
    fetch_job_description_from_url,
    extract_text_from_file,
    generate_feedback,
    analyze_skill_gap,
    generate_summary,
    extract_keywords,
    save_ranked_data,
    load_ranked_data,
    suggest_jobs,
)
from utils.resume_rewriter import rewrite_resume

# ------------------- Config ------------------- #
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "a_super_secret_key_that_is_not_so_secret")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "resumes")
REWRITTEN_FOLDER = os.path.join(BASE_DIR, "rewritten_resumes")
TEMP_STORAGE = os.path.join(BASE_DIR, "temp_files")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
STORAGE_FILE = os.path.join(STORAGE_DIR, "ranked_data.json")

for p in (UPLOAD_FOLDER, REWRITTEN_FOLDER, TEMP_STORAGE, STORAGE_DIR):
    os.makedirs(p, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def unique_filename(filename: str) -> str:
    base = secure_filename(filename)
    return f"{uuid.uuid4().hex}_{base}"


# ------------------- Routes ------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/rank", methods=["POST"])
def rank():
    job_desc = request.form.get("jobdesc", "").strip()
    job_url = request.form.get("joburl", "").strip()
    files = request.files.getlist("resumes")

    if job_url and not job_desc:
        job_desc = fetch_job_description_from_url(job_url)

    if not job_desc:
        flash("Job description is required to rank resumes.", "error")
        return redirect(url_for("index"))

    if not files or not files[0].filename:
        flash("Please upload at least one resume (PDF or DOCX).", "error")
        return redirect(url_for("index"))

    resumes_data = []
    for f in files:
        if not f or not f.filename:
            continue

        if not allowed_file(f.filename):
            flash(f"{f.filename} - unsupported file type. Only PDF and DOCX allowed.", "error")
            continue

        saved_name = unique_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, saved_name)
        try:
            f.save(path)
        except Exception as e:
            current_app.logger.exception("Failed to save uploaded file")
            flash(f"Failed to save file {f.filename}: {e}", "error")
            continue

        text = extract_text_from_file(path)
        if text and text.strip():
            resumes_data.append({"filename": saved_name, "orig_name": f.filename, "text": text})

    if not resumes_data:
        flash("No valid resumes were found. Please upload readable PDF or DOCX files.", "error")
        return redirect(url_for("index"))

    docs = [job_desc] + [r["text"] for r in resumes_data]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        vectors = vectorizer.fit_transform(docs)
        similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    except Exception as e:
        current_app.logger.exception("Vectorisation or similarity failed")
        flash(f"Error computing similarity: {e}", "error")
        return redirect(url_for("index"))

    leaderboard = []
    all_keywords = []
    for i, r in enumerate(resumes_data):
        score = float(similarity[i])
        resume_text = r["text"]
        keywords = extract_keywords(resume_text) or []
        all_keywords.extend(keywords)

        leaderboard.append(
            {
                "file": r["filename"],
                "orig_name": r.get("orig_name", r["filename"]),
                "score": f"{score * 100:.2f}%",
                "feedback": generate_feedback(resume_text),
                "summary": generate_summary(resume_text),
                "skill_gap": ", ".join(analyze_skill_gap(resume_text, job_desc)),
                "keywords": keywords,
                "original_text": resume_text,
                "job_desc": job_desc,
            }
        )

    leaderboard.sort(key=lambda x: float(x["score"].strip("%")), reverse=True)

    suggestions = suggest_jobs(list(dict.fromkeys(all_keywords)))
    try:
        save_ranked_data(leaderboard)
    except TypeError:
        try:
            with open(STORAGE_FILE, "w", encoding="utf-8") as fh:
                json.dump(leaderboard, fh, ensure_ascii=False, indent=2)
        except Exception:
            current_app.logger.exception("Failed to persist ranked data")

    return render_template("index.html", ranked=leaderboard, suggestions=suggestions)


@app.route("/dashboard")
def dashboard():
    try:
        ranked_data = load_ranked_data()
    except TypeError:
        if os.path.exists(STORAGE_FILE):
            try:
                with open(STORAGE_FILE, "r", encoding="utf-8") as fh:
                    ranked_data = json.load(fh)
            except Exception:
                current_app.logger.exception("Failed to load storage file")
                ranked_data = []
        else:
            ranked_data = []
    return render_template("dashboard.html", ranked=ranked_data)


@app.route("/preview_rewrite", methods=["POST"])
def preview_rewrite():
    filename = request.form.get("filename")
    original_text = request.form.get("original_text")
    job_desc = request.form.get("job_desc", "")

    if not filename or not original_text:
        flash("Error: file or content missing.", "error")
        return redirect(url_for("index"))

    # Call rewrite (may use Replicate). It can return a string message on failure.
    try:
        rewritten_text = rewrite_resume(original_text, job_desc)
    except Exception:
        current_app.logger.exception("rewrite_resume failed")
        flash("Failed to rewrite the resume (check server log).", "error")
        return redirect(url_for("index"))

    safe_base = secure_filename(os.path.splitext(filename)[0]) or "resume"
    unique_name = f"rewritten_{safe_base}_{uuid.uuid4().hex}.txt"
    new_path = os.path.join(TEMP_STORAGE, unique_name)

    try:
        with open(new_path, "w", encoding="utf-8") as fh:
            fh.write(rewritten_text)
    except Exception:
        current_app.logger.exception("Failed to save rewritten text")
        flash("Failed to save rewritten text file.", "error")
        return redirect(url_for("index"))

    return render_template(
        "preview.html",
        original_text=original_text,
        rewritten_text=rewritten_text,
        filename=unique_name,
        original_filename=filename,
    )


@app.route("/download_txt/<path:filename>")
def download_txt(filename):
    txt_path = os.path.join(TEMP_STORAGE, filename)
    if not os.path.exists(txt_path):
        current_app.logger.warning("Requested txt not found: %s", txt_path)
        flash(f"Text file {filename} not found.", "error")
        return redirect(url_for("index"))

    try:
        return send_from_directory(TEMP_STORAGE, filename, as_attachment=True)
    except Exception:
        current_app.logger.exception("Failed to send txt file")
        flash("Failed to send the text file.", "error")
        return redirect(url_for("index"))


# ---------- PDF helpers (FPDF + ReportLab fallback) ----------
from fpdf import FPDF

def create_pdf_with_fpdf(text, pdf_path, font_path):
    pdf = FPDF()
    pdf.add_page()

    pdf.add_font(
        "DejaVu",
        "",
        font_path,
        uni=True
    )

    pdf.set_font("DejaVu", size=11)

    for line in text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf.output(pdf_path)


def create_pdf_with_reportlab(text: str, pdf_path: str, font_path: Optional[str] = None) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except Exception as e:
        raise RuntimeError("ReportLab not installed") from e

    font_name = "Helvetica"
    if font_path and os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("DejaVu", font_path))
            font_name = "DejaVu"
        except Exception:
            font_name = "Helvetica"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 40
    line_height = 12
    x = margin
    y = height - margin
    c.setFont(font_name, 10)

    for raw_line in str(text).splitlines():
        line = raw_line or ""
        words = line.split(" ")
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if c.stringWidth(test, font_name, 10) < (width - 2 * margin):
                cur = test
            else:
                c.drawString(x, y, cur)
                y -= line_height
                cur = w
                if y < margin + line_height:
                    c.showPage()
                    c.setFont(font_name, 10)
                    y = height - margin
        if cur:
            c.drawString(x, y, cur)
            y -= line_height
        if y < margin + line_height:
            c.showPage()
            c.setFont(font_name, 10)
            y = height - margin

    c.save()


@app.route("/download_pdf/<path:filename>")
def download_pdf(filename):
    txt_path = os.path.join(TEMP_STORAGE, filename)
    if not os.path.exists(txt_path):
        current_app.logger.warning("Requested txt for PDF not found: %s", txt_path)
        flash(f"Text file {filename} not found for PDF conversion.", "error")
        return redirect(url_for("index"))

    try:
        with open(txt_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        current_app.logger.exception("Failed to read rewritten text file")
        flash("Failed to read the rewritten text file (encoding/read error).", "error")
        return redirect(url_for("index"))

    pdf_filename = f"{os.path.splitext(filename)[0]}.pdf"
    pdf_path = os.path.join(TEMP_STORAGE, pdf_filename)

    font_candidate = os.path.join(BASE_DIR, "static", "fonts", "DejaVuSans.ttf")
    font_path = font_candidate if os.path.exists(font_candidate) else None

    # Try FPDF then ReportLab fallback
    try:
        create_pdf_with_fpdf(text, pdf_path, font_path=font_path)
        current_app.logger.info("PDF created with FPDF: %s", pdf_path)
    except Exception:
        current_app.logger.exception("FPDF failed, attempting ReportLab fallback")
        try:
            create_pdf_with_reportlab(text, pdf_path, font_path=font_path)
            current_app.logger.info("PDF created with ReportLab: %s", pdf_path)
        except Exception:
            current_app.logger.exception("ReportLab fallback failed")
            flash("An error occurred during PDF generation. Check server console for details.", "error")
            return redirect(url_for("index"))

    try:
        return send_from_directory(TEMP_STORAGE, pdf_filename, as_attachment=True)
    except Exception:
        current_app.logger.exception("Failed to send PDF file")
        flash("Failed to send the generated PDF.", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    # optional: increase logger level
    import logging
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
