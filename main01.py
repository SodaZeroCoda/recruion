import os
import json
import tempfile
import http.client
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import docx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login

# 🔐 Hugging Face API-nyckel
login(token=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")

# 🚀 FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Modell
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🔑 API-konstanter
JOOBLE_KEY    = ""
JOOBLE_HOST   = "jooble.org"
JOBINDEX_URL  = "https://api.jobindex.dk/api/v1/job/search"
NAV_URL       = "https://arbeidsplassen.nav.no/public-feed/api/v1/ads"

MAX_JOBS = 100

# Läs in alla nordiska kommuner från citys.json
with open(os.path.join(os.path.dirname(__file__), "citys.json"), encoding="utf-8") as f:
    NORDIC_LOCATIONS = json.load(f)

def get_company_logo(company_name: str) -> str:
    if not company_name:
        return ""
    domain = company_name.lower().replace(" ", "") + ".com"
    return f"https://logo.clearbit.com/{domain}"

def extract_text(uploaded_file: UploadFile) -> str:
    suffix = os.path.splitext(uploaded_file.filename)[1].lower()
    if suffix not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Endast PDF/DOCX tillåts.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            pages = PdfReader(tmp_path).pages
            return "\n".join(p.extract_text() or "" for p in pages)
        else:
            doc = docx.Document(tmp_path)
            return "\n".join(para.text for para in doc.paragraphs)
    finally:
        os.remove(tmp_path)

def fetch_jooble():
    jobs = []
    for loc in NORDIC_LOCATIONS:
        if len(jobs) >= MAX_JOBS:
            break
        try:
            conn = http.client.HTTPSConnection(JOOBLE_HOST)
            payload = json.dumps({"keywords": "", "location": loc})
            conn.request("POST", f"/api/{JOOBLE_KEY}", payload, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status == 200:
                data = json.loads(resp.read())
                for job in data.get("jobs", []):
                    logo = job.get("company_logo") or get_company_logo(job.get("company"))
                    jobs.append({
                        "title":       job.get("title", ""),
                        "company":     job.get("company", ""),
                        "location":    job.get("location", ""),
                        "description": job.get("snippet", ""),
                        "url":         job.get("link", ""),
                        "logo":        logo,
                        "source":      "Jooble"
                    })
            conn.close()
        except Exception:
            pass
    return jobs[:MAX_JOBS]

def fetch_jobindex():
    jobs = []
    try:
        res = requests.get(JOBINDEX_URL, params={"q": "", "limit": MAX_JOBS})
        res.raise_for_status()
        for job in res.json().get("results", []):
            logo = job.get("logo") or get_company_logo(job.get("company"))
            jobs.append({
                "title":       job.get("title", ""),
                "company":     job.get("company", ""),
                "location":    job.get("city", ""),
                "description": job.get("description", ""),
                "url":         job.get("url", ""),
                "logo":        logo,
                "source":      "Jobindex"
            })
    except Exception:
        pass
    return jobs[:MAX_JOBS]

def fetch_nav():
    jobs = []
    try:
        res = requests.get(NAV_URL)
        res.raise_for_status()
        for job in res.json().get("content", []):
            emp = job.get("employer", {}).get("name", "")
            logo = job.get("employer", {}).get("logoUrl") or get_company_logo(emp)
            jobs.append({
                "title":       job.get("title", ""),
                "company":     emp,
                "location":    job.get("location", {}).get("municipal", ""),
                "description": job.get("description", ""),
                "url":         job.get("url", ""),
                "logo":        logo,
                "source":      "NAV"
            })
    except Exception:
        pass
    return jobs[:MAX_JOBS]

@app.get("/jobs", response_class=JSONResponse)
def get_all_jobs():
    all_jobs = (
        fetch_jooble() +
        fetch_jobindex() +
        fetch_nav()
    )
    return {"total": len(all_jobs), "jobs": all_jobs}

@app.post("/match", response_class=JSONResponse)
async def match(cv: UploadFile = File(...)):
    cv_text = extract_text(cv)
    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="CV:t är tomt.")

    jobs = get_all_jobs()["jobs"]
    if not jobs:
        return {"matches": [], "total_found": 0, "filtered": 0}

    texts   = [ (j["title"] + " ") * 3 + j["company"] + " " + j["description"] for j in jobs ]
    cv_emb   = model.encode([cv_text])
    job_embs = model.encode(texts)
    sims     = cosine_similarity(cv_emb, job_embs)[0]

    matches = []
    for j, score in zip(jobs, sims):
        if score >= 0.3:
            matches.append({
                **j,
                "similarity": round(float(score), 3),
                "ats":         min(100, int(score * 120)),
                "keywords":    min(100, int(score * 90 + 10)),
                "format":      min(100, int(score * 80 + 20)),
            })

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return {"matches": matches, "total_found": len(jobs), "filtered": len(matches)}
