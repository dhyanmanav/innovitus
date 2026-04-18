# 🧠 TalentIQ — Multimodal Talent Intelligence System

> **Beyond keyword matching. Evaluate candidates the way a senior engineer actually would.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![No External APIs](https://img.shields.io/badge/External%20AI%20APIs-None-00E5A0?style=flat-square)](#architecture)

---
# The App is now live at http://ai-talent.streamlit.app/

## What is TalentIQ?

TalentIQ is an AI-powered candidate evaluation system that reads **three sources simultaneously** — resume PDF, live GitHub API, and portfolio — and produces an **explainable fit score (0–100)** with skill gap analysis, integrity validation, and recruiter-ready evidence.

Traditional ATS systems count keywords. TalentIQ **verifies claims**.

```
PDF Resume ──┐
GitHub API ──┼──► Skill Extraction ──► Knowledge Graph ──► Fit Score (0–100)
Portfolio ───┘         │                   Expansion           + Explanation
                       └──────────────► Integrity Engine ──► 9 Cross-Source Checks
```

---

## Quick Start

```bash
pip install streamlit plotly PyMuPDF pandas numpy sentence-transformers scikit-learn joblib torch
streamlit run app_v2.py
# Opens at http://localhost:8501
```

> The app works immediately without training. Run `talent_intelligence_training.ipynb` to activate your own fine-tuned model weights.

---

## What it produces

**Candidate view:**
```
FIT SCORE: 78/100  ✅ Strong Match      INTEGRITY: 84/100  Verified

✔ Matched Required:   python, pytorch, bert, nlp, docker
✘ Missing Required:   kubernetes, aws

Learning Path for kubernetes:
  → Step 1: Minikube local cluster
  → Step 2: Pods, Deployments, Services
  → Step 3: Helm charts          (~4–6 weeks)

Best fit across all roles: NLP Engineer 78 > ML Engineer 61 > Data Scientist 54
```

**Recruiter view:**
```
✅ PROCEED TO INTERVIEW
Arjun Sharma → NLP Engineer | Fit 78 | Integrity 84

Skill evidence matrix:
  python     Resume ✅  GitHub ✅  Portfolio ✅  Credibility: 🟢 High
  kubernetes Resume ✗   GitHub ✗   Portfolio ✗   Credibility: 🔴 None

Interview probes (auto-generated):
  Q1: Walk me through a project where you used Kubernetes in production.
  Q2: Your resume has few measurable outcomes — can you quantify impact?
```

---

## Architecture

### System layers

```
Layer 1 — Input
  PDF Resume (PyMuPDF)  |  GitHub API (live fetch)  |  Portfolio text

Layer 2 — Extraction
  Skill extractor: 100+ skills, 40 aliases, regex + knowledge graph
  Name / experience / companies / metrics / hyperlinks

Layer 3 — Representation
  Fine-tuned MiniLM-L6-v2 embeddings  (own weights)
  Knowledge graph parent inference
  GitHub: language map + topic map + text extraction

Layer 4 — Scoring
  Cosine similarity (profile ↔ job description)
  Skill overlap (required × 0.6 + preferred × 0.3 + nice × 0.1)
  Fit score = 0.5×semantic + 0.5×skill_overlap × experience_factor
  GBM regressor trained on 6 features  (own weights)

Layer 5 — Output
  Integrity engine: 9 cross-source checks
  Streamlit: Candidate mode (5 tabs) + Recruiter mode (full evidence view)
```

### Score formula

```python
fit_score = min(100,
    (0.5 × semantic_score + 0.5 × skill_overlap_score) × experience_factor
)

# skill_overlap_score
= (req_matched/req_total × 0.6
 + pref_matched/pref_total × 0.3
 + nice_matched/nice_total × 0.1) × 100

# experience_factor
= 0.85  if under-experienced
  1.0   if within typical range
  0.97  if significantly over-qualified
```

Every number in the score has a named source. Nothing is a black box.

### Knowledge graph (12 domains)

```python
SKILL_GRAPH = {
    "python":           ["numpy", "pandas", "pytorch", "tensorflow", "fastapi", ...],
    "deep_learning":    ["cnn", "rnn", "lstm", "transformers", "bert", "gpt", ...],
    "nlp":              ["tokenization", "sentiment_analysis", "embeddings", "spacy", ...],
    "cloud":            ["aws", "gcp", "azure", "docker", "kubernetes", "terraform", ...],
    "data_engineering": ["sql", "spark", "kafka", "airflow", "etl", "dbt", ...],
    ...  # 7 more domains
}

# Inference: pytorch + bert present → deep_learning inferred automatically
# Prevents penalising specialists who never wrote the domain name explicitly
```

---

## Own Architecture, Own Weights

### Fine-tuned sentence transformer

Base model `all-MiniLM-L6-v2` fine-tuned on 1,258 resume–job description pairs using `CosineSimilarityLoss`. The weights in `saved_models/finetuned_talent_model/` are domain-adapted to technical hiring vocabulary and differ from the base model.

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
model.fit(
    train_objectives=[(train_loader, CosineSimilarityLoss(model))],
    epochs=2,
    output_path='./saved_models/finetuned_talent_model',  # ← our weights
)
```

### Score regressor (trained from scratch)

`GradientBoostingRegressor` with 6 engineered features, trained on our labeled dataset:

| Feature | Description |
|---------|-------------|
| `cosine_similarity` | Embedding similarity between profile and JD |
| `skill_overlap_ratio` | Fraction of all job skills matched |
| `skill_gap_count` | Missing required skills (integer) |
| `experience_years` | Years extracted from text |
| `skill_breadth` | Total distinct skills in expanded set |
| `matched_skill_count` | Absolute matched skill count |

```
MAE: ~4–7 points out of 100     R²: ~0.85–0.90
Saved: saved_models/score_regressor.pkl
```

### Rule-based systems (original design)

- **Skill extractor** — 100+ skills, 40 alias mappings, knowledge graph expansion. Not from any library.
- **Integrity engine** — 9 original cross-source validation checks. No existing tool does this.
- **GitHub skill mapper** — LANG_TO_SKILL + TOPIC_TO_SKILL (30+ mappings from repo data to canonical skills).

---

## The Dataset

Generated by `generate_dataset.py` — 1,480 fully labeled training samples.

| Property | Value |
|----------|-------|
| Total samples | 1,480 |
| Job roles | 8 (balanced, 185 each) |
| Overlap levels | High 35% / Medium 40% / Low 25% |
| Score range | 7.0 – 98.2 |
| Average score | ~49.4 |
| Text fields per sample | 4 (resume, github, portfolio, job description) |
| Label fields | fit_score, fit_label, matched_skills, missing_required, suggestions |

**Sample record:**
```json
{
  "id": "sample_0042",
  "target_role": "NLP Engineer",
  "fit_score": 82.4,
  "fit_label": "high",
  "candidate_skills": ["python", "pytorch", "bert", "nlp", "spacy", "docker"],
  "missing_required": ["kubernetes"],
  "suggestions": ["Learn Kubernetes to meet core requirements"],
  "resume_text": "Arjun Sharma\nB.Tech CS, IIT Bombay...",
  "github_text": "GitHub: github.com/arjunsharma...",
  "job_description": "Role: NLP Engineer. Build NLP pipelines..."
}
```

A noise term of ±5 is added to each label — this prevents trivial overfitting and forces the model to learn from text content, not just structural features.

---

## Integrity Engine

Nine checks that no traditional ATS performs:

| Check | What it catches |
|-------|----------------|
| **Name consistency** | Resume ≠ portfolio ≠ GitHub real name ≠ GitHub handle |
| **Skill inflation** | Skills on resume with no GitHub/portfolio evidence |
| **GitHub corroboration** | Resume claims vs actual repo languages and topic tags |
| **Buzzword penalty** | "guru", "ninja", "passionate about", "responsible for" |
| **Metrics audit** | Resume with no percentages, counts, or scale figures |
| **Hyperlink validation** | All PDF links get HTTP HEAD check — broken links flagged |
| **Recency check** | Most recent year in resume < 2022 |
| **Project matching** | Resume project names vs GitHub repo names (fuzzy) |
| **Portfolio alignment** | Skills evidenced in portfolio but absent from resume |

```python
integrity_score = max(0, min(100,
    (passes×10 − issues×15 − warnings×5) / total × 10 + 70
))
```

---

## GitHub API Integration

Enter any username → the system fetches live data:

```
Fetch: user profile + up to 30 repos
Extract from repos:
  - repo.language     → LANG_TO_SKILL map    (Python → python)
  - repo.topics       → TOPIC_TO_SKILL map   (machine-learning → machine_learning)
  - repo.description  → skill regex extractor
  - user.bio          → skill regex extractor
  - repo.name         → fuzzy match vs resume projects

Output: skills_from_github (verified by actual code, not self-reported)
Cache: @st.cache_data(ttl=3600) — one profile fetch per hour
Rate: 60 requests/hour without token (sufficient for demo)
```

---

## File Structure

```
talentiq/
├── app_v2.py                           ← Main Streamlit app
├── generate_dataset.py                 ← Synthetic dataset generator
├── talent_intelligence_training.ipynb  ← Training notebook → produces own weights
├── requirements.txt
│
├── talent_dataset.json                 ← 1,480 training samples (full text)
├── talent_dataset.csv                  ← Flat CSV for inspection
├── skill_graph.json                    ← 12-domain knowledge graph
├── job_roles.json                      ← 8 role definitions with skill tiers
│
└── saved_models/                       ← Created by notebook
    ├── finetuned_talent_model/         ← Fine-tuned MiniLM weights ✅ OWN
    ├── score_regressor.pkl             ← Trained GBM regressor ✅ OWN
    └── feature_scaler.pkl
```

---

## Dashboard — Two Modes

### Candidate mode (5 tabs)

| Tab | Content |
|-----|---------|
| Skills & Gaps | Donut chart + domain depth bars for all 12 domains |
| Integrity Check | Gauge + identity cross-check + issues/warnings/passes + evidence overlap chart |
| Role Fit | Fit bar chart across all 8 roles + radar chart of domain coverage |
| Learning Path | Step-by-step roadmap per missing skill + timeline |
| vs ATS | Side-by-side comparison + score simulation (ATS keyword sim vs TalentIQ) |

### Recruiter mode (separate full-page view)

- PROCEED / REVIEW / PASS verdict banner with exact reason
- 6-metric scorecard
- 3-column breakdown: Red flags | Strengths | Critical gaps
- **Skill evidence matrix**: per-skill table showing Resume ✅/✗, GitHub ✅/✗, Portfolio ✅/✗, credibility rating
- Auto-generated interview probes (tailored to each gap and integrity issue)
- Cross-role fit chart for routing decisions
- GitHub due-diligence: repo count, stars, language distribution
- Full PDF hyperlink validity table

---

## How it Compares to ATS

| Dimension | Traditional ATS | TalentIQ |
|-----------|----------------|---------|
| Skill detection | Keyword count, one document | 3 sources + knowledge graph + 40 aliases |
| Scoring | Keyword density | Semantic similarity + overlap + experience |
| Lie detection | None | GitHub API verifies all resume skill claims |
| Hyperlinks | Not checked | HTTP HEAD validation on all PDF links |
| Buzzwords | Often rewarded | Penalised in integrity score |
| Explainability | Black box | Score breakdown per component |
| Candidate feedback | None | Personalised learning path per gap |
| Name verification | Not performed | Cross-checked across 3 sources |

---

## Supported Roles

ML Engineer · Data Scientist · NLP Engineer · Backend Engineer · Full Stack Developer · Data Engineer · Computer Vision Engineer · DevOps Engineer

---

## Graceful Degradation

The system runs at any level of dependency:

| Missing | What happens |
|---------|-------------|
| `saved_models/` | TF-IDF fallback — all features still work |
| PyMuPDF | Text paste instead of PDF upload |
| GitHub username | Manual text used |
| plotly | Metrics shown as text |
| pandas | Evidence matrix as plain rows |

---

## Tech Stack

`Python 3.10+` · `Streamlit` · `sentence-transformers` · `scikit-learn` · `PyTorch` · `PyMuPDF` · `Plotly` · `spaCy` · `urllib` (stdlib) · `GitHub REST API v3`

**No external AI APIs.** No OpenAI, Gemini, HuggingFace Inference, or any paid service.

---

## Hackathon — Problem Statement 4 Compliance

| Requirement | Status |
|-------------|--------|
| Parse resumes (PDF) | ✅ PyMuPDF |
| Parse GitHub profiles | ✅ Live REST API |
| Parse portfolios | ✅ Text extraction |
| Structured data extraction | ✅ Skills, experience, companies, metrics |
| Domain expertise identification | ✅ 12-domain depth scoring |
| Semantic matching (NLP/BERT) | ✅ Fine-tuned MiniLM + TF-IDF fallback |
| Knowledge graph | ✅ 12 domains + parent inference |
| Fit score 0–100 | ✅ Weighted, explainable formula |
| Skill gap identification | ✅ Required + preferred, exact list |
| Improvement recommendations | ✅ Step-by-step learning path |
| Own transformer-based model | ✅ Fine-tuned weights in saved_models/ |
| Own weights | ✅ Produced by training notebook |
| Explainable ranking | ✅ Per-component score breakdown |
| Candidate dashboard | ✅ 5-tab Streamlit UI |
| Recruiter dashboard | ✅ Separate view + evidence matrix |
| No external AI APIs | ✅ 100% local |

---

<p align="center">
  Built for Hackathon — Problem Statement 4: Multimodal Talent Intelligence System<br>
  Own architecture · Own weights · Own dataset · No external AI APIs · 100% local
</p>
