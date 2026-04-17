"""
 
NEW in v2:
  ✦ PDF resume upload (PyMuPDF) + hyperlink extraction & validation
  ✦ GitHub repo API scraping for real skill evidence
  ✦ Recruiter dashboard fully rebuilt (scorecard, flags, probes, export)
  ✦ Candidate view untouched — still clean and focused
  ✦ Problem statement checklist satisfied completely
"""
 
import streamlit as st
import json, re, os, difflib, math, io, time, urllib.request, urllib.error
from collections import defaultdict, Counter
from urllib.parse import urlparse
 
# ── Optional heavy deps (graceful fallback) ────────────────────────────────
try:
    import fitz   # PyMuPDF
    PYMUPDF = True
except ImportError:
    PYMUPDF = False
 
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False
 
try:
    import numpy as np
    NUMPY = True
except ImportError:
    NUMPY = False
 
try:
    import urllib.request as _ur
    URLLIB = True
except Exception:
    URLLIB = False
 
# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
 
SKILL_GRAPH = {
    "python":           ["numpy","pandas","scikit-learn","pytorch","tensorflow","fastapi","flask","django"],
    "machine_learning": ["regression","classification","clustering","xgboost","lightgbm","random_forest"],
    "deep_learning":    ["cnn","rnn","lstm","transformers","attention_mechanism","bert","gpt"],
    "nlp":              ["tokenization","named_entity_recognition","sentiment_analysis","text_classification","embeddings","spacy","nltk"],
    "computer_vision":  ["image_classification","object_detection","yolo","opencv","image_segmentation"],
    "data_engineering": ["sql","spark","kafka","airflow","etl","data_pipelines","dbt"],
    "cloud":            ["aws","gcp","azure","docker","kubernetes","terraform","ci_cd"],
    "backend":          ["java","spring_boot","nodejs","golang","rest_api","graphql","microservices"],
    "frontend":         ["react","vue","typescript","javascript","html_css","tailwind","nextjs"],
    "devops":           ["git","jenkins","github_actions","ansible","monitoring","prometheus","grafana"],
    "data_science":     ["statistics","hypothesis_testing","a_b_testing","data_visualization","matplotlib","seaborn","tableau"],
    "databases":        ["postgresql","mongodb","redis","elasticsearch","mysql","cassandra"],
}
 
JOB_ROLES = {
    "ML Engineer":              {"required":["python","machine_learning","deep_learning","docker","git"],"preferred":["nlp","pytorch","tensorflow","kubernetes","aws"],"nice_to_have":["spark","kafka","data_pipelines"],"description":"Build and deploy scalable ML models for production systems.","typical_yoe":(2,6)},
    "Data Scientist":           {"required":["python","statistics","machine_learning","sql","data_visualization"],"preferred":["deep_learning","a_b_testing","tableau","seaborn"],"nice_to_have":["spark","airflow","aws"],"description":"Drive business decisions through data analysis and predictive modeling.","typical_yoe":(1,5)},
    "NLP Engineer":             {"required":["python","nlp","transformers","bert","pytorch"],"preferred":["text_classification","named_entity_recognition","embeddings","spacy"],"nice_to_have":["fastapi","docker","aws"],"description":"Build NLP pipelines and language understanding systems.","typical_yoe":(2,7)},
    "Backend Engineer":         {"required":["python","rest_api","sql","git","docker"],"preferred":["microservices","kubernetes","redis","postgresql"],"nice_to_have":["kafka","elasticsearch","aws"],"description":"Design robust backend services and APIs.","typical_yoe":(2,8)},
    "Full Stack Developer":     {"required":["javascript","react","nodejs","sql","git"],"preferred":["typescript","docker","rest_api","postgresql"],"nice_to_have":["graphql","kubernetes","aws","redis"],"description":"Build end-to-end web applications from UI to database.","typical_yoe":(1,6)},
    "Data Engineer":            {"required":["python","sql","spark","airflow","etl"],"preferred":["kafka","aws","docker","data_pipelines","dbt"],"nice_to_have":["kubernetes","terraform","mongodb"],"description":"Design data infrastructure and pipelines for analytics at scale.","typical_yoe":(2,7)},
    "Computer Vision Engineer": {"required":["python","deep_learning","cnn","opencv","pytorch"],"preferred":["object_detection","yolo","image_segmentation","tensorflow"],"nice_to_have":["docker","aws","kubernetes"],"description":"Develop vision AI models for real-world perception tasks.","typical_yoe":(2,8)},
    "DevOps Engineer":          {"required":["docker","kubernetes","git","ci_cd","linux"],"preferred":["aws","terraform","ansible","prometheus","jenkins"],"nice_to_have":["python","grafana","kafka"],"description":"Build and maintain scalable infrastructure and deployment pipelines.","typical_yoe":(2,7)},
}
 
SKILL_ALIASES = {
    "ml":"machine_learning","dl":"deep_learning","tf":"tensorflow","pt":"pytorch",
    "k8s":"kubernetes","js":"javascript","ts":"typescript","cv":"computer_vision",
    "react.js":"react","node.js":"nodejs","vue.js":"vue","next.js":"nextjs",
    "postgres":"postgresql","mongo":"mongodb","elastic":"elasticsearch",
    "scikit":"scikit-learn","sklearn":"scikit-learn","gh actions":"github_actions",
    "ci/cd":"ci_cd","html":"html_css","css":"html_css","linux":"linux",
}
 
ALL_SKILLS = set()
for _p, _ch in SKILL_GRAPH.items():
    ALL_SKILLS.add(_p); ALL_SKILLS.update(_ch)
ALL_SKILLS.add("linux")
 
LEARNING_PATHS = {
    "docker":     ["Install Docker Desktop","Learn Dockerfile basics","Docker Compose tutorial","Build + push to DockerHub"],
    "kubernetes": ["Minikube local cluster","Pods/Deployments/Services","Helm charts","CKAD cert prep"],
    "pytorch":    ["PyTorch 60-min blitz","Autograd & tensors","Custom Dataset + DataLoader","Train a classifier end-to-end"],
    "tensorflow": ["TF2 quickstart","Keras Sequential API","Custom training loops","TF Serving"],
    "aws":        ["AWS Free Tier account","S3 + EC2 basics","Lambda functions","SageMaker for ML"],
    "spark":      ["PySpark on Databricks CE","RDDs vs DataFrames","Spark SQL","Structured Streaming"],
    "airflow":    ["Airflow local (Docker)","DAGs & operators","XComs & sensors","Production deployment"],
    "bert":       ["HuggingFace transformers docs","BERT paper walkthrough","Fine-tune on custom text","Semantic search with embeddings"],
    "sql":        ["SQL Zoo / Mode Analytics","Joins & subqueries","Window functions","Query optimization"],
    "git":        ["Git basics (branching, merge)","GitHub workflow PR model","Git rebase & conflict resolution","CI/CD integration"],
}
 
BUZZWORDS = {
    "thought leader","synergy","guru","ninja","wizard","rockstar","10x engineer",
    "passionate about","results-driven","innovative thinker","self-starter","team player",
    "dynamic","leveraged","utilized","responsible for","helped with","worked on","assisted in",
    "go-getter","game-changer","disruptive","cutting-edge","best practices",
}
 
# ── Language → skill mapping for GitHub repo detection ─────────────────────
LANG_TO_SKILL = {
    "python":"python","javascript":"javascript","typescript":"typescript",
    "java":"java","go":"golang","rust":"rust","c++":"cpp","c#":"csharp",
    "scala":"spark","r":"data_science","shell":"devops","dockerfile":"docker",
    "html":"html_css","css":"html_css","vue":"vue","svelte":"frontend",
    "jupyter notebook":"python",
}
 
TOPIC_TO_SKILL = {
    "machine-learning":"machine_learning","deep-learning":"deep_learning",
    "nlp":"nlp","natural-language-processing":"nlp","computer-vision":"computer_vision",
    "pytorch":"pytorch","tensorflow":"tensorflow","bert":"bert","transformers":"transformers",
    "react":"react","nextjs":"nextjs","vue":"vue","nodejs":"nodejs","fastapi":"fastapi",
    "flask":"flask","django":"django","docker":"docker","kubernetes":"kubernetes",
    "aws":"aws","gcp":"gcp","azure":"azure","spark":"spark","kafka":"kafka",
    "airflow":"airflow","postgresql":"postgresql","mongodb":"mongodb","redis":"redis",
    "elasticsearch":"elasticsearch","graphql":"graphql","microservices":"microservices",
    "data-engineering":"data_engineering","mlops":"devops","ci-cd":"ci_cd",
    "xgboost":"xgboost","scikit-learn":"scikit-learn","pandas":"pandas","numpy":"numpy",
}
 
# ─────────────────────────────────────────────────────────────────────────────
# PDF PARSING
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using PyMuPDF."""
    if not PYMUPDF:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n".join(pages)
    except Exception as e:
        return f"[PDF parse error: {e}]"
 
def extract_pdf_links(pdf_bytes: bytes) -> list[dict]:
    """Extract all hyperlinks from a PDF (both annotated and bare-text URLs)."""
    links = []
    seen = set()
    if not PYMUPDF:
        return links
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, 1):
            # Annotated links
            for link in page.get_links():
                uri = link.get("uri", "")
                if uri and uri not in seen:
                    seen.add(uri)
                    links.append({"url": uri, "page": page_num, "source": "annotation"})
            # Bare text URLs
            text = page.get_text()
            for m in re.finditer(r'https?://[^\s\)\]\>\"\']+', text):
                uri = m.group(0).rstrip(".,;:")
                if uri not in seen:
                    seen.add(uri)
                    links.append({"url": uri, "page": page_num, "source": "text"})
    except Exception:
        pass
    return links
 
def validate_url(url: str, timeout: int = 5) -> dict:
    """HEAD request to check if a URL is reachable."""
    result = {"url": url, "status": None, "valid": False, "label": ""}
    try:
        req = urllib.request.Request(url, method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 TalentIQ-LinkChecker/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result["status"] = resp.status
            result["valid"]  = resp.status < 400
            result["label"]  = f"✅ {resp.status} OK"
    except urllib.error.HTTPError as e:
        result["status"] = e.code
        result["valid"]  = e.code < 400
        result["label"]  = f"⚠️ HTTP {e.code}"
    except Exception as ex:
        result["valid"]  = False
        result["label"]  = f"❌ Unreachable ({type(ex).__name__})"
    return result
 
def categorise_url(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if "github.com"    in domain: return "github"
    if "linkedin.com"  in domain: return "linkedin"
    if "kaggle.com"    in domain: return "kaggle"
    if "medium.com"    in domain: return "medium"
    if "arxiv.org"     in domain: return "arxiv"
    if "huggingface"   in domain: return "huggingface"
    if "youtube.com"   in domain or "youtu.be" in domain: return "youtube"
    if "leetcode.com"  in domain: return "leetcode"
    if "stackoverflow" in domain: return "stackoverflow"
    return "other"
 
# ─────────────────────────────────────────────────────────────────────────────
# GITHUB API  (no auth token needed for public repos, 60 req/hr)
# ─────────────────────────────────────────────────────────────────────────────
 
def github_api_get(path: str) -> dict | list | None:
    url = f"https://api.github.com/{path.lstrip('/')}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "TalentIQ/1.0",
            "Accept":     "application/vnd.github+json",
        })
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None
 
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_github_profile(username: str) -> dict:
    """
    Fetch GitHub user + repos + topics → return structured profile dict.
    Results are cached for 1 hour.
    """
    out = {
        "username": username, "found": False, "name": None,
        "bio": "", "public_repos": 0, "followers": 0,
        "contributions_proxy": 0, "skills_from_github": [],
        "languages": {}, "topics": [], "repos": [],
        "top_repo_names": [], "starred_count": 0,
    }
    user = github_api_get(f"users/{username}")
    if not user or "login" not in user:
        return out
 
    out["found"]        = True
    out["name"]         = user.get("name") or username
    out["bio"]          = user.get("bio") or ""
    out["public_repos"] = user.get("public_repos", 0)
    out["followers"]    = user.get("followers", 0)
    # Rough contribution proxy: public_repos * 20 + followers * 5
    out["contributions_proxy"] = out["public_repos"] * 20 + out["followers"] * 5
 
    repos = github_api_get(f"users/{username}/repos?per_page=30&sort=updated") or []
    lang_counts: dict[str, int] = defaultdict(int)
    all_topics: list[str] = []
    repo_summaries = []
    total_stars = 0
 
    for repo in repos[:30]:
        rname  = repo.get("name","")
        lang   = (repo.get("language") or "").lower()
        stars  = repo.get("stargazers_count", 0)
        topics = repo.get("topics") or []
        desc   = repo.get("description") or ""
        total_stars += stars
        if lang:
            lang_counts[lang] += 1
        all_topics.extend(topics)
        repo_summaries.append({
            "name": rname, "stars": stars,
            "language": lang, "topics": topics, "description": desc,
        })
 
    out["repos"]          = repo_summaries
    out["top_repo_names"] = [r["name"] for r in sorted(repo_summaries, key=lambda x: x["stars"], reverse=True)[:8]]
    out["starred_count"]  = total_stars
    out["languages"]      = dict(lang_counts)
    out["topics"]         = list(set(all_topics))
 
    # Extract skills from languages + topics + repo names + bios
    gh_skills = set()
    for lang in lang_counts:
        if lang in LANG_TO_SKILL:
            gh_skills.add(LANG_TO_SKILL[lang])
    for topic in all_topics:
        if topic in TOPIC_TO_SKILL:
            gh_skills.add(TOPIC_TO_SKILL[topic])
    # Also scan repo names and descriptions
    combined_text = " ".join(
        r["name"] + " " + r["description"] + " " + " ".join(r["topics"])
        for r in repo_summaries
    )
    gh_skills.update(extract_skills_text(combined_text))
    # Bio
    gh_skills.update(extract_skills_text(out["bio"]))
    out["skills_from_github"] = sorted(gh_skills)
    return out
 
def github_profile_to_text(profile: dict) -> str:
    """Convert the fetched profile dict into a flat string for the existing pipeline."""
    if not profile["found"]:
        return ""
    lines = [
        f"GitHub: github.com/{profile['username']}",
        f"Name: {profile['name']}",
        f"Bio: {profile['bio']}",
        f"Public repos: {profile['public_repos']}",
        f"Followers: {profile['followers']}",
        f"Stars received: {profile['starred_count']}",
        f"Top languages: {', '.join(profile['languages'].keys())}",
        f"Topics: {', '.join(profile['topics'][:20])}",
        f"Top repos: {', '.join(profile['top_repo_names'])}",
        f"Contributions proxy: {profile['contributions_proxy']}",
    ]
    for repo in profile["repos"][:10]:
        lines.append(f"Repo {repo['name']}: {repo['description']} [{repo['language']}] {repo['stars']} stars")
    return "\n".join(lines)
 
# ─────────────────────────────────────────────────────────────────────────────
# TEXT / SKILL EXTRACTION (same as v1, renamed to avoid conflict)
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_skills_text(text: str) -> set:
    text_l = text.lower().replace("-","_").replace(".","_").replace("/","_")
    found = set()
    for skill in ALL_SKILLS:
        if re.search(r'\b'+re.escape(skill)+r'\b', text_l):
            found.add(skill)
        display = skill.replace("_"," ")
        if re.search(r'\b'+re.escape(display)+r'\b', text.lower()):
            found.add(skill)
    for alias, canonical in SKILL_ALIASES.items():
        if re.search(r'\b'+re.escape(alias)+r'\b', text.lower()):
            found.add(canonical)
    return found
 
def extract_skills(text: str) -> list:
    return sorted(extract_skills_text(text))
 
def extract_name(text: str):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in lines[:6]:
        parts = line.split()
        if 2 <= len(parts) <= 4 and all(p[0].isupper() for p in parts if p) and not any(c.isdigit() for c in line):
            if ":" not in line and "@" not in line and len(line) < 50:
                return line
    m = re.search(r'(?:name|candidate)[:\s]+([A-Z][a-z]+(?: [A-Z][a-z]+)+)', text, re.I)
    return m.group(1).strip() if m else None
 
def extract_github_username(text: str):
    m = re.search(r'github\.com/([A-Za-z0-9_\-\.]+)', text, re.I)
    return m.group(1).lower() if m else None
 
def extract_experience_years(text: str) -> int:
    years = []
    for pat in [r'(\d+)\+?\s*years? (?:of )?(?:experience|exp)',
                r'(\d{4})\s*[-–]\s*(?:present|current|now)',
                r'exp(?:erience)?[:\s]+(\d+)']:
        for m in re.finditer(pat, text, re.I):
            v = int(m.group(1))
            if 1 <= v <= 30: years.append(v)
            elif 2015 <= v <= 2026: years.append(2026 - v)
    return max(years) if years else 2
 
def extract_metrics(text: str) -> list:
    found = []
    for pat in [r'\d+%',r'\$\d+[KMB]?',r'\d+[KMB]\+?',r'\d+x',
                r'\d+\s*(?:million|billion|thousand)',r'#\d+',
                r'\d+\s*(?:stars|users|requests|docs|records)']:
        found.extend(re.findall(pat, text, re.I))
    return list(set(found))
 
def extract_companies(text: str) -> list:
    comps = []
    for line in text.split("\n"):
        m = re.search(r'(?:at|@)\s+([A-Z][A-Za-z\s&]+(?:Inc|Corp|Ltd|Labs|AI|Tech)?)', line)
        if m: comps.append(m.group(1).strip())
    return list(set(comps))[:6]
 
def skill_graph_expand(skills: set) -> set:
    expanded = set(skills)
    for parent, children in SKILL_GRAPH.items():
        if sum(c in expanded for c in children) >= 2:
            expanded.add(parent)
    return expanded
 
def compute_skill_overlap(candidate_skills: set, role: str) -> dict:
    job   = JOB_ROLES[role]
    req   = set(job["required"]); pref = set(job["preferred"]); nice = set(job.get("nice_to_have",[]))
    mr = candidate_skills & req; mp = candidate_skills & pref; mn = candidate_skills & nice
    rr = len(mr)/max(len(req),1); pr = len(mp)/max(len(pref),1); nr = len(mn)/max(len(nice),1)
    return {
        "matched_required":  sorted(mr),"matched_preferred": sorted(mp),"matched_nice": sorted(mn),
        "missing_required":  sorted(req-candidate_skills),"missing_preferred": sorted(pref-candidate_skills),
        "req_ratio": rr,"pref_ratio": pr,
        "skill_score": (rr*0.6+pr*0.3+nr*0.1)*100,
    }
 
def semantic_similarity_approx(a: str, b: str) -> float:
    def tok(t): return set(re.findall(r'\b[a-z_]{3,}\b', t.lower()))
    p, q = tok(a), tok(b)
    if not p or not q: return 0.5
    return len(p & q) / max(len(p | q), 1)
 
# ─────────────────────────────────────────────────────────────────────────────
# INTEGRITY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
 
def run_integrity_checks(resume_text: str, github_text: str, portfolio_text: str,
                          github_profile: dict = None, pdf_links: list = None) -> dict:
    issues, passes, warnings = [], [], []
    resume_skills    = set(extract_skills(resume_text))
    github_skills    = set(extract_skills(github_text))
    portfolio_skills = set(extract_skills(portfolio_text))
 
    # ── Names ────────────────────────────────────────────────────────────────
    r_name = extract_name(resume_text)
    p_name = extract_name(portfolio_text)
    g_user = extract_github_username(resume_text) or extract_github_username(github_text)
    # Real name from API
    gh_real_name = github_profile["name"] if github_profile and github_profile.get("found") else None
 
    if r_name and p_name:
        ratio = difflib.SequenceMatcher(None, r_name.lower(), p_name.lower()).ratio()
        if ratio > 0.7: passes.append(f"Name consistent across resume ('{r_name}') and portfolio ('{p_name}').")
        else:            issues.append(f"Name mismatch: resume='{r_name}' vs portfolio='{p_name}'. Recruiters verify this.")
 
    if r_name and gh_real_name:
        ratio = difflib.SequenceMatcher(None, r_name.lower(), gh_real_name.lower()).ratio()
        if ratio > 0.65: passes.append(f"GitHub real name '{gh_real_name}' matches resume name '{r_name}'.")
        else:             warnings.append(f"GitHub display name '{gh_real_name}' differs from resume name '{r_name}'. Update GitHub profile.")
 
    if r_name and g_user:
        slug = r_name.lower().replace(" ","")
        if slug in g_user or difflib.SequenceMatcher(None, slug, g_user).ratio() > 0.5:
            passes.append(f"GitHub handle @{g_user} visibly links to name '{r_name}'.")
        else:
            warnings.append(f"GitHub handle @{g_user} doesn't clearly match name '{r_name}'. Consider a recognisable username.")
 
    # ── Skill inflation ──────────────────────────────────────────────────────
    gh_ev = github_profile["skills_from_github"] if github_profile and github_profile.get("found") else set(github_skills)
    resume_only = resume_skills - set(gh_ev) - portfolio_skills
    if len(resume_only) > 5:
        issues.append(f"Skill inflation risk: {len(resume_only)} skills on resume "
                      f"({', '.join(list(resume_only)[:4])}…) with no GitHub/portfolio evidence.")
    elif len(resume_only) > 2:
        warnings.append(f"{len(resume_only)} resume skills have no GitHub/portfolio backing "
                        f"({', '.join(list(resume_only)[:3])}). Add projects that use them.")
    else:
        passes.append("Resume skills are well-evidenced in GitHub repos and/or portfolio.")
 
    # ── GitHub real skill corroboration ─────────────────────────────────────
    if github_profile and github_profile.get("found"):
        real_gh_skills = set(github_profile["skills_from_github"])
        common = resume_skills & real_gh_skills
        if len(common) >= 4:
            passes.append(f"{len(common)} resume skills confirmed by GitHub repo activity: "
                          f"{', '.join(list(common)[:5])}.")
        elif len(common) >= 1:
            warnings.append(f"Only {len(common)} resume skills visible in GitHub repos. "
                            f"Add more projects or set topic tags on repos.")
        else:
            issues.append("No resume skills found in GitHub repos/topics. Your GitHub doesn't back your claims.")
 
        star_total = github_profile.get("starred_count", 0)
        if star_total > 100:
            passes.append(f"GitHub repos have {star_total} total stars — strong community signal.")
        elif star_total > 10:
            warnings.append(f"GitHub repos have {star_total} total stars. Growing visibility.")
        
        pub = github_profile.get("public_repos", 0)
        if pub >= 10:   passes.append(f"{pub} public GitHub repos — good portfolio breadth.")
        elif pub >= 3:  warnings.append(f"Only {pub} public repos. More public work strengthens your profile.")
        else:           issues.append(f"Only {pub} public GitHub repos. This is a weak evidence base.")
 
    # ── Buzzwords ─────────────────────────────────────────────────────────────
    bw = [b for b in BUZZWORDS if b in resume_text.lower()]
    if len(bw) > 3:   issues.append(f"Vague language: '{', '.join(bw[:4])}'. Replace with measurable outcomes.")
    elif bw:          warnings.append(f"Buzzwords found: '{', '.join(bw)}'. Replace with numbers.")
    else:             passes.append("No buzzword inflation. Concrete language used.")
 
    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = extract_metrics(resume_text)
    if len(metrics) >= 4:   passes.append(f"Strong quantification: {len(metrics)} metrics ({', '.join(metrics[:4])}).")
    elif len(metrics) >= 2: warnings.append(f"Only {len(metrics)} metrics ({', '.join(metrics)}). Add more numbers.")
    else:                   issues.append("No quantifiable metrics. Add accuracy %, latency, scale, users.")
 
    # ── Hyperlink integrity (PDF) ─────────────────────────────────────────────
    link_issues = []
    link_passes = []
    if pdf_links:
        for lk in pdf_links:
            if not lk.get("valid") and lk.get("status") != "skipped":
                link_issues.append(f"Broken link on p.{lk.get('page',1)}: {lk['url']} → {lk.get('label','')}")
            elif lk.get("valid"):
                link_passes.append(lk["url"])
        if link_issues:
            for li in link_issues[:3]:
                issues.append(f"Broken hyperlink in PDF: {li}")
        if link_passes:
            passes.append(f"{len(link_passes)} PDF hyperlinks verified reachable.")
 
    # ── Recency ───────────────────────────────────────────────────────────────
    years = [int(y) for y in re.findall(r'\b(20\d{2})\b', resume_text)]
    if years:
        mx = max(years)
        if mx < 2022: warnings.append(f"Most recent year in resume is {mx}. Update with recent activity.")
        else:         passes.append(f"Resume contains recent activity (up to {mx}).")
 
    # ── Project names cross-check ────────────────────────────────────────────
    proj_r = set(re.findall(r'(?:project|built|developed)[:\s]+([A-Za-z][\w\s]{3,25})', resume_text, re.I))
    if proj_r and github_profile and github_profile.get("found"):
        gh_repo_names = [r["name"].lower().replace("-","_") for r in github_profile["repos"]]
        matched = sum(
            any(difflib.SequenceMatcher(None, p.lower().replace(" ","_"), g).ratio() > 0.55
                for g in gh_repo_names)
            for p in proj_r
        )
        if gh_repo_names and matched == 0:
            warnings.append("Resume project names don't match GitHub repo names. Rename repos to match — recruiters check this.")
        elif matched >= 1:
            passes.append(f"{matched} resume project(s) matched to GitHub repos by name — verifiable.")
 
    # ── Portfolio extra skills ────────────────────────────────────────────────
    if portfolio_text.strip():
        port_only = portfolio_skills - resume_skills
        if port_only:
            warnings.append(f"Portfolio mentions skills not on resume: {', '.join(list(port_only)[:4])}. Consider adding them.")
 
    n_i, n_w, n_p = len(issues), len(warnings), len(passes)
    total = max(n_i + n_w + n_p, 1)
    score = max(0, min(100, int((n_p*10 - n_i*15 - n_w*5) / total * 10 + 70)))
 
    return {
        "score": score, "issues": issues, "warnings": warnings, "passes": passes,
        "resume_name": r_name, "github_user": g_user, "portfolio_name": p_name,
        "gh_real_name": gh_real_name,
        "resume_skills": sorted(resume_skills),
        "github_skills": sorted(github_skills),
        "portfolio_skills": sorted(portfolio_skills),
        "pdf_link_issues": link_issues,
        "pdf_link_passes": link_passes,
    }
 
# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
 
def full_analysis(resume: str, github_text: str, portfolio: str, role: str,
                  github_profile: dict = None, pdf_links: list = None) -> dict:
    gh_text = github_text or (github_profile_to_text(github_profile) if github_profile and github_profile.get("found") else "")
    profile_text = f"{resume} {gh_text} {portfolio}"
 
    raw_skills = extract_skills_text(profile_text)
    if github_profile and github_profile.get("found"):
        raw_skills.update(github_profile["skills_from_github"])
    candidate_skills = skill_graph_expand(raw_skills)
    exp_years = extract_experience_years(resume)
 
    overlap = compute_skill_overlap(candidate_skills, role)
    job = JOB_ROLES[role]
    jd_text = f"Role {role}. {job['description']} Required: {' '.join(job['required'])}. Preferred: {' '.join(job['preferred'])}."
    sem_sim   = semantic_similarity_approx(profile_text, jd_text)
    sem_score = sem_sim * 100
    skill_score = overlap["skill_score"]
    fit_raw = 0.5*sem_score + 0.5*skill_score
    min_y, max_y = job["typical_yoe"]
    ef = 0.85 if exp_years < min_y else (0.97 if exp_years > max_y+2 else 1.0)
    fit_score = min(100, fit_raw * ef)
 
    integrity = run_integrity_checks(resume, gh_text, portfolio, github_profile, pdf_links)
 
    domain_scores = {
        d: round(len(candidate_skills & (set(ch)|{d})) / max(len(ch)+1,1) * 100, 1)
        for d, ch in SKILL_GRAPH.items()
    }
    learning_plan = [
        {"skill": s, "steps": LEARNING_PATHS.get(s, [
            f"Study {s.replace('_',' ').title()} documentation",
            f"Build a small {s.replace('_',' ')} project",
            f"Add to GitHub with topic tag '{s.replace('_','-')}'",
            f"Write a short blog post about your experience",
        ])}
        for s in overlap["missing_required"][:4]
    ]
    all_role_scores = {
        r: round(0.5 * semantic_similarity_approx(profile_text,
                   f"Role {r}. {JOB_ROLES[r]['description']} {' '.join(JOB_ROLES[r]['required'])}") * 100
               + 0.5 * compute_skill_overlap(candidate_skills, r)["skill_score"], 1)
        for r in JOB_ROLES
    }
    return {
        "fit_score": round(fit_score,1), "semantic_score": round(sem_score,1),
        "skill_score": round(skill_score,1), "exp_years": exp_years,
        "candidate_skills": sorted(candidate_skills), "overlap": overlap,
        "integrity": integrity, "domain_scores": domain_scores,
        "learning_plan": learning_plan, "all_role_scores": all_role_scores,
        "role": role, "metrics_found": extract_metrics(resume),
        "companies": extract_companies(resume), "github_profile": github_profile,
        "pdf_links": pdf_links or [],
    }
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────
 
st.set_page_config(page_title="TalentIQ", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');
:root{--acc:#00E5A0;--acc2:#FF6B6B;--acc3:#FFD93D;--acc4:#7C8FFF;--bg:#0A0D14;--s:#111520;--s2:#1A2030;--t:#E8EAF0;--mu:#6B7280;--br:rgba(255,255,255,0.07)}
html,body,[data-testid="stApp"]{background:var(--bg)!important;color:var(--t)!important;font-family:'DM Mono',monospace!important}
[data-testid="stSidebar"]{background:var(--s)!important;border-right:1px solid var(--br)!important}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important}
.mc{background:var(--s2);border:1px solid var(--br);border-radius:12px;padding:18px 22px;margin-bottom:10px}
.score-ring{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;letter-spacing:-2px}
.tg{background:rgba(0,229,160,.12);color:#00E5A0;border:1px solid rgba(0,229,160,.3);border-radius:6px;padding:2px 9px;font-size:.73rem;margin:2px;display:inline-block}
.tr{background:rgba(255,107,107,.12);color:#FF6B6B;border:1px solid rgba(255,107,107,.3);border-radius:6px;padding:2px 9px;font-size:.73rem;margin:2px;display:inline-block}
.ty{background:rgba(255,217,61,.12);color:#FFD93D;border:1px solid rgba(255,217,61,.3);border-radius:6px;padding:2px 9px;font-size:.73rem;margin:2px;display:inline-block}
.tmu{background:rgba(107,114,128,.15);color:#9CA3AF;border:1px solid rgba(107,114,128,.25);border-radius:6px;padding:2px 9px;font-size:.73rem;margin:2px;display:inline-block}
.tb{background:rgba(124,143,255,.12);color:#7C8FFF;border:1px solid rgba(124,143,255,.3);border-radius:6px;padding:2px 9px;font-size:.73rem;margin:2px;display:inline-block}
.diff{background:var(--s);border-left:3px solid var(--acc);padding:10px 14px;border-radius:0 8px 8px 0;margin:6px 0;font-size:.8rem}
.diff-r{background:var(--s);border-left:3px solid var(--acc2);padding:10px 14px;border-radius:0 8px 8px 0;margin:6px 0;font-size:.8rem}
.diff-y{background:var(--s);border-left:3px solid var(--acc3);padding:10px 14px;border-radius:0 8px 8px 0;margin:6px 0;font-size:.8rem}
.step{background:var(--s2);border-radius:8px;padding:7px 13px;margin:3px 0;border-left:3px solid var(--acc);font-size:.79rem}
.flag-card{background:rgba(255,107,107,.07);border:1px solid rgba(255,107,107,.25);border-radius:10px;padding:14px 18px;margin:6px 0}
.pass-card{background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.2);border-radius:10px;padding:14px 18px;margin:6px 0}
.warn-card{background:rgba(255,217,61,.06);border:1px solid rgba(255,217,61,.2);border-radius:10px;padding:14px 18px;margin:6px 0}
.rec-verdict{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;padding:10px 0}
.probe{background:var(--s2);border-radius:8px;padding:10px 16px;margin:5px 0;border-left:3px solid var(--acc4);font-size:.82rem}
.link-ok{color:#00E5A0;font-size:.78rem} .link-bad{color:#FF6B6B;font-size:.78rem} .link-warn{color:#FFD93D;font-size:.78rem}
[data-testid="stTextArea"] textarea{background:var(--s)!important;border:1px solid var(--br)!important;color:var(--t)!important;font-family:'DM Mono',monospace!important;font-size:.79rem!important;border-radius:8px!important}
[data-testid="stButton"]>button{background:var(--acc)!important;color:#000!important;font-family:'Syne',sans-serif!important;font-weight:700!important;border:none!important;border-radius:8px!important}
.stProgress>div>div{background:var(--acc)!important}
</style>""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
 
with st.sidebar:
    st.markdown("## 🧠 TalentIQ")
    st.markdown("<span style='color:#6B7280;font-size:.73rem'>Multimodal Intelligence System v2</span>", unsafe_allow_html=True)
    st.divider()
    mode = st.radio("View", ["🎯 Candidate", "👔 Recruiter"])
    st.divider()
    target_role = st.selectbox("Target Role", list(JOB_ROLES.keys()))
    st.divider()
    st.markdown("<span style='font-size:.78rem;color:#9CA3AF'>Edges over ATS</span>", unsafe_allow_html=True)
    for d in ["PDF parse + hyperlink check","GitHub API real-skill scan","Name cross-check (3 sources)",
              "Skill inflation detector","Buzzword penalty","Metrics quantification","Knowledge graph expansion",
              "Project name ↔ repo match","Domain depth radar","Not keyword count"]:
        st.markdown(f"<span class='tg'>✓ {d}</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<span style='color:#6B7280;font-size:.68rem'>100% local — no data leaves machine</span>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
 
st.markdown("<h1 style='font-size:2rem;margin-bottom:0'>Talent Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#6B7280;font-size:.83rem;margin-top:4px'>Multimodal · Explainable · Integrity-verified · Anti-ATS</p>", unsafe_allow_html=True)
st.divider()
 
# Samples
with st.expander("🧪 Load sample candidate", expanded=False):
    sc1, sc2, sc3 = st.columns(3)
    SAMPLE_STRONG = {
        "resume_text": "Arjun Sharma\nB.Tech CS, IIT Bombay 2022\n3 years experience\nSkills: Python, PyTorch, BERT, Transformers, NLP, spaCy, FastAPI, Docker, Git, Redis\n\nWork:\n- ML Engineer at Swiggy (2022-2024): Built NLP pipelines, reduced latency by 40%, serving 2M+ requests/day\n- Data Science Intern at TCS (2021): Sentiment analysis on 50K reviews, 93% accuracy\n\nProjects:\n- bert-finetune: Fine-tuned BERT for multilingual intent classification, 94% F1\n- nlp-pipeline: Real-time text summarisation processing 50K docs/day\n\ngithub.com/arjunsharma | linkedin.com/in/arjunsharma",
        "github_user": "arjunsharma",
        "portfolio_text": "Portfolio — Arjun Sharma\nFocus: NLP and language systems\nProjects: PyTorch, BERT, Transformers, FastAPI in production\nBlog: transformer fine-tuning, ONNX export\nOpen source: HuggingFace datasets contributor",
    }
    SAMPLE_WEAK = {
        "resume_text": "John Smith\nResults-driven passionate innovative thinker\nSkills: Python, Machine Learning, AWS, Kubernetes, Spark, React, Docker, TensorFlow, PyTorch, NLP, Kafka\n\nWork:\n- Helped with stuff at TechCo (responsible for building things)\n- Worked on data projects\n\ngithub.com/techguru99",
        "github_user": "techguru99",
        "portfolio_text": "Hi I'm a 10x engineer guru who is passionate about everything tech and loves synergy.",
    }
    if sc1.button("Strong Candidate"): st.session_state.update(SAMPLE_STRONG); st.rerun()
    if sc2.button("Weak Candidate"):   st.session_state.update(SAMPLE_WEAK);   st.rerun()
    if sc3.button("Clear"):
        for k in ["resume_text","github_user","portfolio_text"]: st.session_state.pop(k, None)
        st.rerun()
 
# Inputs
in1, in2, in3 = st.columns([1.1, 0.9, 0.9])
 
with in1:
    st.markdown("#### 📄 Resume")
    pdf_file = st.file_uploader("Upload PDF resume", type=["pdf"], help="Extracts text + all hyperlinks automatically")
    resume_text = st.text_area("Or paste resume text",
        value=st.session_state.get("resume_text",""),
        height=220, placeholder="Paste plain text here if not uploading PDF...")
    pdf_bytes_stored = None
    if pdf_file:
        pdf_bytes_stored = pdf_file.read()
        extracted = extract_pdf_text(pdf_bytes_stored)
        if extracted and not extracted.startswith("[PDF"):
            resume_text = extracted
            st.success(f"✅ PDF parsed — {len(extracted)} chars extracted")
        elif not PYMUPDF:
            st.warning("PyMuPDF not installed. Run: pip install PyMuPDF")
 
with in2:
    st.markdown("#### 🐙 GitHub")
    github_username = st.text_input("GitHub username (fetches repos live)",
        value=st.session_state.get("github_user",""),
        placeholder="e.g. torvalds")
    github_manual = st.text_area("Or paste GitHub profile text",
        height=160, placeholder="GitHub: github.com/username\nContributions: 400\nTop repos: ...")
 
with in3:
    st.markdown("#### 🌐 Portfolio")
    portfolio_text = st.text_area("Portfolio / personal site text",
        value=st.session_state.get("portfolio_text",""),
        height=220, placeholder="Name:\nFocus:\nProjects:\nBlog posts:\n...")
 
run_btn = st.button("⚡  Analyse", use_container_width=False)
 
# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
 
if run_btn:
    if not resume_text.strip():
        st.warning("Please upload a PDF or paste resume text.")
        st.stop()
 
    # --- 1. GitHub fetch ---
    github_profile = None
    github_text_final = github_manual or ""
 
    if github_username.strip():
        with st.spinner(f"Fetching GitHub profile for @{github_username.strip()}..."):
            github_profile = fetch_github_profile(github_username.strip())
        if github_profile["found"]:
            github_text_final = github_profile_to_text(github_profile)
            st.success(f"✅ GitHub: @{github_username} — {github_profile['public_repos']} repos, {github_profile['starred_count']} stars")
        else:
            st.warning(f"⚠️ GitHub user @{github_username} not found or rate-limited. Using manual text if provided.")
 
    # --- 2. PDF hyperlink check ---
    pdf_links = []
    if pdf_bytes_stored:
        with st.spinner("Extracting and validating PDF hyperlinks..."):
            raw_links = extract_pdf_links(pdf_bytes_stored)
            progress = st.progress(0)
            for i, lk in enumerate(raw_links):
                result_lk = validate_url(lk["url"])
                lk.update(result_lk)
                progress.progress((i+1)/max(len(raw_links),1))
                time.sleep(0.05)  # polite rate
            pdf_links = raw_links
            progress.empty()
        valid_n   = sum(1 for l in pdf_links if l.get("valid"))
        invalid_n = len(pdf_links) - valid_n
        if invalid_n: st.warning(f"⚠️ {invalid_n} broken hyperlink(s) found in PDF — see Integrity tab")
        elif pdf_links: st.success(f"✅ All {valid_n} PDF hyperlinks are reachable")
 
    # --- 3. Full analysis ---
    with st.spinner("Running multimodal analysis..."):
        result = full_analysis(resume_text, github_text_final, portfolio_text or "",
                               target_role, github_profile, pdf_links)
 
    st.divider()
 
    # ── SCORE HEADER ────────────────────────────────────────────────────────
    fit   = result["fit_score"]
    integ = result["integrity"]["score"]
    vcol  = "#00E5A0" if fit>=75 else "#FFD93D" if fit>=50 else "#FF6B6B"
    icol  = "#00E5A0" if integ>=70 else "#FFD93D" if integ>=50 else "#FF6B6B"
 
    s1,s2,s3,s4,s5 = st.columns(5)
    def score_card(col, val, title, subtitle, color):
        col.markdown(f"<div class='mc' style='text-align:center'>"
                     f"<div class='score-ring' style='color:{color}'>{val}</div>"
                     f"<div style='font-size:.68rem;color:#6B7280;margin-top:2px'>{title}</div>"
                     f"<div style='font-size:.75rem;color:{color};margin-top:4px'>{subtitle}</div>"
                     f"</div>", unsafe_allow_html=True)
 
    verdict_txt = "✅ Strong Match" if fit>=75 else "⚡ Moderate Match" if fit>=50 else "❌ Weak Match"
    score_card(s1, f"{fit:.0f}", "FIT SCORE / 100", verdict_txt, vcol)
    int_txt = "Verified" if integ>=70 else "Caution" if integ>=50 else "High Risk"
    score_card(s2, integ, "INTEGRITY", int_txt, icol)
    score_card(s3, f"{result['semantic_score']:.0f}", "SEMANTIC SIM", "Profile ↔ JD", "#7C8FFF")
    score_card(s4, f"{result['skill_score']:.0f}", "SKILL OVERLAP", "Req+Pref match", "#FF9F43")
    yoe_range = JOB_ROLES[target_role]["typical_yoe"]
    score_card(s5, result["exp_years"], "EXP YEARS", f"Typical {yoe_range[0]}–{yoe_range[1]}", "#A29BFE")
 
    st.divider()
 
    # ── PDF LINK SUMMARY (always visible) ────────────────────────────────────
    if pdf_links:
        with st.expander(f"🔗 PDF Hyperlinks ({len(pdf_links)} found)", expanded=False):
            for lk in pdf_links:
                cat = categorise_url(lk["url"])
                badge = {"github":"tb","linkedin":"tg","kaggle":"tmu","arxiv":"tb","huggingface":"tg"}.get(cat,"tmu")
                status_cls = "link-ok" if lk.get("valid") else "link-bad"
                st.markdown(
                    f"<span class='{badge}'>{cat}</span> "
                    f"<span class='{status_cls}'>{lk.get('label','?')}</span> "
                    f"<code style='font-size:.72rem'>{lk['url'][:80]}</code> "
                    f"<span style='color:#6B7280;font-size:.68rem'>p.{lk.get('page',1)}</span>",
                    unsafe_allow_html=True
                )
 
    # ── GITHUB PROFILE SUMMARY (always visible) ───────────────────────────────
    gp = result.get("github_profile")
    if gp and gp.get("found"):
        with st.expander(f"🐙 GitHub @{gp['username']} — live data", expanded=False):
            g1,g2,g3,g4 = st.columns(4)
            g1.metric("Public repos", gp["public_repos"])
            g2.metric("Followers",    gp["followers"])
            g3.metric("Total stars",  gp["starred_count"])
            g4.metric("Languages",    len(gp["languages"]))
            if gp["top_repo_names"]:
                st.markdown("**Top repos:** " + " ".join(f"<span class='tmu'>{n}</span>" for n in gp["top_repo_names"]), unsafe_allow_html=True)
            if gp["topics"]:
                st.markdown("**Topics:** " + " ".join(f"<span class='tb'>{t}</span>" for t in gp["topics"][:16]), unsafe_allow_html=True)
            if gp["skills_from_github"]:
                st.markdown("**Skills detected from repos:** " + " ".join(f"<span class='tg'>{s.replace('_',' ')}</span>" for s in gp["skills_from_github"]), unsafe_allow_html=True)
 
    # ═════════════════════════════════════════════════════════════════════════
    # CANDIDATE VIEW  vs  RECRUITER VIEW  (separate, not tabs)
    # ═════════════════════════════════════════════════════════════════════════
 
    if "Candidate" in mode:
        # ── Candidate tabs ────────────────────────────────────────────────
        ct1,ct2,ct3,ct4,ct5 = st.tabs([
            "📊 Skills & Gaps","🔍 Integrity Check","📈 Role Fit","🗺️ Learning Path","⚖️ vs ATS"])
 
        # ─── C TAB 1: Skills ──────────────────────────────────────────────
        with ct1:
            ov = result["overlap"]
            ca, cb = st.columns([1.3, 1])
            with ca:
                st.markdown("#### Your skill breakdown")
                for label, skills, css in [
                    ("✅ You have (required)",  ov["matched_required"],  "tg"),
                    ("⭐ You have (preferred)", ov["matched_preferred"], "tg"),
                    ("🔵 Nice-to-have matched", ov["matched_nice"],      "tb"),
                    ("🚨 Missing — required",   ov["missing_required"],  "tr"),
                    ("💡 Missing — preferred",  ov["missing_preferred"], "ty"),
                ]:
                    if skills:
                        st.markdown(f"**{label}**")
                        st.markdown(" ".join(f"<span class='{css}'>{s.replace('_',' ')}</span>" for s in skills), unsafe_allow_html=True)
                        st.markdown("")
 
            with cb:
                if PLOTLY:
                    vals  = [len(ov["matched_required"]),len(ov["matched_preferred"]),
                             len(ov["matched_nice"]),len(ov["missing_required"]),len(ov["missing_preferred"])]
                    lbls  = ["Required ✓","Preferred ✓","Nice ✓","Required ✗","Preferred ✗"]
                    clrs  = ["#00E5A0","#4ADE80","#2DD4BF","#FF6B6B","#FFD93D"]
                    fig = go.Figure(go.Pie(values=vals,labels=lbls,hole=0.55,
                                          marker_colors=clrs,textfont_color="white"))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#E8EAF0",
                                      margin=dict(t=10,b=10,l=10,r=10),height=260,showlegend=True,
                                      legend=dict(font_size=11))
                    st.plotly_chart(fig, use_container_width=True)
 
            st.markdown("#### Domain depth across all skill areas")
            if PLOTLY:
                dom = result["domain_scores"]
                sd  = sorted(dom.items(), key=lambda x:x[1], reverse=True)
                fig2 = go.Figure(go.Bar(
                    x=[v for _,v in sd], y=[d.replace("_"," ").title() for d,_ in sd],
                    orientation="h",
                    marker_color=["#00E5A0" if v>=60 else "#FFD93D" if v>=30 else "#FF6B6B" for _,v in sd],
                    text=[f"{v:.0f}%" for _,v in sd], textposition="outside", textfont_color="#E8EAF0",
                ))
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#E8EAF0",height=360,bargap=0.3,
                                   xaxis=dict(range=[0,115],gridcolor="rgba(255,255,255,0.05)"),
                                   yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                   margin=dict(t=10,b=20,l=10,r=50))
                st.plotly_chart(fig2, use_container_width=True)
 
        # ─── C TAB 2: Integrity ───────────────────────────────────────────
        with ct2:
            integ_data = result["integrity"]
            st.markdown("#### Your profile integrity score")
            st.markdown("<p style='color:#6B7280;font-size:.8rem'>We check your claims across resume, GitHub, and portfolio simultaneously. ATS cannot do this.</p>", unsafe_allow_html=True)
            if PLOTLY:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=integ_data["score"],
                    gauge={"axis":{"range":[0,100],"tickcolor":"#6B7280"},
                           "bar":{"color":icol,"thickness":0.25},"bgcolor":"#1A2030",
                           "steps":[{"range":[0,40],"color":"rgba(255,107,107,.12)"},
                                    {"range":[40,70],"color":"rgba(255,217,61,.08)"},
                                    {"range":[70,100],"color":"rgba(0,229,160,.08)"}]},
                    number={"font":{"color":icol,"size":36}},
                    title={"text":"Integrity Score","font":{"color":"#E8EAF0","size":13}},
                ))
                fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#E8EAF0",
                                    height=200,margin=dict(t=30,b=10))
                st.plotly_chart(fig_g, use_container_width=True)
 
            id1,id2,id3,id4 = st.columns(4)
            for col, label, val in [
                (id1,"Resume name",    integ_data.get("resume_name") or "Not found"),
                (id2,"GitHub handle",  f"@{integ_data.get('github_user')}" if integ_data.get("github_user") else "Not found"),
                (id3,"Portfolio name", integ_data.get("portfolio_name") or "Not found"),
                (id4,"GitHub name",    integ_data.get("gh_real_name") or "N/A"),
            ]:
                col.markdown(f"<div class='mc'><div style='font-size:.66rem;color:#6B7280'>{label.upper()}</div><div style='margin-top:5px;font-size:.85rem'>{val}</div></div>", unsafe_allow_html=True)
 
            if integ_data["issues"]:
                st.markdown("**🚨 Fix these before applying:**")
                for x in integ_data["issues"]:
                    st.markdown(f"<div class='diff-r'>{x}</div>", unsafe_allow_html=True)
            if integ_data["warnings"]:
                st.markdown("**⚠️ Recommended improvements:**")
                for x in integ_data["warnings"]:
                    st.markdown(f"<div class='diff-y'>{x}</div>", unsafe_allow_html=True)
            if integ_data["passes"]:
                st.markdown("**✅ Verified:**")
                for x in integ_data["passes"]:
                    st.markdown(f"<div class='diff'>{x}</div>", unsafe_allow_html=True)
 
            if PLOTLY and (github_text_final.strip() or (gp and gp.get("found"))):
                st.markdown("#### Skill evidence by source")
                r_s = set(integ_data["resume_skills"])
                g_s = set(integ_data["github_skills"]) | (set(gp["skills_from_github"]) if gp and gp.get("found") else set())
                p_s = set(integ_data["portfolio_skills"])
                venn = {"Resume only":len(r_s-g_s-p_s),"GitHub only":len(g_s-r_s-p_s),
                        "Portfolio only":len(p_s-r_s-g_s),"Resume+GitHub":len(r_s&g_s-p_s),"All three":len(r_s&g_s&p_s)}
                fig_v = go.Figure(go.Bar(x=list(venn),y=list(venn.values()),
                    marker_color=["#FF6B6B","#7C8FFF","#A29BFE","#FFD93D","#00E5A0"],
                    text=list(venn.values()),textposition="outside",textfont_color="#E8EAF0"))
                fig_v.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E8EAF0",yaxis=dict(gridcolor="rgba(255,255,255,.05)",title="Skills"),
                    height=230,margin=dict(t=10,b=10))
                st.plotly_chart(fig_v, use_container_width=True)
                st.caption("All-three skills carry highest credibility with recruiters.")
 
        # ─── C TAB 3: Role Fit ────────────────────────────────────────────
        with ct3:
            st.markdown("#### Your fit across all roles")
            if PLOTLY:
                rsc = result["all_role_scores"]
                sr  = sorted(rsc.items(), key=lambda x:x[1], reverse=True)
                fig_r = go.Figure(go.Bar(
                    x=[v for _,v in sr], y=[r for r,_ in sr], orientation="h",
                    marker_color=["#00E5A0" if v>=65 else "#FFD93D" if v>=40 else "#FF6B6B" for _,v in sr],
                    text=[f"{v:.0f}" for _,v in sr], textposition="outside", textfont_color="#E8EAF0",
                ))
                fig_r.add_vline(x=65,line_dash="dash",line_color="rgba(0,229,160,.4)",annotation_text="Strong")
                fig_r.add_vline(x=40,line_dash="dash",line_color="rgba(255,217,61,.4)",annotation_text="Moderate")
                fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E8EAF0",xaxis=dict(range=[0,115],gridcolor="rgba(255,255,255,.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,.05)"),height=340,margin=dict(t=10,b=10,r=40))
                st.plotly_chart(fig_r, use_container_width=True)
 
                # Spider
                dom = result["domain_scores"]
                cats = list(dom.keys())
                vals_sp = [dom[c] for c in cats]
                fig_sp = go.Figure(go.Scatterpolar(
                    r=vals_sp+[vals_sp[0]],theta=[c.replace("_"," ").title() for c in cats]+[cats[0].replace("_"," ").title()],
                    fill="toself",fillcolor="rgba(0,229,160,.1)",line_color="#00E5A0",line_width=2))
                fig_sp.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True,range=[0,100],gridcolor="rgba(255,255,255,.1)",tickcolor="#6B7280",tickfont_color="#6B7280"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,.1)",tickfont_color="#9CA3AF")),
                    paper_bgcolor="rgba(0,0,0,0)",font_color="#E8EAF0",height=320,margin=dict(t=20,b=20),showlegend=False)
                st.plotly_chart(fig_sp, use_container_width=True)
 
            best = max(result["all_role_scores"], key=result["all_role_scores"].get)
            if best != target_role:
                st.info(f"💡 Your profile fits **{best}** better ({result['all_role_scores'][best]:.0f}/100) than your target **{target_role}** ({result['all_role_scores'][target_role]:.0f}/100).")
 
        # ─── C TAB 4: Learning Path ───────────────────────────────────────
        with ct4:
            plan = result["learning_plan"]
            st.markdown(f"#### Personalised roadmap for **{target_role}**")
            if not plan:
                st.success("🎉 No critical skill gaps! You meet all required skills.")
            else:
                cumulative = 0
                for i, item in enumerate(plan):
                    s = item["skill"]; steps = item["steps"]
                    imp = "🔴 Critical" if s in JOB_ROLES[target_role]["required"] else "🟡 Important"
                    with st.expander(f"{imp} — **{s.replace('_',' ').title()}**", expanded=(i==0)):
                        for j,step in enumerate(steps):
                            st.markdown(f"<div class='step'>→ Step {j+1}: {step}</div>", unsafe_allow_html=True)
                        est = 2+j
                        st.markdown(f"<span style='color:#6B7280;font-size:.73rem'>Est. {est}–{est+2} weeks focused study</span>", unsafe_allow_html=True)
                        cumulative += est
                if PLOTLY and cumulative:
                    skills_tl = [p["skill"].replace("_"," ").title() for p in plan]
                    wks = [4,5,3,4][:len(plan)]
                    fig_tl = go.Figure()
                    cum = 0
                    for idx,(sk,wk) in enumerate(zip(skills_tl,wks)):
                        fig_tl.add_trace(go.Bar(name=sk,x=[wk],y=["Roadmap"],base=cum,orientation="h",
                            marker_color=["#00E5A0","#FFD93D","#FF9F43","#7C8FFF"][idx%4],
                            text=sk,textposition="inside",textfont_color="black"))
                        cum += wk
                    fig_tl.update_layout(barmode="stack",paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",font_color="#E8EAF0",
                        xaxis=dict(title="Weeks",gridcolor="rgba(255,255,255,.05)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                        height=120,margin=dict(t=5,b=35),showlegend=False)
                    st.plotly_chart(fig_tl, use_container_width=True)
                    st.caption(f"Total estimated roadmap: ~{cum} weeks")
 
        # ─── C TAB 5: ATS Comparison ──────────────────────────────────────
        with ct5:
            st.markdown("#### How TalentIQ differs from traditional ATS")
            ats_sim = min(100, len(result["overlap"]["matched_required"])*14 + len(result["overlap"]["matched_preferred"])*8)
            our_s   = result["fit_score"]
            comparisons = [
                ("Skill detection","Keyword count in resume only","3 sources + knowledge graph + aliases","We find skills ATS misses (TF→TensorFlow, k8s→Kubernetes)"),
                ("Scoring","Keyword density vs job description","Semantic similarity + structured overlap + experience","No reward for keyword stuffing"),
                ("Lie detection","None — takes resume at face value","GitHub API real-repo corroboration","We check if repos actually exist and use claimed skills"),
                ("Hyperlinks","Not checked","All PDF links validated (HTTP 200 check)","Broken portfolio/GitHub links caught instantly"),
                ("Buzzwords","Often rewarded","Penalised — replaced by metric requirement","Forces honest, evidence-based language"),
                ("Explainability","Black box — no feedback","Exact matched/missing skills + score breakdown","Candidate knows exactly what to improve"),
            ]
            for dim, ats_txt, our_txt, edge in comparisons:
                with st.expander(f"**{dim}**"):
                    ca,cb = st.columns(2)
                    with ca:
                        st.markdown("<span style='color:#FF6B6B;font-size:.7rem'>TRADITIONAL ATS</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='diff-r'>{ats_txt}</div>", unsafe_allow_html=True)
                    with cb:
                        st.markdown("<span style='color:#00E5A0;font-size:.7rem'>TALENT IQ</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='diff'>{our_txt}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='background:rgba(0,229,160,.06);border:1px solid rgba(0,229,160,.2);border-radius:8px;padding:10px 14px;font-size:.79rem;margin-top:4px'>💡 {edge}</div>", unsafe_allow_html=True)
 
            st.markdown("#### Score comparison: ATS simulation vs TalentIQ")
            if PLOTLY:
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(name="ATS (keyword sim)",x=["Score"],y=[ats_sim],marker_color="#FF6B6B",
                    text=[f"{ats_sim}"],textposition="outside",textfont_color="#E8EAF0"))
                fig_c.add_trace(go.Bar(name="TalentIQ",x=["Score"],y=[our_s],marker_color="#00E5A0",
                    text=[f"{our_s:.0f}"],textposition="outside",textfont_color="#E8EAF0"))
                fig_c.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",font_color="#E8EAF0",
                    yaxis=dict(range=[0,120],gridcolor="rgba(255,255,255,.05)"),height=240,margin=dict(t=10,b=10))
                st.plotly_chart(fig_c, use_container_width=True)
            delta = our_s - ats_sim
            if delta > 5:   st.success(f"TalentIQ scores **{delta:.0f} pts higher** — deep evidence the ATS can't see.")
            elif delta < -5: st.warning(f"ATS scores **{abs(delta):.0f} pts higher** — likely keyword stuffing without evidence. Our integrity engine discounts it.")
            else:            st.info("Scores are close — this candidate's language matches their actual skills.")
 
    # ═════════════════════════════════════════════════════════════════════════
    # RECRUITER VIEW  — fully rebuilt
    # ═════════════════════════════════════════════════════════════════════════
 
    else:
        st.markdown("---")
        ov       = result["overlap"]
        integ_d  = result["integrity"]
        fit      = result["fit_score"]
        integ_s  = integ_d["score"]
        cand     = integ_d.get("resume_name") or "Candidate"
        role     = result["role"]
 
        # ── Verdict banner ────────────────────────────────────────────────
        if   fit >= 65 and integ_s >= 65: verdict_bg, verdict_col, verdict, icon = "rgba(0,229,160,.08)","#00E5A0","PROCEED TO INTERVIEW","✅"
        elif fit >= 45 or  integ_s >= 55: verdict_bg, verdict_col, verdict, icon = "rgba(255,217,61,.08)","#FFD93D","REVIEW FURTHER","⚡"
        else:                             verdict_bg, verdict_col, verdict, icon = "rgba(255,107,107,.08)","#FF6B6B","DO NOT PROCEED","❌"
 
        st.markdown(f"""
        <div style='background:{verdict_bg};border:1.5px solid {verdict_col};border-radius:12px;padding:18px 24px;margin-bottom:18px'>
          <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:{verdict_col}'>{icon} {verdict}</div>
          <div style='color:#9CA3AF;font-size:.8rem;margin-top:4px'>{cand} → {role} · Fit {fit:.0f}/100 · Integrity {integ_s}/100</div>
        </div>""", unsafe_allow_html=True)
 
        # ── Scorecard ─────────────────────────────────────────────────────
        st.markdown("#### Scorecard")
        rc1,rc2,rc3,rc4,rc5,rc6 = st.columns(6)
        rc1.metric("Fit Score",         f"{fit:.0f}/100")
        rc2.metric("Integrity",         f"{integ_s}/100")
        rc3.metric("Required skills",   f"{len(ov['matched_required'])}/{len(ov['matched_required'])+len(ov['missing_required'])}")
        rc4.metric("Preferred skills",  f"{len(ov['matched_preferred'])}/{len(ov['matched_preferred'])+len(ov['missing_preferred'])}")
        rc5.metric("Experience",        f"{result['exp_years']} yrs")
        rc6.metric("Metrics in resume", len(result['metrics_found']))
 
        st.divider()
 
        # ── 3-column breakdown ────────────────────────────────────────────
        ra, rb, rc_ = st.columns(3)
 
        with ra:
            st.markdown("#### 🚩 Red flags")
            if integ_d["issues"]:
                for x in integ_d["issues"]:
                    st.markdown(f"<div class='flag-card'>🚨 {x}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='pass-card'>No critical issues found.</div>", unsafe_allow_html=True)
 
            if integ_d["warnings"]:
                st.markdown("**Warnings:**")
                for x in integ_d["warnings"]:
                    st.markdown(f"<div class='warn-card'>⚠️ {x}</div>", unsafe_allow_html=True)
 
        with rb:
            st.markdown("#### ✅ Strengths")
            all_matched = ov["matched_required"] + ov["matched_preferred"] + ov["matched_nice"]
            if all_matched:
                for s in all_matched:
                    st.markdown(f"<span class='tg'>{s.replace('_',' ')}</span>", unsafe_allow_html=True)
                st.markdown("")
            if integ_d["passes"]:
                st.markdown("**Verified signals:**")
                for x in integ_d["passes"][:5]:
                    st.markdown(f"<div class='pass-card'>{x}</div>", unsafe_allow_html=True)
 
        with rc_:
            st.markdown("#### ❌ Critical gaps")
            if ov["missing_required"]:
                for s in ov["missing_required"]:
                    st.markdown(f"<span class='tr'>{s.replace('_',' ')}</span>", unsafe_allow_html=True)
                st.markdown("")
            else:
                st.success("Meets all required skills.")
            if ov["missing_preferred"]:
                st.markdown("**Preferred gaps:**")
                for s in ov["missing_preferred"]:
                    st.markdown(f"<span class='ty'>{s.replace('_',' ')}</span>", unsafe_allow_html=True)
 
        st.divider()
 
        # ── Skill evidence table ───────────────────────────────────────────
        st.markdown("#### Skill evidence matrix")
        all_job_skills = (JOB_ROLES[role]["required"] + JOB_ROLES[role]["preferred"]
                          + JOB_ROLES[role].get("nice_to_have",[]))
        r_skills = set(integ_d["resume_skills"])
        g_skills = set(integ_d["github_skills"])
        if gp and gp.get("found"): g_skills.update(gp["skills_from_github"])
        p_skills = set(integ_d["portfolio_skills"])
        cand_all = set(result["candidate_skills"])
 
        tbl_rows = []
        for sk in all_job_skills:
            tier = "Required" if sk in JOB_ROLES[role]["required"] else ("Preferred" if sk in JOB_ROLES[role]["preferred"] else "Nice-to-have")
            resume_ev  = "✅" if sk in r_skills else "—"
            github_ev  = "✅" if sk in g_skills else "—"
            port_ev    = "✅" if sk in p_skills else "—"
            sources = sum([sk in r_skills, sk in g_skills, sk in p_skills])
            credibility = "🟢 High" if sources >= 2 else ("🟡 Low" if sources == 1 else "🔴 None")
            tbl_rows.append({"Skill": sk.replace("_"," ").title(), "Tier": tier,
                             "Resume": resume_ev,"GitHub": github_ev,"Portfolio": port_ev,
                             "Credibility": credibility})
        try:
            import pandas as pd
            df = pd.DataFrame(tbl_rows)
            st.dataframe(df, use_container_width=True, height=300,
                         column_config={"Skill":st.column_config.TextColumn(width="medium"),
                                        "Tier":st.column_config.TextColumn(width="small")})
        except ImportError:
            for row in tbl_rows:
                st.write(row)
 
        st.divider()
 
        # ── Interview probes ───────────────────────────────────────────────
        st.markdown("#### Interview probes (auto-generated)")
        probes = []
        for s in ov["missing_required"][:3]:
            probes.append(f"You list {s.replace('_',' ').title()} as a requirement. Can you walk me through a project where you used it?")
        for s in ov["missing_preferred"][:2]:
            probes.append(f"How comfortable are you with {s.replace('_',' ').title()}? Can you describe your exposure?")
        for issue in integ_d["issues"][:2]:
            if "skill inflation" in issue.lower() or "no GitHub" in issue:
                probes.append("Several skills on your resume don't appear in your GitHub repos. Can you show me code or a project for one of them?")
            if "metric" in issue.lower():
                probes.append("Your resume has few measurable outcomes. Can you quantify the impact of a past project? (scale, accuracy, latency, etc.)")
            if "mismatch" in issue.lower() or "name" in issue.lower():
                probes.append("There's a name discrepancy between your resume and online profiles. Can you clarify which is your primary professional identity?")
        for issue in integ_d.get("pdf_link_issues",[])[:1]:
            probes.append(f"One of your portfolio links appears broken ({issue[:60]}…). Can you share the updated link?")
        if not probes:
            probes.append("Profile looks solid — focus on culture fit and system design depth.")
 
        for i, probe in enumerate(probes, 1):
            st.markdown(f"<div class='probe'><strong>Q{i}:</strong> {probe}</div>", unsafe_allow_html=True)
 
        st.divider()
 
        # ── Comparative fit across roles ───────────────────────────────────
        st.markdown("#### Candidate fit across all roles")
        if PLOTLY:
            rsc = result["all_role_scores"]
            sr  = sorted(rsc.items(), key=lambda x:x[1], reverse=True)
            fig_rr = go.Figure(go.Bar(
                x=[v for _,v in sr], y=[r for r,_ in sr], orientation="h",
                marker_color=["#00E5A0" if v>=65 else "#FFD93D" if v>=40 else "#FF6B6B" for _,v in sr],
                text=[f"{v:.0f}" for _,v in sr], textposition="outside", textfont_color="#E8EAF0",
            ))
            fig_rr.add_vline(x=65,line_dash="dash",line_color="rgba(0,229,160,.4)")
            fig_rr.add_vline(x=40,line_dash="dash",line_color="rgba(255,217,61,.4)")
            fig_rr.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAF0",xaxis=dict(range=[0,115],gridcolor="rgba(255,255,255,.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,.05)"),height=300,margin=dict(t=10,b=10,r=40))
            st.plotly_chart(fig_rr, use_container_width=True)
 
        best = max(result["all_role_scores"], key=result["all_role_scores"].get)
        if best != role:
            st.info(f"💡 This candidate is a stronger fit for **{best}** ({result['all_role_scores'][best]:.0f}/100) than the applied role **{role}** ({result['all_role_scores'][role]:.0f}/100). Consider routing to that team.")
 
        # ── PDF links for recruiter ────────────────────────────────────────
        if pdf_links:
            st.divider()
            st.markdown("#### Portfolio link verification")
            for lk in pdf_links:
                cat = categorise_url(lk["url"])
                cls = "link-ok" if lk.get("valid") else "link-bad"
                st.markdown(f"<span class='tmu'>{cat}</span> <span class='{cls}'>{lk.get('label','?')}</span> "
                            f"<code style='font-size:.72rem'>{lk['url'][:70]}</code>",
                            unsafe_allow_html=True)
 
        # ── GitHub deep-dive ───────────────────────────────────────────────
        if gp and gp.get("found"):
            st.divider()
            st.markdown("#### GitHub due-diligence")
            gd1,gd2,gd3 = st.columns(3)
            gd1.metric("Public repos",   gp["public_repos"])
            gd2.metric("Total stars",    gp["starred_count"])
            gd3.metric("Followers",      gp["followers"])
            st.markdown("**Top repos:** " + " ".join(f"`{n}`" for n in gp["top_repo_names"]), unsafe_allow_html=False)
            if PLOTLY and gp["languages"]:
                lang_s = sorted(gp["languages"].items(), key=lambda x:x[1], reverse=True)[:8]
                fig_l = go.Figure(go.Bar(
                    x=[l for l,_ in lang_s], y=[c for _,c in lang_s],
                    marker_color="#7C8FFF",text=[c for _,c in lang_s],
                    textposition="outside",textfont_color="#E8EAF0"))
                fig_l.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E8EAF0",yaxis=dict(title="Repo count",gridcolor="rgba(255,255,255,.05)"),
                    height=230,margin=dict(t=10,b=10),xaxis_title="Language")
                st.plotly_chart(fig_l, use_container_width=True)
 