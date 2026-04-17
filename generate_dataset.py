"""
Synthetic Dataset Generator for Multimodal Talent Intelligence System
Generates resume-job pairs with skill overlap labels for training/fine-tuning
"""

import json
import random
import csv
from itertools import combinations

# ─────────────────────────────────────────────
# SKILL TAXONOMY (Knowledge Graph Base)
# ─────────────────────────────────────────────
SKILL_GRAPH = {
    "python": ["numpy", "pandas", "scikit-learn", "pytorch", "tensorflow", "fastapi", "flask", "django"],
    "machine_learning": ["regression", "classification", "clustering", "xgboost", "lightgbm", "random_forest"],
    "deep_learning": ["cnn", "rnn", "lstm", "transformers", "attention_mechanism", "bert", "gpt"],
    "nlp": ["tokenization", "named_entity_recognition", "sentiment_analysis", "text_classification", "embeddings", "spacy", "nltk"],
    "computer_vision": ["image_classification", "object_detection", "yolo", "opencv", "image_segmentation"],
    "data_engineering": ["sql", "spark", "kafka", "airflow", "etl", "data_pipelines", "dbt"],
    "cloud": ["aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci_cd"],
    "backend": ["java", "spring_boot", "nodejs", "golang", "rest_api", "graphql", "microservices"],
    "frontend": ["react", "vue", "typescript", "javascript", "html_css", "tailwind", "nextjs"],
    "devops": ["git", "jenkins", "github_actions", "ansible", "monitoring", "prometheus", "grafana"],
    "data_science": ["statistics", "hypothesis_testing", "a_b_testing", "data_visualization", "matplotlib", "seaborn", "tableau"],
    "databases": ["postgresql", "mongodb", "redis", "elasticsearch", "mysql", "cassandra"],
}

DOMAINS = {
    "ML Engineer": {
        "required": ["python", "machine_learning", "deep_learning", "docker", "git"],
        "preferred": ["nlp", "pytorch", "tensorflow", "kubernetes", "aws"],
        "nice_to_have": ["spark", "kafka", "data_pipelines"],
        "description": "Build and deploy scalable ML models for production systems."
    },
    "Data Scientist": {
        "required": ["python", "statistics", "machine_learning", "sql", "data_visualization"],
        "preferred": ["deep_learning", "a_b_testing", "tableau", "r"],
        "nice_to_have": ["spark", "airflow", "cloud"],
        "description": "Drive business decisions through data analysis and predictive modeling."
    },
    "NLP Engineer": {
        "required": ["python", "nlp", "transformers", "bert", "pytorch"],
        "preferred": ["text_classification", "named_entity_recognition", "embeddings", "spacy"],
        "nice_to_have": ["fastapi", "docker", "aws"],
        "description": "Build NLP pipelines and language understanding systems."
    },
    "Backend Engineer": {
        "required": ["python", "rest_api", "sql", "git", "docker"],
        "preferred": ["microservices", "kubernetes", "redis", "postgresql"],
        "nice_to_have": ["kafka", "elasticsearch", "aws"],
        "description": "Design and develop robust backend services and APIs."
    },
    "Full Stack Developer": {
        "required": ["javascript", "react", "nodejs", "sql", "git"],
        "preferred": ["typescript", "docker", "rest_api", "postgresql", "css"],
        "nice_to_have": ["graphql", "kubernetes", "aws", "redis"],
        "description": "Build end-to-end web applications from UI to database."
    },
    "Data Engineer": {
        "required": ["python", "sql", "spark", "airflow", "etl"],
        "preferred": ["kafka", "aws", "docker", "data_pipelines", "dbt"],
        "nice_to_have": ["kubernetes", "terraform", "mongodb"],
        "description": "Design data infrastructure and pipelines for analytics at scale."
    },
    "Computer Vision Engineer": {
        "required": ["python", "deep_learning", "cnn", "opencv", "pytorch"],
        "preferred": ["object_detection", "yolo", "image_segmentation", "tensorflow"],
        "nice_to_have": ["docker", "aws", "kubernetes"],
        "description": "Develop vision AI models for real-world perception tasks."
    },
    "DevOps Engineer": {
        "required": ["docker", "kubernetes", "git", "ci_cd", "linux"],
        "preferred": ["aws", "terraform", "ansible", "prometheus", "jenkins"],
        "nice_to_have": ["python", "grafana", "kafka"],
        "description": "Build and maintain scalable infrastructure and deployment pipelines."
    },
}

# ─────────────────────────────────────────────
# CANDIDATE PROFILES (templates)
# ─────────────────────────────────────────────

NAMES = [
    "Arjun Sharma", "Priya Patel", "Rahul Verma", "Sneha Iyer", "Kiran Reddy",
    "Aditya Nair", "Divya Mehta", "Rohit Gupta", "Ananya Singh", "Vivek Kumar",
    "Meera Pillai", "Siddharth Joshi", "Neha Rao", "Akash Bose", "Kavya Thomas",
    "Varun Mishra", "Lakshmi Venkat", "Harsh Tiwari", "Pooja Saxena", "Abhinav Das",
    "Emma Wilson", "James Chen", "Sarah Okonkwo", "Carlos Rivera", "Yuki Tanaka",
    "Ali Hassan", "Maria Santos", "David Kowalski", "Fatima Al-Rashid", "Lucas Müller",
]

UNIVERSITIES = [
    "IIT Bombay", "IIT Delhi", "BITS Pilani", "NIT Trichy", "IIT Madras",
    "VIT Vellore", "IIIT Hyderabad", "DTU Delhi", "RVCE Bangalore", "PSG Tech",
    "MIT", "Stanford University", "UC Berkeley", "Carnegie Mellon", "Georgia Tech",
    "University of Toronto", "ETH Zurich", "TU Munich", "University of Waterloo",
]

COMPANIES = [
    "Google", "Microsoft", "Amazon", "Flipkart", "Ola", "Swiggy", "Razorpay",
    "CRED", "Zomato", "Meesho", "Paytm", "PhonePe", "Byju's", "Freshworks",
    "TCS", "Infosys", "Wipro", "HCL", "Capgemini", "Accenture",
    "JP Morgan", "Goldman Sachs", "Deutsche Bank", "Morgan Stanley",
    "Startup Alpha", "DataCo Labs", "NeuralBridge AI", "QuantEdge", "ByteForge",
]

PROJECT_TEMPLATES = {
    "ml": [
        "Built a {task} model achieving {metric}% accuracy using {tech}",
        "Developed {task} pipeline with {tech}, reducing inference time by {num}%",
        "Implemented {task} system for {domain} domain using {tech}",
    ],
    "nlp": [
        "Fine-tuned BERT for {task} achieving F1 score of {metric}",
        "Built {task} classifier processing {num}K documents daily using {tech}",
        "Developed multilingual {task} system using transformer embeddings",
    ],
    "data": [
        "Designed data pipeline processing {num}TB daily using {tech}",
        "Built real-time analytics dashboard with {tech}",
        "Automated {task} workflow saving {num} hours/week using {tech}",
    ],
    "backend": [
        "Developed RESTful API serving {num}K requests/day using {tech}",
        "Built microservices architecture handling {num}M users using {tech}",
        "Optimized database queries reducing latency by {num}% in {tech}",
    ],
}

# ─────────────────────────────────────────────
# GENERATOR FUNCTIONS
# ─────────────────────────────────────────────

def pick_skills(domain_key, overlap_level="high"):
    """
    Generate candidate skill set with controlled overlap to job requirements.
    overlap_level: 'high' (0.75-1.0), 'medium' (0.45-0.74), 'low' (0.1-0.44)
    """
    job = DOMAINS[domain_key]
    all_job_skills = job["required"] + job["preferred"] + job["nice_to_have"]

    if overlap_level == "high":
        # Has most required + some preferred
        must = job["required"][:]
        extra = random.sample(job["preferred"], min(len(job["preferred"]), random.randint(2, 4)))
        candidate_skills = list(set(must + extra))
        # Add 2-4 unrelated skills
        all_flat = [s for skills in SKILL_GRAPH.values() for s in skills]
        bonus = random.sample(all_flat, 3)
        candidate_skills = list(set(candidate_skills + bonus))
    
    elif overlap_level == "medium":
        # Has some required, misses some
        must = random.sample(job["required"], max(1, len(job["required"]) - 2))
        extra = random.sample(job["preferred"], min(len(job["preferred"]), 1))
        candidate_skills = list(set(must + extra))
        all_flat = [s for skills in SKILL_GRAPH.values() for s in skills]
        bonus = random.sample(all_flat, 4)
        candidate_skills = list(set(candidate_skills + bonus))
    
    else:  # low
        # Mostly unrelated skills, few matches
        must = random.sample(job["required"], max(1, len(job["required"]) - 3))
        all_flat = [s for skills in SKILL_GRAPH.values() for s in skills]
        unrelated = random.sample([s for s in all_flat if s not in all_job_skills], 5)
        candidate_skills = list(set(must[:1] + unrelated))
    
    return candidate_skills


def compute_fit_score(candidate_skills, job_key):
    """Compute ground-truth fit score for training labels."""
    job = DOMAINS[job_key]
    
    required = set(job["required"])
    preferred = set(job["preferred"])
    nice = set(job["nice_to_have"])
    candidate = set(candidate_skills)
    
    req_match = len(required & candidate) / len(required) if required else 0
    pref_match = len(preferred & candidate) / len(preferred) if preferred else 0
    nice_match = len(nice & candidate) / len(nice) if nice else 0
    
    # Weighted score
    score = (req_match * 0.6) + (pref_match * 0.3) + (nice_match * 0.1)
    
    # Add noise ±5 for realism
    noise = random.uniform(-0.05, 0.05)
    score = max(0.0, min(1.0, score + noise))
    
    return round(score * 100, 1)


def generate_resume_text(name, skills, experience_years, job_key=None):
    """Generate realistic resume text."""
    uni = random.choice(UNIVERSITIES)
    grad_year = 2024 - experience_years
    companies_worked = random.sample(COMPANIES, min(experience_years, 3))
    
    skill_str = ", ".join(s.replace("_", " ").title() for s in skills[:12])
    
    projects = []
    for _ in range(random.randint(2, 4)):
        ptype = random.choice(["ml", "data", "backend", "nlp"])
        tmpl = random.choice(PROJECT_TEMPLATES[ptype])
        proj = tmpl.format(
            task=random.choice(["fraud detection", "recommendation", "churn prediction", "sentiment analysis", "image recognition"]),
            tech=random.choice(skills[:5]).replace("_", " "),
            metric=random.randint(88, 98),
            num=random.randint(20, 80),
            domain=random.choice(["e-commerce", "finance", "healthcare", "logistics"]),
        )
        projects.append(proj)
    
    resume = f"""
Name: {name}
Education: B.Tech Computer Science, {uni}, {grad_year}
Experience: {experience_years} years

Skills: {skill_str}

Work Experience:
"""
    for i, company in enumerate(companies_worked):
        role = random.choice(["Software Engineer", "Data Scientist", "ML Engineer", "Backend Developer", "Analyst"])
        resume += f"- {role} at {company} ({grad_year + i} - {grad_year + i + 2})\n"
    
    resume += "\nProjects:\n"
    for p in projects:
        resume += f"- {p}\n"
    
    resume += f"\nGitHub: github.com/{name.lower().replace(' ', '')}"
    
    return resume.strip()


def generate_github_text(skills, experience_years):
    """Simulate GitHub profile text."""
    repos = []
    for skill in random.sample(skills, min(5, len(skills))):
        stars = random.randint(0, 500)
        repos.append(f"{skill.replace('_', '-')}-project ({stars} stars, {random.choice(['Python', 'JavaScript', 'Go'])})")
    
    contributions = experience_years * random.randint(100, 400)
    
    return f"""
GitHub Profile:
Contributions last year: {contributions}
Top Languages: {', '.join(random.sample([s.replace('_',' ') for s in skills[:4]], min(3, len(skills))))}
Repositories: {len(repos)} public repos
Notable: {'; '.join(repos[:3])}
""".strip()


def generate_portfolio_text(skills, name):
    """Simulate portfolio summary."""
    focus = random.choice(["building scalable ML systems", "NLP and text understanding", 
                           "data-driven product development", "full-stack AI applications"])
    return f"""
Portfolio - {name}
Focus: {focus}
Key Projects: Applied {', '.join(s.replace('_',' ') for s in random.sample(skills, min(3, len(skills))))} in real-world settings.
Blog posts on {random.choice(['model deployment', 'distributed systems', 'NLP fine-tuning', 'MLOps'])}.
Open source contributor.
""".strip()


def generate_job_description(job_key):
    """Generate job description text."""
    job = DOMAINS[job_key]
    req = ", ".join(s.replace("_", " ").title() for s in job["required"])
    pref = ", ".join(s.replace("_", " ").title() for s in job["preferred"])
    
    return f"""
Role: {job_key}
{job['description']}

Required Skills: {req}
Preferred: {pref}
Experience: {random.randint(2, 6)}+ years

Responsibilities:
- Design and implement scalable solutions
- Collaborate with cross-functional teams
- Drive technical decisions and code reviews
- Mentor junior engineers
""".strip()


# ─────────────────────────────────────────────
# MAIN DATASET GENERATION
# ─────────────────────────────────────────────

def generate_dataset(n_samples=1500):
    dataset = []
    
    overlap_distribution = {
        "high": 0.35,    # 35% strong matches (score 70-100)
        "medium": 0.40,  # 40% medium matches (score 40-70)
        "low": 0.25,     # 25% weak matches (score 10-40)
    }
    
    sample_id = 0
    
    for job_key in DOMAINS.keys():
        per_job = n_samples // len(DOMAINS)
        
        for level, pct in overlap_distribution.items():
            n_level = int(per_job * pct)
            
            for _ in range(n_level):
                name = random.choice(NAMES)
                experience = random.randint(1, 10)
                skills = pick_skills(job_key, overlap_level=level)
                
                fit_score = compute_fit_score(skills, job_key)
                
                job = DOMAINS[job_key]
                matched_skills = list(set(skills) & set(job["required"] + job["preferred"] + job["nice_to_have"]))
                missing_required = list(set(job["required"]) - set(skills))
                missing_preferred = list(set(job["preferred"]) - set(skills))
                
                record = {
                    "id": f"sample_{sample_id:04d}",
                    "candidate_name": name,
                    "target_role": job_key,
                    "experience_years": experience,
                    "overlap_level": level,
                    
                    # Raw text inputs (for embedding)
                    "resume_text": generate_resume_text(name, skills, experience, job_key),
                    "github_text": generate_github_text(skills, experience),
                    "portfolio_text": generate_portfolio_text(skills, name),
                    "job_description": generate_job_description(job_key),
                    
                    # Structured features
                    "candidate_skills": skills,
                    "required_skills": job["required"],
                    "preferred_skills": job["preferred"],
                    
                    # Labels (ground truth for training)
                    "fit_score": fit_score,                          # Regression target (0-100)
                    "fit_label": "high" if fit_score >= 65 else ("medium" if fit_score >= 35 else "low"),
                    "matched_skills": matched_skills,
                    "missing_required": missing_required,
                    "missing_preferred": missing_preferred,
                    
                    # Explainability fields
                    "skill_gap_count": len(missing_required),
                    "skill_overlap_ratio": round(len(matched_skills) / max(len(job["required"] + job["preferred"]), 1), 3),
                    "suggestions": [
                        f"Learn {s.replace('_', ' ').title()} to meet core requirements" 
                        for s in missing_required[:2]
                    ] + [
                        f"Add {s.replace('_', ' ').title()} to strengthen profile"
                        for s in missing_preferred[:2]
                    ],
                }
                
                dataset.append(record)
                sample_id += 1
    
    random.shuffle(dataset)
    print(f"✅ Generated {len(dataset)} samples across {len(DOMAINS)} job roles")
    return dataset


def save_dataset(dataset, prefix="talent_dataset"):
    # Save as JSON (full, for training)
    with open(f"{prefix}.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"📁 Saved {prefix}.json ({len(dataset)} records)")
    
    # Save as CSV (lightweight, for quick inspection)
    flat_records = []
    for d in dataset:
        flat_records.append({
            "id": d["id"],
            "candidate_name": d["candidate_name"],
            "target_role": d["target_role"],
            "experience_years": d["experience_years"],
            "overlap_level": d["overlap_level"],
            "fit_score": d["fit_score"],
            "fit_label": d["fit_label"],
            "skill_overlap_ratio": d["skill_overlap_ratio"],
            "skill_gap_count": d["skill_gap_count"],
            "candidate_skills": "|".join(d["candidate_skills"]),
            "missing_required": "|".join(d["missing_required"]),
            "resume_snippet": d["resume_text"][:200].replace("\n", " "),
        })
    
    with open(f"{prefix}.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_records[0].keys())
        writer.writeheader()
        writer.writerows(flat_records)
    print(f"📁 Saved {prefix}.csv (flat format for inspection)")
    
    # Save skill graph
    with open("skill_graph.json", "w") as f:
        json.dump(SKILL_GRAPH, f, indent=2)
    print(f"📁 Saved skill_graph.json (knowledge graph)")
    
    # Save job descriptions
    job_desc = {k: v for k, v in DOMAINS.items()}
    with open("job_roles.json", "w") as f:
        json.dump(job_desc, f, indent=2)
    print(f"📁 Saved job_roles.json")


if __name__ == "__main__":
    print("🚀 Generating Talent Intelligence Dataset...")
    dataset = generate_dataset(n_samples=1500)
    
    # Print stats
    from collections import Counter
    labels = Counter(d["fit_label"] for d in dataset)
    roles = Counter(d["target_role"] for d in dataset)
    
    print("\n📊 Dataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Fit label distribution: {dict(labels)}")
    print(f"  Samples per role: {dict(roles)}")
    
    scores = [d["fit_score"] for d in dataset]
    print(f"  Score range: {min(scores):.1f} – {max(scores):.1f}")
    print(f"  Average score: {sum(scores)/len(scores):.1f}")
    
    save_dataset(dataset)
    print("\n✅ Dataset generation complete!")
