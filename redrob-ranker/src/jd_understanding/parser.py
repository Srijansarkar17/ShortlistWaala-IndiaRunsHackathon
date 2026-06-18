import re
import json
from typing import List, Optional
from pydantic import BaseModel, Field

class ExperienceRange(BaseModel):
    min_years: Optional[float] = None
    max_years: Optional[float] = None

class JDProfile(BaseModel):
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience_range: ExperienceRange = Field(default_factory=ExperienceRange)
    location_requirements: List[str] = Field(default_factory=list)
    industry_requirements: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)

# Keyword dictionaries mapping lowercase query strings to clean formatted versions
TECH_SKILLS_MAP = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "c++": "C++",
    "java": "Java",
    "scala": "Scala",
    "kotlin": "Kotlin",
    "ruby": "Ruby",
    "php": "PHP",
    "c#": "C#",
    "sql": "SQL",
    "nosql": "NoSQL",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "scikit-learn",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "llm": "LLMs",
    "llms": "LLMs",
    "rag": "RAG",
    "prompt engineering": "Prompt Engineering",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
    "computer vision": "Computer Vision",
    "opencv": "OpenCV",
    "peft": "PEFT",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "milvus": "Milvus",
    "faiss": "FAISS",
    "qdrant": "Qdrant",
    "pinecone": "Pinecone",
    "weaviate": "Weaviate",
    "vector search": "Vector Search",
    "opensearch": "OpenSearch",
    "elasticsearch": "Elasticsearch",
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "gans": "GANs",
    "reinforcement learning": "Reinforcement Learning",
    "statistical modeling": "Statistical Modeling",
    "embeddings": "Embeddings",
    "semantic search": "Semantic Search",
    "yolo": "YOLO",
    "cnn": "CNN",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "terraform": "Terraform",
    "git": "Git",
    "mlflow": "MLflow",
    "kubeflow": "Kubeflow",
    "bentoml": "BentoML"
}

SOFT_SKILLS_MAP = {
    "communication": "Communication",
    "collaboration": "Collaboration",
    "leadership": "Leadership",
    "problem-solving": "Problem-solving",
    "problem solving": "Problem-solving",
    "teamwork": "Teamwork",
    "adaptability": "Adaptability",
    "creativity": "Creativity",
    "critical thinking": "Critical thinking",
    "mentorship": "Mentorship",
    "initiative": "Initiative",
    "presentation": "Presentation",
    "organizational": "Organizational",
    "time management": "Time management"
}

INDUSTRIES_MAP = {
    "fintech": "FinTech",
    "e-commerce": "E-commerce",
    "ecommerce": "E-commerce",
    "saas": "SaaS",
    "healthcare": "Healthcare",
    "finance": "Finance",
    "banking": "Banking",
    "automotive": "Automotive",
    "blockchain": "Blockchain",
    "crypto": "Crypto",
    "web3": "Web3",
    "cybersecurity": "Cybersecurity",
    "gaming": "Gaming",
    "education": "Education",
    "edtech": "EdTech",
    "retail": "Retail",
    "insurance": "Insurance"
}

LOCATIONS_MAP = {
    "remote": "Remote",
    "hybrid": "Hybrid",
    "onsite": "Onsite",
    "on-site": "Onsite",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "san francisco": "San Francisco",
    "sf": "San Francisco",
    "london": "London",
    "new york": "New York",
    "india": "India",
    "us": "US",
    "usa": "US",
    "uk": "UK"
}

class JDParser:
    """Deterministic local parser for Job Description text."""

    def parse(self, text: str) -> JDProfile:
        """Parse raw JD text and return a validated JDProfile."""
        # Normalize text and split into lines/paragraphs
        lines = [line.strip() for line in text.splitlines()]
        
        required_skills_set = set()
        preferred_skills_set = set()
        soft_skills_set = set()
        industries_set = set()
        locations_set = set()
        
        # Section classification state
        # 0 = General, 1 = Required Skills/Qualifications, 2 = Preferred Skills/Pluses
        section_state = 0
        
        for line in lines:
            if not line:
                continue
                
            line_lower = line.lower()
            
            # 1. Update Section State
            preferred_headers = ["preferred", "nice to have", "plus", "desirable", "good to have", "bonus", "assets"]
            required_headers = ["required", "must have", "requirements", "qualifications", "essential", "about you", "what you need"]
            
            # Simple header detection (e.g. if the line is short and matches keyword)
            is_header = len(line) < 50
            
            if is_header:
                if any(h in line_lower for h in preferred_headers):
                    section_state = 2
                elif any(h in line_lower for h in required_headers):
                    section_state = 1
                    
            # 2. Extract Technical Skills
            # Use regex to find whole words matching skills
            for keyword, clean_name in TECH_SKILLS_MAP.items():
                # Escape special chars like c++
                escaped_kw = re.escape(keyword)
                # Word boundaries for alphanumeric, but also handle special chars
                pattern = rf'\b{escaped_kw}\b'
                if keyword in ["c++", "c#"]:
                    pattern = rf'(?:^|[\s,;]){escaped_kw}(?:$|[\s,;])'
                
                if re.search(pattern, line_lower):
                    if section_state == 2:
                        preferred_skills_set.add(clean_name)
                    else:
                        required_skills_set.add(clean_name)
                        
            # 3. Extract Soft Skills
            for keyword, clean_name in SOFT_SKILLS_MAP.items():
                pattern = rf'\b{re.escape(keyword)}\b'
                if re.search(pattern, line_lower):
                    soft_skills_set.add(clean_name)
                    
            # 4. Extract Industry Requirements
            for keyword, clean_name in INDUSTRIES_MAP.items():
                pattern = rf'\b{re.escape(keyword)}\b'
                if re.search(pattern, line_lower):
                    industries_set.add(clean_name)
                    
            # 5. Extract Location Requirements
            for keyword, clean_name in LOCATIONS_MAP.items():
                pattern = rf'\b{re.escape(keyword)}\b'
                if re.search(pattern, line_lower):
                    locations_set.add(clean_name)
                    
        # 6. Extract Experience Range
        # Find matches for years of experience
        experience_patterns = [
            # E.g. "5-8 years of experience", "5 to 8 years"
            re.compile(r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?', re.IGNORECASE),
            # E.g. "5+ years", "5 years+"
            re.compile(r'(\d+)\s*\+\s*years?', re.IGNORECASE),
            re.compile(r'(\d+)\s*years?\s*\+', re.IGNORECASE),
            # E.g. "at least 5 years", "minimum of 5 years"
            re.compile(r'(?:at least|minimum of|min)\s*(\d+)\s*years?', re.IGNORECASE),
            # E.g. "5 years of experience"
            re.compile(r'(\d+)\s*years?\s*(?:of\s*)?experience', re.IGNORECASE)
        ]
        
        min_years = None
        max_years = None
        
        # Search the entire text for experience patterns
        text_normalized = " ".join(lines)
        for pattern in experience_patterns:
            matches = list(pattern.finditer(text_normalized))
            if matches:
                # Pick the first match that is likely referring to work experience
                # (since we look for "years" / "experience" / "background")
                for match in matches:
                    groups = match.groups()
                    if len(groups) == 2 and groups[0] and groups[1]:
                        min_years = float(groups[0])
                        max_years = float(groups[1])
                        break
                    elif len(groups) == 1 and groups[0]:
                        min_years = float(groups[0])
                        max_years = None
                        break
                if min_years is not None:
                    break
                    
        # If any required skills are also classified as preferred, keep them in required and drop from preferred
        preferred_skills_set = preferred_skills_set - required_skills_set

        return JDProfile(
            required_skills=sorted(list(required_skills_set)),
            preferred_skills=sorted(list(preferred_skills_set)),
            experience_range=ExperienceRange(min_years=min_years, max_years=max_years),
            location_requirements=sorted(list(locations_set)),
            industry_requirements=sorted(list(industries_set)),
            soft_skills=sorted(list(soft_skills_set))
        )
