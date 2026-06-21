from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger

class TrapDetector:
    """Detects 11 different candidate profile and behavioral traps with confidence scores."""

    def detect(
        self,
        candidates_parquet: str | Path,
        features_parquet: str | Path,
        reference_date: datetime = datetime(2026, 6, 21)
    ) -> pd.DataFrame:
        """
        Loads candidate data and detects traps.
        
        Parameters
        ----------
        candidates_parquet : str | Path
            Path to the normalized candidates database.
        features_parquet : str | Path
            Path to the candidate precomputed feature store.
        reference_date : datetime
            The evaluation reference date for calculating inactivity.
        """
        logger.info("Loading candidates and features data for trap detection...")
        
        candidates_df = pd.read_parquet(candidates_parquet)
        features_df = pd.read_parquet(features_parquet)
        
        # Join tables on candidate_id
        combined_df = candidates_df.merge(features_df, on="candidate_id", how="inner", suffixes=("", "_f"))
        logger.info(f"Loaded and joined data for {len(combined_df):,} candidates.")
        
        records = []
        
        # Hardcoded lists for heuristics
        consulting_kws = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy", "hcl", "tech mahindra", "l&t", "deloitte", "pwc", "ey", "kpmg"]
        research_kws = ["researcher", "research associate", "phd scholar", "postdoc", "postdoctoral", "fellow", "academic", "professor", "lecturer", "academic lab"]
        traditional_ml_skills = ["pytorch", "tensorflow", "scikit-learn", "keras", "machine learning", "nlp", "computer vision", "pandas", "numpy", "feature engineering", "deep learning"]
        architect_kws = ["architect", "director", "manager", "tech lead", "principal engineer", "technical architect"]
        ai_buzzwords = ["ai tools", "chatgpt", "generative ai", "prompt engineering", "artificial intelligence"]
        marketing_kws = ["marketing", "sales", "brand manager", "seo", "copywriter", "social media", "public relations", "ad operations"]
        robotics_skills = ["robotics", "ros", "lidar", "uav", "robot", "control theory", "kinematics", "slam", "path planning", "motion planning"]
        speech_skills = ["speech", "tts", "asr", "whisper", "audio", "voice", "speech recognition", "text-to-speech", "speech-to-text"]
        nlp_skills = ["nlp", "natural language", "bert", "embeddings", "elasticsearch", "vector search", "pinecone", "milvus", "weaviate", "qdrant", "rag", "retrieval", "text"]
        
        for row in combined_df.itertuples():
            candidate_id = row.candidate_id
            
            # Extract lists
            skill_names = []
            cert_names = []
            career_companies = []
            career_titles = []
            career_descriptions = []
            
            try:
                if isinstance(row.skill_names_json, str):
                    skill_names = [s.lower() for s in json.loads(row.skill_names_json)]
                if isinstance(row.certification_names_json, str):
                    cert_names = [c.lower() for c in json.loads(row.certification_names_json)]
                if isinstance(row.career_companies_json, str):
                    career_companies = [c.lower() for c in json.loads(row.career_companies_json)]
                if isinstance(row.career_titles_json, str):
                    career_titles = [t.lower() for t in json.loads(row.career_titles_json)]
                if isinstance(row.career_descriptions_json, str):
                    career_descriptions = [d.lower() for d in json.loads(row.career_descriptions_json)]
            except Exception:
                pass
                
            skill_set = set(skill_names)
            desc_text = " ".join(career_descriptions).lower()
            title_text = " ".join(career_titles).lower()
            summary = (row.summary or "").lower()
            current_title = (row.current_title or "").lower()
            
            # --- 1. Consulting Only ---
            # Stated companies are only consulting firms
            if not career_companies:
                trap_consulting_only = 0.0
            else:
                consulting_count = sum(1 for c in career_companies if any(kw in c for kw in consulting_kws))
                trap_consulting_only = float(consulting_count / len(career_companies))
                
            # --- 2. Research Only ---
            # Careers in pure research labs / academic posts
            all_titles = career_titles + ([current_title] if current_title else [])
            if not all_titles:
                trap_research_only = 0.0
            else:
                research_count = sum(1 for t in all_titles if any(kw in t for kw in research_kws))
                trap_research_only = float(research_count / len(all_titles))
                
            # --- 3. LangChain Only ---
            # Possesses wrapper experience but no traditional ML skills
            has_langchain = any("langchain" in s or "openai" in s or "chatgpt" in s for s in skill_names)
            has_traditional = any(m in skill_set for m in traditional_ml_skills)
            
            if has_langchain and not has_traditional:
                trap_langchain_only = 1.0
            elif has_langchain:
                trap_langchain_only = 0.3
            else:
                trap_langchain_only = 0.0
                
            # --- 4. Architect Who Does Not Code ---
            # High-level titles, zero coding activity
            is_arch = any(kw in current_title for kw in architect_kws)
            github_score = getattr(row, "signal_github_activity_score", -1.0)
            if github_score is None:
                github_score = -1.0
                
            if is_arch and github_score <= 0:
                trap_architect_no_code = 1.0
            elif is_arch and github_score < 20:
                trap_architect_no_code = 0.7
            elif is_arch:
                trap_architect_no_code = 0.2
            else:
                trap_architect_no_code = 0.0
                
            # --- 5. Job Hopper ---
            # Switches companies every 1-1.5 years on average (min 3 entries)
            num_jobs = getattr(row, "num_career_entries", 0)
            avg_tenure = getattr(row, "avg_tenure", 0.0) # in years
            
            if num_jobs >= 3:
                if avg_tenure <= 1.0:
                    trap_job_hopper = 1.0
                elif avg_tenure <= 1.5:
                    trap_job_hopper = 0.7
                elif avg_tenure <= 1.8:
                    trap_job_hopper = 0.3
                else:
                    trap_job_hopper = 0.0
            else:
                trap_job_hopper = 0.0
                
            # --- 6. Keyword Stuffer ---
            # Too many listed skills (e.g. >25 or >35)
            num_skills = getattr(row, "num_skills", 0)
            if num_skills > 35:
                trap_keyword_stuffer = 1.0
            elif num_skills > 25:
                trap_keyword_stuffer = 0.7
            elif num_skills > 15:
                trap_keyword_stuffer = 0.3
            else:
                trap_keyword_stuffer = 0.0
                
            # --- 7. Marketing Summary Trap ---
            # AI summary terms, but actual titles are in marketing/sales
            has_ai_buzz = any(bw in summary for bw in ai_buzzwords)
            is_mkt = any(kw in current_title for kw in marketing_kws) or any(kw in title_text for kw in marketing_kws)
            
            if has_ai_buzz and is_mkt:
                trap_marketing_summary = 1.0
            elif is_mkt:
                trap_marketing_summary = 0.3
            else:
                trap_marketing_summary = 0.0
                
            # --- 8. Inactive Candidate ---
            # Last active >90 days, or zero response rate
            last_active_s = getattr(row, "signal_last_active_date", None)
            response_rate = getattr(row, "signal_recruiter_response_rate", 1.0)
            if response_rate is None:
                response_rate = 1.0
                
            days_inactive = 0
            if last_active_s:
                try:
                    active_date = datetime.strptime(last_active_s[:10], "%Y-%m-%d")
                    days_inactive = (reference_date - active_date).days
                except Exception:
                    pass
                    
            if days_inactive > 180 or response_rate < 0.1:
                trap_inactive_candidate = 1.0
            elif days_inactive > 90 or response_rate < 0.3:
                trap_inactive_candidate = 0.7
            elif days_inactive > 30:
                trap_inactive_candidate = 0.3
            else:
                trap_inactive_candidate = 0.0
                
            # --- 9. CV Only ---
            # Low completeness score, zero active platform indicators
            comp_score = getattr(row, "profile_completeness_score", 100.0)
            if comp_score is None:
                comp_score = 100.0
            connections = getattr(row, "signal_connection_count", 0) or 0
            apps = getattr(row, "signal_applications_submitted_30d", 0) or 0
            views = getattr(row, "signal_profile_views_received_30d", 0) or 0
            
            if comp_score < 40 and connections == 0 and apps == 0 and views == 0:
                trap_cv_only = 1.0
            elif comp_score < 60 and connections <= 1 and apps == 0:
                trap_cv_only = 0.6
            else:
                trap_cv_only = 0.0
                
            # --- 10. Robotics Only ---
            # Robotics focus, zero NLP exposure
            has_robotics = any(any(kw in s for kw in robotics_skills) for s in skill_names)
            has_nlp = any(any(kw in s for kw in nlp_skills) for s in skill_names)
            
            if has_robotics and not has_nlp:
                trap_robotics_only = 1.0
            elif has_robotics:
                trap_robotics_only = 0.4
            else:
                trap_robotics_only = 0.0
                
            # --- 11. Speech Only ---
            # Speech/ASR/TTS focus, zero NLP exposure
            has_speech = any(any(kw in s for kw in speech_skills) for s in skill_names)
            
            if has_speech and not has_nlp:
                trap_speech_only = 1.0
            elif has_speech:
                trap_speech_only = 0.4
            else:
                trap_speech_only = 0.0
                
            records.append({
                "candidate_id": candidate_id,
                "trap_consulting_only": float(trap_consulting_only),
                "trap_research_only": float(trap_research_only),
                "trap_langchain_only": float(trap_langchain_only),
                "trap_architect_no_code": float(trap_architect_no_code),
                "trap_job_hopper": float(trap_job_hopper),
                "trap_keyword_stuffer": float(trap_keyword_stuffer),
                "trap_marketing_summary": float(trap_marketing_summary),
                "trap_inactive_candidate": float(trap_inactive_candidate),
                "trap_cv_only": float(trap_cv_only),
                "trap_robotics_only": float(trap_robotics_only),
                "trap_speech_only": float(trap_speech_only)
            })
            
        return pd.DataFrame(records)
