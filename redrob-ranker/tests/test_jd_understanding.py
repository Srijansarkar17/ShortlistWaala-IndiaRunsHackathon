import pytest
from src.jd_understanding.parser import JDParser, JDProfile, ExperienceRange

def test_jd_parser_basic():
    parser = JDParser()
    jd_text = """
    Job Title: Senior ML Engineer
    
    Required Qualifications:
    - 5+ years of experience in Software Engineering
    - Deep understanding of Python and PyTorch
    - Hands-on experience with SQL
    
    Preferred Qualifications:
    - Knowledge of LangChain and Hugging Face
    - Experience in SaaS products
    
    Soft Skills:
    - Excellent communication and collaboration
    - Proven leadership abilities
    
    Location:
    - Onsite in Bengaluru, India
    """
    
    profile = parser.parse(jd_text)
    
    # Check required skills (sorted list)
    assert "Python" in profile.required_skills
    assert "PyTorch" in profile.required_skills
    assert "SQL" in profile.required_skills
    
    # Check preferred skills (sorted list)
    assert "LangChain" in profile.preferred_skills
    assert "Hugging Face" in profile.preferred_skills
    
    # Ensure a required skill is not duplicated in preferred
    assert "Python" not in profile.preferred_skills
    
    # Check experience range
    assert profile.experience_range.min_years == 5.0
    assert profile.experience_range.max_years is None
    
    # Check soft skills
    assert "Communication" in profile.soft_skills
    assert "Collaboration" in profile.soft_skills
    assert "Leadership" in profile.soft_skills
    
    # Check location requirements
    assert "Onsite" in profile.location_requirements
    assert "Bengaluru" in profile.location_requirements
    assert "India" in profile.location_requirements
    
    # Check industry requirements
    assert "SaaS" in profile.industry_requirements

def test_jd_parser_experience_formats():
    parser = JDParser()
    
    # Case 1: Range format
    profile1 = parser.parse("Looking for someone with 3-5 years of experience.")
    assert profile1.experience_range.min_years == 3.0
    assert profile1.experience_range.max_years == 5.0
    
    # Case 2: Range with "to" word
    profile2 = parser.parse("Requires 2 to 4 years of background.")
    assert profile2.experience_range.min_years == 2.0
    assert profile2.experience_range.max_years == 4.0
    
    # Case 3: "At least" format
    profile3 = parser.parse("Must have at least 8 years in the field.")
    assert profile3.experience_range.min_years == 8.0
    assert profile3.experience_range.max_years is None
    
    # Case 4: No experience specified
    profile4 = parser.parse("Graduate role with zero requirements.")
    assert profile4.experience_range.min_years is None
    assert profile4.experience_range.max_years is None

def test_jd_parser_empty_and_noise():
    parser = JDParser()
    
    profile = parser.parse("")
    assert len(profile.required_skills) == 0
    assert len(profile.preferred_skills) == 0
    assert profile.experience_range.min_years is None
    assert profile.experience_range.max_years is None
    assert len(profile.location_requirements) == 0
    assert len(profile.industry_requirements) == 0
    assert len(profile.soft_skills) == 0

def test_jd_parser_special_skill_characters():
    parser = JDParser()
    
    # C++ and C# special patterns
    profile = parser.parse("Requirements: Experience with C++, C#, Python")
    assert "C++" in profile.required_skills
    assert "C#" in profile.required_skills
    assert "Python" in profile.required_skills
