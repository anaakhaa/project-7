import joblib
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import PyPDF2
import docx
import base64
import pandas as pd
from PyPDF2.errors import PdfReadError

# Load NLP model
nlp = spacy.load('en_core_web_sm')

# Load the pre-trained models
classifier = joblib.load("C:/Users/acer/PycharmProjects/resumemyself/models/resume_classifier.pkl")
vectorizer = joblib.load("C:/Users/acer/PycharmProjects/resumemyself/models/vectorizer.pkl")

# Define common skills for matching
COMMON_SKILLS = [
    "Python", "Java", "JavaScript", "HTML", "CSS", "Adobe Photoshop", "Adobe Illustrator",
    "UI/UX Design", "Graphic Design", "InDesign", "Premiere Pro", "After Effects", "WordPress",
    "Prototyping", "Invision", "Axure", "Communication", "Project Management",
    "Financial Analysis", "Excel", "Data Visualization", "Investment Strategies", "SQL",
    "Branding", "Creative Suite"
]

# Company credentials
COMPANY_ID = "luxeo123"
COMPANY_PASSWORD = "luxeo@global"

# Add background image function with base64 encoding
def add_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to extract text from files
def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode('utf-8', errors='ignore')
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
    except PdfReadError as e:
        st.error(f"Error reading PDF file: {uploaded_file.name}. {e}")
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {e}")

    return text.strip() or "Empty"

# Extract features function
def extract_features(text):
    doc = nlp(text)
    skills = [token.text for token in doc if token.text in COMMON_SKILLS]
    return {"skills": list(set(skills))}

# Improved function to extract candidate name
def extract_candidate_name(text):
    """Extracts the candidate's name using Named Entity Recognition (NER) and heuristic methods."""
    doc = nlp(text)

    # Extract names labeled as PERSON
    candidate_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    # Heuristic 1: Look for labels like 'Name:'
    lines = text.splitlines()
    for line in lines:
        if "name:" in line.lower():
            potential_name = line.split(':', 1)[-1].strip()
            if potential_name:
                return potential_name

    # Heuristic 2: Check for the largest heading or prominently formatted text
    headings = [line.strip() for line in lines if line.isupper() or line.istitle()]
    for heading in headings:
        if heading.lower() not in ["profile", "experience", "skills", "honors", "education"]:
            return heading

    # Heuristic 3: Validate PERSON entities as fallback
    if candidate_names:
        excluded_words = ["Collaborate", "Experience", "Profile", "Skills", "Honors"]
        for name in candidate_names:
            if name not in excluded_words:
                return name

    return "Unknown"  # If no name is found

# Enhanced function to extract name with font prioritization for PDFs
def extract_name_with_fonts(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        largest_text = ""
        largest_font_size = 0

        for page in reader.pages:
            if "/FontSize" in page:
                for obj in page["/FontSize"]:
                    size = obj.get("/FontSize", 0)
                    text = obj.get("/Text", "")
                    if size > largest_font_size:
                        largest_font_size = size
                        largest_text = text

        if largest_text.strip():
            return largest_text
    except Exception:
        pass

    return ""

# Match Resume with Job
def match_resume_with_job(resume_features, job_features):
    resume_text = ' '.join(resume_features["skills"])
    job_text = ' '.join(job_features["skills"])

    if not resume_text.strip() or not job_text.strip():
        return None

    vectorizer = TfidfVectorizer()
    resume_tfidf = vectorizer.fit_transform([resume_text])
    job_tfidf = vectorizer.transform([job_text])
    similarity = cosine_similarity(resume_tfidf, job_tfidf)

    return similarity[0][0] * 10

# Suggestions for improvements
def suggest_improvements(resume_features, job_features):
    missing_skills = [skill for skill in job_features["skills"] if skill not in resume_features["skills"]]
    return {"missing_skills": missing_skills}

# Function to handle page navigation using session state
def set_page(page):
    st.session_state.page = page

# Initialize session state for page if not already set
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Home Page
def home_page():
    add_bg_image("C:/Users/acer/PycharmProjects/resumemyself/images/res1.jpg")
    st.markdown("<h1 style='color:white; text-align:center;'>Welcome to the Smart Resume Analyzer</h1>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Applicant Form", key="applicant_form_button_home"):
            set_page("applicant_form")
    with col2:
        if st.button("Go to Company Login", key="company_login_button_home"):
            set_page("company_login")

# Applicant Form Page
def applicant_form():
    add_bg_image("C:/Users/acer/PycharmProjects/resumemyself/images/analyres.jpg")
    st.markdown("<h1 style='color:white; text-align:center;'>Applicant Form</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        label {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    name = st.text_input("Name", key="name_input")
    job_description = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])
    resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Submit Application"):
            if name and resume and job_description:
                applicant_id = str(uuid.uuid4())
                resume_text = extract_text_from_file(resume)
                job_text = extract_text_from_file(job_description)

                if resume_text == "Empty" or job_text == "Empty":
                    st.error("Resume or Job Description is empty or contains only invalid content.")
                else:
                    resume_features = extract_features(resume_text)
                    job_features = extract_features(job_text)

                    match_score = match_resume_with_job(resume_features, job_features)
                    if match_score is not None:
                        suggestions = suggest_improvements(resume_features, job_features)

                        st.session_state.applicant_result = {
                            "name": name,
                            "score": match_score,
                            "suggestions": suggestions
                        }
                        set_page("applicant_result")
                    else:
                        st.error("Insufficient content in resume or job description to calculate a match score.")
            else:
                st.error("Please fill in all fields.")
    with col2:
        if st.button("Back to Home"):
            set_page("home")

# Applicant Result Page
def applicant_result_page():
    add_bg_image("C:/Users/acer/PycharmProjects/resumemyself/images/background 2.jpg")
    st.markdown("<h1 style='color:white; text-align:center;'>Application Result</h1>", unsafe_allow_html=True)

    if 'applicant_result' in st.session_state:
        result = st.session_state.applicant_result
        st.write(f"Candidate Name: {result['name']}")
        st.write(f"Your match score: {result['score']:.1f}/10")

        if result['score'] >= 7.0:
            st.success("You are a good fit for this job!")
        else:
            missing_skills = ', '.join(result['suggestions']['missing_skills'])
            st.warning(f"Sorry!you are unfit for this job role. Consider improving the following skills: {missing_skills}")

        if st.button("Back to Applicant Form"):
            set_page("applicant_form")
    else:
        st.error("No result found. Please submit the application first.")

# Company Login Page
def company_login_page():
    add_bg_image("C:/Users/acer/PycharmProjects/resumemyself/images/analyres.jpg")
    st.markdown("<h1 style='color:white; text-align:center;'>Company Login</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        label {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    company_id = st.text_input("Company ID", type="password")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if company_id == COMPANY_ID and password == COMPANY_PASSWORD:
            set_page("company_dashboard")
        else:
            st.error("Invalid credentials. Please try again.")

# Company Dashboard Page
def company_dashboard_page():
    add_bg_image("C:/Users/acer/PycharmProjects/resumemyself/images/background 2.jpg")
    st.markdown("<h1 style='color:white; text-align:center;'>Company Dashboard</h1>", unsafe_allow_html=True)

    job_description = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"], key="job_desc_upload")
    resumes = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if job_description and resumes:
        job_text = extract_text_from_file(job_description)
        if job_text == "Empty":
            st.error("Job Description is empty or contains only invalid content.")
        else:
            job_features = extract_features(job_text)

            results = []
            for idx, resume in enumerate(resumes):
                try:
                    resume_text = extract_text_from_file(resume)
                    if resume_text != "Empty":
                        resume_features = extract_features(resume_text)
                        match_score = match_resume_with_job(resume_features, job_features)

                        if match_score is not None:
                            candidate_name = extract_name_with_fonts(resume) or extract_candidate_name(resume_text)
                            eligibility = "Eligible" if match_score >= 7 else "Not Eligible"
                            results.append({
                                "Serial Number": idx + 1,
                                "Candidate Name": candidate_name,
                                "Score": f"{match_score:.1f}",
                                "Rank": "",
                                "Eligibility Status": eligibility
                            })
                        else:
                            st.error(f"Error calculating match score for resume {idx + 1}")
                    else:
                        st.error(f"Resume {idx + 1} contains no valid content.")
                except Exception as e:
                    st.error(f"Error processing resume {idx + 1}: {e}")

            # Rank candidates
            results = sorted(results, key=lambda x: float(x["Score"]), reverse=True)
            for i, result in enumerate(results):
                result["Rank"] = i + 1

            # Display results as a table
            df = pd.DataFrame(results)
            st.dataframe(df)

# Main Page Routing
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "applicant_form":
    applicant_form()
elif st.session_state.page == "applicant_result":
    applicant_result_page()
elif st.session_state.page == "company_login":
    company_login_page()
elif st.session_state.page == "company_dashboard":
    company_dashboard_page()
