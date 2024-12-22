from sklearn.feature_extraction.text import TfidfVectorizer
def extract_features():
    return None
# utils.py



def extract_features(resume_text, job_text):
    """
    Extract features from the resume and job description text.
    Here, we simply concatenate the resume and job description for vectorization.
    """
    combined_text = resume_text + " " + job_text
    return combined_text
