import streamlit as st
from back import BackgroundCSSGenerator


st.set_page_config(page_title="AI DocMate Prototype", layout="wide")
st.title("📄 AI DocMate – GenAI-Powered Document Workflow")

img1_path = r"C:\Users\praje\OneDrive\AppData\Desktop\project\JobAI\img2.jpg"
img2_path = r"C:\Users\praje\OneDrive\AppData\Desktop\project\JobAI\img2.jpg"
background_generator = BackgroundCSSGenerator(img1_path, img2_path)
page_bg_img = background_generator.generate_background_css()
st.markdown(page_bg_img, unsafe_allow_html=True)



# Custom CSS for modern styling and professional font
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        font-size: 17px;
        line-height: 1.6;
        color: #2E2E2E;
    }
    h1, h2, h3, h4, h5 {
        font-family: 'Segoe UI Semibold', sans-serif;
        color: #1A1A1A;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    .css-1v0mbdj { background-color: #f9f9fb !important; }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .metric-container {
        background-color: #fff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        padding: 0.6rem 1.6rem;
        border-radius: 10px;
        border: none;
        font-size: 17px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        font-size: 16px;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stRadio>div>label, .stSelectbox>div>div {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar Navigation ---
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Jump to Section", [
    "🏁 Upload Document",
    "📥 Ingestor Agent",
    "🧾 Extractor Agent",
    "🧠 Classifier Agent",
    "🚚 Router Agent",
    "📊 Admin Dashboard"
])

# --- Upload Section ---
if section == "🏁 Upload Document":
    st.subheader("📤 Upload or Ingest Document")
    col1, col2 = st.columns(2)
    with col1:
        st.file_uploader("Drag & drop your document", type=["pdf", "jpg", "png", "docx"])
    with col2:
        st.text_input("📧 IMAP Email Hook", "docs@company.com")
        st.text_input("📂 Folder Path (Dropbox/Drive)", "/shared/folder")
        st.button("🚀 Simulate Ingestion")

# --- Ingestor Agent ---
elif section == "📥 Ingestor Agent":
    st.subheader("📥 Ingestor Agent")
    st.info("This agent captures files, extracts metadata, and stores them.")
    st.code("""def ingest(file):
    # capture metadata and file path
    return doc_id""")
    st.button("Trigger Metadata Extraction")

# --- Extractor Agent ---
elif section == "🧾 Extractor Agent":
    st.subheader("🧾 Text Extractor Agent")
    st.success("Uses PyMuPDF & OCR to extract content from PDFs & images.")
    st.code("""def extract_text(file_path):
    # OCR + PDF text extraction
    return cleaned_text""")
    st.button("🧠 Simulate Text Extraction")

# --- Classifier Agent ---
elif section == "🧠 Classifier Agent":
    st.subheader("🧠 LLM Classifier Agent")
    st.warning("Groq LLM used to classify documents with confidence + explanation")
    st.code("""def classify(text):
    # LLM prompt to classify + explain
    return { 'type': 'Invoice', 'confidence': 0.92, 'reason': 'Contains invoice number + amount' }""")
    st.button("🔍 Simulate Classification")

# --- Router Agent ---
elif section == "🚚 Router Agent":
    st.subheader("🚚 Smart Router Agent")
    st.markdown("""
    - Uploads to **ImageKit**, sends **Slack** alert, or routes to **ERP**.
    - Includes fallback queue for failures.
    """)
    col1, col2 = st.columns(2)
    with col1:
        dest = st.selectbox("Route To", ["ImageKit", "ERP System", "Slack"])
    with col2:
        st.button("🚀 Simulate Routing")
    st.code("""def route(doc, destination):
    # API-based routing
    return status""")

# --- Admin Dashboard ---
elif section == "📊 Admin Dashboard":
    st.subheader("📊 Admin Dashboard")
    st.caption("Monitor document lifecycle, reclassify, and manage LLM prompts")

    doc_id = st.text_input("🔎 Search Document by ID")
    st.progress(0.75)

    col1, col2 = st.columns(2)
    with col1:
        st.radio("🔁 Manual Action", ["Re-classify", "Re-route", "Flag for Review"])
    with col2:
        st.text_area("🧠 Custom Prompt (LLM Override)", "Classify this document based on legal terms...")
        st.button("Submit Prompt")

    st.markdown("---")
    cols1, cols2,cols3 = st.columns(3)
    with cols1:
        st.metric(label="Docs Processed Today", value="154", delta="+8")
    with cols2:
        st.metric(label="Avg Confidence", value="93.6%", delta="+1.4")
    with cols3:
        st.metric(label="Manual Overrides", value="6", delta="-2")
