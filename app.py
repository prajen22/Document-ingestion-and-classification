import streamlit as st
from back import BackgroundCSSGenerator
import os
import tempfile
import fitz  # PyMuPDF
import uuid
from imagekitio import ImageKit
import requests
import hashlib
from PIL import Image
import pytesseract
from docx import Document
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
import easyocr
import pandas as pd
import plotly.express as px
from tinydb import TinyDB
import datetime
import random

from db_utils import insert_document_status  # or inline if in same file


reader = easyocr.Reader(['en'], gpu=False)

# === Document Workflow Configuration ===
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"
IMAGEKIT_PUBLIC_KEY = "public_D6gYyMPgt2Ie/28UYNm9izah2io="
IMAGEKIT_PRIVATE_KEY = "private_UplpBpqaxFOJCy+7htWPS6tdJnE="
IMAGEKIT_URL_ENDPOINT = "https://ik.imagekit.io/r68bqoqtkb"

imagekit = ImageKit(
    public_key=IMAGEKIT_PUBLIC_KEY,
    private_key=IMAGEKIT_PRIVATE_KEY,
    url_endpoint=IMAGEKIT_URL_ENDPOINT
)

import streamlit as st

st.set_page_config(
    page_title="AI DocMate Prototype – Smart Document Workflow",
    page_icon="🗃️",  # Non-white document workflow icon
    layout="wide",
    initial_sidebar_state="expanded"
)




st.sidebar.markdown("---")

st.title("📄 AI DocMate – GenAI-Powered Document Workflow")

img1_path = r"gif4.gif"
img2_path = r"img2.jpg"
background_generator = BackgroundCSSGenerator(img1_path, img2_path)
page_bg_img = background_generator.generate_background_css()
st.markdown(page_bg_img, unsafe_allow_html=True)

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
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        padding: 0.6rem 1.6rem;
        border-radius: 10px;
        font-size: 17px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

if "event_log" not in st.session_state:
    st.session_state.event_log = []

def log_event(stage, detail):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.event_log.append(f"[{timestamp}] {stage}: {detail}")

def upload_document_ui():
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "jpg", "png", "docx"])
    sender = st.text_input("Sender Email (for priority check)")
    if uploaded_file:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(tempfile.gettempdir(), f"{file_id}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Uploaded: {uploaded_file.name}")
        st.session_state["file_path"] = file_path
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["upload_timestamp"] = str(datetime.datetime.now())
        st.session_state["sender"] = sender
        log_event("Upload", f"File: {uploaded_file.name} from {sender}")
        return True
    return False

def ingest_file():
    st.subheader("📥 Ingestor Agent")
    if "file_path" in st.session_state:
        st.session_state["doc_id"] = str(uuid.uuid4())
        sender = st.session_state.get("sender", "").lower()
        priority = "High" if sender in ["ceo@company.com", "legal@company.com"] else "Normal"
        st.session_state["priority"] = priority
        st.success(f"Ingested Document ID: {st.session_state['doc_id']} | Priority: {priority}")
        log_event("Ingest", f"Document ID: {st.session_state['doc_id']} | Priority: {priority}")
        return True
    else:
        st.warning("Please upload a document first.")
        return False

def extract_text():
    st.subheader("🧾 Extractor Agent")
    if "file_path" in st.session_state:
        file_path = st.session_state["file_path"]
        text = ""
        metadata = {}

        if file_path.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                meta = doc.metadata
                metadata = {
                    "Title": meta.get("title", "N/A"),
                    "Author": meta.get("author", "N/A"),
                    "Creation Date": meta.get("creationDate", "N/A"),
                    "Modification Date": meta.get("modDate", "N/A"),
                    "Producer": meta.get("producer", "N/A"),
                    "Number of Pages": len(doc)
                }
                for page in doc:
                    content = page.get_text()
                    if content.strip():
                        text += content
                    else:
                        pix = page.get_pixmap()
                        temp_img = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.png")
                        pix.save(temp_img)
                        results = reader.readtext(temp_img, detail=0, paragraph=True)
                        text += "\n".join(results)

        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            core_props = doc.core_properties
            metadata = {
                "Title": core_props.title or "N/A",
                "Author": core_props.author or "N/A",
                "Created": str(core_props.created) or "N/A",
                "Last Modified By": core_props.last_modified_by or "N/A",
            }

        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            try:
                results = reader.readtext(file_path, detail=0, paragraph=True)
                text = "\n".join(results)
                metadata = {
                    "Type": "Image File",
                    "Dimensions": str(Image.open(file_path).size),
                }
            except Exception as e:
                st.error(f"OCR failed: {str(e)}")
                return False
        else:
            st.error("❌ Unsupported file type.")
            return False

        st.session_state["extracted_text"] = text
        st.session_state["metadata"] = metadata

        st.text_area("Extracted Text", text, height=200)
        with st.expander("📌 Metadata", expanded=False):
            for key, value in metadata.items():
                st.write(f"**{key}**: {value}")

        st.success("Text & Metadata extraction complete.")
        log_event("Extract", "Text and metadata extracted from document")
        return True
    else:
        st.warning("Please upload and ingest a document first.")
        return False


def classify_document():
    st.subheader("🧠 Classifier Agent – Groq LLM")
    if "extracted_text" not in st.session_state:
        st.warning("Please extract text first.")
        return False

    valid_folders = [
        "resume", "tender", "invoice", "report", "offer_letter",
        "agreement", "payslip", "legal", "purchase_order", "bank_statement",
        "aadhaar", "pan_card", "passport", "bill", "contract"
    ]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    classify_prompt = (
        "You are a document classification assistant. "
        "Classify the following document into one of the following folders ONLY:\n"
        f"{', '.join(valid_folders)}.\n"
        "Respond ONLY with the folder name in lowercase."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": classify_prompt},
            {"role": "user", "content": st.session_state["extracted_text"]}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        reply = result['choices'][0]['message']['content'].lower().strip()

        st.text_area("LLM Raw Classification", reply, height=100)

        if reply in valid_folders:
            st.session_state["classification_result"] = reply
            st.session_state["destination_folder"] = reply
            st.success(f"Document classified as: {reply}")
            log_event("Classify", f"Classified as {reply}")
        else:
            st.warning(f"LLM returned unknown folder: {reply}")

        override = st.checkbox("Not satisfied with classification? Check to override")
        if override:
            manual_label = st.selectbox("Manually select correct folder:", valid_folders)
            if manual_label:
                st.session_state["destination_folder"] = manual_label
                explain_prompt = (
                    f"You are an assistant. Based on the content below, justify why the document is best suited for folder: '{manual_label}'. "
                    "Also, provide a confidence score (0–100)."
                )

                explain_payload = {
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": explain_prompt},
                        {"role": "user", "content": st.session_state["extracted_text"]}
                    ]
                }

                explain_response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=explain_payload)

                if explain_response.status_code == 200:
                    explanation = explain_response.json()['choices'][0]['message']['content']
                    st.text_area("📄 LLM Explanation for Manual Classification", explanation, height=200)
                    log_event("Manual Classification", f"Overridden to {manual_label} with LLM explanation")
                    st.success(f"✅ Document manually classified as: {manual_label}")
                else:
                    st.error("❌ Could not fetch explanation from LLM")

        return True
    else:
        st.error("❌ Classification failed.")
        return False

import io
import zipfile
import uuid
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

from elasticsearch import Elasticsearch
import uuid
import io
import zipfile
import streamlit as st
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from elasticsearch import Elasticsearch
ELASTICSEARCH_URL = "https://my-elasticsearch-project-a9cf9b.es.us-central1.gcp.elastic.cloud:443"
API_KEY = "b0pzTG9wZ0J3eXhEZHo1TVlOdE46THRYTVlIaFQyMzAyOV9NMC0yOG5jUQ=="
es = Elasticsearch(
    ELASTICSEARCH_URL,
    api_key=API_KEY
)

def route_document():
    st.subheader("🚚 Router Agent")
    if "file_path" in st.session_state and "destination_folder" in st.session_state:
        file_path = st.session_state["file_path"]
        file_name = st.session_state["file_name"]
        destination_folder = st.session_state["destination_folder"]
        doc_id = st.session_state.get("doc_id", str(uuid.uuid4()))
        sender = st.session_state.get("sender", "unknown")

        upload_success = False
        upload_url = ""
        max_retries = 3

        # Create compressed ZIP in memory
        compressed_buffer = io.BytesIO()
        with zipfile.ZipFile(compressed_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(file_path, arcname=file_name)
        compressed_buffer.seek(0)  # Reset pointer to start

        compressed_name = file_name.rsplit(".", 1)[0] + ".zip"

        for attempt in range(max_retries):
            try:
                options = UploadFileRequestOptions(
                    use_unique_file_name=True,
                    folder=f"/{destination_folder}"
                )

                response = imagekit.upload_file(
                    file=compressed_buffer.getvalue(),
                    file_name=compressed_name,
                    options=options
                )

                if isinstance(response, dict) and "url" in response:
                    upload_url = response["url"]
                    upload_success = True
                    break
                elif hasattr(response, "url"):
                    upload_url = response.url
                    upload_success = True
                    break

            except Exception as e:
                st.warning(f"Attempt {attempt+1} failed: {e}")

        if upload_success:
            st.session_state["imagekit_url"] = upload_url
            st.success(f"✅ Uploaded compressed file to ImageKit in folder {destination_folder}")
            log_event("Route", f"Uploaded compressed file to ImageKit folder: {destination_folder}")

            # Store metadata in database
            insert_document_status(
                doc_id=doc_id,
                file_name=compressed_name,
                folder=destination_folder,
                url=upload_url,
                status="success",
                sender=sender
            )

            # --- Elasticsearch Indexing ---
            try:
                # Optional: Extract text from original file (if PDF, DOCX, etc.)
                file_text = ""  
                try:
                    import fitz  # PyMuPDF for PDFs
                    if file_path.lower().endswith(".pdf"):
                        doc = fitz.open(file_path)
                        file_text = "\n".join(page.get_text() for page in doc)
                except Exception as e:
                    st.warning(f"Text extraction failed: {e}")

                es.index(
                    index="documents",
                    id=doc_id,
                    document={
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "folder": destination_folder,
                        "url": upload_url,
                        "sender": sender,
                        "content": file_text,
                        "tags": [],
                    }
                )
                st.success("📂 Document indexed in Elasticsearch for search.")
            except Exception as e:
                st.error(f"❌ Elasticsearch indexing failed: {e}")

            return True
        else:
            st.error("❌ All upload attempts failed.")

            insert_document_status(
                doc_id=doc_id,
                file_name=compressed_name,
                folder=destination_folder,
                url="N/A",
                status="fail",
                sender=sender
            )
            return False
    else:
        st.warning("Classification or file missing.")
        return False


INDEX_NAME = "documents"  # Your Elasticsearch index
def summarize_from_es():
    """
    Gets a user query from Streamlit input,
    searches Elasticsearch for relevant content,
    and summarizes it using Groq LLM.
    """
    st.subheader("💬 Document Summarizer")

    # Get query from Streamlit input
    query = st.text_input("Enter your question:")

    if not query:
        return

    try:
        #
        # Search in Elasticsearch
        search_body = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": 1
        }
        res = es.search(index=INDEX_NAME, body=search_body)

        if res["hits"]["total"]["value"] == 0:
            st.warning("No matching documents found.")
            return

        best_doc = res["hits"]["hits"][0]["_source"]["content"]

        # Prepare Groq request
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = f"Summarize the following text relevant to the query:\n\nDocument:\n{best_doc}\n\nUser Query:\n{query}"
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes relevant text from documents."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }

        # Call Groq API
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            summary = response.json()["choices"][0]["message"]["content"].strip()
            
            # Styled box using st.markdown with HTML
            st.markdown(
    f"""
    <div style="
        padding: 15px; 
        border: 2px solid #1E3A8A; 
        border-radius: 10px; 
        background-color: #1E3A8A; 
        color: white;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    ">
        <strong>📄 Summary:</strong><br>{summary}
    </div>
    """,
    unsafe_allow_html=True
)

        else:
            st.error(f"Groq API request failed: {response.status_code} {response.text}")


    except Exception as e:
        st.error(f"Error: {e}")



st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Jump to Section", [
    "🏁 Upload Document",
    "📥 Ingestor Agent",
    "🧾 Extractor Agent",
    "🧠 Classifier Agent",
    "🚚 Router Agent",
    "📊 Admin Dashboard",
    "💬 chatbot",
    "📂 search and summarize from elasticsearch"
])


from tinydb import TinyDB
import json
import streamlit as st
from groq import Groq
import os



from tinydb import TinyDB
import json
import streamlit as st
import requests

def chatbot_from_db():
    """
    Chatbot function that loads data from TinyDB and answers user queries using Groq API.
    """
    st.subheader("💬 Document Query Chatbot")

    # Load the database
    db = TinyDB("document_status.json")
    docs = db.all()

    if not docs:
        st.warning("📂 No documents found in the database.")
        return

    # Convert DB content into a single text block
    db_text = json.dumps(docs, indent=2)

    # User input
    user_query = st.text_input("Ask me anything about the uploaded documents:")

    if user_query:
        try:
            # Use the same headers style as classify_document
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",  # using same global var
                "Content-Type": "application/json"
            }

            # Create the prompt
            prompt = f"""
            You are a helpful assistant. The following is the document metadata from the database:

            {db_text}

            Based on this information, answer the user's query:

            User: {user_query}
            """

            payload = {
                "model": GROQ_MODEL,  # also reuse same variable
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant for answering document status queries and answer for all the questions."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 500
            }

            # API call
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"].strip()
                st.markdown(f"**🤖 Chatbot:** {answer}")
            else:
                st.error(f"❌ API request failed: {response.status_code} {response.text}")

        except Exception as e:
            st.error(f"❌ Error communicating with Groq: {e}")



if section == "🏁 Upload Document":
    upload_document_ui()
elif section == "📥 Ingestor Agent":
    ingest_file()
elif section == "🧾 Extractor Agent":
    extract_text()
elif section == "🧠 Classifier Agent":
    classify_document()
elif section == "🚚 Router Agent":

    
    route_document()

elif section == "💬 chatbot":
    chatbot_from_db()

elif section == "📂 search and summarize from elasticsearch":
    summarize_from_es()
elif section == "📊 Admin Dashboard":


    st.subheader("📊 Admin Dashboard")

    db = TinyDB("document_status.json")
    docs = db.all()

    if not docs:
        st.warning("No document data available yet.")
    else:
        df = pd.DataFrame(docs)

        # === Metrics ===
        total_docs = len(df)
        successful_docs = df[df['status'].str.strip().str.lower() == 'success'].shape[0]

        failed_docs = df[df['status'] == 'Failed'].shape[0]
        unique_folders = df['folder'].nunique()

        row1 = st.columns(4)
        with row1[0]:
            st.metric("📄 Total Documents", total_docs)
        with row1[1]:
            st.metric("✅ Successful Uploads", successful_docs)
        with row1[2]:
            st.metric("❌ Failed Uploads", failed_docs)
        with row1[3]:
            st.metric("🗂️ Unique Classifications", unique_folders)


        st.markdown("---------------------------------------------------------------------")

        # === Charts ===
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Clean and count document types
            if 'folder' in df.columns:
                df['folder'] = df['folder'].str.strip().str.lower()  # clean column values
                class_counts = df['folder'].value_counts().reset_index()
                class_counts.columns = ['Document Type', 'Count']


                # Chart selector UI
                chart_type = st.selectbox("📈 Select Chart Type", ["Bar", "Horizontal Bar", "Funnel"], key="chart_type_doc")

                # Generate chart
                if chart_type == "Bar":
                    fig = px.bar(
                        class_counts,
                        x='Document Type',
                        y='Count',
                        color='Document Type',
                        text='Count',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                elif chart_type == "Horizontal Bar":
                    fig = px.bar(
                        class_counts,
                        y='Document Type',
                        x='Count',
                        orientation='h',
                        color='Document Type',
                        text='Count',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                elif chart_type == "Funnel":
                    fig = px.funnel(
                        class_counts,
                        x='Count',
                        y='Document Type',
                        color='Document Type'
                    )

                # Layout and render
                fig.update_layout(
                    title="📂 Document Type Distribution",
                    xaxis_title="Count",
                    yaxis_title="Document Type",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No classification data available in the database.")



        with col_chart2:
            if "folder" in df.columns and not df["folder"].isnull().all():
                df['folder'] = df['folder'].str.strip().str.title()
                folder_counts = df['folder'].value_counts().reset_index()
                folder_counts.columns = ['Folder', 'Count']

                viz_type = st.selectbox("📂 Choose Visualization Type", ["Treemap", "Sunburst", "Strip Plot"], key="folder_chart")

                if viz_type == "Treemap":
                    fig2 = px.treemap(
                        folder_counts,
                        path=['Folder'],
                        values='Count',
                        color='Count',
                        color_continuous_scale='Aggrnyl',
                        title="📁 Treemap of Documents by Folder"
                    )
                elif viz_type == "Sunburst":
                    fig2 = px.sunburst(
                        folder_counts,
                        path=['Folder'],
                        values='Count',
                        color='Count',
                        color_continuous_scale='Blues',
                        title="📁 Sunburst Chart: Document Types"
                    )
                elif viz_type == "Strip Plot":
                    fig2 = px.strip(
                        folder_counts,
                        x='Folder',
                        y='Count',
                        color='Folder',
                        title="📁 Strip Plot of Document Counts",
                        stripmode='overlay'
                    )

                fig2.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("📭 No folder classification data available.")

        # === Document Table ===
        st.subheader("📁 Full Upload History")
        df_display = df[["timestamp", "file_name", "folder", "status", "url"]].copy()
        df_display.columns = ["📅 Timestamp", "📄 File", "📂 Folder", "✅ Status", "🔗 URL"]
        st.dataframe(df_display.sort_values(by="📅 Timestamp", ascending=False), use_container_width=True)


