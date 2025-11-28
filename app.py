import streamlit as st
import requests
import json
import time
import os
import tempfile
import sys

# Import Whisper model loading
try:
    import whisper
except ImportError:
    st.error("The 'openai-whisper' library is not installed. Please install it using 'pip install openai-whisper'.")
    st.stop()


# --- Configuration ---
# 1. Load API Key securely from st.secrets (from .streamlit/secrets.toml)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🚨 API Key Error: Please set 'GEMINI_API_KEY' in your Streamlit secrets file.")
    st.stop()

# 2. Use a stable model name and construct the API URL
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# --- Utility Functions ---

def retry_fetch(url, payload, headers, max_retries=3):
    """Fetches API response with exponential backoff."""
    delay = 1
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise exception for 4xx or 5xx errors
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Log the warning but don't stop the process
                st.toast(f"API Call Attempt {attempt+1} failed. Retrying in {delay}s. (Error: {e})", icon='⚠️')
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"API Call Failed after {max_retries} attempts: {e}")
                return None
    return None


@st.cache_resource
def load_whisper_model():
    """Load the Whisper model once and cache it for efficiency."""
    st.info("Loading Whisper 'base.en' model... (This initialization happens only once)")
    # Set fp16=False for reliability on CPU-only machines
    try:
        return whisper.load_model("base.en")
    except Exception as e:
        st.error(f"Failed to load Whisper model. Error: {e}")
        st.error("Hint: This may be due to a dependency issue with PyTorch or a missing FFmpeg package on the server.")
        return None

def transcribe_video_with_whisper(uploaded_file):
    """
    Transcribes the audio from the uploaded video file using OpenAI Whisper.
    Uses tempfile.NamedTemporaryFile for safer handling of uploaded bytes.
    """
    model = load_whisper_model()
    if model is None:
        return None
        
    temp_path = None
    try:
        # 1. Save uploaded file to a temporary disk path
        file_suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # 2. Run Whisper on the temporary file
        st.markdown(f"Running Whisper on file: **{uploaded_file.name}**...")
        
        # Using the base.en model and setting fp16=False for stability
        result = model.transcribe(temp_path, language="en", fp16=False) 
        transcript = result["text"]
        return transcript
            
    except Exception as e:
        # If FFmpeg is missing or improperly installed, this exception is caught here.
        st.error(f"Whisper Transcription Failed. This is often due to missing FFmpeg dependency.")
        st.error(f"Error Details: {e}")
        return None
            
    finally:
        # 3. Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def summarize_text(transcript_text):
    """
    Summarizes the English transcript using the Gemini AI model.
    Uses the corrected payload structure (contents + config).
    """
    st.info("Step 2/2: Sending transcript to Gemini AI for summarization...")
    
    # System Instruction to guide the model's output format and persona
    system_prompt = (
        "You are a professional video summarizer. Your task is to analyze the following English video transcript "
        "and extract the 5 most critical learning points, concepts, or steps discussed. Present the output using clear, concise bullet points."
    )

    # User Query containing the transcript
    user_query = f"Please summarize the following video transcript:\n\n---\n\n{transcript_text}"
    
    # CORRECTED PAYLOAD STRUCTURE
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_query}]}
        ],
        "config": {
            # System Instruction must be passed inside the config block
            "systemInstruction": system_prompt
        }
    }
    
    headers = {'Content-Type': 'application/json'}
    
    # Call the API using the retry mechanism
    response_data = retry_fetch(API_URL, payload, headers)
    
    if response_data and 'candidates' in response_data:
        try:
            summary_text = response_data['candidates'][0]['content']['parts'][0]['text']
            return summary_text
        except (IndexError, KeyError):
            st.error("Gemini API returned an unexpected response format.")
            return "Summarization failed due to invalid API response format."
            
    return "Summarization failed. Check the error messages above for connection or request issues."


# --- Streamlit UI ---
st.set_page_config(page_title="Real-Time English Video Summarizer", layout="centered")

# Custom CSS styling
st.markdown("""
<style>
    /* Custom Styling for aesthetics */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        transition: background-color 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .main-header {
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎙️ Real-Time English Video Summarizer (Whisper + Gemini)</h1>', unsafe_allow_html=True)
st.warning("⚠️ **Note:** For cloud deployment, the app relies on FFmpeg being installed on the server (e.g., via `packages.txt`).")
st.write("Upload an English video or audio recording (MP4, MOV, WAV, MP3) to generate a full transcript and a key point summary.")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Video or Audio File",
    type=["mp4", "mov", "wav", "mp3", "m4a"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    st.success(f"File uploaded successfully: **{uploaded_file.name}**")
    
    if st.button("Generate Transcript and Summary"):
        
        # 1. Transcription Step
        with st.spinner("Step 1/2: Generating Transcript using Whisper AI... (Processing audio takes time)"):
            transcript = transcribe_video_with_whisper(uploaded_file)
            
            if transcript:
                st.subheader("📝 Extracted Transcript")
                
                # Show the long transcript in an expander
                with st.expander("Click to view full transcript text"):
                    st.code(transcript, language="text")

                # 2. Summarization Step
                summary = summarize_text(transcript)
                
                st.subheader("✅ Summary (Generated by Gemini)")
                st.markdown(summary)
                
                if not summary.startswith("Summarization failed"):
                    st.success("Process complete: Transcript generated and Summary extracted.")
                else:
                    st.error("Process failed during summarization. Check error details above.")
            else:
                st.error("Cannot proceed to summarization. Transcription failed.")