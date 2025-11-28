import streamlit as st
import requests
import json
import time
import os
import tempfile
import sys
import re

# Import Whisper model loading
try:
    import whisper
except ImportError:
    st.error("The 'openai-whisper' library is not installed. Please install it using 'pip install openai-whisper'.")
    st.stop()


# --- Configuration ---
# 1. Load API Key securely from st.secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🚨 API Key Error: Please set 'GEMINI_API_KEY' in your Streamlit secrets file.")
    st.stop()

# Check for placeholder key
if API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not API_KEY:
    st.error("🚨 Configuration Error: The API Key is still set to the placeholder value or is missing. Please update it in your Streamlit secrets.")
    st.stop()

# 2. Use a stable model name and construct the API URL
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# --- Utility Functions ---

def retry_fetch(url, payload, headers, max_retries=3):
    """Fetches API response with exponential backoff."""
    delay = 1
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise exception for 4xx or 5xx errors
            return response.json()
        except requests.exceptions.RequestException as e:
            # Check for 400 Bad Request specifically using the response object
            if response is not None and response.status_code == 400:
                 error_message = f"API Call Failed (400 Bad Request). Raw response: {response.text}"
                 st.error(error_message)
                 # Stop retrying if we hit a known structure error
                 return None
                 
            if attempt < max_retries - 1:
                st.toast(f"API Call Attempt {attempt+1} failed. Retrying in {delay}s. (Error: {e})", icon='⚠️')
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"API Call Failed after {max_retries} attempts. Last error: {e}")
                return None
    return None


@st.cache_resource
def load_whisper_model():
    """Load the Whisper model once and cache it for efficiency."""
    st.info("Loading Whisper 'base.en' model... (This initialization happens only once)")
    try:
        return whisper.load_model("base.en")
    except Exception as e:
        st.error(f"Failed to load Whisper model. Error: {e}")
        st.error("Hint: Ensure the PyTorch/Whisper dependencies are correctly installed via requirements.txt.")
        return None

def transcribe_video_with_whisper(uploaded_file):
    """Transcribes the audio from the uploaded video or audio file using OpenAI Whisper."""
    model = load_whisper_model()
    if model is None:
        return None
        
    temp_path = None
    try:
        file_suffix = os.path.splitext(uploaded_file.name)[1]
        
        # Ensure the file pointer is at the start before writing
        uploaded_file.seek(0) 
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        st.markdown(f"Running Whisper on file: **{uploaded_file.name}**...")
        
        # Whisper requires FFmpeg to extract audio from video/process various audio files
        result = model.transcribe(temp_path, language="en", fp16=False) 
        
        # Clean up the transcript: strip whitespace 
        transcript = result.get("text", "").strip()
        
        if len(transcript) < 50:
             st.warning("Whisper completed, but the extracted transcript is very short. This might mean the file had no audible speech.")
             # Return None if transcript is too short to summarize meaningfully
             return None 

        return transcript
            
    except Exception as e:
        st.error(f"Whisper Transcription Failed. This is typically due to a missing FFmpeg dependency or an unusual file format.")
        st.exception(e) 
        return None
            
    finally:
        # Clean up the temporary file regardless of success or failure
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def summarize_text(transcript_text):
    """
    Summarizes the transcript using the Gemini AI model.
    Uses the corrected payload structure: systemInstruction is top-level.
    """
    if not transcript_text or transcript_text.isspace():
        st.error("Cannot summarize: Transcript content is empty or contains only spaces.")
        return "Summarization failed: Empty transcript content."
        
    st.info("Step 2/2: Sending transcript to Gemini AI for summarization...")
    
    system_prompt = (
        "You are a professional summarizer. Your task is to analyze the following English transcript "
        "and extract the 5 most critical discussion points, concepts, or decisions. Present the output using clear, concise bullet points."
    )

    user_query = f"Please summarize the following transcript:\n\n---\n\n{transcript_text}"
    
    # CORRECTED PAYLOAD STRUCTURE: systemInstruction is a top-level key.
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_query}]}
        ],
        "systemInstruction": system_prompt 
    }
    
    headers = {'Content-Type': 'application/json'}
    
    response_data = retry_fetch(API_URL, payload, headers)
    
    if response_data and 'candidates' in response_data:
        try:
            summary_text = response_data['candidates'][0]['content']['parts'][0]['text']
            return summary_text
        except (IndexError, KeyError):
            st.error("Gemini API returned an unexpected response format.")
            st.json(response_data) 
            return "Summarization failed due to invalid API response format."
            
    return "Summarization failed. Check the error messages above."


# --- Streamlit UI ---
st.set_page_config(page_title="Universal Video/Audio Summarizer", layout="centered")

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


st.markdown('<h1 class="main-header">🎙️ Universal Video/Audio Summarizer (Whisper + Gemini)</h1>', unsafe_allow_html=True)
st.warning("⚠️ **Cloud Deployment Note:** The ability to process all file types relies on the FFmpeg system package being installed on the server (via `packages.txt`).")
st.write("Upload **any** English video or audio file (MP4, MOV, MP3, WAV, MKV, FLAC, etc.) to generate a full transcript and a key point summary. Perfect for meeting recordings or lecture videos.")

# File Uploader - Expanded file types for universality
ALL_MEDIA_TYPES = [
    "mp4", "mov", "wav", "mp3", "m4a", "mkv", "avi", "flv", "wmv", 
    "ogg", "flac", "wma", "aac", "aiff", "webm"
]

uploaded_file = st.file_uploader(
    "Upload Video or Audio File",
    type=ALL_MEDIA_TYPES,
    accept_multiple_files=False
)

if uploaded_file is not None:
    st.success(f"File uploaded successfully: **{uploaded_file.name}**")
    
    if st.button("Generate Transcript and Summary"):
        
        # 1. Transcription Step
        with st.spinner("Step 1/2: Generating Transcript using Whisper AI... (Processing audio takes time)"):
            transcript = transcribe_video_with_whisper(uploaded_file)
            
            if transcript is not None:
                st.subheader("📝 Extracted Transcript")
                
                # Show the long transcript in an expander
                with st.expander("Click to view full transcript text"):
                    if transcript:
                         st.code(transcript, language="text")
                    else:
                         st.warning("Transcript is empty or blank. Cannot proceed to summarization.")

                # 2. Summarization Step (Only runs if a valid transcript was returned)
                if transcript:
                    summary = summarize_text(transcript)
                    
                    st.subheader("✅ Summary (Generated by Gemini)")
                    st.markdown(summary)
                    
                    if not summary.startswith("Summarization failed"):
                        st.success("Process complete: Transcript generated and Summary extracted.")
                    else:
                        st.error("Process failed during summarization. Check error details above.")
                
            else:
                st.error("Transcription failed. Please ensure the audio quality is clear, the file is valid, and FFmpeg is properly installed on the server.")
