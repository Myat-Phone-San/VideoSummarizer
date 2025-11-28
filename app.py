import streamlit as st
import os
import tempfile
import sys

# --- Gemini SDK Imports ---
from google import genai
from google.genai import types
from google.genai.errors import APIError # For handling specific Gemini errors

# Import Whisper model loading
try:
    import whisper
except ImportError:
    st.error("The 'openai-whisper' library is not installed. Please install it using 'pip install openai-whisper'.")
    st.stop()


# --- Configuration and Client Initialization ---
# 1. Load API Key securely from st.secrets and initialize the client
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("🚨 API Key Error: Please set 'GEMINI_API_KEY' in your Streamlit secrets file.")
    st.stop()

# Initialize the Gemini Client globally (Streamlit handles resource caching)
try:
    # Use the SDK client for API calls
    client = genai.Client(api_key=API_KEY) 
except Exception as e:
    st.error(f"Error initializing AI client. Details: {e}")
    st.stop()
    
# Model name used for text generation
MODEL_NAME = "gemini-2.5-flash"


# --- Utility Functions (HTTP/Retry code removed) ---

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
    """
    model = load_whisper_model()
    if model is None:
        return None
        
    temp_path = None
    try:
        # 1. Save uploaded file to a temporary disk path
        file_suffix = os.path.splitext(uploaded_file.name)[1]
        
        # Ensure the file pointer is at the start before writing
        uploaded_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # 2. Run Whisper on the temporary file
        st.markdown(f"Running Whisper on file: **{uploaded_file.name}**...")
        
        # Using the base.en model and setting fp16=False for stability
        result = model.transcribe(temp_path, language="en", fp16=False) 
        transcript = result["text"].strip()
        
        if len(transcript) < 50:
            st.warning("Whisper completed, but the extracted transcript is very short. Cannot proceed to summarization.")
            return None

        return transcript
            
    except Exception as e:
        st.error(f"Whisper Transcription Failed. This is often due to missing FFmpeg dependency.")
        st.error(f"Error Details: {e}")
        return None
            
    finally:
        # 3. Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def summarize_text(transcript_text):
    """
    Summarizes the English transcript using the Gemini AI SDK client.
    Handles system instruction using GenerateContentConfig.
    """
    if not transcript_text or transcript_text.isspace():
        return "Summarization failed: Empty transcript content."
        
    st.info("Step 2/2: Sending transcript to Gemini AI for summarization...")
    
    # Define the System Instruction
    system_prompt = (
        "You are a professional video summarizer. Your task is to analyze the following English video transcript "
        "and extract the 5 most critical learning points, concepts, or steps discussed. Present the output using clear, concise bullet points."
    )

    # Define the User Query
    user_query = f"Please summarize the following video transcript:\n\n---\n\n{transcript_text}"
    
    # --- FIX APPLIED HERE ---
    # The SDK automatically wraps the string into types.Content(parts=[types.Part.from_text(...)])
    contents = [user_query]
    # -------------------------
    
    # 2. Prepare the configuration (including system instruction)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
    )
    
    try:
        # 3. Call the SDK's generate_content method
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents, # Now uses the simpler, fixed content definition
            config=config,
        )
        
        return response.text
        
    except APIError as e:
        # Handle specific API errors (e.g., rate limiting, bad request)
        st.error(f"Gemini API Call Failed (SDK Error): {e}")
        return "Summarization failed due to API connection error."
    except Exception as e:
        # Handle other unexpected errors
        st.error(f"An unexpected error occurred during summarization: {e}")
        return "Summarization failed due to an unexpected error."


# --- Streamlit UI ---
st.set_page_config(page_title="Universal Video/Audio Summarizer (SDK)", layout="centered")

# Custom CSS styling (Kept for aesthetics)
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


st.markdown('<h1 class="main-header">🎙️ Universal Video/Audio Summarizer (Whisper + Gemini SDK)</h1>', unsafe_allow_html=True)
st.warning("⚠️ **Cloud Deployment Note:** The ability to process all file types relies on the FFmpeg system package being installed on the server (via `packages.txt`).")
st.write("Upload **any** English video or audio file (MP4, MOV, MP3, WAV, etc.) to generate a full transcript and a key point summary. Now using the official Google GenAI SDK.")

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
