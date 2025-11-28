import streamlit as st
import os
import tempfile
import sys

# --- Gemini SDK Imports ---
from google import genai
from google.genai import types
from google.genai.errors import APIError 

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

# Initialize the Gemini Client globally
try:
    # Use the SDK client for API calls
    client = genai.Client(api_key=API_KEY)  
except Exception as e:
    st.error(f"Error initializing AI client. Details: {e}")
    st.stop()
    
# Model name used for text generation
MODEL_NAME = "gemini-2.5-flash"


# --- Core Functions ---

@st.cache_resource
def load_whisper_model():
    """
    Load the multilingual Whisper model once and cache it for efficiency.
    Using 'small' for better multilingual accuracy.
    """
    # CHANGED: Switched from 'base.en' to 'small' for multilingual support
    st.info("Loading Whisper 'small' multilingual model... (This initialization happens only once)")
    # Set fp16=False for reliability on CPU-only machines
    try:
        return whisper.load_model("small")
    except Exception as e:
        st.error(f"Failed to load Whisper model. Error: {e}")
        st.error("Hint: This may be due to a dependency issue with PyTorch or a missing FFmpeg package on the server.")
        return None

def transcribe_video_with_whisper(uploaded_file):
    """
    Transcribes the audio from the uploaded video file using OpenAI Whisper.
    Auto-detects the language. Returns a tuple: (transcript, detected_lang_code)
    """
    model = load_whisper_model()
    if model is None:
        return None
        
    temp_path = None
    try:
        # 1. Save uploaded file to a temporary disk path
        file_suffix = os.path.splitext(uploaded_file.name)[1]
        uploaded_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # 2. Run Whisper on the temporary file
        st.markdown(f"Running Whisper on file: **{uploaded_file.name}**...")
        
        # CHANGED: Removed language="en" to enable auto-detection (multilingual)
        result = model.transcribe(temp_path, fp16=False)  
        transcript = result["text"].strip()
        detected_lang_code = result["language"]
        
        # Friendly display name for the detected language
        if detected_lang_code == 'my':
             detected_lang_name = "Burmese (Myanmar) 🇲🇲"
        elif detected_lang_code == 'en':
             detected_lang_name = "English 🇬🇧"
        else:
             detected_lang_name = f"Other Language (Code: {detected_lang_code})"

        st.success(f"Language Detected: **{detected_lang_name}**")

        if len(transcript) < 50:
            st.warning("Whisper completed, but the extracted transcript is very short. Cannot proceed to summarization.")
            return None

        # Return the transcript and the language code
        return transcript, detected_lang_code
            
    except Exception as e:
        st.error(f"Whisper Transcription Failed. This is often due to missing FFmpeg dependency.")
        st.error(f"Error Details: {e}")
        return None
            
    finally:
        # 3. Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def summarize_text(transcript_text, detected_lang_code):
    """
    Summarizes the transcript using the Gemini AI SDK client.
    The system prompt is dynamically set to handle the detected language.
    """
    if not transcript_text or transcript_text.isspace():
        return "Summarization failed: Empty transcript content."
        
    st.info("Step 2/2: Sending transcript to Gemini AI for summarization...")
    
    # Dynamic prompt based on detected language
    if detected_lang_code == 'my':
        lang_prompt = "The transcript is in Burmese (Myanmar). Translate the key points into English and then summarize."
    else:
        lang_prompt = "The transcript is in English."
    
    # Define the System Instruction
    system_prompt = (
        f"You are a professional video summarizer. {lang_prompt} Your task is to analyze the following "
        "video transcript and extract the 5 most critical learning points, concepts, or steps discussed. "
        "Present the final output using clear, concise bullet points, **in English**."
    )

    # Define the User Query
    user_query = f"Please summarize the following video transcript:\n\n---\n\n{transcript_text}"
    contents = [user_query]
    
    # Prepare the configuration (including system instruction)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
    )
    
    try:
        # Call the SDK's generate_content method
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )
        
        return response.text
            
    except APIError as e:
        # Handle specific API errors
        st.error(f"Gemini API Call Failed (SDK Error): {e}")
        return "Summarization failed due to API connection error."
    except Exception as e:
        # Handle other unexpected errors
        st.error(f"An unexpected error occurred during summarization: {e}")
        return "Summarization failed due to an unexpected error."


# --- Streamlit UI ---
st.set_page_config(page_title="Universal Video/Audio Summarizer (SDK)", layout="centered")

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


st.markdown('<h1 class="main-header">🎙️ Universal Video/Audio Summarizer (Whisper + Gemini SDK)</h1>', unsafe_allow_html=True)
st.warning("⚠️ **Multilingual Support:** This app now uses a multilingual model for **English** and **Burmese** transcription. Requires FFmpeg on the server.")
st.write("Upload **any** video or audio file (MP4, MOV, MP3, WAV, etc.) to generate a full transcript and a key point summary.")

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
            # CHANGED: transcribe_video_with_whisper now returns a tuple
            whisper_result = transcribe_video_with_whisper(uploaded_file)
            
            if whisper_result is not None:
                transcript, detected_lang_code = whisper_result
                
                st.subheader("📝 Extracted Transcript")
                
                # Show the long transcript in an expander
                with st.expander("Click to view full transcript text"):
                    if transcript:
                        st.code(transcript, language="text")
                    else:
                        st.warning("Transcript is empty or blank. Cannot proceed to summarization.")

                # 2. Summarization Step
                if transcript:
                    # CHANGED: Passed the detected language code to the summarizer
                    summary = summarize_text(transcript, detected_lang_code)
                    
                    st.subheader("✅ Summary (Generated by Gemini)")
                    st.markdown(summary)
                    
                    if not summary.startswith("Summarization failed"):
                        st.success("Process complete: Transcript generated and Summary extracted.")
                    else:
                        st.error("Process failed during summarization. Check error details above.")
                
            else:
                st.error("Transcription failed. Please ensure the audio quality is clear, the file is valid, and FFmpeg is properly installed on the server.")
