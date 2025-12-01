import streamlit as st
import os
import tempfile
import sys
import time

# --- Gemini SDK Imports ---
try:
    from google import genai
    from google.genai import types  # For GenerateContentConfig
    from google.genai.errors import APIError as GeminiAPIError
except ImportError:
    st.error("The 'google-genai' library is not installed. Please install it using 'pip install google-genai'.")
    st.stop()

# Import Whisper model loading
try:
    import whisper
except ImportError:
    st.error("The 'openai-whisper' library is not installed. Please install it using 'pip install openai-whisper'.")
    st.stop()


# --- Configuration and Client Initialization ---
# ❗ IMPORTANT: Ensure GEMINI_API_KEY is set in Streamlit secrets
try:
    # Attempt to retrieve API Key from Streamlit secrets
    API_KEY = st.secrets["GEMINI_API_KEY"] 
except KeyError:
    st.error("🚨 API Key Error: Please set 'GEMINI_API_KEY' in your Streamlit secrets file or Streamlit Cloud Secrets.")
    st.stop()

try:
    # Initialize the Gemini Client globally
    client = genai.Client(api_key=API_KEY)  
except Exception as e:
    st.error(f"Error initializing AI client. Details: {e}")
    st.stop()
    
# Model name used for text generation
MODEL_NAME = "gemini-2.5-flash" 

# Define language codes for Whisper
LANG_CODE_MY = "my" # ISO code for Burmese/Myanmar


# --- Utility Functions ---

@st.cache_resource
def load_whisper_model():
    """
    Load the Whisper 'base' model once to meet Streamlit Cloud's memory constraints.
    """
    st.info("Loading Whisper **'base'** model... (This initialization happens only once)")
    try:
        # Using 'base' (approx. 1GB RAM) for stability.
        # Set device="cpu" to avoid GPU dependency.
        return whisper.load_model("base", device="cpu") 
    except Exception as e:
        st.error(f"Failed to load Whisper model.")
        st.error("Error Hint: If you see 'Failed to load model', check PyTorch/Whisper installation and memory limits.")
        return None

def transcribe_video_with_whisper(uploaded_file):
    """
    Transcribes the audio from the uploaded video file, automatically detecting language.
    Returns: (transcript, detected_language_code)
    """
    model = load_whisper_model()
    if model is None:
        return None, None
        
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
        start_time = time.time()
        
        # Run Whisper (fp16=False for CPU stability)
        result = model.transcribe(temp_path, fp16=False) 
        
        end_time = time.time()
        
        detected_lang = result.get("language", "en") 
        transcript = result["text"].strip()
        
        st.success(f"Language detected by Whisper: **{detected_lang.upper()}**. Transcription completed in {end_time - start_time:.2f} seconds.")

        if len(transcript) < 50:
            st.warning("Whisper completed, but the extracted transcript is too short. Cannot proceed to summarization.")
            return None, None

        return transcript, detected_lang
            
    except Exception as e:
        st.error(f"Whisper Transcription Failed. (Check FFmpeg installation and file integrity)")
        st.error(f"Error Details: {e}")
        return None, None
            
    finally:
        # 3. Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def summarize_text(transcript_text, detected_lang):
    """
    Summarizes the transcript using the Gemini AI client with language-specific system instructions.
    """
    if not transcript_text or transcript_text.isspace():
        return "Summarization failed: Empty transcript content."
        
    st.info("Step 2/2: Sending transcript to Gemini AI for summarization...")
    
    # --- Dynamic Prompts ---
    if detected_lang == LANG_CODE_MY:
        # Burmese System Instruction 
        system_instruction = (
            "သင်သည် ပရော်ဖက်ရှင်နယ် ဗီဒီယို အနှစ်ချုပ်သူ ဖြစ်သည်။ သင်၏တာဝန်မှာ အောက်ပါ မြန်မာဗီဒီယို စာသားကို ခွဲခြမ်းစိတ်ဖြာပြီး ဆွေးနွေးထားသော အရေးကြီးဆုံး သင်ယူမှုအချက် ၅ ချက်၊ အယူအဆ ၅ ချက် သို့မဟုတ် အဆင့် ၅ ဆင့်ကို ထုတ်နုတ်ရန်ဖြစ်သည်။ ရလဒ်ကို ရှင်းလင်းပြတ်သားသော အချက်အလက်စာရင်း (bullet points) များဖြင့် ဖော်ပြပါ။"
        )
        user_query = f"ကျေးဇူးပြု၍ အောက်ပါ ဗီဒီယို စာသားကို အကျဉ်းချုပ်ပေးပါ။:\n\n---\n\n{transcript_text}"
        
    else: # Default to English for all other languages
        # English System Instruction 
        system_instruction = (
            "You are a professional video summarizer. Your task is to analyze the following video transcript "
            "and extract the 5 most critical learning points, concepts, or steps discussed. Present the output using clear, concise bullet points."
        )
        user_query = f"Please summarize the following video transcript:\n\n---\n\n{transcript_text}"
    
    # --- Gemini API Call Structure ---
    prompt_contents = [user_query] 

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_contents,
            config=types.GenerateContentConfig( 
                system_instruction=system_instruction, 
                temperature=0.0 # Keep summarization factual
            )
        )
        
        return response.text
        
    except GeminiAPIError as e: 
        st.error(f"Gemini API Call Failed (SDK Error): {e}")
        return "Summarization failed due to API connection error."
    except Exception as e:
        st.error(f"An unexpected error occurred during summarization: {e}")
        return "Summarization failed due to an unexpected error."


# --- Streamlit UI ---
st.set_page_config(page_title="Universal Video/Audio Summarizer (Gemini)", layout="centered")

st.markdown("""
<style>
    /* Custom Styling for aesthetics */
    .stButton>button {
        background-color: #0A66C2; 
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        transition: background-color 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #004182;
    }
    .main-header {
        color: #0A66C2; 
        font-weight: bold;
        text-align: center;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">🎙️ Universal Video/Audio Summarizer (Whisper + Gemini SDK)</h1>', unsafe_allow_html=True)
st.warning("⚠️ **System Note:** Using Whisper **'base'** model for memory compatibility. Burmese transcription accuracy may be lower than with 'large' or 'medium' models.")
st.write("Upload **any** English or **Myanmar (Burmese)** video/audio file to generate a full transcript and a key point summary.")

# File Uploader
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
        with st.spinner("Step 1/2: Generating Transcript using Whisper AI..."):
            transcript, detected_lang = transcribe_video_with_whisper(uploaded_file)
            
            if transcript is not None:
                st.subheader("📝 Extracted Transcript (ထုတ်ယူထားသော စာသား)")
                
                # Show the long transcript in an expander
                with st.expander("Click to view full transcript text"):
                    if transcript:
                        st.code(transcript, language="text") 
                    else:
                        st.warning("Transcript is empty or blank. Cannot proceed to summarization.")

                # 2. Summarization Step (Only runs if a valid transcript was returned)
                if transcript:
                    summary = summarize_text(transcript, detected_lang)
                    
                    st.subheader("✅ Summary (Generated by Gemini - အနှစ်ချုပ်)")
                    st.markdown(summary)
                    
                    if not summary.startswith("Summarization failed"):
                        st.success("Process complete: Transcript generated and Summary extracted.")
                    else:
                        st.error("Process failed during summarization. Check error details above.")
                
            else:
                st.error("Transcription failed. Please ensure the audio quality is clear, the file is valid, and FFmpeg is properly installed on the server.")