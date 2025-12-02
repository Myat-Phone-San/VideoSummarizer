import streamlit as st
import os
import tempfile
import time
import sys
import io

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

# --- Optional Transcript Parsing Library Imports ---
# These are required for PDF and DOCX support.
try:
    import pypdf
except ImportError:
    st.info("Optional: Install 'pypdf' (`pip install pypdf`) for PDF transcript support.")
    pypdf = None

try:
    import docx
except ImportError:
    st.info("Optional: Install 'python-docx' (`pip install python-docx`) for Word transcript support.")
    docx = None


# --- Configuration and Client Initialization ---
try:
    # Safely get the API key
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        # Fallback for local development if set as environment variable
        API_KEY = os.environ.get("GEMINI_API_KEY")
        if not API_KEY:
             st.error("🚨 API Key Error: Please set 'GEMINI_API_KEY' in your Streamlit secrets file or Environment variables.")
             st.stop()

except Exception:
    st.error("🚨 API Key Error: Please set 'GEMINI_API_KEY' in your Streamlit secrets file or Environment variables.")
    st.stop()

try:
    # IMPORTANT: Ensure you have replaced the LEAKED API_KEY with a NEW one
    client = genai.Client(api_key=API_KEY)  
except Exception as e:
    st.error(f"Error initializing AI client. Details: {e}")
    st.stop()
    
MODEL_NAME = "gemini-2.5-flash"
LANG_CODE_MY = "my" # ISO code for Burmese/Myanmar
LANG_CODE_EN = "en" # ISO code for English


# --- Session State Initialization ---
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'detected_lang' not in st.session_state:
    # This will be 'my' if transcribed, or assumed 'en' for text input unless user specifies
    st.session_state.detected_lang = ""
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False


# --- Utility Functions ---

@st.cache_resource(max_entries=1) # Ensure the model is loaded only once
def load_whisper_model():
    """
    Load the Whisper 'small' model for best Burmese transcription accuracy.
    """
    st.info("Loading Whisper **'small'** model for better **Burmese accuracy**... (Requires ~3GB RAM)")
    try:
        # Load the 'small' model and force CPU usage to reduce potential GPU memory errors
        model = whisper.load_model("small", device="cpu") 
        st.success("Whisper model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        st.error("⚠️ Memory Error Hint: The 'small' model is too large. You might try reverting to 'base' for smaller environments.")
        return None

def transcribe_video_with_whisper(uploaded_file):
    """
    Transcribes the audio from the uploaded media file.
    
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
        
        # Force 'my' language for best Burmese transcription with the 'small' model
        result = model.transcribe(temp_path, fp16=False, language="my") 
        
        end_time = time.time()
        
        # Whisper still detects the language, but we forced 'my' for transcription.
        # We'll use 'my' as the assumed source for better context in summarization.
        detected_lang = result.get("language", LANG_CODE_MY) 
        transcript = result["text"].strip()
        
        st.success(f"Language detected by Whisper: **{detected_lang.upper()}**. Transcription completed in {end_time - start_time:.2f} seconds.")

        if len(transcript) < 20: 
            st.warning("Whisper completed, but the extracted transcript is too short. Please verify the audio quality or try a different file.")
            return None, None

        return transcript, detected_lang
            
    except Exception as e:
        st.error(f"Whisper Transcription Failed. (Likely file decoding issue or unexpected error)")
        st.error(f"Error Details: {e}")
        return None, None
            
    finally:
        # 3. Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                st.warning(f"Could not fully remove temporary file: {e}")


def summarize_text(transcript_text, target_lang):
    """
    Summarizes the transcript using the Gemini AI client, tailored to the target language.
    """
    if not transcript_text or transcript_text.isspace():
        return "Summarization failed: Empty transcript content."
        
    st.info(f"Sending transcript to Gemini AI for summarization in **{target_lang.upper()}**...")
    
    # --- Dynamic Prompts based on Target Language ---
    
    # Common core prompt for Gemini
    core_query = "Please summarize the following text by extracting the 5 most critical learning points, concepts, or steps discussed. Present the result using clear, concise bullet points."
    
    if target_lang == LANG_CODE_MY:
        # Burmese System Instruction (Requesting summary IN Burmese)
        system_instruction = (
            "သင်သည် ပရော်ဖက်ရှင်နယ် အနှစ်ချုပ်သူ ဖြစ်သည်။ သင်၏တာဝန်မှာ အောက်ပါ စာသားကို ခွဲခြမ်းစိတ်ဖြာပြီး ဆွေးနွေးထားသော အရေးကြီးဆုံး သင်ယူမှုအချက် ၅ ချက်၊ အယူအဆ ၅ ချက် သို့မဟုတ် အဆင့် ၅ ဆင့်ကို မြန်မာဘာသာဖြင့်သာ ထုတ်နုတ်ဖော်ပြရန်ဖြစ်သည်။ ရလဒ်ကို ရှင်းလင်းပြတ်သားသော အချက်အလက်စာရင်း (bullet points) များဖြင့် ဖော်ပြပါ။"
        )
        # Burmese User Query (Providing the core task)
        user_query = f"{core_query} အောက်ပါ စာသားကို အကျဉ်းချုပ်ပေးပါ။:\n\n---\n\n{transcript_text}"
        
    else: # Default to English (LANG_CODE_EN)
        # English System Instruction (Requesting summary IN English)
        system_instruction = (
            "You are a professional summarizer. Your task is to analyze the following text and extract the 5 most critical learning points, concepts, or steps discussed. Present the output using clear, concise bullet points in English."
        )
        # English User Query
        user_query = f"{core_query}\n\n---\n\n{transcript_text}"
    
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
        return "Summarization failed due to API connection error. (Check Console for details)"
    except Exception as e:
        st.error(f"An unexpected error occurred during summarization: {e}")
        return "Summarization failed due to an unexpected error."


# --- Streamlit UI ---
st.set_page_config(page_title="Universal Media/Text Summarizer (Gemini)", layout="centered")

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
        margin: 5px; /* Added margin for side-by-side buttons */
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

st.markdown('<h1 class="main-header">🎙️ Universal Media/Text Summarizer (Whisper + Gemini SDK)</h1>', unsafe_allow_html=True)
st.warning("⚠️ **ACTION REQUIRED:** Your previous **Gemini API Key is leaked** and is now blocked. **You MUST replace the API Key** with a new one in your Streamlit secrets file.")
st.write("This tool uses Whisper for media transcription and Gemini for multi-lingual summarization.")


# --- Input Method Selection ---
input_method = st.radio(
    "Select Input Method (ထည့်သွင်းမှုပုံစံကို ရွေးပါ):",
    ("Upload Media (Audio/Video)", "Upload Transcript File (.txt, .md, .pdf, .docx)", "Paste Text Directly"),
    index=0
)

# Reset state when input method changes
if st.session_state.get('last_input_method') != input_method:
    st.session_state.transcript = ""
    st.session_state.detected_lang = ""
    st.session_state.processing_complete = False
    st.session_state.last_input_method = input_method
    st.rerun()

st.divider()

# --- Conditional Input Handling ---

if input_method == "Upload Media (Audio/Video)":
    
    # Define acceptable media types
    ALL_MEDIA_TYPES = [
        "mp4", "mov", "wav", "mp3", "m4a", "mkv", "avi", "flv", "wmv", 
        "ogg", "flac", "wma", "aac", "aiff", "webm"
    ]

    uploaded_file = st.file_uploader(
        "Upload Video or Audio File (ဗီဒီယို သို့မဟုတ် အသံဖိုင် တင်ပါ)",
        type=ALL_MEDIA_TYPES,
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        st.success(f"File uploaded successfully: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")
        
        # Display the uploaded media
        if uploaded_file.type.startswith('audio'):
            st.audio(uploaded_file, format=uploaded_file.type)
        else:
            st.video(uploaded_file, format=uploaded_file.type)

        if st.button("Generate Transcript (စာသားထုတ်ယူရန်)"):
            # 1. Transcription Step
            with st.spinner("Step 1/2: Generating Transcript using Whisper AI..."):
                transcript, detected_lang = transcribe_video_with_whisper(uploaded_file)
                
                if transcript is not None:
                    st.session_state.transcript = transcript
                    st.session_state.detected_lang = detected_lang
                    st.session_state.processing_complete = True
                else:
                    st.session_state.processing_complete = False

elif input_method == "Upload Transcript File (.txt, .md, .pdf, .docx)":
    
    uploaded_transcript_file = st.file_uploader(
        "Upload Transcript File (.txt, .md, .pdf, .docx) (စာသားဖိုင် တင်ပါ)",
        type=['txt', 'md', 'pdf', 'docx'],
        accept_multiple_files=False
    )
    
    if uploaded_transcript_file is not None:
        try:
            file_extension = os.path.splitext(uploaded_transcript_file.name)[1].lower()
            transcript_content = ""
            
            if file_extension in ['.txt', '.md']:
                # Standard text files
                transcript_content = uploaded_transcript_file.read().decode("utf-8")
                
            elif file_extension == '.pdf':
                # PDF file handling
                if pypdf:
                    # Need to read the file content into an in-memory buffer for pypdf
                    uploaded_transcript_file.seek(0)
                    reader = pypdf.PdfReader(uploaded_transcript_file)
                    for page in reader.pages:
                        transcript_content += page.extract_text() or ""
                    if not transcript_content:
                        st.warning("Could not extract text from PDF. The file may contain images only or be encrypted.")
                else:
                    st.error("Cannot read PDF. Please install the 'pypdf' library (`pip install pypdf`).")
                    st.session_state.processing_complete = False
                    st.stop() # Corrected: Replaced return with st.stop()
            
            elif file_extension == '.docx':
                # DOCX file handling
                if docx:
                    # docx.Document requires a file path or file-like object
                    uploaded_transcript_file.seek(0)
                    document = docx.Document(uploaded_transcript_file)
                    paragraphs = [p.text for p in document.paragraphs]
                    transcript_content = "\n".join(paragraphs)
                    if not transcript_content:
                        st.warning("Could not extract text from DOCX. The file may be empty or encrypted.")
                else:
                    st.error("Cannot read DOCX. Please install the 'python-docx' library (`pip install python-docx`).")
                    st.session_state.processing_complete = False
                    st.stop() # Corrected: Replaced return with st.stop()

            # Common processing for all file types
            if transcript_content.strip():
                st.session_state.transcript = transcript_content
                st.session_state.detected_lang = "manual/unknown" 
                st.session_state.processing_complete = True
                st.success(f"Transcript file '{uploaded_transcript_file.name}' loaded successfully.")
            else:
                 st.error(f"Transcript file '{uploaded_transcript_file.name}' is empty or text extraction failed.")
                 st.session_state.processing_complete = False

        except Exception as e:
            st.error(f"Error reading transcript file: {e}")
            st.session_state.processing_complete = False


elif input_method == "Paste Text Directly":
    
    pasted_text = st.text_area(
        "Paste your text/transcript here (စာသားထည့်ပါ)",
        height=300,
        placeholder="Paste your video or audio transcript here..."
    )
    
    if st.button("Use Pasted Text (စာသားအသုံးပြုရန်)"):
        if len(pasted_text.strip()) > 20:
            st.session_state.transcript = pasted_text.strip()
            # Assume language is Burmese or English for transcription context if not specified
            st.session_state.detected_lang = "manual/unknown"
            st.session_state.processing_complete = True
            st.success("Text accepted. Ready for summarization.")
        else:
            st.error("Please paste at least 20 characters of text.")
            st.session_state.processing_complete = False


# --- Summarization and Output Section ---

if st.session_state.processing_complete:
    
    st.divider()
    st.subheader("📝 Extracted Transcript (ထုတ်ယူထားသော စာသား)")
    
    # Show the long transcript in an expander
    with st.expander(f"Click to view full transcript text (Source: {st.session_state.detected_lang.upper()})"):
        st.code(st.session_state.transcript, language="text") 

    st.subheader("2. Choose Summarization Language (အနှစ်ချုပ်ဘာသာစကား ရွေးပါ)")
    
    col1, col2 = st.columns(2)

    with col1:
        # Button 1: Summarize in English
        if st.button("Summarize in English (အင်္ဂလိပ်ဘာသာ)", use_container_width=True):
            st.subheader("✅ English Summary (Generated by Gemini)")
            with st.spinner("Generating English Summary..."):
                summary = summarize_text(st.session_state.transcript, LANG_CODE_EN)
                st.markdown(summary)
            if not summary.startswith("Summarization failed"):
                 st.balloons()

    with col2:
        # Button 2: Summarize in Burmese
        if st.button("Summarize in Burmese (မြန်မာဘာသာ)", use_container_width=True):
            st.subheader("✅ Burmese Summary (Generated by Gemini - မြန်မာအနှစ်ချုပ်)")
            with st.spinner("Generating Burmese Summary..."):
                summary = summarize_text(st.session_state.transcript, LANG_CODE_MY)
                st.markdown(summary)
            if not summary.startswith("Summarization failed"):
                 st.balloons()
                 
else:
    if st.session_state.transcript:
         st.divider()
         st.info("Transcript loaded. Press one of the summary buttons above.")
    elif input_method != "Upload Media (Audio/Video)" and not st.session_state.transcript:
        st.info("Please upload a file or paste text and click the 'Use' button to proceed.")
    # For media, the button is "Generate Transcript" and already handled.
