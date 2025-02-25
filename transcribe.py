import streamlit as st
import time
from openai import OpenAI
import os
from pydub import AudioSegment
import math
import logging
from dotenv import load_dotenv
from moviepy import VideoFileClip
import re
import subprocess
from datetime import timedelta
from pyannote.audio import Pipeline

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")

# Hugging Face Access Token for pyannote.audio
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

if not openai_api_key:
    st.warning("Please set your OpenAI API key in the .env file or as an environment variable.  See the README for instructions.")
    client = None  # Disable OpenAI client if API key is missing
else:
    client = OpenAI(api_key=openai_api_key)

MAX_FILE_SIZE = 50 * 1024 * 1024  # Increase to 50 MB
CHUNK_SIZE = 24 * 1024 * 1024  # Keep 24 MB chunk size for safety
ALLOWED_EXTENSIONS = {'.mp3', '.mp4', '.wav', '.m4a'}

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)
SPACER_DURATION = 2000  # 2 seconds in milliseconds
MAX_CHUNK_DURATION = 10 * 60 * 1000  # Reduce to 10 minutes in milliseconds
SUPPORTED_EXTENSIONS = {
    "audio": [".mp3", ".wav", ".m4a"],
    "video": [".mp4"]
}

def create_temp_file_path(extension):
    """Creates a unique temporary file path."""
    timestamp = int(time.time())
    return os.path.join(TEMP_DIR, f"temp_{timestamp}{extension}")

def convert_to_wav(input_file_path, output_file_path):
    """Converts an audio/video file to WAV format using FFmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_file_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_file_path],
            check=True, capture_output=True, text=True
        )
        logging.info(f"Successfully converted {input_file_path} to {output_file_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion failed: {e.stderr}")
        st.error(f"Failed to convert file to WAV format. FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during conversion: {e}")
        st.error(f"An unexpected error occurred during conversion: {e}")
        return False

def split_audio(file_path, chunk_duration=MAX_CHUNK_DURATION, spacer_duration=SPACER_DURATION):
    """Splits a large audio file into smaller chunks with intelligent segmentation."""
    logging.info(f"Splitting audio file: {file_path}")
    try:
        audio = AudioSegment.from_file(file_path)
        st.info(f"Loading audio file: {os.path.basename(file_path)}") 
    except Exception as e:
        logging.error(f"Failed to load audio file with pydub: {e}")
        st.error(f"Failed to load audio file: {e}")
        return []

    total_duration = len(audio)
    
    # Intelligent chunk sizing
    if total_duration <= chunk_duration:
        logging.info("Audio file is short enough, no need to split.")
        return [file_path]

    chunk_paths = []
    spacer = AudioSegment.silent(duration=spacer_duration)

    # Calculate number of chunks with some overlap for continuity
    overlap_duration = 5000  # 5 seconds of overlap between chunks
    effective_chunk_duration = chunk_duration - overlap_duration

    num_chunks = math.ceil(total_duration / effective_chunk_duration)
    logging.info(f"Total duration: {total_duration}, Number of chunks: {num_chunks}")
    
    # Use st.info for progress communication
    st.info(f"Splitting into {num_chunks} chunks...")

    for i in range(num_chunks):
        start_time = i * effective_chunk_duration
        end_time = min((i + 1) * chunk_duration, total_duration)

        logging.info(f"Processing chunk {i+1}/{num_chunks}: start={start_time}, end={end_time}")
        
        # Extract chunk with optional overlap
        chunk = audio[start_time:end_time]
        
        # Add spacer only if it's not the last chunk
        if i < num_chunks - 1:
            chunk += spacer

        chunk_path = create_temp_file_path(f"_chunk_{i}.wav")
        
        # Use st.info for progress communication
        st.info(f"Saving chunk {i+1}/{num_chunks}...")
        
        try:
            chunk.export(chunk_path, format="wav")
            logging.info(f"Chunk {i+1} saved to {chunk_path}")
            chunk_paths.append(chunk_path)
        except Exception as e:
            logging.error(f"Failed to export chunk {i+1}: {e}")
            st.error(f"Failed to export chunk {i+1}: {e}")
            return []  # Return empty list to indicate failure

    logging.info(f"Successfully split audio into {num_chunks} chunks.")
    st.info("Audio splitting complete...")
    return chunk_paths

def transcribe_audio(file_path, include_timestamps=False):
    """Transcribes an audio file using OpenAI's Whisper model."""
    logging.info(f"Transcribing audio file: {file_path}")
    try:
        with open(file_path, "rb") as audio_file:
            if 'progress_bar' in globals():
                progress_bar.progress(0.85, text=f"Transcribing {os.path.basename(file_path)}...")  
            
            file_size = os.path.getsize(file_path)
            logging.info(f"Transcribing file: {file_path}, Size: {file_size} bytes")

            try:
                if include_timestamps:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                    transcription = ""
                    for segment in response.segments:
                        start = timedelta(seconds=segment.start)
                        end = timedelta(seconds=segment.end)
                        text = segment.text
                        transcription += f"[{start}] - [{end}] {text.strip()}\n"
                else:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                    transcription = response if isinstance(response, str) else response.text

                logging.info(f"Transcription successful for {file_path}")
                if 'progress_bar' in globals():
                    progress_bar.progress(0.95, text=f"Transcription of {os.path.basename(file_path)} complete...")
                return transcription

            except Exception as api_error:
                if "413" in str(api_error) or "content size limit" in str(api_error).lower():
                    logging.warning(f"Chunk too large, will be processed in smaller pieces: {api_error}")
                    # Return empty string instead of None to continue processing
                    return ""
                else:
                    raise api_error  # Re-raise other API errors

    except Exception as e:
        logging.error(f"An error occurred during transcription: {e}")
        st.error(f"An error occurred during transcription: {e}")
        return None

def cleanup_temp_files(file_paths):
    """Removes temporary files."""
    for file_path in file_paths:
        try:
            os.remove(file_path)
            logging.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logging.error(f"Failed to delete temporary file {file_path}: {e}")

def get_file_extension(file):
    """Extracts the file extension from a file object or path."""
    if isinstance(file, str):  # Handle file paths
        return os.path.splitext(file)[1].lower()
    else:  # Handle UploadedFile objects
        return os.path.splitext(file.name)[1].lower()

def process_file(file, include_timestamps):
    """Processes the uploaded file (audio or video)."""
    # Remove the initial file size check since we'll handle large files by splitting
    
    file_extension = get_file_extension(file)
    logging.info(f"Processing file: {file.name if hasattr(file, 'name') else file}, extension: {file_extension}")

    temp_input_path = create_temp_file_path(file_extension)
    temp_wav_path = create_temp_file_path(".wav")

    # Save the uploaded file to a temporary location
    if hasattr(file, 'read'):  # Check if it's an UploadedFile object
        with open(temp_input_path, "wb") as f:
            f.write(file.read())
    else:
        temp_input_path = file # Already a path (from splitting)

    if file_extension in SUPPORTED_EXTENSIONS["video"]:
        try:
            # Extract audio from video
            progress_bar.progress(0.05, text="Extracting audio from video...") # Update progress
            video_clip = VideoFileClip(temp_input_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(temp_wav_path, codec='pcm_s16le', fps=16000, nbytes=2, ffmpeg_params=["-ac", "1"]) #Ensure mono channel
            audio_clip.close()
            video_clip.close()
        except Exception as e:
            logging.error(f"Error extracting audio from video: {e}")
            st.error(f"Error extracting audio from video: {e}")
            return None, [temp_input_path, temp_wav_path]

    elif file_extension in SUPPORTED_EXTENSIONS["audio"]:
        progress_bar.progress(0.05, text="Converting audio to WAV...")  # Update progress
        if not convert_to_wav(temp_input_path, temp_wav_path):
            return None, [temp_input_path]  # Conversion failed
    else:
        st.error(f"Unsupported file format: {file_extension}")
        logging.error(f"Unsupported file format: {file_extension}")
        return None, [temp_input_path]

    # Split the audio into chunks if necessary
    chunk_paths = split_audio(temp_wav_path)
    all_files_to_cleanup = [temp_input_path, temp_wav_path] + chunk_paths

    if not chunk_paths:  # Splitting failed
        return None, all_files_to_cleanup

    # Transcribe each chunk
    full_transcription = ""
    for chunk_path in chunk_paths:
        # Progress update is handled within transcribe_audio
        transcription = transcribe_audio(chunk_path, include_timestamps)
        if transcription:
            full_transcription += transcription + "\n"
        else:
            return None, all_files_to_cleanup # Transcription failed

    return full_transcription, all_files_to_cleanup

# --- Speaker Diarization Functions ---

def perform_diarization(audio_file_path):
    """Performs speaker diarization using pyannote.audio."""
    logging.info(f"Performing diarization on {audio_file_path}")
    try:
        progress_bar.progress(0.9, text="Performing speaker diarization...") # Update progress
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=HF_ACCESS_TOKEN)
        diarization = pipeline({'uri': 'audio', 'audio': audio_file_path})

        # Convert diarization to a list of segments with speaker labels
        diarization_list = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(segment.start * 1000)  # Convert to milliseconds
            end_ms = int(segment.end * 1000)
            diarization_list.append({"start": start_ms, "end": end_ms, "speaker": speaker})

        logging.info(f"Diarization successful for {audio_file_path}")
        progress_bar.progress(0.98, text="Speaker diarization complete...") # Update progress
        return diarization_list
    except Exception as e:
        logging.error(f"Diarization failed: {e}")
        st.error(f"Diarization failed: {e}.  Check your Hugging Face token and ensure you have accepted the user conditions for the pyannote models.")
        return []

def combine_transcription_and_diarization(transcription, diarization_result):
    """Combines Whisper transcription with pyannote diarization results."""
    combined_output = []
    transcription_lines = transcription.strip().split('\n')

    # Parse transcription lines into segments with start, end, and text
    transcription_segments = []
    for line in transcription_lines:
        match = re.match(r'\[(.*?)\] - \[(.*?)\] (.*)', line)
        if match:
            start_str, end_str, text = match.groups()
            start = int(timedelta(hours=int(start_str.split(':')[0]), minutes=int(start_str.split(':')[1]), seconds=float(start_str.split(':')[2])).total_seconds() * 1000)
            end = int(timedelta(hours=int(end_str.split(':')[0]), minutes=int(end_str.split(':')[1]), seconds=float(end_str.split(':')[2])).total_seconds() * 1000)
            transcription_segments.append({"start": start, "end": end, "text": text})
        else:  # Handle lines without timestamps (plain text transcription)
            transcription_segments.append({"start": None, "end": None, "text": line})

    # Merge diarization and transcription
    for trans_segment in transcription_segments:
        start_time = trans_segment["start"]
        end_time = trans_segment["end"]
        text = trans_segment["text"]

        if start_time is None or end_time is None:  # Plain text transcription
            combined_output.append(f"{text}")
            continue

        # Find the speaker for this segment
        speaker = "Speaker 1"
        for diar_segment in diarization_result:
            if diar_segment["start"] <= start_time and diar_segment["end"] >= end_time:
                speaker = diar_segment["speaker"]
                break

        combined_output.append(f"[{timedelta(milliseconds=start_time)}] - [{timedelta(milliseconds=end_time)}] {speaker}: {text}")

    return "\n".join(combined_output)

# --- Streamlit UI Start ---

st.title("Audio/Video Transcription with Speaker Identification")
st.write("Upload an audio file (MP3, MP4, WAV, or M4A) to transcribe. Large files will be automatically split and processed.")

# Clear previous file on page reload
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.file_uploader("Upload an audio or video file", type=list(SUPPORTED_EXTENSIONS["audio"]) + list(SUPPORTED_EXTENSIONS["video"]))

# Select box for transcription options
transcription_option = st.selectbox(
    "Select Transcription Option:",
    ("Full Transcription", "Timestamped Transcription", "Transcription with Timestamps and Speaker ID")
)

# Set flags based on the selected option
include_timestamps = transcription_option != "Full Transcription"
perform_diarization_flag = transcription_option == "Transcription with Timestamps and Speaker ID"

# More prominent experimental feature warning (before the button)
if perform_diarization_flag:
    st.warning("""
        **Speaker identification is an experimental feature and is not foolproof.**  
        It may misidentify speakers or fail to distinguish between them, especially in noisy environments or with overlapping speech.
    """)

# Store the uploaded file in session state
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if uploaded_file is not None:
    if st.button("Transcribe"):  # Explicit "Transcribe" button
        if client is None:
            st.error("Please configure your OpenAI API key to enable transcription.")
        else:
            start_time = time.time() # Record start time
            progress_bar = st.progress(0, text="Starting transcription process...")
            log_messages = []

            # Redirect logging to the Streamlit UI
            class StreamlitHandler(logging.Handler):
                def emit(self, record):
                    progress_bar.text("\n".join(log_messages + [self.format(record)]))

            logger.addHandler(StreamlitHandler())

            # Main processing block
            with st.spinner("Transcribing....Results will appear below when complete."):
                transcription, temp_files = process_file(uploaded_file, include_timestamps)

                if not transcription:
                    st.error("Transcription failed. Please check the file and try again.")
                    cleanup_temp_files(temp_files)
                    st.stop()

                st.subheader("Transcription:")
                if perform_diarization_flag:
                        temp_wav_for_diarization = create_temp_file_path(".wav")
                        if not convert_to_wav(temp_files[0], temp_wav_for_diarization):
                            st.error("Failed to prepare file for diarization.")
                        else:
                            diarization_result = perform_diarization(temp_wav_for_diarization)
                            if diarization_result:
                                combined_transcription = combine_transcription_and_diarization(transcription, diarization_result)
                                st.text_area("", combined_transcription, height=300)  # Display combined result
                            else:
                                st.text_area("", transcription, height=300) # Fallback to regular transcription.

                            temp_files.append(temp_wav_for_diarization) # Add to cleanup list.
                else:
                    st.text_area("", transcription, height=300)

                progress_bar.progress(1.0, text="Transcription complete!") # Final progress update
                st.download_button(
                    label="Download Transcription",
                    data=transcription,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                end_time = time.time() # Record end time
                total_time = end_time - start_time
                st.success(f"Transcription completed in {total_time:.2f} seconds.")

            cleanup_temp_files(temp_files)
            logger.removeHandler(StreamlitHandler())
            progress_bar.empty()  # Remove the progress bar after completion
else:
    st.write("Please upload a file to transcribe.")