import streamlit as st
from openai import OpenAI
import os
from pathlib import Path
import tempfile
from pydub import AudioSegment
import math
import logging
import atexit
from dotenv import load_dotenv
from moviepy import VideoFileClip
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.warning("Please set your OpenAI API key in the .env file or as an environment variable.  See the README for instructions.")
    client = None  # Disable OpenAI client if API key is missing
else:
    client = OpenAI(api_key=openai_api_key)

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
CHUNK_SIZE = 24 * 1024 * 1024  # 24 MB for safety
ALLOWED_EXTENSIONS = {'.mp3', '.mp4', '.wav', '.m4a'}

def split_audio(file_path, chunk_size):
    """Splits an audio file into smaller chunks, exporting as MP3."""
    logger.info(f"Splitting audio file: {file_path}")
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        logger.error(f"Error loading audio with pydub: {e}")
        st.error(f"Error loading audio file.  Please ensure it's a supported format (mp3, mp4, wav, m4a).  Error: {e}")
        return []  # Return an empty list to indicate failure

    duration = len(audio)
    chunk_duration = math.ceil((chunk_size / len(audio.raw_data)) * duration)
    chunks = []

    for i in range(0, duration, chunk_duration):
        chunk = audio[i:i+chunk_duration]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:  # Always use .mp3
            chunk.export(temp_file.name, format="mp3")  # Export as MP3
            chunks.append(temp_file.name)

    logger.info(f"Split into {len(chunks)} chunks")
    return chunks

def cleanup_temp_files():
    """Clean up temporary files on exit."""
    logger.info("Cleaning up temporary files")
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.endswith(tuple(ALLOWED_EXTENSIONS)) or file.endswith('.txt'):
            try:
                os.remove(os.path.join(temp_dir, file))
            except Exception as e:
                logger.error(f"Error deleting temporary file {file}: {e}")

atexit.register(cleanup_temp_files)

def transcribe_file(uploaded_file, timestamp_option):
    """Transcribes an audio or video file, handling large files by splitting.

    Args:
        uploaded_file: The file uploaded by the user.
        timestamp_option: Boolean, whether to include timestamps.
    """
    if uploaded_file is None:
        return "Please upload a file.", None

    if client is None:
        return "OpenAI API key not configured. Please set it to use this feature.", None

    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            file_path = Path(temp_file.name)

        logger.info(f"Processing file: {file_path}")

        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return f"Please upload a file with one of these extensions: {', '.join(ALLOWED_EXTENSIONS)}", None

        if file_path.suffix.lower() == '.mp4':  # Handle video files
            try:
                # Extract audio from video using moviepy
                video_clip = VideoFileClip(str(file_path))
                audio_clip = video_clip.audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_temp_file:
                    audio_clip.write_audiofile(audio_temp_file.name)
                    audio_file_path = audio_temp_file.name

                video_clip.close() # close the video clip
                audio_clip.close() #close the audio clip

            except Exception as e:
                logger.error(f"Error extracting audio from video: {e}")
                return f"Error extracting audio from video: {e}", None

            # Now, use the extracted audio_file_path for transcription
            file_path_to_transcribe = audio_file_path

        else: #It is an audio file
            file_path_to_transcribe = file_path

        if os.path.getsize(file_path_to_transcribe) > MAX_FILE_SIZE:
            chunks = split_audio(file_path_to_transcribe, CHUNK_SIZE)
            if not chunks:  # Check if split_audio returned an empty list (error)
                return "Error splitting audio file.", None
            full_transcript = ""

            # Create a placeholder for progress updates
            progress_placeholder = st.empty()
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Transcribing chunk {i}/{len(chunks)}")
                with open(chunk, "rb") as audio_file:
                    # Request word-level timestamps or plain text based on user choice
                    if timestamp_option:
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="verbose_json"  # Get detailed JSON output
                        )
                                                
                        # Iterate through the segments and construct timestamped output
                        for segment in response.segments:
                            start_time = segment.start
                            end_time = segment.end
                            text = segment.text
                            full_transcript += f"[{start_time:.2f} - {end_time:.2f}] {text.strip()}\n"
                    else:
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text"  # Get plain text output
                        )
                        full_transcript += response + " "

                os.unlink(chunk)  # Delete the temporary chunk
                # Update the progress in the placeholder
                progress_placeholder.text(f"Processed chunk {i}/{len(chunks)}")

            # Save full transcript to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as transcript_file:
                transcript_file.write(full_transcript)
            logger.info("Transcription complete")
            return full_transcript, transcript_file.name

        else:
            logger.info("Transcribing single file")
            with open(file_path_to_transcribe, "rb") as audio_file:
                # Request word-level timestamps or plain text based on user choice
                if timestamp_option:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                    full_transcript = ""
                    for segment in response.segments:
                        start_time = segment.start
                        end_time = segment.end
                        text = segment.text
                        full_transcript += f"[{start_time:.2f} - {end_time:.2f}] {text.strip()}\n"

                else:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                    full_transcript = response  # Directly use the response string

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as transcript_file:
                transcript_file.write(full_transcript)
            logger.info("Transcription complete")
            return full_transcript, transcript_file.name
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}", None
    finally:
        # Clean up the initial temp file (for both single and multi-part)
        if 'file_path' in locals():
            try:
                os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file {file_path}: {e}")
        #Clean up the audio file if we extracted from video
        if 'file_path_to_transcribe' in locals() and 'audio_file_path' in locals() and file_path_to_transcribe == audio_file_path:
            try:
                os.unlink(file_path_to_transcribe)
            except Exception as e:
                logger.error(f"Error deleting temporary file {file_path_to_transcribe}: {e}")

# Streamlit UI
st.title("Audio/Video Transcription with OpenAI Whisper")
st.write("Upload an audio file (MP3, MP4, WAV, or M4A) to transcribe. Large files will be automatically split and processed.")

# Clear previous file on page reload
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.file_uploader("Upload an audio or video file", type=list(ALLOWED_EXTENSIONS))

# Store the uploaded file in session state
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

# Add a checkbox for timestamp option
timestamp_option = st.checkbox("Include timestamps?")

if st.session_state.uploaded_file is not None:
    if client is None:
        st.error("Please configure your OpenAI API key to enable transcription.")
    else:
        with st.spinner("Transcribing..."):
            transcript, transcript_file_path = transcribe_file(st.session_state.uploaded_file, timestamp_option)

            if transcript:
                # st.write("Transcription:")
                st.text_area("Transcription:", transcript, height=300) # Larger text area

                if transcript_file_path:
                    try:
                        with open(transcript_file_path, "rb") as f:
                            st.download_button(
                                label="Download Transcript",
                                data=f,
                                file_name=Path(st.session_state.uploaded_file.name).stem + "_transcript.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"Error preparing download: {e}")
                        logger.error(f"Error opening transcript file for download: {e}")
                    finally:
                        #Clean up transcript file after download attempt
                        try:
                            os.unlink(transcript_file_path)
                        except Exception as e:
                             logger.error(f"Error deleting transcript file {transcript_file_path}: {e}")
            else:
                st.error("Transcription failed. See logs for details.")

        # Clear the uploaded file after processing
        st.session_state.uploaded_file = None