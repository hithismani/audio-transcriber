# Audio/Video Transcription App

This application uses OpenAI's Whisper model to transcribe audio and video files with a simple interface using Streamlit. Also uses pyannote (via Hugging Face) for speaker diarization.

## Demo

View the demo of the app on [X](https://x.com/megabored/status/1893641574413742102).

## Features

- Supports MP3, MP4, WAV, and M4A file formats
- Automatically splits large files for processing (up to 25 MB)
- Provides a simple web interface using Streamlit
- Multiple transcription options:
  * Full Transcription
  * Timestamped Transcription
  * Optional Transcription with Timestamps and Speaker Identification
- Handles both audio and video file transcription
- Robust error handling and logging

## Prerequisites: Installing FFmpeg

FFmpeg is a critical dependency for this application. Follow the installation instructions for your operating system:

### Windows, macOS, and Linux Installation Instructions

#### Windows
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract the downloaded zip file
3. Add the `bin` folder to your system PATH

#### macOS (using Homebrew)
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/hithismani/audio-transcriber.git
   cd audio-transcriber
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   HF_ACCESS_TOKEN=your_huggingface_token_here  # Optional, for speaker identification
   ```

## Hugging Face Authentication (Optional Speaker Identification)

### Speaker Identification Setup
1. Create a Hugging Face account: [https://huggingface.co/](https://huggingface.co/)
2. Accept user conditions for these models:
   - [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
   - [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
3. Create a Hugging Face access token (read role): [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Add the token to your `.env` file: `HF_ACCESS_TOKEN=your_huggingface_token`

**Note:** Speaker identification is an experimental feature and is optional. The app will function fully without this token.

## Usage

Run the application:
```bash
streamlit run transcribe.py
```

### Transcription Options
- Upload audio or video files (MP3, MP4, WAV, M4A)
- Choose transcription type:
  * Full Transcription
  * Timestamped Transcription
  * Optional Transcription with Timestamps and Speaker Identification
- Automatic handling of large files by splitting them into chunks
- Direct download of transcription results

## Roadmap: Upcoming Features

### ✅ Completed Features
- [x] Speaker Identification (Optional)
  * Uses pyannote.audio for experimental speaker diarization
  * Identifies distinct speakers in audio
  * Works best with clear, separated speech
  * Optional feature that can be enabled/disabled
  * Provides speaker labels like "SPEAKER_00", "SPEAKER_01"

- [x] Chunk Serialization
  * Automatically splits large audio files (> 24 MB)
  * Preserves audio quality during splitting
  * Adds short silence between chunks to prevent cut-off words
  * Supports files up to 25 MB
  * Handles various audio formats (MP3, WAV, M4A, MP4)
  * Seamlessly transcribes split chunks
  * Reconstructs full transcription from individual chunks

### 🚧 Planned Features

1. **Multi-Model Support**
   - Allow users to choose from multiple AI transcription models
   - Support for:
     * OpenAI Whisper
     * Google Speech-to-Text
     * Amazon Transcribe
     * Local open-source models (Whisper, wav2vec, etc.)
   - Ability to select specific sub-models within each provider
   - Comparative analysis of transcription accuracy

2. **Advanced Transcription Organization**
   - Automatic chapter/section detection
   - Manual chapter creation and editing
   - Timestamp-based chapter segmentation
   - Export chapters as separate files or with hierarchical structure

## Privacy Considerations

When using this transcription application, be aware of:
- Audio/video files are sent to OpenAI's servers for transcription
- Ensure you have necessary rights and permissions for content
- OpenAI API data is not used for model training
- Obtain consent before transcribing recordings of others

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper transcription model
- Streamlit for the web interface
- Pyannote for speaker diarization
- Pydub for audio processing
- MoviePy for video file handling

