# Audio/Video Transcription App

This application uses OpenAI's Whisper model to transcribe audio and video files with a simple interface using Streamlit.

## Demo

View the demo of the app on [X](https://x.com/megabored/status/1893641574413742102).

## Features

- Supports MP3, MP4, WAV, and M4A file formats
- Automatically splits large files for processing (up to 25 MB)
- Provides a simple web interface using Streamlit
- Option to include timestamps in transcription
- Handles both audio and video file transcription
- Robust error handling and logging

## Prerequisites: Installing FFmpeg

FFmpeg is a critical dependency for this application. Follow the installation instructions for your operating system:

### Windows, macOS, and Linux Installation Instructions
(Keep the existing detailed FFmpeg installation guide)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/hithismani/audio-transcriber.git
   cd audio-transcriber
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the application using one of the following methods:

### Using Python Directly
```
python transcribe.py
```

### Using Pipenv
```bash
# Install pipenv if not already installed
pip install pipenv

# Install dependencies
pipenv install

# Run the application
pipenv run dev
```

Then open a web browser and go to the URL displayed in the console (usually http://localhost:8501).

### Transcription Options
- Upload audio or video files (MP3, MP4, WAV, M4A)
- Toggle timestamp inclusion for more detailed transcripts
- Automatic handling of large files by splitting them into chunks
- Direct download of transcription results

## Inspiration and Background

This project was inspired by [@niallmcnulty/transcript_app](https://github.com/niallmcnulty/transcript_app). While the original project used Gradio, we transitioned to Streamlit to expand the feature set and address some implementation challenges.

The script was a requirement we had at [TheMindClan.com](https://web.TheMindClan.com) to help transcribe corporate mental health workshops and client calls, making our recorded content more accessible and easier to review.

## Privacy Considerations

When using this transcription application, it's important to be aware of the following privacy implications:

- **Data Transmission**: Audio and video files are sent to OpenAI's servers for transcription. Ensure you have the necessary rights and permissions for the content you're transcribing.
- **Sensitive Information**: Be cautious about transcribing files containing personal, confidential, or sensitive information.
- **API Data Usage**: According to OpenAI's official policy, **data sent through the API is not used for training models**. As stated in their Business Terms: "We will not use Customer Content to develop or improve the Services." [[Source]](https://community.openai.com/t/does-open-ai-api-use-api-data-for-training/659053)
- **Data Retention**: OpenAI has endpoint-specific data retention policies, with data typically being deleted after a certain number of days.
- **Consent**: If transcribing recordings of other individuals, obtain their consent before processing.
- **Local Processing**: For maximum privacy, consider using local transcription models that don't require sending data to external servers.

> **Important Note**: Always review the most current [OpenAI API Terms of Service](https://openai.com/policies/api-terms/) for the most up-to-date privacy and data handling information.

## Technical Details

- Uses OpenAI's Whisper model for transcription
- Supports files up to 25 MB (larger files are automatically split)
- Provides both timestamped and plain text transcription options
- Robust error handling and temporary file management

## Roadmap: Upcoming Features

Our vision for the future of this transcription tool includes:

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


## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## Notes

- Ensure FFmpeg is correctly installed and accessible in your system PATH
- Check that your OpenAI API key is correct and has the necessary permissions
- Temporary files are automatically cleaned up after processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the Whisper transcription model
- Streamlit for the intuitive web interface framework
- Pydub for audio file handling
- MoviePy for video file processing

