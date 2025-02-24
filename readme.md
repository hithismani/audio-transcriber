# Audio/Video Transcription App

This application uses OpenAI's Whisper model to transcribe audio and video files.

## Features

- Supports MP3, MP4, WAV, and M4A file formats
- Automatically splits large files for processing
- Provides a simple web interface using Gradio

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/transcription-app.git
   cd transcription-app
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

Run the application:
```
python transcript.py
```

Then open a web browser and go to the URL displayed in the console (usually http://localhost:7860).

## Privacy Considerations

When using this transcription application, it's important to be aware of the following privacy implications:

- **Data Transmission**: Audio and video files are sent to OpenAI's servers for transcription. Ensure you have the necessary rights and permissions for the content you're transcribing.
- **Sensitive Information**: Be cautious about transcribing files containing personal, confidential, or sensitive information.
- **API Data Usage**: According to OpenAI's official policy, **data sent through the API is not used for training models**. As stated in their Business Terms: "We will not use Customer Content to develop or improve the Services." [[Source]](https://community.openai.com/t/does-open-ai-api-use-api-data-for-training/659053)
- **Data Retention**: OpenAI has endpoint-specific data retention policies, with data typically being deleted after a certain number of days.
- **Consent**: If transcribing recordings of other individuals, obtain their consent before processing.
- **Local Processing**: For maximum privacy, consider using local transcription models that don't require sending data to external servers.

> **Important Note**: Always review the most current [OpenAI API Terms of Service](https://openai.com/policies/api-terms/) for the most up-to-date privacy and data handling information.

## Notes

- If you encounter issues with file processing, ensure FFmpeg is correctly installed and accessible in your system PATH.
- For API-related errors, check that your OpenAI API key is correct and has the necessary permissions.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the Whisper transcription model
- Gradio for the easy-to-use web interface framework
- Pydub for audio file handling

