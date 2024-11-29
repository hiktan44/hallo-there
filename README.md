# Hallo There! 🎥

An advanced AI-powered video generation tool that creates realistic talking avatars from images and audio.

## 🌟 Features

- 🎭 Avatar Generation: Create realistic talking avatars from static images
- 🗣️ Voice Processing: Advanced audio diarization using pyannote.audio
- 🎬 Video Synthesis: High-quality video generation with customizable settings
- 🔄 Multi-pose Support: Generate videos with multiple facial poses
- 🎨 Background Customization: Flexible background handling options

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg
- Hugging Face account and access token

## 🚀 Installation

1. Set up Python environment:
```bash
conda create --name hallo-there
conda activate hallo-there
```

2. Clone the repository:
```bash
git clone https://github.com/hiktan44/hallo-there.git
cd hallo-there
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install .
```

4. Install FFmpeg:
- Linux: `sudo apt-get install ffmpeg`
- Windows: Download from official FFmpeg website and add to system PATH

## ⚙️ Configuration

1. Create Hugging Face access token:
   - Visit [Hugging Face Token Settings](https://huggingface.co/settings/tokens)
   - Generate new token with required permissions

2. Set up diarization:
```bash
python diarization.py -access_token <YOUR_HUGGING_FACE_TOKEN>
```

## 📁 Project Structure

```
hallo-there/
├── source_images/      # Input images (512x512)
├── audio/             # Input audio files
├── diarization/       # Diarization output
├── output/           # Generated video clips
└── docs/             # Documentation
```

## 🎮 Usage

1. Prepare source images:
   - 512x512 pixel squares
   - Face should occupy 50-70% of image
   - Place in `source_images/` directory

2. Prepare audio:
   - Convert to WAV format
   - Place in `audio/input_audio.wav`

3. Generate video:
```bash
python generate_videos.py
python combine_videos.py
```

## 🔧 Advanced Options

- `-mode full`: Enable subtle head movements during silence
- `-background custom`: Use custom background image
- `-quality high`: Generate higher quality output

## 📚 Documentation

Detailed documentation available in [docs/](docs/) directory:
- [Installation Guide](docs/installation.md)
- [Configuration Options](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for audio diarization
- Hugging Face for AI models and infrastructure