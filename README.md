# 🐳 Dockerized Speaker Diarization + ASR Application

A complete solution for speaker diarization and automatic speech recognition, packaged in Docker for easy deployment.

## 🚀 Features

- **Speaker Diarization**: Identify who spoke when using pyannote.audio
- **Automatic Speech Recognition**: Transcribe speech using faster-whisper
- **Combined Output**: Get transcripts organized by speaker
- **Multiple Export Formats**: 
  - Play format (like a script)
  - Individual speaker transcripts
  - RTTM format for diarization results
  - Comprehensive ZIP package with all results
- **Dockerized**: Pre-downloaded models for fast startup
- **Advanced Settings**: Fine-tune model parameters through the UI

## 🐳 Setup & Deployment

### Prerequisites
- Docker and Docker Compose installed
- Hugging Face account with access to pyannote models

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/speaker-diarization-app.git
cd speaker-diarization-app
