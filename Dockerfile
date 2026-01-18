# Base image with Python 3.9
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Download Whisper medium model during build
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='float32')"

# Download pyannote models during build
# We'll use a script to pre-download models
ARG HF_TOKEN_BUILD=""
ENV HF_TOKEN=${HF_TOKEN_BUILD}
COPY download_models.py .
RUN --mount=type=cache,target=/root/.cache/huggingface/hub python download_models.py

# Copy application code
COPY app.py .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Start the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]