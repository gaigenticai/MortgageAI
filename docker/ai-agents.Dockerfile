# AI Agents Dockerfile for Python-based ML services
FROM python:3.11-slim-bullseye

# Install system dependencies for OpenCV, OCR, and build tools
RUN apt-get update && apt-get install -y \
    # Build tools for compiling Python packages
    build-essential \
    g++ \
    gcc \
    make \
    cmake \
    pkg-config \
    # Tesseract OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-nld \
    # OpenCV dependencies for headless operation
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Graphics libraries for OpenCV
    libgl1-mesa-glx \
    libglu1-mesa \
    libegl1-mesa \
    libgl1-mesa-dev \
    libgl1 \
    libglx0 \
    libglx-mesa0 \
    # Additional dependencies
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libtiff5-dev \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Utilities
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables for headless OpenCV operation
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV OPENCV_LOG_LEVEL=ERROR

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI agents code
COPY backend/agents/ ./agents/

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create models directory
RUN mkdir -p models

# Set proper permissions
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import agents.compliance.agent; print('AI Agents healthy')" || exit 1

# Keep container running for API calls
CMD ["python", "-m", "agents.api"]
