# Backend Dockerfile with automatic port conflict resolution
FROM node:18-alpine

# Install system dependencies for OCR and document processing
RUN apk add --no-cache \
    tesseract-ocr \
    tesseract-ocr-data-eng \
    tesseract-ocr-data-nld \
    poppler-utils \
    imagemagick \
    curl \
    && rm -rf /var/cache/apk/*

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --only=production && npm cache clean --force

# Copy application code
COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p uploads logs public

# Set proper permissions
RUN chown -R node:node /app
USER node

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start the application
CMD ["npm", "start"]
