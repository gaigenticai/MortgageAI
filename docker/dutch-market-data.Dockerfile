# Dutch Market Data Service Dockerfile
# Handles AFM regulations, BKR credit data, NHG validation, and property valuations

FROM node:18-alpine

# Install system dependencies for Dutch integrations
RUN apk add --no-cache \
    curl \
    jq \
    openssl \
    ca-certificates \
    && rm -rf /var/cache/apk/*

# Set working directory
WORKDIR /app

# Copy package files
COPY backend/package*.json ./

# Install production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy backend source code
COPY backend/ ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache

# Create non-root user for security
RUN addgroup -g 1001 -S dutchdata && \
    adduser -S dutchmarket -u 1001

# Set proper permissions
RUN chown -R dutchmarket:dutchdata /app
USER dutchmarket

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${DUTCH_DATA_PORT:-8001}/health || exit 1

# Expose port
EXPOSE ${DUTCH_DATA_PORT:-8001}

# Start the Dutch Market Data service
CMD ["node", "services/dutch-market-data-server.js"]
