# AFM Compliance Monitor Dockerfile
# Real-time AFM compliance monitoring and reporting service

FROM node:18-alpine

# Install system dependencies for AFM monitoring
RUN apk add --no-cache \
    curl \
    jq \
    openssl \
    ca-certificates \
    postgresql-client \
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
RUN mkdir -p /app/logs /app/compliance-reports /app/cache

# Create non-root user for security
RUN addgroup -g 1001 -S afmmonitor && \
    adduser -S afmmonitor -u 1001

# Set proper permissions
RUN chown -R afmmonitor:afmmonitor /app
USER afmmonitor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${AFM_MONITOR_PORT:-8003}/health || exit 1

# Expose port
EXPOSE ${AFM_MONITOR_PORT:-8003}

# Start the AFM Monitor service
CMD ["node", "services/afm-monitor-server.js"]
