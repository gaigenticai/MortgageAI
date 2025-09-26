# Lender Integration Service Dockerfile
# Handles integration with Dutch lenders (Stater, Quion, ING, Rabobank, ABN AMRO)

FROM node:18-alpine

# Install system dependencies for lender integrations
RUN apk add --no-cache \
    curl \
    jq \
    openssl \
    ca-certificates \
    xmlsec \
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
RUN mkdir -p /app/logs /app/uploads /app/cache

# Create non-root user for security
RUN addgroup -g 1001 -S lender && \
    adduser -S lenderintegration -u 1001

# Set proper permissions
RUN chown -R lenderintegration:lender /app
USER lenderintegration

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${LENDER_PORT:-8002}/health || exit 1

# Expose port
EXPOSE ${LENDER_PORT:-8002}

# Start the Lender Integration service
CMD ["node", "services/lender-integration-server.js"]
