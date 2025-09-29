# AI Chat WebSocket Service Dockerfile
# Production-grade container for real-time conversational AI

FROM node:18-alpine AS base

# Install system dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates \
    tzdata

# Set timezone
ENV TZ=Europe/Amsterdam

# Create app directory
WORKDIR /app

# Copy package files
COPY backend/package*.json ./

# Install dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy application code
COPY backend/services/chat-websocket-server.js ./
COPY backend/agents ./agents/

# Create logs directory
RUN mkdir -p /app/logs

# Add health check endpoint
RUN echo 'const http = require("http"); \
const server = http.createServer((req, res) => { \
  if (req.url === "/health") { \
    res.writeHead(200, {"Content-Type": "application/json"}); \
    res.end(JSON.stringify({status: "healthy", service: "chat-websocket", timestamp: new Date().toISOString()})); \
  } else { \
    res.writeHead(404); \
    res.end("Not Found"); \
  } \
}); \
server.listen(8005, () => console.log("Health check server running on port 8005"));' > health-server.js

# Create startup script
RUN printf '#!/bin/sh\nnode /app/health-server.js &\nnode /app/chat-websocket-server.js\n' > start.sh && chmod +x start.sh

# Set user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001 && \
    chown -R nodejs:nodejs /app

USER nodejs

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Start the service
CMD ["/app/start.sh"]


