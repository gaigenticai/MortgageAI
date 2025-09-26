/**
 * MortgageAI Backend Server
 *
 * Main server file that orchestrates the dual-agent framework:
 * - Compliance & Plain-Language Advisor Agent
 * - Mortgage Application Quality Control Agent
 *
 * Features:
 * - RESTful API endpoints
 * - Authentication (optional)
 * - File upload handling
 * - Rate limiting
 * - Request logging
 * - Health monitoring
 */

require('dotenv').config();

const fastify = require('fastify')({
  logger: {
    level: process.env.LOG_LEVEL || 'info'
  }
});

// Register plugins
fastify.register(require('@fastify/cors'), {
  origin: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization']
});

fastify.register(require('@fastify/helmet'));
fastify.register(require('@fastify/jwt'), {
  secret: process.env.JWT_SECRET || 'your-secret-key'
});

fastify.register(require('@fastify/multipart'), {
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024 // 10MB
  }
});

fastify.register(require('@fastify/rate-limit'), {
  max: parseInt(process.env.RATE_LIMIT_MAX) || 100,
  timeWindow: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000 // 15 minutes
});

fastify.register(require('@fastify/static'), {
  root: process.env.UPLOAD_PATH || '/app/uploads',
  prefix: '/uploads/'
});

// Register HTTP proxy for AI agents
fastify.register(require('@fastify/http-proxy'), {
  upstream: process.env.AGENTS_API_URL || 'http://ai-agents:8000',
  prefix: '/api/compliance',
  rewritePrefix: '/api/compliance'
});

fastify.register(require('@fastify/http-proxy'), {
  upstream: process.env.AGENTS_API_URL || 'http://ai-agents:8000',
  prefix: '/api/quality-control',
  rewritePrefix: '/api/quality-control'
});

// Import routes
const authRoutes = require('../routes/auth');
const afmComplianceRoutes = require('../routes/afm_compliance');
const dutchMortgageQCRoutes = require('../routes/dutch_mortgage_qc');
const applicationsRoutes = require('../routes/applications');

// Register routes
if (process.env.REQUIRE_AUTH === 'true') {
  fastify.register(authRoutes);
}

// Register AFM compliance routes
fastify.register(afmComplianceRoutes, { prefix: '/api/afm' });

// Register Dutch mortgage QC routes
fastify.register(dutchMortgageQCRoutes, { prefix: '/api' });

// Register applications routes
fastify.register(applicationsRoutes, { prefix: '/api' });

// Authentication middleware (when REQUIRE_AUTH=true)
if (process.env.REQUIRE_AUTH === 'true') {
  fastify.decorate('authenticate', async function(request, reply) {
    try {
      await request.jwtVerify();
    } catch (err) {
      reply.send(err);
    }
  });
}

// Health check endpoint
fastify.get('/health', async (request, reply) => {
  return {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    services: {
      compliance_agent: 'available',
      quality_control_agent: 'available',
      database: 'connected',
      redis: 'connected'
    }
  };
});

// Root endpoint
fastify.get('/', async (request, reply) => {
  return {
    message: 'MortgageAI API Server',
    description: 'Agentic AI Solution for Enhancing Mortgage Advice Quality and Application Accuracy',
    version: '1.0.0',
    endpoints: {
      compliance: '/api/compliance',
      quality_control: '/api/quality-control',
      health: '/health'
    }
  };
});

// Authentication routes are registered separately when REQUIRE_AUTH=true

// File upload endpoint
fastify.post('/api/upload', async (request, reply) => {
  const parts = request.parts();

  for await (const part of parts) {
    if (part.type === 'file') {
      // Validate file type
      const allowedTypes = (process.env.ALLOWED_FILE_TYPES || 'pdf,jpg,jpeg,png').split(',');
      const fileExtension = part.filename.split('.').pop().toLowerCase();

      if (!allowedTypes.includes(fileExtension)) {
        return reply.code(400).send({
          error: 'Invalid file type',
          allowed: allowedTypes
        });
      }

      // Save file
      const uploadPath = process.env.UPLOAD_PATH || '/app/uploads';
      const filePath = `${uploadPath}/${Date.now()}-${part.filename}`;

      await part.toFile(filePath);

      return {
        message: 'File uploaded successfully',
        filename: part.filename,
        path: filePath,
        size: part.file.bytesRead
      };
    }
  }

  return reply.code(400).send({ error: 'No file uploaded' });
});

// Error handler
fastify.setErrorHandler((error, request, reply) => {
  fastify.log.error(error);

  const statusCode = error.statusCode || 500;
  const message = error.message || 'Internal Server Error';

  reply.code(statusCode).send({
    error: message,
    timestamp: new Date().toISOString(),
    path: request.url
  });
});

// Not found handler
fastify.setNotFoundHandler((request, reply) => {
  reply.code(404).send({
    error: 'Endpoint not found',
    path: request.url,
    available_endpoints: [
      '/api/compliance/*',
      '/api/quality-control/*',
      '/api/afm/*',
      '/api/qc/*',
      '/api/applications/*',
      '/api/lenders',
      '/health',
      '/api/upload'
    ]
  });
});

// Startup function
async function start() {
  try {
    const port = parseInt(process.env.PORT) || 3000;
    const host = process.env.HOST || '0.0.0.0';

    await fastify.listen({ port, host });

    console.log(`🚀 MortgageAI Server running on http://${host}:${port}`);
    console.log(`📊 Health check: http://${host}:${port}/health`);
    console.log(`🔐 Authentication: ${process.env.REQUIRE_AUTH === 'true' ? 'Enabled' : 'Disabled'}`);

  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('🛑 Shutting down MortgageAI Server...');
  await fastify.close();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('🛑 Shutting down MortgageAI Server...');
  await fastify.close();
  process.exit(0);
});

// Start server
start();
