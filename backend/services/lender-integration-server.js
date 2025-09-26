/**
 * Lender Integration Service Server
 *
 * Main server file for the Lender Integration microservice that provides:
 * - Multi-lender API integration (Stater, Quion, ING, Rabobank, ABN AMRO)
 * - Lender-specific mortgage application submission
 * - Real-time lender status monitoring
 * - Automated lender communication
 *
 * This service handles all lender-specific operations for Dutch mortgage applications.
 */

require('dotenv').config();

const express = require('express');
const winston = require('winston');
const expressWinston = require('express-winston');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');

// Import service module
const LenderIntegrationService = require('./lender-integration.js').LenderIntegrationService;

// Initialize logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'lender-integration' },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({
      filename: process.env.LOG_FILE || '/app/logs/lender-integration.log'
    })
  ]
});

// Initialize Express app
const app = express();
const PORT = process.env.LENDER_PORT || 8002;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.NODE_ENV === 'production' ? false : true,
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Compression middleware
app.use(compression());

// Request logging
app.use(expressWinston.logger({
  winstonInstance: logger,
  meta: true,
  msg: "HTTP {{req.method}} {{req.url}} {{res.statusCode}} {{res.responseTime}}ms",
  expressFormat: true,
  colorize: false,
  ignoreRoute: function (req, res) { return false; }
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Initialize service
const lenderService = new LenderIntegrationService();

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    service: 'lender-integration',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// Get supported lenders
app.get('/api/lenders', async (req, res) => {
  try {
    const lenders = await lenderService.getSupportedLenders();
    res.json({ success: true, data: lenders });
  } catch (error) {
    logger.error('Failed to fetch lenders', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to fetch lenders' });
  }
});

// Submit application to lender
app.post('/api/lenders/:lenderId/submit', async (req, res) => {
  try {
    const { lenderId } = req.params;
    const { applicationData } = req.body;

    const result = await lenderService.submitToLender(lenderId, applicationData);
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('Lender submission failed', {
      error: error.message,
      lenderId: req.params.lenderId,
      applicationId: req.body.applicationData?.id
    });
    res.status(500).json({ success: false, error: 'Submission failed' });
  }
});

// Get lender status
app.get('/api/lenders/:lenderId/status', async (req, res) => {
  try {
    const { lenderId } = req.params;
    const status = await lenderService.getLenderStatus(lenderId);
    res.json({ success: true, data: status });
  } catch (error) {
    logger.error('Failed to get lender status', { error: error.message, lenderId: req.params.lenderId });
    res.status(500).json({ success: false, error: 'Failed to get status' });
  }
});

// Get application status from lender
app.get('/api/applications/:applicationId/status', async (req, res) => {
  try {
    const { applicationId } = req.params;
    const status = await lenderService.getApplicationStatus(applicationId);
    res.json({ success: true, data: status });
  } catch (error) {
    logger.error('Failed to get application status', { error: error.message, applicationId: req.params.applicationId });
    res.status(500).json({ success: false, error: 'Failed to get application status' });
  }
});

// Get lender rates
app.get('/api/lenders/:lenderId/rates', async (req, res) => {
  try {
    const { lenderId } = req.params;
    const { loanAmount, loanTerm, propertyValue } = req.query;

    const rates = await lenderService.getLenderRates(lenderId, {
      loanAmount: parseFloat(loanAmount),
      loanTerm: parseInt(loanTerm),
      propertyValue: parseFloat(propertyValue)
    });

    res.json({ success: true, data: rates });
  } catch (error) {
    logger.error('Failed to get lender rates', { error: error.message, lenderId: req.params.lenderId });
    res.status(500).json({ success: false, error: 'Failed to get rates' });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error', { error: error.message, stack: error.stack });
  res.status(500).json({ success: false, error: 'Internal server error' });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ success: false, error: 'Endpoint not found' });
});

// Start server
app.listen(PORT, () => {
  logger.info(`Lender Integration Service running on port ${PORT}`, {
    port: PORT,
    environment: process.env.NODE_ENV || 'development'
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

module.exports = app;
