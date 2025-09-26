/**
 * Dutch Market Data Service Server
 *
 * Main server file for the Dutch Market Data microservice that provides:
 * - AFM Regulation compliance validation
 * - BKR Credit scoring integration
 * - NHG eligibility validation
 * - Property valuation services
 *
 * This service provides real-time Dutch mortgage market data and compliance checking.
 */

require('dotenv').config();

const express = require('express');
const winston = require('winston');
const expressWinston = require('express-winston');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');

// Import service modules
const { 
  AFMRegulationService, 
  BKRCreditService, 
  NHGValidationService, 
  PropertyValuationService,
  initializeServices
} = require('./dutch-market-data.js');

// Initialize logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'dutch-market-data' },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({
      filename: process.env.LOG_FILE || '/app/logs/dutch-market-data.log'
    })
  ]
});

// Initialize Express app
const app = express();
const PORT = process.env.DUTCH_DATA_PORT || 8001;

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

// Initialize services
const afmService = new AFMRegulationService();
const bkrService = new BKRCreditService();
const nhgService = new NHGValidationService();
const propertyService = new PropertyValuationService();

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    service: 'dutch-market-data',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// AFM Regulation endpoints
app.get('/api/afm/regulations', async (req, res) => {
  try {
    const regulations = await afmService.getAllRegulations();
    res.json({ success: true, data: regulations });
  } catch (error) {
    logger.error('Failed to fetch AFM regulations', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to fetch regulations' });
  }
});

app.post('/api/afm/validate', async (req, res) => {
  try {
    const { regulation, requirement, context } = req.body;
    const result = await afmService.checkRegulationCompliance(regulation, requirement, context);
    res.json({ success: true, compliant: result });
  } catch (error) {
    logger.error('AFM validation failed', { error: error.message });
    res.status(500).json({ success: false, error: 'Validation failed' });
  }
});

// BKR Credit check endpoints
app.post('/api/bkr/check', async (req, res) => {
  try {
    const { bsn, applicationId } = req.body;
    const result = await bkrService.performCreditCheck(bsn, applicationId);
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('BKR credit check failed', { error: error.message, bsn, applicationId });
    res.status(500).json({ success: false, error: 'Credit check failed' });
  }
});

// NHG Validation endpoints
app.post('/api/nhg/validate', async (req, res) => {
  try {
    const { applicationData } = req.body;
    const result = await nhgService.validateNHGEligibility(applicationData);
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('NHG validation failed', { error: error.message });
    res.status(500).json({ success: false, error: 'NHG validation failed' });
  }
});

// Property valuation endpoints
app.post('/api/property/valuate', async (req, res) => {
  try {
    const { propertyData } = req.body;
    const result = await propertyService.getPropertyValuation(propertyData);
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('Property valuation failed', { error: error.message });
    res.status(500).json({ success: false, error: 'Property valuation failed' });
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

// Server will be started by the startServer() function below

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

// Start the server
async function startServer() {
  try {
    // Initialize all services
    await initializeServices();
    
    // Start HTTP server
    app.listen(PORT, () => {
      logger.info(`🚀 Dutch Market Data Service running on port ${PORT}`, {
        environment: process.env.NODE_ENV || 'production',
        port: PORT,
        service: 'dutch-market-data'
      });
    });
  } catch (error) {
    logger.error('Failed to start Dutch Market Data service:', error);
    process.exit(1);
  }
}

// Start the server
startServer();

module.exports = app;
