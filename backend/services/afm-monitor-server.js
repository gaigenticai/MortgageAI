/**
 * AFM Monitor Service Server
 *
 * Main server file for the AFM Monitor microservice that provides:
 * - Real-time AFM compliance monitoring
 * - Automated regulatory alerts
 * - Compliance audit trail generation
 * - Regulatory reporting and notifications
 *
 * This service ensures continuous compliance with AFM regulations.
 */

require('dotenv').config();

const express = require('express');
const winston = require('winston');
const expressWinston = require('express-winston');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');

// Import service module
const AFMMonitorService = require('./afm-monitor.js').AFMMonitorService;

// Initialize logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'afm-monitor' },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({
      filename: process.env.LOG_FILE || '/app/logs/afm-monitor.log'
    })
  ]
});

// Initialize Express app
const app = express();
const PORT = process.env.AFM_MONITOR_PORT || 8003;

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
const afmMonitor = new AFMMonitorService();

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'healthy',
    service: 'afm-monitor',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// Get monitoring status
app.get('/api/monitor/status', async (req, res) => {
  try {
    const status = await afmMonitor.getMonitoringStatus();
    res.json({ success: true, data: status });
  } catch (error) {
    logger.error('Failed to get monitoring status', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to get status' });
  }
});

// Get compliance alerts
app.get('/api/monitor/alerts', async (req, res) => {
  try {
    const { severity, limit = 50 } = req.query;
    const alerts = await afmMonitor.getComplianceAlerts({
      severity,
      limit: parseInt(limit)
    });
    res.json({ success: true, data: alerts });
  } catch (error) {
    logger.error('Failed to get compliance alerts', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to get alerts' });
  }
});

// Create compliance alert
app.post('/api/monitor/alerts', async (req, res) => {
  try {
    const { type, severity, message, clientId, applicationId, details } = req.body;

    const alert = await afmMonitor.createComplianceAlert({
      type,
      severity,
      message,
      clientId,
      applicationId,
      details
    });

    res.status(201).json({ success: true, data: alert });
  } catch (error) {
    logger.error('Failed to create compliance alert', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to create alert' });
  }
});

// Get audit trail
app.get('/api/monitor/audit/:clientId', async (req, res) => {
  try {
    const { clientId } = req.params;
    const { startDate, endDate, limit = 100 } = req.query;

    const auditTrail = await afmMonitor.getAuditTrail(clientId, {
      startDate,
      endDate,
      limit: parseInt(limit)
    });

    res.json({ success: true, data: auditTrail });
  } catch (error) {
    logger.error('Failed to get audit trail', { error: error.message, clientId: req.params.clientId });
    res.status(500).json({ success: false, error: 'Failed to get audit trail' });
  }
});

// Generate compliance report
app.post('/api/monitor/reports', async (req, res) => {
  try {
    const { clientId, applicationId, reportType, dateRange } = req.body;

    const report = await afmMonitor.generateComplianceReport({
      clientId,
      applicationId,
      reportType,
      dateRange
    });

    res.json({ success: true, data: report });
  } catch (error) {
    logger.error('Failed to generate compliance report', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to generate report' });
  }
});

// Webhook for AFM regulation updates
app.post('/api/monitor/webhook/afm-updates', async (req, res) => {
  try {
    const { regulations, source } = req.body;

    await afmMonitor.processAFMRegulationUpdate(regulations, source);

    res.json({ success: true, message: 'Regulation update processed' });
  } catch (error) {
    logger.error('Failed to process AFM regulation update', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to process update' });
  }
});

// Get monitoring metrics
app.get('/api/monitor/metrics', async (req, res) => {
  try {
    const metrics = await afmMonitor.getMonitoringMetrics();
    res.json({ success: true, data: metrics });
  } catch (error) {
    logger.error('Failed to get monitoring metrics', { error: error.message });
    res.status(500).json({ success: false, error: 'Failed to get metrics' });
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
  logger.info(`AFM Monitor Service running on port ${PORT}`, {
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
