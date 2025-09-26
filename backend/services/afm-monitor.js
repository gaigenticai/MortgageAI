/**
 * AFM Compliance Monitor Service
 * Real-time AFM compliance monitoring and reporting
 *
 * This service provides:
 * - Continuous compliance monitoring
 * - Automated audit trail generation
 * - Regulatory reporting and alerts
 * - Compliance violation detection
 * - AFM regulation updates tracking
 */

const express = require('express');
const axios = require('axios');
const { Client } = require('pg');
const Redis = require('redis');
const winston = require('winston');
const nodemailer = require('nodemailer');
const cron = require('node-cron');
const Joi = require('joi');
const fs = require('fs').promises;
const path = require('path');

const app = express();

// Environment configuration
const PORT = process.env.AFM_MONITOR_PORT || 8003;
const AFM_WEBHOOK_URL = process.env.AFM_AUDIT_WEBHOOK_URL;
const COMPLIANCE_EMAIL = process.env.COMPLIANCE_ALERT_EMAIL;

// Email configuration
const emailTransporter = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  port: process.env.SMTP_PORT || 587,
  secure: false,
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS
  }
});

// Database and cache clients
let dbClient;
let redisClient;

// Logger configuration
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'afm-monitor' },
  transports: [
    new winston.transports.File({ filename: '/app/logs/afm-monitor.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Database connection
async function connectDatabase() {
  try {
    const dbConfig = {
      host: process.env.DB_HOST || 'postgres',
      port: process.env.DB_PORT || 5432,
      database: process.env.DB_NAME || 'mortgage_db',
      user: process.env.DB_USER || 'mortgage_user',
      password: process.env.DB_PASSWORD || 'mortgage_pass',
      ssl: false
    };

    dbClient = new Client(dbConfig);
    await dbClient.connect();
    logger.info('Connected to PostgreSQL database');
  } catch (error) {
    logger.error('Failed to connect to database:', error);
    throw error;
  }
}

// Redis connection
async function connectRedis() {
 try {
    redisClient = Redis.createClient({
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    });

    redisClient.on('error', (err) => logger.error('Redis Client Error', err));
    redisClient.on('connect', () => logger.info('Connected to Redis'));

    await redisClient.connect();
  } catch (error) {
    logger.error('Failed to connect to Redis:', error);
    throw error;
  }
}

// AFM Compliance Monitor
class AFMComplianceMonitor {
  constructor() {
    this.activeAlerts = new Map();
    this.complianceMetrics = new Map();
    this.violationThresholds = {
      critical: 95,  // Compliance score below 95 is critical
      high: 85,      // Below 85 is high priority
      medium: 75,    // Below 75 is medium priority
      low: 65        // Below 65 is low priority
    };
  }

  async monitorCompliance() {
    try {
      // Get recent compliance checks
      const recentChecks = await this.getRecentComplianceChecks();

      for (const check of recentChecks) {
        await this.analyzeComplianceCheck(check);
      }

      // Generate compliance metrics
      await this.updateComplianceMetrics();

      logger.info(`Compliance monitoring completed for ${recentChecks.length} checks`);
    } catch (error) {
      logger.error('Compliance monitoring failed:', error);
    }
  }

  async analyzeComplianceCheck(check) {
    const complianceScore = check.compliance_score || 0;
    const riskLevel = this.determineRiskLevel(complianceScore);

    // Check for violations
    if (riskLevel === 'critical' || riskLevel === 'high') {
      await this.handleComplianceViolation(check, riskLevel);
    }

    // Update compliance metrics
    await this.updateCheckMetrics(check, riskLevel);

    // Send webhook notification if configured
    if (AFM_WEBHOOK_URL) {
      await this.sendWebhookNotification(check, riskLevel);
    }
  }

  determineRiskLevel(complianceScore) {
    if (complianceScore >= this.violationThresholds.critical) return 'compliant';
    if (complianceScore >= this.violationThresholds.high) return 'low';
    if (complianceScore >= this.violationThresholds.medium) return 'medium';
    if (complianceScore >= this.violationThresholds.low) return 'high';
    return 'critical';
  }

  async handleComplianceViolation(check, riskLevel) {
    const alertKey = `violation_${check.session_id}_${check.check_type}`;

    // Check if alert already exists
    if (this.activeAlerts.has(alertKey)) {
      return; // Alert already active
    }

    const alert = {
      id: alertKey,
      session_id: check.session_id,
      check_type: check.check_type,
      compliance_score: check.compliance_score,
      risk_level: riskLevel,
      detected_at: new Date().toISOString(),
      status: 'active',
      remediation_required: riskLevel === 'critical'
    };

    // Store alert
    this.activeAlerts.set(alertKey, alert);

    // Store in database
    await this.storeComplianceAlert(alert);

    // Send email alert
    await this.sendComplianceAlert(alert);

    logger.warn(`Compliance violation detected: ${alertKey}`, {
      session_id: check.session_id,
      risk_level: riskLevel,
      compliance_score: check.compliance_score
    });
  }

  async storeComplianceAlert(alert) {
    try {
      await dbClient.query(`
        INSERT INTO compliance_alerts (
          alert_id, session_id, alert_type, risk_level,
          details, status, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
      `, [
        alert.id,
        alert.session_id,
        alert.check_type,
        alert.risk_level,
        JSON.stringify(alert),
        alert.status
      ]);
    } catch (error) {
      logger.error('Failed to store compliance alert:', error);
    }
  }

  async sendComplianceAlert(alert) {
    if (!COMPLIANCE_EMAIL) return;

    try {
      const mailOptions = {
        from: process.env.FROM_EMAIL,
        to: COMPLIANCE_EMAIL,
        subject: `AFM Compliance Alert - ${alert.risk_level.toUpperCase()} Risk`,
        html: `
          <h2>AFM Compliance Violation Detected</h2>
          <p><strong>Risk Level:</strong> ${alert.risk_level.toUpperCase()}</p>
          <p><strong>Session ID:</strong> ${alert.session_id}</p>
          <p><strong>Check Type:</strong> ${alert.check_type}</p>
          <p><strong>Compliance Score:</strong> ${alert.compliance_score}%</p>
          <p><strong>Detected At:</strong> ${alert.detected_at}</p>
          <p><strong>Remediation Required:</strong> ${alert.remediation_required ? 'YES' : 'NO'}</p>
          <br>
          <p>Please review the compliance check details and take appropriate action.</p>
        `
      };

      await emailTransporter.sendMail(mailOptions);
      logger.info(`Compliance alert email sent for ${alert.id}`);
    } catch (error) {
      logger.error('Failed to send compliance alert email:', error);
    }
  }

  async sendWebhookNotification(check, riskLevel) {
    try {
      const payload = {
        event: 'compliance_check',
        session_id: check.session_id,
        risk_level: riskLevel,
        compliance_score: check.compliance_score,
        check_type: check.check_type,
        timestamp: new Date().toISOString(),
        details: check
      };

      await axios.post(AFM_WEBHOOK_URL, payload, {
        headers: {
          'Content-Type': 'application/json',
          'X-AFM-Monitor': 'v1.0'
        },
        timeout: 10000
      });

      logger.info(`Webhook notification sent for session ${check.session_id}`);
    } catch (error) {
      logger.error('Failed to send webhook notification:', error);
    }
  }

  async getRecentComplianceChecks(hours = 1) {
    try {
      const result = await dbClient.query(`
        SELECT acl.session_id, acl.check_type, acl.check_result,
               acl.details, s.afm_compliance_score as compliance_score,
               s.created_at, s.completed_at
        FROM afm_compliance_logs acl
        JOIN afm_advice_sessions s ON acl.session_id = s.id
        WHERE acl.checked_at >= NOW() - INTERVAL '${hours} hours'
        ORDER BY acl.checked_at DESC
      `);

      return result.rows.map(row => ({
        session_id: row.session_id,
        check_type: row.check_type,
        check_result: row.check_result,
        compliance_score: row.afm_compliance_score,
        details: row.details ? JSON.parse(row.details) : {},
        created_at: row.created_at,
        completed_at: row.completed_at
      }));
    } catch (error) {
      logger.error('Failed to get recent compliance checks:', error);
      return [];
    }
  }

  async updateComplianceMetrics() {
    try {
      // Calculate compliance metrics for the last 24 hours
      const result = await dbClient.query(`
        SELECT
          COUNT(*) as total_checks,
          AVG(afm_compliance_score) as avg_compliance_score,
          MIN(afm_compliance_score) as min_compliance_score,
          MAX(afm_compliance_score) as max_compliance_score,
          COUNT(CASE WHEN afm_compliance_score < 75 THEN 1 END) as low_compliance_count,
          COUNT(CASE WHEN afm_compliance_score < 85 THEN 1 END) as medium_compliance_count,
          COUNT(CASE WHEN afm_compliance_score < 95 THEN 1 END) as high_compliance_count
        FROM afm_advice_sessions
        WHERE completed_at >= NOW() - INTERVAL '24 hours'
      `);

      if (result.rows.length > 0) {
        const metrics = result.rows[0];
        this.complianceMetrics.set('daily', {
          ...metrics,
          calculated_at: new Date().toISOString(),
          compliance_rate: metrics.total_checks > 0 ?
            ((metrics.total_checks - metrics.low_compliance_count) / metrics.total_checks * 100).toFixed(2) : 0
        });

        // Cache metrics
        await redisClient.setEx('compliance_metrics_daily', 3600, JSON.stringify(metrics));

        logger.info('Compliance metrics updated', metrics);
      }
    } catch (error) {
      logger.error('Failed to update compliance metrics:', error);
    }
  }

  async updateCheckMetrics(check, riskLevel) {
    const key = `metrics_${check.check_type}_${riskLevel}`;
    const current = await redisClient.get(key) || 0;
    await redisClient.setEx(key, 86400, parseInt(current) + 1); // 24 hour expiry
  }

  async generateComplianceReport(startDate, endDate) {
    try {
      const result = await dbClient.query(`
        SELECT
          DATE_TRUNC('day', completed_at) as date,
          COUNT(*) as total_sessions,
          AVG(afm_compliance_score) as avg_compliance,
          MIN(afm_compliance_score) as min_compliance,
          MAX(afm_compliance_score) as max_compliance,
          COUNT(CASE WHEN afm_compliance_score >= 95 THEN 1 END) as compliant_sessions,
          COUNT(CASE WHEN afm_compliance_score < 75 THEN 1 END) as critical_sessions
        FROM afm_advice_sessions
        WHERE completed_at BETWEEN $1 AND $2
        GROUP BY DATE_TRUNC('day', completed_at)
        ORDER BY date DESC
      `, [startDate, endDate]);

      const report = {
        generated_at: new Date().toISOString(),
        period: { start: startDate, end: endDate },
        summary: {
          total_days: result.rows.length,
          total_sessions: result.rows.reduce((sum, row) => sum + parseInt(row.total_sessions), 0),
          avg_compliance: result.rows.length > 0 ?
            result.rows.reduce((sum, row) => sum + parseFloat(row.avg_compliance), 0) / result.rows.length : 0
        },
        daily_breakdown: result.rows
      };

      // Save report to file
      const reportPath = path.join('/app/compliance-reports', `compliance-report-${Date.now()}.json`);
      await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

      logger.info(`Compliance report generated: ${reportPath}`);
      return report;

    } catch (error) {
      logger.error('Failed to generate compliance report:', error);
      throw error;
    }
  }

  async getActiveAlerts() {
    const alerts = Array.from(this.activeAlerts.values());
    return alerts.filter(alert => alert.status === 'active');
  }

  async resolveAlert(alertId, resolution) {
    const alert = this.activeAlerts.get(alertId);
    if (!alert) {
      throw new Error('Alert not found');
    }

    alert.status = 'resolved';
    alert.resolved_at = new Date().toISOString();
    alert.resolution = resolution;

    // Update in database
    await dbClient.query(`
      UPDATE compliance_alerts
      SET status = $1, details = details || $2, updated_at = CURRENT_TIMESTAMP
      WHERE alert_id = $1
    `, [
      'resolved',
      JSON.stringify({ resolved_at: alert.resolved_at, resolution })
    ]);

    // Remove from active alerts
    this.activeAlerts.delete(alertId);

    logger.info(`Alert resolved: ${alertId}`);
    return alert;
  }
}

// AFM Regulation Tracker
class AFMRegulationTracker {
  constructor() {
    this.lastCheck = null;
    this.updateInterval = 24 * 60 * 60 * 1000; // 24 hours
  }

  async checkForRegulationUpdates() {
    try {
      // Check if we need to check for updates
      if (this.lastCheck && (Date.now() - this.lastCheck) < this.updateInterval) {
        return; // Too soon to check again
      }

      const response = await axios.get(process.env.AFM_REGULATION_FEED_URL, {
        headers: {
          'Authorization': `Bearer ${process.env.AFM_API_KEY}`,
          'Accept': 'application/json'
        },
        timeout: 30000
      });

      const updates = response.data.updates || [];
      let newUpdates = 0;

      for (const update of updates) {
        // Check if this update already exists
        const existing = await dbClient.query(`
          SELECT id FROM afm_regulation_updates
          WHERE regulation_code = $1 AND update_date = $2
        `, [update.regulation_code, update.effective_date]);

        if (existing.rows.length === 0) {
          // Store new update
          await dbClient.query(`
            INSERT INTO afm_regulation_updates (
              regulation_code, update_type, changes, effective_date,
              update_date, source_url
            ) VALUES ($1, $2, $3, $4, $5, $6)
          `, [
            update.regulation_code,
            update.update_type,
            JSON.stringify(update.changes),
            update.effective_date,
            update.update_date,
            update.source_url
          ]);

          newUpdates++;

          // Send notification about regulation update
          await this.notifyRegulationUpdate(update);
        }
      }

      if (newUpdates > 0) {
        logger.info(`Processed ${newUpdates} AFM regulation updates`);

        // Send summary email
        await this.sendRegulationUpdateSummary(updates);
      }

      this.lastCheck = Date.now();

    } catch (error) {
      logger.error('Failed to check for AFM regulation updates:', error);
    }
  }

  async notifyRegulationUpdate(update) {
    if (!COMPLIANCE_EMAIL) return;

    try {
      const mailOptions = {
        from: process.env.FROM_EMAIL,
        to: COMPLIANCE_EMAIL,
        subject: `AFM Regulation Update - ${update.regulation_code}`,
        html: `
          <h2>AFM Regulation Update Notification</h2>
          <p><strong>Regulation:</strong> ${update.regulation_code}</p>
          <p><strong>Update Type:</strong> ${update.update_type}</p>
          <p><strong>Effective Date:</strong> ${update.effective_date}</p>
          <p><strong>Changes:</strong></p>
          <pre>${JSON.stringify(update.changes, null, 2)}</pre>
          <br>
          <p>Please review the regulation changes and update compliance procedures as needed.</p>
        `
      };

      await emailTransporter.sendMail(mailOptions);
    } catch (error) {
      logger.error('Failed to send regulation update notification:', error);
    }
  }

  async sendRegulationUpdateSummary(updates) {
    if (!COMPLIANCE_EMAIL || updates.length === 0) return;

    try {
      const summary = updates.map(u => `${u.regulation_code} (${u.update_type})`).join(', ');

      const mailOptions = {
        from: process.env.FROM_EMAIL,
        to: COMPLIANCE_EMAIL,
        subject: `AFM Regulation Update Summary - ${updates.length} Updates`,
        html: `
          <h2>AFM Regulation Update Summary</h2>
          <p><strong>Total Updates:</strong> ${updates.length}</p>
          <p><strong>Updated Regulations:</strong> ${summary}</p>
          <p><strong>Check Date:</strong> ${new Date().toISOString()}</p>
          <br>
          <p>Please review all regulation updates and ensure compliance procedures are current.</p>
        `
      };

      await emailTransporter.sendMail(mailOptions);
      logger.info('Regulation update summary email sent');
    } catch (error) {
      logger.error('Failed to send regulation update summary:', error);
    }
  }
}

// Service instances
const complianceMonitor = new AFMComplianceMonitor();
const regulationTracker = new AFMRegulationTracker();

// API Routes

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'afm-monitor',
    timestamp: new Date().toISOString(),
    active_alerts: complianceMonitor.activeAlerts.size,
    metrics: Object.fromEntries(complianceMonitor.complianceMetrics)
  });
});

// Get compliance metrics
app.get('/api/compliance/metrics', async (req, res) => {
  try {
    const { period = 'daily' } = req.query;

    const metrics = await redisClient.get(`compliance_metrics_${period}`);
    const parsedMetrics = metrics ? JSON.parse(metrics) : null;

    res.json({
      success: true,
      period,
      metrics: parsedMetrics,
      retrieved_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Failed to get compliance metrics:', error);
    res.status(500).json({ error: 'Failed to retrieve metrics' });
  }
});

// Get active compliance alerts
app.get('/api/compliance/alerts', async (req, res) => {
  try {
    const alerts = await complianceMonitor.getActiveAlerts();

    res.json({
      success: true,
      alerts,
      total_active: alerts.length,
      retrieved_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Failed to get compliance alerts:', error);
    res.status(500).json({ error: 'Failed to retrieve alerts' });
  }
});

// Resolve compliance alert
app.post('/api/compliance/alerts/:alertId/resolve', async (req, res) => {
  try {
    const { alertId } = req.params;
    const { resolution } = req.body;

    const alert = await complianceMonitor.resolveAlert(alertId, resolution);

    res.json({
      success: true,
      alert,
      resolved_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error(`Failed to resolve alert ${req.params.alertId}:`, error);
    res.status(500).json({ error: 'Failed to resolve alert' });
  }
});

// Generate compliance report
app.post('/api/compliance/reports', async (req, res) => {
  try {
    const { start_date, end_date } = req.body;

    const report = await complianceMonitor.generateComplianceReport(
      start_date || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(), // 30 days ago
      end_date || new Date().toISOString()
    );

    res.json({
      success: true,
      report,
      generated_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Failed to generate compliance report:', error);
    res.status(500).json({ error: 'Failed to generate report' });
  }
});

// Get regulation updates
app.get('/api/afm/regulation-updates', async (req, res) => {
  try {
    const { limit = 50 } = req.query;

    const result = await dbClient.query(`
      SELECT * FROM afm_regulation_updates
      ORDER BY update_date DESC
      LIMIT $1
    `, [parseInt(limit)]);

    res.json({
      success: true,
      updates: result.rows,
      total_updates: result.rows.length,
      retrieved_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Failed to get regulation updates:', error);
    res.status(500).json({ error: 'Failed to retrieve regulation updates' });
  }
});

// Trigger manual compliance check
app.post('/api/compliance/check', async (req, res) => {
  try {
    await complianceMonitor.monitorCompliance();

    res.json({
      success: true,
      message: 'Compliance check completed',
      checked_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Manual compliance check failed:', error);
    res.status(500).json({ error: 'Compliance check failed' });
  }
});

// Trigger regulation update check
app.post('/api/afm/check-updates', async (req, res) => {
  try {
    await regulationTracker.checkForRegulationUpdates();

    res.json({
      success: true,
      message: 'Regulation update check completed',
      checked_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Regulation update check failed:', error);
    res.status(500).json({ error: 'Regulation update check failed' });
  }
});

// Get compliance dashboard data
app.get('/api/compliance/dashboard', async (req, res) => {
  try {
    const [metrics, alerts, updates] = await Promise.all([
      redisClient.get('compliance_metrics_daily'),
      complianceMonitor.getActiveAlerts(),
      dbClient.query(`
        SELECT COUNT(*) as total_updates
        FROM afm_regulation_updates
        WHERE update_date >= NOW() - INTERVAL '7 days'
      `)
    ]);

    const dashboard = {
      compliance_score: metrics ? JSON.parse(metrics).avg_compliance_score : 0,
      active_alerts: alerts.length,
      recent_regulation_updates: updates.rows[0].total_updates,
      risk_distribution: {
        critical: alerts.filter(a => a.risk_level === 'critical').length,
        high: alerts.filter(a => a.risk_level === 'high').length,
        medium: alerts.filter(a => a.risk_level === 'medium').length,
        low: alerts.filter(a => a.risk_level === 'low').length
      },
      last_updated: new Date().toISOString()
    };

    res.json({
      success: true,
      dashboard
    });

  } catch (error) {
    logger.error('Failed to get compliance dashboard:', error);
    res.status(500).json({ error: 'Failed to retrieve dashboard data' });
  }
});

// Scheduled tasks
async function initializeScheduledTasks() {
  // Run compliance monitoring every 15 minutes
  cron.schedule('*/15 * * * *', async () => {
    try {
      logger.info('Running scheduled compliance monitoring');
      await complianceMonitor.monitorCompliance();
    } catch (error) {
      logger.error('Scheduled compliance monitoring failed:', error);
    }
  });

  // Check for AFM regulation updates daily at 6 AM
  cron.schedule('0 6 * * *', async () => {
    try {
      logger.info('Checking for AFM regulation updates');
      await regulationTracker.checkForRegulationUpdates();
    } catch (error) {
      logger.error('Scheduled regulation update check failed:', error);
    }
  });

  // Generate daily compliance report at midnight
  cron.schedule('0 0 * * *', async () => {
    try {
      logger.info('Generating daily compliance report');
      const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000);
      await complianceMonitor.generateComplianceReport(
        yesterday.toISOString().split('T')[0] + 'T00:00:00Z',
        new Date().toISOString().split('T')[0] + 'T23:59:59Z'
      );
    } catch (error) {
      logger.error('Scheduled compliance report generation failed:', error);
    }
  });
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('Shutting down AFM Monitor service...');

  if (dbClient) {
    await dbClient.end();
  }

  if (redisClient?.isOpen) {
    await redisClient.quit();
  }

  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('Received SIGINT, shutting down gracefully...');

  if (dbClient) {
    await dbClient.end();
  }

  if (redisClient?.isOpen) {
    await redisClient.quit();
  }

  process.exit(0);
});

// Start server
async function startServer() {
  try {
    // Initialize connections
    await connectDatabase();
    await connectRedis();

    // Initialize scheduled tasks
    await initializeScheduledTasks();

    // Run initial compliance check
    setTimeout(() => {
      complianceMonitor.monitorCompliance().catch(error =>
        logger.error('Initial compliance check failed:', error)
      );
    }, 5000); // Wait 5 seconds after startup

    // Start HTTP server
    app.listen(PORT, () => {
      logger.info(`🚀 AFM Monitor Service running on port ${PORT}`);
      logger.info(`📊 Active alerts: ${complianceMonitor.activeAlerts.size}`);
      logger.info(`⚖️ Health check: http://localhost:${PORT}/health`);
    });

  } catch (error) {
    logger.error('Failed to start AFM Monitor service:', error);
    process.exit(1);
  }
}

// Export service class for use by other modules
class AFMMonitorService {
  constructor() {
    this.complianceMonitor = complianceMonitor;
  }

  async startMonitoring() {
    return await this.complianceMonitor.startMonitoring();
  }

  async checkCompliance(sessionId) {
    return await this.complianceMonitor.checkCompliance(sessionId);
  }

  async getActiveAlerts() {
    return await this.complianceMonitor.getActiveAlerts();
  }
}

module.exports = {
  AFMMonitorService
};

// Export service class for use by other modules - do not start server here
// The server is started by afm-monitor-server.js
