/**
 * Lender Integration Service
 * Handles integration with Dutch lenders (Stater, Quion, ING, Rabobank, ABN AMRO)
 *
 * This service provides unified API for:
 * - Application submission to multiple lenders
 * - Status tracking and updates
 * - Document processing and validation
 * - Lender-specific requirement handling
 * - Automated follow-up and communication
 */

const express = require('express');
const axios = require('axios');
const { Client } = require('pg');
const Redis = require('redis');
const winston = require('winston');
const Joi = require('joi');
const fs = require('fs').promises;
const path = require('path');

const app = express();

// Environment configuration
const PORT = process.env.LENDER_PORT || 8002;

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
  defaultMeta: { service: 'lender-integration' },
  transports: [
    new winston.transports.File({ filename: '/app/logs/lender-integration.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Lender configurations
const LENDER_CONFIGS = {
  stater: {
    name: 'Stater',
    apiUrl: process.env.STATER_API_URL,
    apiKey: process.env.STATER_API_KEY,
    submitEndpoint: '/api/applications',
    statusEndpoint: '/api/applications/{reference}',
    documentEndpoint: '/api/applications/{reference}/documents',
    supportedProducts: ['fixed_rate_5yr', 'fixed_rate_10yr', 'fixed_rate_20yr', 'variable_rate'],
    processingTime: '3-5 business days',
    maxLTV: 100,
    minIncome: 35000
  },
  quion: {
    name: 'Quion',
    apiUrl: process.env.QUION_API_URL,
    apiKey: process.env.QUION_API_KEY,
    submitEndpoint: '/api/mortgage-applications',
    statusEndpoint: '/api/mortgage-applications/{reference}',
    documentEndpoint: '/api/mortgage-applications/{reference}/documents',
    supportedProducts: ['fixed_rate_5yr', 'fixed_rate_10yr', 'fixed_rate_15yr', 'fixed_rate_20yr', 'variable_rate'],
    processingTime: '2-4 business days',
    maxLTV: 100,
    minIncome: 40000
  },
  ing: {
    name: 'ING',
    apiUrl: process.env.ING_API_URL,
    apiKey: process.env.ING_API_KEY,
    submitEndpoint: '/api/applications',
    statusEndpoint: '/api/applications/{reference}',
    documentEndpoint: '/api/applications/{reference}/documents',
    supportedProducts: ['fixed_rate_5yr', 'fixed_rate_10yr', 'fixed_rate_20yr', 'fixed_rate_30yr', 'variable_rate'],
    processingTime: '5-10 business days',
    maxLTV: 100,
    minIncome: 30000
  },
  rabobank: {
    name: 'Rabobank',
    apiUrl: process.env.RABOBANK_API_URL,
    apiKey: process.env.RABOBANK_API_KEY,
    submitEndpoint: '/api/applications',
    statusEndpoint: '/api/applications/{reference}',
    documentEndpoint: '/api/applications/{reference}/documents',
    supportedProducts: ['fixed_rate_5yr', 'fixed_rate_10yr', 'fixed_rate_15yr', 'fixed_rate_20yr', 'variable_rate'],
    processingTime: '5-10 business days',
    maxLTV: 100,
    minIncome: 35000
  },
  abn_amro: {
    name: 'ABN AMRO',
    apiUrl: process.env.ABN_AMRO_API_URL,
    apiKey: process.env.ABN_AMRO_API_KEY,
    submitEndpoint: '/api/applications',
    statusEndpoint: '/api/applications/{reference}',
    documentEndpoint: '/api/applications/{reference}/documents',
    supportedProducts: ['fixed_rate_5yr', 'fixed_rate_10yr', 'fixed_rate_20yr', 'fixed_rate_30yr', 'variable_rate'],
    processingTime: '5-10 business days',
    maxLTV: 100,
    minIncome: 30000
  }
};

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));

// Database connection
async function connectDatabase() {
  try {
    dbClient = new Client({
      connectionString: process.env.DATABASE_URL,
      ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
    });
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

// Lender Integration Manager
class LenderIntegrationManager {
  constructor() {
    this.activeConnections = new Map();
    this.rateLimits = new Map();
    this.requestQueue = new Map();
  }

  async submitApplication(lenderName, applicationData) {
    const config = LENDER_CONFIGS[lenderName.toLowerCase()];
    if (!config) {
      throw new Error(`Unsupported lender: ${lenderName}`);
    }

    // Check rate limits
    await this.checkRateLimit(lenderName);

    try {
      logger.info(`Submitting application to ${config.name}`, {
        lender: lenderName,
        application_id: applicationData.application_id
      });

      // Transform application data to lender format
      const lenderData = await this.transformApplicationData(applicationData, lenderName);

      // Submit to lender API
      const response = await axios.post(`${config.apiUrl}${config.submitEndpoint}`, lenderData, {
        headers: {
          'Authorization': `Bearer ${config.apiKey}`,
          'Content-Type': 'application/json',
          'X-API-Version': '2025.1'
        },
        timeout: 60000 // 60 second timeout for submissions
      });

      const submissionResult = {
        reference_number: response.data.reference_number || response.data.application_id,
        status: response.data.status || 'submitted',
        estimated_processing_time: config.processingTime,
        submitted_at: new Date().toISOString(),
        lender_response: response.data
      };

      // Store submission in database
      await this.storeSubmission(applicationData.application_id, lenderName, submissionResult);

      // Update application status
      await this.updateApplicationStatus(applicationData.application_id, lenderName, submissionResult);

      logger.info(`Application submitted successfully to ${config.name}`, {
        lender: lenderName,
        reference: submissionResult.reference_number
      });

      return submissionResult;

    } catch (error) {
      logger.error(`Application submission failed for ${config.name}:`, error);

      // Store failed submission
      await this.storeFailedSubmission(applicationData.application_id, lenderName, error);

      throw new Error(`${config.name} submission failed: ${error.response?.data?.message || error.message}`);
    }
  }

  async checkApplicationStatus(lenderName, referenceNumber) {
    const config = LENDER_CONFIGS[lenderName.toLowerCase()];
    if (!config) {
      throw new Error(`Unsupported lender: ${lenderName}`);
    }

    try {
      const statusEndpoint = config.statusEndpoint.replace('{reference}', referenceNumber);

      const response = await axios.get(`${config.apiUrl}${statusEndpoint}`, {
        headers: {
          'Authorization': `Bearer ${config.apiKey}`,
          'Accept': 'application/json'
        },
        timeout: 30000
      });

      const statusData = {
        status: response.data.status,
        status_description: response.data.status_description || this.mapStatus(response.data.status),
        last_updated: response.data.last_updated || new Date().toISOString(),
        next_steps: response.data.next_steps || [],
        documents_required: response.data.documents_required || [],
        approval_probability: response.data.approval_probability,
        estimated_completion: response.data.estimated_completion,
        comments: response.data.comments || []
      };

      // Cache status for 5 minutes
      await redisClient.setEx(`lender_status_${lenderName}_${referenceNumber}`, 300, JSON.stringify(statusData));

      return statusData;

    } catch (error) {
      logger.error(`Status check failed for ${config.name} reference ${referenceNumber}:`, error);
      throw new Error(`Status check failed: ${error.response?.data?.message || error.message}`);
    }
  }

  async uploadDocument(lenderName, referenceNumber, documentType, documentData) {
    const config = LENDER_CONFIGS[lenderName.toLowerCase()];
    if (!config) {
      throw new Error(`Unsupported lender: ${lenderName}`);
    }

    try {
      const documentEndpoint = config.documentEndpoint.replace('{reference}', referenceNumber);

      const formData = new FormData();
      formData.append('document_type', documentType);
      formData.append('file', documentData);

      const response = await axios.post(`${config.apiUrl}${documentEndpoint}`, formData, {
        headers: {
          'Authorization': `Bearer ${config.apiKey}`,
          'Content-Type': 'multipart/form-data'
        },
        timeout: 120000 // 2 minute timeout for uploads
      });

      return {
        document_id: response.data.document_id,
        uploaded_at: new Date().toISOString(),
        status: 'uploaded'
      };

    } catch (error) {
      logger.error(`Document upload failed for ${config.name}:`, error);
      throw new Error(`Document upload failed: ${error.response?.data?.message || error.message}`);
    }
  }

  async transformApplicationData(applicationData, lenderName) {
    // Transform our internal application format to lender-specific format
    const lenderConfig = LENDER_CONFIGS[lenderName.toLowerCase()];

    // Base transformation
    const lenderData = {
      applicant: {
        first_name: applicationData.client_data.first_name,
        last_name: applicationData.client_data.last_name,
        bsn: applicationData.client_data.bsn,
        date_of_birth: applicationData.client_data.date_of_birth,
        email: applicationData.client_data.email || '',
        phone: applicationData.client_data.phone || ''
      },
      property: {
        address: applicationData.client_data.address,
        type: applicationData.mortgage_details.property_type || 'house',
        value: applicationData.mortgage_details.property_value,
        usage: 'residential'
      },
      mortgage: {
        amount: applicationData.mortgage_details.loan_amount,
        term_years: applicationData.mortgage_details.term_years,
        interest_type: applicationData.mortgage_details.interest_type,
        product: applicationData.product_selection.product_name,
        nhg_requested: applicationData.mortgage_details.nhg_requested
      },
      financials: {
        annual_income: applicationData.client_data.financial_data.gross_annual_income,
        monthly_income: applicationData.client_data.financial_data.net_monthly_income,
        existing_debts: applicationData.client_data.financial_data.existing_debts,
        savings: applicationData.client_data.financial_data.savings,
        investments: applicationData.client_data.financial_data.investments
      },
      documents: applicationData.documents || [],
      metadata: {
        source: 'mortgage_ai',
        submitted_at: new Date().toISOString(),
        version: '2025.1'
      }
    };

    // Lender-specific transformations
    return this.applyLenderSpecificTransformations(lenderData, lenderName);
  }

  applyLenderSpecificTransformations(data, lenderName) {
    const lender = lenderName.toLowerCase();

    switch (lender) {
      case 'stater':
        // Stater requires specific field mappings
        return {
          ...data,
          applicant: {
            ...data.applicant,
            citizen_service_number: data.applicant.bsn // Stater uses different field name
          },
          mortgage: {
            ...data.mortgage,
            loan_to_value_ratio: ((data.mortgage.amount / data.property.value) * 100).toFixed(2)
          }
        };

      case 'quion':
        // Quion requires additional verification fields
        return {
          ...data,
          verification: {
            income_verified: false,
            identity_verified: false,
            credit_checked: true
          }
        };

      case 'ing':
        // ING has specific product naming
        return {
          ...data,
          mortgage: {
            ...data.mortgage,
            product_code: this.mapProductToING(data.mortgage.product)
          }
        };

      default:
        return data;
    }
  }

  mapProductToING(productName) {
    const productMap = {
      'ING Fixed 30 Years': 'ING_FIXED_30YR',
      'ING Fixed 20 Years': 'ING_FIXED_20YR',
      'ING Fixed 10 Years': 'ING_FIXED_10YR',
      'ING Variable Rate': 'ING_VARIABLE'
    };
    return productMap[productName] || 'ING_FIXED_20YR';
  }

  mapStatus(lenderStatus) {
    // Standardize lender status codes
    const statusMap = {
      'received': 'Application Received',
      'under_review': 'Under Review',
      'documents_requested': 'Additional Documents Required',
      'valuation_scheduled': 'Property Valuation Scheduled',
      'approved': 'Approved',
      'conditionally_approved': 'Conditionally Approved',
      'rejected': 'Rejected',
      'withdrawn': 'Withdrawn'
    };

    return statusMap[lenderStatus] || lenderStatus;
  }

  async checkRateLimit(lenderName) {
    const key = `ratelimit_${lenderName}`;
    const current = await redisClient.get(key);

    if (current && parseInt(current) >= 100) { // 100 requests per hour
      throw new Error(`Rate limit exceeded for ${lenderName}`);
    }

    await redisClient.incr(key);
    await redisClient.expire(key, 3600); // 1 hour expiry
  }

  async storeSubmission(applicationId, lenderName, submissionResult) {
    try {
      await dbClient.query(`
        INSERT INTO lender_submissions (
          application_id, lender_name, reference_number, status,
          submission_data, submitted_at, estimated_processing_time
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      `, [
        applicationId,
        lenderName,
        submissionResult.reference_number,
        submissionResult.status,
        JSON.stringify(submissionResult),
        submissionResult.submitted_at,
        submissionResult.estimated_processing_time
      ]);
    } catch (error) {
      logger.error('Failed to store lender submission:', error);
    }
  }

  async storeFailedSubmission(applicationId, lenderName, error) {
    try {
      await dbClient.query(`
        INSERT INTO lender_submissions (
          application_id, lender_name, status, error_message, submitted_at
        ) VALUES ($1, $2, $3, $4, $5)
      `, [
        applicationId,
        lenderName,
        'failed',
        error.message,
        new Date().toISOString()
      ]);
    } catch (dbError) {
      logger.error('Failed to store failed submission:', dbError);
    }
  }

  async updateApplicationStatus(applicationId, lenderName, submissionResult) {
    try {
      await dbClient.query(`
        UPDATE dutch_mortgage_applications
        SET lender_name = $1, lender_reference = $2, lender_validation_status = $3,
            submitted_at = $4, updated_at = CURRENT_TIMESTAMP
        WHERE id = $5
      `, [
        lenderName,
        submissionResult.reference_number,
        'submitted',
        submissionResult.submitted_at,
        applicationId
      ]);
    } catch (error) {
      logger.error('Failed to update application status:', error);
    }
  }

  async getSupportedLenders() {
    const lenders = Object.entries(LENDER_CONFIGS).map(([key, config]) => ({
      id: key,
      name: config.name,
      api_available: !!config.apiKey,
      supported_products: config.supportedProducts,
      processing_time: config.processingTime,
      max_ltv: config.maxLTV,
      min_income: config.minIncome,
      status: 'active'
    }));

    return lenders;
  }
}

// Service instance
const lenderManager = new LenderIntegrationManager();

// API Routes

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'lender-integration',
    timestamp: new Date().toISOString(),
    lenders: {
      stater: !!LENDER_CONFIGS.stater.apiKey,
      quion: !!LENDER_CONFIGS.quion.apiKey,
      ing: !!LENDER_CONFIGS.ing.apiKey,
      rabobank: !!LENDER_CONFIGS.rabobank.apiKey,
      abn_amro: !!LENDER_CONFIGS.abn_amro.apiKey
    }
  });
});

// Get supported lenders
app.get('/api/lenders', async (req, res) => {
  try {
    const lenders = await lenderManager.getSupportedLenders();
    res.json({ lenders });
  } catch (error) {
    logger.error('Failed to get supported lenders:', error);
    res.status(500).json({ error: 'Failed to retrieve lenders' });
  }
});

// Submit application to lender
app.post('/api/lenders/:lenderName/applications', async (req, res) => {
  try {
    const { lenderName } = req.params;
    const applicationData = req.body;

    const result = await lenderManager.submitApplication(lenderName, applicationData);

    res.json({
      success: true,
      submission: result,
      lender: lenderName,
      submitted_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error(`Application submission failed for ${req.params.lenderName}:`, error);
    res.status(500).json({
      error: 'Application submission failed',
      message: error.message
    });
  }
});

// Check application status with lender
app.get('/api/lenders/:lenderName/applications/:reference', async (req, res) => {
  try {
    const { lenderName, reference } = req.params;

    // Check cache first
    const cacheKey = `lender_status_${lenderName}_${reference}`;
    const cached = await redisClient.get(cacheKey);

    if (cached) {
      return res.json({
        success: true,
        status: JSON.parse(cached),
        cached: true
      });
    }

    const status = await lenderManager.checkApplicationStatus(lenderName, reference);

    res.json({
      success: true,
      status,
      lender: lenderName,
      reference_number: reference
    });

  } catch (error) {
    logger.error(`Status check failed for ${req.params.lenderName}/${req.params.reference}:`, error);
    res.status(500).json({
      error: 'Status check failed',
      message: error.message
    });
  }
});

// Upload document to lender
app.post('/api/lenders/:lenderName/applications/:reference/documents', async (req, res) => {
  try {
    const { lenderName, reference } = req.params;
    const { document_type, file } = req.body;

    const result = await lenderManager.uploadDocument(lenderName, reference, document_type, file);

    res.json({
      success: true,
      document: result,
      lender: lenderName,
      reference_number: reference
    });

  } catch (error) {
    logger.error(`Document upload failed for ${req.params.lenderName}/${req.params.reference}:`, error);
    res.status(500).json({
      error: 'Document upload failed',
      message: error.message
    });
  }
});

// Get lender submission history for application
app.get('/api/applications/:applicationId/lenders', async (req, res) => {
  try {
    const { applicationId } = req.params;

    const result = await dbClient.query(`
      SELECT lender_name, reference_number, status, submitted_at,
             estimated_processing_time, error_message
      FROM lender_submissions
      WHERE application_id = $1
      ORDER BY submitted_at DESC
    `, [applicationId]);

    const submissions = result.rows.map(row => ({
      lender_name: row.lender_name,
      reference_number: row.reference_number,
      status: row.status,
      submitted_at: row.submitted_at,
      estimated_processing_time: row.estimated_processing_time,
      error_message: row.error_message
    }));

    res.json({
      success: true,
      application_id: applicationId,
      submissions,
      total_submissions: submissions.length
    });

  } catch (error) {
    logger.error(`Failed to get lender submissions for application ${req.params.applicationId}:`, error);
    res.status(500).json({
      error: 'Failed to retrieve lender submissions',
      message: error.message
    });
  }
});

// Bulk status check for all lender submissions
app.post('/api/lenders/status/bulk', async (req, res) => {
  try {
    const { application_ids } = req.body;
    const results = [];

    for (const appId of application_ids) {
      try {
        const result = await dbClient.query(`
          SELECT lender_name, reference_number
          FROM lender_submissions
          WHERE application_id = $1 AND status = 'submitted'
          ORDER BY submitted_at DESC LIMIT 1
        `, [appId]);

        if (result.rows.length > 0) {
          const { lender_name, reference_number } = result.rows[0];
          const status = await lenderManager.checkApplicationStatus(lender_name, reference_number);

          results.push({
            application_id: appId,
            lender_name,
            reference_number,
            status,
            checked_at: new Date().toISOString()
          });
        }
      } catch (appError) {
        results.push({
          application_id: appId,
          error: appError.message,
          checked_at: new Date().toISOString()
        });
      }
    }

    res.json({
      success: true,
      results,
      total_checked: results.length
    });

  } catch (error) {
    logger.error('Bulk status check failed:', error);
    res.status(500).json({
      error: 'Bulk status check failed',
      message: error.message
    });
  }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('Shutting down Lender Integration service...');

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

    // Log available lenders
    const availableLenders = Object.entries(LENDER_CONFIGS)
      .filter(([_, config]) => config.apiKey)
      .map(([key, config]) => config.name);

    logger.info(`Available lenders: ${availableLenders.join(', ')}`);

    // Start HTTP server
    app.listen(PORT, () => {
      logger.info(`🚀 Lender Integration Service running on port ${PORT}`);
      logger.info(`🏛️ Available lenders: ${availableLenders.length}`);
      logger.info(`🔍 Health check: http://localhost:${PORT}/health`);
    });

  } catch (error) {
    logger.error('Failed to start Lender Integration service:', error);
    process.exit(1);
  }
}

// Export service class for use by other modules
class LenderIntegrationService {
  constructor() {
    this.lenderManager = lenderManager;
  }

  async submitApplication(lenderCode, applicationData) {
    return await this.lenderManager.submitApplication(lenderCode, applicationData);
  }

  async getApplicationStatus(lenderCode, applicationId) {
    return await this.lenderManager.getApplicationStatus(lenderCode, applicationId);
  }

  async getSupportedLenders() {
    return await this.lenderManager.getSupportedLenders();
  }
}

module.exports = {
  LenderIntegrationService
};
