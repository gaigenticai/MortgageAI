/**
 * Dutch Mortgage Quality Control API Routes
 * Integrates with Dutch lenders (Stater, Quion) and validation systems (BKR, NHG)
 *
 * This module provides RESTful endpoints for mortgage application analysis,
 * BKR credit checks, NHG eligibility assessment, and lender submission.
 * All endpoints include comprehensive error handling, validation,
 * and database operations for production use.
 */

// Dutch Mortgage Quality Control API Routes - Fastify Version
// Integrates with Dutch lenders (Stater, Quion) and validation systems (BKR, NHG)
//
// This module provides RESTful endpoints for mortgage application analysis,
// BKR credit checks, NHG eligibility assessment, and lender submission.
// All endpoints include comprehensive error handling, validation,
// and database operations for production use.
const { Client } = require('pg');
const axios = require('axios');
const Joi = require('joi');
const { v4: uuidv4 } = require('uuid');

// Database configuration
const dbConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'mortgage_ai',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD || '',
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
};

// Agent service configuration
const AGENTS_API_URL = process.env.AGENTS_API_URL || 'http://ai-agents:8000';

// External API configurations
const BKR_API_CONFIG = {
  url: process.env.BKR_API_URL || 'https://api.bkr.nl/v2',
  key: process.env.BKR_API_KEY,
  timeout: 15000
};

const NHG_API_CONFIG = {
  url: process.env.NHG_API_URL || 'https://api.nhg.nl/v1',
  key: process.env.NHG_API_KEY,
  timeout: 15000
};

const LENDER_APIS = {
  stater: {
    api_url: process.env.STATER_API_URL || 'https://api.stater.nl/v2',
    api_key: process.env.STATER_API_KEY,
    submission_endpoint: '/applications',
    status_endpoint: '/applications/{reference}/status'
  },
  quion: {
    api_url: process.env.QUION_API_URL || 'https://api.quion.nl/v1',
    api_key: process.env.QUION_API_KEY,
    submission_endpoint: '/mortgage-applications',
    status_endpoint: '/mortgage-applications/{reference}'
  },
  ing: {
    api_url: process.env.ING_API_URL || 'https://api.ing.nl/mortgages/v1',
    api_key: process.env.ING_API_KEY,
    submission_endpoint: '/applications',
    status_endpoint: '/applications/{reference}'
  },
  rabobank: {
    api_url: process.env.RABOBANK_API_URL || 'https://api.rabobank.nl/mortgages/v1',
    api_key: process.env.RABOBANK_API_KEY,
    submission_endpoint: '/applications',
    status_endpoint: '/applications/{reference}'
  },
  abn_amro: {
    api_url: process.env.ABN_AMRO_API_URL || 'https://api.abnamro.nl/mortgages/v1',
    api_key: process.env.ABN_AMRO_API_KEY,
    submission_endpoint: '/applications',
    status_endpoint: '/applications/{reference}'
  }
};

// Validation schemas
const applicationDataSchema = Joi.object({
  application_id: Joi.string().uuid().required(),
  client_data: Joi.object({
    bsn: Joi.string().pattern(/^\d{9}$/).required(),
    first_name: Joi.string().required(),
    last_name: Joi.string().required(),
    date_of_birth: Joi.date().required(),
    address: Joi.object({
      street: Joi.string().required(),
      house_number: Joi.string().required(),
      postal_code: Joi.string().pattern(/^\d{4}[A-Z]{2}$/).required(),
      city: Joi.string().required()
    }).required(),
    financial_data: Joi.object({
      gross_annual_income: Joi.number().min(0).required(),
      net_monthly_income: Joi.number().min(0).required(),
      existing_debts: Joi.array().items(Joi.object({
        type: Joi.string().required(),
        creditor: Joi.string().required(),
        monthly_payment: Joi.number().min(0).required(),
        remaining_amount: Joi.number().min(0).required()
      })).default([]),
      savings: Joi.number().min(0).default(0),
      investments: Joi.number().min(0).default(0)
    }).required()
  }).required(),
  mortgage_details: Joi.object({
    property_value: Joi.number().min(0).required(),
    loan_amount: Joi.number().min(0).required(),
    down_payment: Joi.number().min(0).default(0),
    term_years: Joi.number().min(1).max(40).required(),
    interest_type: Joi.string().valid('fixed', 'variable').required(),
    nhg_requested: Joi.boolean().default(false)
  }).required(),
  documents: Joi.array().items(Joi.object({
    type: Joi.string().required(),
    filename: Joi.string().required(),
    uploaded_at: Joi.date().optional()
  })).default([]),
  product_selection: Joi.object({
    lender_name: Joi.string().valid('stater', 'quion', 'ing', 'rabobank', 'abn_amro').required(),
    product_name: Joi.string().required(),
    interest_rate: Joi.number().min(0).required(),
    features: Joi.object().required()
  }).required()
});

const bkrCheckSchema = Joi.object({
  bsn: Joi.string().pattern(/^\d{9}$/).required(),
  consent_given: Joi.boolean().valid(true).required()
});

const lenderSubmissionSchema = Joi.object({
  lender_name: Joi.string().valid('stater', 'quion', 'ing', 'rabobank', 'abn_amro').required(),
  additional_documents: Joi.array().items(Joi.object({
    type: Joi.string().required(),
    filename: Joi.string().required(),
    content: Joi.binary().required()
  })).default([])
});

// Database helper functions
async function getDbConnection() {
  const client = new Client(dbConfig);
  await client.connect();
  return client;
}

async function storeMortgageQCResults(applicationId, qcResults) {
  const client = await getDbConnection();

  try {
    const query = `
      INSERT INTO dutch_mortgage_qc_results (
        id, application_id, qc_report, overall_score, ready_for_submission,
        ftr_probability, analyzed_at, qc_agent_version
      ) VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP, $7)
      ON CONFLICT (application_id)
      DO UPDATE SET
        qc_report = EXCLUDED.qc_report,
        overall_score = EXCLUDED.overall_score,
        ready_for_submission = EXCLUDED.ready_for_submission,
        ftr_probability = EXCLUDED.ftr_probability,
        analyzed_at = CURRENT_TIMESTAMP
    `;

    await client.query(query, [
      uuidv4(),
      applicationId,
      JSON.stringify(qcResults),
      qcResults.qc_summary?.overall_score || 0,
      qcResults.qc_summary?.ready_for_submission || false,
      qcResults.qc_summary?.first_time_right_probability || 0,
      '2.0'
    ]);
  } finally {
    await client.end();
  }
}

async function getMortgageApplication(applicationId) {
  const client = await getDbConnection();

  try {
    const query = `
      SELECT ma.*, mq.qc_report as qc_results, mq.ready_for_submission,
             mq.ftr_probability as first_time_right_probability
      FROM dutch_mortgage_applications ma
      LEFT JOIN dutch_mortgage_qc_results mq ON ma.application_number = mq.application_id
      WHERE ma.id = $1
    `;

    const result = await client.query(query, [applicationId]);

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];
    return {
      id: row.id,
      application_number: row.application_number,
      status: row.qc_status,
      applicant_data: typeof row.application_data === 'string' ? JSON.parse(row.application_data) : row.application_data,
      documents: typeof row.documents === 'string' ? JSON.parse(row.documents) : row.documents,
      qc_results: row.qc_results ? (typeof row.qc_results === 'string' ? JSON.parse(row.qc_results) : row.qc_results) : null,
      ready_for_submission: row.ready_for_submission,
      first_time_right_probability: row.first_time_right_probability,
      lender_name: row.lender_name,
      lender_reference: row.lender_reference,
      submission_status: row.lender_validation_status,
      submitted_at: row.submitted_at,
      created_at: row.created_at,
      updated_at: row.updated_at
    };
  } finally {
    await client.end();
  }
}

async function getMortgageQCResults(applicationId) {
  const client = await getDbConnection();

  try {
    const query = `
      SELECT * FROM dutch_mortgage_qc_results
      WHERE application_id = $1
      ORDER BY analyzed_at DESC
      LIMIT 1
    `;

    const result = await client.query(query, [applicationId]);

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];
    return {
      id: row.id,
      application_id: row.application_id,
      qc_data: typeof row.qc_report === 'string' ? JSON.parse(row.qc_report) : row.qc_report,
      overall_score: row.overall_score,
      ready_for_submission: row.ready_for_submission,
      first_time_right_probability: row.ftr_probability,
      created_at: row.analyzed_at,
      updated_at: row.analyzed_at
    };
  } finally {
    await client.end();
  }
}

async function storeBKRCheck(bkrData) {
  const client = await getDbConnection();

  try {
    const query = `
      INSERT INTO bkr_checks (
        id, client_id, bkr_reference, credit_score,
        negative_registrations, suitability_analysis, checked_at, created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
    `;

    await client.query(query, [
      uuidv4(),
      bkrData.client_id,
      bkrData.bkr_reference,
      bkrData.credit_score,
      JSON.stringify(bkrData.negative_registrations || []),
      JSON.stringify(bkrData.suitability_analysis || {}),
      bkrData.checked_at
    ]);
  } finally {
    await client.end();
  }
}

async function updateMortgageApplication(applicationId, updates) {
  const client = await getDbConnection();

  try {
    const setClause = Object.keys(updates).map((key, index) => `${key} = $${index + 2}`).join(', ');
    const values = [applicationId, ...Object.values(updates)];

    const query = `
      UPDATE dutch_mortgage_applications
      SET ${setClause}
      WHERE id = $1
    `;

    await client.query(query, values);
  } finally {
    await client.end();
  }
}

// External service integrations
async function callBKRService(bsn) {
  try {
    const response = await axios.post(`${BKR_API_CONFIG.url}/credit-check`, {
      bsn: bsn,
      purpose: 'mortgage_application',
      consent_timestamp: new Date().toISOString()
    }, {
      headers: {
        'Authorization': `Bearer ${BKR_API_CONFIG.key}`,
        'Content-Type': 'application/json'
      },
      timeout: BKR_API_CONFIG.timeout
    });

    return response.data;
  } catch (error) {
    console.error('BKR API call failed:', error.message);
    throw new Error(`BKR service unavailable: ${error.message}`);
  }
}

async function callNHGService(eligibilityData) {
  try {
    const response = await axios.post(`${NHG_API_CONFIG.url}/eligibility-check`, eligibilityData, {
      headers: {
        'Authorization': `Bearer ${NHG_API_CONFIG.key}`,
        'Content-Type': 'application/json'
      },
      timeout: NHG_API_CONFIG.timeout
    });

    return response.data;
  } catch (error) {
    console.error('NHG API call failed:', error.message);
    throw new Error(`NHG service unavailable: ${error.message}`);
  }
}

async function submitToLender(lenderName, submissionData) {
  const lenderConfig = LENDER_APIS[lenderName.toLowerCase()];
  if (!lenderConfig) {
    throw new Error(`Unsupported lender: ${lenderName}`);
  }

  try {
    const response = await axios.post(
      `${lenderConfig.api_url}${lenderConfig.submission_endpoint}`,
      submissionData,
      {
        headers: {
          'Authorization': `Bearer ${lenderConfig.api_key}`,
          'Content-Type': 'application/json'
        },
        timeout: 30000
      }
    );

    return response.data;
  } catch (error) {
    console.error(`${lenderName} API submission failed:`, error.message);
    throw new Error(`${lenderName} submission failed: ${error.message}`);
  }
}

async function checkLenderStatus(lenderName, referenceNumber) {
  const lenderConfig = LENDER_APIS[lenderName.toLowerCase()];
  if (!lenderConfig) {
    throw new Error(`Unsupported lender: ${lenderName}`);
  }

  try {
    const endpoint = lenderConfig.status_endpoint.replace('{reference}', referenceNumber);
    const response = await axios.get(`${lenderConfig.api_url}${endpoint}`, {
      headers: {
        'Authorization': `Bearer ${lenderConfig.api_key}`
      },
      timeout: 15000
    });

    return response.data;
  } catch (error) {
    console.error(`${lenderName} status check failed:`, error.message);
    throw new Error(`${lenderName} status check failed: ${error.message}`);
  }
}

// Quality Control Agent integration
async function callQCComplianceAgent(endpoint, data) {
  try {
    const response = await axios.post(`${AGENTS_API_URL}/api/quality-control${endpoint}`, data, {
      timeout: 45000, // 45 second timeout for complex QC operations
      headers: {
        'Content-Type': 'application/json'
      }
    });

    return response.data;
  } catch (error) {
    console.error(`QC Agent call failed for ${endpoint}:`, error.message);
    throw new Error(`Quality control service unavailable: ${error.message}`);
  }
}

// Utility functions
function calculateResolutionTime(remediationPlan) {
  if (!remediationPlan || remediationPlan.length === 0) {
    return '1 hour';
  }

  const timeEstimates = {
    'low': 2,
    'medium': 8,
    'high': 24,
    'critical': 72
  };

  const maxSeverity = remediationPlan.reduce((max, item) =>
    timeEstimates[item.severity] > timeEstimates[max] ? item.severity : max,
    'low'
  );

  return `${timeEstimates[maxSeverity]} hours`;
}

// Dutch Mortgage QC Routes for Fastify
async function dutchMortgageQCRoutes(fastify, options) {

  // Run QC analysis for application (matching frontend expectation)
  fastify.post('/qc/run/:applicationId', async (request, reply) => {
    try {
      const { applicationId } = request.params;

      // Validate request data
      const validationResult = applicationDataSchema.validate(request.body, { abortEarly: false });
      if (validationResult.error) {
        reply.code(400).send({
          error: 'Validation failed',
          details: validationResult.error.details.map(detail => ({
            field: detail.path.join('.'),
            message: detail.message
          }))
        });
        return;
      }

      const applicationData = {
        ...request.body,
        application_id: applicationId
      };

      console.log(`Analyzing mortgage application ${applicationId}`);

      // Call QC agent for comprehensive analysis
      const qcResults = await callQCComplianceAgent('/analyze-application', applicationData);

      // Store QC results
      await storeMortgageQCResults(applicationData.application_id, qcResults);

      // Prepare processing recommendation
      const recommendation = {
        action: qcResults.qc_summary.ready_for_submission ? 'submit_to_lender' : 'complete_remediation',
        priority: qcResults.qc_summary.first_time_right_probability >= 80 ? 'high' : 'medium',
        estimated_resolution_time: calculateResolutionTime(qcResults.remediation_plan),
        confidence_score: qcResults.qc_summary.overall_score
      };

      reply.send({
        success: true,
        qc_results: qcResults,
        application_status: qcResults.qc_summary.ready_for_submission ? 'ready' : 'requires_attention',
        processing_recommendation: recommendation,
        metadata: {
          analyzed_at: new Date().toISOString(),
          application_id: applicationData.application_id,
          processing_time: Date.now() - request.startTime || 0
        }
      });

    } catch (error) {
      console.error('Mortgage application analysis error:', error);
      reply.code(500).send({
        error: 'Mortgage application analysis failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

// BKR credit check integration
router.post('/bkr-check/:client_id', async (req, res) => {
  try {
    const { client_id } = req.params;
    const { bsn, consent_given } = req.body;

    // Validate BKR check request
    const validationResult = bkrCheckSchema.validate({ bsn, consent_given }, { abortEarly: false });
    if (validationResult.error) {
      return res.status(400).json({
        error: 'Validation failed',
        details: validationResult.error.details.map(detail => ({
          field: detail.path.join('.'),
          message: detail.message
        }))
      });
    }

    if (!consent_given) {
      return res.status(400).json({
        error: 'Client consent required for BKR check',
        code: 'CONSENT_REQUIRED'
      });
    }

    console.log(`Performing BKR credit check for client ${client_id}`);

    // Perform BKR credit check
    const bkrResult = await callBKRService(bsn);

    // Analyze BKR results for mortgage suitability
    const suitabilityAnalysis = await callQCComplianceAgent('/analyze-bkr-suitability', {
      bkr_data: bkrResult,
      client_id: client_id
    });

    // Store BKR check results
    await storeBKRCheck({
      client_id,
      bkr_reference: bkrResult.reference_number || `BKR_${Date.now()}`,
      credit_score: bkrResult.credit_score,
      negative_registrations: bkrResult.negative_registrations || [],
      suitability_analysis: suitabilityAnalysis,
      checked_at: new Date().toISOString()
    });

    res.json({
      success: true,
      bkr_results: {
        credit_score: bkrResult.credit_score,
        risk_assessment: suitabilityAnalysis.risk_level,
        approval_likelihood: suitabilityAnalysis.approval_likelihood,
        negative_factors: bkrResult.negative_registrations.length,
        recommendations: suitabilityAnalysis.recommendations
      },
      mortgage_impact: {
        affects_eligibility: suitabilityAnalysis.blocks_mortgage,
        interest_rate_impact: suitabilityAnalysis.rate_impact,
        required_actions: suitabilityAnalysis.required_actions
      },
      metadata: {
        checked_at: new Date().toISOString(),
        bkr_reference: bkrResult.reference_number,
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('BKR check error:', error);
    res.status(500).json({
      error: 'BKR credit check failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// NHG eligibility assessment
router.post('/nhg-eligibility/:application_id', async (req, res) => {
  try {
    const { application_id } = req.params;

    console.log(`Assessing NHG eligibility for application ${application_id}`);

    // Retrieve application data
    const applicationData = await getMortgageApplication(application_id);
    if (!applicationData) {
      return res.status(404).json({
        error: 'Application not found',
        application_id: application_id
      });
    }

    // Check NHG eligibility using QC agent
    const nhgEligibility = await callQCComplianceAgent('/validate-nhg-eligibility', {
      application_data: applicationData
    });

    // Calculate NHG benefits and costs
    const nhgAnalysis = await callNHGService({
      property_value: applicationData.applicant_data.mortgage_details.property_value,
      loan_amount: applicationData.applicant_data.mortgage_details.loan_amount,
      applicant_income: applicationData.applicant_data.financial_data.gross_annual_income,
      existing_debts: applicationData.applicant_data.financial_data.existing_debts
    });

    // Determine recommendation
    const netBenefit = nhgAnalysis.total_savings - nhgAnalysis.nhg_premium;
    const recommendation = {
      apply_for_nhg: nhgEligibility.eligible && netBenefit > 0,
      reasons: nhgAnalysis.recommendation_reasons,
      estimated_savings: nhgAnalysis.total_savings,
      nhg_costs: nhgAnalysis.nhg_premium,
      net_benefit: netBenefit
    };

    res.json({
      success: true,
      nhg_eligibility: nhgEligibility,
      financial_analysis: nhgAnalysis,
      recommendation: recommendation,
      metadata: {
        assessed_at: new Date().toISOString(),
        application_id: application_id,
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('NHG eligibility check error:', error);
    res.status(500).json({
      error: 'NHG eligibility assessment failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Submit to lender (Stater, Quion, etc.)
router.post('/submit-to-lender/:application_id', async (req, res) => {
  try {
    const { application_id } = req.params;
    const { lender_name, additional_documents } = req.body;

    // Validate lender submission request
    const validationResult = lenderSubmissionSchema.validate(
      { lender_name, additional_documents },
      { abortEarly: false }
    );
    if (validationResult.error) {
      return res.status(400).json({
        error: 'Validation failed',
        details: validationResult.error.details.map(detail => ({
          field: detail.path.join('.'),
          message: detail.message
        }))
      });
    }

    console.log(`Submitting application ${application_id} to lender ${lender_name}`);

    // Retrieve application and QC results
    const applicationData = await getMortgageApplication(application_id);
    const qcResults = await getMortgageQCResults(application_id);

    if (!qcResults || !qcResults.qc_data.qc_summary.ready_for_submission) {
      return res.status(400).json({
        error: 'Application not ready for submission',
        required_actions: qcResults?.qc_data?.remediation_plan?.filter(r => r.severity === 'critical') || [],
        code: 'NOT_READY_FOR_SUBMISSION'
      });
    }

    // Prepare lender-specific submission package
    const submissionPackage = await callQCComplianceAgent('/prepare-lender-submission', {
      application_data: applicationData,
      lender_name: lender_name,
      additional_documents: additional_documents || []
    });

    // Submit to lender via their API
    const lenderResponse = await submitToLender(lender_name, submissionPackage);

    // Update application status
    await updateMortgageApplication(application_id, {
      lender_name: lender_name,
      submission_status: 'submitted',
      lender_reference: lenderResponse.reference_number,
      submitted_at: new Date().toISOString(),
      estimated_response_time: lenderResponse.estimated_processing_time || '5-10 business days'
    });

    res.json({
      success: true,
      submission_status: 'submitted',
      lender_response: lenderResponse,
      tracking: {
        reference_number: lenderResponse.reference_number,
        estimated_processing_time: lenderResponse.estimated_processing_time || '5-10 business days',
        status_check_url: `/dutch-mortgage-qc/check-status/${application_id}`,
        lender_name: lender_name
      },
      metadata: {
        submitted_at: new Date().toISOString(),
        application_id: application_id,
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Lender submission error:', error);
    res.status(500).json({
      error: 'Lender submission failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Check application status with lender
router.get('/check-status/:application_id', async (req, res) => {
  try {
    const { application_id } = req.params;

    console.log(`Checking lender status for application ${application_id}`);

    // Retrieve application
    const applicationData = await getMortgageApplication(application_id);
    if (!applicationData || !applicationData.lender_reference) {
      return res.status(400).json({
        error: 'Application not yet submitted to lender',
        application_id: application_id,
        code: 'NOT_SUBMITTED'
      });
    }

    // Check status with lender
    const lenderStatus = await checkLenderStatus(
      applicationData.lender_name,
      applicationData.lender_reference
    );

    // Update application with latest status
    await updateMortgageApplication(application_id, {
      lender_status: lenderStatus.status,
      lender_status_updated: new Date().toISOString(),
      lender_comments: lenderStatus.comments || null
    });

    // Calculate processing progress
    const processingProgress = {
      current_stage: lenderStatus.processing_stage || 'unknown',
      completion_percentage: lenderStatus.completion_percentage || 0,
      next_milestone: lenderStatus.next_milestone || null,
      estimated_completion: lenderStatus.estimated_completion || null,
      days_since_submission: applicationData.submitted_at ?
        Math.floor((Date.now() - new Date(applicationData.submitted_at).getTime()) / (1000 * 60 * 60 * 24)) : 0
    };

    res.json({
      success: true,
      application_status: lenderStatus,
      processing_progress: processingProgress,
      metadata: {
        checked_at: new Date().toISOString(),
        application_id: application_id,
        lender_name: applicationData.lender_name,
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Status check error:', error);
    res.status(500).json({
      error: 'Status check failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get supported lenders
router.get('/supported-lenders', async (req, res) => {
  try {
    const lenders = Object.keys(LENDER_APIS).map(lenderKey => ({
      name: lenderKey,
      display_name: lenderKey.charAt(0).toUpperCase() + lenderKey.slice(1),
      api_available: !!LENDER_APIS[lenderKey].api_key,
      submission_endpoint: LENDER_APIS[lenderKey].submission_endpoint,
      status_endpoint: LENDER_APIS[lenderKey].status_endpoint
    }));

    res.json({
      success: true,
      lenders: lenders,
      metadata: {
        total_supported: lenders.length,
        retrieved_at: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Supported lenders retrieval error:', error);
    res.status(500).json({
      error: 'Failed to retrieve supported lenders',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get QC results for application (matching frontend expectation)
router.get('/qc/result/:applicationId', async (req, res) => {
  try {
    const { applicationId } = req.params;

    console.log(`Retrieving QC results for application ${applicationId}`);

    // Get QC results from database
    const qcResults = await getMortgageQCResults(applicationId);

    if (!qcResults) {
      return res.status(404).json({
        error: 'QC results not found',
        application_id: applicationId,
        message: 'Run QC analysis first'
      });
    }

    res.json({
      success: true,
      qc_result: {
        application_id: applicationId,
        overall_score: qcResults.overall_score,
        ready_for_submission: qcResults.ready_for_submission,
        ftr_probability: qcResults.first_time_right_probability,
        qc_report: qcResults.qc_data,
        analyzed_at: qcResults.created_at
      },
      metadata: {
        retrieved_at: new Date().toISOString(),
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('QC results retrieval error:', error);
    res.status(500).json({
      error: 'QC results retrieval failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get supported lenders (matching frontend expectation)
router.get('/lenders', async (req, res) => {
  try {
    const lenders = Object.keys(LENDER_APIS).map(lenderKey => ({
      id: lenderKey,
      name: lenderKey.charAt(0).toUpperCase() + lenderKey.slice(1),
      display_name: lenderKey.charAt(0).toUpperCase() + lenderKey.slice(1).replace('_', ' '),
      api_available: !!LENDER_APIS[lenderKey].api_key,
      supported_products: ['fixed_rate', 'variable_rate', 'nhg_mortgages'],
      processing_time_days: lenderKey === 'stater' ? '3-5' : lenderKey === 'quion' ? '2-4' : '5-10'
    }));

    res.json({
      success: true,
      lenders: lenders,
      metadata: {
        total_supported: lenders.length,
        retrieved_at: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Supported lenders retrieval error:', error);
    res.status(500).json({
      error: 'Failed to retrieve supported lenders',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// BKR Credit Check endpoints
router.post('/bkr-check', async (req, res) => {
  try {
    const { client_id, bsn, purpose, include_payment_history, include_inquiries, consent_given, consent_timestamp } = req.body;

    if (!consent_given) {
      return res.status(400).json({
        error: 'Client consent required for BKR check',
        code: 'CONSENT_REQUIRED'
      });
    }

    console.log(`Requesting BKR credit check for client ${client_id}`);

    // Create credit check request record
    const requestId = uuidv4();
    const client = await getDbConnection();

    try {
      await client.query(`
        INSERT INTO bkr_credit_check_requests (
          request_id, client_id, bsn, purpose, status, include_payment_history,
          include_inquiries, consent_given, consent_timestamp, created_at
        ) VALUES ($1, $2, $3, $4, 'pending', $5, $6, $7, $8, NOW())
      `, [requestId, client_id, bsn, purpose, include_payment_history, include_inquiries, consent_given, consent_timestamp]);

      // Call BKR service
      const bkrResult = await callBKRService(bsn);

      // Store credit report
      const reportId = uuidv4();
      await client.query(`
        INSERT INTO bkr_credit_reports (
          report_id, request_id, client_id, bsn, credit_score, score_range,
          report_data, payment_history, credit_utilization, inquiries, recommendations,
          mortgage_eligibility, risk_indicators, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW())
      `, [
        reportId, requestId, client_id, bsn, bkrResult.credit_score,
        bkrResult.score_range, JSON.stringify(bkrResult),
        JSON.stringify(bkrResult.payment_history),
        JSON.stringify(bkrResult.credit_utilization),
        JSON.stringify(bkrResult.inquiries),
        JSON.stringify(bkrResult.recommendations),
        JSON.stringify(bkrResult.mortgage_eligibility),
        JSON.stringify(bkrResult.risk_indicators)
      ]);

      // Update request status
      await client.query(
        'UPDATE bkr_credit_check_requests SET status = $1, completed_at = NOW() WHERE request_id = $2',
        ['completed', requestId]
      );

      res.json({
        success: true,
        request_id: requestId,
        status: 'completed',
        estimated_completion_time: new Date().toISOString(),
        report: bkrResult
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('BKR credit check request error:', error);
    res.status(500).json({
      error: 'BKR credit check failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get credit check status
router.get('/bkr-check/:requestId/status', async (req, res) => {
  try {
    const { requestId } = req.params;
    const client = await getDbConnection();

    try {
      const result = await client.query(
        'SELECT * FROM bkr_credit_check_requests WHERE request_id = $1',
        [requestId]
      );

      if (result.rows.length === 0) {
        return res.status(404).json({
          error: 'Credit check request not found'
        });
      }

      const request = result.rows[0];
      let report = null;

      if (request.status === 'completed') {
        const reportResult = await client.query(
          'SELECT * FROM bkr_credit_reports WHERE request_id = $1',
          [requestId]
        );
        if (reportResult.rows.length > 0) {
          report = reportResult.rows[0];
        }
      }

      res.json({
        request_id: requestId,
        status: request.status,
        estimated_completion_time: request.status === 'pending' ?
          new Date(Date.now() + 300000).toISOString() : // 5 minutes from now
          request.completed_at,
        report: report
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('Credit check status error:', error);
    res.status(500).json({
      error: 'Failed to get credit check status',
      message: error.message
    });
  }
});

// Get credit report by ID
router.get('/bkr-report/:reportId', async (req, res) => {
  try {
    const { reportId } = req.params;
    const client = await getDbConnection();

    try {
      const result = await client.query(
        'SELECT * FROM bkr_credit_reports WHERE report_id = $1',
        [reportId]
      );

      if (result.rows.length === 0) {
        return res.status(404).json({
          error: 'Credit report not found'
        });
      }

      const report = result.rows[0];

      res.json({
        id: report.report_id,
        bsn: report.bsn,
        credit_score: report.credit_score,
        score_range: report.score_range,
        report_date: report.created_at,
        last_updated: report.created_at,
        active_loans: JSON.parse(report.report_data).active_loans || [],
        payment_history: JSON.parse(report.payment_history),
        credit_utilization: JSON.parse(report.credit_utilization),
        inquiries: JSON.parse(report.inquiries),
        recommendations: JSON.parse(report.recommendations),
        mortgage_eligibility: JSON.parse(report.mortgage_eligibility),
        risk_indicators: JSON.parse(report.risk_indicators)
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('Credit report retrieval error:', error);
    res.status(500).json({
      error: 'Failed to get credit report',
      message: error.message
    });
  }
});

// Get credit report by client ID
router.get('/bkr-client/:clientId/credit-report', async (req, res) => {
  try {
    const { clientId } = req.params;
    const client = await getDbConnection();

    try {
      const result = await client.query(`
        SELECT r.* FROM bkr_credit_reports r
        JOIN bkr_credit_check_requests req ON r.request_id = req.request_id
        WHERE req.client_id = $1
        ORDER BY r.created_at DESC
        LIMIT 1
      `, [clientId]);

      if (result.rows.length === 0) {
        return res.status(404).json({
          error: 'No credit report found for client'
        });
      }

      const report = result.rows[0];

      res.json({
        id: report.report_id,
        bsn: report.bsn,
        credit_score: report.credit_score,
        score_range: report.score_range,
        report_date: report.created_at,
        last_updated: report.created_at,
        active_loans: JSON.parse(report.report_data).active_loans || [],
        payment_history: JSON.parse(report.payment_history),
        credit_utilization: JSON.parse(report.credit_utilization),
        inquiries: JSON.parse(report.inquiries),
        recommendations: JSON.parse(report.recommendations),
        mortgage_eligibility: JSON.parse(report.mortgage_eligibility),
        risk_indicators: JSON.parse(report.risk_indicators)
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('Client credit report retrieval error:', error);
    res.status(500).json({
      error: 'Failed to get client credit report',
      message: error.message
    });
  }
});

// Refresh credit report
router.post('/bkr-report/:reportId/refresh', async (req, res) => {
  try {
    const { reportId } = req.params;
    const client = await getDbConnection();

    try {
      // Get existing report
      const existingResult = await client.query(
        'SELECT * FROM bkr_credit_reports WHERE report_id = $1',
        [reportId]
      );

      if (existingResult.rows.length === 0) {
        return res.status(404).json({
          error: 'Credit report not found'
        });
      }

      const existingReport = existingResult.rows[0];

      // Request new BKR check
      const newBkrResult = await callBKRService(existingReport.bsn);

      // Update existing report
      await client.query(`
        UPDATE bkr_credit_reports SET
          credit_score = $1, score_range = $2, report_data = $3,
          payment_history = $4, credit_utilization = $5, inquiries = $6,
          recommendations = $7, mortgage_eligibility = $8, risk_indicators = $9,
          updated_at = NOW()
        WHERE report_id = $10
      `, [
        newBkrResult.credit_score, newBkrResult.score_range,
        JSON.stringify(newBkrResult),
        JSON.stringify(newBkrResult.payment_history),
        JSON.stringify(newBkrResult.credit_utilization),
        JSON.stringify(newBkrResult.inquiries),
        JSON.stringify(newBkrResult.recommendations),
        JSON.stringify(newBkrResult.mortgage_eligibility),
        JSON.stringify(newBkrResult.risk_indicators),
        reportId
      ]);

      res.json({
        success: true,
        request_id: uuidv4(),
        status: 'completed',
        estimated_completion_time: new Date().toISOString(),
        report: newBkrResult
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('Credit report refresh error:', error);
    res.status(500).json({
      error: 'Failed to refresh credit report',
      message: error.message
    });
  }
});

// Get credit score history for client
router.get('/bkr-client/:clientId/score-history', async (req, res) => {
  try {
    const { clientId } = req.params;
    const { months = 24 } = req.query;
    const client = await getDbConnection();

    try {
      const monthsAgo = new Date();
      monthsAgo.setMonth(monthsAgo.getMonth() - parseInt(months));

      const result = await client.query(`
        SELECT credit_score, created_at
        FROM bkr_credit_reports r
        JOIN bkr_credit_check_requests req ON r.request_id = req.request_id
        WHERE req.client_id = $1 AND r.created_at >= $2
        ORDER BY r.created_at DESC
      `, [clientId, monthsAgo]);

      const history = result.rows.map(row => ({
        date: row.created_at.toISOString().split('T')[0],
        score: row.credit_score,
        change: 0 // TODO: Calculate change from previous score
      }));

      // Calculate changes
      for (let i = 0; i < history.length - 1; i++) {
        history[i].change = history[i].score - history[i + 1].score;
      }

      res.json({
        success: true,
        history: history
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('Credit score history error:', error);
    res.status(500).json({
      error: 'Failed to get credit score history',
      message: error.message
    });
  }
});

// NHG Eligibility endpoints
router.get('/nhg/client/:clientId/eligibility', async (req, res) => {
  try {
    const { clientId } = req.params;
    const client = await getDbConnection();

    try {
      // Get latest NHG eligibility assessment for client
      const result = await client.query(`
        SELECT * FROM nhg_eligibility_assessments
        WHERE client_id = $1
        ORDER BY created_at DESC
        LIMIT 1
      `, [clientId]);

      if (result.rows.length === 0) {
        return res.status(404).json({
          error: 'No NHG eligibility assessment found for client'
        });
      }

      const assessment = result.rows[0];

      res.json({
        id: assessment.assessment_id,
        client_id: assessment.client_id,
        assessment_date: assessment.created_at,
        eligible: assessment.eligible,
        mortgage_amount: assessment.mortgage_amount,
        nhg_costs: assessment.nhg_costs,
        benefits: JSON.parse(assessment.benefits),
        requirements: JSON.parse(assessment.requirements),
        risk_assessment: assessment.risk_assessment,
        recommendations: JSON.parse(assessment.recommendations)
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('NHG eligibility retrieval error:', error);
    res.status(500).json({
      error: 'Failed to get NHG eligibility assessment',
      message: error.message
    });
  }
});

// Market Insights endpoints
router.get('/market/insights', async (req, res) => {
  try {
    const client = await getDbConnection();

    try {
      // Get latest market indicators
      const indicatorsResult = await client.query(`
        SELECT * FROM market_indicators
        ORDER BY last_updated DESC
        LIMIT 20
      `);

      // Get lender rates
      const ratesResult = await client.query(`
        SELECT * FROM lender_rates
        ORDER BY last_updated DESC
      `);

      // Get regulatory updates
      const updatesResult = await client.query(`
        SELECT * FROM regulatory_updates
        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        ORDER BY date DESC
        LIMIT 10
      `);

      const indicators = indicatorsResult.rows.map(row => ({
        id: row.id,
        name: row.name,
        value: row.value,
        change: row.change,
        trend: row.trend,
        unit: row.unit,
        last_updated: row.last_updated
      }));

      const lender_rates = ratesResult.rows.map(row => ({
        lender: row.lender,
        fixed_10yr: row.fixed_10yr,
        fixed_20yr: row.fixed_20yr,
        variable: row.variable,
        last_updated: row.last_updated
      }));

      const regulatory_updates = updatesResult.rows.map(row => ({
        id: row.id,
        title: row.title,
        summary: row.summary,
        impact: row.impact,
        date: row.date,
        category: row.category
      }));

      // Calculate market summary
      const market_summary = {
        overall_trend: 'neutral', // TODO: Calculate from indicators
        key_drivers: indicators.filter(i => Math.abs(i.change) > 1).map(i => i.name),
        forecast_3m: 'Stable market conditions expected',
        risk_factors: ['Interest rate volatility', 'Economic uncertainty']
      };

      res.json({
        success: true,
        indicators: indicators,
        lender_rates: lender_rates,
        regulatory_updates: regulatory_updates,
        market_summary: market_summary
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('Market insights retrieval error:', error);
    res.status(500).json({
      error: 'Failed to get market insights',
      message: error.message
    });
  }
});

// Refresh market data
router.post('/market/refresh', async (req, res) => {
  try {
    // This would typically call external market data APIs
    // For now, return mock updated data
    res.json({
      success: true,
      message: 'Market data refreshed',
      last_updated: new Date().toISOString()
    });

  } catch (error) {
    console.error('Market data refresh error:', error);
    res.status(500).json({
      error: 'Failed to refresh market data',
      message: error.message
    });
  }
});

// BSN validation endpoint
router.post('/validation/bsn', async (req, res) => {
  try {
    const { bsn } = req.body;

    // Basic BSN validation (Dutch Social Security Number format)
    const bsnRegex = /^\d{9}$/;
    if (!bsnRegex.test(bsn)) {
      return res.json({
        is_valid: false,
        message: 'BSN must be exactly 9 digits'
      });
    }

    // Check if BSN follows the 11-proof (Dutch checksum)
    const digits = bsn.split('').map(Number);
    let sum = 0;
    for (let i = 0; i < 8; i++) {
      sum += digits[i] * (9 - i);
    }
    sum += digits[8] * 1;

    const isValid = sum % 11 === 0;

    res.json({
      is_valid: isValid,
      message: isValid ? 'Valid BSN' : 'Invalid BSN checksum'
    });

  } catch (error) {
    console.error('BSN validation error:', error);
    res.status(500).json({
      error: 'BSN validation failed',
      message: error.message
    });
  }
});

module.exports = router;
