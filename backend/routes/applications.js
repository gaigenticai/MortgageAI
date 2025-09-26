/**
 * Mortgage Applications API Routes
 * Basic CRUD operations for mortgage applications
 *
 * This module provides RESTful endpoints for managing mortgage applications,
 * including creation, retrieval, updates, and submission tracking.
 * Integrates with the existing database schema.
 */

const express = require('express');
const router = express.Router();
const { Client } = require('pg');
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

// Validation schemas
const applicationSchema = Joi.object({
  application_number: Joi.string().required(),
  applicant_data: Joi.object().required(),
  documents: Joi.array().default([]),
  qc_score: Joi.number().min(0).max(100).optional(),
  compliance_score: Joi.number().min(0).max(100).optional(),
  advice_draft: Joi.string().optional(),
  final_advice: Joi.string().optional(),
  status: Joi.string().valid('draft', 'under_review', 'approved', 'rejected', 'submitted').default('draft')
});

// Database helper functions
async function getDbConnection() {
  const client = new Client(dbConfig);
  await client.connect();
  return client;
}

async function createApplication(applicationData) {
  const client = await getDbConnection();

  try {
    const query = `
      INSERT INTO mortgage_applications (
        id, application_number, applicant_data, documents,
        status, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
      RETURNING id
    `;

    const values = [
      uuidv4(),
      applicationData.application_number,
      JSON.stringify(applicationData.applicant_data),
      JSON.stringify(applicationData.documents || []),
      applicationData.status || 'draft'
    ];

    const result = await client.query(query, values);
    return { id: result.rows[0].id, ...applicationData };
  } finally {
    await client.end();
  }
}

async function getApplication(applicationId) {
  const client = await getDbConnection();

  try {
    const query = `
      SELECT * FROM mortgage_applications
      WHERE id = $1
    `;

    const result = await client.query(query, [applicationId]);

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];
    return {
      id: row.id,
      application_number: row.application_number,
      status: row.status,
      applicant_data: typeof row.applicant_data === 'string' ? JSON.parse(row.applicant_data) : row.applicant_data,
      documents: typeof row.documents === 'string' ? JSON.parse(row.documents) : row.documents,
      qc_score: row.qc_score,
      compliance_score: row.compliance_score,
      advice_draft: row.advice_draft,
      final_advice: row.final_advice,
      created_at: row.created_at,
      updated_at: row.updated_at,
      submitted_at: row.submitted_at,
      approved_at: row.approved_at
    };
  } finally {
    await client.end();
  }
}

async function getApplicationsByClient(clientId) {
  const client = await getDbConnection();

  try {
    const query = `
      SELECT ma.* FROM mortgage_applications ma
      JOIN dutch_mortgage_applications dma ON ma.id = dma.id
      WHERE dma.client_id = $1
      ORDER BY ma.created_at DESC
    `;

    const result = await client.query(query, [clientId]);
    return result.rows.map(row => ({
      id: row.id,
      application_number: row.application_number,
      status: row.status,
      applicant_data: typeof row.applicant_data === 'string' ? JSON.parse(row.applicant_data) : row.applicant_data,
      created_at: row.created_at,
      updated_at: row.updated_at
    }));
  } finally {
    await client.end();
  }
}

async function updateApplication(applicationId, updates) {
  const client = await getDbConnection();

  try {
    const setClause = Object.keys(updates).map((key, index) => `${key} = $${index + 2}`).join(', ');
    const values = [applicationId, ...Object.values(updates)];

    const query = `
      UPDATE mortgage_applications
      SET ${setClause}, updated_at = CURRENT_TIMESTAMP
      WHERE id = $1
      RETURNING *
    `;

    const result = await client.query(query, values);

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];
    return {
      id: row.id,
      application_number: row.application_number,
      status: row.status,
      applicant_data: typeof row.applicant_data === 'string' ? JSON.parse(row.applicant_data) : row.applicant_data,
      documents: typeof row.documents === 'string' ? JSON.parse(row.documents) : row.documents,
      qc_score: row.qc_score,
      compliance_score: row.compliance_score,
      updated_at: row.updated_at
    };
  } finally {
    await client.end();
  }
}

// Route handlers

// Create new application
router.post('/applications', async (req, res) => {
  try {
    const validationResult = applicationSchema.validate(req.body, { abortEarly: false });
    if (validationResult.error) {
      return res.status(400).json({
        error: 'Validation failed',
        details: validationResult.error.details.map(detail => ({
          field: detail.path.join('.'),
          message: detail.message
        }))
      });
    }

    const applicationData = req.body;
    console.log(`Creating mortgage application ${applicationData.application_number}`);

    const application = await createApplication(applicationData);

    res.status(201).json({
      success: true,
      application: application,
      metadata: {
        created_at: new Date().toISOString(),
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Application creation error:', error);
    res.status(500).json({
      error: 'Application creation failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get application by ID
router.get('/applications/:applicationId', async (req, res) => {
  try {
    const { applicationId } = req.params;

    console.log(`Retrieving application ${applicationId}`);

    const application = await getApplication(applicationId);
    if (!application) {
      return res.status(404).json({
        error: 'Application not found',
        application_id: applicationId
      });
    }

    res.json({
      success: true,
      application: application,
      metadata: {
        retrieved_at: new Date().toISOString(),
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Application retrieval error:', error);
    res.status(500).json({
      error: 'Application retrieval failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get applications by client ID
router.get('/applications/client/:clientId', async (req, res) => {
  try {
    const { clientId } = req.params;

    console.log(`Retrieving applications for client ${clientId}`);

    const applications = await getApplicationsByClient(clientId);

    res.json({
      success: true,
      applications: applications,
      total_count: applications.length,
      metadata: {
        retrieved_at: new Date().toISOString(),
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Client applications retrieval error:', error);
    res.status(500).json({
      error: 'Client applications retrieval failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Update application
router.put('/applications/:applicationId', async (req, res) => {
  try {
    const { applicationId } = req.params;
    const updates = req.body;

    console.log(`Updating application ${applicationId}`);

    const application = await updateApplication(applicationId, updates);
    if (!application) {
      return res.status(404).json({
        error: 'Application not found',
        application_id: applicationId
      });
    }

    res.json({
      success: true,
      application: application,
      metadata: {
        updated_at: new Date().toISOString(),
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Application update error:', error);
    res.status(500).json({
      error: 'Application update failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Submit application
router.post('/applications/:applicationId/submit', async (req, res) => {
  try {
    const { applicationId } = req.params;

    console.log(`Submitting application ${applicationId}`);

    const updates = {
      status: 'submitted',
      submitted_at: new Date().toISOString()
    };

    const application = await updateApplication(applicationId, updates);
    if (!application) {
      return res.status(404).json({
        error: 'Application not found',
        application_id: applicationId
      });
    }

    res.json({
      success: true,
      application: application,
      message: 'Application submitted successfully',
      metadata: {
        submitted_at: new Date().toISOString(),
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Application submission error:', error);
    res.status(500).json({
      error: 'Application submission failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get application documents
router.get('/applications/:applicationId/documents', async (req, res) => {
  try {
    const { applicationId } = req.params;

    console.log(`Retrieving documents for application ${applicationId}`);

    const application = await getApplication(applicationId);
    if (!application) {
      return res.status(404).json({
        error: 'Application not found',
        application_id: applicationId
      });
    }

    res.json({
      success: true,
      documents: application.documents || [],
      metadata: {
        application_id: applicationId,
        document_count: (application.documents || []).length,
        retrieved_at: new Date().toISOString(),
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Application documents retrieval error:', error);
    res.status(500).json({
      error: 'Application documents retrieval failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Cancel application
router.post('/applications/:applicationId/cancel', async (req, res) => {
  try {
    const { applicationId } = req.params;
    const { reason } = req.body;

    console.log(`Cancelling application ${applicationId}`);

    const updates = {
      status: 'cancelled',
      processing_notes: reason || 'Cancelled by user'
    };

    const application = await updateApplication(applicationId, updates);
    if (!application) {
      return res.status(404).json({
        error: 'Application not found',
        application_id: applicationId
      });
    }

    res.json({
      success: true,
      application: application,
      message: 'Application cancelled successfully',
      metadata: {
        cancelled_at: new Date().toISOString(),
        processing_time: Date.now() - req.startTime
      }
    });

  } catch (error) {
    console.error('Application cancellation error:', error);
    res.status(500).json({
      error: 'Application cancellation failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Get application status updates
router.get('/applications/:applicationId/status-updates', async (req, res) => {
  try {
    const { applicationId } = req.params;
    const { since } = req.query;

    const client = await getDbConnection();

    try {
      let query = `
        SELECT status, created_at, updated_at, lender_status, lender_status_updated,
               lender_comments, submission_status, submitted_at, qc_status, qc_score,
               compliance_score
        FROM mortgage_applications
        WHERE id = $1
      `;
      const params = [applicationId];

      if (since) {
        query += ' AND updated_at > $2';
        params.push(since);
      }

      query += ' ORDER BY updated_at DESC';

      const result = await client.query(query, params);

      if (result.rows.length === 0) {
        return res.status(404).json({
          error: 'Application not found',
          application_id: applicationId
        });
      }

      const updates = result.rows.map(row => ({
        id: `${applicationId}_${row.updated_at.getTime()}`,
        application_id: applicationId,
        status: row.status,
        timestamp: row.updated_at,
        changes: {
          qc_status: row.qc_status,
          qc_score: row.qc_score,
          compliance_score: row.compliance_score,
          lender_status: row.lender_status,
          submission_status: row.submission_status,
          lender_comments: row.lender_comments
        },
        description: `Application status updated to ${row.status}`
      }));

      res.json({
        success: true,
        updates: updates,
        metadata: {
          application_id: applicationId,
          total_updates: updates.length,
          retrieved_at: new Date().toISOString()
        }
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('Application status updates retrieval error:', error);
    res.status(500).json({
      error: 'Failed to get application status updates',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Upload document to application
router.post('/applications/:applicationId/documents', async (req, res) => {
  try {
    const { applicationId } = req.params;
    const { document_type, comments } = req.body;

    // In a real implementation, you'd handle file upload here
    // For now, we'll simulate document upload

    const client = await getDbConnection();

    try {
      // Check if application exists
      const appResult = await client.query(
        'SELECT id FROM mortgage_applications WHERE id = $1',
        [applicationId]
      );

      if (appResult.rows.length === 0) {
        return res.status(404).json({
          error: 'Application not found',
          application_id: applicationId
        });
      }

      // Create document record
      const documentId = uuidv4();
      const filename = `doc_${Date.now()}_${document_type}.pdf`; // Simulated filename

      await client.query(`
        INSERT INTO application_documents (
          id, application_id, document_type, filename, status, uploaded_at, comments
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      `, [
        documentId,
        applicationId,
        document_type,
        filename,
        'pending', // Status will be updated after processing
        new Date(),
        comments || null
      ]);

      res.json({
        success: true,
        document: {
          id: documentId,
          application_id: applicationId,
          type: document_type,
          filename: filename,
          status: 'pending',
          uploaded_at: new Date().toISOString(),
          comments: comments
        },
        metadata: {
          application_id: applicationId,
          uploaded_at: new Date().toISOString(),
          processing_status: 'Document queued for processing'
        }
      });

    } finally {
      await client.end();
    }

  } catch (error) {
    console.error('Document upload error:', error);
    res.status(500).json({
      error: 'Failed to upload document',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

module.exports = router;
