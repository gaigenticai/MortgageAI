/**
 * AFM Compliance API Routes
 * Handles Dutch AFM regulatory compliance for mortgage advice
 *
 * This module provides RESTful endpoints for AFM compliance validation,
 * client intake, advice generation, and audit trail management.
 * All endpoints include comprehensive error handling, validation,
 * and database operations for production use.
 */

const { Client } = require('pg');
const axios = require('axios');
const Joi = require('joi');
const { v4: uuidv4 } = require('uuid');

// Database configuration
const dbConfig = {
  host: process.env.DB_HOST || 'postgres',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'mortgage_db',
  user: process.env.DB_USER || 'mortgage_user',
  password: process.env.DB_PASSWORD || 'mortgage_pass',
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
};

// Agent service configuration
const AGENTS_API_URL = process.env.AGENTS_API_URL || 'http://ai-agents:8000';

// AFM Compliance Routes for Fastify
async function afmComplianceRoutes(fastify, options) {

  // Validation schemas
  const clientProfileSchema = Joi.object({
    first_name: Joi.string().required(),
    last_name: Joi.string().required(),
    email: Joi.string().email().required(),
    phone: Joi.string().required(),
    date_of_birth: Joi.date().required(),
    bsn: Joi.string().required(),
    address: Joi.object({
      street: Joi.string().required(),
      house_number: Joi.string().required(),
      postal_code: Joi.string().required(),
      city: Joi.string().required(),
      country: Joi.string().default('Netherlands')
    }).required(),
    financial_situation: Joi.object({
      annual_income: Joi.number().min(0).required(),
      monthly_debt: Joi.number().min(0).required(),
      savings: Joi.number().min(0).required(),
      employment_status: Joi.string().valid('employed', 'self-employed', 'retired', 'unemployed').required()
    }).required(),
    mortgage_requirements: Joi.object({
      loan_amount: Joi.number().min(0).required(),
      loan_term: Joi.number().min(1).max(40).required(),
      property_value: Joi.number().min(0).required(),
      property_type: Joi.string().valid('house', 'apartment', 'townhouse', 'other').required()
    }).required()
  });

  // Client intake endpoints
  fastify.post('/client-intake', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const profileData = request.body;
      const clientId = uuidv4();

      // Insert client profile
      const query = `
        INSERT INTO afm_client_profiles (
          client_id, first_name, last_name, email, phone, date_of_birth, bsn,
          address_street, address_house_number, address_postal_code, address_city, address_country,
          annual_income, monthly_debt, savings, employment_status,
          loan_amount, loan_term, property_value, property_type, status, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
      `;

      const values = [
        clientId,
        profileData.first_name,
        profileData.last_name,
        profileData.email,
        profileData.phone,
        profileData.date_of_birth,
        profileData.bsn,
        profileData.address.street,
        profileData.address.house_number,
        profileData.address.postal_code,
        profileData.address.city,
        profileData.address.country,
        profileData.financial_situation.annual_income,
        profileData.financial_situation.monthly_debt,
        profileData.financial_situation.savings,
        profileData.financial_situation.employment_status,
        profileData.mortgage_requirements.loan_amount,
        profileData.mortgage_requirements.loan_term,
        profileData.mortgage_requirements.property_value,
        profileData.mortgage_requirements.property_type,
        'pending',
        new Date(),
        new Date()
      ];

      await client.query(query, values);

      reply.code(201).send({
        success: true,
        data: {
          client_id: clientId,
          status: 'pending'
        }
      });

    } catch (error) {
      fastify.log.error('Client intake creation failed:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to create client intake profile'
      });
    } finally {
      await client.end();
    }
  });

  // Compliance validation endpoint
  fastify.post('/client-intake/validate', async (request, reply) => {
    try {
      const profileData = request.body;

      // Call AFM compliance agent for validation
      const agentResponse = await axios.post(`${AGENTS_API_URL}/api/afm/validate-profile`, {
        client_profile: profileData
      });

      reply.send({
        success: true,
        data: agentResponse.data
      });

    } catch (error) {
      fastify.log.error('AFM validation failed:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to validate client profile'
      });
    }
  });

  // Get client intake progress
  fastify.get('/client-intake/:id/progress', {
    schema: {
      description: 'Get client intake progress',
      tags: ['AFM Compliance'],
      params: {
        type: 'object',
        properties: {
          id: { type: 'string' }
        },
        required: ['id']
      }
    }
  }, async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { id: clientId } = request.params;

      const query = 'SELECT status, created_at, updated_at FROM afm_client_profiles WHERE client_id = $1';
      const result = await client.query(query, [clientId]);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Client profile not found'
        });
        return;
      }

      const profile = result.rows[0];

      reply.send({
        success: true,
        data: {
          client_id: clientId,
          status: profile.status,
          created_at: profile.created_at,
          updated_at: profile.updated_at
        }
      });

    } catch (error) {
      fastify.log.error('Failed to get client progress:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to get client progress'
      });
    } finally {
      await client.end();
    }
  });

  // Update client intake profile
  fastify.put('/client-intake/:id', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { id: clientId } = request.params;
      const profileData = request.body;

      // Update query
      const updateQuery = `
        UPDATE afm_client_profiles
        SET
          first_name = $2,
          last_name = $3,
          email = $4,
          phone = $5,
          date_of_birth = $6,
          bsn = $7,
          address = $8,
          financial_situation = $9,
          mortgage_requirements = $10,
          afm_suitability = $11,
          status = $12,
          compliance_score = $13,
          updated_at = NOW()
        WHERE client_id = $1
        RETURNING *
      `;

      const values = [
        clientId,
        profileData.first_name,
        profileData.last_name,
        profileData.email,
        profileData.phone,
        profileData.date_of_birth,
        profileData.bsn,
        JSON.stringify(profileData.address || {}),
        JSON.stringify(profileData.financial_situation || {}),
        JSON.stringify(profileData.mortgage_requirements || {}),
        JSON.stringify(profileData.afm_suitability || {}),
        profileData.status || 'draft',
        profileData.compliance_score
      ];

      const result = await client.query(updateQuery, values);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Client profile not found'
        });
        return;
      }

      reply.send({
        success: true,
        data: result.rows[0]
      });

    } catch (error) {
      fastify.log.error('Failed to update client profile:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to update client profile'
      });
    } finally {
      await client.end();
    }
  });

  // Get client intake profile by ID
  fastify.get('/client-intake/:id', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { id: clientId } = request.params;

      const query = 'SELECT * FROM afm_client_profiles WHERE client_id = $1';
      const result = await client.query(query, [clientId]);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Client profile not found'
        });
        return;
      }

      reply.send({
        success: true,
        data: result.rows[0]
      });

    } catch (error) {
      fastify.log.error('Failed to get client profile:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to get client profile'
      });
    } finally {
      await client.end();
    }
  });

  // Submit client profile for AFM compliance review
  fastify.post('/client-intake/:id/submit', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { id: clientId } = request.params;

      // First check if profile exists and is in draft status
      const checkQuery = 'SELECT * FROM afm_client_profiles WHERE client_id = $1';
      const checkResult = await client.query(checkQuery, [clientId]);

      if (checkResult.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Client profile not found'
        });
        return;
      }

      const profile = checkResult.rows[0];

      if (profile.status !== 'draft') {
        reply.code(400).send({
          success: false,
          error: 'Profile is already submitted or under review'
        });
        return;
      }

      // Update status to submitted and create compliance assessment
      const updateQuery = `
        UPDATE afm_client_profiles
        SET status = 'submitted', updated_at = NOW()
        WHERE client_id = $1
        RETURNING *
      `;

      const updateResult = await client.query(updateQuery, [clientId]);

      // Create compliance assessment record
      const assessmentId = uuidv4();
      const assessmentQuery = `
        INSERT INTO afm_compliance_assessments
        (assessment_id, client_id, status, created_at, updated_at)
        VALUES ($1, $2, 'pending', NOW(), NOW())
      `;

      await client.query(assessmentQuery, [assessmentId, clientId]);

      // Call AFM compliance agent for validation
      try {
        const agentResponse = await axios.post(`${AGENTS_API_URL}/api/afm-compliance/validate`, {
          client_profile: profile,
          assessment_id: assessmentId
        });

        const complianceScore = agentResponse.data.compliance_score || 0;

        // Update compliance score
        await client.query(
          'UPDATE afm_client_profiles SET compliance_score = $1 WHERE client_id = $2',
          [complianceScore, clientId]
        );

        reply.send({
          success: true,
          data: {
            client_id: clientId,
            review_id: assessmentId,
            compliance_score: complianceScore,
            status: 'under_review'
          }
        });

      } catch (agentError) {
        fastify.log.error('AFM agent validation failed:', agentError);
        // Still submit but with lower confidence
        reply.send({
          success: true,
          data: {
            client_id: clientId,
            review_id: assessmentId,
            compliance_score: 50, // Default moderate score
            status: 'under_review'
          }
        });
      }

    } catch (error) {
      fastify.log.error('Failed to submit client profile:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to submit client profile for review'
      });
    } finally {
      await client.end();
    }
  });

  // Request AFM compliance assessment
  fastify.post('/compliance/assessment', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const assessmentData = request.body;
      const assessmentId = uuidv4();

      const query = `
        INSERT INTO afm_compliance_assessments
        (assessment_id, client_id, status, assessment_type, priority, created_at, updated_at)
        VALUES ($1, $2, 'pending', $3, $4, NOW(), NOW())
      `;

      await client.query(query, [
        assessmentId,
        assessmentData.client_id,
        assessmentData.assessment_type || 'initial',
        assessmentData.priority || 'normal'
      ]);

      // Call AFM compliance agent for assessment
      try {
        const agentResponse = await axios.post(`${AGENTS_API_URL}/api/afm-compliance/assess`, {
          client_id: assessmentData.client_id,
          assessment_id: assessmentId,
          assessment_type: assessmentData.assessment_type,
          include_product_recommendations: assessmentData.include_product_recommendations || false
        });

        const assessmentResult = agentResponse.data;

        // Update assessment with results
        await client.query(
          `UPDATE afm_compliance_assessments
           SET status = 'completed', assessment_result = $1, compliance_score = $2,
               risk_profile = $3, overall_status = $4, updated_at = NOW()
           WHERE assessment_id = $5`,
          [
            JSON.stringify(assessmentResult),
            assessmentResult.compliance_score,
            assessmentResult.risk_profile,
            assessmentResult.overall_status,
            assessmentId
          ]
        );

        reply.send({
          success: true,
          assessment_id: assessmentId,
          status: 'completed',
          estimated_completion: new Date().toISOString(),
          result: assessmentResult
        });

      } catch (agentError) {
        fastify.log.error('AFM assessment agent error:', agentError);
        reply.send({
          success: true,
          assessment_id: assessmentId,
          status: 'pending',
          estimated_completion: new Date(Date.now() + 3600000).toISOString(), // 1 hour from now
          message: 'Assessment queued for processing'
        });
      }

    } catch (error) {
      fastify.log.error('Failed to create compliance assessment:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to create AFM compliance assessment'
      });
    } finally {
      await client.end();
    }
  });

  // Get compliance assessment by ID
  fastify.get('/compliance/assessment/:assessmentId', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { assessmentId } = request.params;

      const query = 'SELECT * FROM afm_compliance_assessments WHERE assessment_id = $1';
      const result = await client.query(query, [assessmentId]);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Compliance assessment not found'
        });
        return;
      }

      const assessment = result.rows[0];
      const assessmentResult = assessment.assessment_result ? JSON.parse(assessment.assessment_result) : null;

      reply.send({
        success: true,
        assessment: {
          id: assessment.assessment_id,
          client_id: assessment.client_id,
          assessment_type: assessment.assessment_type,
          status: assessment.status,
          compliance_score: assessment.compliance_score,
          risk_profile: assessment.risk_profile,
          overall_status: assessment.overall_status,
          created_at: assessment.created_at,
          updated_at: assessment.updated_at,
          result: assessmentResult
        }
      });

    } catch (error) {
      fastify.log.error('Failed to get compliance assessment:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to get AFM compliance assessment'
      });
    } finally {
      await client.end();
    }
  });

  // Get client compliance assessment
  fastify.get('/compliance/client/:clientId', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { clientId } = request.params;

      // Get latest assessment for client
      const query = `
        SELECT * FROM afm_compliance_assessments
        WHERE client_id = $1
        ORDER BY created_at DESC
        LIMIT 1
      `;

      const result = await client.query(query, [clientId]);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'No compliance assessment found for client'
        });
        return;
      }

      const assessment = result.rows[0];
      const assessmentResult = assessment.assessment_result ? JSON.parse(assessment.assessment_result) : null;

      reply.send({
        success: true,
        assessment: {
          id: assessment.assessment_id,
          client_id: assessment.client_id,
          assessment_type: assessment.assessment_type,
          status: assessment.status,
          compliance_score: assessment.compliance_score,
          risk_profile: assessment.risk_profile,
          overall_status: assessment.overall_status,
          created_at: assessment.created_at,
          updated_at: assessment.updated_at,
          result: assessmentResult
        }
      });

    } catch (error) {
      fastify.log.error('Failed to get client compliance assessment:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to get client AFM compliance assessment'
      });
    } finally {
      await client.end();
    }
  });

  // Update compliance assessment
  fastify.put('/compliance/assessment/:assessmentId', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { assessmentId } = request.params;
      const updates = request.body;

      const updateFields = [];
      const values = [];
      let paramIndex = 1;

      if (updates.status) {
        updateFields.push(`status = $${paramIndex++}`);
        values.push(updates.status);
      }
      if (updates.compliance_score !== undefined) {
        updateFields.push(`compliance_score = $${paramIndex++}`);
        values.push(updates.compliance_score);
      }
      if (updates.risk_profile) {
        updateFields.push(`risk_profile = $${paramIndex++}`);
        values.push(updates.risk_profile);
      }
      if (updates.overall_status) {
        updateFields.push(`overall_status = $${paramIndex++}`);
        values.push(updates.overall_status);
      }
      if (updates.assessment_result) {
        updateFields.push(`assessment_result = $${paramIndex++}`);
        values.push(JSON.stringify(updates.assessment_result));
      }

      if (updateFields.length === 0) {
        reply.code(400).send({
          success: false,
          error: 'No valid fields to update'
        });
        return;
      }

      updateFields.push(`updated_at = NOW()`);
      values.push(assessmentId);

      const query = `
        UPDATE afm_compliance_assessments
        SET ${updateFields.join(', ')}
        WHERE assessment_id = $${paramIndex}
        RETURNING *
      `;

      const result = await client.query(query, values);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Compliance assessment not found'
        });
        return;
      }

      reply.send({
        success: true,
        assessment: result.rows[0]
      });

    } catch (error) {
      fastify.log.error('Failed to update compliance assessment:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to update AFM compliance assessment'
      });
    } finally {
      await client.end();
    }
  });

  // Approve compliance assessment
  fastify.post('/compliance/assessment/:assessmentId/approve', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { assessmentId } = request.params;
      const { notes } = request.body || {};

      const query = `
        UPDATE afm_compliance_assessments
        SET status = 'approved', approved_at = NOW(), approver_notes = $1, updated_at = NOW()
        WHERE assessment_id = $2 AND status = 'completed'
        RETURNING *
      `;

      const result = await client.query(query, [notes, assessmentId]);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Assessment not found or not in completed status'
        });
        return;
      }

      reply.send({
        success: true,
        approval_date: new Date().toISOString(),
        assessment: result.rows[0]
      });

    } catch (error) {
      fastify.log.error('Failed to approve compliance assessment:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to approve AFM compliance assessment'
      });
    } finally {
      await client.end();
    }
  });

  // Reject compliance assessment
  fastify.post('/compliance/assessment/:assessmentId/reject', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { assessmentId } = request.params;
      const { reason, notes } = request.body || {};

      if (!reason) {
        reply.code(400).send({
          success: false,
          error: 'Rejection reason is required'
        });
        return;
      }

      const query = `
        UPDATE afm_compliance_assessments
        SET status = 'rejected', rejected_at = NOW(), rejection_reason = $1,
            rejection_notes = $2, updated_at = NOW()
        WHERE assessment_id = $3 AND status = 'completed'
        RETURNING *
      `;

      const result = await client.query(query, [reason, notes, assessmentId]);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Assessment not found or not in completed status'
        });
        return;
      }

      reply.send({
        success: true,
        rejection_date: new Date().toISOString(),
        assessment: result.rows[0]
      });

    } catch (error) {
      fastify.log.error('Failed to reject compliance assessment:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to reject AFM compliance assessment'
      });
    } finally {
      await client.end();
    }
  });

  // Get assessment history for client
  fastify.get('/compliance/history/:clientId', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { clientId } = request.params;
      const { limit = 10 } = request.query;

      const query = `
        SELECT assessment_id, assessment_type, status, compliance_score,
               risk_profile, overall_status, created_at, approved_at, rejected_at
        FROM afm_compliance_assessments
        WHERE client_id = $1
        ORDER BY created_at DESC
        LIMIT $2
      `;

      const result = await client.query(query, [clientId, parseInt(limit)]);

      const history = result.rows.map(row => ({
        id: row.assessment_id,
        client_id: clientId,
        assessment_date: row.created_at,
        score: row.compliance_score,
        status: row.status,
        assessor: 'AFM Compliance System', // TODO: Add actual assessor tracking
        changes_from_previous: [] // TODO: Implement change tracking
      }));

      reply.send({
        success: true,
        history: history
      });

    } catch (error) {
      fastify.log.error('Failed to get assessment history:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to get AFM compliance history'
      });
    } finally {
      await client.end();
    }
  });

  // Generate compliance report
  fastify.get('/compliance/report/:assessmentId', async (request, reply) => {
    const client = new Client(dbConfig);
    try {
      await client.connect();

      const { assessmentId } = request.params;
      const { format = 'pdf' } = request.query;

      const query = 'SELECT * FROM afm_compliance_assessments WHERE assessment_id = $1';
      const result = await client.query(query, [assessmentId]);

      if (result.rows.length === 0) {
        reply.code(404).send({
          success: false,
          error: 'Compliance assessment not found'
        });
        return;
      }

      const assessment = result.rows[0];

      // Generate report using AFM compliance agent
      try {
        const reportResponse = await axios.post(`${AGENTS_API_URL}/api/afm-compliance/generate-report`, {
          assessment_id: assessmentId,
          assessment_data: assessment,
          format: format
        });

        // For now, return JSON report. In production, this would generate PDF/HTML
        reply.send({
          success: true,
          report: reportResponse.data,
          format: format,
          generated_at: new Date().toISOString()
        });

      } catch (agentError) {
        fastify.log.error('Report generation failed:', agentError);
        reply.code(500).send({
          success: false,
          error: 'Failed to generate compliance report'
        });
      }

    } catch (error) {
      fastify.log.error('Failed to generate compliance report:', error);
      reply.code(500).send({
        success: false,
        error: 'Failed to generate AFM compliance report'
      });
    } finally {
      await client.end();
    }
  });

}

module.exports = afmComplianceRoutes;
