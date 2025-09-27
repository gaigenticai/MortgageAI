/**
 * Advanced Field Validation API Routes
 * Created: 2024-01-15
 * Author: MortgageAI Development Team
 * 
 * Fastify routes for the Advanced Field Validation Engine with real-time
 * validation, error correction suggestions, and AFM compliance checking.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const crypto = require('crypto');

/**
 * Advanced Field Validation API Routes
 * @param {import('fastify').FastifyInstance} fastify
 * @param {Object} options
 */
async function advancedFieldValidationRoutes(fastify, options) {
  const VALIDATION_TIMEOUT = parseInt(process.env.FIELD_VALIDATION_TIMEOUT) || 30000;
  const MAX_VALIDATION_SIZE = parseInt(process.env.MAX_VALIDATION_DATA_SIZE) || 10485760; // 10MB
  const PYTHON_SCRIPT_PATH = path.join(__dirname, '..', 'agents', 'utils', 'advanced_field_validation_executor.py');

  // Utility function to execute Python validation script
  const executeValidation = (operation, args) => {
    return new Promise((resolve, reject) => {
      const pythonArgs = [PYTHON_SCRIPT_PATH, operation];
      
      // Add operation-specific arguments
      Object.entries(args).forEach(([key, value]) => {
        pythonArgs.push(`--${key}`);
        if (typeof value !== 'boolean') {
          pythonArgs.push(String(value));
        }
      });

      fastify.log.info(`Executing validation: ${operation} with args:`, Object.keys(args));

      const python = spawn('python3', pythonArgs, {
        env: { ...process.env },
        timeout: VALIDATION_TIMEOUT
      });

      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (parseError) {
            fastify.log.error('Failed to parse validation result:', parseError);
            reject(new Error('Invalid validation response format'));
          }
        } else {
          fastify.log.error(`Validation process failed with code ${code}:`, stderr);
          reject(new Error(`Validation failed: ${stderr || 'Unknown error'}`));
        }
      });

      python.on('error', (error) => {
        fastify.log.error('Failed to start validation process:', error);
        reject(new Error(`Failed to start validation: ${error.message}`));
      });
    });
  };

  // Schema definitions for request validation
  const validateFieldSchema = {
    body: {
      type: 'object',
      required: ['field_path', 'value'],
      properties: {
        field_path: { type: 'string', minLength: 1, maxLength: 200 },
        value: { type: ['string', 'number', 'boolean'] },
        field_type: { type: 'string', enum: ['text', 'email', 'phone', 'currency', 'percentage', 'date', 'integer', 'decimal', 'boolean', 'enum', 'bsn', 'iban', 'postcode', 'address', 'name', 'custom'] },
        context: { type: 'object' },
        check_afm_compliance: { type: 'boolean', default: true },
        generate_suggestions: { type: 'boolean', default: true }
      },
      additionalProperties: false
    }
  };

  const validateDataSchema = {
    body: {
      type: 'object',
      required: ['data'],
      properties: {
        data: { type: 'object' },
        validation_config: { type: 'object' },
        check_afm_compliance: { type: 'boolean', default: true },
        generate_suggestions: { type: 'boolean', default: true },
        include_scores: { type: 'boolean', default: true }
      },
      additionalProperties: false
    }
  };

  const ruleSchema = {
    type: 'object',
    required: ['rule_name', 'field_path', 'field_type', 'rule_type'],
    properties: {
      rule_name: { type: 'string', minLength: 1, maxLength: 255 },
      field_path: { type: 'string', minLength: 1, maxLength: 200 },
      field_type: { type: 'string', enum: ['text', 'email', 'phone', 'currency', 'percentage', 'date', 'integer', 'decimal', 'boolean', 'enum', 'bsn', 'iban', 'postcode', 'address', 'name', 'custom'] },
      rule_type: { type: 'string', enum: ['required', 'format', 'range', 'length', 'pattern', 'custom', 'afm_compliance'] },
      parameters: { type: 'object' },
      error_message: { type: 'string', maxLength: 500 },
      suggestion_template: { type: 'string', maxLength: 500 },
      severity: { type: 'string', enum: ['info', 'warning', 'error', 'critical'], default: 'error' },
      is_active: { type: 'boolean', default: true },
      afm_article: { type: 'string', maxLength: 50 },
      priority: { type: 'integer', minimum: 1, maximum: 5, default: 1 },
      dependencies: { type: 'array', items: { type: 'string' } },
      conditions: { type: 'object' }
    },
    additionalProperties: false
  };

  // Request size validation middleware
  fastify.addContentTypeParser('application/json', { parseAs: 'buffer', limit: MAX_VALIDATION_SIZE }, async (req, body) => {
    try {
      return JSON.parse(body.toString());
    } catch (err) {
      throw new Error('Invalid JSON in request body');
    }
  });

  // Rate limiting for validation endpoints
  await fastify.register(require('@fastify/rate-limit'), {
    max: parseInt(process.env.VALIDATION_RATE_LIMIT_MAX) || 100,
    timeWindow: parseInt(process.env.VALIDATION_RATE_LIMIT_WINDOW) || 60000 // 1 minute
  });

  /**
   * Validate Single Field
   * POST /validate/field
   * 
   * Validates a single field value against all applicable rules
   */
  fastify.post('/validate/field', { schema: validateFieldSchema }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const { field_path, value, field_type, context, check_afm_compliance, generate_suggestions } = request.body;
      
      fastify.log.info(`Validating field: ${field_path} with value type: ${typeof value}`);
      
      const validationArgs = {
        field_path,
        value: JSON.stringify(value),
        field_type: field_type || 'text',
        context: JSON.stringify(context || {}),
        check_afm_compliance: check_afm_compliance !== false,
        generate_suggestions: generate_suggestions !== false
      };
      
      const result = await executeValidation('validate_field', validationArgs);
      
      if (result.success) {
        const processingTime = Date.now() - startTime;
        
        return {
          success: true,
          field_path,
          is_valid: result.messages.length === 0,
          messages: result.messages || [],
          field_score: result.field_score || 100,
          suggestions: result.suggestions || [],
          processing_time: processingTime,
          timestamp: new Date().toISOString(),
          metadata: {
            field_type: field_type || 'text',
            afm_compliance_checked: check_afm_compliance !== false,
            suggestions_generated: generate_suggestions !== false
          }
        };
      } else {
        throw new Error(result.error || 'Field validation failed');
      }
    } catch (error) {
      fastify.log.error('Field validation error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Field validation failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Validate Data Set
   * POST /validate/data
   * 
   * Validates a complete data set against all applicable rules
   */
  fastify.post('/validate/data', { schema: validateDataSchema }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const { data, validation_config, check_afm_compliance, generate_suggestions, include_scores } = request.body;
      
      // Validate data size
      const dataString = JSON.stringify(data);
      if (dataString.length > MAX_VALIDATION_SIZE) {
        return reply.code(413).send({
          success: false,
          error: 'Request too large',
          message: `Data size exceeds maximum allowed size (${MAX_VALIDATION_SIZE} bytes)`,
          timestamp: new Date().toISOString()
        });
      }
      
      fastify.log.info(`Validating data set with ${Object.keys(data).length} root fields`);
      
      const validationArgs = {
        data: dataString,
        validation_config: JSON.stringify(validation_config || {}),
        check_afm_compliance: check_afm_compliance !== false,
        generate_suggestions: generate_suggestions !== false,
        include_scores: include_scores !== false
      };
      
      const result = await executeValidation('validate_data', validationArgs);
      
      if (result.success) {
        const processingTime = Date.now() - startTime;
        
        return {
          success: true,
          is_valid: result.is_valid || false,
          total_fields: result.total_fields || 0,
          validated_fields: result.validated_fields || 0,
          errors: result.errors || 0,
          warnings: result.warnings || 0,
          infos: result.infos || 0,
          messages: result.messages || [],
          field_scores: result.field_scores || {},
          overall_score: result.overall_score || 0,
          compliance_score: result.compliance_score || 0,
          processing_time: processingTime,
          timestamp: new Date().toISOString(),
          metadata: {
            ...result.metadata,
            request_size: dataString.length,
            afm_compliance_checked: check_afm_compliance !== false,
            suggestions_generated: generate_suggestions !== false
          }
        };
      } else {
        throw new Error(result.error || 'Data validation failed');
      }
    } catch (error) {
      fastify.log.error('Data validation error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Data validation failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Get Validation Rules
   * GET /rules
   * 
   * Retrieves validation rules with optional filtering
   */
  fastify.get('/rules', async (request, reply) => {
    try {
      const { field_path, field_type, rule_type, active_only, limit, offset } = request.query;
      
      const queryArgs = {
        field_path: field_path || '',
        field_type: field_type || '',
        rule_type: rule_type || '',
        active_only: active_only === 'true',
        limit: parseInt(limit) || 100,
        offset: parseInt(offset) || 0
      };
      
      const result = await executeValidation('list_rules', queryArgs);
      
      if (result.success) {
        return {
          success: true,
          rules: result.rules || [],
          total_rules: result.total_rules || 0,
          active_rules: result.active_rules || 0,
          filters: {
            field_path: field_path || null,
            field_type: field_type || null,
            rule_type: rule_type || null,
            active_only: active_only === 'true'
          },
          pagination: {
            limit: parseInt(limit) || 100,
            offset: parseInt(offset) || 0,
            has_more: (result.rules || []).length === parseInt(limit || 100)
          },
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error(result.error || 'Failed to retrieve rules');
      }
    } catch (error) {
      fastify.log.error('Rules retrieval error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to retrieve validation rules',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Create Validation Rule
   * POST /rules
   * 
   * Creates a new validation rule
   */
  fastify.post('/rules', { 
    schema: { body: ruleSchema }
  }, async (request, reply) => {
    try {
      const ruleData = request.body;
      
      // Generate unique rule ID
      const ruleId = `rule_${crypto.randomBytes(8).toString('hex')}_${Date.now()}`;
      
      fastify.log.info(`Creating validation rule: ${ruleData.rule_name}`);
      
      const createArgs = {
        rule_id: ruleId,
        rule_data: JSON.stringify(ruleData)
      };
      
      const result = await executeValidation('create_rule', createArgs);
      
      if (result.success) {
        return {
          success: true,
          rule_id: ruleId,
          created: true,
          rule_configuration: result.rule_configuration || ruleData,
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error(result.error || 'Failed to create rule');
      }
    } catch (error) {
      fastify.log.error('Rule creation error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to create validation rule',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Get Validation Rule
   * GET /rules/:ruleId
   * 
   * Retrieves a specific validation rule
   */
  fastify.get('/rules/:ruleId', async (request, reply) => {
    try {
      const { ruleId } = request.params;
      
      const result = await executeValidation('get_rule', { rule_id: ruleId });
      
      if (result.success) {
        if (result.rule) {
          return {
            success: true,
            rule: result.rule,
            timestamp: new Date().toISOString()
          };
        } else {
          return reply.code(404).send({
            success: false,
            error: 'Rule not found',
            rule_id: ruleId,
            timestamp: new Date().toISOString()
          });
        }
      } else {
        throw new Error(result.error || 'Failed to retrieve rule');
      }
    } catch (error) {
      fastify.log.error('Rule retrieval error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to retrieve validation rule',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Update Validation Rule
   * PUT /rules/:ruleId
   * 
   * Updates an existing validation rule
   */
  fastify.put('/rules/:ruleId', {
    schema: { body: ruleSchema }
  }, async (request, reply) => {
    try {
      const { ruleId } = request.params;
      const ruleData = request.body;
      
      fastify.log.info(`Updating validation rule: ${ruleId}`);
      
      const updateArgs = {
        rule_id: ruleId,
        rule_data: JSON.stringify(ruleData)
      };
      
      const result = await executeValidation('update_rule', updateArgs);
      
      if (result.success) {
        if (result.updated) {
          return {
            success: true,
            rule_id: ruleId,
            updated: true,
            rule_configuration: result.rule_configuration || ruleData,
            timestamp: new Date().toISOString()
          };
        } else {
          return reply.code(404).send({
            success: false,
            error: 'Rule not found',
            rule_id: ruleId,
            timestamp: new Date().toISOString()
          });
        }
      } else {
        throw new Error(result.error || 'Failed to update rule');
      }
    } catch (error) {
      fastify.log.error('Rule update error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to update validation rule',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Delete Validation Rule
   * DELETE /rules/:ruleId
   * 
   * Deletes a validation rule
   */
  fastify.delete('/rules/:ruleId', async (request, reply) => {
    try {
      const { ruleId } = request.params;
      
      fastify.log.info(`Deleting validation rule: ${ruleId}`);
      
      const result = await executeValidation('delete_rule', { rule_id: ruleId });
      
      if (result.success) {
        if (result.deleted) {
          return {
            success: true,
            rule_id: ruleId,
            deleted: true,
            timestamp: new Date().toISOString()
          };
        } else {
          return reply.code(404).send({
            success: false,
            error: 'Rule not found',
            rule_id: ruleId,
            timestamp: new Date().toISOString()
          });
        }
      } else {
        throw new Error(result.error || 'Failed to delete rule');
      }
    } catch (error) {
      fastify.log.error('Rule deletion error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to delete validation rule',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Get Field Suggestions
   * POST /suggestions
   * 
   * Gets correction suggestions for invalid field values
   */
  fastify.post('/suggestions', async (request, reply) => {
    try {
      const { field_path, value, field_type, validation_error } = request.body;
      
      if (!field_path || !value) {
        return reply.code(400).send({
          success: false,
          error: 'Missing required fields',
          message: 'field_path and value are required',
          timestamp: new Date().toISOString()
        });
      }
      
      fastify.log.info(`Generating suggestions for field: ${field_path}`);
      
      const suggestionsArgs = {
        field_path,
        value: JSON.stringify(value),
        field_type: field_type || 'text',
        validation_error: validation_error || ''
      };
      
      const result = await executeValidation('get_suggestions', suggestionsArgs);
      
      if (result.success) {
        return {
          success: true,
          field_path,
          original_value: value,
          suggestions: result.suggestions || [],
          corrected_value: result.corrected_value || null,
          confidence_score: result.confidence_score || 0,
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error(result.error || 'Failed to generate suggestions');
      }
    } catch (error) {
      fastify.log.error('Suggestions generation error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to generate suggestions',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Check AFM Compliance
   * POST /compliance/check
   * 
   * Performs AFM compliance validation on provided data
   */
  fastify.post('/compliance/check', async (request, reply) => {
    try {
      const { field_path, value, context } = request.body;
      
      if (!field_path) {
        return reply.code(400).send({
          success: false,
          error: 'Missing required field',
          message: 'field_path is required',
          timestamp: new Date().toISOString()
        });
      }
      
      fastify.log.info(`Checking AFM compliance for field: ${field_path}`);
      
      const complianceArgs = {
        field_path,
        value: JSON.stringify(value || ''),
        context: JSON.stringify(context || {})
      };
      
      const result = await executeValidation('check_afm_compliance', complianceArgs);
      
      if (result.success) {
        return {
          success: true,
          field_path,
          is_compliant: result.is_compliant || false,
          compliance_score: result.compliance_score || 0,
          compliance_messages: result.compliance_messages || [],
          afm_requirements: result.afm_requirements || [],
          recommendations: result.recommendations || [],
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error(result.error || 'AFM compliance check failed');
      }
    } catch (error) {
      fastify.log.error('AFM compliance check error:', error);
      return reply.code(500).send({
        success: false,
        error: 'AFM compliance check failed',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Get Validation Statistics
   * GET /statistics
   * 
   * Retrieves validation engine performance statistics
   */
  fastify.get('/statistics', async (request, reply) => {
    try {
      const result = await executeValidation('get_statistics', {});
      
      if (result.success) {
        return {
          success: true,
          statistics: result.statistics || {},
          engine_info: result.engine_info || {},
          performance_metrics: result.performance_metrics || {},
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error(result.error || 'Failed to retrieve statistics');
      }
    } catch (error) {
      fastify.log.error('Statistics retrieval error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to retrieve validation statistics',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Export Validation Rules
   * GET /rules/export
   * 
   * Exports validation rules in specified format
   */
  fastify.get('/rules/export', async (request, reply) => {
    try {
      const { format } = request.query;
      const exportFormat = format || 'json';
      
      if (!['json', 'csv', 'yaml'].includes(exportFormat)) {
        return reply.code(400).send({
          success: false,
          error: 'Invalid export format',
          message: 'Supported formats: json, csv, yaml',
          timestamp: new Date().toISOString()
        });
      }
      
      fastify.log.info(`Exporting validation rules in ${exportFormat} format`);
      
      const result = await executeValidation('export_rules', { format: exportFormat });
      
      if (result.success) {
        const filename = `validation_rules_${new Date().toISOString().split('T')[0]}.${exportFormat}`;
        
        reply.header('Content-Disposition', `attachment; filename="${filename}"`);
        
        switch (exportFormat) {
          case 'json':
            reply.type('application/json');
            break;
          case 'csv':
            reply.type('text/csv');
            break;
          case 'yaml':
            reply.type('text/yaml');
            break;
        }
        
        return result.exported_data;
      } else {
        throw new Error(result.error || 'Export failed');
      }
    } catch (error) {
      fastify.log.error('Rules export error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to export validation rules',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Import Validation Rules
   * POST /rules/import
   * 
   * Imports validation rules from uploaded data
   */
  fastify.post('/rules/import', async (request, reply) => {
    try {
      const { rules_data, format, merge } = request.body;
      
      if (!rules_data) {
        return reply.code(400).send({
          success: false,
          error: 'Missing rules data',
          message: 'rules_data field is required',
          timestamp: new Date().toISOString()
        });
      }
      
      const importFormat = format || 'json';
      const mergeMod = merge !== false;
      
      fastify.log.info(`Importing validation rules from ${importFormat} format (merge: ${mergeMod})`);
      
      const importArgs = {
        rules_data: JSON.stringify(rules_data),
        format: importFormat,
        merge: mergeMod
      };
      
      const result = await executeValidation('import_rules', importArgs);
      
      if (result.success) {
        return {
          success: true,
          imported_count: result.imported_count || 0,
          failed_count: result.failed_count || 0,
          merge_mode: mergeMod,
          format: importFormat,
          timestamp: new Date().toISOString()
        };
      } else {
        throw new Error(result.error || 'Import failed');
      }
    } catch (error) {
      fastify.log.error('Rules import error:', error);
      return reply.code(500).send({
        success: false,
        error: 'Failed to import validation rules',
        message: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Health Check
   * GET /health
   * 
   * Performs health check of the validation engine
   */
  fastify.get('/health', async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const result = await executeValidation('health_check', {});
      const responseTime = Date.now() - startTime;
      
      if (result.success) {
        return {
          success: true,
          status: result.status || 'healthy',
          services: result.services || {},
          performance: {
            ...result.performance || {},
            api_response_time: responseTime
          },
          dependencies: result.dependencies || {},
          version: result.version || '1.0.0',
          timestamp: new Date().toISOString()
        };
      } else {
        return reply.code(503).send({
          success: false,
          status: 'unhealthy',
          error: result.error || 'Health check failed',
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      fastify.log.error('Health check error:', error);
      return reply.code(503).send({
        success: false,
        status: 'unhealthy',
        error: 'Health check failed',
        message: error.message,
        response_time: Date.now() - startTime,
        timestamp: new Date().toISOString()
      });
    }
  });

  // Error handler for this plugin
  fastify.setErrorHandler(async (error, request, reply) => {
    fastify.log.error(`Advanced Field Validation API error on ${request.method} ${request.url}:`, error);
    
    // Handle specific error types
    if (error.statusCode === 413) {
      return reply.code(413).send({
        success: false,
        error: 'Request too large',
        message: 'Request size exceeds maximum allowed limit',
        timestamp: new Date().toISOString()
      });
    }
    
    if (error.statusCode === 429) {
      return reply.code(429).send({
        success: false,
        error: 'Rate limit exceeded',
        message: 'Too many requests, please try again later',
        timestamp: new Date().toISOString()
      });
    }
    
    // Default error response
    const statusCode = error.statusCode || 500;
    return reply.code(statusCode).send({
      success: false,
      error: error.name || 'Internal Server Error',
      message: error.message || 'An unexpected error occurred',
      timestamp: new Date().toISOString()
    });
  });

  // Register shutdown handler
  fastify.addHook('onClose', async () => {
    fastify.log.info('Advanced Field Validation API routes shutting down');
  });

  fastify.log.info('Advanced Field Validation API routes registered successfully');
}

module.exports = advancedFieldValidationRoutes;
