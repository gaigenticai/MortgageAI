/**
 * Anomaly Detection Interface API Routes
 * 
 * Provides comprehensive API endpoints for real-time pattern recognition,
 * alert management, and investigation tools for the Dutch mortgage platform.
 * 
 * Created: 2024-01-15
 * Author: MortgageAI Development Team
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const crypto = require('crypto');

/**
 * Anomaly Detection Interface Routes Plugin
 * @param {Object} fastify - Fastify instance
 * @param {Object} options - Plugin options
 */
async function anomalyDetectionInterfaceRoutes(fastify, options) {
  
  // Helper function to execute Python anomaly detection scripts
  const executePythonScript = (scriptName, args = []) => {
    return new Promise((resolve, reject) => {
      const pythonPath = process.env.PYTHON_PATH || 'python3';
      const scriptPath = path.join(__dirname, '..', 'agents', 'utils', scriptName);
      
      const pythonProcess = spawn(pythonPath, [scriptPath, ...args]);
      
      let stdout = '';
      let stderr = '';
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (parseError) {
            resolve({ success: true, data: stdout });
          }
        } else {
          reject(new Error(`Python script failed: ${stderr}`));
        }
      });
      
      // Set timeout for long-running operations
      setTimeout(() => {
        pythonProcess.kill();
        reject(new Error('Anomaly detection operation timed out'));
      }, parseInt(process.env.ANOMALY_DETECTION_TIMEOUT || '180000')); // 3 minutes default
    });
  };
  
  // Helper function to validate detection parameters
  const validateDetectionParams = (params) => {
    const allowedMethods = ['statistical', 'ml_based', 'rule_based', 'hybrid'];
    const allowedSeverities = ['low', 'medium', 'high', 'critical'];
    const allowedCategories = ['application', 'behavior', 'transaction', 'compliance', 'data_quality'];
    
    if (params.methods && !params.methods.every(m => allowedMethods.includes(m))) {
      throw new Error(`Invalid detection methods. Must be one of: ${allowedMethods.join(', ')}`);
    }
    
    if (params.severity_threshold && !allowedSeverities.includes(params.severity_threshold)) {
      throw new Error(`Invalid severity threshold. Must be one of: ${allowedSeverities.join(', ')}`);
    }
    
    if (params.category && !allowedCategories.includes(params.category)) {
      throw new Error(`Invalid category. Must be one of: ${allowedCategories.join(', ')}`);
    }
    
    return true;
  };

  // Schema definitions for request/response validation
  const detectionRequestSchema = {
    type: 'object',
    properties: {
      data: { 
        type: 'object',
        description: 'Data to analyze for anomalies'
      },
      methods: {
        type: 'array',
        items: { 
          type: 'string',
          enum: ['statistical', 'ml_based', 'rule_based', 'hybrid']
        },
        default: ['statistical', 'ml_based']
      },
      severity_threshold: {
        type: 'string',
        enum: ['low', 'medium', 'high', 'critical'],
        default: 'medium'
      },
      detection_options: {
        type: 'object',
        properties: {
          statistical_method: { 
            type: 'string',
            enum: ['z_score', 'modified_z_score', 'iqr', 'grubbs_test', 'dixon_test'],
            default: 'z_score'
          },
          ml_method: {
            type: 'string',
            enum: ['isolation_forest', 'local_outlier_factor', 'elliptic_envelope', 'dbscan'],
            default: 'isolation_forest'
          },
          contamination: { type: 'number', minimum: 0.01, maximum: 0.5, default: 0.1 },
          confidence_threshold: { type: 'number', minimum: 0.1, maximum: 1.0, default: 0.8 }
        },
        default: {}
      }
    },
    required: ['data'],
    additionalProperties: false
  };
  
  const alertRuleSchema = {
    type: 'object',
    required: ['rule_name', 'rule_type', 'category', 'conditions'],
    properties: {
      rule_name: { type: 'string', minLength: 1 },
      rule_type: { 
        type: 'string',
        enum: ['threshold', 'statistical', 'pattern', 'ml_based']
      },
      category: {
        type: 'string',
        enum: ['application', 'behavior', 'compliance', 'performance']
      },
      conditions: { type: 'object' },
      thresholds: { type: 'object', default: {} },
      parameters: { type: 'object', default: {} },
      severity_mapping: { type: 'object', default: {} },
      escalation_rules: { type: 'array', default: [] },
      notification_channels: { 
        type: 'array',
        items: { type: 'string' },
        default: ['in_app']
      },
      suppression_rules: { type: 'object', default: {} },
      is_active: { type: 'boolean', default: true }
    },
    additionalProperties: false
  };

  // Route: Detect anomalies in data
  fastify.post('/detect', {
    schema: {
      description: 'Detect anomalies in provided data using multiple methods',
      tags: ['Anomaly Detection'],
      body: detectionRequestSchema,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            detection_id: { type: 'string' },
            anomalies_detected: { type: 'integer' },
            anomalies: { type: 'array' },
            detection_summary: { type: 'object' },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const { 
        data, 
        methods = ['statistical', 'ml_based'], 
        severity_threshold = 'medium',
        detection_options = {}
      } = request.body;
      
      // Validate parameters
      validateDetectionParams({ methods, severity_threshold });
      
      const detectionId = crypto.randomUUID();
      
      fastify.log.info(`Starting anomaly detection: ${detectionId}`);
      
      // Execute anomaly detection
      const detectionArgs = [
        'detect_anomalies',
        '--detection_id', detectionId,
        '--data', JSON.stringify(data),
        '--methods', methods.join(','),
        '--severity_threshold', severity_threshold,
        '--detection_options', JSON.stringify(detection_options),
        '--output_format', 'json'
      ];
      
      const result = await executePythonScript('anomaly_detection_executor.py', detectionArgs);
      
      const processingTime = (Date.now() - startTime) / 1000;
      
      reply.code(200).send({
        success: true,
        detection_id: detectionId,
        anomalies_detected: result.anomalies?.length || 0,
        anomalies: result.anomalies || [],
        detection_summary: result.detection_summary || {},
        processing_time: processingTime,
        methods_used: methods,
        severity_threshold
      });
      
    } catch (error) {
      fastify.log.error(`Error detecting anomalies: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to detect anomalies',
        message: error.message
      });
    }
  });
  
  // Route: Get anomaly detection summary
  fastify.get('/summary', {
    schema: {
      description: 'Get summary of detected anomalies',
      tags: ['Anomaly Detection', 'Summary'],
      querystring: {
        type: 'object',
        properties: {
          time_period: { 
            type: 'string',
            pattern: '^\\d+[hdwm]$',
            default: '24h'
          },
          category: {
            type: 'string',
            enum: ['application', 'behavior', 'transaction', 'compliance', 'data_quality']
          },
          severity: {
            type: 'string',
            enum: ['low', 'medium', 'high', 'critical']
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { 
        time_period = '24h', 
        category,
        severity
      } = request.query;
      
      fastify.log.info(`Retrieving anomaly summary for period: ${time_period}`);
      
      const summaryArgs = [
        'get_anomaly_summary',
        '--time_period', time_period,
        '--output_format', 'json'
      ];
      
      if (category) {
        summaryArgs.push('--category', category);
      }
      
      if (severity) {
        summaryArgs.push('--severity', severity);
      }
      
      const result = await executePythonScript('anomaly_detection_executor.py', summaryArgs);
      
      reply.code(200).send({
        success: true,
        time_period,
        summary: result.summary || {},
        filters: { category, severity }
      });
      
    } catch (error) {
      fastify.log.error(`Error retrieving anomaly summary: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve anomaly summary',
        message: error.message
      });
    }
  });
  
  // Route: Get specific anomaly details
  fastify.get('/anomaly/:anomalyId', {
    schema: {
      description: 'Get detailed information about a specific anomaly',
      tags: ['Anomaly Detection', 'Details'],
      params: {
        type: 'object',
        required: ['anomalyId'],
        properties: {
          anomalyId: { type: 'string' }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { anomalyId } = request.params;
      
      fastify.log.info(`Retrieving anomaly details: ${anomalyId}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'get_anomaly_details',
        '--anomaly_id', anomalyId,
        '--output_format', 'json'
      ]);
      
      if (!result.anomaly) {
        reply.code(404).send({
          success: false,
          error: 'Anomaly not found',
          anomaly_id: anomalyId
        });
        return;
      }
      
      reply.code(200).send({
        success: true,
        anomaly: result.anomaly,
        investigation_suggestions: result.investigation_suggestions || [],
        related_anomalies: result.related_anomalies || []
      });
      
    } catch (error) {
      fastify.log.error(`Error retrieving anomaly details: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve anomaly details',
        message: error.message
      });
    }
  });
  
  // Route: Update anomaly feedback
  fastify.put('/anomaly/:anomalyId/feedback', {
    schema: {
      description: 'Update feedback for an anomaly detection',
      tags: ['Anomaly Detection', 'Feedback'],
      params: {
        type: 'object',
        required: ['anomalyId'],
        properties: {
          anomalyId: { type: 'string' }
        }
      },
      body: {
        type: 'object',
        required: ['is_true_positive'],
        properties: {
          is_true_positive: { type: 'boolean' },
          feedback_notes: { type: 'string' },
          resolution_action: { type: 'string' },
          confidence_rating: { 
            type: 'integer',
            minimum: 1,
            maximum: 5
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { anomalyId } = request.params;
      const { 
        is_true_positive, 
        feedback_notes,
        resolution_action,
        confidence_rating
      } = request.body;
      
      fastify.log.info(`Updating feedback for anomaly: ${anomalyId}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'update_anomaly_feedback',
        '--anomaly_id', anomalyId,
        '--is_true_positive', is_true_positive.toString(),
        '--feedback_notes', feedback_notes || '',
        '--resolution_action', resolution_action || '',
        '--confidence_rating', (confidence_rating || 3).toString(),
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        anomaly_id: anomalyId,
        feedback_updated: result.updated || false,
        updated_statistics: result.detection_statistics || {}
      });
      
    } catch (error) {
      fastify.log.error(`Error updating anomaly feedback: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to update anomaly feedback',
        message: error.message
      });
    }
  });
  
  // Route: Manage alert rules
  fastify.post('/rules', {
    schema: {
      description: 'Create a new anomaly detection rule',
      tags: ['Anomaly Detection', 'Rules'],
      body: alertRuleSchema
    }
  }, async (request, reply) => {
    try {
      const ruleData = request.body;
      const ruleId = crypto.randomUUID();
      
      fastify.log.info(`Creating anomaly detection rule: ${ruleData.rule_name}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'create_alert_rule',
        '--rule_id', ruleId,
        '--rule_data', JSON.stringify(ruleData),
        '--output_format', 'json'
      ]);
      
      reply.code(201).send({
        success: true,
        rule_id: ruleId,
        rule_name: ruleData.rule_name,
        created: result.created || false,
        rule_configuration: result.rule_configuration || {}
      });
      
    } catch (error) {
      fastify.log.error(`Error creating alert rule: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to create alert rule',
        message: error.message
      });
    }
  });
  
  // Route: List alert rules
  fastify.get('/rules', {
    schema: {
      description: 'List all anomaly detection rules',
      tags: ['Anomaly Detection', 'Rules'],
      querystring: {
        type: 'object',
        properties: {
          active_only: { type: 'boolean', default: true },
          category: {
            type: 'string',
            enum: ['application', 'behavior', 'compliance', 'performance']
          },
          rule_type: {
            type: 'string',
            enum: ['threshold', 'statistical', 'pattern', 'ml_based']
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { active_only = true, category, rule_type } = request.query;
      
      fastify.log.info('Retrieving anomaly detection rules');
      
      const listArgs = [
        'list_alert_rules',
        '--active_only', active_only.toString(),
        '--output_format', 'json'
      ];
      
      if (category) {
        listArgs.push('--category', category);
      }
      
      if (rule_type) {
        listArgs.push('--rule_type', rule_type);
      }
      
      const result = await executePythonScript('anomaly_detection_executor.py', listArgs);
      
      reply.code(200).send({
        success: true,
        rules: result.rules || [],
        total_rules: result.total_rules || 0,
        active_rules: result.active_rules || 0,
        filters: { active_only, category, rule_type }
      });
      
    } catch (error) {
      fastify.log.error(`Error retrieving alert rules: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve alert rules',
        message: error.message
      });
    }
  });
  
  // Route: Update alert rule
  fastify.put('/rules/:ruleId', {
    schema: {
      description: 'Update an existing anomaly detection rule',
      tags: ['Anomaly Detection', 'Rules'],
      params: {
        type: 'object',
        required: ['ruleId'],
        properties: {
          ruleId: { type: 'string' }
        }
      },
      body: alertRuleSchema
    }
  }, async (request, reply) => {
    try {
      const { ruleId } = request.params;
      const ruleData = request.body;
      
      fastify.log.info(`Updating alert rule: ${ruleId}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'update_alert_rule',
        '--rule_id', ruleId,
        '--rule_data', JSON.stringify(ruleData),
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        rule_id: ruleId,
        updated: result.updated || false,
        rule_configuration: result.rule_configuration || {}
      });
      
    } catch (error) {
      fastify.log.error(`Error updating alert rule: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to update alert rule',
        message: error.message
      });
    }
  });
  
  // Route: Delete alert rule
  fastify.delete('/rules/:ruleId', {
    schema: {
      description: 'Delete an anomaly detection rule',
      tags: ['Anomaly Detection', 'Rules'],
      params: {
        type: 'object',
        required: ['ruleId'],
        properties: {
          ruleId: { type: 'string' }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { ruleId } = request.params;
      
      fastify.log.info(`Deleting alert rule: ${ruleId}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'delete_alert_rule',
        '--rule_id', ruleId,
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        rule_id: ruleId,
        deleted: result.deleted || false
      });
      
    } catch (error) {
      fastify.log.error(`Error deleting alert rule: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to delete alert rule',
        message: error.message
      });
    }
  });
  
  // Route: Start investigation session
  fastify.post('/investigations', {
    schema: {
      description: 'Start a new anomaly investigation session',
      tags: ['Anomaly Detection', 'Investigation'],
      body: {
        type: 'object',
        required: ['anomaly_ids', 'session_name'],
        properties: {
          anomaly_ids: {
            type: 'array',
            items: { type: 'string' },
            minItems: 1
          },
          session_name: { type: 'string', minLength: 1 },
          investigator_id: { type: 'string' },
          priority_level: {
            type: 'string',
            enum: ['low', 'medium', 'high', 'urgent'],
            default: 'medium'
          },
          initial_hypothesis: {
            type: 'array',
            items: { type: 'string' },
            default: []
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { 
        anomaly_ids, 
        session_name, 
        investigator_id,
        priority_level = 'medium',
        initial_hypothesis = []
      } = request.body;
      
      const sessionId = crypto.randomUUID();
      
      fastify.log.info(`Starting investigation session: ${session_name}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'start_investigation',
        '--session_id', sessionId,
        '--anomaly_ids', anomaly_ids.join(','),
        '--session_name', session_name,
        '--investigator_id', investigator_id || 'anonymous',
        '--priority_level', priority_level,
        '--initial_hypothesis', JSON.stringify(initial_hypothesis),
        '--output_format', 'json'
      ]);
      
      reply.code(201).send({
        success: true,
        session_id: sessionId,
        session_name,
        anomaly_count: anomaly_ids.length,
        investigation_status: 'active',
        initial_suggestions: result.initial_suggestions || [],
        created_at: new Date().toISOString()
      });
      
    } catch (error) {
      fastify.log.error(`Error starting investigation: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to start investigation',
        message: error.message
      });
    }
  });
  
  // Route: Get investigation sessions
  fastify.get('/investigations', {
    schema: {
      description: 'List investigation sessions',
      tags: ['Anomaly Detection', 'Investigation'],
      querystring: {
        type: 'object',
        properties: {
          status: {
            type: 'string',
            enum: ['active', 'paused', 'completed', 'cancelled']
          },
          investigator_id: { type: 'string' },
          limit: { type: 'integer', minimum: 1, maximum: 100, default: 10 }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { status, investigator_id, limit = 10 } = request.query;
      
      fastify.log.info('Retrieving investigation sessions');
      
      const listArgs = [
        'list_investigations',
        '--limit', limit.toString(),
        '--output_format', 'json'
      ];
      
      if (status) {
        listArgs.push('--status', status);
      }
      
      if (investigator_id) {
        listArgs.push('--investigator_id', investigator_id);
      }
      
      const result = await executePythonScript('anomaly_detection_executor.py', listArgs);
      
      reply.code(200).send({
        success: true,
        investigations: result.investigations || [],
        total_investigations: result.total_investigations || 0,
        filters: { status, investigator_id, limit }
      });
      
    } catch (error) {
      fastify.log.error(`Error retrieving investigations: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve investigations',
        message: error.message
      });
    }
  });
  
  // Route: Get investigation details
  fastify.get('/investigations/:sessionId', {
    schema: {
      description: 'Get detailed information about an investigation session',
      tags: ['Anomaly Detection', 'Investigation'],
      params: {
        type: 'object',
        required: ['sessionId'],
        properties: {
          sessionId: { type: 'string' }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { sessionId } = request.params;
      
      fastify.log.info(`Retrieving investigation details: ${sessionId}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'get_investigation_details',
        '--session_id', sessionId,
        '--output_format', 'json'
      ]);
      
      if (!result.investigation) {
        reply.code(404).send({
          success: false,
          error: 'Investigation not found',
          session_id: sessionId
        });
        return;
      }
      
      reply.code(200).send({
        success: true,
        investigation: result.investigation,
        analysis_results: result.analysis_results || {},
        progress_summary: result.progress_summary || {}
      });
      
    } catch (error) {
      fastify.log.error(`Error retrieving investigation details: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve investigation details',
        message: error.message
      });
    }
  });
  
  // Route: Add investigation evidence
  fastify.post('/investigations/:sessionId/evidence', {
    schema: {
      description: 'Add evidence to an investigation session',
      tags: ['Anomaly Detection', 'Investigation'],
      params: {
        type: 'object',
        required: ['sessionId'],
        properties: {
          sessionId: { type: 'string' }
        }
      },
      body: {
        type: 'object',
        required: ['evidence_type', 'evidence_data'],
        properties: {
          evidence_type: { 
            type: 'string',
            enum: ['data_analysis', 'log_review', 'user_input', 'system_check', 'external_source']
          },
          evidence_data: { type: 'object' },
          evidence_description: { type: 'string' },
          confidence_level: { 
            type: 'number',
            minimum: 0.0,
            maximum: 1.0,
            default: 0.8
          },
          source: { type: 'string' },
          tags: { 
            type: 'array',
            items: { type: 'string' },
            default: []
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { sessionId } = request.params;
      const evidenceData = request.body;
      
      fastify.log.info(`Adding evidence to investigation: ${sessionId}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'add_investigation_evidence',
        '--session_id', sessionId,
        '--evidence_data', JSON.stringify(evidenceData),
        '--output_format', 'json'
      ]);
      
      reply.code(201).send({
        success: true,
        session_id: sessionId,
        evidence_added: result.evidence_added || false,
        evidence_id: result.evidence_id,
        total_evidence: result.total_evidence || 0
      });
      
    } catch (error) {
      fastify.log.error(`Error adding investigation evidence: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to add investigation evidence',
        message: error.message
      });
    }
  });
  
  // Route: Perform pattern analysis
  fastify.post('/investigations/:sessionId/analyze', {
    schema: {
      description: 'Perform pattern analysis on investigation data',
      tags: ['Anomaly Detection', 'Investigation'],
      params: {
        type: 'object',
        required: ['sessionId'],
        properties: {
          sessionId: { type: 'string' }
        }
      },
      body: {
        type: 'object',
        required: ['analysis_data'],
        properties: {
          analysis_data: { type: 'object' },
          analysis_type: {
            type: 'string',
            enum: ['temporal', 'statistical', 'correlation', 'comprehensive'],
            default: 'comprehensive'
          },
          include_visualizations: { type: 'boolean', default: true }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { sessionId } = request.params;
      const { analysis_data, analysis_type = 'comprehensive', include_visualizations = true } = request.body;
      
      fastify.log.info(`Performing pattern analysis for investigation: ${sessionId}`);
      
      const result = await executePythonScript('anomaly_detection_executor.py', [
        'analyze_investigation_patterns',
        '--session_id', sessionId,
        '--analysis_data', JSON.stringify(analysis_data),
        '--analysis_type', analysis_type,
        '--include_visualizations', include_visualizations.toString(),
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        session_id: sessionId,
        analysis_type,
        analysis_results: result.analysis_results || {},
        patterns_found: result.patterns_found || 0,
        insights: result.insights || [],
        visualizations: include_visualizations ? (result.visualizations || {}) : {}
      });
      
    } catch (error) {
      fastify.log.error(`Error performing pattern analysis: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to perform pattern analysis',
        message: error.message
      });
    }
  });
  
  // Route: Get system configuration
  fastify.get('/config', {
    schema: {
      description: 'Get anomaly detection system configuration',
      tags: ['Anomaly Detection', 'Configuration']
    }
  }, async (request, reply) => {
    try {
      reply.code(200).send({
        success: true,
        configuration: {
          anomaly_detection_enabled: process.env.ANOMALY_DETECTION_ENABLED === 'true',
          timeout: parseInt(process.env.ANOMALY_DETECTION_TIMEOUT || '180000'),
          max_data_points: parseInt(process.env.MAX_ANOMALY_DATA_POINTS || '10000'),
          default_methods: (process.env.DEFAULT_ANOMALY_METHODS || 'statistical,ml_based').split(','),
          supported_methods: ['statistical', 'ml_based', 'rule_based', 'hybrid'],
          statistical_methods: ['z_score', 'modified_z_score', 'iqr', 'grubbs_test', 'dixon_test'],
          ml_methods: ['isolation_forest', 'local_outlier_factor', 'elliptic_envelope', 'dbscan'],
          severity_levels: ['low', 'medium', 'high', 'critical'],
          categories: ['application', 'behavior', 'transaction', 'compliance', 'data_quality'],
          alert_channels: ['email', 'sms', 'slack', 'in_app'],
          real_time_detection: process.env.ENABLE_REAL_TIME_ANOMALY_DETECTION === 'true'
        },
        feature_flags: {
          statistical_detection: process.env.ENABLE_STATISTICAL_ANOMALY_DETECTION !== 'false',
          ml_detection: process.env.ENABLE_ML_ANOMALY_DETECTION !== 'false',
          rule_based_detection: process.env.ENABLE_RULE_BASED_ANOMALY_DETECTION !== 'false',
          investigation_tools: process.env.ENABLE_INVESTIGATION_TOOLS !== 'false',
          alert_management: process.env.ENABLE_ANOMALY_ALERT_MANAGEMENT !== 'false',
          pattern_analysis: process.env.ENABLE_PATTERN_ANALYSIS !== 'false',
          feedback_learning: process.env.ENABLE_ANOMALY_FEEDBACK_LEARNING !== 'false'
        }
      });
    } catch (error) {
      fastify.log.error(`Error retrieving configuration: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve configuration',
        message: error.message
      });
    }
  });
  
  // Route: Health check
  fastify.get('/health', {
    schema: {
      description: 'Health check for anomaly detection services',
      tags: ['Anomaly Detection', 'Health']
    }
  }, async (request, reply) => {
    try {
      const healthCheck = await executePythonScript('anomaly_detection_executor.py', [
        'health_check',
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        status: 'healthy',
        timestamp: new Date().toISOString(),
        services: healthCheck.services || {},
        performance: healthCheck.performance || {},
        dependencies: healthCheck.dependencies || {},
        detection_statistics: healthCheck.detection_statistics || {}
      });
      
    } catch (error) {
      fastify.log.error(`Anomaly detection health check failed: ${error.message}`);
      reply.code(503).send({
        success: false,
        status: 'unhealthy',
        error: 'Anomaly detection services unavailable',
        message: error.message
      });
    }
  });
}

module.exports = anomalyDetectionInterfaceRoutes;
