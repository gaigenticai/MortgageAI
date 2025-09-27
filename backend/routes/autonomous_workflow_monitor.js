/**
 * Autonomous Workflow Monitor API Routes
 * 
 * This module provides RESTful API endpoints for autonomous workflow monitoring,
 * including real-time agent decision tracking, learning pattern analysis, and
 * performance analytics with predictive insights.
 * 
 * Features:
 * - Real-time workflow monitoring and tracking
 * - Agent decision logging and analysis
 * - Learning pattern detection and visualization
 * - Performance analytics and optimization recommendations
 * - Bottleneck detection and workflow optimization
 * - Automated alerting and notification system
 * - Comprehensive reporting and dashboard data
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

/**
 * Autonomous Workflow Monitor Routes Plugin
 * @param {object} fastify - Fastify instance
 * @param {object} options - Plugin options
 */
async function autonomousWorkflowMonitorRoutes(fastify, options) {
  // Configuration
  const PYTHON_SCRIPT_PATH = path.join(__dirname, '../agents/utils/workflow_monitor_executor.py');
  const MONITORING_DATA_DIR = path.join(__dirname, '../../data/workflow_monitoring');
  const MONITORING_CACHE_DIR = path.join(__dirname, '../../cache/workflow_monitoring');
  const MONITORING_LOGS_DIR = path.join(__dirname, '../../logs/workflow_monitoring');
  
  // Ensure directories exist
  await ensureDirectoryExists(MONITORING_DATA_DIR);
  await ensureDirectoryExists(MONITORING_CACHE_DIR);
  await ensureDirectoryExists(MONITORING_LOGS_DIR);

  // In-memory monitoring state (in production, this would be Redis/database)
  let monitoringActive = false;
  let monitoringSession = null;

  // Schema definitions
  const agentDecisionSchema = {
    type: 'object',
    required: ['agent_id', 'decision_type', 'input_data', 'output_data', 'confidence_score', 'processing_time'],
    properties: {
      agent_id: { type: 'string' },
      decision_type: { 
        type: 'string',
        enum: ['classification', 'recommendation', 'validation', 'escalation', 'approval', 'rejection', 'routing', 'optimization', 'prediction', 'intervention']
      },
      input_data: { type: 'object' },
      output_data: { type: 'object' },
      confidence_score: { type: 'number', minimum: 0, maximum: 1 },
      processing_time: { type: 'number', minimum: 0 },
      workflow_id: { type: 'string' },
      context: { type: 'object' },
      metadata: { type: 'object' },
      correctness_score: { type: 'number', minimum: 0, maximum: 1 },
      user_feedback_score: { type: 'number', minimum: 0, maximum: 1 }
    }
  };

  const workflowExecutionSchema = {
    type: 'object',
    required: ['workflow_name', 'steps'],
    properties: {
      workflow_name: { type: 'string' },
      steps: { type: 'array', items: { type: 'string' } },
      dependencies: { type: 'object' },
      parallel_branches: { type: 'array' },
      input_parameters: { type: 'object' },
      output_results: { type: 'object' },
      metadata: { type: 'object' }
    }
  };

  const performanceMetricSchema = {
    type: 'object',
    required: ['agent_id', 'metric_type', 'value'],
    properties: {
      agent_id: { type: 'string' },
      metric_type: {
        type: 'string',
        enum: ['accuracy', 'precision', 'recall', 'f1_score', 'processing_time', 'throughput', 'error_rate', 'resource_utilization', 'user_satisfaction', 'cost_per_decision']
      },
      value: { type: 'number' },
      workflow_id: { type: 'string' },
      decision_id: { type: 'string' },
      measurement_context: { type: 'object' },
      measurement_confidence: { type: 'number', minimum: 0, maximum: 1 },
      data_quality_score: { type: 'number', minimum: 0, maximum: 1 }
    }
  };

  // Helper function to ensure directory exists
  async function ensureDirectoryExists(dirPath) {
    try {
      await fs.access(dirPath);
    } catch (error) {
      await fs.mkdir(dirPath, { recursive: true });
    }
  }

  // Helper function to execute Python scripts
  async function executePythonScript(scriptArgs, inputData = null) {
    return new Promise((resolve, reject) => {
      const python = spawn('python3', [PYTHON_SCRIPT_PATH, ...scriptArgs], {
        cwd: path.dirname(PYTHON_SCRIPT_PATH),
        stdio: ['pipe', 'pipe', 'pipe']
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
            reject(new Error(`Failed to parse Python script output: ${parseError.message}`));
          }
        } else {
          reject(new Error(`Python script failed with code ${code}: ${stderr}`));
        }
      });

      python.on('error', (error) => {
        reject(new Error(`Failed to spawn Python script: ${error.message}`));
      });

      // Send input data if provided
      if (inputData) {
        python.stdin.write(JSON.stringify(inputData));
        python.stdin.end();
      }
    });
  }

  // Helper function to generate cache key
  function generateCacheKey(data) {
    return crypto.createHash('sha256').update(JSON.stringify(data)).digest('hex');
  }

  // Helper function to get cached result
  async function getCachedResult(cacheKey, maxAgeMinutes = 30) {
    try {
      const cacheFile = path.join(MONITORING_CACHE_DIR, `${cacheKey}.json`);
      const stats = await fs.stat(cacheFile);
      
      // Check if cache is still valid
      const ageMinutes = (Date.now() - stats.mtime.getTime()) / (1000 * 60);
      if (ageMinutes <= maxAgeMinutes) {
        const cachedData = await fs.readFile(cacheFile, 'utf8');
        return JSON.parse(cachedData);
      }
    } catch (error) {
      // Cache miss or error - will proceed to compute fresh result
    }
    return null;
  }

  // Helper function to cache result
  async function setCachedResult(cacheKey, result) {
    try {
      const cacheFile = path.join(MONITORING_CACHE_DIR, `${cacheKey}.json`);
      await fs.writeFile(cacheFile, JSON.stringify(result, null, 2));
    } catch (error) {
      fastify.log.warn(`Failed to cache result: ${error.message}`);
    }
  }

  // Route: Start workflow monitoring
  fastify.post('/start-monitoring', {
    schema: {
      description: 'Start autonomous workflow monitoring',
      tags: ['Autonomous Workflow Monitor'],
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            session_id: { type: 'string' },
            monitoring_status: { type: 'string' },
            start_time: { type: 'string', format: 'date-time' },
            configuration: { type: 'object' }
          }
        },
        400: {
          type: 'object',
          properties: {
            error: { type: 'string' },
            details: { type: 'string' }
          }
        },
        500: {
          type: 'object',
          properties: {
            error: { type: 'string' },
            details: { type: 'string' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      if (monitoringActive) {
        return reply.status(400).send({
          error: 'Monitoring already active',
          details: 'Stop current monitoring session before starting a new one'
        });
      }

      // Start monitoring session
      const sessionId = uuidv4();
      const startTime = new Date().toISOString();

      const result = await executePythonScript(
        ['start_monitoring'],
        {
          session_id: sessionId,
          configuration: {
            metrics_collection_interval: parseInt(process.env.MONITORING_COLLECTION_INTERVAL) || 60,
            enable_real_time_analysis: process.env.ENABLE_REAL_TIME_ANALYSIS === 'true',
            alert_thresholds: {
              accuracy: parseFloat(process.env.ALERT_THRESHOLD_ACCURACY) || 0.8,
              processing_time: parseFloat(process.env.ALERT_THRESHOLD_PROCESSING_TIME) || 10.0,
              error_rate: parseFloat(process.env.ALERT_THRESHOLD_ERROR_RATE) || 0.1
            }
          }
        }
      );

      if (result.success) {
        monitoringActive = true;
        monitoringSession = {
          session_id: sessionId,
          start_time: startTime,
          configuration: result.configuration
        };

        return reply.send({
          success: true,
          session_id: sessionId,
          monitoring_status: 'active',
          start_time: startTime,
          configuration: result.configuration
        });
      } else {
        throw new Error(result.error || 'Failed to start monitoring');
      }

    } catch (error) {
      fastify.log.error(`Error starting workflow monitoring: ${error.message}`);
      return reply.status(500).send({
        error: 'Failed to start workflow monitoring',
        details: error.message
      });
    }
  });

  // Route: Stop workflow monitoring
  fastify.post('/stop-monitoring', {
    schema: {
      description: 'Stop autonomous workflow monitoring',
      tags: ['Autonomous Workflow Monitor'],
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            session_id: { type: 'string' },
            monitoring_status: { type: 'string' },
            end_time: { type: 'string', format: 'date-time' },
            session_summary: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      if (!monitoringActive || !monitoringSession) {
        return reply.status(400).send({
          error: 'No active monitoring session',
          details: 'Start monitoring before attempting to stop'
        });
      }

      const endTime = new Date().toISOString();

      const result = await executePythonScript(
        ['stop_monitoring'],
        {
          session_id: monitoringSession.session_id,
          end_time: endTime
        }
      );

      monitoringActive = false;
      const sessionData = monitoringSession;
      monitoringSession = null;

      return reply.send({
        success: true,
        session_id: sessionData.session_id,
        monitoring_status: 'stopped',
        end_time: endTime,
        session_summary: result.session_summary || {}
      });

    } catch (error) {
      fastify.log.error(`Error stopping workflow monitoring: ${error.message}`);
      return reply.status(500).send({
        error: 'Failed to stop workflow monitoring',
        details: error.message
      });
    }
  });

  // Route: Log agent decision
  fastify.post('/log-decision', {
    schema: {
      description: 'Log an agent decision for monitoring and analysis',
      tags: ['Autonomous Workflow Monitor'],
      body: agentDecisionSchema,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            decision_id: { type: 'string' },
            logged_at: { type: 'string', format: 'date-time' },
            analysis_results: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const decisionData = request.body;
      
      // Generate decision ID if not provided
      const decisionId = decisionData.decision_id || uuidv4();
      const loggedAt = new Date().toISOString();

      const result = await executePythonScript(
        ['log_decision'],
        {
          decision_id: decisionId,
          agent_id: decisionData.agent_id,
          decision_type: decisionData.decision_type,
          input_data: decisionData.input_data,
          output_data: decisionData.output_data,
          confidence_score: decisionData.confidence_score,
          processing_time: decisionData.processing_time,
          workflow_id: decisionData.workflow_id || 'default_workflow',
          context: decisionData.context || {},
          metadata: decisionData.metadata || {},
          correctness_score: decisionData.correctness_score,
          user_feedback_score: decisionData.user_feedback_score,
          logged_at: loggedAt
        }
      );

      return reply.send({
        success: true,
        decision_id: decisionId,
        logged_at: loggedAt,
        analysis_results: result.analysis_results || {}
      });

    } catch (error) {
      fastify.log.error(`Error logging agent decision: ${error.message}`);
      return reply.status(500).send({
        error: 'Failed to log agent decision',
        details: error.message
      });
    }
  });

  // Route: Log workflow execution
  fastify.post('/log-workflow', {
    schema: {
      description: 'Log a complete workflow execution for monitoring',
      tags: ['Autonomous Workflow Monitor'],
      body: workflowExecutionSchema,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            workflow_id: { type: 'string' },
            execution_id: { type: 'string' },
            logged_at: { type: 'string', format: 'date-time' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const workflowData = request.body;
      
      const workflowId = workflowData.workflow_id || uuidv4();
      const executionId = uuidv4();
      const loggedAt = new Date().toISOString();

      const result = await executePythonScript(
        ['log_workflow'],
        {
          workflow_id: workflowId,
          execution_id: executionId,
          workflow_name: workflowData.workflow_name,
          steps: workflowData.steps,
          dependencies: workflowData.dependencies || {},
          parallel_branches: workflowData.parallel_branches || [],
          input_parameters: workflowData.input_parameters || {},
          output_results: workflowData.output_results || {},
          metadata: workflowData.metadata || {},
          logged_at: loggedAt
        }
      );

      return reply.send({
        success: true,
        workflow_id: workflowId,
        execution_id: executionId,
        logged_at: loggedAt
      });

    } catch (error) {
      fastify.log.error(`Error logging workflow execution: ${error.message}`);
      return reply.status(500).send({
        error: 'Failed to log workflow execution',
        details: error.message
      });
    }
  });

  // Route: Log performance metric
  fastify.post('/log-metric', {
    schema: {
      description: 'Log a performance metric for monitoring',
      tags: ['Autonomous Workflow Monitor'],
      body: performanceMetricSchema,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            metric_id: { type: 'string' },
            logged_at: { type: 'string', format: 'date-time' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const metricData = request.body;
      
      const metricId = uuidv4();
      const loggedAt = new Date().toISOString();

      const result = await executePythonScript(
        ['log_metric'],
        {
          metric_id: metricId,
          agent_id: metricData.agent_id,
          metric_type: metricData.metric_type,
          value: metricData.value,
          workflow_id: metricData.workflow_id,
          decision_id: metricData.decision_id,
          measurement_context: metricData.measurement_context || {},
          measurement_confidence: metricData.measurement_confidence || 1.0,
          data_quality_score: metricData.data_quality_score || 1.0,
          logged_at: loggedAt
        }
      );

      return reply.send({
        success: true,
        metric_id: metricId,
        logged_at: loggedAt
      });

    } catch (error) {
      fastify.log.error(`Error logging performance metric: ${error.message}`);
      return reply.status(500).send({
        error: 'Failed to log performance metric',
        details: error.message
      });
    }
  });

  // Route: Get performance dashboard data
  fastify.get('/dashboard', {
    schema: {
      description: 'Get comprehensive performance dashboard data',
      tags: ['Autonomous Workflow Monitor'],
      querystring: {
        type: 'object',
        properties: {
          time_range: { type: 'string', enum: ['1h', '24h', '7d', '30d'], default: '24h' },
          agent_filter: { type: 'string' },
          include_predictions: { type: 'boolean', default: true },
          include_optimization: { type: 'boolean', default: true }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            dashboard_data: {
              type: 'object',
              properties: {
                timestamp: { type: 'string', format: 'date-time' },
                performance_summary: { type: 'object' },
                learning_patterns: { type: 'array' },
                workflow_statistics: { type: 'object' },
                optimization_recommendations: { type: 'array' },
                active_alerts: { type: 'integer' },
                system_health: { type: 'object' }
              }
            },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const { time_range = '24h', agent_filter, include_predictions = true, include_optimization = true } = request.query;

      // Check cache first
      const cacheKey = generateCacheKey({ time_range, agent_filter, include_predictions, include_optimization });
      let cachedResult = await getCachedResult(cacheKey, 5); // 5-minute cache for dashboard data
      
      if (cachedResult) {
        fastify.log.info(`Returning cached dashboard data for key: ${cacheKey}`);
        return reply.send({
          ...cachedResult,
          cached: true
        });
      }

      const result = await executePythonScript(
        ['get_dashboard_data'],
        {
          time_range,
          agent_filter,
          include_predictions,
          include_optimization,
          monitoring_session: monitoringSession
        }
      );

      if (!result.success) {
        throw new Error(result.error || 'Failed to get dashboard data');
      }

      const processingTime = (Date.now() - startTime) / 1000;
      const response = {
        success: true,
        dashboard_data: result.dashboard_data,
        processing_time: processingTime,
        cached: false
      };

      // Cache the result
      await setCachedResult(cacheKey, response);

      return reply.send(response);

    } catch (error) {
      fastify.log.error(`Error getting dashboard data: ${error.message}`);
      return reply.status(500).send({
        error: 'Failed to get dashboard data',
        details: error.message
      });
    }
  });

  // Route: Analyze learning patterns
  fastify.post('/analyze-patterns', {
    schema: {
      description: 'Analyze learning patterns in agent behavior',
      tags: ['Autonomous Workflow Monitor'],
      body: {
        type: 'object',
        properties: {
          time_range: { type: 'string', enum: ['1h', '24h', '7d', '30d'], default: '24h' },
          agent_ids: { type: 'array', items: { type: 'string' } },
          pattern_types: { type: 'array', items: { type: 'string' } },
          minimum_confidence: { type: 'number', minimum: 0, maximum: 1, default: 0.5 },
          include_recommendations: { type: 'boolean', default: true }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            analysis_id: { type: 'string' },
            patterns_detected: { type: 'array' },
            pattern_summary: { type: 'object' },
            insights: { type: 'array' },
            recommendations: { type: 'array' },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const {
        time_range = '24h',
        agent_ids = [],
        pattern_types = [],
        minimum_confidence = 0.5,
        include_recommendations = true
      } = request.body;

      const result = await executePythonScript(
        ['analyze_patterns'],
        {
          time_range,
          agent_ids,
          pattern_types,
          minimum_confidence,
          include_recommendations
        }
      );

      if (!result.success) {
        throw new Error(result.error || 'Pattern analysis failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        analysis_id: result.analysis_id,
        patterns_detected: result.patterns_detected,
        pattern_summary: result.pattern_summary,
        insights: result.insights,
        recommendations: result.recommendations,
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Error analyzing learning patterns: ${error.message}`);
      return reply.status(500).send({
        error: 'Learning pattern analysis failed',
        details: error.message
      });
    }
  });

  // Route: Get optimization recommendations
  fastify.post('/optimize-workflow', {
    schema: {
      description: 'Get workflow optimization recommendations',
      tags: ['Autonomous Workflow Monitor'],
      body: {
        type: 'object',
        properties: {
          workflow_ids: { type: 'array', items: { type: 'string' } },
          optimization_scope: { type: 'string', enum: ['performance', 'cost', 'quality', 'comprehensive'], default: 'comprehensive' },
          include_bottleneck_analysis: { type: 'boolean', default: true },
          include_resource_optimization: { type: 'boolean', default: true },
          include_predictions: { type: 'boolean', default: true }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            optimization_id: { type: 'string' },
            bottlenecks: { type: 'array' },
            optimization_opportunities: { type: 'array' },
            predicted_improvements: { type: 'object' },
            implementation_recommendations: { type: 'array' },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const {
        workflow_ids = [],
        optimization_scope = 'comprehensive',
        include_bottleneck_analysis = true,
        include_resource_optimization = true,
        include_predictions = true
      } = request.body;

      const result = await executePythonScript(
        ['optimize_workflow'],
        {
          workflow_ids,
          optimization_scope,
          include_bottleneck_analysis,
          include_resource_optimization,
          include_predictions
        }
      );

      if (!result.success) {
        throw new Error(result.error || 'Workflow optimization failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        optimization_id: result.optimization_id,
        bottlenecks: result.bottlenecks,
        optimization_opportunities: result.optimization_opportunities,
        predicted_improvements: result.predicted_improvements,
        implementation_recommendations: result.implementation_recommendations,
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Error optimizing workflow: ${error.message}`);
      return reply.status(500).send({
        error: 'Workflow optimization failed',
        details: error.message
      });
    }
  });

  // Route: Generate learning insights report
  fastify.post('/generate-report', {
    schema: {
      description: 'Generate comprehensive learning insights report',
      tags: ['Autonomous Workflow Monitor'],
      body: {
        type: 'object',
        properties: {
          report_type: { type: 'string', enum: ['learning_insights', 'performance_summary', 'optimization_report', 'comprehensive'], default: 'comprehensive' },
          time_period: { type: 'string', enum: ['24h', '7d', '30d'], default: '24h' },
          include_visualizations: { type: 'boolean', default: true },
          include_recommendations: { type: 'boolean', default: true },
          export_format: { type: 'string', enum: ['json', 'pdf', 'html'], default: 'json' }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            report_id: { type: 'string' },
            report_data: { type: 'object' },
            export_url: { type: 'string' },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const {
        report_type = 'comprehensive',
        time_period = '24h',
        include_visualizations = true,
        include_recommendations = true,
        export_format = 'json'
      } = request.body;

      const result = await executePythonScript(
        ['generate_report'],
        {
          report_type,
          time_period,
          include_visualizations,
          include_recommendations,
          export_format
        }
      );

      if (!result.success) {
        throw new Error(result.error || 'Report generation failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        report_id: result.report_id,
        report_data: result.report_data,
        export_url: result.export_url || null,
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Error generating report: ${error.message}`);
      return reply.status(500).send({
        error: 'Report generation failed',
        details: error.message
      });
    }
  });

  // Route: Set alert thresholds
  fastify.post('/set-alert-thresholds', {
    schema: {
      description: 'Set custom alert thresholds for monitoring',
      tags: ['Autonomous Workflow Monitor'],
      body: {
        type: 'object',
        properties: {
          thresholds: {
            type: 'object',
            properties: {
              accuracy: { type: 'number', minimum: 0, maximum: 1 },
              processing_time: { type: 'number', minimum: 0 },
              error_rate: { type: 'number', minimum: 0, maximum: 1 },
              throughput: { type: 'number', minimum: 0 },
              resource_utilization: { type: 'number', minimum: 0, maximum: 1 }
            }
          },
          agent_specific_thresholds: { type: 'object' },
          notification_settings: { type: 'object' }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            thresholds_updated: { type: 'object' },
            updated_at: { type: 'string', format: 'date-time' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { thresholds = {}, agent_specific_thresholds = {}, notification_settings = {} } = request.body;

      const result = await executePythonScript(
        ['set_alert_thresholds'],
        {
          thresholds,
          agent_specific_thresholds,
          notification_settings
        }
      );

      if (!result.success) {
        throw new Error(result.error || 'Failed to set alert thresholds');
      }

      return reply.send({
        success: true,
        thresholds_updated: result.thresholds_updated,
        updated_at: new Date().toISOString()
      });

    } catch (error) {
      fastify.log.error(`Error setting alert thresholds: ${error.message}`);
      return reply.status(500).send({
        error: 'Failed to set alert thresholds',
        details: error.message
      });
    }
  });

  // Route: Export monitoring data
  fastify.post('/export-data', {
    schema: {
      description: 'Export monitoring data for analysis or backup',
      tags: ['Autonomous Workflow Monitor'],
      body: {
        type: 'object',
        properties: {
          export_type: { type: 'string', enum: ['decisions', 'workflows', 'metrics', 'patterns', 'all'], default: 'all' },
          time_range: { type: 'string', enum: ['1h', '24h', '7d', '30d'], default: '24h' },
          format: { type: 'string', enum: ['json', 'csv', 'excel'], default: 'json' },
          include_metadata: { type: 'boolean', default: true },
          agent_filter: { type: 'array', items: { type: 'string' } }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            export_id: { type: 'string' },
            export_url: { type: 'string' },
            file_size: { type: 'integer' },
            record_count: { type: 'integer' },
            expires_at: { type: 'string', format: 'date-time' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const {
        export_type = 'all',
        time_range = '24h',
        format = 'json',
        include_metadata = true,
        agent_filter = []
      } = request.body;

      const result = await executePythonScript(
        ['export_data'],
        {
          export_type,
          time_range,
          format,
          include_metadata,
          agent_filter,
          export_id: uuidv4()
        }
      );

      if (!result.success) {
        throw new Error(result.error || 'Data export failed');
      }

      return reply.send({
        success: true,
        export_id: result.export_id,
        export_url: result.export_url,
        file_size: result.file_size,
        record_count: result.record_count,
        expires_at: result.expires_at
      });

    } catch (error) {
      fastify.log.error(`Error exporting monitoring data: ${error.message}`);
      return reply.status(500).send({
        error: 'Data export failed',
        details: error.message
      });
    }
  });

  // Route: Get monitoring status
  fastify.get('/status', {
    schema: {
      description: 'Get current monitoring status and configuration',
      tags: ['Autonomous Workflow Monitor'],
      response: {
        200: {
          type: 'object',
          properties: {
            monitoring_active: { type: 'boolean' },
            session_info: { type: 'object' },
            system_health: { type: 'object' },
            configuration: { type: 'object' },
            statistics: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const result = await executePythonScript(['get_status']);

      return reply.send({
        monitoring_active: monitoringActive,
        session_info: monitoringSession,
        system_health: result.system_health || {},
        configuration: result.configuration || {},
        statistics: result.statistics || {}
      });

    } catch (error) {
      fastify.log.error(`Error getting monitoring status: ${error.message}`);
      return reply.status(500).send({
        error: 'Failed to get monitoring status',
        details: error.message
      });
    }
  });

  // Route: Health check
  fastify.get('/health', {
    schema: {
      description: 'Health check for autonomous workflow monitor service',
      tags: ['Autonomous Workflow Monitor'],
      response: {
        200: {
          type: 'object',
          properties: {
            status: { type: 'string' },
            service: { type: 'string' },
            timestamp: { type: 'string', format: 'date-time' },
            python_available: { type: 'boolean' },
            monitoring_directories: { type: 'object' },
            cache_status: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      // Test Python availability
      let pythonAvailable = false;
      try {
        await executePythonScript(['health_check']);
        pythonAvailable = true;
      } catch (error) {
        fastify.log.warn(`Python script not available: ${error.message}`);
      }

      // Test directory access
      const directoryStatus = {};
      const testDirectories = [MONITORING_DATA_DIR, MONITORING_CACHE_DIR, MONITORING_LOGS_DIR];
      
      for (const dir of testDirectories) {
        try {
          await fs.access(dir, fs.constants.W_OK);
          directoryStatus[path.basename(dir)] = 'accessible';
        } catch (error) {
          directoryStatus[path.basename(dir)] = 'inaccessible';
        }
      }

      // Test cache functionality
      const cacheStatus = {};
      try {
        const testKey = 'health_check_test';
        const testData = { test: true, timestamp: Date.now() };
        await setCachedResult(testKey, testData);
        const retrieved = await getCachedResult(testKey, 1);
        cacheStatus.write = 'ok';
        cacheStatus.read = retrieved ? 'ok' : 'failed';
      } catch (error) {
        cacheStatus.error = error.message;
      }

      return reply.send({
        status: 'healthy',
        service: 'autonomous-workflow-monitor',
        timestamp: new Date().toISOString(),
        python_available: pythonAvailable,
        monitoring_directories: directoryStatus,
        cache_status: cacheStatus
      });

    } catch (error) {
      fastify.log.error(`Health check error: ${error.message}`);
      return reply.status(500).send({
        status: 'unhealthy',
        service: 'autonomous-workflow-monitor',
        timestamp: new Date().toISOString(),
        error: error.message
      });
    }
  });

  // Add request/response logging
  fastify.addHook('onRequest', async (request, reply) => {
    fastify.log.info(`Autonomous Workflow Monitor API: ${request.method} ${request.url}`);
  });

  fastify.addHook('onResponse', async (request, reply) => {
    fastify.log.info(`Autonomous Workflow Monitor API Response: ${request.method} ${request.url} - ${reply.statusCode}`);
  });
}

module.exports = autonomousWorkflowMonitorRoutes;
