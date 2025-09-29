/**
 * Agent Performance Metrics API Routes - Fastify Implementation
 * Created: 2024-01-15
 * Author: MortgageAI Development Team
 * 
 * Comprehensive API routes for Agent Performance Metrics Dashboard with detailed analytics,
 * success rates tracking, optimization recommendations, and real-time monitoring.
 * 
 * Features:
 * - Agent performance metrics collection and analysis
 * - Real-time performance monitoring and alerting
 * - Comprehensive dashboard generation with visualizations
 * - Optimization recommendations and improvement suggestions
 * - Success rates tracking and trend analysis
 * - Comparative performance analysis across agents
 * - AFM compliance performance monitoring
 * - Batch processing and scheduled analysis
 * - Health monitoring and system diagnostics
 * - Export and reporting capabilities
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');

// Rate limiting configuration
const rateLimitConfig = {
  max: parseInt(process.env.PERFORMANCE_RATE_LIMIT_MAX) || 100,
  timeWindow: parseInt(process.env.PERFORMANCE_RATE_LIMIT_WINDOW) || 60000
};

// Agent Performance Metrics Routes
async function agentPerformanceMetricsRoutes(fastify, options) {
  // Apply rate limiting to all routes
  await fastify.register(require('@fastify/rate-limit'), rateLimitConfig);
  
  // Validation schemas
  const metricsCollectionSchema = {
    body: {
      type: 'object',
      properties: {
        agent_id: { type: 'string', minLength: 1 },
        time_period: { 
          type: 'string', 
          enum: ['hourly', 'daily', 'weekly', 'monthly'],
          default: 'daily' 
        },
        include_context: { type: 'boolean', default: true },
        metrics_types: {
          type: 'array',
          items: { 
            type: 'string',
            enum: ['basic', 'quality', 'efficiency', 'compliance', 'interaction']
          }
        }
      },
      required: ['agent_id']
    }
  };

  const performanceAnalysisSchema = {
    body: {
      type: 'object',
      properties: {
        agent_id: { type: 'string', minLength: 1 },
        analysis_period: { 
          type: 'string', 
          enum: ['daily', 'weekly', 'monthly', 'quarterly'],
          default: 'monthly' 
        },
        include_forecasting: { type: 'boolean', default: true },
        include_recommendations: { type: 'boolean', default: true },
        focus_areas: {
          type: 'array',
          items: { 
            type: 'string',
            enum: ['performance', 'quality', 'efficiency', 'compliance', 'satisfaction']
          }
        }
      },
      required: ['agent_id']
    }
  };

  const dashboardGenerationSchema = {
    querystring: {
      type: 'object',
      properties: {
        agent_ids: { type: 'string' }, // Comma-separated list
        time_range: { 
          type: 'string', 
          enum: ['last_7_days', 'last_30_days', 'last_90_days', 'last_year'],
          default: 'last_30_days' 
        },
        dashboard_type: { 
          type: 'string', 
          enum: ['comprehensive', 'summary', 'comparative', 'executive'],
          default: 'comprehensive' 
        },
        include_forecasts: { type: 'boolean', default: false },
        include_benchmarks: { type: 'boolean', default: true }
      }
    }
  };

  const optimizationRecommendationsSchema = {
    body: {
      type: 'object',
      properties: {
        agent_id: { type: 'string', minLength: 1 },
        focus_areas: {
          type: 'array',
          items: { 
            type: 'string',
            enum: ['workflow', 'resource', 'training', 'automation', 'quality', 'efficiency', 'compliance']
          }
        },
        priority_filter: { 
          type: 'string', 
          enum: ['low', 'normal', 'high', 'urgent', 'critical']
        },
        include_implementation_plan: { type: 'boolean', default: true },
        include_roi_analysis: { type: 'boolean', default: true }
      },
      required: ['agent_id']
    }
  };

  /**
   * Helper function to execute Python script
   */
  async function executePythonScript(scriptName, args = []) {
    return new Promise((resolve, reject) => {
      const pythonPath = process.env.PYTHON_PATH || 'python3';
      const scriptPath = path.join(__dirname, '..', 'agents', 'utils', scriptName);
      
      const startTime = Date.now();
      const python = spawn(pythonPath, [scriptPath, ...args]);
      
      let stdout = '';
      let stderr = '';
      
      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      python.on('close', (code) => {
        const executionTime = (Date.now() - startTime) / 1000;
        
        if (code !== 0) {
          reject(new Error(`Python script failed with code ${code}: ${stderr}`));
        } else {
          try {
            const result = JSON.parse(stdout);
            result.execution_time = executionTime;
            resolve(result);
          } catch (parseError) {
            reject(new Error(`Failed to parse Python output: ${parseError.message}`));
          }
        }
      });
      
      python.on('error', (error) => {
        reject(new Error(`Failed to start Python process: ${error.message}`));
      });
    });
  }

  // Performance Metrics Configuration
  // METRICS COLLECTION ENDPOINTS
  // Performance Metrics Configuration

  /**
   * Collect performance metrics for a specific agent
   */
  fastify.post('/metrics/collect', {
    schema: metricsCollectionSchema,
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const { 
        agent_id, 
        time_period, 
        include_context, 
        metrics_types 
      } = request.body;
      
      // Validate agent existence
      const agentExists = await fastify.db.query(
        'SELECT agent_id FROM agents WHERE agent_id = $1',
        [agent_id]
      );
      
      if (agentExists.rows.length === 0) {
        return reply.code(404).send({
          success: false,
          error: 'Agent not found',
          error_code: 'AGENT_NOT_FOUND',
          agent_id
        });
      }

      // Execute metrics collection
      const scriptArgs = [
        '--action', 'collect_metrics',
        '--agent_id', agent_id,
        '--time_period', time_period,
        '--include_context', include_context ? 'true' : 'false',
        '--user_id', request.user?.id || 'system'
      ];

      if (metrics_types && metrics_types.length > 0) {
        scriptArgs.push('--metrics_types', metrics_types.join(','));
      }

      const result = await executePythonScript('agent_performance_metrics.py', scriptArgs);
      
      // Log metrics collection event
      await fastify.db.query(`
        INSERT INTO agent_performance_logs 
        (log_id, agent_id, user_id, action, parameters, result_summary, processing_time, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
      `, [
        uuidv4(),
        agent_id,
        request.user?.id || null,
        'metrics_collection',
        JSON.stringify({ time_period, include_context, metrics_types }),
        JSON.stringify({ 
          success: result.success, 
          metrics_count: Object.keys(result.metrics || {}).length,
          overall_score: result.metrics?.overall_performance_score 
        }),
        (Date.now() - startTime) / 1000,
        new Date()
      ]);

      reply.send({
        success: true,
        message: 'Performance metrics collected successfully',
        data: result,
        metadata: {
          agent_id,
          time_period,
          collection_timestamp: new Date().toISOString(),
          processing_time: (Date.now() - startTime) / 1000
        }
      });

    } catch (error) {
      request.log.error('Metrics collection failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to collect performance metrics',
        error_code: 'METRICS_COLLECTION_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  /**
   * Collect metrics for multiple agents (batch operation)
   */
  fastify.post('/metrics/collect-batch', {
    schema: {
      body: {
        type: 'object',
        properties: {
          agent_ids: {
            type: 'array',
            items: { type: 'string', minLength: 1 },
            minItems: 1,
            maxItems: 50
          },
          time_period: { 
            type: 'string', 
            enum: ['hourly', 'daily', 'weekly', 'monthly'],
            default: 'daily' 
          },
          include_context: { type: 'boolean', default: true },
          parallel_processing: { type: 'boolean', default: true }
        },
        required: ['agent_ids']
      }
    },
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const { agent_ids, time_period, include_context, parallel_processing } = request.body;
      
      // Execute batch metrics collection
      const scriptArgs = [
        '--action', 'collect_batch_metrics',
        '--agent_ids', agent_ids.join(','),
        '--time_period', time_period,
        '--include_context', include_context ? 'true' : 'false',
        '--parallel', parallel_processing ? 'true' : 'false',
        '--user_id', request.user?.id || 'system'
      ];

      const result = await executePythonScript('agent_performance_metrics.py', scriptArgs);
      
      // Log batch operation
      await fastify.db.query(`
        INSERT INTO agent_performance_logs 
        (log_id, user_id, action, parameters, result_summary, processing_time, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
      `, [
        uuidv4(),
        request.user?.id || null,
        'batch_metrics_collection',
        JSON.stringify({ agent_count: agent_ids.length, time_period, parallel_processing }),
        JSON.stringify({ 
          success: result.success, 
          successful_collections: result.successful_collections,
          failed_collections: result.failed_collections 
        }),
        (Date.now() - startTime) / 1000,
        new Date()
      ]);

      reply.send({
        success: true,
        message: 'Batch metrics collection completed',
        data: result,
        metadata: {
          agent_count: agent_ids.length,
          time_period,
          processing_time: (Date.now() - startTime) / 1000
        }
      });

    } catch (error) {
      request.log.error('Batch metrics collection failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to collect batch performance metrics',
        error_code: 'BATCH_METRICS_COLLECTION_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  // Performance Metrics Configuration
  // PERFORMANCE ANALYSIS ENDPOINTS
  // Performance Metrics Configuration

  /**
   * Analyze agent performance with detailed insights and recommendations
   */
  fastify.post('/analysis/performance', {
    schema: performanceAnalysisSchema,
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const { 
        agent_id, 
        analysis_period, 
        include_forecasting, 
        include_recommendations,
        focus_areas 
      } = request.body;
      
      // Execute performance analysis
      const scriptArgs = [
        '--action', 'analyze_performance',
        '--agent_id', agent_id,
        '--analysis_period', analysis_period,
        '--include_forecasting', include_forecasting ? 'true' : 'false',
        '--include_recommendations', include_recommendations ? 'true' : 'false',
        '--user_id', request.user?.id || 'system'
      ];

      if (focus_areas && focus_areas.length > 0) {
        scriptArgs.push('--focus_areas', focus_areas.join(','));
      }

      const result = await executePythonScript('agent_performance_metrics.py', scriptArgs);
      
      // Store analysis results
      if (result.success && result.analysis) {
        await fastify.db.query(`
          INSERT INTO agent_performance_analyses 
          (analysis_id, agent_id, user_id, analysis_type, analysis_period, analysis_data, 
           overall_grade, key_insights, recommendations_count, created_at)
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        `, [
          result.analysis.analysis_id,
          agent_id,
          request.user?.id || null,
          'comprehensive',
          analysis_period,
          JSON.stringify(result.analysis),
          result.analysis.overall_grade || 'N/A',
          JSON.stringify(result.analysis.key_insights || []),
          result.analysis.recommendations?.length || 0,
          new Date()
        ]);
      }

      reply.send({
        success: true,
        message: 'Performance analysis completed successfully',
        data: result,
        metadata: {
          agent_id,
          analysis_period,
          analysis_timestamp: new Date().toISOString(),
          processing_time: (Date.now() - startTime) / 1000
        }
      });

    } catch (error) {
      request.log.error('Performance analysis failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to analyze agent performance',
        error_code: 'PERFORMANCE_ANALYSIS_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  /**
   * Get comparative performance analysis across multiple agents
   */
  fastify.get('/analysis/comparative', {
    schema: {
      querystring: {
        type: 'object',
        properties: {
          agent_ids: { type: 'string' }, // Comma-separated
          time_period: { 
            type: 'string', 
            enum: ['weekly', 'monthly', 'quarterly'],
            default: 'monthly' 
          },
          metrics: { type: 'string' }, // Comma-separated metrics to compare
          include_rankings: { type: 'boolean', default: true },
          include_benchmarks: { type: 'boolean', default: true }
        }
      }
    },
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const { agent_ids, time_period, metrics, include_rankings, include_benchmarks } = request.query;
      
      // Parse agent IDs
      const agentIdList = agent_ids ? agent_ids.split(',').map(id => id.trim()) : [];
      
      if (agentIdList.length === 0) {
        // Get all active agents if none specified
        const activeAgents = await fastify.db.query(`
          SELECT DISTINCT agent_id 
          FROM agent_performance_metrics 
          WHERE created_at >= NOW() - INTERVAL '30 days'
          ORDER BY agent_id
        `);
        agentIdList.push(...activeAgents.rows.map(row => row.agent_id));
      }

      // Execute comparative analysis
      const scriptArgs = [
        '--action', 'comparative_analysis',
        '--agent_ids', agentIdList.join(','),
        '--time_period', time_period,
        '--include_rankings', include_rankings ? 'true' : 'false',
        '--include_benchmarks', include_benchmarks ? 'true' : 'false',
        '--user_id', request.user?.id || 'system'
      ];

      if (metrics) {
        scriptArgs.push('--metrics', metrics);
      }

      const result = await executePythonScript('agent_performance_metrics.py', scriptArgs);

      reply.send({
        success: true,
        message: 'Comparative analysis completed successfully',
        data: result,
        metadata: {
          agent_count: agentIdList.length,
          time_period,
          comparison_metrics: metrics ? metrics.split(',') : 'all',
          processing_time: (Date.now() - startTime) / 1000
        }
      });

    } catch (error) {
      request.log.error('Comparative analysis failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to generate comparative analysis',
        error_code: 'COMPARATIVE_ANALYSIS_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  // Performance Metrics Configuration
  // DASHBOARD GENERATION ENDPOINTS
  // Performance Metrics Configuration

  /**
   * Generate comprehensive performance dashboard
   */
  fastify.get('/dashboard/generate', {
    schema: dashboardGenerationSchema,
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const { 
        agent_ids, 
        time_range, 
        dashboard_type, 
        include_forecasts, 
        include_benchmarks 
      } = request.query;
      
      // Parse agent IDs
      const agentIdList = agent_ids ? agent_ids.split(',').map(id => id.trim()) : null;

      // Execute dashboard generation
      const scriptArgs = [
        '--action', 'generate_dashboard',
        '--time_range', time_range,
        '--dashboard_type', dashboard_type,
        '--include_forecasts', include_forecasts ? 'true' : 'false',
        '--include_benchmarks', include_benchmarks ? 'true' : 'false',
        '--user_id', request.user?.id || 'system'
      ];

      if (agentIdList) {
        scriptArgs.push('--agent_ids', agentIdList.join(','));
      }

      const result = await executePythonScript('agent_performance_metrics.py', scriptArgs);
      
      // Cache dashboard data
      const cacheKey = `dashboard_${dashboard_type}_${time_range}_${agentIdList ? agentIdList.join('_') : 'all'}`;
      const cacheExpiry = 15 * 60; // 15 minutes
      
      if (result.success && result.dashboard) {
        // Store in Redis cache
        await fastify.redis.setex(
          cacheKey, 
          cacheExpiry, 
          JSON.stringify(result.dashboard)
        );
        
        // Log dashboard generation
        await fastify.db.query(`
          INSERT INTO agent_dashboard_generations 
          (generation_id, user_id, dashboard_type, time_range, agent_count, 
           generation_time, cached_until, created_at)
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        `, [
          result.dashboard.dashboard_id,
          request.user?.id || null,
          dashboard_type,
          time_range,
          result.dashboard.agent_count,
          (Date.now() - startTime) / 1000,
          new Date(Date.now() + cacheExpiry * 1000),
          new Date()
        ]);
      }

      reply.send({
        success: true,
        message: 'Dashboard generated successfully',
        data: result,
        metadata: {
          dashboard_type,
          time_range,
          agent_count: result.dashboard?.agent_count || 0,
          cached_until: new Date(Date.now() + cacheExpiry * 1000).toISOString(),
          processing_time: (Date.now() - startTime) / 1000
        }
      });

    } catch (error) {
      request.log.error('Dashboard generation failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to generate performance dashboard',
        error_code: 'DASHBOARD_GENERATION_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  /**
   * Get real-time performance metrics for live dashboard
   */
  fastify.get('/dashboard/realtime', {
    schema: {
      querystring: {
        type: 'object',
        properties: {
          agent_ids: { type: 'string' },
          metrics: { type: 'string' }, // Comma-separated metrics
          refresh_interval: { type: 'number', minimum: 5, maximum: 300, default: 30 }
        }
      }
    },
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const { agent_ids, metrics, refresh_interval } = request.query;
      
      // Parse agent IDs
      const agentIdList = agent_ids ? agent_ids.split(',').map(id => id.trim()) : null;
      
      // Execute real-time metrics collection
      const scriptArgs = [
        '--action', 'realtime_metrics',
        '--refresh_interval', refresh_interval.toString(),
        '--user_id', request.user?.id || 'system'
      ];

      if (agentIdList) {
        scriptArgs.push('--agent_ids', agentIdList.join(','));
      }

      if (metrics) {
        scriptArgs.push('--metrics', metrics);
      }

      const result = await executePythonScript('agent_performance_metrics.py', scriptArgs);

      reply.send({
        success: true,
        message: 'Real-time metrics retrieved successfully',
        data: result,
        metadata: {
          refresh_interval,
          next_refresh: new Date(Date.now() + refresh_interval * 1000).toISOString(),
          metrics_included: metrics ? metrics.split(',') : 'all'
        }
      });

    } catch (error) {
      request.log.error('Real-time metrics failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve real-time metrics',
        error_code: 'REALTIME_METRICS_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  // Performance Metrics Configuration
  // OPTIMIZATION RECOMMENDATIONS ENDPOINTS
  // Performance Metrics Configuration

  /**
   * Get optimization recommendations for an agent
   */
  fastify.post('/recommendations/optimize', {
    schema: optimizationRecommendationsSchema,
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const { 
        agent_id, 
        focus_areas, 
        priority_filter, 
        include_implementation_plan,
        include_roi_analysis 
      } = request.body;
      
      // Execute optimization recommendations generation
      const scriptArgs = [
        '--action', 'generate_recommendations',
        '--agent_id', agent_id,
        '--include_implementation', include_implementation_plan ? 'true' : 'false',
        '--include_roi', include_roi_analysis ? 'true' : 'false',
        '--user_id', request.user?.id || 'system'
      ];

      if (focus_areas && focus_areas.length > 0) {
        scriptArgs.push('--focus_areas', focus_areas.join(','));
      }

      if (priority_filter) {
        scriptArgs.push('--priority_filter', priority_filter);
      }

      const result = await executePythonScript('agent_performance_metrics.py', scriptArgs);
      
      // Store recommendations
      if (result.success && result.recommendations) {
        for (const recommendation of result.recommendations) {
          await fastify.db.query(`
            INSERT INTO agent_optimization_recommendations 
            (recommendation_id, agent_id, user_id, recommendation_type, priority, 
             title, description, expected_impact, implementation_effort, 
             estimated_roi, confidence_level, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
          `, [
            recommendation.recommendation_id || uuidv4(),
            agent_id,
            request.user?.id || null,
            recommendation.recommendation_type,
            recommendation.priority,
            recommendation.title,
            recommendation.description,
            recommendation.expected_impact,
            recommendation.implementation_effort,
            recommendation.estimated_roi || 0,
            recommendation.confidence_level || 0,
            new Date()
          ]);
        }
      }

      reply.send({
        success: true,
        message: 'Optimization recommendations generated successfully',
        data: result,
        metadata: {
          agent_id,
          focus_areas: focus_areas || ['all'],
          priority_filter: priority_filter || 'all',
          recommendations_count: result.recommendations?.length || 0,
          processing_time: (Date.now() - startTime) / 1000
        }
      });

    } catch (error) {
      request.log.error('Optimization recommendations failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to generate optimization recommendations',
        error_code: 'OPTIMIZATION_RECOMMENDATIONS_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  /**
   * Get recommendation implementation status
   */
  fastify.get('/recommendations/:recommendationId/status', {
    schema: {
      params: {
        type: 'object',
        properties: {
          recommendationId: { type: 'string', minLength: 1 }
        },
        required: ['recommendationId']
      }
    },
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const { recommendationId } = request.params;
      
      // Get recommendation details and implementation status
      const recommendation = await fastify.db.query(`
        SELECT r.*, ri.implementation_status, ri.progress_percentage, 
               ri.implementation_notes, ri.results_achieved, ri.updated_at as status_updated_at
        FROM agent_optimization_recommendations r
        LEFT JOIN recommendation_implementations ri ON r.recommendation_id = ri.recommendation_id
        WHERE r.recommendation_id = $1
      `, [recommendationId]);

      if (recommendation.rows.length === 0) {
        return reply.code(404).send({
          success: false,
          error: 'Recommendation not found',
          error_code: 'RECOMMENDATION_NOT_FOUND'
        });
      }

      const recommendationData = recommendation.rows[0];

      reply.send({
        success: true,
        data: {
          recommendation_id: recommendationData.recommendation_id,
          agent_id: recommendationData.agent_id,
          title: recommendationData.title,
          description: recommendationData.description,
          priority: recommendationData.priority,
          expected_impact: recommendationData.expected_impact,
          estimated_roi: recommendationData.estimated_roi,
          implementation_status: recommendationData.implementation_status || 'not_started',
          progress_percentage: recommendationData.progress_percentage || 0,
          implementation_notes: recommendationData.implementation_notes,
          results_achieved: recommendationData.results_achieved,
          created_at: recommendationData.created_at,
          status_updated_at: recommendationData.status_updated_at
        }
      });

    } catch (error) {
      request.log.error('Recommendation status retrieval failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve recommendation status',
        error_code: 'RECOMMENDATION_STATUS_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  // Performance Metrics Configuration
  // MONITORING AND ALERTING ENDPOINTS
  // Performance Metrics Configuration

  /**
   * Get performance alerts and notifications
   */
  fastify.get('/alerts/performance', {
    schema: {
      querystring: {
        type: 'object',
        properties: {
          agent_id: { type: 'string' },
          severity: { 
            type: 'string', 
            enum: ['low', 'medium', 'high', 'critical']
          },
          status: { 
            type: 'string', 
            enum: ['active', 'acknowledged', 'resolved']
          },
          time_range: { 
            type: 'string', 
            enum: ['last_hour', 'last_day', 'last_week'],
            default: 'last_day' 
          },
          limit: { type: 'number', minimum: 1, maximum: 100, default: 50 }
        }
      }
    },
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const { agent_id, severity, status, time_range, limit } = request.query;
      
      // Build query conditions
      let whereConditions = ['1=1'];
      let queryParams = [];
      let paramIndex = 1;

      if (agent_id) {
        whereConditions.push(`agent_id = $${paramIndex}`);
        queryParams.push(agent_id);
        paramIndex++;
      }

      if (severity) {
        whereConditions.push(`severity = $${paramIndex}`);
        queryParams.push(severity);
        paramIndex++;
      }

      if (status) {
        whereConditions.push(`alert_status = $${paramIndex}`);
        queryParams.push(status);
        paramIndex++;
      }

      // Time range condition
      let timeCondition = '';
      switch (time_range) {
        case 'last_hour':
          timeCondition = `created_at >= NOW() - INTERVAL '1 hour'`;
          break;
        case 'last_day':
          timeCondition = `created_at >= NOW() - INTERVAL '1 day'`;
          break;
        case 'last_week':
          timeCondition = `created_at >= NOW() - INTERVAL '1 week'`;
          break;
      }
      
      if (timeCondition) {
        whereConditions.push(timeCondition);
      }

      const alertsQuery = `
        SELECT alert_id, agent_id, alert_type, severity, alert_status, 
               title, description, metric_values, threshold_values,
               acknowledged_by, acknowledged_at, resolved_at, created_at
        FROM agent_performance_alerts
        WHERE ${whereConditions.join(' AND ')}
        ORDER BY created_at DESC, severity DESC
        LIMIT $${paramIndex}
      `;
      
      queryParams.push(limit);

      const alerts = await fastify.db.query(alertsQuery, queryParams);

      // Get alert statistics
      const statsQuery = `
        SELECT 
          COUNT(*) as total_alerts,
          COUNT(CASE WHEN alert_status = 'active' THEN 1 END) as active_alerts,
          COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts,
          COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_alerts
        FROM agent_performance_alerts
        WHERE ${timeCondition || '1=1'}
      `;

      const stats = await fastify.db.query(statsQuery);

      reply.send({
        success: true,
        data: {
          alerts: alerts.rows,
          statistics: stats.rows[0],
          filters: {
            agent_id,
            severity,
            status,
            time_range,
            limit
          }
        },
        metadata: {
          total_count: alerts.rows.length,
          query_time_range: time_range,
          retrieved_at: new Date().toISOString()
        }
      });

    } catch (error) {
      request.log.error('Performance alerts retrieval failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve performance alerts',
        error_code: 'PERFORMANCE_ALERTS_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  /**
   * System health check for performance monitoring system
   */
  fastify.get('/health', async (request, reply) => {
    try {
      const healthStartTime = Date.now();
      
      // Execute health check
      const result = await executePythonScript('agent_performance_metrics.py', [
        '--action', 'health_check'
      ]);
      
      // Additional system checks
      const systemHealth = {
        api_status: 'healthy',
        database_status: 'healthy',
        cache_status: 'healthy',
        python_integration: result.success ? 'healthy' : 'unhealthy',
        response_time: (Date.now() - healthStartTime) / 1000
      };

      // Test database connection
      try {
        await fastify.db.query('SELECT 1');
      } catch (dbError) {
        systemHealth.database_status = 'unhealthy';
        systemHealth.database_error = dbError.message;
      }

      // Test Redis connection
      try {
        await fastify.redis.ping();
      } catch (redisError) {
        systemHealth.cache_status = 'unhealthy';
        systemHealth.cache_error = redisError.message;
      }

      const overallStatus = Object.values(systemHealth)
        .filter(status => typeof status === 'string')
        .every(status => status === 'healthy') ? 'healthy' : 'degraded';

      reply.send({
        success: true,
        status: overallStatus,
        system: systemHealth,
        performance_engine: result.success ? result.health : { status: 'unhealthy' },
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      reply.code(500).send({
        success: false,
        status: 'unhealthy',
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  /**
   * Get performance monitoring statistics
   */
  fastify.get('/statistics', {
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      // Execute statistics collection
      const result = await executePythonScript('agent_performance_metrics.py', [
        '--action', 'get_statistics',
        '--user_id', request.user?.id || 'system'
      ]);

      reply.send({
        success: true,
        message: 'Performance statistics retrieved successfully',
        data: result,
        metadata: {
          retrieved_at: new Date().toISOString()
        }
      });

    } catch (error) {
      request.log.error('Statistics retrieval failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve performance statistics',
        error_code: 'STATISTICS_RETRIEVAL_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });

  // Performance Metrics Configuration
  // EXPORT AND REPORTING ENDPOINTS
  // Performance Metrics Configuration

  /**
   * Export performance data and reports
   */
  fastify.post('/export', {
    schema: {
      body: {
        type: 'object',
        properties: {
          export_type: { 
            type: 'string', 
            enum: ['metrics', 'analysis', 'dashboard', 'recommendations'],
            default: 'metrics' 
          },
          agent_ids: {
            type: 'array',
            items: { type: 'string' }
          },
          time_range: { 
            type: 'string', 
            enum: ['last_week', 'last_month', 'last_quarter', 'last_year'],
            default: 'last_month' 
          },
          format: { 
            type: 'string', 
            enum: ['json', 'csv', 'xlsx', 'pdf'],
            default: 'json' 
          },
          include_charts: { type: 'boolean', default: false }
        },
        required: ['export_type']
      }
    },
    preHandler: fastify.auth([fastify.verifyToken], { relation: 'and' })
  }, async (request, reply) => {
    try {
      const { export_type, agent_ids, time_range, format, include_charts } = request.body;
      
      // Execute export operation
      const scriptArgs = [
        '--action', 'export_data',
        '--export_type', export_type,
        '--time_range', time_range,
        '--format', format,
        '--include_charts', include_charts ? 'true' : 'false',
        '--user_id', request.user?.id || 'system'
      ];

      if (agent_ids && agent_ids.length > 0) {
        scriptArgs.push('--agent_ids', agent_ids.join(','));
      }

      const result = await executePythonScript('agent_performance_metrics.py', scriptArgs);
      
      if (result.success && result.export_path) {
        // Set appropriate headers for file download
        const filename = path.basename(result.export_path);
        const contentType = format === 'pdf' ? 'application/pdf' : 
                          format === 'xlsx' ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' :
                          format === 'csv' ? 'text/csv' : 'application/json';
        
        reply.header('Content-Type', contentType);
        reply.header('Content-Disposition', `attachment; filename="${filename}"`);
        
        // Read and send file
        const fileContent = await fs.readFile(result.export_path);
        reply.send(fileContent);
      } else {
        reply.send({
          success: true,
          message: 'Export completed successfully',
          data: result
        });
      }

    } catch (error) {
      request.log.error('Export failed:', error);
      
      reply.code(500).send({
        success: false,
        error: 'Failed to export performance data',
        error_code: 'EXPORT_FAILED',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  });
}

module.exports = agentPerformanceMetricsRoutes;
