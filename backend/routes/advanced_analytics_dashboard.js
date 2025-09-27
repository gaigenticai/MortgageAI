/**
 * Advanced Analytics Dashboard API Routes
 * 
 * Provides comprehensive API endpoints for Dutch mortgage market analytics,
 * predictive modeling, and advanced reporting capabilities.
 * 
 * Created: 2024-01-15
 * Author: MortgageAI Development Team
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const crypto = require('crypto');

/**
 * Advanced Analytics Dashboard Routes Plugin
 * @param {Object} fastify - Fastify instance
 * @param {Object} options - Plugin options
 */
async function advancedAnalyticsDashboardRoutes(fastify, options) {
  
  // Helper function to execute Python analytics scripts
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
      
      // Set timeout for long-running analyses
      setTimeout(() => {
        pythonProcess.kill();
        reject(new Error('Analytics operation timed out'));
      }, parseInt(process.env.ANALYTICS_TIMEOUT || '300000')); // 5 minutes default
    });
  };
  
  // Helper function to validate analysis parameters
  const validateAnalysisParams = (params) => {
    const allowedTypes = ['market_analysis', 'risk_assessment', 'performance', 'compliance', 'custom'];
    const allowedPeriods = ['1m', '3m', '6m', '12m', '24m'];
    
    if (params.analysis_type && !allowedTypes.includes(params.analysis_type)) {
      throw new Error(`Invalid analysis_type. Must be one of: ${allowedTypes.join(', ')}`);
    }
    
    if (params.time_period && !allowedPeriods.includes(params.time_period)) {
      throw new Error(`Invalid time_period. Must be one of: ${allowedPeriods.join(', ')}`);
    }
    
    return true;
  };

  // Schema definitions for request/response validation
  const analysisRequestSchema = {
    type: 'object',
    properties: {
      analysis_type: { 
        type: 'string', 
        enum: ['market_analysis', 'risk_assessment', 'performance', 'compliance', 'custom'],
        default: 'market_analysis'
      },
      time_period: { 
        type: 'string', 
        enum: ['1m', '3m', '6m', '12m', '24m'],
        default: '12m'
      },
      include_forecasts: { type: 'boolean', default: true },
      include_visualizations: { type: 'boolean', default: true },
      custom_metrics: { 
        type: 'array', 
        items: { type: 'string' },
        default: []
      },
      filters: {
        type: 'object',
        properties: {
          regions: { type: 'array', items: { type: 'string' } },
          lender_types: { type: 'array', items: { type: 'string' } },
          product_types: { type: 'array', items: { type: 'string' } }
        },
        additionalProperties: true,
        default: {}
      }
    },
    additionalProperties: false
  };
  
  const forecastRequestSchema = {
    type: 'object',
    required: ['target_variable', 'input_data'],
    properties: {
      target_variable: { type: 'string' },
      model_type: { 
        type: 'string', 
        enum: ['linear', 'random_forest', 'gradient_boosting', 'ensemble'],
        default: 'ensemble'
      },
      input_data: { type: 'object' },
      forecast_periods: { type: 'integer', minimum: 1, maximum: 24, default: 6 },
      confidence_interval: { type: 'number', minimum: 0.8, maximum: 0.99, default: 0.95 }
    },
    additionalProperties: false
  };

  // Route: Get comprehensive market analysis
  fastify.get('/analysis/comprehensive', {
    schema: {
      description: 'Generate comprehensive Dutch mortgage market analysis',
      tags: ['Analytics'],
      querystring: {
        type: 'object',
        properties: {
          analysis_type: { type: 'string', default: 'market_analysis' },
          time_period: { type: 'string', default: '12m' },
          include_forecasts: { type: 'boolean', default: true },
          include_visualizations: { type: 'boolean', default: true }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            report_id: { type: 'string' },
            analysis_type: { type: 'string' },
            time_period: { type: 'string' },
            insights_count: { type: 'integer' },
            metrics_count: { type: 'integer' },
            visualizations_count: { type: 'integer' },
            executive_summary: { type: 'string' },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      const { analysis_type = 'market_analysis', time_period = '12m', include_forecasts = true, include_visualizations = true } = request.query;
      
      // Validate parameters
      validateAnalysisParams({ analysis_type, time_period });
      
      fastify.log.info(`Generating comprehensive analysis: type=${analysis_type}, period=${time_period}`);
      
      // Execute the analytics dashboard analysis
      const analysisArgs = [
        'generate_comprehensive_analysis',
        '--analysis_type', analysis_type,
        '--time_period', time_period,
        '--include_forecasts', include_forecasts.toString(),
        '--include_visualizations', include_visualizations.toString(),
        '--output_format', 'json'
      ];
      
      const result = await executePythonScript('analytics_dashboard_executor.py', analysisArgs);
      
      const processingTime = (Date.now() - startTime) / 1000;
      
      reply.code(200).send({
        success: true,
        report_id: result.report_id,
        analysis_type: result.analysis_type,
        time_period: result.time_period,
        insights_count: result.insights?.length || 0,
        metrics_count: Object.keys(result.metrics || {}).length,
        visualizations_count: Object.keys(result.visualizations || {}).length,
        executive_summary: result.executive_summary,
        processing_time: processingTime,
        data: result
      });
      
    } catch (error) {
      fastify.log.error(`Error generating comprehensive analysis: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to generate comprehensive analysis',
        message: error.message
      });
    }
  });
  
  // Route: Get real-time market metrics
  fastify.get('/metrics/realtime', {
    schema: {
      description: 'Get real-time Dutch mortgage market metrics',
      tags: ['Analytics', 'Metrics'],
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            timestamp: { type: 'string' },
            market_status: { type: 'string' },
            key_indicators: { type: 'object' },
            alert_levels: { type: 'object' },
            performance_indicators: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const startTime = Date.now();
      
      fastify.log.info('Retrieving real-time market metrics');
      
      const result = await executePythonScript('analytics_dashboard_executor.py', [
        'get_real_time_metrics',
        '--output_format', 'json'
      ]);
      
      const processingTime = (Date.now() - startTime) / 1000;
      
      reply.code(200).send({
        success: true,
        processing_time: processingTime,
        ...result
      });
      
    } catch (error) {
      fastify.log.error(`Error retrieving real-time metrics: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve real-time metrics',
        message: error.message
      });
    }
  });
  
  // Route: Get market insights
  fastify.get('/insights', {
    schema: {
      description: 'Get market insights and analysis',
      tags: ['Analytics', 'Insights'],
      querystring: {
        type: 'object',
        properties: {
          insight_types: { 
            type: 'array',
            items: { 
              type: 'string',
              enum: ['trend_analysis', 'risk_assessment', 'opportunity_detection', 'regulatory_impact']
            },
            default: []
          },
          time_period: { type: 'string', default: '12m' },
          confidence_threshold: { type: 'number', minimum: 0.0, maximum: 1.0, default: 0.7 }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { insight_types = [], time_period = '12m', confidence_threshold = 0.7 } = request.query;
      
      fastify.log.info(`Generating market insights: types=${insight_types.join(',')}, period=${time_period}`);
      
      const result = await executePythonScript('analytics_dashboard_executor.py', [
        'generate_insights',
        '--insight_types', insight_types.join(','),
        '--time_period', time_period,
        '--confidence_threshold', confidence_threshold.toString(),
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        insights_count: result.insights?.length || 0,
        insights: result.insights || [],
        metadata: {
          time_period,
          confidence_threshold,
          generated_at: new Date().toISOString()
        }
      });
      
    } catch (error) {
      fastify.log.error(`Error generating insights: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to generate insights',
        message: error.message
      });
    }
  });
  
  // Route: Create predictive forecast
  fastify.post('/forecasts/create', {
    schema: {
      description: 'Create predictive forecasts for market variables',
      tags: ['Analytics', 'Forecasting'],
      body: forecastRequestSchema
    }
  }, async (request, reply) => {
    try {
      const { target_variable, model_type = 'ensemble', input_data, forecast_periods = 6, confidence_interval = 0.95 } = request.body;
      
      fastify.log.info(`Creating forecast for ${target_variable} using ${model_type} model`);
      
      const result = await executePythonScript('analytics_dashboard_executor.py', [
        'create_forecast',
        '--target_variable', target_variable,
        '--model_type', model_type,
        '--input_data', JSON.stringify(input_data),
        '--forecast_periods', forecast_periods.toString(),
        '--confidence_interval', confidence_interval.toString(),
        '--output_format', 'json'
      ]);
      
      reply.code(201).send({
        success: true,
        model_id: result.model_id,
        target_variable,
        model_type,
        forecast_periods,
        accuracy_metrics: result.accuracy_metrics,
        forecasts: result.forecasts,
        created_at: new Date().toISOString()
      });
      
    } catch (error) {
      fastify.log.error(`Error creating forecast: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to create forecast',
        message: error.message
      });
    }
  });
  
  // Route: Get historical trends
  fastify.get('/trends/:metric', {
    schema: {
      description: 'Get historical trends for a specific metric',
      tags: ['Analytics', 'Trends'],
      params: {
        type: 'object',
        required: ['metric'],
        properties: {
          metric: { type: 'string' }
        }
      },
      querystring: {
        type: 'object',
        properties: {
          periods: { type: 'integer', minimum: 6, maximum: 60, default: 24 },
          aggregation: { 
            type: 'string', 
            enum: ['daily', 'weekly', 'monthly', 'quarterly'],
            default: 'monthly'
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { metric } = request.params;
      const { periods = 24, aggregation = 'monthly' } = request.query;
      
      fastify.log.info(`Retrieving historical trends for ${metric}: ${periods} ${aggregation} periods`);
      
      const result = await executePythonScript('analytics_dashboard_executor.py', [
        'get_historical_trends',
        '--metric', metric,
        '--periods', periods.toString(),
        '--aggregation', aggregation,
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        metric,
        periods,
        aggregation,
        trend_data: result.trend_data,
        trend_analysis: result.trend_analysis
      });
      
    } catch (error) {
      fastify.log.error(`Error retrieving trends for ${metric}: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to retrieve trends',
        message: error.message
      });
    }
  });
  
  // Route: Generate custom report
  fastify.post('/reports/generate', {
    schema: {
      description: 'Generate custom analytics report',
      tags: ['Analytics', 'Reports'],
      body: {
        type: 'object',
        required: ['report_name', 'report_type'],
        properties: {
          report_name: { type: 'string', minLength: 1 },
          report_type: { 
            type: 'string',
            enum: ['market_analysis', 'risk_assessment', 'performance', 'compliance', 'custom']
          },
          time_period: { type: 'string', default: '12m' },
          sections: {
            type: 'array',
            items: { type: 'string' },
            default: ['overview', 'metrics', 'insights', 'forecasts', 'recommendations']
          },
          export_format: {
            type: 'string',
            enum: ['pdf', 'html', 'json', 'excel'],
            default: 'json'
          },
          custom_parameters: { type: 'object', default: {} }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { 
        report_name, 
        report_type, 
        time_period = '12m', 
        sections = ['overview', 'metrics', 'insights', 'forecasts', 'recommendations'],
        export_format = 'json',
        custom_parameters = {}
      } = request.body;
      
      const reportId = crypto.randomUUID();
      
      fastify.log.info(`Generating custom report: ${report_name} (${report_type})`);
      
      const result = await executePythonScript('analytics_dashboard_executor.py', [
        'generate_custom_report',
        '--report_id', reportId,
        '--report_name', report_name,
        '--report_type', report_type,
        '--time_period', time_period,
        '--sections', sections.join(','),
        '--export_format', export_format,
        '--custom_parameters', JSON.stringify(custom_parameters),
        '--output_format', 'json'
      ]);
      
      reply.code(201).send({
        success: true,
        report_id: reportId,
        report_name,
        report_type,
        export_format,
        generated_at: new Date().toISOString(),
        download_url: `/api/analytics/reports/${reportId}/download`,
        report_data: result.report_data
      });
      
    } catch (error) {
      fastify.log.error(`Error generating custom report: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to generate custom report',
        message: error.message
      });
    }
  });
  
  // Route: Get dashboard visualizations
  fastify.get('/visualizations/dashboard', {
    schema: {
      description: 'Get dashboard visualizations and charts',
      tags: ['Analytics', 'Visualizations'],
      querystring: {
        type: 'object',
        properties: {
          chart_types: {
            type: 'array',
            items: { 
              type: 'string',
              enum: ['gauge', 'line', 'bar', 'pie', 'heatmap', 'scatter']
            },
            default: ['gauge', 'line', 'bar', 'pie']
          },
          time_period: { type: 'string', default: '12m' },
          refresh_cache: { type: 'boolean', default: false }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { chart_types = ['gauge', 'line', 'bar', 'pie'], time_period = '12m', refresh_cache = false } = request.query;
      
      fastify.log.info(`Generating dashboard visualizations: ${chart_types.join(',')}`);
      
      const result = await executePythonScript('analytics_dashboard_executor.py', [
        'create_dashboard_visualizations',
        '--chart_types', chart_types.join(','),
        '--time_period', time_period,
        '--refresh_cache', refresh_cache.toString(),
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        visualizations: result.visualizations,
        metadata: {
          chart_types,
          time_period,
          generated_at: new Date().toISOString(),
          cache_refreshed: refresh_cache
        }
      });
      
    } catch (error) {
      fastify.log.error(`Error generating visualizations: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to generate visualizations',
        message: error.message
      });
    }
  });
  
  // Route: Get market benchmarks
  fastify.get('/benchmarks', {
    schema: {
      description: 'Get market benchmarks and comparative analysis',
      tags: ['Analytics', 'Benchmarks'],
      querystring: {
        type: 'object',
        properties: {
          benchmark_type: {
            type: 'string',
            enum: ['peer_comparison', 'historical_comparison', 'regional_comparison', 'international_comparison'],
            default: 'peer_comparison'
          },
          metrics: {
            type: 'array',
            items: { type: 'string' },
            default: ['interest_rates', 'approval_rates', 'processing_times', 'loan_amounts']
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const { benchmark_type = 'peer_comparison', metrics = ['interest_rates', 'approval_rates', 'processing_times', 'loan_amounts'] } = request.query;
      
      fastify.log.info(`Generating benchmarks: ${benchmark_type} for metrics ${metrics.join(',')}`);
      
      const result = await executePythonScript('analytics_dashboard_executor.py', [
        'generate_benchmarks',
        '--benchmark_type', benchmark_type,
        '--metrics', metrics.join(','),
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        benchmark_type,
        metrics,
        benchmarks: result.benchmarks,
        comparative_analysis: result.comparative_analysis,
        generated_at: new Date().toISOString()
      });
      
    } catch (error) {
      fastify.log.error(`Error generating benchmarks: ${error.message}`);
      reply.code(500).send({
        success: false,
        error: 'Failed to generate benchmarks',
        message: error.message
      });
    }
  });
  
  // Route: Get analytics configuration
  fastify.get('/config', {
    schema: {
      description: 'Get analytics dashboard configuration',
      tags: ['Analytics', 'Configuration']
    }
  }, async (request, reply) => {
    try {
      reply.code(200).send({
        success: true,
        configuration: {
          analytics_enabled: process.env.ANALYTICS_DASHBOARD_ENABLED === 'true',
          timeout: parseInt(process.env.ANALYTICS_TIMEOUT || '300000'),
          cache_ttl: parseInt(process.env.ANALYTICS_CACHE_TTL || '900'),
          max_forecast_periods: parseInt(process.env.MAX_FORECAST_PERIODS || '24'),
          supported_formats: ['json', 'pdf', 'html', 'excel'],
          available_models: ['linear', 'random_forest', 'gradient_boosting', 'ensemble'],
          market_data_sources: ['CBS', 'DNB', 'Kadaster', 'AFM', 'NHG', 'BKR'],
          real_time_updates: process.env.ENABLE_REAL_TIME_ANALYTICS === 'true'
        },
        feature_flags: {
          predictive_modeling: process.env.ENABLE_PREDICTIVE_MODELING === 'true',
          advanced_visualizations: process.env.ENABLE_ADVANCED_VISUALIZATIONS === 'true',
          export_functionality: process.env.ENABLE_ANALYTICS_EXPORT === 'true',
          benchmark_analysis: process.env.ENABLE_BENCHMARK_ANALYSIS === 'true'
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
  
  // Route: Health check for analytics services
  fastify.get('/health', {
    schema: {
      description: 'Health check for analytics dashboard services',
      tags: ['Analytics', 'Health']
    }
  }, async (request, reply) => {
    try {
      const healthCheck = await executePythonScript('analytics_dashboard_executor.py', [
        'health_check',
        '--output_format', 'json'
      ]);
      
      reply.code(200).send({
        success: true,
        status: 'healthy',
        timestamp: new Date().toISOString(),
        services: healthCheck.services,
        performance: healthCheck.performance,
        dependencies: healthCheck.dependencies
      });
      
    } catch (error) {
      fastify.log.error(`Analytics health check failed: ${error.message}`);
      reply.code(503).send({
        success: false,
        status: 'unhealthy',
        error: 'Analytics services unavailable',
        message: error.message
      });
    }
  });
}

module.exports = advancedAnalyticsDashboardRoutes;
