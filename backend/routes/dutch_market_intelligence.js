/**
 * Dutch Market Intelligence API Routes
 * ===================================
 * 
 * Fastify routes for Dutch mortgage market intelligence with real-time data feeds,
 * comprehensive trend analysis, and advanced predictive insights.
 * 
 * This module provides production-grade API endpoints including:
 * - Real-time market data collection from Dutch sources (CBS, DNB, Kadaster, AFM, NHG, BKR)
 * - Advanced trend analysis with statistical modeling and pattern recognition
 * - Predictive analytics with machine learning models and forecasting
 * - Market sentiment analysis and risk assessment
 * - Comprehensive market intelligence reporting and visualization
 * - Task management and execution monitoring
 * - Performance analytics and system health monitoring
 * 
 * Features:
 * - RESTful API design with comprehensive error handling
 * - Real-time data integration and processing
 * - Advanced analytics and machine learning capabilities
 * - Comprehensive validation and security measures
 * - Performance monitoring and optimization
 * - Production-grade logging and audit trails
 * 
 * @author MortgageAI Development Team
 * @date 2025-01-27
 * @version 1.0.0
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

/**
 * Dutch Market Intelligence API Routes Plugin
 * @param {Object} fastify - Fastify instance
 * @param {Object} options - Plugin options
 */
async function dutchMarketIntelligenceRoutes(fastify, options) {
  
  // Schema definitions for validation
  const schemas = {
    
    // Data collection request schema
    dataCollectionRequest: {
      type: 'object',
      required: ['source', 'metrics', 'date_range'],
      properties: {
        source: { 
          type: 'string', 
          enum: ['cbs', 'dnb', 'kadaster', 'afm', 'nhg', 'bkr', 'ecb', 'eurostat', 'custom']
        },
        metrics: {
          type: 'array',
          items: { type: 'string' },
          minItems: 1,
          maxItems: 20
        },
        date_range: {
          type: 'object',
          required: ['start_date', 'end_date'],
          properties: {
            start_date: { type: 'string', format: 'date' },
            end_date: { type: 'string', format: 'date' }
          }
        },
        filters: {
          type: 'object',
          additionalProperties: true
        },
        options: {
          type: 'object',
          additionalProperties: true
        }
      }
    },
    
    // Trend analysis request schema
    trendAnalysisRequest: {
      type: 'object',
      required: ['data_points'],
      properties: {
        data_points: {
          type: 'array',
          items: {
            type: 'object',
            required: ['source', 'metric_name', 'value', 'timestamp'],
            properties: {
              source: { type: 'string' },
              metric_name: { type: 'string' },
              value: { type: 'number' },
              timestamp: { type: 'string', format: 'date-time' },
              metadata: { type: 'object' },
              quality_score: { type: 'number', minimum: 0, maximum: 1 },
              confidence_level: { type: 'number', minimum: 0, maximum: 1 }
            }
          },
          minItems: 10
        },
        analysis_type: {
          type: 'string',
          enum: ['trend_analysis', 'predictive_modeling', 'sentiment_analysis', 'risk_assessment', 'correlation_analysis', 'seasonal_analysis', 'volatility_analysis', 'comparative_analysis'],
          default: 'trend_analysis'
        },
        parameters: {
          type: 'object',
          additionalProperties: true
        }
      }
    },
    
    // Predictive model request schema
    predictiveModelRequest: {
      type: 'object',
      required: ['training_data', 'target_metric', 'features'],
      properties: {
        training_data: {
          type: 'array',
          items: {
            type: 'object',
            required: ['source', 'metric_name', 'value', 'timestamp'],
            properties: {
              source: { type: 'string' },
              metric_name: { type: 'string' },
              value: { type: 'number' },
              timestamp: { type: 'string', format: 'date-time' },
              metadata: { type: 'object' }
            }
          },
          minItems: 50
        },
        target_metric: { type: 'string' },
        features: {
          type: 'array',
          items: { type: 'string' },
          minItems: 1,
          maxItems: 50
        },
        model_type: {
          type: 'string',
          enum: ['random_forest', 'gradient_boosting', 'linear_regression', 'ridge', 'lasso'],
          default: 'random_forest'
        },
        parameters: {
          type: 'object',
          additionalProperties: true
        }
      }
    },
    
    // Market insights request schema
    marketInsightsRequest: {
      type: 'object',
      required: ['analysis_results'],
      properties: {
        analysis_results: {
          type: 'array',
          items: {
            type: 'object',
            required: ['type'],
            properties: {
              type: { type: 'string', enum: ['trend_analysis', 'predictive_model'] }
            }
          },
          minItems: 1
        },
        context: {
          type: 'object',
          additionalProperties: true
        },
        filters: {
          type: 'object',
          properties: {
            min_importance: { type: 'number', minimum: 0, maximum: 1 },
            risk_levels: {
              type: 'array',
              items: { type: 'string', enum: ['very_low', 'low', 'moderate', 'high', 'very_high', 'critical'] }
            }
          }
        }
      }
    },
    
    // Comprehensive report request schema
    comprehensiveReportRequest: {
      type: 'object',
      required: ['insights'],
      properties: {
        insights: {
          type: 'array',
          items: {
            type: 'object',
            required: ['insight_id', 'title', 'description', 'category', 'importance_score', 'confidence_level', 'implications', 'recommendations', 'risk_level', 'time_horizon', 'generated_at'],
            properties: {
              insight_id: { type: 'string' },
              title: { type: 'string' },
              description: { type: 'string' },
              category: { type: 'string' },
              importance_score: { type: 'number', minimum: 0, maximum: 1 },
              confidence_level: { type: 'number', minimum: 0, maximum: 1 },
              implications: { type: 'array', items: { type: 'string' } },
              recommendations: { type: 'array', items: { type: 'string' } },
              risk_level: { type: 'string', enum: ['very_low', 'low', 'moderate', 'high', 'very_high', 'critical'] },
              time_horizon: { type: 'string' },
              generated_at: { type: 'string', format: 'date-time' }
            }
          },
          minItems: 1
        },
        include_visualizations: { type: 'boolean', default: true },
        report_format: { type: 'string', enum: ['json', 'pdf', 'html'], default: 'json' },
        parameters: {
          type: 'object',
          additionalProperties: true
        }
      }
    },
    
    // Task submission schema
    taskSubmissionRequest: {
      type: 'object',
      required: ['task_type', 'task_name', 'parameters'],
      properties: {
        task_type: {
          type: 'string',
          enum: ['collect_market_data', 'perform_trend_analysis', 'generate_predictive_model', 'generate_market_insights', 'generate_comprehensive_report', 'data_validation', 'cache_management', 'system_health_check', 'performance_optimization']
        },
        task_name: { type: 'string', minLength: 1, maxLength: 500 },
        parameters: {
          type: 'object',
          additionalProperties: true
        },
        priority: {
          type: 'string',
          enum: ['critical', 'high', 'normal', 'low', 'background'],
          default: 'normal'
        },
        timeout_seconds: { type: 'integer', minimum: 60, maximum: 7200, default: 1800 },
        max_retries: { type: 'integer', minimum: 0, maximum: 5, default: 3 },
        dependencies: {
          type: 'array',
          items: { type: 'string' },
          default: []
        },
        scheduled_at: { type: 'string', format: 'date-time' },
        metadata: {
          type: 'object',
          additionalProperties: true,
          default: {}
        }
      }
    }
  };

  /**
   * Execute Python script with parameters
   * @param {string} scriptName - Name of the Python script
   * @param {Object} parameters - Parameters to pass to the script
   * @returns {Promise<Object>} - Script execution result
   */
  async function executePythonScript(scriptName, parameters) {
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(__dirname, '..', 'agents', 'utils', scriptName);
      const pythonProcess = spawn('python3', [scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: parseInt(process.env.INTELLIGENCE_SCRIPT_TIMEOUT) || 300000 // 5 minutes default
      });

      let stdout = '';
      let stderr = '';

      // Send parameters as JSON to stdin
      pythonProcess.stdin.write(JSON.stringify(parameters));
      pythonProcess.stdin.end();

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
            reject(new Error(`Failed to parse Python script output: ${parseError.message}`));
          }
        } else {
          reject(new Error(`Python script failed with code ${code}: ${stderr}`));
        }
      });

      pythonProcess.on('error', (error) => {
        reject(new Error(`Failed to execute Python script: ${error.message}`));
      });
    });
  }

  /**
   * Generate execution ID for tracking
   * @returns {string} - Unique execution ID
   */
  function generateExecutionId() {
    return `intel_${Date.now()}_${crypto.randomBytes(8).toString('hex')}`;
  }

  /**
   * Validate date range
   * @param {string} startDate - Start date
   * @param {string} endDate - End date
   */
  function validateDateRange(startDate, endDate) {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const now = new Date();
    
    if (start >= end) {
      throw new Error('Start date must be before end date');
    }
    
    if (end > now) {
      throw new Error('End date cannot be in the future');
    }
    
    const maxDays = parseInt(process.env.INTELLIGENCE_MAX_DATE_RANGE_DAYS) || 365;
    const daysDiff = (end - start) / (1000 * 60 * 60 * 24);
    
    if (daysDiff > maxDays) {
      throw new Error(`Date range cannot exceed ${maxDays} days`);
    }
  }

  // =============================================================================
  // MARKET DATA COLLECTION ENDPOINTS
  // =============================================================================

  /**
   * Collect market data from specified Dutch sources
   * POST /collect-data
   */
  fastify.post('/collect-data', {
    schema: {
      tags: ['Market Data'],
      summary: 'Collect market data from Dutch sources',
      description: 'Collect real-time market data from various Dutch sources including CBS, DNB, Kadaster, AFM, NHG, and BKR',
      body: schemas.dataCollectionRequest,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            execution_id: { type: 'string' },
            data: {
              type: 'object',
              properties: {
                source: { type: 'string' },
                metrics: { type: 'array', items: { type: 'string' } },
                data_points: { type: 'array' },
                collection_metadata: { type: 'object' }
              }
            },
            metadata: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const executionId = generateExecutionId();
    
    try {
      fastify.log.info({ executionId, source: request.body.source }, 'Starting market data collection');
      
      // Validate date range
      validateDateRange(request.body.date_range.start_date, request.body.date_range.end_date);
      
      // Prepare parameters for Python script
      const parameters = {
        operation: 'collect_market_data',
        execution_id: executionId,
        source: request.body.source,
        metrics: request.body.metrics,
        date_range: request.body.date_range,
        filters: request.body.filters || {},
        options: request.body.options || {},
        user_id: request.user?.id || 'anonymous',
        timestamp: new Date().toISOString()
      };
      
      // Execute data collection
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      fastify.log.info({ 
        executionId, 
        dataPoints: result.data?.data_points?.length || 0 
      }, 'Market data collection completed');
      
      return {
        success: true,
        execution_id: executionId,
        data: result.data || {},
        metadata: {
          execution_time: result.execution_time || 0,
          timestamp: new Date().toISOString(),
          source: request.body.source,
          metrics_count: request.body.metrics.length
        }
      };
      
    } catch (error) {
      fastify.log.error({ 
        executionId, 
        error: error.message,
        stack: error.stack
      }, 'Market data collection failed');
      
      reply.code(500);
      return {
        success: false,
        execution_id: executionId,
        error: {
          message: 'Market data collection failed',
          details: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
          code: 'DATA_COLLECTION_ERROR'
        }
      };
    }
  });

  /**
   * Get available data sources and metrics
   * GET /data-sources
   */
  fastify.get('/data-sources', {
    schema: {
      tags: ['Market Data'],
      summary: 'Get available data sources and metrics',
      description: 'Retrieve information about available Dutch market data sources and their supported metrics',
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            data_sources: {
              type: 'object',
              additionalProperties: {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  description: { type: 'string' },
                  available_metrics: { type: 'array', items: { type: 'string' } },
                  update_frequency: { type: 'string' },
                  data_quality: { type: 'string' }
                }
              }
            }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      const dataSources = {
        cbs: {
          name: 'Centraal Bureau voor de Statistiek',
          description: 'Dutch national statistics office providing economic and demographic data',
          available_metrics: [
            'house_prices', 'construction_permits', 'mortgage_rates', 'economic_growth', 
            'unemployment', 'inflation', 'consumer_confidence', 'housing_starts',
            'population_statistics', 'gdp_growth', 'interest_rates'
          ],
          update_frequency: 'daily',
          data_quality: 'high'
        },
        dnb: {
          name: 'De Nederlandsche Bank',
          description: 'Dutch central bank providing financial and monetary policy data',
          available_metrics: [
            'bank_lending_rates', 'mortgage_lending_volume', 'household_debt',
            'financial_stability_indicators', 'credit_default_rates', 'systemic_risk_indicators',
            'money_supply', 'exchange_rates', 'banking_statistics'
          ],
          update_frequency: 'daily',
          data_quality: 'very_high'
        },
        kadaster: {
          name: 'Netherlands Cadastre',
          description: 'Dutch land registry providing property and real estate data',
          available_metrics: [
            'property_transactions', 'property_values', 'land_use_changes',
            'construction_activity', 'property_registrations', 'ownership_transfers',
            'mortgage_registrations', 'property_tax_assessments'
          ],
          update_frequency: 'weekly',
          data_quality: 'high'
        },
        afm: {
          name: 'Autoriteit Financiële Markten',
          description: 'Dutch financial markets authority providing regulatory and compliance data',
          available_metrics: [
            'regulatory_changes', 'compliance_indicators', 'market_conduct_metrics',
            'consumer_protection_metrics', 'supervisory_actions', 'enforcement_actions',
            'market_integrity_indicators', 'prudential_requirements'
          ],
          update_frequency: 'weekly',
          data_quality: 'high'
        },
        nhg: {
          name: 'Nationale Hypotheek Garantie',
          description: 'National mortgage guarantee providing mortgage insurance data',
          available_metrics: [
            'guarantee_volumes', 'guarantee_rates', 'claim_rates', 'coverage_ratios',
            'default_rates', 'recovery_rates', 'risk_assessments', 'premium_calculations'
          ],
          update_frequency: 'monthly',
          data_quality: 'high'
        },
        bkr: {
          name: 'Bureau Krediet Registratie',
          description: 'Dutch credit registration bureau providing credit market data',
          available_metrics: [
            'credit_registrations', 'default_rates', 'credit_inquiries', 'debt_levels',
            'payment_behavior', 'credit_utilization', 'risk_profiles', 'collection_activities'
          ],
          update_frequency: 'daily',
          data_quality: 'high'
        }
      };
      
      return {
        success: true,
        data_sources: dataSources,
        metadata: {
          total_sources: Object.keys(dataSources).length,
          last_updated: new Date().toISOString()
        }
      };
      
    } catch (error) {
      fastify.log.error({ error: error.message }, 'Failed to get data sources');
      
      reply.code(500);
      return {
        success: false,
        error: {
          message: 'Failed to retrieve data sources information',
          code: 'DATA_SOURCES_ERROR'
        }
      };
    }
  });

  // =============================================================================
  // TREND ANALYSIS ENDPOINTS
  // =============================================================================

  /**
   * Perform trend analysis on market data
   * POST /analyze-trends
   */
  fastify.post('/analyze-trends', {
    schema: {
      tags: ['Analytics'],
      summary: 'Perform trend analysis on market data',
      description: 'Analyze market trends using statistical methods and pattern recognition',
      body: schemas.trendAnalysisRequest,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            execution_id: { type: 'string' },
            analysis: { type: 'object' },
            metadata: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const executionId = generateExecutionId();
    
    try {
      fastify.log.info({ 
        executionId, 
        dataPoints: request.body.data_points.length,
        analysisType: request.body.analysis_type
      }, 'Starting trend analysis');
      
      // Prepare parameters for Python script
      const parameters = {
        operation: 'perform_trend_analysis',
        execution_id: executionId,
        data_points: request.body.data_points,
        analysis_type: request.body.analysis_type || 'trend_analysis',
        parameters: request.body.parameters || {},
        user_id: request.user?.id || 'anonymous',
        timestamp: new Date().toISOString()
      };
      
      // Execute trend analysis
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      fastify.log.info({ 
        executionId,
        trendType: result.analysis?.trend_type,
        confidence: result.analysis?.confidence
      }, 'Trend analysis completed');
      
      return {
        success: true,
        execution_id: executionId,
        analysis: result.analysis || {},
        metadata: {
          execution_time: result.execution_time || 0,
          timestamp: new Date().toISOString(),
          data_points_analyzed: request.body.data_points.length,
          analysis_type: request.body.analysis_type
        }
      };
      
    } catch (error) {
      fastify.log.error({ 
        executionId, 
        error: error.message,
        stack: error.stack
      }, 'Trend analysis failed');
      
      reply.code(500);
      return {
        success: false,
        execution_id: executionId,
        error: {
          message: 'Trend analysis failed',
          details: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
          code: 'TREND_ANALYSIS_ERROR'
        }
      };
    }
  });

  // =============================================================================
  // PREDICTIVE MODELING ENDPOINTS
  // =============================================================================

  /**
   * Generate predictive model for market forecasting
   * POST /generate-model
   */
  fastify.post('/generate-model', {
    schema: {
      tags: ['Predictive Analytics'],
      summary: 'Generate predictive model for market forecasting',
      description: 'Create machine learning models for predicting market trends and values',
      body: schemas.predictiveModelRequest,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            execution_id: { type: 'string' },
            model: { type: 'object' },
            metadata: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const executionId = generateExecutionId();
    
    try {
      fastify.log.info({ 
        executionId,
        targetMetric: request.body.target_metric,
        modelType: request.body.model_type,
        trainingSize: request.body.training_data.length
      }, 'Starting predictive model generation');
      
      // Prepare parameters for Python script
      const parameters = {
        operation: 'generate_predictive_model',
        execution_id: executionId,
        training_data: request.body.training_data,
        target_metric: request.body.target_metric,
        features: request.body.features,
        model_type: request.body.model_type || 'random_forest',
        parameters: request.body.parameters || {},
        user_id: request.user?.id || 'anonymous',
        timestamp: new Date().toISOString()
      };
      
      // Execute model generation
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      fastify.log.info({ 
        executionId,
        accuracy: result.model?.accuracy_score,
        r2Score: result.model?.r2_score
      }, 'Predictive model generation completed');
      
      return {
        success: true,
        execution_id: executionId,
        model: result.model || {},
        metadata: {
          execution_time: result.execution_time || 0,
          timestamp: new Date().toISOString(),
          training_samples: request.body.training_data.length,
          features_count: request.body.features.length,
          model_type: request.body.model_type
        }
      };
      
    } catch (error) {
      fastify.log.error({ 
        executionId, 
        error: error.message,
        stack: error.stack
      }, 'Predictive model generation failed');
      
      reply.code(500);
      return {
        success: false,
        execution_id: executionId,
        error: {
          message: 'Predictive model generation failed',
          details: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
          code: 'PREDICTIVE_MODEL_ERROR'
        }
      };
    }
  });

  // =============================================================================
  // MARKET INSIGHTS ENDPOINTS
  // =============================================================================

  /**
   * Generate market insights from analysis results
   * POST /generate-insights
   */
  fastify.post('/generate-insights', {
    schema: {
      tags: ['Market Intelligence'],
      summary: 'Generate market insights from analysis results',
      description: 'Generate actionable market insights from trend analysis and predictive models',
      body: schemas.marketInsightsRequest,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            execution_id: { type: 'string' },
            insights: { type: 'array' },
            metadata: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const executionId = generateExecutionId();
    
    try {
      fastify.log.info({ 
        executionId,
        analysisResultsCount: request.body.analysis_results.length
      }, 'Starting market insights generation');
      
      // Prepare parameters for Python script
      const parameters = {
        operation: 'generate_market_insights',
        execution_id: executionId,
        analysis_results: request.body.analysis_results,
        context: request.body.context || {},
        filters: request.body.filters || {},
        user_id: request.user?.id || 'anonymous',
        timestamp: new Date().toISOString()
      };
      
      // Execute insights generation
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      fastify.log.info({ 
        executionId,
        insightsGenerated: result.insights?.length || 0
      }, 'Market insights generation completed');
      
      return {
        success: true,
        execution_id: executionId,
        insights: result.insights || [],
        metadata: {
          execution_time: result.execution_time || 0,
          timestamp: new Date().toISOString(),
          insights_count: result.insights?.length || 0,
          analysis_inputs: request.body.analysis_results.length
        }
      };
      
    } catch (error) {
      fastify.log.error({ 
        executionId, 
        error: error.message,
        stack: error.stack
      }, 'Market insights generation failed');
      
      reply.code(500);
      return {
        success: false,
        execution_id: executionId,
        error: {
          message: 'Market insights generation failed',
          details: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
          code: 'INSIGHTS_GENERATION_ERROR'
        }
      };
    }
  });

  // =============================================================================
  // COMPREHENSIVE REPORTING ENDPOINTS
  // =============================================================================

  /**
   * Generate comprehensive market intelligence report
   * POST /generate-report
   */
  fastify.post('/generate-report', {
    schema: {
      tags: ['Reporting'],
      summary: 'Generate comprehensive market intelligence report',
      description: 'Create comprehensive reports with market analysis, insights, and recommendations',
      body: schemas.comprehensiveReportRequest,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            execution_id: { type: 'string' },
            report: { type: 'object' },
            metadata: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const executionId = generateExecutionId();
    
    try {
      fastify.log.info({ 
        executionId,
        insightsCount: request.body.insights.length,
        reportFormat: request.body.report_format
      }, 'Starting comprehensive report generation');
      
      // Prepare parameters for Python script
      const parameters = {
        operation: 'generate_comprehensive_report',
        execution_id: executionId,
        insights: request.body.insights,
        include_visualizations: request.body.include_visualizations !== false,
        report_format: request.body.report_format || 'json',
        parameters: request.body.parameters || {},
        user_id: request.user?.id || 'anonymous',
        timestamp: new Date().toISOString()
      };
      
      // Execute report generation
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      fastify.log.info({ 
        executionId,
        reportSections: Object.keys(result.report || {}).length
      }, 'Comprehensive report generation completed');
      
      return {
        success: true,
        execution_id: executionId,
        report: result.report || {},
        metadata: {
          execution_time: result.execution_time || 0,
          timestamp: new Date().toISOString(),
          insights_processed: request.body.insights.length,
          report_format: request.body.report_format,
          include_visualizations: request.body.include_visualizations
        }
      };
      
    } catch (error) {
      fastify.log.error({ 
        executionId, 
        error: error.message,
        stack: error.stack
      }, 'Comprehensive report generation failed');
      
      reply.code(500);
      return {
        success: false,
        execution_id: executionId,
        error: {
          message: 'Comprehensive report generation failed',
          details: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
          code: 'REPORT_GENERATION_ERROR'
        }
      };
    }
  });

  // =============================================================================
  // TASK MANAGEMENT ENDPOINTS
  // =============================================================================

  /**
   * Submit intelligence task for execution
   * POST /submit-task
   */
  fastify.post('/submit-task', {
    schema: {
      tags: ['Task Management'],
      summary: 'Submit intelligence task for execution',
      description: 'Submit market intelligence tasks for asynchronous execution',
      body: schemas.taskSubmissionRequest,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            task_id: { type: 'string' },
            status: { type: 'string' },
            metadata: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const taskId = uuidv4();
    
    try {
      fastify.log.info({ 
        taskId,
        taskType: request.body.task_type,
        priority: request.body.priority
      }, 'Submitting intelligence task');
      
      // Prepare task definition
      const taskDefinition = {
        task_id: taskId,
        task_type: request.body.task_type,
        task_name: request.body.task_name,
        parameters: request.body.parameters,
        priority: request.body.priority || 'normal',
        timeout_seconds: request.body.timeout_seconds || 1800,
        max_retries: request.body.max_retries || 3,
        dependencies: request.body.dependencies || [],
        created_at: new Date().toISOString(),
        scheduled_at: request.body.scheduled_at || null,
        user_id: request.user?.id || 'anonymous',
        metadata: request.body.metadata || {}
      };
      
      // Submit task to executor
      const parameters = {
        operation: 'submit_task',
        task_definition: taskDefinition,
        timestamp: new Date().toISOString()
      };
      
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      fastify.log.info({ 
        taskId,
        status: result.status
      }, 'Intelligence task submitted successfully');
      
      return {
        success: true,
        task_id: taskId,
        status: result.status || 'queued',
        metadata: {
          submission_time: new Date().toISOString(),
          estimated_completion: result.estimated_completion,
          priority: request.body.priority,
          task_type: request.body.task_type
        }
      };
      
    } catch (error) {
      fastify.log.error({ 
        taskId, 
        error: error.message,
        stack: error.stack
      }, 'Intelligence task submission failed');
      
      reply.code(500);
      return {
        success: false,
        task_id: taskId,
        error: {
          message: 'Task submission failed',
          details: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
          code: 'TASK_SUBMISSION_ERROR'
        }
      };
    }
  });

  /**
   * Get task status and results
   * GET /task/:taskId
   */
  fastify.get('/task/:taskId', {
    schema: {
      tags: ['Task Management'],
      summary: 'Get task status and results',
      description: 'Retrieve the current status and results of a submitted task',
      params: {
        type: 'object',
        required: ['taskId'],
        properties: {
          taskId: { type: 'string', format: 'uuid' }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            task_id: { type: 'string' },
            status: { type: 'string' },
            result: { type: 'object' },
            metadata: { type: 'object' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const { taskId } = request.params;
    
    try {
      fastify.log.info({ taskId }, 'Getting task status');
      
      // Get task status from executor
      const parameters = {
        operation: 'get_task_status',
        task_id: taskId,
        timestamp: new Date().toISOString()
      };
      
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      if (!result.task_found) {
        reply.code(404);
        return {
          success: false,
          task_id: taskId,
          error: {
            message: 'Task not found',
            code: 'TASK_NOT_FOUND'
          }
        };
      }
      
      return {
        success: true,
        task_id: taskId,
        status: result.status,
        result: result.result_data || null,
        metadata: {
          start_time: result.start_time,
          end_time: result.end_time,
          execution_time: result.execution_time,
          retry_count: result.retry_count,
          resource_usage: result.resource_usage
        }
      };
      
    } catch (error) {
      fastify.log.error({ 
        taskId, 
        error: error.message
      }, 'Failed to get task status');
      
      reply.code(500);
      return {
        success: false,
        task_id: taskId,
        error: {
          message: 'Failed to get task status',
          details: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error',
          code: 'TASK_STATUS_ERROR'
        }
      };
    }
  });

  // =============================================================================
  // SYSTEM MONITORING ENDPOINTS
  // =============================================================================

  /**
   * Get system health status
   * GET /health
   */
  fastify.get('/health', {
    schema: {
      tags: ['System'],
      summary: 'Get system health status',
      description: 'Check the health and status of the market intelligence system',
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            status: { type: 'string' },
            services: { type: 'object' },
            metrics: { type: 'object' },
            timestamp: { type: 'string' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      // Get system health from executor
      const parameters = {
        operation: 'system_health_check',
        timestamp: new Date().toISOString()
      };
      
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      return {
        success: true,
        status: result.overall_status || 'unknown',
        services: result.service_checks || {},
        metrics: result.metrics || {},
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      fastify.log.error({ 
        error: error.message
      }, 'Health check failed');
      
      // Return degraded status on error
      reply.code(503);
      return {
        success: false,
        status: 'degraded',
        error: {
          message: 'Health check failed',
          code: 'HEALTH_CHECK_ERROR'
        },
        timestamp: new Date().toISOString()
      };
    }
  });

  /**
   * Get system performance metrics
   * GET /metrics
   */
  fastify.get('/metrics', {
    schema: {
      tags: ['System'],
      summary: 'Get system performance metrics',
      description: 'Retrieve detailed performance metrics for the market intelligence system',
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            metrics: { type: 'object' },
            timestamp: { type: 'string' }
          }
        }
      }
    }
  }, async (request, reply) => {
    try {
      // Get system metrics from executor
      const parameters = {
        operation: 'get_metrics',
        timestamp: new Date().toISOString()
      };
      
      const result = await executePythonScript('dutch_market_intelligence_executor.py', parameters);
      
      return {
        success: true,
        metrics: result.metrics || {},
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      fastify.log.error({ 
        error: error.message
      }, 'Metrics retrieval failed');
      
      reply.code(500);
      return {
        success: false,
        error: {
          message: 'Failed to retrieve system metrics',
          code: 'METRICS_ERROR'
        },
        timestamp: new Date().toISOString()
      };
    }
  });
}

// Export the plugin with metadata
module.exports = dutchMarketIntelligenceRoutes;
module.exports[Symbol.for('plugin-meta')] = {
  name: 'dutch-market-intelligence-routes',
  version: '1.0.0',
  description: 'Dutch Market Intelligence API Routes for MortgageAI'
};
