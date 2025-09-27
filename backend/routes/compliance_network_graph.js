/**
 * Compliance Network Graph Visualization API Routes
 * 
 * This module provides RESTful API endpoints for compliance network graph analysis,
 * risk propagation modeling, and interactive visualization capabilities.
 * 
 * Features:
 * - Network construction and analysis
 * - Risk propagation simulation
 * - Interactive graph exploration
 * - Regulatory change impact assessment
 * - Anomaly detection and alerting
 * - Comprehensive visualization data generation
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

/**
 * Compliance Network Graph Routes Plugin
 * @param {object} fastify - Fastify instance
 * @param {object} options - Plugin options
 */
async function complianceNetworkGraphRoutes(fastify, options) {
  // Configuration
  const PYTHON_SCRIPT_PATH = path.join(__dirname, '../agents/utils/compliance_network_executor.py');
  const NETWORK_DATA_DIR = path.join(__dirname, '../../uploads/network_data');
  const ANALYSIS_CACHE_DIR = path.join(__dirname, '../../cache/network_analysis');
  
  // Ensure directories exist
  await ensureDirectoryExists(NETWORK_DATA_DIR);
  await ensureDirectoryExists(ANALYSIS_CACHE_DIR);

  // Schema definitions
  const networkNodeSchema = {
    type: 'object',
    required: ['node_id', 'node_type', 'label'],
    properties: {
      node_id: { type: 'string' },
      node_type: { 
        type: 'string', 
        enum: ['client', 'advisor', 'regulation', 'mortgage_product', 'lender', 'compliance_rule', 'risk_factor', 'document', 'process', 'audit_event'] 
      },
      label: { type: 'string' },
      properties: { type: 'object' },
      risk_score: { type: 'number', minimum: 0, maximum: 1 },
      compliance_status: { type: 'string' }
    }
  };

  const networkEdgeSchema = {
    type: 'object',
    required: ['source_node', 'target_node', 'edge_type'],
    properties: {
      edge_id: { type: 'string' },
      source_node: { type: 'string' },
      target_node: { type: 'string' },
      edge_type: { 
        type: 'string',
        enum: ['advises', 'applies_to', 'complies_with', 'violates', 'depends_on', 'influences', 'requires', 'produces', 'validates', 'triggers', 'escalates']
      },
      weight: { type: 'number', minimum: 0 },
      confidence: { type: 'number', minimum: 0, maximum: 1 },
      risk_contribution: { type: 'number', minimum: 0, maximum: 1 },
      properties: { type: 'object' }
    }
  };

  const networkDataSchema = {
    type: 'object',
    required: ['clients', 'advisors', 'regulations', 'relationships'],
    properties: {
      clients: { type: 'array', items: { type: 'object' } },
      advisors: { type: 'array', items: { type: 'object' } },
      regulations: { type: 'array', items: { type: 'object' } },
      relationships: { type: 'array', items: networkEdgeSchema }
    }
  };

  const riskPropagationSchema = {
    type: 'object',
    required: ['source_nodes'],
    properties: {
      source_nodes: { type: 'array', items: { type: 'string' } },
      propagation_type: { 
        type: 'string', 
        enum: ['linear', 'exponential', 'logarithmic', 'threshold', 'cascade'],
        default: 'linear'
      },
      max_steps: { type: 'integer', minimum: 1, maximum: 50, default: 10 },
      convergence_tolerance: { type: 'number', minimum: 0.0001, maximum: 0.1, default: 0.001 }
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
  async function getCachedResult(cacheKey, maxAgeMinutes = 60) {
    try {
      const cacheFile = path.join(ANALYSIS_CACHE_DIR, `${cacheKey}.json`);
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
      const cacheFile = path.join(ANALYSIS_CACHE_DIR, `${cacheKey}.json`);
      await fs.writeFile(cacheFile, JSON.stringify(result, null, 2));
    } catch (error) {
      fastify.log.warn(`Failed to cache result: ${error.message}`);
    }
  }

  // Route: Build and analyze compliance network
  fastify.post('/analyze', {
    schema: {
      description: 'Build compliance network from data and perform comprehensive analysis',
      tags: ['Compliance Network Graph'],
      body: networkDataSchema,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            analysis_id: { type: 'string' },
            network_stats: { type: 'object' },
            centrality_analysis: { type: 'object' },
            community_structure: { type: 'object' },
            risk_assessment: { type: 'object' },
            anomaly_detection: { type: 'object' },
            recommendations: { type: 'array', items: { type: 'string' } },
            visualization_data: { type: 'object' },
            analysis_timestamp: { type: 'string', format: 'date-time' },
            processing_time: { type: 'number' }
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
    const startTime = Date.now();
    
    try {
      const { clients, advisors, regulations, relationships } = request.body;

      // Validate input data
      if (!clients || !advisors || !regulations || !relationships) {
        return reply.status(400).send({
          error: 'Missing required network data',
          details: 'clients, advisors, regulations, and relationships are required'
        });
      }

      // Check cache
      const cacheKey = generateCacheKey(request.body);
      let cachedResult = await getCachedResult(cacheKey, 30); // 30 minutes cache
      
      if (cachedResult) {
        fastify.log.info(`Returning cached network analysis result for key: ${cacheKey}`);
        return reply.send({
          ...cachedResult,
          cached: true,
          cache_key: cacheKey
        });
      }

      // Execute network analysis
      const analysisResult = await executePythonScript(
        ['analyze_network'],
        {
          clients,
          advisors,
          regulations,
          relationships,
          analysis_options: {
            enable_centrality: true,
            enable_communities: true,
            enable_anomaly_detection: true,
            enable_visualization: true,
            layout_algorithm: 'spring'
          }
        }
      );

      if (!analysisResult.success) {
        throw new Error(analysisResult.error || 'Network analysis failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;
      const response = {
        success: true,
        analysis_id: analysisResult.analysis_id,
        network_stats: analysisResult.network_stats,
        centrality_analysis: analysisResult.centrality_analysis,
        community_structure: analysisResult.community_structure,
        risk_assessment: analysisResult.risk_assessment,
        anomaly_detection: analysisResult.anomaly_detection,
        recommendations: analysisResult.recommendations,
        visualization_data: analysisResult.visualization_data,
        analysis_timestamp: new Date().toISOString(),
        processing_time: processingTime,
        cached: false
      };

      // Cache the result
      await setCachedResult(cacheKey, response);

      return reply.send(response);

    } catch (error) {
      fastify.log.error(`Compliance network analysis error: ${error.message}`);
      return reply.status(500).send({
        error: 'Network analysis failed',
        details: error.message
      });
    }
  });

  // Route: Analyze risk propagation
  fastify.post('/risk-propagation', {
    schema: {
      description: 'Analyze risk propagation through the compliance network',
      tags: ['Compliance Network Graph'],
      body: {
        type: 'object',
        required: ['network_data', 'propagation_request'],
        properties: {
          network_data: networkDataSchema,
          propagation_request: riskPropagationSchema
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            propagation_id: { type: 'string' },
            source_nodes: { type: 'array', items: { type: 'string' } },
            affected_nodes: { type: 'object' },
            propagation_paths: { type: 'array' },
            total_risk_increase: { type: 'number' },
            propagation_time_steps: { type: 'integer' },
            convergence_achieved: { type: 'boolean' },
            critical_paths: { type: 'array' },
            mitigation_recommendations: { type: 'array', items: { type: 'string' } },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const { network_data, propagation_request } = request.body;

      // Execute risk propagation analysis
      const propagationResult = await executePythonScript(
        ['analyze_risk_propagation'],
        {
          network_data,
          propagation_request
        }
      );

      if (!propagationResult.success) {
        throw new Error(propagationResult.error || 'Risk propagation analysis failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        propagation_id: propagationResult.propagation_id,
        source_nodes: propagationResult.source_nodes,
        affected_nodes: propagationResult.affected_nodes,
        propagation_paths: propagationResult.propagation_paths,
        total_risk_increase: propagationResult.total_risk_increase,
        propagation_time_steps: propagationResult.propagation_time_steps,
        convergence_achieved: propagationResult.convergence_achieved,
        critical_paths: propagationResult.critical_paths,
        mitigation_recommendations: propagationResult.mitigation_recommendations,
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Risk propagation analysis error: ${error.message}`);
      return reply.status(500).send({
        error: 'Risk propagation analysis failed',
        details: error.message
      });
    }
  });

  // Route: Simulate regulatory change impact
  fastify.post('/simulate-regulatory-impact', {
    schema: {
      description: 'Simulate the impact of regulatory changes on the compliance network',
      tags: ['Compliance Network Graph'],
      body: {
        type: 'object',
        required: ['current_network_data', 'regulatory_changes'],
        properties: {
          current_network_data: networkDataSchema,
          regulatory_changes: {
            type: 'object',
            properties: {
              new_regulations: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    id: { type: 'string' },
                    title: { type: 'string' },
                    severity: { type: 'number', minimum: 0, maximum: 1 },
                    affects: { type: 'array', items: { type: 'string' } },
                    risk_impact: { type: 'number', minimum: 0, maximum: 1 }
                  }
                }
              },
              regulation_updates: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    regulation_id: { type: 'string' },
                    new_severity: { type: 'number', minimum: 0, maximum: 1 }
                  }
                }
              }
            }
          }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            simulation_id: { type: 'string' },
            original_network_stats: { type: 'object' },
            changes_applied: { type: 'object' },
            impact_analysis: { type: 'object' },
            recommendations: { type: 'array', items: { type: 'string' } },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const { current_network_data, regulatory_changes } = request.body;

      // Execute regulatory impact simulation
      const simulationResult = await executePythonScript(
        ['simulate_regulatory_impact'],
        {
          current_network_data,
          regulatory_changes
        }
      );

      if (!simulationResult.success) {
        throw new Error(simulationResult.error || 'Regulatory impact simulation failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        simulation_id: simulationResult.simulation_id,
        original_network_stats: simulationResult.original_network_stats,
        changes_applied: simulationResult.changes_applied,
        impact_analysis: simulationResult.impact_analysis,
        recommendations: simulationResult.recommendations,
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Regulatory impact simulation error: ${error.message}`);
      return reply.status(500).send({
        error: 'Regulatory impact simulation failed',
        details: error.message
      });
    }
  });

  // Route: Get network statistics
  fastify.post('/statistics', {
    schema: {
      description: 'Get comprehensive statistics for a compliance network',
      tags: ['Compliance Network Graph'],
      body: networkDataSchema,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            node_count: { type: 'integer' },
            edge_count: { type: 'integer' },
            density: { type: 'number' },
            is_connected: { type: 'boolean' },
            node_type_distribution: { type: 'object' },
            edge_type_distribution: { type: 'object' },
            risk_statistics: { type: 'object' },
            degree_statistics: { type: 'object' },
            connected_components: { type: 'integer' },
            clustering_coefficient: { type: 'number' },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const networkData = request.body;

      // Execute network statistics calculation
      const statisticsResult = await executePythonScript(
        ['network_statistics'],
        { network_data: networkData }
      );

      if (!statisticsResult.success) {
        throw new Error(statisticsResult.error || 'Network statistics calculation failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        ...statisticsResult.statistics,
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Network statistics calculation error: ${error.message}`);
      return reply.status(500).send({
        error: 'Network statistics calculation failed',
        details: error.message
      });
    }
  });

  // Route: Detect network anomalies
  fastify.post('/detect-anomalies', {
    schema: {
      description: 'Detect anomalies in the compliance network',
      tags: ['Compliance Network Graph'],
      body: networkDataSchema,
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            anomalous_nodes: { type: 'array' },
            anomalous_edges: { type: 'array' },
            structural_anomalies: { type: 'array' },
            temporal_anomalies: { type: 'array' },
            anomaly_summary: { type: 'object' },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const networkData = request.body;

      // Execute anomaly detection
      const anomalyResult = await executePythonScript(
        ['detect_anomalies'],
        { network_data: networkData }
      );

      if (!anomalyResult.success) {
        throw new Error(anomalyResult.error || 'Anomaly detection failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        anomalous_nodes: anomalyResult.anomalous_nodes,
        anomalous_edges: anomalyResult.anomalous_edges,
        structural_anomalies: anomalyResult.structural_anomalies,
        temporal_anomalies: anomalyResult.temporal_anomalies,
        anomaly_summary: {
          total_anomalies: 
            (anomalyResult.anomalous_nodes?.length || 0) +
            (anomalyResult.anomalous_edges?.length || 0) +
            (anomalyResult.structural_anomalies?.length || 0) +
            (anomalyResult.temporal_anomalies?.length || 0),
          node_anomalies: anomalyResult.anomalous_nodes?.length || 0,
          edge_anomalies: anomalyResult.anomalous_edges?.length || 0,
          structural_anomalies: anomalyResult.structural_anomalies?.length || 0,
          temporal_anomalies: anomalyResult.temporal_anomalies?.length || 0
        },
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Anomaly detection error: ${error.message}`);
      return reply.status(500).send({
        error: 'Anomaly detection failed',
        details: error.message
      });
    }
  });

  // Route: Generate visualization data
  fastify.post('/visualization', {
    schema: {
      description: 'Generate visualization data for the compliance network',
      tags: ['Compliance Network Graph'],
      body: {
        type: 'object',
        required: ['network_data'],
        properties: {
          network_data: networkDataSchema,
          layout_algorithm: {
            type: 'string',
            enum: ['spring', 'circular', 'hierarchical', 'force_atlas'],
            default: 'spring'
          },
          include_centrality: { type: 'boolean', default: true },
          include_communities: { type: 'boolean', default: true },
          filter_options: {
            type: 'object',
            properties: {
              min_risk_score: { type: 'number', minimum: 0, maximum: 1 },
              node_types: { type: 'array', items: { type: 'string' } },
              edge_types: { type: 'array', items: { type: 'string' } }
            }
          }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            visualization_data: {
              type: 'object',
              properties: {
                nodes: { type: 'array' },
                edges: { type: 'array' },
                layout: { type: 'string' },
                statistics: { type: 'object' },
                legends: { type: 'object' }
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
      const { network_data, layout_algorithm = 'spring', include_centrality = true, include_communities = true, filter_options } = request.body;

      // Execute visualization data generation
      const visualizationResult = await executePythonScript(
        ['generate_visualization'],
        {
          network_data,
          layout_algorithm,
          include_centrality,
          include_communities,
          filter_options
        }
      );

      if (!visualizationResult.success) {
        throw new Error(visualizationResult.error || 'Visualization data generation failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        visualization_data: visualizationResult.visualization_data,
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Visualization data generation error: ${error.message}`);
      return reply.status(500).send({
        error: 'Visualization data generation failed',
        details: error.message
      });
    }
  });

  // Route: Find shortest paths between nodes
  fastify.post('/shortest-paths', {
    schema: {
      description: 'Find shortest paths between nodes in the compliance network',
      tags: ['Compliance Network Graph'],
      body: {
        type: 'object',
        required: ['network_data', 'source_nodes', 'target_nodes'],
        properties: {
          network_data: networkDataSchema,
          source_nodes: { type: 'array', items: { type: 'string' } },
          target_nodes: { type: 'array', items: { type: 'string' } },
          max_path_length: { type: 'integer', minimum: 1, maximum: 20, default: 10 },
          weight_attribute: { type: 'string', default: 'weight' }
        }
      },
      response: {
        200: {
          type: 'object',
          properties: {
            success: { type: 'boolean' },
            paths: { type: 'array' },
            path_statistics: { type: 'object' },
            processing_time: { type: 'number' }
          }
        }
      }
    }
  }, async (request, reply) => {
    const startTime = Date.now();
    
    try {
      const { network_data, source_nodes, target_nodes, max_path_length = 10, weight_attribute = 'weight' } = request.body;

      // Execute shortest path calculation
      const pathResult = await executePythonScript(
        ['find_shortest_paths'],
        {
          network_data,
          source_nodes,
          target_nodes,
          max_path_length,
          weight_attribute
        }
      );

      if (!pathResult.success) {
        throw new Error(pathResult.error || 'Shortest path calculation failed');
      }

      const processingTime = (Date.now() - startTime) / 1000;

      return reply.send({
        success: true,
        paths: pathResult.paths,
        path_statistics: pathResult.path_statistics,
        processing_time: processingTime
      });

    } catch (error) {
      fastify.log.error(`Shortest path calculation error: ${error.message}`);
      return reply.status(500).send({
        error: 'Shortest path calculation failed',
        details: error.message
      });
    }
  });

  // Route: Health check
  fastify.get('/health', {
    schema: {
      description: 'Health check for compliance network graph service',
      tags: ['Compliance Network Graph'],
      response: {
        200: {
          type: 'object',
          properties: {
            status: { type: 'string' },
            service: { type: 'string' },
            timestamp: { type: 'string', format: 'date-time' },
            python_available: { type: 'boolean' },
            cache_directory_writable: { type: 'boolean' }
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

      // Test cache directory
      let cacheDirectoryWritable = false;
      try {
        const testFile = path.join(ANALYSIS_CACHE_DIR, 'test.txt');
        await fs.writeFile(testFile, 'test');
        await fs.unlink(testFile);
        cacheDirectoryWritable = true;
      } catch (error) {
        fastify.log.warn(`Cache directory not writable: ${error.message}`);
      }

      return reply.send({
        status: 'healthy',
        service: 'compliance-network-graph',
        timestamp: new Date().toISOString(),
        python_available: pythonAvailable,
        cache_directory_writable: cacheDirectoryWritable
      });

    } catch (error) {
      fastify.log.error(`Health check error: ${error.message}`);
      return reply.status(500).send({
        status: 'unhealthy',
        service: 'compliance-network-graph',
        timestamp: new Date().toISOString(),
        error: error.message
      });
    }
  });

  // Add request/response logging
  fastify.addHook('onRequest', async (request, reply) => {
    fastify.log.info(`Compliance Network Graph API: ${request.method} ${request.url}`);
  });

  fastify.addHook('onResponse', async (request, reply) => {
    fastify.log.info(`Compliance Network Graph API Response: ${request.method} ${request.url} - ${reply.statusCode}`);
  });
}

module.exports = complianceNetworkGraphRoutes;
