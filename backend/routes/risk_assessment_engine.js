/**
 * Advanced Risk Assessment Engine API Routes
 * Production-grade Fastify routes for comprehensive risk analysis and management
 * 
 * Features:
 * - Multi-dimensional risk scoring with 15+ risk factors
 * - Advanced predictive modeling using ensemble machine learning
 * - Real-time risk monitoring with dynamic threshold adjustment
 * - Comprehensive mitigation strategy recommendations
 * - Stress testing and scenario analysis capabilities
 * - Portfolio risk aggregation and concentration analysis
 * - Advanced analytics and reporting
 */

const fastify = require('fastify')({ logger: true });
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const WebSocket = require('ws');
const crypto = require('crypto');

// Database and Redis connections
let dbPool, redisClient;

// WebSocket server for real-time updates
let wsServer;

// Python agent process
let pythonAgent = null;

// Performance metrics
const performanceMetrics = {
    assessments_performed: 0,
    stress_tests_executed: 0,
    models_trained: 0,
    alerts_generated: 0,
    avg_assessment_time: 0,
    assessment_times: []
};

/**
 * Initialize Risk Assessment Engine routes
 */
async function initializeRiskAssessmentEngineRoutes(app, database, redis) {
    dbPool = database;
    redisClient = redis;

    // Initialize WebSocket server
    wsServer = new WebSocket.Server({ port: 8013 });
    
    wsServer.on('connection', (ws) => {
        fastify.log.info('WebSocket client connected to Risk Assessment Engine');
        
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                handleWebSocketMessage(ws, data);
            } catch (error) {
                ws.send(JSON.stringify({ error: 'Invalid message format' }));
            }
        });
        
        ws.on('close', () => {
            fastify.log.info('WebSocket client disconnected from Risk Assessment Engine');
        });
    });

    // Start Python agent
    await startPythonAgent();

    // Register all routes
    await registerRiskAssessmentRoutes(app);
    await registerStressTestingRoutes(app);
    await registerMitigationRoutes(app);
    await registerPortfolioRiskRoutes(app);
    await registerModelManagementRoutes(app);
    await registerAlertManagementRoutes(app);
    await registerAnalyticsRoutes(app);
    await registerBenchmarkRoutes(app);
    await registerConfigurationRoutes(app);

    fastify.log.info('Risk Assessment Engine routes initialized');
}

/**
 * Start Python agent process
 */
async function startPythonAgent() {
    try {
        const agentPath = path.join(__dirname, '../agents/risk_assessment_engine/agent.py');
        
        pythonAgent = spawn('python3', [agentPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: {
                ...process.env,
                PYTHONPATH: path.join(__dirname, '../agents'),
                DATABASE_URL: process.env.DATABASE_URL,
                REDIS_URL: process.env.REDIS_URL
            }
        });

        pythonAgent.stdout.on('data', (data) => {
            fastify.log.info(`Risk Assessment Python Agent: ${data}`);
        });

        pythonAgent.stderr.on('data', (data) => {
            fastify.log.error(`Risk Assessment Python Agent Error: ${data}`);
        });

        pythonAgent.on('close', (code) => {
            fastify.log.warn(`Risk Assessment Python Agent exited with code ${code}`);
            // Auto-restart after 5 seconds
            setTimeout(startPythonAgent, 5000);
        });

        fastify.log.info('Risk Assessment Engine Python agent started');
    } catch (error) {
        fastify.log.error('Failed to start Risk Assessment Python agent:', error);
        throw error;
    }
}

/**
 * Call Python agent function
 */
async function callPythonAgent(functionName, params = {}) {
    return new Promise((resolve, reject) => {
        const requestId = crypto.randomUUID();
        const request = {
            id: requestId,
            function: functionName,
            params: params,
            timestamp: new Date().toISOString()
        };

        const timeout = setTimeout(() => {
            reject(new Error('Risk Assessment Python agent timeout'));
        }, 300000); // 5 minute timeout for complex calculations

        const responseHandler = (data) => {
            try {
                const response = JSON.parse(data.toString());
                if (response.id === requestId) {
                    clearTimeout(timeout);
                    pythonAgent.stdout.removeListener('data', responseHandler);
                    
                    if (response.error) {
                        reject(new Error(response.error));
                    } else {
                        resolve(response.result);
                    }
                }
            } catch (error) {
                // Not a JSON response, ignore
            }
        };

        pythonAgent.stdout.on('data', responseHandler);
        pythonAgent.stdin.write(JSON.stringify(request) + '\n');
    });
}

/**
 * Handle WebSocket messages
 */
function handleWebSocketMessage(ws, data) {
    switch (data.type) {
        case 'subscribe_risk_assessments':
            ws.riskSubscription = true;
            break;
        case 'subscribe_alerts':
            ws.alertSubscription = true;
            break;
        case 'subscribe_stress_tests':
            ws.stressTestSubscription = true;
            break;
        case 'get_real_time_dashboard':
            sendRealTimeDashboard(ws);
            break;
        default:
            ws.send(JSON.stringify({ error: 'Unknown message type' }));
    }
}

/**
 * Broadcast to WebSocket clients
 */
function broadcastToClients(data) {
    wsServer.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify(data));
        }
    });
}

/**
 * Send real-time dashboard data
 */
async function sendRealTimeDashboard(ws) {
    try {
        const dashboardData = await getRiskDashboardData();
        ws.send(JSON.stringify({
            type: 'real_time_dashboard',
            data: dashboardData,
            timestamp: new Date().toISOString()
        }));
    } catch (error) {
        ws.send(JSON.stringify({ error: 'Failed to get dashboard data' }));
    }
}

/**
 * Get risk dashboard data
 */
async function getRiskDashboardData() {
    try {
        const assessmentStats = await dbPool.query(`
            SELECT 
                risk_level,
                COUNT(*) as count,
                AVG(overall_risk_score) as avg_score
            FROM risk_assessments_advanced 
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY risk_level
        `);

        const alertStats = await dbPool.query(`
            SELECT 
                severity,
                COUNT(*) as count
            FROM risk_monitoring_alerts 
            WHERE alert_status = 'active'
            GROUP BY severity
        `);

        const portfolioStats = await dbPool.query(`
            SELECT 
                COUNT(DISTINCT entity_id) as total_entities,
                SUM(expected_loss) as total_expected_loss,
                AVG(value_at_risk) as avg_var
            FROM risk_assessments_advanced 
            WHERE created_at > NOW() - INTERVAL '30 days'
        `);

        return {
            assessments: assessmentStats.rows,
            alerts: alertStats.rows,
            portfolio: portfolioStats.rows[0] || {},
            system_metrics: performanceMetrics
        };
    } catch (error) {
        fastify.log.error('Failed to get dashboard data:', error);
        throw error;
    }
}

/**
 * Register risk assessment routes
 */
async function registerRiskAssessmentRoutes(app) {
    // Perform comprehensive risk assessment
    app.post('/api/risk-assessment/assess', async (request, reply) => {
        try {
            const { entity_id, entity_type, input_data } = request.body;

            if (!entity_id || !entity_type || !input_data) {
                reply.status(400).send({
                    success: false,
                    error: 'Entity ID, entity type, and input data are required'
                });
                return;
            }

            const startTime = Date.now();

            // Call Python agent for comprehensive assessment
            const assessment = await callPythonAgent('perform_comprehensive_risk_assessment', {
                entity_id,
                entity_type,
                input_data
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.assessments_performed++;
            performanceMetrics.assessment_times.push(processingTime);

            // Update average processing time
            if (performanceMetrics.assessment_times.length > 1000) {
                performanceMetrics.assessment_times = performanceMetrics.assessment_times.slice(-1000);
            }
            performanceMetrics.avg_assessment_time = 
                performanceMetrics.assessment_times.reduce((a, b) => a + b, 0) / 
                performanceMetrics.assessment_times.length;

            // Generate alerts if high risk
            if (assessment.overall_risk_score > 0.7) {
                await generateRiskAlert(assessment);
            }

            // Broadcast to WebSocket clients
            broadcastToClients({
                type: 'risk_assessment_completed',
                data: {
                    assessment_id: assessment.assessment_id,
                    entity_id,
                    risk_level: assessment.risk_level,
                    risk_score: assessment.overall_risk_score,
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                assessment,
                processing_time_ms: processingTime
            };

        } catch (error) {
            fastify.log.error('Risk assessment failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'Risk assessment failed'
            });
        }
    });

    // Get risk assessment by ID
    app.get('/api/risk-assessment/assessments/:assessmentId', async (request, reply) => {
        try {
            const { assessmentId } = request.params;

            const result = await dbPool.query(`
                SELECT * FROM risk_assessments_advanced WHERE assessment_id = $1
            `, [assessmentId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Risk assessment not found'
                });
                return;
            }

            const assessment = result.rows[0];

            // Get related mitigation strategies
            const strategiesResult = await dbPool.query(`
                SELECT * FROM risk_mitigation_strategies WHERE assessment_id = $1
            `, [assessmentId]);

            // Get related alerts
            const alertsResult = await dbPool.query(`
                SELECT * FROM risk_monitoring_alerts WHERE assessment_id = $1
            `, [assessmentId]);

            return {
                success: true,
                assessment: {
                    ...assessment,
                    mitigation_strategies: strategiesResult.rows,
                    monitoring_alerts: alertsResult.rows
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get risk assessment:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve risk assessment'
            });
        }
    });

    // Get risk assessments with filtering
    app.get('/api/risk-assessment/assessments', async (request, reply) => {
        try {
            const {
                entity_type,
                entity_id,
                risk_level,
                min_risk_score,
                max_risk_score,
                days = 30,
                page = 1,
                limit = 50
            } = request.query;

            let query = `
                SELECT assessment_id, entity_id, entity_type, overall_risk_score,
                       risk_level, predicted_default_probability, expected_loss,
                       data_quality_score, risk_appetite_alignment, created_at
                FROM risk_assessments_advanced
                WHERE created_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (entity_type) {
                query += ` AND entity_type = $${paramIndex}`;
                params.push(entity_type);
                paramIndex++;
            }

            if (entity_id) {
                query += ` AND entity_id = $${paramIndex}`;
                params.push(entity_id);
                paramIndex++;
            }

            if (risk_level) {
                query += ` AND risk_level = $${paramIndex}`;
                params.push(risk_level);
                paramIndex++;
            }

            if (min_risk_score) {
                query += ` AND overall_risk_score >= $${paramIndex}`;
                params.push(parseFloat(min_risk_score));
                paramIndex++;
            }

            if (max_risk_score) {
                query += ` AND overall_risk_score <= $${paramIndex}`;
                params.push(parseFloat(max_risk_score));
                paramIndex++;
            }

            query += ` ORDER BY created_at DESC`;
            query += ` LIMIT $${paramIndex} OFFSET $${paramIndex + 1}`;
            params.push(parseInt(limit));
            params.push((parseInt(page) - 1) * parseInt(limit));

            const result = await dbPool.query(query, params);

            return {
                success: true,
                assessments: result.rows,
                pagination: {
                    page: parseInt(page),
                    limit: parseInt(limit)
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get risk assessments:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve risk assessments'
            });
        }
    });

    // Update risk assessment
    app.put('/api/risk-assessment/assessments/:assessmentId', async (request, reply) => {
        try {
            const { assessmentId } = request.params;
            const updates = request.body;

            const allowedUpdates = ['risk_appetite_alignment', 'next_review_date', 'monitoring_alerts'];
            const updateFields = [];
            const updateValues = [];
            let paramIndex = 1;

            for (const [key, value] of Object.entries(updates)) {
                if (allowedUpdates.includes(key)) {
                    if (typeof value === 'object') {
                        updateFields.push(`${key} = $${paramIndex}`);
                        updateValues.push(JSON.stringify(value));
                    } else {
                        updateFields.push(`${key} = $${paramIndex}`);
                        updateValues.push(value);
                    }
                    paramIndex++;
                }
            }

            if (updateFields.length === 0) {
                reply.status(400).send({
                    success: false,
                    error: 'No valid update fields provided'
                });
                return;
            }

            updateFields.push('updated_at = CURRENT_TIMESTAMP');
            updateValues.push(assessmentId);

            const updateQuery = `
                UPDATE risk_assessments_advanced
                SET ${updateFields.join(', ')}
                WHERE assessment_id = $${paramIndex}
                RETURNING *
            `;

            const result = await dbPool.query(updateQuery, updateValues);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Risk assessment not found'
                });
                return;
            }

            // Create audit record
            await dbPool.query(`
                INSERT INTO risk_assessment_audit (
                    assessment_id, audit_action, new_values, changed_by, change_reason
                ) VALUES ($1, $2, $3, $4, $5)
            `, [
                assessmentId, 'assessment_updated', JSON.stringify(updates),
                request.user?.id || 'system', 'API update request'
            ]);

            return {
                success: true,
                assessment: result.rows[0],
                message: 'Risk assessment updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update risk assessment:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update risk assessment'
            });
        }
    });
}

/**
 * Register stress testing routes
 */
async function registerStressTestingRoutes(app) {
    // Execute stress test
    app.post('/api/risk-assessment/stress-test', async (request, reply) => {
        try {
            const { assessment_id, scenario_ids, custom_scenarios } = request.body;

            if (!assessment_id && !custom_scenarios) {
                reply.status(400).send({
                    success: false,
                    error: 'Assessment ID or custom scenarios are required'
                });
                return;
            }

            const startTime = Date.now();

            // Get assessment data
            let assessmentData = null;
            if (assessment_id) {
                const assessmentResult = await dbPool.query(`
                    SELECT * FROM risk_assessments_advanced WHERE assessment_id = $1
                `, [assessment_id]);

                if (assessmentResult.rows.length === 0) {
                    reply.status(404).send({
                        success: false,
                        error: 'Risk assessment not found'
                    });
                    return;
                }
                assessmentData = assessmentResult.rows[0];
            }

            // Get stress test scenarios
            let scenarios = [];
            if (scenario_ids && scenario_ids.length > 0) {
                const scenariosResult = await dbPool.query(`
                    SELECT * FROM stress_test_scenarios 
                    WHERE scenario_id = ANY($1) AND is_active = true
                `, [scenario_ids]);
                scenarios = scenariosResult.rows;
            }

            if (custom_scenarios) {
                scenarios = scenarios.concat(custom_scenarios);
            }

            // Execute stress test
            const stressTestResults = await callPythonAgent('execute_stress_test', {
                assessment_data: assessmentData,
                scenarios
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.stress_tests_executed++;

            // Store stress test results
            for (const result of stressTestResults) {
                await dbPool.query(`
                    INSERT INTO stress_test_results (
                        test_id, assessment_id, scenario_id, baseline_risk_score,
                        stressed_risk_score, risk_impact, baseline_default_probability,
                        stressed_default_probability, expected_loss_baseline,
                        expected_loss_stressed, capital_impact, test_duration_ms, model_version
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                `, [
                    crypto.randomUUID(), assessment_id, result.scenario_id,
                    result.baseline_risk_score, result.stressed_risk_score, result.risk_impact,
                    result.baseline_default_probability, result.stressed_default_probability,
                    result.expected_loss_baseline, result.expected_loss_stressed,
                    result.capital_impact, processingTime, '2025.1'
                ]);
            }

            // Broadcast results
            broadcastToClients({
                type: 'stress_test_completed',
                data: {
                    assessment_id,
                    scenarios_tested: scenarios.length,
                    max_impact: Math.max(...stressTestResults.map(r => r.risk_impact || 0)),
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                stress_test_results: stressTestResults,
                processing_time_ms: processingTime,
                scenarios_tested: scenarios.length
            };

        } catch (error) {
            fastify.log.error('Stress test execution failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'Stress test execution failed'
            });
        }
    });

    // Get stress test scenarios
    app.get('/api/risk-assessment/stress-scenarios', async (request, reply) => {
        try {
            const { scenario_type, severity, active_only = true } = request.query;

            let query = `
                SELECT * FROM stress_test_scenarios
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (scenario_type) {
                query += ` AND scenario_type = $${paramIndex}`;
                params.push(scenario_type);
                paramIndex++;
            }

            if (severity) {
                query += ` AND severity = $${paramIndex}`;
                params.push(severity);
                paramIndex++;
            }

            if (active_only === 'true') {
                query += ` AND is_active = true`;
            }

            query += ' ORDER BY severity DESC, probability DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                scenarios: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get stress scenarios:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve stress scenarios'
            });
        }
    });

    // Create custom stress scenario
    app.post('/api/risk-assessment/stress-scenarios', async (request, reply) => {
        try {
            const scenarioData = request.body;

            const requiredFields = ['name', 'description', 'scenario_type', 'severity', 'parameters', 'impact_factors'];
            for (const field of requiredFields) {
                if (!scenarioData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            const scenarioId = crypto.randomUUID();
            const result = await dbPool.query(`
                INSERT INTO stress_test_scenarios (
                    scenario_id, name, description, scenario_type, severity, probability,
                    parameters, impact_factors, duration_months, recovery_assumptions, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING *
            `, [
                scenarioId, scenarioData.name, scenarioData.description,
                scenarioData.scenario_type, scenarioData.severity, scenarioData.probability || 0.1,
                JSON.stringify(scenarioData.parameters), JSON.stringify(scenarioData.impact_factors),
                scenarioData.duration_months || 12, JSON.stringify(scenarioData.recovery_assumptions || {}),
                request.user?.id || 'api_user'
            ]);

            return {
                success: true,
                scenario: result.rows[0],
                message: 'Stress test scenario created successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to create stress scenario:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to create stress scenario'
            });
        }
    });
}

/**
 * Register mitigation strategy routes
 */
async function registerMitigationRoutes(app) {
    // Get mitigation recommendations
    app.get('/api/risk-assessment/mitigation/:assessmentId', async (request, reply) => {
        try {
            const { assessmentId } = request.params;
            const { budget_constraint } = request.query;

            const result = await dbPool.query(`
                SELECT * FROM risk_mitigation_strategies 
                WHERE assessment_id = $1
                ORDER BY effectiveness_score DESC, expected_risk_reduction DESC
            `, [assessmentId]);

            let strategies = result.rows;

            // Apply budget constraint if specified
            if (budget_constraint) {
                const budget = parseFloat(budget_constraint);
                let totalCost = 0;
                strategies = strategies.filter(strategy => {
                    if (totalCost + strategy.implementation_cost <= budget) {
                        totalCost += strategy.implementation_cost;
                        return true;
                    }
                    return false;
                });
            }

            return {
                success: true,
                mitigation_strategies: strategies,
                total_strategies: strategies.length,
                total_cost: strategies.reduce((sum, s) => sum + (s.implementation_cost || 0), 0),
                expected_risk_reduction: strategies.reduce((sum, s) => sum + (s.expected_risk_reduction || 0), 0)
            };

        } catch (error) {
            fastify.log.error('Failed to get mitigation strategies:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve mitigation strategies'
            });
        }
    });

    // Update mitigation strategy implementation
    app.put('/api/risk-assessment/mitigation/:strategyId', async (request, reply) => {
        try {
            const { strategyId } = request.params;
            const updates = request.body;

            const allowedUpdates = [
                'implementation_status', 'implementation_start_date', 'implementation_completion_date',
                'actual_risk_reduction', 'cost_effectiveness_ratio'
            ];

            const updateFields = [];
            const updateValues = [];
            let paramIndex = 1;

            for (const [key, value] of Object.entries(updates)) {
                if (allowedUpdates.includes(key)) {
                    updateFields.push(`${key} = $${paramIndex}`);
                    updateValues.push(value);
                    paramIndex++;
                }
            }

            if (updateFields.length === 0) {
                reply.status(400).send({
                    success: false,
                    error: 'No valid update fields provided'
                });
                return;
            }

            updateFields.push('updated_at = CURRENT_TIMESTAMP');
            updateValues.push(strategyId);

            const updateQuery = `
                UPDATE risk_mitigation_strategies
                SET ${updateFields.join(', ')}
                WHERE strategy_id = $${paramIndex}
                RETURNING *
            `;

            const result = await dbPool.query(updateQuery, updateValues);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Mitigation strategy not found'
                });
                return;
            }

            return {
                success: true,
                strategy: result.rows[0],
                message: 'Mitigation strategy updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update mitigation strategy:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update mitigation strategy'
            });
        }
    });
}

/**
 * Register portfolio risk routes
 */
async function registerPortfolioRiskRoutes(app) {
    // Calculate portfolio risk metrics
    app.post('/api/risk-assessment/portfolio-risk', async (request, reply) => {
        try {
            const { portfolio_id, entity_ids, aggregation_method = 'weighted_average' } = request.body;

            if (!portfolio_id && !entity_ids) {
                reply.status(400).send({
                    success: false,
                    error: 'Portfolio ID or entity IDs are required'
                });
                return;
            }

            const startTime = Date.now();

            // Get individual assessments
            let assessments = [];
            if (entity_ids) {
                const result = await dbPool.query(`
                    SELECT * FROM risk_assessments_advanced 
                    WHERE entity_id = ANY($1)
                    ORDER BY created_at DESC
                `, [entity_ids]);
                assessments = result.rows;
            } else {
                // Use portfolio function
                const portfolioResult = await dbPool.query(`
                    SELECT * FROM calculate_portfolio_risk_metrics($1)
                `, [portfolio_id]);
                
                return {
                    success: true,
                    portfolio_metrics: portfolioResult.rows[0]
                };
            }

            // Calculate portfolio risk using Python agent
            const portfolioRisk = await callPythonAgent('calculate_portfolio_risk', {
                assessments,
                aggregation_method
            });

            const processingTime = Date.now() - startTime;

            // Store portfolio aggregation
            if (portfolio_id) {
                await dbPool.query(`
                    INSERT INTO risk_portfolio_aggregation (
                        portfolio_id, portfolio_name, aggregation_date, total_exposure,
                        weighted_average_risk_score, portfolio_default_probability,
                        portfolio_expected_loss, portfolio_var_99, number_of_exposures
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                `, [
                    portfolio_id, `Portfolio ${portfolio_id}`, new Date(),
                    portfolioRisk.total_exposure, portfolioRisk.weighted_average_risk_score,
                    portfolioRisk.portfolio_default_probability, portfolioRisk.portfolio_expected_loss,
                    portfolioRisk.portfolio_var_99, assessments.length
                ]);
            }

            return {
                success: true,
                portfolio_risk: portfolioRisk,
                processing_time_ms: processingTime,
                assessments_analyzed: assessments.length
            };

        } catch (error) {
            fastify.log.error('Portfolio risk calculation failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Portfolio risk calculation failed'
            });
        }
    });

    // Get portfolio risk history
    app.get('/api/risk-assessment/portfolio-history/:portfolioId', async (request, reply) => {
        try {
            const { portfolioId } = request.params;
            const { days = 90 } = request.query;

            const result = await dbPool.query(`
                SELECT * FROM risk_portfolio_aggregation
                WHERE portfolio_id = $1 
                AND aggregation_date > NOW() - INTERVAL '${days} days'
                ORDER BY aggregation_date DESC
            `, [portfolioId]);

            return {
                success: true,
                portfolio_history: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get portfolio history:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve portfolio history'
            });
        }
    });
}

/**
 * Register model management routes
 */
async function registerModelManagementRoutes(app) {
    // Train risk models
    app.post('/api/risk-assessment/train-models', async (request, reply) => {
        try {
            const { model_types, training_parameters } = request.body;

            const startTime = Date.now();

            const trainingResults = await callPythonAgent('train_risk_models', {
                model_types: model_types || ['credit_scoring'],
                parameters: training_parameters || {}
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.models_trained++;

            // Store model performance
            for (const [modelType, performance] of Object.entries(trainingResults)) {
                if (!performance.error) {
                    await dbPool.query(`
                        INSERT INTO risk_model_performance (
                            model_type, model_version, training_date, training_samples,
                            validation_samples, mse, mae, r2_score, rmse, feature_importance
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    `, [
                        modelType, '2025.1', new Date(), performance.training_samples || 0,
                        performance.validation_samples || 0, performance.mse, performance.mae,
                        performance.r2, performance.rmse, JSON.stringify(performance.feature_importance || {})
                    ]);
                }
            }

            return {
                success: true,
                training_results: trainingResults,
                processing_time_ms: processingTime
            };

        } catch (error) {
            fastify.log.error('Model training failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Model training failed'
            });
        }
    });

    // Get model performance
    app.get('/api/risk-assessment/model-performance', async (request, reply) => {
        try {
            const { model_type } = request.query;

            let query = `
                SELECT * FROM risk_model_performance
                WHERE is_active = true
            `;
            const params = [];

            if (model_type) {
                query += ' AND model_type = $1';
                params.push(model_type);
            }

            query += ' ORDER BY training_date DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                model_performance: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get model performance:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve model performance'
            });
        }
    });
}

/**
 * Register alert management routes
 */
async function registerAlertManagementRoutes(app) {
    // Get risk alerts
    app.get('/api/risk-assessment/alerts', async (request, reply) => {
        try {
            const { severity, status, days = 7 } = request.query;

            let query = `
                SELECT * FROM active_risk_alerts
                WHERE created_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (severity) {
                query += ` AND severity = $${paramIndex}`;
                params.push(severity);
                paramIndex++;
            }

            if (status) {
                query += ` AND alert_status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            const result = await dbPool.query(query, params);

            return {
                success: true,
                alerts: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get risk alerts:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve risk alerts'
            });
        }
    });

    // Acknowledge alert
    app.put('/api/risk-assessment/alerts/:alertId/acknowledge', async (request, reply) => {
        try {
            const { alertId } = request.params;
            const { acknowledgment_notes } = request.body;

            const result = await dbPool.query(`
                UPDATE risk_monitoring_alerts
                SET alert_status = 'acknowledged', acknowledged_by = $1, 
                    acknowledged_at = CURRENT_TIMESTAMP, resolution_notes = $2
                WHERE alert_id = $3
                RETURNING *
            `, [request.user?.id || 'system', acknowledgment_notes, alertId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Alert not found'
                });
                return;
            }

            return {
                success: true,
                alert: result.rows[0],
                message: 'Alert acknowledged successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to acknowledge alert:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to acknowledge alert'
            });
        }
    });

    // Generate risk alerts manually
    app.post('/api/risk-assessment/generate-alerts', async (request, reply) => {
        try {
            const alertCount = await dbPool.query('SELECT generate_risk_alerts()');
            const generatedAlerts = alertCount.rows[0].generate_risk_alerts;

            performanceMetrics.alerts_generated += generatedAlerts;

            return {
                success: true,
                alerts_generated: generatedAlerts,
                message: `Generated ${generatedAlerts} new risk alerts`
            };

        } catch (error) {
            fastify.log.error('Failed to generate alerts:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to generate alerts'
            });
        }
    });
}

/**
 * Register analytics routes
 */
async function registerAnalyticsRoutes(app) {
    // Get risk analytics dashboard
    app.get('/api/risk-assessment/analytics/dashboard', async (request, reply) => {
        try {
            const { days = 30 } = request.query;

            const dashboardData = await dbPool.query(`
                SELECT * FROM risk_dashboard_summary
                WHERE assessment_date >= CURRENT_DATE - INTERVAL '${days} days'
            `);

            const trendData = await dbPool.query(`
                SELECT 
                    DATE(created_at) as date,
                    AVG(overall_risk_score) as avg_risk_score,
                    COUNT(*) as assessment_count
                FROM risk_assessments_advanced
                WHERE created_at >= NOW() - INTERVAL '${days} days'
                GROUP BY DATE(created_at)
                ORDER BY date
            `);

            const categoryAnalysis = await dbPool.query(`
                SELECT 
                    jsonb_object_keys(category_scores) as risk_category,
                    AVG(CAST(jsonb_extract_path_text(category_scores, jsonb_object_keys(category_scores)) AS DECIMAL)) as avg_score
                FROM risk_assessments_advanced
                WHERE created_at >= NOW() - INTERVAL '${days} days'
                GROUP BY jsonb_object_keys(category_scores)
            `);

            return {
                success: true,
                analytics: {
                    dashboard_summary: dashboardData.rows,
                    trend_data: trendData.rows,
                    category_analysis: categoryAnalysis.rows,
                    system_metrics: performanceMetrics
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get risk analytics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve risk analytics'
            });
        }
    });

    // Get risk correlation analysis
    app.get('/api/risk-assessment/analytics/correlations', async (request, reply) => {
        try {
            const result = await dbPool.query(`
                SELECT * FROM risk_correlation_matrix
                WHERE is_current = true
                ORDER BY ABS(correlation_coefficient) DESC
            `);

            return {
                success: true,
                correlations: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get risk correlations:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve risk correlations'
            });
        }
    });
}

/**
 * Register benchmark management routes
 */
async function registerBenchmarkRoutes(app) {
    // Get risk factor benchmarks
    app.get('/api/risk-assessment/benchmarks', async (request, reply) => {
        try {
            const { factor_id, benchmark_type } = request.query;

            let query = `
                SELECT * FROM risk_factor_benchmarks
                WHERE is_current = true
            `;
            const params = [];
            let paramIndex = 1;

            if (factor_id) {
                query += ` AND factor_id = $${paramIndex}`;
                params.push(factor_id);
                paramIndex++;
            }

            if (benchmark_type) {
                query += ` AND benchmark_type = $${paramIndex}`;
                params.push(benchmark_type);
                paramIndex++;
            }

            query += ' ORDER BY factor_id, benchmark_type';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                benchmarks: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get benchmarks:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve benchmarks'
            });
        }
    });

    // Update benchmark
    app.put('/api/risk-assessment/benchmarks/:factorId', async (request, reply) => {
        try {
            const { factorId } = request.params;
            const { benchmark_type, benchmark_value, benchmark_source } = request.body;

            if (!benchmark_type || benchmark_value === undefined) {
                reply.status(400).send({
                    success: false,
                    error: 'Benchmark type and value are required'
                });
                return;
            }

            // Deactivate current benchmark
            await dbPool.query(`
                UPDATE risk_factor_benchmarks 
                SET is_current = false 
                WHERE factor_id = $1 AND benchmark_type = $2
            `, [factorId, benchmark_type]);

            // Insert new benchmark
            const result = await dbPool.query(`
                INSERT INTO risk_factor_benchmarks (
                    factor_id, benchmark_type, benchmark_value, benchmark_source,
                    effective_date, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING *
            `, [
                factorId, benchmark_type, benchmark_value, benchmark_source,
                new Date(), request.user?.id || 'system'
            ]);

            return {
                success: true,
                benchmark: result.rows[0],
                message: 'Benchmark updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update benchmark:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update benchmark'
            });
        }
    });
}

/**
 * Register configuration routes
 */
async function registerConfigurationRoutes(app) {
    // Get risk appetite configuration
    app.get('/api/risk-assessment/config/risk-appetite', async (request, reply) => {
        try {
            const result = await dbPool.query(`
                SELECT * FROM risk_appetite_config
                WHERE is_active = true
                ORDER BY risk_category, threshold_type
            `);

            return {
                success: true,
                risk_appetite_config: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get risk appetite config:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve risk appetite configuration'
            });
        }
    });

    // Update risk appetite threshold
    app.put('/api/risk-assessment/config/risk-appetite/:configId', async (request, reply) => {
        try {
            const { configId } = request.params;
            const { threshold_value, business_justification } = request.body;

            if (threshold_value === undefined) {
                reply.status(400).send({
                    success: false,
                    error: 'Threshold value is required'
                });
                return;
            }

            const result = await dbPool.query(`
                UPDATE risk_appetite_config
                SET threshold_value = $1, business_justification = $2, updated_at = CURRENT_TIMESTAMP
                WHERE id = $3
                RETURNING *
            `, [threshold_value, business_justification, configId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Risk appetite configuration not found'
                });
                return;
            }

            return {
                success: true,
                config: result.rows[0],
                message: 'Risk appetite threshold updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update risk appetite config:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update risk appetite configuration'
            });
        }
    });
}

/**
 * Generate risk alert
 */
async function generateRiskAlert(assessment) {
    try {
        const alertId = crypto.randomUUID();
        
        await dbPool.query(`
            INSERT INTO risk_monitoring_alerts (
                alert_id, assessment_id, alert_type, severity, title, description,
                threshold_breached, current_value, recommended_actions, escalation_level
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        `, [
            alertId, assessment.assessment_id, 'high_risk_assessment',
            assessment.risk_level === 'critical' ? 'critical' : 'high',
            `High Risk Assessment: ${assessment.entity_id}`,
            `Risk assessment shows ${assessment.risk_level} risk level requiring immediate attention`,
            0.7, assessment.overall_risk_score,
            JSON.stringify(assessment.mitigation_recommendations.slice(0, 3)),
            assessment.overall_risk_score > 0.9 ? 3 : 2
        ]);

        // Broadcast alert
        broadcastToClients({
            type: 'risk_alert_generated',
            data: {
                alert_id: alertId,
                assessment_id: assessment.assessment_id,
                severity: assessment.risk_level === 'critical' ? 'critical' : 'high',
                risk_score: assessment.overall_risk_score
            }
        });

        performanceMetrics.alerts_generated++;

    } catch (error) {
        fastify.log.error('Failed to generate risk alert:', error);
    }
}

module.exports = {
    initializeRiskAssessmentEngineRoutes
};