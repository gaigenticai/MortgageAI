/**
 * Advanced Lender Integration Manager API Routes
 * Production-grade Fastify routes for comprehensive lender integration
 * 
 * Features:
 * - AI-powered approval likelihood prediction
 * - Real-time lender health monitoring
 * - Advanced validation rules engine
 * - Circuit breaker patterns
 * - Rate limiting and load balancing
 * - Comprehensive audit trails
 * - Multi-lender optimization
 * - Real-time notifications
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

/**
 * Initialize Advanced Lender Integration Manager routes
 */
async function initializeLenderIntegrationRoutes(app, database, redis) {
    dbPool = database;
    redisClient = redis;

    // Initialize WebSocket server
    wsServer = new WebSocket.Server({ port: 8010 });
    
    wsServer.on('connection', (ws) => {
        fastify.log.info('WebSocket client connected to Advanced Lender Integration');
        
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                handleWebSocketMessage(ws, data);
            } catch (error) {
                ws.send(JSON.stringify({ error: 'Invalid message format' }));
            }
        });
        
        ws.on('close', () => {
            fastify.log.info('WebSocket client disconnected from Advanced Lender Integration');
        });
    });

    // Start Python agent
    await startPythonAgent();

    // Register all routes
    await registerLenderConfigurationRoutes(app);
    await registerSubmissionRoutes(app);
    await registerValidationRoutes(app);
    await registerPredictionRoutes(app);
    await registerHealthMonitoringRoutes(app);
    await registerOptimizationRoutes(app);
    await registerTrainingRoutes(app);
    await registerReportingRoutes(app);
    await registerNotificationRoutes(app);

    fastify.log.info('Advanced Lender Integration Manager routes initialized');
}

/**
 * Start Python agent process
 */
async function startPythonAgent() {
    try {
        const agentPath = path.join(__dirname, '../agents/lender_integration_manager/agent.py');
        
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
            fastify.log.info(`Python Agent: ${data}`);
        });

        pythonAgent.stderr.on('data', (data) => {
            fastify.log.error(`Python Agent Error: ${data}`);
        });

        pythonAgent.on('close', (code) => {
            fastify.log.warn(`Python Agent exited with code ${code}`);
            // Auto-restart after 5 seconds
            setTimeout(startPythonAgent, 5000);
        });

        fastify.log.info('Advanced Lender Integration Python agent started');
    } catch (error) {
        fastify.log.error('Failed to start Python agent:', error);
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

        // Set up response handler
        const timeout = setTimeout(() => {
            reject(new Error('Python agent timeout'));
        }, 60000); // 60 second timeout

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
        case 'subscribe_health':
            // Subscribe to health updates
            ws.healthSubscription = true;
            break;
        case 'subscribe_submissions':
            // Subscribe to submission updates
            ws.submissionSubscription = true;
            break;
        case 'get_real_time_metrics':
            // Send current metrics
            sendRealTimeMetrics(ws);
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
 * Send real-time metrics to WebSocket client
 */
async function sendRealTimeMetrics(ws) {
    try {
        const healthMetrics = await dbPool.query(`
            SELECT lender_name, status, response_time_ms, success_rate, 
                   error_rate, consecutive_failures, circuit_breaker_open
            FROM lender_health_metrics 
            ORDER BY lender_name
        `);

        const submissionStats = await dbPool.query(`
            SELECT lender_name, status, COUNT(*) as count
            FROM lender_submissions_advanced 
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY lender_name, status
            ORDER BY lender_name, status
        `);

        ws.send(JSON.stringify({
            type: 'real_time_metrics',
            data: {
                health_metrics: healthMetrics.rows,
                submission_stats: submissionStats.rows,
                timestamp: new Date().toISOString()
            }
        }));
    } catch (error) {
        ws.send(JSON.stringify({ error: 'Failed to get real-time metrics' }));
    }
}

/**
 * Register lender configuration routes
 */
async function registerLenderConfigurationRoutes(app) {
    // Get all lender configurations
    app.get('/api/lender-integration-advanced/lenders', async (request, reply) => {
        try {
            const result = await dbPool.query(`
                SELECT id, name, supported_products, max_loan_amount, min_loan_amount,
                       max_ltv, min_income, processing_time_days, priority_score,
                       fees, is_active, created_at, updated_at
                FROM lender_configurations
                WHERE is_active = true
                ORDER BY priority_score DESC, name
            `);

            return {
                success: true,
                lenders: result.rows,
                total: result.rows.length
            };
        } catch (error) {
            fastify.log.error('Failed to get lender configurations:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve lender configurations'
            });
        }
    });

    // Get specific lender configuration
    app.get('/api/lender-integration-advanced/lenders/:lenderName', async (request, reply) => {
        try {
            const { lenderName } = request.params;
            
            const result = await dbPool.query(`
                SELECT * FROM lender_configurations
                WHERE LOWER(name) = LOWER($1) AND is_active = true
            `, [lenderName]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Lender configuration not found'
                });
                return;
            }

            // Remove sensitive data
            const config = result.rows[0];
            delete config.api_key_encrypted;
            delete config.backup_api_key_encrypted;

            return {
                success: true,
                lender: config
            };
        } catch (error) {
            fastify.log.error('Failed to get lender configuration:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve lender configuration'
            });
        }
    });

    // Add new lender configuration
    app.post('/api/lender-integration-advanced/lenders', async (request, reply) => {
        try {
            const config = request.body;
            
            // Validate required fields
            const requiredFields = ['name', 'api_url', 'api_key', 'supported_products', 
                                   'max_loan_amount', 'min_loan_amount', 'max_ltv', 'min_income'];
            
            for (const field of requiredFields) {
                if (!config[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            // Encrypt API keys
            const cipher = crypto.createCipher('aes-256-cbc', process.env.ENCRYPTION_KEY || 'default-key');
            const apiKeyEncrypted = cipher.update(config.api_key, 'utf8', 'hex') + cipher.final('hex');
            
            let backupApiKeyEncrypted = null;
            if (config.backup_api_key) {
                const backupCipher = crypto.createCipher('aes-256-cbc', process.env.ENCRYPTION_KEY || 'default-key');
                backupApiKeyEncrypted = backupCipher.update(config.backup_api_key, 'utf8', 'hex') + backupCipher.final('hex');
            }

            const result = await dbPool.query(`
                INSERT INTO lender_configurations (
                    name, api_url, api_key_encrypted, backup_api_url, backup_api_key_encrypted,
                    supported_products, max_loan_amount, min_loan_amount, max_ltv, min_income,
                    processing_time_days, rate_limit_per_hour, timeout_seconds, retry_attempts,
                    circuit_breaker_threshold, priority_score, fees, requirements,
                    validation_rules, approval_criteria, document_requirements,
                    notification_webhooks, ssl_verify, custom_headers
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24
                ) RETURNING id, name, created_at
            `, [
                config.name, config.api_url, apiKeyEncrypted, config.backup_api_url, backupApiKeyEncrypted,
                JSON.stringify(config.supported_products), config.max_loan_amount, config.min_loan_amount,
                config.max_ltv, config.min_income, config.processing_time_days || 7,
                config.rate_limit_per_hour || 100, config.timeout_seconds || 60, config.retry_attempts || 3,
                config.circuit_breaker_threshold || 5, config.priority_score || 5,
                JSON.stringify(config.fees || {}), JSON.stringify(config.requirements || {}),
                JSON.stringify(config.validation_rules || []), JSON.stringify(config.approval_criteria || {}),
                JSON.stringify(config.document_requirements || []), JSON.stringify(config.notification_webhooks || []),
                config.ssl_verify !== false, JSON.stringify(config.custom_headers || {})
            ]);

            // Initialize health metrics
            await dbPool.query(`
                INSERT INTO lender_health_metrics (
                    lender_name, status, response_time_ms, success_rate, error_rate,
                    uptime_percentage, rate_limit_remaining
                ) VALUES ($1, 'active', 0, 1.0, 0.0, 100.0, $2)
            `, [config.name, config.rate_limit_per_hour || 100]);

            // Notify Python agent about new lender
            try {
                await callPythonAgent('add_lender', config);
            } catch (error) {
                fastify.log.warn('Failed to notify Python agent about new lender:', error);
            }

            return {
                success: true,
                lender: result.rows[0],
                message: 'Lender configuration added successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to add lender configuration:', error);
            
            if (error.code === '23505') { // Unique constraint violation
                reply.status(409).send({
                    success: false,
                    error: 'Lender with this name already exists'
                });
            } else {
                reply.status(500).send({
                    success: false,
                    error: 'Failed to add lender configuration'
                });
            }
        }
    });

    // Update lender configuration
    app.put('/api/lender-integration-advanced/lenders/:lenderName', async (request, reply) => {
        try {
            const { lenderName } = request.params;
            const updates = request.body;

            // Get current configuration
            const currentResult = await dbPool.query(`
                SELECT * FROM lender_configurations
                WHERE LOWER(name) = LOWER($1)
            `, [lenderName]);

            if (currentResult.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Lender configuration not found'
                });
                return;
            }

            // Build update query dynamically
            const updateFields = [];
            const updateValues = [];
            let paramIndex = 1;

            for (const [key, value] of Object.entries(updates)) {
                if (key === 'api_key') {
                    const cipher = crypto.createCipher('aes-256-cbc', process.env.ENCRYPTION_KEY || 'default-key');
                    const encrypted = cipher.update(value, 'utf8', 'hex') + cipher.final('hex');
                    updateFields.push(`api_key_encrypted = $${paramIndex}`);
                    updateValues.push(encrypted);
                } else if (key === 'backup_api_key') {
                    const cipher = crypto.createCipher('aes-256-cbc', process.env.ENCRYPTION_KEY || 'default-key');
                    const encrypted = cipher.update(value, 'utf8', 'hex') + cipher.final('hex');
                    updateFields.push(`backup_api_key_encrypted = $${paramIndex}`);
                    updateValues.push(encrypted);
                } else if (typeof value === 'object') {
                    updateFields.push(`${key} = $${paramIndex}`);
                    updateValues.push(JSON.stringify(value));
                } else {
                    updateFields.push(`${key} = $${paramIndex}`);
                    updateValues.push(value);
                }
                paramIndex++;
            }

            updateFields.push('updated_at = CURRENT_TIMESTAMP');
            updateValues.push(lenderName);

            const updateQuery = `
                UPDATE lender_configurations
                SET ${updateFields.join(', ')}
                WHERE LOWER(name) = LOWER($${paramIndex})
                RETURNING id, name, updated_at
            `;

            const result = await dbPool.query(updateQuery, updateValues);

            return {
                success: true,
                lender: result.rows[0],
                message: 'Lender configuration updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update lender configuration:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update lender configuration'
            });
        }
    });

    // Delete lender configuration (soft delete)
    app.delete('/api/lender-integration-advanced/lenders/:lenderName', async (request, reply) => {
        try {
            const { lenderName } = request.params;

            const result = await dbPool.query(`
                UPDATE lender_configurations
                SET is_active = false, updated_at = CURRENT_TIMESTAMP
                WHERE LOWER(name) = LOWER($1)
                RETURNING id, name
            `, [lenderName]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Lender configuration not found'
                });
                return;
            }

            return {
                success: true,
                message: 'Lender configuration deactivated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to delete lender configuration:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to delete lender configuration'
            });
        }
    });
}

/**
 * Register application submission routes
 */
async function registerSubmissionRoutes(app) {
    // Submit application to lender
    app.post('/api/lender-integration-advanced/submit', async (request, reply) => {
        try {
            const { lender_name, application_data, options = {} } = request.body;

            if (!lender_name || !application_data) {
                reply.status(400).send({
                    success: false,
                    error: 'Missing required fields: lender_name, application_data'
                });
                return;
            }

            // Call Python agent for submission
            const result = await callPythonAgent('submit_application', {
                lender_name,
                application_data,
                options
            });

            // Broadcast update to WebSocket clients
            broadcastToClients({
                type: 'application_submitted',
                data: {
                    lender_name,
                    submission_id: result.submission_id,
                    status: result.status,
                    timestamp: new Date().toISOString()
                }
            });

            return result;

        } catch (error) {
            fastify.log.error('Application submission failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'Application submission failed'
            });
        }
    });

    // Submit to multiple lenders (batch submission)
    app.post('/api/lender-integration-advanced/submit-batch', async (request, reply) => {
        try {
            const { lender_names, application_data, options = {} } = request.body;

            if (!lender_names || !Array.isArray(lender_names) || !application_data) {
                reply.status(400).send({
                    success: false,
                    error: 'Missing required fields: lender_names (array), application_data'
                });
                return;
            }

            const results = [];
            const errors = [];

            // Submit to each lender
            for (const lender_name of lender_names) {
                try {
                    const result = await callPythonAgent('submit_application', {
                        lender_name,
                        application_data,
                        options
                    });
                    results.push(result);

                    // Broadcast update
                    broadcastToClients({
                        type: 'application_submitted',
                        data: {
                            lender_name,
                            submission_id: result.submission_id,
                            status: result.status,
                            timestamp: new Date().toISOString()
                        }
                    });

                } catch (error) {
                    errors.push({
                        lender_name,
                        error: error.message
                    });
                }
            }

            return {
                success: true,
                results,
                errors,
                total_submitted: results.length,
                total_failed: errors.length
            };

        } catch (error) {
            fastify.log.error('Batch submission failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Batch submission failed'
            });
        }
    });

    // Get submission status
    app.get('/api/lender-integration-advanced/submissions/:submissionId', async (request, reply) => {
        try {
            const { submissionId } = request.params;

            const result = await dbPool.query(`
                SELECT las.*, ap.probability, ap.confidence_interval_lower, ap.confidence_interval_upper,
                       ap.risk_factors, ap.positive_factors, ap.recommendation
                FROM lender_submissions_advanced las
                LEFT JOIN approval_predictions ap ON las.id = ap.submission_id
                WHERE las.id = $1
            `, [submissionId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Submission not found'
                });
                return;
            }

            const submission = result.rows[0];

            // Get latest status from lender if available
            if (submission.lender_reference && submission.status !== 'failed') {
                try {
                    const lenderStatus = await callPythonAgent('get_application_status', {
                        lender_name: submission.lender_name,
                        reference_number: submission.lender_reference
                    });
                    submission.lender_status = lenderStatus;
                } catch (error) {
                    fastify.log.warn('Failed to get lender status:', error);
                }
            }

            return {
                success: true,
                submission
            };

        } catch (error) {
            fastify.log.error('Failed to get submission:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve submission'
            });
        }
    });

    // Get all submissions for an application
    app.get('/api/lender-integration-advanced/applications/:applicationId/submissions', async (request, reply) => {
        try {
            const { applicationId } = request.params;

            const result = await dbPool.query(`
                SELECT las.*, ap.probability, ap.recommendation
                FROM lender_submissions_advanced las
                LEFT JOIN approval_predictions ap ON las.id = ap.submission_id
                WHERE las.application_data->>'application_id' = $1
                ORDER BY las.created_at DESC
            `, [applicationId]);

            return {
                success: true,
                submissions: result.rows,
                total: result.rows.length
            };

        } catch (error) {
            fastify.log.error('Failed to get application submissions:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve application submissions'
            });
        }
    });

    // Retry failed submission
    app.post('/api/lender-integration-advanced/submissions/:submissionId/retry', async (request, reply) => {
        try {
            const { submissionId } = request.params;

            // Get original submission
            const submissionResult = await dbPool.query(`
                SELECT * FROM lender_submissions_advanced WHERE id = $1
            `, [submissionId]);

            if (submissionResult.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Submission not found'
                });
                return;
            }

            const originalSubmission = submissionResult.rows[0];

            if (originalSubmission.status !== 'failed') {
                reply.status(400).send({
                    success: false,
                    error: 'Only failed submissions can be retried'
                });
                return;
            }

            // Retry submission
            const result = await callPythonAgent('submit_application', {
                lender_name: originalSubmission.lender_name,
                application_data: originalSubmission.application_data,
                options: { retry: true, original_submission_id: submissionId }
            });

            return result;

        } catch (error) {
            fastify.log.error('Submission retry failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Submission retry failed'
            });
        }
    });
}

/**
 * Register validation routes
 */
async function registerValidationRoutes(app) {
    // Validate application data
    app.post('/api/lender-integration-advanced/validate', async (request, reply) => {
        try {
            const { application_data, lender_name = null } = request.body;

            if (!application_data) {
                reply.status(400).send({
                    success: false,
                    error: 'Missing required field: application_data'
                });
                return;
            }

            const result = await callPythonAgent('validate_application', {
                application_data,
                lender_name
            });

            return {
                success: true,
                validation_results: result
            };

        } catch (error) {
            fastify.log.error('Validation failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Validation failed'
            });
        }
    });

    // Get validation rules
    app.get('/api/lender-integration-advanced/validation-rules', async (request, reply) => {
        try {
            const { lender_name, field, rule_type } = request.query;

            let query = `
                SELECT * FROM validation_rules_advanced
                WHERE is_active = true
            `;
            const params = [];
            let paramIndex = 1;

            if (lender_name) {
                query += ` AND (lender_specific = $${paramIndex} OR lender_specific IS NULL)`;
                params.push(lender_name);
                paramIndex++;
            }

            if (field) {
                query += ` AND field = $${paramIndex}`;
                params.push(field);
                paramIndex++;
            }

            if (rule_type) {
                query += ` AND rule_type = $${paramIndex}`;
                params.push(rule_type);
                paramIndex++;
            }

            query += ' ORDER BY priority DESC, created_at';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                rules: result.rows,
                total: result.rows.length
            };

        } catch (error) {
            fastify.log.error('Failed to get validation rules:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve validation rules'
            });
        }
    });

    // Add validation rule
    app.post('/api/lender-integration-advanced/validation-rules', async (request, reply) => {
        try {
            const rule = request.body;

            const requiredFields = ['rule_id', 'name', 'field', 'rule_type', 'severity', 'error_message'];
            for (const field of requiredFields) {
                if (!rule[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            const result = await dbPool.query(`
                INSERT INTO validation_rules_advanced (
                    rule_id, name, description, field, rule_type, parameters,
                    severity, error_message, correction_suggestion, conditions,
                    priority, lender_specific
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING *
            `, [
                rule.rule_id, rule.name, rule.description, rule.field, rule.rule_type,
                JSON.stringify(rule.parameters || {}), rule.severity, rule.error_message,
                rule.correction_suggestion, JSON.stringify(rule.conditions || []),
                rule.priority || 1, rule.lender_specific
            ]);

            return {
                success: true,
                rule: result.rows[0],
                message: 'Validation rule added successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to add validation rule:', error);
            
            if (error.code === '23505') {
                reply.status(409).send({
                    success: false,
                    error: 'Validation rule with this ID already exists'
                });
            } else {
                reply.status(500).send({
                    success: false,
                    error: 'Failed to add validation rule'
                });
            }
        }
    });
}

/**
 * Register prediction routes
 */
async function registerPredictionRoutes(app) {
    // Get approval predictions
    app.post('/api/lender-integration-advanced/predict', async (request, reply) => {
        try {
            const { application_data, lender_names = null } = request.body;

            if (!application_data) {
                reply.status(400).send({
                    success: false,
                    error: 'Missing required field: application_data'
                });
                return;
            }

            const result = await callPythonAgent('get_approval_predictions', {
                application_data,
                lender_names
            });

            return {
                success: true,
                predictions: result
            };

        } catch (error) {
            fastify.log.error('Prediction failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Prediction failed'
            });
        }
    });

    // Get optimized lender recommendations
    app.post('/api/lender-integration-advanced/optimize', async (request, reply) => {
        try {
            const { application_data, criteria = {} } = request.body;

            if (!application_data) {
                reply.status(400).send({
                    success: false,
                    error: 'Missing required field: application_data'
                });
                return;
            }

            const result = await callPythonAgent('optimize_lender_selection', {
                application_data,
                criteria
            });

            return {
                success: true,
                recommendations: result
            };

        } catch (error) {
            fastify.log.error('Optimization failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Optimization failed'
            });
        }
    });
}

/**
 * Register health monitoring routes
 */
async function registerHealthMonitoringRoutes(app) {
    // Get health status
    app.get('/api/lender-integration-advanced/health', async (request, reply) => {
        try {
            const result = await callPythonAgent('get_health_status');

            return {
                success: true,
                health: result
            };

        } catch (error) {
            fastify.log.error('Failed to get health status:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve health status'
            });
        }
    });

    // Get lender health metrics
    app.get('/api/lender-integration-advanced/health/lenders', async (request, reply) => {
        try {
            const result = await dbPool.query(`
                SELECT * FROM lender_health_metrics
                ORDER BY lender_name
            `);

            return {
                success: true,
                metrics: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get lender health metrics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve lender health metrics'
            });
        }
    });

    // Get circuit breaker events
    app.get('/api/lender-integration-advanced/health/circuit-breaker', async (request, reply) => {
        try {
            const { lender_name, hours = 24 } = request.query;

            let query = `
                SELECT * FROM circuit_breaker_events
                WHERE created_at > NOW() - INTERVAL '${hours} hours'
            `;
            const params = [];

            if (lender_name) {
                query += ' AND lender_name = $1';
                params.push(lender_name);
            }

            query += ' ORDER BY created_at DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                events: result.rows,
                total: result.rows.length
            };

        } catch (error) {
            fastify.log.error('Failed to get circuit breaker events:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve circuit breaker events'
            });
        }
    });
}

/**
 * Register optimization routes
 */
async function registerOptimizationRoutes(app) {
    // Get lender comparison
    app.post('/api/lender-integration-advanced/compare', async (request, reply) => {
        try {
            const { application_data, lender_names } = request.body;

            if (!application_data || !lender_names || !Array.isArray(lender_names)) {
                reply.status(400).send({
                    success: false,
                    error: 'Missing required fields: application_data, lender_names (array)'
                });
                return;
            }

            const comparisons = [];

            for (const lender_name of lender_names) {
                try {
                    // Get lender configuration
                    const configResult = await dbPool.query(`
                        SELECT * FROM lender_configurations
                        WHERE LOWER(name) = LOWER($1) AND is_active = true
                    `, [lender_name]);

                    if (configResult.rows.length === 0) {
                        continue;
                    }

                    const config = configResult.rows[0];

                    // Get approval prediction
                    const prediction = await callPythonAgent('predict_approval', {
                        lender_name,
                        application_data
                    });

                    // Calculate estimated costs
                    const loanAmount = application_data.loan_amount || 0;
                    const interestRate = config.fees.interest_rate || 3.5;
                    const loanTerm = application_data.loan_term_years || 30;
                    
                    const monthlyPayment = calculateMonthlyPayment(loanAmount, interestRate, loanTerm);
                    const totalCost = monthlyPayment * loanTerm * 12;

                    comparisons.push({
                        lender_name,
                        approval_probability: prediction.probability,
                        estimated_interest_rate: interestRate,
                        monthly_payment: monthlyPayment,
                        total_cost: totalCost,
                        processing_time_days: config.processing_time_days,
                        fees: config.fees,
                        risk_factors: prediction.risk_factors,
                        positive_factors: prediction.positive_factors,
                        recommendation: prediction.recommendation
                    });

                } catch (error) {
                    fastify.log.warn(`Failed to compare lender ${lender_name}:`, error);
                }
            }

            // Sort by approval probability
            comparisons.sort((a, b) => b.approval_probability - a.approval_probability);

            return {
                success: true,
                comparisons,
                total: comparisons.length
            };

        } catch (error) {
            fastify.log.error('Comparison failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Comparison failed'
            });
        }
    });
}

/**
 * Register training routes
 */
async function registerTrainingRoutes(app) {
    // Train prediction models
    app.post('/api/lender-integration-advanced/train', async (request, reply) => {
        try {
            const { lender_names = null, training_data = null } = request.body;

            let result;

            if (training_data) {
                // Use provided training data
                result = await callPythonAgent('train_prediction_models', {
                    historical_data: training_data
                });
            } else {
                // Use historical data from database
                const historicalData = {};

                const lendersToTrain = lender_names || 
                    (await dbPool.query('SELECT DISTINCT lender_name FROM lender_training_data')).rows.map(r => r.lender_name);

                for (const lenderName of lendersToTrain) {
                    const dataResult = await dbPool.query(`
                        SELECT application_features, approved, rejection_reason,
                               processing_time_days, final_interest_rate, loan_amount
                        FROM lender_training_data
                        WHERE lender_name = $1
                        ORDER BY created_at DESC
                        LIMIT 10000
                    `, [lenderName]);

                    historicalData[lenderName] = dataResult.rows;
                }

                result = await callPythonAgent('train_prediction_models', {
                    historical_data: historicalData
                });
            }

            return {
                success: true,
                training_results: result,
                message: 'Model training completed'
            };

        } catch (error) {
            fastify.log.error('Model training failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Model training failed'
            });
        }
    });

    // Add training data
    app.post('/api/lender-integration-advanced/training-data', async (request, reply) => {
        try {
            const { lender_name, application_features, approved, rejection_reason, 
                    processing_time_days, final_interest_rate, loan_amount } = request.body;

            if (!lender_name || !application_features || approved === undefined) {
                reply.status(400).send({
                    success: false,
                    error: 'Missing required fields: lender_name, application_features, approved'
                });
                return;
            }

            const result = await dbPool.query(`
                INSERT INTO lender_training_data (
                    lender_name, application_features, approved, rejection_reason,
                    processing_time_days, final_interest_rate, loan_amount
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id, created_at
            `, [
                lender_name, JSON.stringify(application_features), approved,
                rejection_reason, processing_time_days, final_interest_rate, loan_amount
            ]);

            return {
                success: true,
                training_record: result.rows[0],
                message: 'Training data added successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to add training data:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to add training data'
            });
        }
    });
}

/**
 * Register reporting routes
 */
async function registerReportingRoutes(app) {
    // Get submission statistics
    app.get('/api/lender-integration-advanced/reports/submissions', async (request, reply) => {
        try {
            const { start_date, end_date, lender_name, status } = request.query;

            let query = `
                SELECT 
                    lender_name,
                    status,
                    COUNT(*) as count,
                    AVG(response_time_ms) as avg_response_time,
                    DATE_TRUNC('day', created_at) as date
                FROM lender_submissions_advanced
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (start_date) {
                query += ` AND created_at >= $${paramIndex}`;
                params.push(start_date);
                paramIndex++;
            }

            if (end_date) {
                query += ` AND created_at <= $${paramIndex}`;
                params.push(end_date);
                paramIndex++;
            }

            if (lender_name) {
                query += ` AND lender_name = $${paramIndex}`;
                params.push(lender_name);
                paramIndex++;
            }

            if (status) {
                query += ` AND status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            query += `
                GROUP BY lender_name, status, DATE_TRUNC('day', created_at)
                ORDER BY date DESC, lender_name
            `;

            const result = await dbPool.query(query, params);

            return {
                success: true,
                statistics: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get submission statistics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve submission statistics'
            });
        }
    });

    // Get approval rate statistics
    app.get('/api/lender-integration-advanced/reports/approval-rates', async (request, reply) => {
        try {
            const { lender_name, days = 30 } = request.query;

            let query = `
                SELECT 
                    ap.lender_name,
                    COUNT(*) as total_predictions,
                    AVG(ap.probability) as avg_probability,
                    COUNT(CASE WHEN las.status = 'approved' THEN 1 END) as approved_count,
                    COUNT(CASE WHEN las.status = 'rejected' THEN 1 END) as rejected_count,
                    (COUNT(CASE WHEN las.status = 'approved' THEN 1 END)::FLOAT / 
                     NULLIF(COUNT(CASE WHEN las.status IN ('approved', 'rejected') THEN 1 END), 0)) * 100 as actual_approval_rate
                FROM approval_predictions ap
                LEFT JOIN lender_submissions_advanced las ON ap.submission_id = las.id
                WHERE ap.created_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];

            if (lender_name) {
                query += ' AND ap.lender_name = $1';
                params.push(lender_name);
            }

            query += ' GROUP BY ap.lender_name ORDER BY avg_probability DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                approval_rates: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get approval rate statistics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve approval rate statistics'
            });
        }
    });

    // Get performance metrics
    app.get('/api/lender-integration-advanced/reports/performance', async (request, reply) => {
        try {
            const { lender_name, hours = 24 } = request.query;

            let query = `
                SELECT 
                    lender_name,
                    AVG(response_time_ms) as avg_response_time,
                    MIN(response_time_ms) as min_response_time,
                    MAX(response_time_ms) as max_response_time,
                    COUNT(*) as total_requests,
                    COUNT(CASE WHEN success = true THEN 1 END) as successful_requests,
                    (COUNT(CASE WHEN success = true THEN 1 END)::FLOAT / COUNT(*)) * 100 as success_rate
                FROM lender_api_logs
                WHERE created_at > NOW() - INTERVAL '${hours} hours'
            `;
            const params = [];

            if (lender_name) {
                query += ' AND lender_name = $1';
                params.push(lender_name);
            }

            query += ' GROUP BY lender_name ORDER BY success_rate DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                performance_metrics: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get performance metrics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve performance metrics'
            });
        }
    });
}

/**
 * Register notification routes
 */
async function registerNotificationRoutes(app) {
    // Get notifications
    app.get('/api/lender-integration-advanced/notifications', async (request, reply) => {
        try {
            const { type, status, hours = 24 } = request.query;

            let query = `
                SELECT * FROM lender_notifications
                WHERE created_at > NOW() - INTERVAL '${hours} hours'
            `;
            const params = [];
            let paramIndex = 1;

            if (type) {
                query += ` AND type = $${paramIndex}`;
                params.push(type);
                paramIndex++;
            }

            if (status) {
                query += ` AND status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            query += ' ORDER BY created_at DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                notifications: result.rows,
                total: result.rows.length
            };

        } catch (error) {
            fastify.log.error('Failed to get notifications:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve notifications'
            });
        }
    });

    // Mark notification as read
    app.put('/api/lender-integration-advanced/notifications/:notificationId/read', async (request, reply) => {
        try {
            const { notificationId } = request.params;

            const result = await dbPool.query(`
                UPDATE lender_notifications
                SET status = 'read'
                WHERE id = $1
                RETURNING *
            `, [notificationId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Notification not found'
                });
                return;
            }

            return {
                success: true,
                notification: result.rows[0]
            };

        } catch (error) {
            fastify.log.error('Failed to mark notification as read:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update notification'
            });
        }
    });
}

/**
 * Calculate monthly payment
 */
function calculateMonthlyPayment(loanAmount, annualRate, years) {
    const monthlyRate = annualRate / 100 / 12;
    const numPayments = years * 12;
    
    if (monthlyRate === 0) {
        return loanAmount / numPayments;
    }
    
    return loanAmount * (monthlyRate * Math.pow(1 + monthlyRate, numPayments)) / 
           (Math.pow(1 + monthlyRate, numPayments) - 1);
}

module.exports = {
    initializeLenderIntegrationRoutes
};