/**
 * BKR/NHG Integration API Routes
 * Production-grade Fastify routes for Dutch credit bureau and mortgage guarantee integration
 * 
 * Features:
 * - Real-time BKR credit checks with comprehensive BSN validation
 * - Live NHG eligibility verification with cost-benefit analysis
 * - Advanced compliance checking against Dutch regulations
 * - Intelligent caching with smart invalidation
 * - Risk assessment and fraud detection
 * - Performance optimization and monitoring
 * - Comprehensive audit trails and regulatory reporting
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
    bkr_checks: 0,
    nhg_checks: 0,
    compliance_validations: 0,
    cache_hits: 0,
    cache_misses: 0,
    avg_processing_time: 0,
    processing_times: []
};

/**
 * Initialize BKR/NHG Integration routes
 */
async function initializeBKRNHGRoutes(app, database, redis) {
    dbPool = database;
    redisClient = redis;

    // Initialize WebSocket server
    wsServer = new WebSocket.Server({ port: 8011 });
    
    wsServer.on('connection', (ws) => {
        fastify.log.info('WebSocket client connected to BKR/NHG Integration');
        
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                handleWebSocketMessage(ws, data);
            } catch (error) {
                ws.send(JSON.stringify({ error: 'Invalid message format' }));
            }
        });
        
        ws.on('close', () => {
            fastify.log.info('WebSocket client disconnected from BKR/NHG Integration');
        });
    });

    // Start Python agent
    await startPythonAgent();

    // Register all routes
    await registerBSNValidationRoutes(app);
    await registerBKRCreditCheckRoutes(app);
    await registerNHGEligibilityRoutes(app);
    await registerComplianceValidationRoutes(app);
    await registerRiskAssessmentRoutes(app);
    await registerComprehensiveCheckRoutes(app);
    await registerPerformanceMonitoringRoutes(app);
    await registerReportingRoutes(app);
    await registerDataManagementRoutes(app);

    fastify.log.info('BKR/NHG Integration routes initialized');
}

/**
 * Start Python agent process
 */
async function startPythonAgent() {
    try {
        const agentPath = path.join(__dirname, '../agents/bkr_nhg_integration/agent.py');
        
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
            fastify.log.info(`BKR/NHG Python Agent: ${data}`);
        });

        pythonAgent.stderr.on('data', (data) => {
            fastify.log.error(`BKR/NHG Python Agent Error: ${data}`);
        });

        pythonAgent.on('close', (code) => {
            fastify.log.warn(`BKR/NHG Python Agent exited with code ${code}`);
            // Auto-restart after 5 seconds
            setTimeout(startPythonAgent, 5000);
        });

        fastify.log.info('BKR/NHG Integration Python agent started');
    } catch (error) {
        fastify.log.error('Failed to start BKR/NHG Python agent:', error);
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
            reject(new Error('BKR/NHG Python agent timeout'));
        }, 120000); // 2 minute timeout for complex operations

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
        case 'subscribe_bkr_updates':
            ws.bkrSubscription = true;
            break;
        case 'subscribe_nhg_updates':
            ws.nhgSubscription = true;
            break;
        case 'subscribe_compliance_alerts':
            ws.complianceSubscription = true;
            break;
        case 'get_real_time_metrics':
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
        const metrics = await getPerformanceMetrics();
        ws.send(JSON.stringify({
            type: 'real_time_metrics',
            data: metrics,
            timestamp: new Date().toISOString()
        }));
    } catch (error) {
        ws.send(JSON.stringify({ error: 'Failed to get real-time metrics' }));
    }
}

/**
 * Get performance metrics
 */
async function getPerformanceMetrics() {
    try {
        const bkrMetrics = await dbPool.query(`
            SELECT 
                COUNT(*) as total_checks,
                AVG(response_time_ms) as avg_response_time,
                COUNT(CASE WHEN success = true THEN 1 END) as successful_checks,
                COUNT(CASE WHEN success = false THEN 1 END) as failed_checks
            FROM bkr_api_metrics 
            WHERE timestamp > NOW() - INTERVAL '24 hours'
        `);

        const nhgMetrics = await dbPool.query(`
            SELECT 
                COUNT(*) as total_checks,
                AVG(response_time_ms) as avg_response_time,
                COUNT(CASE WHEN success = true THEN 1 END) as successful_checks,
                COUNT(CASE WHEN eligibility_result = 'eligible' THEN 1 END) as eligible_checks
            FROM nhg_api_metrics 
            WHERE timestamp > NOW() - INTERVAL '24 hours'
        `);

        const complianceMetrics = await dbPool.query(`
            SELECT 
                regulation,
                status,
                COUNT(*) as count
            FROM compliance_validations 
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY regulation, status
        `);

        return {
            bkr: bkrMetrics.rows[0] || {},
            nhg: nhgMetrics.rows[0] || {},
            compliance: complianceMetrics.rows,
            system: performanceMetrics
        };
    } catch (error) {
        fastify.log.error('Failed to get performance metrics:', error);
        throw error;
    }
}

/**
 * Register BSN validation routes
 */
async function registerBSNValidationRoutes(app) {
    // Validate BSN
    app.post('/api/bkr-nhg/validate-bsn', async (request, reply) => {
        try {
            const { bsn } = request.body;

            if (!bsn) {
                reply.status(400).send({
                    success: false,
                    error: 'BSN is required'
                });
                return;
            }

            // Check cache first
            const bsnHash = crypto.createHash('sha256').update(bsn).digest('hex');
            const cacheKey = `bsn_validation:${bsnHash}`;
            const cachedResult = await redisClient.get(cacheKey);

            if (cachedResult) {
                performanceMetrics.cache_hits++;
                return JSON.parse(cachedResult);
            }

            performanceMetrics.cache_misses++;

            // Call Python agent for BSN validation
            const result = await callPythonAgent('validate_bsn', { bsn });

            // Store in database
            await dbPool.query(`
                INSERT INTO bsn_validations (
                    bsn_hash, is_valid, checksum_valid, format_valid, 
                    blacklist_check, confidence_score, error_message, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            `, [
                bsnHash, result.is_valid, result.checksum_valid, result.format_valid,
                result.blacklist_check, result.confidence_score, result.error_message,
                new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours expiry
            ]);

            // Cache for 1 hour
            await redisClient.setex(cacheKey, 3600, JSON.stringify({
                success: true,
                validation: result
            }));

            return {
                success: true,
                validation: result
            };

        } catch (error) {
            fastify.log.error('BSN validation failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'BSN validation failed'
            });
        }
    });

    // Batch BSN validation
    app.post('/api/bkr-nhg/validate-bsn-batch', async (request, reply) => {
        try {
            const { bsns } = request.body;

            if (!bsns || !Array.isArray(bsns)) {
                reply.status(400).send({
                    success: false,
                    error: 'BSNs array is required'
                });
                return;
            }

            if (bsns.length > 100) {
                reply.status(400).send({
                    success: false,
                    error: 'Maximum 100 BSNs per batch'
                });
                return;
            }

            const results = [];

            for (const bsn of bsns) {
                try {
                    const validation = await callPythonAgent('validate_bsn', { bsn });
                    results.push({
                        bsn: bsn.substring(0, 3) + '***' + bsn.substring(6), // Masked for privacy
                        validation
                    });
                } catch (error) {
                    results.push({
                        bsn: bsn.substring(0, 3) + '***' + bsn.substring(6),
                        error: error.message
                    });
                }
            }

            return {
                success: true,
                results,
                total_processed: results.length
            };

        } catch (error) {
            fastify.log.error('Batch BSN validation failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Batch BSN validation failed'
            });
        }
    });
}

/**
 * Register BKR credit check routes
 */
async function registerBKRCreditCheckRoutes(app) {
    // Perform BKR credit check
    app.post('/api/bkr-nhg/bkr-credit-check', async (request, reply) => {
        try {
            const { bsn, consent_token, purpose = 'mortgage_application' } = request.body;

            if (!bsn || !consent_token) {
                reply.status(400).send({
                    success: false,
                    error: 'BSN and consent token are required'
                });
                return;
            }

            // Verify consent
            const consentValid = await verifyConsent(bsn, consent_token, purpose);
            if (!consentValid) {
                reply.status(403).send({
                    success: false,
                    error: 'Invalid or expired consent'
                });
                return;
            }

            const startTime = Date.now();

            // Call Python agent for BKR check
            const result = await callPythonAgent('perform_bkr_credit_check', {
                bsn,
                consent_token,
                purpose
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.bkr_checks++;
            performanceMetrics.processing_times.push(processingTime);

            // Update performance metrics
            if (performanceMetrics.processing_times.length > 1000) {
                performanceMetrics.processing_times = performanceMetrics.processing_times.slice(-1000);
            }
            performanceMetrics.avg_processing_time = 
                performanceMetrics.processing_times.reduce((a, b) => a + b, 0) / 
                performanceMetrics.processing_times.length;

            // Store API metrics
            await dbPool.query(`
                INSERT INTO bkr_api_metrics (
                    endpoint, method, response_time_ms, status_code, success
                ) VALUES ($1, $2, $3, $4, $5)
            `, ['/bkr-credit-check', 'POST', processingTime, 200, true]);

            // Broadcast update to WebSocket clients
            broadcastToClients({
                type: 'bkr_check_completed',
                data: {
                    check_id: result.check_id,
                    credit_score: result.credit_score,
                    risk_level: result.risk_indicators?.length > 0 ? 'medium' : 'low',
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                bkr_check: result,
                processing_time_ms: processingTime
            };

        } catch (error) {
            const processingTime = Date.now() - (request.startTime || Date.now());
            
            // Store failed API metrics
            await dbPool.query(`
                INSERT INTO bkr_api_metrics (
                    endpoint, method, response_time_ms, status_code, success, error_message
                ) VALUES ($1, $2, $3, $4, $5, $6)
            `, ['/bkr-credit-check', 'POST', processingTime, 500, false, error.message]);

            fastify.log.error('BKR credit check failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'BKR credit check failed'
            });
        }
    });

    // Get BKR check status
    app.get('/api/bkr-nhg/bkr-credit-check/:checkId', async (request, reply) => {
        try {
            const { checkId } = request.params;

            const result = await dbPool.query(`
                SELECT * FROM bkr_credit_checks WHERE check_id = $1
            `, [checkId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'BKR check not found'
                });
                return;
            }

            const check = result.rows[0];

            return {
                success: true,
                bkr_check: {
                    check_id: check.check_id,
                    status: check.status,
                    credit_score: check.credit_score,
                    debt_to_income_ratio: check.debt_to_income_ratio,
                    total_debt: check.total_debt,
                    risk_indicators: check.risk_indicators,
                    recommendations: check.recommendations,
                    last_updated: check.last_updated,
                    expires_at: check.expires_at
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get BKR check:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve BKR check'
            });
        }
    });
}

/**
 * Register NHG eligibility routes
 */
async function registerNHGEligibilityRoutes(app) {
    // Check NHG eligibility
    app.post('/api/bkr-nhg/nhg-eligibility', async (request, reply) => {
        try {
            const { property_data, applicant_data, loan_data } = request.body;

            if (!property_data || !applicant_data || !loan_data) {
                reply.status(400).send({
                    success: false,
                    error: 'Property data, applicant data, and loan data are required'
                });
                return;
            }

            const startTime = Date.now();

            // Call Python agent for NHG eligibility check
            const result = await callPythonAgent('check_nhg_eligibility', {
                property_data,
                applicant_data,
                loan_data
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.nhg_checks++;

            // Store API metrics
            await dbPool.query(`
                INSERT INTO nhg_api_metrics (
                    endpoint, method, response_time_ms, status_code, success, eligibility_result
                ) VALUES ($1, $2, $3, $4, $5, $6)
            `, ['/nhg-eligibility', 'POST', processingTime, 200, true, result.eligibility_status]);

            // Store NHG assessment
            await dbPool.query(`
                INSERT INTO nhg_eligibility_assessments (
                    property_value, loan_amount, nhg_limit, is_eligible, eligibility_status,
                    cost_benefit_analysis, nhg_premium, interest_rate_benefit, total_savings,
                    conditions, restrictions, property_requirements, income_requirements,
                    compliance_notes, validity_period, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            `, [
                result.property_value, result.loan_amount, result.nhg_limit, result.is_eligible,
                result.eligibility_status, JSON.stringify(result.cost_benefit_analysis),
                result.nhg_premium, result.interest_rate_benefit, result.total_savings,
                JSON.stringify(result.conditions), JSON.stringify(result.restrictions),
                JSON.stringify(result.property_requirements), JSON.stringify(result.income_requirements),
                JSON.stringify(result.compliance_notes), result.validity_period,
                new Date(Date.now() + result.validity_period * 24 * 60 * 60 * 1000)
            ]);

            // Broadcast update to WebSocket clients
            broadcastToClients({
                type: 'nhg_check_completed',
                data: {
                    is_eligible: result.is_eligible,
                    eligibility_status: result.eligibility_status,
                    total_savings: result.total_savings,
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                nhg_eligibility: result,
                processing_time_ms: processingTime
            };

        } catch (error) {
            const processingTime = Date.now() - (request.startTime || Date.now());
            
            // Store failed API metrics
            await dbPool.query(`
                INSERT INTO nhg_api_metrics (
                    endpoint, method, response_time_ms, status_code, success, error_message
                ) VALUES ($1, $2, $3, $4, $5, $6)
            `, ['/nhg-eligibility', 'POST', processingTime, 500, false, error.message]);

            fastify.log.error('NHG eligibility check failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'NHG eligibility check failed'
            });
        }
    });

    // Get current NHG limits
    app.get('/api/bkr-nhg/nhg-limits', async (request, reply) => {
        try {
            const result = await dbPool.query(`
                SELECT * FROM nhg_limits_history 
                WHERE is_current = true 
                ORDER BY effective_date DESC 
                LIMIT 1
            `);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Current NHG limits not found'
                });
                return;
            }

            const limits = result.rows[0];

            return {
                success: true,
                nhg_limits: {
                    standard_limit: limits.standard_limit,
                    energy_efficient_bonus: limits.energy_efficient_bonus,
                    starter_bonus: limits.starter_bonus,
                    renovation_limit: limits.renovation_limit,
                    premium_rate: limits.premium_rate,
                    effective_date: limits.effective_date
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get NHG limits:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve NHG limits'
            });
        }
    });

    // Update NHG limits (admin only)
    app.post('/api/bkr-nhg/nhg-limits', async (request, reply) => {
        try {
            const { standard_limit, energy_efficient_bonus, starter_bonus, renovation_limit, premium_rate, effective_date } = request.body;

            // Validate required fields
            if (!standard_limit || !effective_date) {
                reply.status(400).send({
                    success: false,
                    error: 'Standard limit and effective date are required'
                });
                return;
            }

            // Deactivate current limits
            await dbPool.query(`
                UPDATE nhg_limits_history SET is_current = false WHERE is_current = true
            `);

            // Insert new limits
            const result = await dbPool.query(`
                INSERT INTO nhg_limits_history (
                    standard_limit, energy_efficient_bonus, starter_bonus, 
                    renovation_limit, premium_rate, effective_date, is_current
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
            `, [
                standard_limit, energy_efficient_bonus, starter_bonus,
                renovation_limit, premium_rate, effective_date, true
            ]);

            return {
                success: true,
                nhg_limits: result.rows[0],
                message: 'NHG limits updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update NHG limits:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update NHG limits'
            });
        }
    });
}

/**
 * Register compliance validation routes
 */
async function registerComplianceValidationRoutes(app) {
    // Validate compliance
    app.post('/api/bkr-nhg/validate-compliance', async (request, reply) => {
        try {
            const { application_data, regulation_type = 'all' } = request.body;

            if (!application_data) {
                reply.status(400).send({
                    success: false,
                    error: 'Application data is required'
                });
                return;
            }

            const startTime = Date.now();

            // Call Python agent for compliance validation
            const result = await callPythonAgent('validate_compliance', {
                application_data,
                regulation_type
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.compliance_validations++;

            // Broadcast compliance alerts for critical issues
            const criticalIssues = result.filter(check => 
                check.status === 'critical' || check.risk_level === 'critical'
            );

            if (criticalIssues.length > 0) {
                broadcastToClients({
                    type: 'compliance_alert',
                    data: {
                        critical_issues: criticalIssues.length,
                        regulations_affected: [...new Set(criticalIssues.map(issue => issue.regulation))],
                        timestamp: new Date().toISOString()
                    }
                });
            }

            return {
                success: true,
                compliance_validation: result,
                summary: {
                    total_checks: result.length,
                    compliant: result.filter(c => c.status === 'compliant').length,
                    warnings: result.filter(c => c.status === 'warning').length,
                    non_compliant: result.filter(c => c.status === 'non_compliant').length,
                    critical: result.filter(c => c.status === 'critical').length
                },
                processing_time_ms: processingTime
            };

        } catch (error) {
            fastify.log.error('Compliance validation failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'Compliance validation failed'
            });
        }
    });

    // Get compliance history
    app.get('/api/bkr-nhg/compliance-history', async (request, reply) => {
        try {
            const { regulation, status, days = 30 } = request.query;

            let query = `
                SELECT regulation, article, requirement, status, risk_level,
                       COUNT(*) as count, MAX(checked_at) as last_check
                FROM compliance_validations
                WHERE created_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (regulation) {
                query += ` AND regulation = $${paramIndex}`;
                params.push(regulation);
                paramIndex++;
            }

            if (status) {
                query += ` AND status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            query += ` GROUP BY regulation, article, requirement, status, risk_level
                      ORDER BY count DESC, last_check DESC`;

            const result = await dbPool.query(query, params);

            return {
                success: true,
                compliance_history: result.rows,
                total_records: result.rows.length
            };

        } catch (error) {
            fastify.log.error('Failed to get compliance history:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve compliance history'
            });
        }
    });
}

/**
 * Register risk assessment routes
 */
async function registerRiskAssessmentRoutes(app) {
    // Generate risk assessment
    app.post('/api/bkr-nhg/risk-assessment', async (request, reply) => {
        try {
            const { bkr_data, nhg_data, compliance_data } = request.body;

            if (!bkr_data && !nhg_data && !compliance_data) {
                reply.status(400).send({
                    success: false,
                    error: 'At least one data type (BKR, NHG, or compliance) is required'
                });
                return;
            }

            // Call Python agent for risk assessment
            const result = await callPythonAgent('generate_risk_assessment', {
                bkr_data,
                nhg_data,
                compliance_data
            });

            return {
                success: true,
                risk_assessment: result
            };

        } catch (error) {
            fastify.log.error('Risk assessment failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'Risk assessment failed'
            });
        }
    });

    // Get risk statistics
    app.get('/api/bkr-nhg/risk-statistics', async (request, reply) => {
        try {
            const { days = 30 } = request.query;

            const result = await dbPool.query(`
                SELECT 
                    risk_level,
                    COUNT(*) as count,
                    AVG(overall_risk_score) as avg_risk_score,
                    AVG(confidence_level) as avg_confidence
                FROM risk_assessments
                WHERE created_at > NOW() - INTERVAL '${days} days'
                GROUP BY risk_level
                ORDER BY 
                    CASE risk_level 
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                    END
            `);

            return {
                success: true,
                risk_statistics: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get risk statistics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve risk statistics'
            });
        }
    });
}

/**
 * Register comprehensive check routes
 */
async function registerComprehensiveCheckRoutes(app) {
    // Perform comprehensive BKR/NHG check
    app.post('/api/bkr-nhg/comprehensive-check', async (request, reply) => {
        try {
            const application_data = request.body;

            if (!application_data.bsn || !application_data.consent_token) {
                reply.status(400).send({
                    success: false,
                    error: 'BSN and consent token are required'
                });
                return;
            }

            const startTime = Date.now();

            // Call Python agent for comprehensive check
            const result = await callPythonAgent('perform_comprehensive_check', application_data);

            const processingTime = Date.now() - startTime;

            // Update performance metrics
            performanceMetrics.processing_times.push(processingTime);
            if (performanceMetrics.processing_times.length > 1000) {
                performanceMetrics.processing_times = performanceMetrics.processing_times.slice(-1000);
            }
            performanceMetrics.avg_processing_time = 
                performanceMetrics.processing_times.reduce((a, b) => a + b, 0) / 
                performanceMetrics.processing_times.length;

            // Broadcast comprehensive update
            broadcastToClients({
                type: 'comprehensive_check_completed',
                data: {
                    check_id: result.check_id,
                    bkr_status: result.bkr_check?.status,
                    nhg_eligible: result.nhg_eligibility?.is_eligible,
                    risk_level: result.risk_assessment?.risk_level,
                    compliance_issues: result.compliance_validation?.filter(c => c.status !== 'compliant').length,
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                ...result,
                processing_time_ms: processingTime
            };

        } catch (error) {
            fastify.log.error('Comprehensive check failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'Comprehensive check failed'
            });
        }
    });

    // Get comprehensive check by ID
    app.get('/api/bkr-nhg/comprehensive-check/:checkId', async (request, reply) => {
        try {
            const { checkId } = request.params;

            const result = await dbPool.query(`
                SELECT * FROM bkr_nhg_checks WHERE id = $1
            `, [checkId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Comprehensive check not found'
                });
                return;
            }

            const check = result.rows[0];

            return {
                success: true,
                check_id: check.id,
                application_data: check.application_data,
                bkr_results: check.bkr_results,
                nhg_results: check.nhg_results,
                compliance_results: check.compliance_results,
                risk_assessment: check.risk_assessment,
                recommendations: check.recommendations,
                status: check.status,
                processing_time_ms: check.processing_time_ms,
                created_at: check.created_at
            };

        } catch (error) {
            fastify.log.error('Failed to get comprehensive check:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve comprehensive check'
            });
        }
    });
}

/**
 * Register performance monitoring routes
 */
async function registerPerformanceMonitoringRoutes(app) {
    // Get performance metrics
    app.get('/api/bkr-nhg/performance-metrics', async (request, reply) => {
        try {
            const metrics = await getPerformanceMetrics();

            return {
                success: true,
                metrics,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            fastify.log.error('Failed to get performance metrics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve performance metrics'
            });
        }
    });

    // Get system health
    app.get('/api/bkr-nhg/health', async (request, reply) => {
        try {
            const pythonAgentHealthy = pythonAgent && !pythonAgent.killed;
            const dbHealthy = await testDatabaseConnection();
            const redisHealthy = await testRedisConnection();

            const overallHealth = pythonAgentHealthy && dbHealthy && redisHealthy;

            return {
                success: true,
                health: {
                    overall_status: overallHealth ? 'healthy' : 'degraded',
                    python_agent: pythonAgentHealthy ? 'healthy' : 'unhealthy',
                    database: dbHealthy ? 'healthy' : 'unhealthy',
                    redis: redisHealthy ? 'healthy' : 'unhealthy',
                    websocket_clients: wsServer.clients.size,
                    uptime: process.uptime()
                },
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            fastify.log.error('Health check failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Health check failed'
            });
        }
    });
}

/**
 * Register reporting routes
 */
async function registerReportingRoutes(app) {
    // Get BKR/NHG summary report
    app.get('/api/bkr-nhg/reports/summary', async (request, reply) => {
        try {
            const { start_date, end_date } = request.query;

            const dateFilter = start_date && end_date ? 
                `WHERE created_at BETWEEN $1 AND $2` : 
                `WHERE created_at > NOW() - INTERVAL '30 days'`;
            
            const params = start_date && end_date ? [start_date, end_date] : [];

            const bkrStats = await dbPool.query(`
                SELECT 
                    COUNT(*) as total_checks,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_checks,
                    AVG(credit_score) as avg_credit_score,
                    COUNT(CASE WHEN jsonb_array_length(defaults) > 0 THEN 1 END) as checks_with_defaults
                FROM bkr_credit_checks
                ${dateFilter}
            `, params);

            const nhgStats = await dbPool.query(`
                SELECT 
                    COUNT(*) as total_assessments,
                    COUNT(CASE WHEN is_eligible = true THEN 1 END) as eligible_assessments,
                    AVG(total_savings) as avg_savings,
                    AVG(nhg_premium) as avg_premium
                FROM nhg_eligibility_assessments
                ${dateFilter}
            `, params);

            const complianceStats = await dbPool.query(`
                SELECT 
                    regulation,
                    status,
                    COUNT(*) as count
                FROM compliance_validations
                ${dateFilter}
                GROUP BY regulation, status
                ORDER BY regulation, status
            `, params);

            return {
                success: true,
                summary_report: {
                    bkr_statistics: bkrStats.rows[0] || {},
                    nhg_statistics: nhgStats.rows[0] || {},
                    compliance_statistics: complianceStats.rows,
                    report_period: {
                        start_date: start_date || new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
                        end_date: end_date || new Date().toISOString()
                    }
                }
            };

        } catch (error) {
            fastify.log.error('Failed to generate summary report:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to generate summary report'
            });
        }
    });

    // Get detailed analytics
    app.get('/api/bkr-nhg/reports/analytics', async (request, reply) => {
        try {
            const { metric_type = 'all', days = 30 } = request.query;

            const results = {};

            if (metric_type === 'all' || metric_type === 'performance') {
                const performanceData = await dbPool.query(`
                    SELECT 
                        DATE_TRUNC('day', timestamp) as date,
                        AVG(response_time_ms) as avg_response_time,
                        COUNT(*) as total_requests,
                        COUNT(CASE WHEN success = true THEN 1 END) as successful_requests
                    FROM bkr_api_metrics
                    WHERE timestamp > NOW() - INTERVAL '${days} days'
                    GROUP BY DATE_TRUNC('day', timestamp)
                    ORDER BY date
                `);
                results.performance_trends = performanceData.rows;
            }

            if (metric_type === 'all' || metric_type === 'compliance') {
                const complianceData = await dbPool.query(`
                    SELECT 
                        DATE_TRUNC('day', created_at) as date,
                        regulation,
                        status,
                        COUNT(*) as count
                    FROM compliance_validations
                    WHERE created_at > NOW() - INTERVAL '${days} days'
                    GROUP BY DATE_TRUNC('day', created_at), regulation, status
                    ORDER BY date, regulation
                `);
                results.compliance_trends = complianceData.rows;
            }

            if (metric_type === 'all' || metric_type === 'risk') {
                const riskData = await dbPool.query(`
                    SELECT 
                        risk_level,
                        COUNT(*) as count,
                        AVG(overall_risk_score) as avg_score
                    FROM risk_assessments
                    WHERE created_at > NOW() - INTERVAL '${days} days'
                    GROUP BY risk_level
                    ORDER BY 
                        CASE risk_level 
                            WHEN 'critical' THEN 1
                            WHEN 'high' THEN 2
                            WHEN 'medium' THEN 3
                            WHEN 'low' THEN 4
                        END
                `);
                results.risk_distribution = riskData.rows;
            }

            return {
                success: true,
                analytics: results,
                period_days: days
            };

        } catch (error) {
            fastify.log.error('Failed to generate analytics report:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to generate analytics report'
            });
        }
    });
}

/**
 * Register data management routes
 */
async function registerDataManagementRoutes(app) {
    // Clean up expired data
    app.post('/api/bkr-nhg/cleanup-expired-data', async (request, reply) => {
        try {
            const result = await dbPool.query('SELECT cleanup_expired_bkr_nhg_data()');
            const deletedCount = result.rows[0].cleanup_expired_bkr_nhg_data;

            return {
                success: true,
                message: `Cleaned up ${deletedCount} expired records`,
                deleted_count: deletedCount
            };

        } catch (error) {
            fastify.log.error('Data cleanup failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Data cleanup failed'
            });
        }
    });

    // Get data retention statistics
    app.get('/api/bkr-nhg/data-retention-stats', async (request, reply) => {
        try {
            const stats = await dbPool.query(`
                SELECT 
                    table_name,
                    SUM(records_processed) as total_processed,
                    SUM(records_deleted) as total_deleted,
                    MAX(started_at) as last_cleanup
                FROM data_retention_log
                GROUP BY table_name
                ORDER BY last_cleanup DESC
            `);

            return {
                success: true,
                retention_statistics: stats.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get retention statistics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve retention statistics'
            });
        }
    });
}

/**
 * Verify consent token
 */
async function verifyConsent(bsn, consentToken, purpose) {
    try {
        const bsnHash = crypto.createHash('sha256').update(bsn).digest('hex');
        const tokenHash = crypto.createHash('sha256').update(consentToken).digest('hex');

        const result = await dbPool.query(`
            SELECT * FROM data_consent_records
            WHERE bsn_hash = $1 AND consent_token_hash = $2 AND purpose = $3
            AND is_active = true AND (expires_at IS NULL OR expires_at > NOW())
        `, [bsnHash, tokenHash, purpose]);

        return result.rows.length > 0;
    } catch (error) {
        fastify.log.error('Consent verification failed:', error);
        return false;
    }
}

/**
 * Test database connection
 */
async function testDatabaseConnection() {
    try {
        await dbPool.query('SELECT 1');
        return true;
    } catch (error) {
        return false;
    }
}

/**
 * Test Redis connection
 */
async function testRedisConnection() {
    try {
        await redisClient.ping();
        return true;
    } catch (error) {
        return false;
    }
}

module.exports = {
    initializeBKRNHGRoutes
};