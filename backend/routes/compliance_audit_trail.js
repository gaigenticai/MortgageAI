/**
 * Comprehensive Compliance Audit Trail API Routes
 * Production-grade Fastify routes for advanced compliance logging and investigation
 * 
 * Features:
 * - Immutable audit event logging with hash chain verification
 * - Real-time compliance violation detection and alerting
 * - Advanced investigation management with evidence tracking
 * - Comprehensive regulatory reporting with automated generation
 * - Pattern recognition and anomaly detection
 * - Data lineage tracking and impact analysis
 * - Advanced search and filtering capabilities
 * - Stakeholder management and notification system
 */

const fastify = require('fastify')({ logger: true });
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const WebSocket = require('ws');
const crypto = require('crypto');
const { Parser } = require('json2csv');

// Database and Redis connections
let dbPool, redisClient;

// WebSocket server for real-time updates
let wsServer;

// Python agent process
let pythonAgent = null;

// Performance metrics
const performanceMetrics = {
    events_logged: 0,
    violations_detected: 0,
    investigations_created: 0,
    reports_generated: 0,
    integrity_checks: 0,
    search_queries: 0,
    avg_response_time: 0,
    response_times: []
};

/**
 * Initialize Compliance Audit Trail routes
 */
async function initializeComplianceAuditTrailRoutes(app, database, redis) {
    dbPool = database;
    redisClient = redis;

    // Initialize WebSocket server
    wsServer = new WebSocket.Server({ port: 8012 });
    
    wsServer.on('connection', (ws) => {
        fastify.log.info('WebSocket client connected to Compliance Audit Trail');
        
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                handleWebSocketMessage(ws, data);
            } catch (error) {
                ws.send(JSON.stringify({ error: 'Invalid message format' }));
            }
        });
        
        ws.on('close', () => {
            fastify.log.info('WebSocket client disconnected from Compliance Audit Trail');
        });
    });

    // Start Python agent
    await startPythonAgent();

    // Register all routes
    await registerAuditEventRoutes(app);
    await registerViolationManagementRoutes(app);
    await registerInvestigationRoutes(app);
    await registerReportingRoutes(app);
    await registerIntegrityVerificationRoutes(app);
    await registerPatternAnalysisRoutes(app);
    await registerSearchRoutes(app);
    await registerStakeholderRoutes(app);
    await registerMetricsRoutes(app);
    await registerNotificationRoutes(app);
    await registerDataLineageRoutes(app);

    fastify.log.info('Compliance Audit Trail routes initialized');
}

/**
 * Start Python agent process
 */
async function startPythonAgent() {
    try {
        const agentPath = path.join(__dirname, '../agents/compliance_audit_trail/agent.py');
        
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
            fastify.log.info(`Compliance Audit Python Agent: ${data}`);
        });

        pythonAgent.stderr.on('data', (data) => {
            fastify.log.error(`Compliance Audit Python Agent Error: ${data}`);
        });

        pythonAgent.on('close', (code) => {
            fastify.log.warn(`Compliance Audit Python Agent exited with code ${code}`);
            // Auto-restart after 5 seconds
            setTimeout(startPythonAgent, 5000);
        });

        fastify.log.info('Compliance Audit Trail Python agent started');
    } catch (error) {
        fastify.log.error('Failed to start Compliance Audit Python agent:', error);
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
            reject(new Error('Compliance Audit Python agent timeout'));
        }, 180000); // 3 minute timeout for complex operations

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
        case 'subscribe_audit_events':
            ws.auditSubscription = true;
            break;
        case 'subscribe_violations':
            ws.violationSubscription = true;
            break;
        case 'subscribe_investigations':
            ws.investigationSubscription = true;
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
        const dashboardData = await getDashboardData();
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
 * Get dashboard data
 */
async function getDashboardData() {
    try {
        const today = new Date().toISOString().split('T')[0];
        
        const eventStats = await dbPool.query(`
            SELECT 
                COUNT(*) as total_events,
                COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_events,
                COUNT(CASE WHEN compliance_status != 'compliant' THEN 1 END) as non_compliant_events,
                AVG(risk_score) as avg_risk_score
            FROM compliance_audit_events 
            WHERE DATE(timestamp) = $1
        `, [today]);

        const violationStats = await dbPool.query(`
            SELECT 
                COUNT(*) as total_violations,
                COUNT(CASE WHEN remediation_status = 'completed' THEN 1 END) as resolved_violations,
                COUNT(CASE WHEN investigation_required = true THEN 1 END) as investigation_required
            FROM compliance_violations 
            WHERE DATE(detection_timestamp) = $1
        `, [today]);

        const investigationStats = await dbPool.query(`
            SELECT 
                COUNT(*) as total_investigations,
                COUNT(CASE WHEN status = 'open' THEN 1 END) as open_investigations,
                COUNT(CASE WHEN priority = 'critical' THEN 1 END) as critical_investigations
            FROM compliance_investigations 
            WHERE DATE(created_timestamp) >= $1 - INTERVAL '7 days'
        `, [today]);

        return {
            events: eventStats.rows[0] || {},
            violations: violationStats.rows[0] || {},
            investigations: investigationStats.rows[0] || {},
            system_metrics: performanceMetrics
        };
    } catch (error) {
        fastify.log.error('Failed to get dashboard data:', error);
        throw error;
    }
}

/**
 * Register audit event routes
 */
async function registerAuditEventRoutes(app) {
    // Log compliance event
    app.post('/api/compliance-audit/events', async (request, reply) => {
        try {
            const eventData = request.body;
            const startTime = Date.now();

            // Validate required fields
            const requiredFields = ['event_type', 'entity_type', 'entity_id', 'action', 'details'];
            for (const field of requiredFields) {
                if (!eventData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            // Extract client information
            const clientInfo = {
                ip_address: request.ip,
                user_agent: request.headers['user-agent'],
                session_id: request.headers['x-session-id']
            };

            // Call Python agent to log event
            const eventId = await callPythonAgent('log_compliance_event', {
                ...eventData,
                ...clientInfo
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.events_logged++;
            performanceMetrics.response_times.push(processingTime);

            // Update average response time
            if (performanceMetrics.response_times.length > 1000) {
                performanceMetrics.response_times = performanceMetrics.response_times.slice(-1000);
            }
            performanceMetrics.avg_response_time = 
                performanceMetrics.response_times.reduce((a, b) => a + b, 0) / 
                performanceMetrics.response_times.length;

            // Broadcast to WebSocket clients
            broadcastToClients({
                type: 'audit_event_logged',
                data: {
                    event_id: eventId,
                    event_type: eventData.event_type,
                    severity: eventData.severity || 'info',
                    regulation: eventData.regulation,
                    timestamp: new Date().toISOString()
                }
            });

            return {
                success: true,
                event_id: eventId,
                processing_time_ms: processingTime,
                message: 'Compliance event logged successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to log compliance event:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'Failed to log compliance event'
            });
        }
    });

    // Get audit events with advanced filtering
    app.get('/api/compliance-audit/events', async (request, reply) => {
        try {
            const {
                start_date,
                end_date,
                user_id,
                entity_type,
                entity_id,
                regulation,
                severity,
                compliance_status,
                event_type,
                page = 1,
                limit = 50,
                sort_by = 'timestamp',
                sort_order = 'desc'
            } = request.query;

            // Build dynamic query
            let query = `
                SELECT event_id, event_type, timestamp, user_id, entity_type, entity_id,
                       action, regulation, compliance_status, severity, risk_score,
                       investigation_id, tags, created_at
                FROM compliance_audit_events
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            // Add filters
            if (start_date) {
                query += ` AND timestamp >= $${paramIndex}`;
                params.push(start_date);
                paramIndex++;
            }

            if (end_date) {
                query += ` AND timestamp <= $${paramIndex}`;
                params.push(end_date);
                paramIndex++;
            }

            if (user_id) {
                query += ` AND user_id = $${paramIndex}`;
                params.push(user_id);
                paramIndex++;
            }

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

            if (regulation) {
                query += ` AND regulation = $${paramIndex}`;
                params.push(regulation);
                paramIndex++;
            }

            if (severity) {
                query += ` AND severity = $${paramIndex}`;
                params.push(severity);
                paramIndex++;
            }

            if (compliance_status) {
                query += ` AND compliance_status = $${paramIndex}`;
                params.push(compliance_status);
                paramIndex++;
            }

            if (event_type) {
                query += ` AND event_type = $${paramIndex}`;
                params.push(event_type);
                paramIndex++;
            }

            // Add sorting and pagination
            query += ` ORDER BY ${sort_by} ${sort_order.toUpperCase()}`;
            query += ` LIMIT $${paramIndex} OFFSET $${paramIndex + 1}`;
            params.push(parseInt(limit));
            params.push((parseInt(page) - 1) * parseInt(limit));

            const result = await dbPool.query(query, params);

            // Get total count for pagination
            let countQuery = query.substring(0, query.indexOf('ORDER BY')).replace(
                'SELECT event_id, event_type, timestamp, user_id, entity_type, entity_id, action, regulation, compliance_status, severity, risk_score, investigation_id, tags, created_at',
                'SELECT COUNT(*)'
            );
            const countParams = params.slice(0, -2); // Remove limit and offset
            const countResult = await dbPool.query(countQuery, countParams);
            const totalCount = parseInt(countResult.rows[0].count);

            return {
                success: true,
                events: result.rows,
                pagination: {
                    page: parseInt(page),
                    limit: parseInt(limit),
                    total: totalCount,
                    pages: Math.ceil(totalCount / parseInt(limit))
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get audit events:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve audit events'
            });
        }
    });

    // Get specific audit event with full details
    app.get('/api/compliance-audit/events/:eventId', async (request, reply) => {
        try {
            const { eventId } = request.params;

            const result = await dbPool.query(`
                SELECT * FROM compliance_audit_events WHERE event_id = $1
            `, [eventId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Audit event not found'
                });
                return;
            }

            const event = result.rows[0];

            // Get related violations
            const violationsResult = await dbPool.query(`
                SELECT * FROM compliance_violations WHERE event_id = $1
            `, [eventId]);

            // Get related investigations
            const investigationsResult = await dbPool.query(`
                SELECT investigation_id, title, status, priority
                FROM compliance_investigations 
                WHERE $1 = ANY(string_to_array(related_events::text, ','))
            `, [eventId]);

            return {
                success: true,
                event: {
                    ...event,
                    related_violations: violationsResult.rows,
                    related_investigations: investigationsResult.rows
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get audit event:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve audit event'
            });
        }
    });

    // Batch log multiple events
    app.post('/api/compliance-audit/events/batch', async (request, reply) => {
        try {
            const { events } = request.body;

            if (!events || !Array.isArray(events)) {
                reply.status(400).send({
                    success: false,
                    error: 'Events array is required'
                });
                return;
            }

            if (events.length > 100) {
                reply.status(400).send({
                    success: false,
                    error: 'Maximum 100 events per batch'
                });
                return;
            }

            const results = [];
            const errors = [];

            for (const eventData of events) {
                try {
                    const eventId = await callPythonAgent('log_compliance_event', {
                        ...eventData,
                        ip_address: request.ip,
                        user_agent: request.headers['user-agent']
                    });
                    results.push({ event_id: eventId, status: 'logged' });
                } catch (error) {
                    errors.push({ event_data: eventData, error: error.message });
                }
            }

            performanceMetrics.events_logged += results.length;

            return {
                success: true,
                results,
                errors,
                total_processed: events.length,
                successful: results.length,
                failed: errors.length
            };

        } catch (error) {
            fastify.log.error('Batch event logging failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Batch event logging failed'
            });
        }
    });
}

/**
 * Register violation management routes
 */
async function registerViolationManagementRoutes(app) {
    // Get compliance violations
    app.get('/api/compliance-audit/violations', async (request, reply) => {
        try {
            const {
                regulation,
                severity,
                status,
                assigned_to,
                days = 30,
                page = 1,
                limit = 50
            } = request.query;

            let query = `
                SELECT cv.*, cae.user_id, cae.timestamp as event_timestamp, cae.action
                FROM compliance_violations cv
                JOIN compliance_audit_events cae ON cv.event_id = cae.event_id
                WHERE cv.detection_timestamp > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (regulation) {
                query += ` AND cv.regulation = $${paramIndex}`;
                params.push(regulation);
                paramIndex++;
            }

            if (severity) {
                query += ` AND cv.severity = $${paramIndex}`;
                params.push(severity);
                paramIndex++;
            }

            if (status) {
                query += ` AND cv.remediation_status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            if (assigned_to) {
                query += ` AND cv.compliance_officer_assigned = $${paramIndex}`;
                params.push(assigned_to);
                paramIndex++;
            }

            query += ` ORDER BY cv.detection_timestamp DESC`;
            query += ` LIMIT $${paramIndex} OFFSET $${paramIndex + 1}`;
            params.push(parseInt(limit));
            params.push((parseInt(page) - 1) * parseInt(limit));

            const result = await dbPool.query(query, params);

            return {
                success: true,
                violations: result.rows,
                pagination: {
                    page: parseInt(page),
                    limit: parseInt(limit)
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get violations:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve violations'
            });
        }
    });

    // Update violation status
    app.put('/api/compliance-audit/violations/:violationId', async (request, reply) => {
        try {
            const { violationId } = request.params;
            const updates = request.body;

            const allowedUpdates = [
                'remediation_status', 'remediation_actions', 'remediation_deadline',
                'compliance_officer_assigned', 'resolution_details', 'escalation_level'
            ];

            const updateFields = [];
            const updateValues = [];
            let paramIndex = 1;

            for (const [key, value] of Object.entries(updates)) {
                if (allowedUpdates.includes(key)) {
                    if (key === 'remediation_actions' && Array.isArray(value)) {
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

            // Add resolution timestamp if status is completed
            if (updates.remediation_status === 'completed') {
                updateFields.push('resolution_timestamp = CURRENT_TIMESTAMP');
            }

            updateFields.push('updated_at = CURRENT_TIMESTAMP');
            updateValues.push(violationId);

            const updateQuery = `
                UPDATE compliance_violations
                SET ${updateFields.join(', ')}
                WHERE violation_id = $${paramIndex}
                RETURNING *
            `;

            const result = await dbPool.query(updateQuery, updateValues);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Violation not found'
                });
                return;
            }

            // Log the update as an audit event
            await callPythonAgent('log_compliance_event', {
                event_type: 'system_action',
                entity_type: 'compliance_violation',
                entity_id: violationId,
                action: 'violation_updated',
                details: { updates, updated_by: request.user?.id || 'system' },
                severity: 'info'
            });

            // Broadcast update
            broadcastToClients({
                type: 'violation_updated',
                data: {
                    violation_id: violationId,
                    status: updates.remediation_status,
                    timestamp: new Date().toISOString()
                }
            });

            return {
                success: true,
                violation: result.rows[0],
                message: 'Violation updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update violation:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update violation'
            });
        }
    });

    // Create manual violation
    app.post('/api/compliance-audit/violations', async (request, reply) => {
        try {
            const violationData = request.body;

            const requiredFields = ['regulation', 'violation_type', 'description', 'severity'];
            for (const field of requiredFields) {
                if (!violationData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            // Create associated audit event first
            const eventId = await callPythonAgent('log_compliance_event', {
                event_type: 'violation_detected',
                entity_type: 'manual_violation',
                entity_id: crypto.randomUUID(),
                action: 'manual_violation_created',
                details: violationData,
                regulation: violationData.regulation,
                severity: violationData.severity,
                user_id: request.user?.id
            });

            // Create violation record
            const violationId = crypto.randomUUID();
            await dbPool.query(`
                INSERT INTO compliance_violations (
                    violation_id, event_id, regulation, article, violation_type,
                    description, severity, risk_impact, affected_entities,
                    detection_method, detection_timestamp, remediation_actions,
                    investigation_required, escalation_level
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            `, [
                violationId, eventId, violationData.regulation, violationData.article || '',
                violationData.violation_type, violationData.description, violationData.severity,
                violationData.risk_impact || 'medium', JSON.stringify(violationData.affected_entities || []),
                'manual_entry', new Date(), JSON.stringify(violationData.remediation_actions || []),
                violationData.investigation_required || false, violationData.escalation_level || 1
            ]);

            performanceMetrics.violations_detected++;

            return {
                success: true,
                violation_id: violationId,
                event_id: eventId,
                message: 'Violation created successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to create violation:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to create violation'
            });
        }
    });
}

/**
 * Register investigation routes
 */
async function registerInvestigationRoutes(app) {
    // Create investigation
    app.post('/api/compliance-audit/investigations', async (request, reply) => {
        try {
            const investigationData = request.body;

            const requiredFields = ['title', 'description', 'investigation_type', 'priority', 'assigned_investigator'];
            for (const field of requiredFields) {
                if (!investigationData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            const investigationId = await callPythonAgent('create_investigation', investigationData);

            performanceMetrics.investigations_created++;

            // Broadcast to WebSocket clients
            broadcastToClients({
                type: 'investigation_created',
                data: {
                    investigation_id: investigationId,
                    title: investigationData.title,
                    priority: investigationData.priority,
                    assigned_investigator: investigationData.assigned_investigator,
                    timestamp: new Date().toISOString()
                }
            });

            return {
                success: true,
                investigation_id: investigationId,
                message: 'Investigation created successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to create investigation:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to create investigation'
            });
        }
    });

    // Get investigations
    app.get('/api/compliance-audit/investigations', async (request, reply) => {
        try {
            const { status, priority, assigned_to, days = 90 } = request.query;

            let query = `
                SELECT * FROM compliance_investigation_summary
                WHERE created_timestamp > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (status) {
                query += ` AND status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            if (priority) {
                query += ` AND priority = $${paramIndex}`;
                params.push(priority);
                paramIndex++;
            }

            if (assigned_to) {
                query += ` AND assigned_investigator = $${paramIndex}`;
                params.push(assigned_to);
                paramIndex++;
            }

            query += ' ORDER BY created_timestamp DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                investigations: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get investigations:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve investigations'
            });
        }
    });

    // Update investigation
    app.put('/api/compliance-audit/investigations/:investigationId', async (request, reply) => {
        try {
            const { investigationId } = request.params;
            const updates = request.body;

            const allowedUpdates = [
                'status', 'findings', 'evidence_collected', 'conclusions',
                'recommendations', 'actions_taken', 'stakeholders'
            ];

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

            updateFields.push('updated_timestamp = CURRENT_TIMESTAMP');
            updateValues.push(investigationId);

            const updateQuery = `
                UPDATE compliance_investigations
                SET ${updateFields.join(', ')}
                WHERE investigation_id = $${paramIndex}
                RETURNING *
            `;

            const result = await dbPool.query(updateQuery, updateValues);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Investigation not found'
                });
                return;
            }

            // Log the update
            await callPythonAgent('log_compliance_event', {
                event_type: 'system_action',
                entity_type: 'compliance_investigation',
                entity_id: investigationId,
                action: 'investigation_updated',
                details: { updates, updated_by: request.user?.id || 'system' },
                severity: 'info'
            });

            return {
                success: true,
                investigation: result.rows[0],
                message: 'Investigation updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update investigation:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update investigation'
            });
        }
    });
}

/**
 * Register reporting routes
 */
async function registerReportingRoutes(app) {
    // Generate compliance report
    app.post('/api/compliance-audit/reports', async (request, reply) => {
        try {
            const { report_type, regulation, start_date, end_date, format = 'json' } = request.body;

            if (!report_type || !start_date || !end_date) {
                reply.status(400).send({
                    success: false,
                    error: 'Report type, start date, and end date are required'
                });
                return;
            }

            const startTime = Date.now();

            // Generate report using Python agent
            const report = await callPythonAgent('generate_compliance_report', {
                report_type,
                regulation,
                start_date,
                end_date
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.reports_generated++;

            // Handle different output formats
            if (format === 'csv' && report.success) {
                // Convert to CSV format
                const csvData = await convertReportToCSV(report);
                reply.type('text/csv').send(csvData);
                return;
            } else if (format === 'pdf' && report.success) {
                // Generate PDF (would integrate with PDF generation library)
                reply.status(501).send({
                    success: false,
                    error: 'PDF format not yet implemented'
                });
                return;
            }

            return {
                success: true,
                report,
                processing_time_ms: processingTime,
                format
            };

        } catch (error) {
            fastify.log.error('Failed to generate report:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to generate report'
            });
        }
    });

    // Get existing reports
    app.get('/api/compliance-audit/reports', async (request, reply) => {
        try {
            const { report_type, regulation, days = 90 } = request.query;

            let query = `
                SELECT report_id, report_type, regulation, period_start, period_end,
                       generated_at, generated_by, file_path, file_size_bytes,
                       access_count, last_accessed
                FROM compliance_reports
                WHERE generated_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (report_type) {
                query += ` AND report_type = $${paramIndex}`;
                params.push(report_type);
                paramIndex++;
            }

            if (regulation) {
                query += ` AND regulation = $${paramIndex}`;
                params.push(regulation);
                paramIndex++;
            }

            query += ' ORDER BY generated_at DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                reports: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get reports:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve reports'
            });
        }
    });

    // Download report
    app.get('/api/compliance-audit/reports/:reportId/download', async (request, reply) => {
        try {
            const { reportId } = request.params;

            const result = await dbPool.query(`
                SELECT report_data, report_type, regulation, generated_at
                FROM compliance_reports 
                WHERE report_id = $1
            `, [reportId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Report not found'
                });
                return;
            }

            const report = result.rows[0];

            // Update access count
            await dbPool.query(`
                UPDATE compliance_reports 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE report_id = $1
            `, [reportId]);

            // Log access
            await callPythonAgent('log_compliance_event', {
                event_type: 'data_access',
                entity_type: 'compliance_report',
                entity_id: reportId,
                action: 'report_downloaded',
                details: { report_type: report.report_type, regulation: report.regulation },
                severity: 'info',
                user_id: request.user?.id
            });

            reply.type('application/json').send(report.report_data);

        } catch (error) {
            fastify.log.error('Failed to download report:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to download report'
            });
        }
    });
}

/**
 * Register integrity verification routes
 */
async function registerIntegrityVerificationRoutes(app) {
    // Verify audit trail integrity
    app.post('/api/compliance-audit/verify-integrity', async (request, reply) => {
        try {
            const { start_date, end_date } = request.body;

            const startTime = Date.now();

            const verificationResult = await callPythonAgent('verify_audit_trail_integrity', {
                start_date,
                end_date
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.integrity_checks++;

            // Store verification result
            await dbPool.query(`
                INSERT INTO audit_trail_integrity (
                    verification_id, verification_timestamp, period_start, period_end,
                    total_events_verified, valid_events, invalid_events,
                    integrity_score, integrity_status, verification_details,
                    verified_by, verification_duration_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            `, [
                crypto.randomUUID(), new Date(), start_date, end_date,
                verificationResult.total_events, verificationResult.valid_events,
                verificationResult.invalid_events, verificationResult.integrity_score,
                verificationResult.integrity_status, JSON.stringify(verificationResult.verification_details),
                request.user?.id || 'system', processingTime
            ]);

            return {
                success: true,
                verification: verificationResult,
                processing_time_ms: processingTime
            };

        } catch (error) {
            fastify.log.error('Integrity verification failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Integrity verification failed'
            });
        }
    });

    // Get integrity verification history
    app.get('/api/compliance-audit/integrity-history', async (request, reply) => {
        try {
            const { days = 30 } = request.query;

            const result = await dbPool.query(`
                SELECT verification_id, verification_timestamp, period_start, period_end,
                       total_events_verified, integrity_score, integrity_status,
                       verified_by, verification_duration_ms
                FROM audit_trail_integrity
                WHERE verification_timestamp > NOW() - INTERVAL '${days} days'
                ORDER BY verification_timestamp DESC
            `);

            return {
                success: true,
                integrity_history: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get integrity history:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve integrity history'
            });
        }
    });
}

/**
 * Register pattern analysis routes
 */
async function registerPatternAnalysisRoutes(app) {
    // Analyze compliance patterns
    app.post('/api/compliance-audit/analyze-patterns', async (request, reply) => {
        try {
            const { regulation, days = 30 } = request.body;

            const analysisResult = await callPythonAgent('analyze_compliance_patterns', {
                regulation,
                days
            });

            return {
                success: true,
                analysis: analysisResult
            };

        } catch (error) {
            fastify.log.error('Pattern analysis failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Pattern analysis failed'
            });
        }
    });

    // Get detected patterns
    app.get('/api/compliance-audit/patterns', async (request, reply) => {
        try {
            const { pattern_type, active_only = true } = request.query;

            let query = `
                SELECT * FROM compliance_patterns
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (pattern_type) {
                query += ` AND pattern_type = $${paramIndex}`;
                params.push(pattern_type);
                paramIndex++;
            }

            if (active_only === 'true') {
                query += ` AND is_active = true`;
            }

            query += ' ORDER BY confidence_score DESC, first_detected DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                patterns: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get patterns:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve patterns'
            });
        }
    });
}

/**
 * Register advanced search routes
 */
async function registerSearchRoutes(app) {
    // Advanced search across audit trail
    app.post('/api/compliance-audit/search', async (request, reply) => {
        try {
            const { query, filters = {}, page = 1, limit = 50 } = request.body;

            if (!query) {
                reply.status(400).send({
                    success: false,
                    error: 'Search query is required'
                });
                return;
            }

            const startTime = Date.now();

            // Full-text search using PostgreSQL
            let searchQuery = `
                SELECT csi.entity_id, csi.entity_type, cae.timestamp, cae.action,
                       cae.regulation, cae.severity, cae.user_id,
                       ts_rank(to_tsvector('english', csi.searchable_content), plainto_tsquery('english', $1)) as rank
                FROM compliance_search_index csi
                JOIN compliance_audit_events cae ON csi.entity_id = cae.event_id::text
                WHERE to_tsvector('english', csi.searchable_content) @@ plainto_tsquery('english', $1)
            `;
            const params = [query];
            let paramIndex = 2;

            // Add filters
            if (filters.regulation) {
                searchQuery += ` AND cae.regulation = $${paramIndex}`;
                params.push(filters.regulation);
                paramIndex++;
            }

            if (filters.severity) {
                searchQuery += ` AND cae.severity = $${paramIndex}`;
                params.push(filters.severity);
                paramIndex++;
            }

            if (filters.start_date) {
                searchQuery += ` AND cae.timestamp >= $${paramIndex}`;
                params.push(filters.start_date);
                paramIndex++;
            }

            if (filters.end_date) {
                searchQuery += ` AND cae.timestamp <= $${paramIndex}`;
                params.push(filters.end_date);
                paramIndex++;
            }

            searchQuery += ` ORDER BY rank DESC, cae.timestamp DESC`;
            searchQuery += ` LIMIT $${paramIndex} OFFSET $${paramIndex + 1}`;
            params.push(parseInt(limit));
            params.push((parseInt(page) - 1) * parseInt(limit));

            const result = await dbPool.query(searchQuery, params);

            const processingTime = Date.now() - startTime;
            performanceMetrics.search_queries++;

            return {
                success: true,
                results: result.rows,
                query,
                filters,
                processing_time_ms: processingTime,
                pagination: {
                    page: parseInt(page),
                    limit: parseInt(limit)
                }
            };

        } catch (error) {
            fastify.log.error('Search failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Search failed'
            });
        }
    });
}

/**
 * Register stakeholder management routes
 */
async function registerStakeholderRoutes(app) {
    // Get compliance stakeholders
    app.get('/api/compliance-audit/stakeholders', async (request, reply) => {
        try {
            const { role, department, active_only = true } = request.query;

            let query = `
                SELECT * FROM compliance_stakeholders
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (role) {
                query += ` AND role = $${paramIndex}`;
                params.push(role);
                paramIndex++;
            }

            if (department) {
                query += ` AND department = $${paramIndex}`;
                params.push(department);
                paramIndex++;
            }

            if (active_only === 'true') {
                query += ` AND is_active = true`;
            }

            query += ' ORDER BY role, name';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                stakeholders: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get stakeholders:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve stakeholders'
            });
        }
    });

    // Add compliance stakeholder
    app.post('/api/compliance-audit/stakeholders', async (request, reply) => {
        try {
            const stakeholderData = request.body;

            const requiredFields = ['stakeholder_id', 'name', 'role', 'email'];
            for (const field of requiredFields) {
                if (!stakeholderData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            const result = await dbPool.query(`
                INSERT INTO compliance_stakeholders (
                    stakeholder_id, name, role, department, email, phone,
                    responsibilities, regulations_assigned, notification_preferences,
                    escalation_level
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING *
            `, [
                stakeholderData.stakeholder_id, stakeholderData.name, stakeholderData.role,
                stakeholderData.department, stakeholderData.email, stakeholderData.phone,
                JSON.stringify(stakeholderData.responsibilities || []),
                JSON.stringify(stakeholderData.regulations_assigned || []),
                JSON.stringify(stakeholderData.notification_preferences || {}),
                stakeholderData.escalation_level || 1
            ]);

            return {
                success: true,
                stakeholder: result.rows[0],
                message: 'Stakeholder added successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to add stakeholder:', error);
            
            if (error.code === '23505') {
                reply.status(409).send({
                    success: false,
                    error: 'Stakeholder with this ID already exists'
                });
            } else {
                reply.status(500).send({
                    success: false,
                    error: 'Failed to add stakeholder'
                });
            }
        }
    });
}

/**
 * Register metrics routes
 */
async function registerMetricsRoutes(app) {
    // Get compliance metrics
    app.get('/api/compliance-audit/metrics', async (request, reply) => {
        try {
            const { metric_name, regulation, days = 30 } = request.query;

            let query = `
                SELECT * FROM compliance_metrics
                WHERE measurement_period_start > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (metric_name) {
                query += ` AND metric_name = $${paramIndex}`;
                params.push(metric_name);
                paramIndex++;
            }

            if (regulation) {
                query += ` AND regulation = $${paramIndex}`;
                params.push(regulation);
                paramIndex++;
            }

            query += ' ORDER BY calculated_at DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                metrics: result.rows,
                system_metrics: performanceMetrics
            };

        } catch (error) {
            fastify.log.error('Failed to get metrics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve metrics'
            });
        }
    });

    // Calculate compliance metrics manually
    app.post('/api/compliance-audit/calculate-metrics', async (request, reply) => {
        try {
            await dbPool.query('SELECT calculate_compliance_metrics()');

            return {
                success: true,
                message: 'Compliance metrics calculated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to calculate metrics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to calculate metrics'
            });
        }
    });
}

/**
 * Register notification routes
 */
async function registerNotificationRoutes(app) {
    // Get compliance notifications
    app.get('/api/compliance-audit/notifications', async (request, reply) => {
        try {
            const { notification_type, severity, status, days = 7 } = request.query;

            let query = `
                SELECT * FROM compliance_notifications
                WHERE created_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (notification_type) {
                query += ` AND notification_type = $${paramIndex}`;
                params.push(notification_type);
                paramIndex++;
            }

            if (severity) {
                query += ` AND severity = $${paramIndex}`;
                params.push(severity);
                paramIndex++;
            }

            if (status) {
                query += ` AND delivery_status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            query += ' ORDER BY created_at DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                notifications: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get notifications:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve notifications'
            });
        }
    });
}

/**
 * Register data lineage routes
 */
async function registerDataLineageRoutes(app) {
    // Track data lineage
    app.post('/api/compliance-audit/data-lineage', async (request, reply) => {
        try {
            const lineageData = request.body;

            const requiredFields = ['source_entity_type', 'source_entity_id', 'target_entity_type', 'target_entity_id', 'transformation_type'];
            for (const field of requiredFields) {
                if (!lineageData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            const lineageId = crypto.randomUUID();
            await dbPool.query(`
                INSERT INTO compliance_data_lineage (
                    lineage_id, source_entity_type, source_entity_id,
                    target_entity_type, target_entity_id, transformation_type,
                    transformation_details, data_classification, compliance_impact,
                    lineage_timestamp, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            `, [
                lineageId, lineageData.source_entity_type, lineageData.source_entity_id,
                lineageData.target_entity_type, lineageData.target_entity_id,
                lineageData.transformation_type, JSON.stringify(lineageData.transformation_details || {}),
                lineageData.data_classification || 'internal', lineageData.compliance_impact || 'medium',
                new Date(), request.user?.id || 'system'
            ]);

            return {
                success: true,
                lineage_id: lineageId,
                message: 'Data lineage recorded successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to record data lineage:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to record data lineage'
            });
        }
    });

    // Get data lineage
    app.get('/api/compliance-audit/data-lineage/:entityType/:entityId', async (request, reply) => {
        try {
            const { entityType, entityId } = request.params;

            const upstreamResult = await dbPool.query(`
                SELECT * FROM compliance_data_lineage
                WHERE target_entity_type = $1 AND target_entity_id = $2
                ORDER BY lineage_timestamp DESC
            `, [entityType, entityId]);

            const downstreamResult = await dbPool.query(`
                SELECT * FROM compliance_data_lineage
                WHERE source_entity_type = $1 AND source_entity_id = $2
                ORDER BY lineage_timestamp DESC
            `, [entityType, entityId]);

            return {
                success: true,
                lineage: {
                    entity_type: entityType,
                    entity_id: entityId,
                    upstream: upstreamResult.rows,
                    downstream: downstreamResult.rows
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get data lineage:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve data lineage'
            });
        }
    });
}

/**
 * Convert report to CSV format
 */
async function convertReportToCSV(report) {
    try {
        // Extract tabular data from report
        const fields = ['timestamp', 'regulation', 'event_type', 'severity', 'compliance_status'];
        const opts = { fields };
        const parser = new Parser(opts);
        
        // This would extract relevant data from the report structure
        const csvData = parser.parse([]);
        return csvData;
    } catch (error) {
        throw new Error(`CSV conversion failed: ${error.message}`);
    }
}

module.exports = {
    initializeComplianceAuditTrailRoutes
};