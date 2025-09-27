/**
 * Document Authenticity Checker API Routes
 * Production-grade Fastify routes for comprehensive document verification and fraud detection
 * 
 * Features:
 * - Advanced computer vision document analysis
 * - Blockchain-based verification and tamper detection
 * - Machine learning fraud detection with ensemble methods
 * - Forensic-level document analysis with metadata extraction
 * - OCR with advanced text recognition and validation
 * - Image forensics including ELA, copy-move detection, and noise analysis
 * - Comprehensive audit trails and compliance reporting
 * - Real-time verification with batch processing capabilities
 */

const fastify = require('fastify')({ logger: true });
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');
const WebSocket = require('ws');
const crypto = require('crypto');
const multer = require('multer');
const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

// Database and Redis connections
let dbPool, redisClient;

// WebSocket server for real-time updates
let wsServer;

// Python agent process
let pythonAgent = null;

// Performance metrics
const performanceMetrics = {
    documents_verified: 0,
    fraud_detected: 0,
    blockchain_registrations: 0,
    avg_processing_time: 0,
    processing_times: [],
    batch_verifications: 0
};

/**
 * Initialize Document Authenticity Checker routes
 */
async function initializeDocumentAuthenticityRoutes(app, database, redis) {
    dbPool = database;
    redisClient = redis;

    // Initialize WebSocket server
    wsServer = new WebSocket.Server({ port: 8014 });
    
    wsServer.on('connection', (ws) => {
        fastify.log.info('WebSocket client connected to Document Authenticity Checker');
        
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                handleWebSocketMessage(ws, data);
            } catch (error) {
                ws.send(JSON.stringify({ error: 'Invalid message format' }));
            }
        });
        
        ws.on('close', () => {
            fastify.log.info('WebSocket client disconnected from Document Authenticity Checker');
        });
    });

    // Start Python agent
    await startPythonAgent();

    // Register multipart content type parser
    app.register(require('@fastify/multipart'));

    // Register all routes
    await registerDocumentVerificationRoutes(app);
    await registerBatchVerificationRoutes(app);
    await registerBlockchainRoutes(app);
    await registerFraudAnalysisRoutes(app);
    await registerTemplateManagementRoutes(app);
    await registerForensicsRoutes(app);
    await registerAnalyticsRoutes(app);
    await registerAlertManagementRoutes(app);
    await registerModelManagementRoutes(app);

    fastify.log.info('Document Authenticity Checker routes initialized');
}

/**
 * Start Python agent process
 */
async function startPythonAgent() {
    try {
        const agentPath = path.join(__dirname, '../agents/document_authenticity_checker/agent.py');
        
        pythonAgent = spawn('python3', [agentPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: {
                ...process.env,
                PYTHONPATH: path.join(__dirname, '../agents'),
                DATABASE_URL: process.env.DATABASE_URL,
                REDIS_URL: process.env.REDIS_URL,
                TESSDATA_PREFIX: process.env.TESSDATA_PREFIX || '/usr/share/tesseract-ocr/4.00/tessdata'
            }
        });

        pythonAgent.stdout.on('data', (data) => {
            fastify.log.info(`Document Authenticity Python Agent: ${data}`);
        });

        pythonAgent.stderr.on('data', (data) => {
            fastify.log.error(`Document Authenticity Python Agent Error: ${data}`);
        });

        pythonAgent.on('close', (code) => {
            fastify.log.warn(`Document Authenticity Python Agent exited with code ${code}`);
            // Auto-restart after 5 seconds
            setTimeout(startPythonAgent, 5000);
        });

        fastify.log.info('Document Authenticity Checker Python agent started');
    } catch (error) {
        fastify.log.error('Failed to start Document Authenticity Python agent:', error);
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
            reject(new Error('Document Authenticity Python agent timeout'));
        }, 600000); // 10 minute timeout for complex document analysis

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
        case 'subscribe_verifications':
            ws.verificationSubscription = true;
            break;
        case 'subscribe_fraud_alerts':
            ws.fraudAlertSubscription = true;
            break;
        case 'subscribe_blockchain_events':
            ws.blockchainSubscription = true;
            break;
        case 'get_real_time_stats':
            sendRealTimeStats(ws);
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
 * Send real-time statistics
 */
async function sendRealTimeStats(ws) {
    try {
        const stats = await getVerificationStats();
        ws.send(JSON.stringify({
            type: 'real_time_stats',
            data: stats,
            timestamp: new Date().toISOString()
        }));
    } catch (error) {
        ws.send(JSON.stringify({ error: 'Failed to get real-time stats' }));
    }
}

/**
 * Get verification statistics
 */
async function getVerificationStats() {
    try {
        const today = new Date().toISOString().split('T')[0];
        
        const dailyStats = await dbPool.query(`
            SELECT 
                COUNT(*) as total_verifications,
                COUNT(CASE WHEN authenticity_status = 'authentic' THEN 1 END) as authentic_count,
                COUNT(CASE WHEN authenticity_status = 'suspicious' THEN 1 END) as suspicious_count,
                COUNT(CASE WHEN authenticity_status = 'fraudulent' THEN 1 END) as fraudulent_count,
                AVG(confidence_score) as avg_confidence,
                AVG(processing_time_ms) as avg_processing_time
            FROM document_authenticity_verifications 
            WHERE DATE(verification_timestamp) = $1
        `, [today]);

        const alertStats = await dbPool.query(`
            SELECT 
                severity,
                COUNT(*) as count
            FROM document_verification_alerts 
            WHERE alert_status = 'active'
            GROUP BY severity
        `);

        return {
            daily_stats: dailyStats.rows[0] || {},
            alert_stats: alertStats.rows,
            system_metrics: performanceMetrics
        };
    } catch (error) {
        fastify.log.error('Failed to get verification stats:', error);
        throw error;
    }
}

/**
 * Register document verification routes
 */
async function registerDocumentVerificationRoutes(app) {
    // Single document verification
    app.post('/api/document-authenticity/verify', async (request, reply) => {
        try {
            const data = await request.file();
            
            if (!data) {
                reply.status(400).send({
                    success: false,
                    error: 'Document file is required'
                });
                return;
            }

            const startTime = Date.now();

            // Get file buffer
            const fileBuffer = await data.toBuffer();
            
            // Extract metadata from request
            const metadata = {
                filename: data.filename,
                mimetype: data.mimetype,
                encoding: data.encoding,
                upload_timestamp: new Date().toISOString(),
                uploaded_by: request.user?.id || 'anonymous'
            };

            // Determine document type from filename or content
            const documentType = determineDocumentType(data.filename, data.mimetype);

            // Call Python agent for verification
            const verificationResult = await callPythonAgent('verify_document_authenticity', {
                document_data: fileBuffer.toString('base64'),
                document_type: documentType,
                metadata: metadata
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.documents_verified++;
            performanceMetrics.processing_times.push(processingTime);

            // Update average processing time
            if (performanceMetrics.processing_times.length > 1000) {
                performanceMetrics.processing_times = performanceMetrics.processing_times.slice(-1000);
            }
            performanceMetrics.avg_processing_time = 
                performanceMetrics.processing_times.reduce((a, b) => a + b, 0) / 
                performanceMetrics.processing_times.length;

            // Check for fraud detection
            if (verificationResult.authenticity_status === 'fraudulent' || 
                verificationResult.authenticity_status === 'suspicious') {
                performanceMetrics.fraud_detected++;
                
                // Generate alert
                await generateFraudAlert(verificationResult);
            }

            // Broadcast to WebSocket clients
            broadcastToClients({
                type: 'document_verified',
                data: {
                    document_id: verificationResult.document_id,
                    authenticity_status: verificationResult.authenticity_status,
                    confidence_score: verificationResult.confidence_score,
                    fraud_indicators: verificationResult.fraud_indicators,
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                verification: verificationResult,
                processing_time_ms: processingTime,
                message: 'Document verification completed'
            };

        } catch (error) {
            fastify.log.error('Document verification failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'Document verification failed'
            });
        }
    });

    // Get verification result by ID
    app.get('/api/document-authenticity/verifications/:verificationId', async (request, reply) => {
        try {
            const { verificationId } = request.params;

            const result = await dbPool.query(`
                SELECT dav.*, dma.*, doa.*, dif.*, dfa.*
                FROM document_authenticity_verifications dav
                LEFT JOIN document_metadata_analysis dma ON dav.document_id = dma.document_id
                LEFT JOIN document_ocr_analysis doa ON dav.document_id = doa.document_id
                LEFT JOIN document_image_forensics dif ON dav.document_id = dif.document_id
                LEFT JOIN document_fraud_analysis dfa ON dav.document_id = dfa.document_id
                WHERE dav.verification_id = $1
            `, [verificationId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Verification result not found'
                });
                return;
            }

            const verification = result.rows[0];

            // Get blockchain verification if available
            const blockchainResult = await dbPool.query(`
                SELECT * FROM document_blockchain_verification WHERE document_id = $1
            `, [verification.document_id]);

            // Get digital signature verification
            const signatureResult = await dbPool.query(`
                SELECT * FROM document_digital_signatures WHERE document_id = $1
            `, [verification.document_id]);

            return {
                success: true,
                verification: {
                    ...verification,
                    blockchain_verification: blockchainResult.rows[0] || null,
                    digital_signature_verification: signatureResult.rows[0] || null
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get verification result:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve verification result'
            });
        }
    });

    // Get verification history
    app.get('/api/document-authenticity/verifications', async (request, reply) => {
        try {
            const {
                document_type,
                authenticity_status,
                start_date,
                end_date,
                page = 1,
                limit = 50
            } = request.query;

            let query = `
                SELECT dav.verification_id, dav.document_id, dav.document_type,
                       dav.authenticity_status, dav.confidence_score, dav.verification_timestamp,
                       dav.processing_time_ms, dav.file_name, dav.file_size,
                       dfa.overall_fraud_probability, dfa.fraud_classification
                FROM document_authenticity_verifications dav
                LEFT JOIN document_fraud_analysis dfa ON dav.document_id = dfa.document_id
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (document_type) {
                query += ` AND dav.document_type = $${paramIndex}`;
                params.push(document_type);
                paramIndex++;
            }

            if (authenticity_status) {
                query += ` AND dav.authenticity_status = $${paramIndex}`;
                params.push(authenticity_status);
                paramIndex++;
            }

            if (start_date) {
                query += ` AND dav.verification_timestamp >= $${paramIndex}`;
                params.push(start_date);
                paramIndex++;
            }

            if (end_date) {
                query += ` AND dav.verification_timestamp <= $${paramIndex}`;
                params.push(end_date);
                paramIndex++;
            }

            query += ` ORDER BY dav.verification_timestamp DESC`;
            query += ` LIMIT $${paramIndex} OFFSET $${paramIndex + 1}`;
            params.push(parseInt(limit));
            params.push((parseInt(page) - 1) * parseInt(limit));

            const result = await dbPool.query(query, params);

            return {
                success: true,
                verifications: result.rows,
                pagination: {
                    page: parseInt(page),
                    limit: parseInt(limit)
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get verification history:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve verification history'
            });
        }
    });

    // Update verification status (manual review)
    app.put('/api/document-authenticity/verifications/:verificationId', async (request, reply) => {
        try {
            const { verificationId } = request.params;
            const { authenticity_status, analyst_notes, reason } = request.body;

            if (!authenticity_status) {
                reply.status(400).send({
                    success: false,
                    error: 'Authenticity status is required'
                });
                return;
            }

            // Get current verification
            const currentResult = await dbPool.query(`
                SELECT * FROM document_authenticity_verifications WHERE verification_id = $1
            `, [verificationId]);

            if (currentResult.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Verification not found'
                });
                return;
            }

            const currentVerification = currentResult.rows[0];

            // Update verification
            const result = await dbPool.query(`
                UPDATE document_authenticity_verifications
                SET authenticity_status = $1, analyst_notes = $2
                WHERE verification_id = $3
                RETURNING *
            `, [authenticity_status, analyst_notes, verificationId]);

            // Create audit record
            await dbPool.query(`
                INSERT INTO document_verification_audit (
                    document_id, audit_action, performed_by, previous_status,
                    new_status, reason, action_details
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            `, [
                currentVerification.document_id, 'manual_review_update',
                request.user?.id || 'system', currentVerification.authenticity_status,
                authenticity_status, reason, JSON.stringify({ analyst_notes })
            ]);

            return {
                success: true,
                verification: result.rows[0],
                message: 'Verification status updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update verification:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update verification'
            });
        }
    });
}

/**
 * Register batch verification routes
 */
async function registerBatchVerificationRoutes(app) {
    // Batch document verification
    app.post('/api/document-authenticity/verify-batch', async (request, reply) => {
        try {
            const parts = await request.saveRequestFiles();
            
            if (!parts || parts.length === 0) {
                reply.status(400).send({
                    success: false,
                    error: 'No documents provided for batch verification'
                });
                return;
            }

            if (parts.length > 50) {
                reply.status(400).send({
                    success: false,
                    error: 'Maximum 50 documents per batch'
                });
                return;
            }

            const startTime = Date.now();
            const verificationResults = [];
            const errors = [];

            // Process each document
            for (const part of parts) {
                try {
                    const fileBuffer = await fs.readFile(part.filepath);
                    
                    const metadata = {
                        filename: part.filename,
                        mimetype: part.mimetype,
                        upload_timestamp: new Date().toISOString(),
                        batch_id: crypto.randomUUID()
                    };

                    const documentType = determineDocumentType(part.filename, part.mimetype);

                    const verificationResult = await callPythonAgent('verify_document_authenticity', {
                        document_data: fileBuffer.toString('base64'),
                        document_type: documentType,
                        metadata: metadata
                    });

                    verificationResults.push(verificationResult);

                    // Clean up temporary file
                    await fs.unlink(part.filepath);

                } catch (error) {
                    errors.push({
                        filename: part.filename,
                        error: error.message
                    });
                }
            }

            const processingTime = Date.now() - startTime;
            performanceMetrics.batch_verifications++;
            performanceMetrics.documents_verified += verificationResults.length;

            // Count fraud detections
            const fraudDetected = verificationResults.filter(r => 
                r.authenticity_status === 'fraudulent' || r.authenticity_status === 'suspicious'
            ).length;
            performanceMetrics.fraud_detected += fraudDetected;

            // Broadcast batch completion
            broadcastToClients({
                type: 'batch_verification_completed',
                data: {
                    total_documents: parts.length,
                    successful_verifications: verificationResults.length,
                    fraud_detected: fraudDetected,
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                batch_results: {
                    total_documents: parts.length,
                    successful_verifications: verificationResults.length,
                    failed_verifications: errors.length,
                    fraud_detected: fraudDetected,
                    processing_time_ms: processingTime
                },
                verifications: verificationResults,
                errors
            };

        } catch (error) {
            fastify.log.error('Batch verification failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Batch verification failed'
            });
        }
    });

    // Get batch verification status
    app.get('/api/document-authenticity/batch-status/:batchId', async (request, reply) => {
        try {
            const { batchId } = request.params;

            // Query for documents with matching batch ID in metadata
            const result = await dbPool.query(`
                SELECT dav.*, dma.*, dfa.fraud_classification, dfa.overall_fraud_probability
                FROM document_authenticity_verifications dav
                LEFT JOIN document_metadata_analysis dma ON dav.document_id = dma.document_id
                LEFT JOIN document_fraud_analysis dfa ON dav.document_id = dfa.document_id
                WHERE dav.verification_id IN (
                    SELECT verification_id FROM document_authenticity_verifications
                    WHERE verification_id::text LIKE '%${batchId}%'
                )
                ORDER BY dav.verification_timestamp DESC
            `);

            const batchStats = {
                total_documents: result.rows.length,
                authentic_documents: result.rows.filter(r => r.authenticity_status === 'authentic').length,
                suspicious_documents: result.rows.filter(r => r.authenticity_status === 'suspicious').length,
                fraudulent_documents: result.rows.filter(r => r.authenticity_status === 'fraudulent').length,
                avg_confidence: result.rows.length > 0 
                    ? result.rows.reduce((sum, r) => sum + (r.confidence_score || 0), 0) / result.rows.length 
                    : 0,
                avg_processing_time: result.rows.length > 0
                    ? result.rows.reduce((sum, r) => sum + (r.processing_time_ms || 0), 0) / result.rows.length
                    : 0
            };

            return {
                success: true,
                batch_id: batchId,
                batch_statistics: batchStats,
                documents: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get batch status:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve batch status'
            });
        }
    });
}

/**
 * Register blockchain verification routes
 */
async function registerBlockchainRoutes(app) {
    // Register document on blockchain
    app.post('/api/document-authenticity/blockchain/register', async (request, reply) => {
        try {
            const { document_hash, metadata } = request.body;

            if (!document_hash) {
                reply.status(400).send({
                    success: false,
                    error: 'Document hash is required'
                });
                return;
            }

            const blockchainResult = await callPythonAgent('register_document_on_blockchain', {
                document_hash,
                metadata: metadata || {}
            });

            if (blockchainResult.verification_status) {
                performanceMetrics.blockchain_registrations++;

                // Store blockchain verification
                await dbPool.query(`
                    INSERT INTO document_blockchain_verification (
                        document_id, blockchain_hash, transaction_id, block_number,
                        smart_contract_address, gas_used, confirmations, ipfs_hash,
                        verification_status, blockchain_timestamp
                    ) VALUES (
                        (SELECT document_id FROM document_authenticity_verifications WHERE document_hash_sha256 = $1),
                        $2, $3, $4, $5, $6, $7, $8, $9, $10
                    )
                `, [
                    document_hash, blockchainResult.blockchain_hash, blockchainResult.transaction_id,
                    blockchainResult.block_number, blockchainResult.smart_contract_address,
                    blockchainResult.gas_used, blockchainResult.confirmations, blockchainResult.ipfs_hash,
                    blockchainResult.verification_status, blockchainResult.timestamp
                ]);

                // Broadcast blockchain registration
                broadcastToClients({
                    type: 'blockchain_registration',
                    data: {
                        document_hash,
                        transaction_id: blockchainResult.transaction_id,
                        block_number: blockchainResult.block_number
                    }
                });
            }

            return {
                success: true,
                blockchain_verification: blockchainResult
            };

        } catch (error) {
            fastify.log.error('Blockchain registration failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Blockchain registration failed'
            });
        }
    });

    // Verify document on blockchain
    app.get('/api/document-authenticity/blockchain/verify/:documentHash', async (request, reply) => {
        try {
            const { documentHash } = request.params;

            const blockchainResult = await callPythonAgent('verify_document_on_blockchain', {
                document_hash: documentHash
            });

            return {
                success: true,
                blockchain_verification: blockchainResult
            };

        } catch (error) {
            fastify.log.error('Blockchain verification failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Blockchain verification failed'
            });
        }
    });
}

/**
 * Register fraud analysis routes
 */
async function registerFraudAnalysisRoutes(app) {
    // Get fraud analysis for document
    app.get('/api/document-authenticity/fraud-analysis/:documentId', async (request, reply) => {
        try {
            const { documentId } = request.params;

            const result = await dbPool.query(`
                SELECT dfa.*, dav.document_type, dav.authenticity_status
                FROM document_fraud_analysis dfa
                JOIN document_authenticity_verifications dav ON dfa.document_id = dav.document_id
                WHERE dfa.document_id = $1
            `, [documentId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Fraud analysis not found'
                });
                return;
            }

            return {
                success: true,
                fraud_analysis: result.rows[0]
            };

        } catch (error) {
            fastify.log.error('Failed to get fraud analysis:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve fraud analysis'
            });
        }
    });

    // Get fraud statistics
    app.get('/api/document-authenticity/fraud-statistics', async (request, reply) => {
        try {
            const { days = 30, document_type } = request.query;

            let query = `
                SELECT 
                    fraud_classification,
                    COUNT(*) as count,
                    AVG(overall_fraud_probability) as avg_probability,
                    AVG(structure_fraud_score) as avg_structure_score,
                    AVG(content_fraud_score) as avg_content_score,
                    AVG(technical_fraud_score) as avg_technical_score
                FROM document_fraud_analysis dfa
                JOIN document_authenticity_verifications dav ON dfa.document_id = dav.document_id
                WHERE dfa.created_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];

            if (document_type) {
                query += ' AND dav.document_type = $1';
                params.push(document_type);
            }

            query += ' GROUP BY fraud_classification ORDER BY count DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                fraud_statistics: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get fraud statistics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve fraud statistics'
            });
        }
    });
}

/**
 * Register template management routes
 */
async function registerTemplateManagementRoutes(app) {
    // Get document templates
    app.get('/api/document-authenticity/templates', async (request, reply) => {
        try {
            const { document_type, issuing_authority, active_only = true } = request.query;

            let query = `
                SELECT * FROM document_templates
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (document_type) {
                query += ` AND document_type = $${paramIndex}`;
                params.push(document_type);
                paramIndex++;
            }

            if (issuing_authority) {
                query += ` AND issuing_authority = $${paramIndex}`;
                params.push(issuing_authority);
                paramIndex++;
            }

            if (active_only === 'true') {
                query += ` AND is_active = true`;
            }

            query += ' ORDER BY document_type, template_name';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                templates: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get templates:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve templates'
            });
        }
    });

    // Add document template
    app.post('/api/document-authenticity/templates', async (request, reply) => {
        try {
            const templateData = request.body;

            const requiredFields = ['template_name', 'document_type', 'issuing_authority', 'template_features'];
            for (const field of requiredFields) {
                if (!templateData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            const templateId = crypto.randomUUID();
            const result = await dbPool.query(`
                INSERT INTO document_templates (
                    template_id, template_name, document_type, issuing_authority,
                    template_version, valid_from, valid_to, template_features,
                    security_features, layout_specifications, font_specifications,
                    color_specifications, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING *
            `, [
                templateId, templateData.template_name, templateData.document_type,
                templateData.issuing_authority, templateData.template_version || '1.0',
                templateData.valid_from || new Date(), templateData.valid_to,
                JSON.stringify(templateData.template_features),
                JSON.stringify(templateData.security_features || {}),
                JSON.stringify(templateData.layout_specifications || {}),
                JSON.stringify(templateData.font_specifications || {}),
                JSON.stringify(templateData.color_specifications || {}),
                request.user?.id || 'system'
            ]);

            return {
                success: true,
                template: result.rows[0],
                message: 'Document template added successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to add template:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to add template'
            });
        }
    });
}

/**
 * Register forensics analysis routes
 */
async function registerForensicsRoutes(app) {
    // Get forensics analysis for document
    app.get('/api/document-authenticity/forensics/:documentId', async (request, reply) => {
        try {
            const { documentId } = request.params;

            const result = await dbPool.query(`
                SELECT * FROM document_image_forensics WHERE document_id = $1
            `, [documentId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Forensics analysis not found'
                });
                return;
            }

            return {
                success: true,
                forensics_analysis: result.rows[0]
            };

        } catch (error) {
            fastify.log.error('Failed to get forensics analysis:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve forensics analysis'
            });
        }
    });

    // Get OCR analysis for document
    app.get('/api/document-authenticity/ocr/:documentId', async (request, reply) => {
        try {
            const { documentId } = request.params;

            const result = await dbPool.query(`
                SELECT * FROM document_ocr_analysis WHERE document_id = $1
            `, [documentId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'OCR analysis not found'
                });
                return;
            }

            return {
                success: true,
                ocr_analysis: result.rows[0]
            };

        } catch (error) {
            fastify.log.error('Failed to get OCR analysis:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve OCR analysis'
            });
        }
    });
}

/**
 * Register analytics routes
 */
async function registerAnalyticsRoutes(app) {
    // Get verification analytics
    app.get('/api/document-authenticity/analytics/dashboard', async (request, reply) => {
        try {
            const { days = 30 } = request.query;

            const verificationTrends = await dbPool.query(`
                SELECT * FROM document_verification_summary
                WHERE verification_date >= CURRENT_DATE - INTERVAL '${days} days'
            `);

            const fraudTrends = await dbPool.query(`
                SELECT * FROM document_fraud_trends
                WHERE week_start >= CURRENT_DATE - INTERVAL '${days} days'
            `);

            const performanceMetrics = await dbPool.query(`
                SELECT 
                    document_type,
                    AVG(processing_time_ms) as avg_processing_time,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(*) as total_verifications
                FROM document_authenticity_verifications
                WHERE verification_timestamp >= NOW() - INTERVAL '${days} days'
                GROUP BY document_type
                ORDER BY total_verifications DESC
            `);

            return {
                success: true,
                analytics: {
                    verification_trends: verificationTrends.rows,
                    fraud_trends: fraudTrends.rows,
                    performance_metrics: performanceMetrics.rows,
                    system_metrics: performanceMetrics
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get analytics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve analytics'
            });
        }
    });

    // Calculate verification statistics
    app.post('/api/document-authenticity/calculate-stats', async (request, reply) => {
        try {
            await dbPool.query('SELECT calculate_document_verification_stats()');

            return {
                success: true,
                message: 'Verification statistics calculated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to calculate stats:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to calculate statistics'
            });
        }
    });
}

/**
 * Register alert management routes
 */
async function registerAlertManagementRoutes(app) {
    // Get verification alerts
    app.get('/api/document-authenticity/alerts', async (request, reply) => {
        try {
            const { severity, status, days = 7 } = request.query;

            let query = `
                SELECT * FROM active_document_alerts
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
            fastify.log.error('Failed to get alerts:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve alerts'
            });
        }
    });

    // Acknowledge alert
    app.put('/api/document-authenticity/alerts/:alertId/acknowledge', async (request, reply) => {
        try {
            const { alertId } = request.params;
            const { acknowledgment_notes } = request.body;

            const result = await dbPool.query(`
                UPDATE document_verification_alerts
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
}

/**
 * Register model management routes
 */
async function registerModelManagementRoutes(app) {
    // Get ML model performance
    app.get('/api/document-authenticity/models', async (request, reply) => {
        try {
            const { model_type, active_only = true } = request.query;

            let query = `
                SELECT * FROM document_ml_models
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (model_type) {
                query += ` AND model_type = $${paramIndex}`;
                params.push(model_type);
                paramIndex++;
            }

            if (active_only === 'true') {
                query += ` AND is_active = true`;
            }

            query += ' ORDER BY training_date DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                models: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get models:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve models'
            });
        }
    });

    // Train fraud detection model
    app.post('/api/document-authenticity/train-model', async (request, reply) => {
        try {
            const { model_type, training_parameters } = request.body;

            if (!model_type) {
                reply.status(400).send({
                    success: false,
                    error: 'Model type is required'
                });
                return;
            }

            const trainingResult = await callPythonAgent('train_fraud_detection_model', {
                model_type,
                parameters: training_parameters || {}
            });

            return {
                success: true,
                training_result: trainingResult
            };

        } catch (error) {
            fastify.log.error('Model training failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Model training failed'
            });
        }
    });
}

/**
 * Determine document type from filename and MIME type
 */
function determineDocumentType(filename, mimetype) {
    const lowerFilename = filename.toLowerCase();
    
    if (lowerFilename.includes('passport') || lowerFilename.includes('id')) {
        return 'identity_document';
    } else if (lowerFilename.includes('bank') || lowerFilename.includes('statement')) {
        return 'financial_statement';
    } else if (lowerFilename.includes('payslip') || lowerFilename.includes('employment')) {
        return 'employment_document';
    } else if (lowerFilename.includes('property') || lowerFilename.includes('deed')) {
        return 'property_document';
    } else if (mimetype === 'application/pdf') {
        return 'financial_statement';  // Default for PDFs
    } else {
        return 'unknown';
    }
}

/**
 * Generate fraud alert
 */
async function generateFraudAlert(verificationResult) {
    try {
        const alertId = crypto.randomUUID();
        
        await dbPool.query(`
            INSERT INTO document_verification_alerts (
                alert_id, document_id, alert_type, severity, title, description,
                fraud_probability, recommended_actions, escalation_level
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        `, [
            alertId, verificationResult.document_id, 'fraud_detection',
            verificationResult.authenticity_status === 'fraudulent' ? 'critical' : 'high',
            `Document Fraud Detected: ${verificationResult.document_id}`,
            `Document verification detected ${verificationResult.authenticity_status} status`,
            verificationResult.fraud_analysis?.overall_fraud_probability || 0.8,
            JSON.stringify(verificationResult.recommendations.slice(0, 3)),
            verificationResult.authenticity_status === 'fraudulent' ? 3 : 2
        ]);

        // Broadcast alert
        broadcastToClients({
            type: 'fraud_alert_generated',
            data: {
                alert_id: alertId,
                document_id: verificationResult.document_id,
                authenticity_status: verificationResult.authenticity_status,
                confidence_score: verificationResult.confidence_score
            }
        });

    } catch (error) {
        fastify.log.error('Failed to generate fraud alert:', error);
    }
}

module.exports = {
    initializeDocumentAuthenticityRoutes
};