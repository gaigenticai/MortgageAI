/**
 * Computer Vision Document Verification API Routes (Fastify)
 * 
 * Provides RESTful API endpoints for document verification using advanced computer vision
 * including forgery detection, signature analysis, tampering detection, and authenticity scoring.
 * 
 * Features:
 * - Single document verification
 * - Batch document verification
 * - Signature verification with reference comparison
 * - Real-time verification status
 * - Comprehensive audit trails
 * - Blockchain-based verification logging
 */

const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');

/**
 * Execute Python CV verification script
 */
async function executeCVVerification(documentPath, referencePaths = [], metadata = {}) {
    return new Promise((resolve, reject) => {
        const pythonScript = path.join(__dirname, '../agents/utils/cv_verification_executor.py');
        const args = [
            pythonScript,
            '--document', documentPath,
            '--metadata', JSON.stringify(metadata)
        ];
        
        if (referencePaths.length > 0) {
            args.push('--references', JSON.stringify(referencePaths));
        }
        
        const pythonProcess = spawn('python3', args);
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
                } catch (error) {
                    reject(new Error(`Failed to parse CV verification result: ${error.message}`));
                }
            } else {
                reject(new Error(`CV verification failed with code ${code}: ${stderr}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            reject(new Error(`Failed to start CV verification: ${error.message}`));
        });
    });
}

/**
 * Store verification results in database
 */
async function storeVerificationResult(fastify, verificationData) {
    const query = `
        INSERT INTO cv_verification_results (
            id, document_hash, document_path, verification_status, 
            overall_confidence, forgery_probability, signature_authenticity,
            tampering_evidence, metadata_analysis, image_forensics,
            blockchain_hash, verification_timestamp, processing_time,
            user_id, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
        ) RETURNING id
    `;
    
    const values = [
        verificationData.id,
        verificationData.document_hash,
        verificationData.document_path,
        verificationData.verification_status,
        verificationData.overall_confidence,
        verificationData.forgery_probability,
        verificationData.signature_authenticity,
        JSON.stringify(verificationData.tampering_evidence),
        JSON.stringify(verificationData.metadata_analysis),
        JSON.stringify(verificationData.image_forensics),
        verificationData.blockchain_hash,
        verificationData.verification_timestamp,
        verificationData.processing_time,
        verificationData.user_id,
        new Date()
    ];
    
    const result = await fastify.pg.query(query, values);
    return result.rows[0];
}

/**
 * Save uploaded file to disk
 */
async function saveUploadedFile(data, filename) {
    const uploadDir = path.join(process.cwd(), 'uploads', 'cv_verification');
    await fs.mkdir(uploadDir, { recursive: true });
    
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const safeFilename = `cv_verification_${uniqueSuffix}_${filename}`;
    const filePath = path.join(uploadDir, safeFilename);
    
    await fs.writeFile(filePath, data.buffer);
    return filePath;
}

/**
 * Register CV verification routes with Fastify
 */
async function cvVerificationRoutes(fastify, options) {
    
    // Register database connection plugin if not already registered
    if (!fastify.pg) {
        await fastify.register(require('@fastify/postgres'), {
            connectionString: `postgresql://${process.env.DB_USER}:${process.env.DB_PASSWORD}@${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_NAME}`
        });
    }
    
    /**
     * GET /status
     * Get verification system status and statistics
     */
    fastify.get('/status', async (request, reply) => {
        try {
            // Get verification statistics
            const statsQuery = `
                SELECT 
                    COUNT(*) as total_verifications,
                    COUNT(CASE WHEN verification_status = 'authentic' THEN 1 END) as authentic_count,
                    COUNT(CASE WHEN verification_status = 'suspicious' THEN 1 END) as suspicious_count,
                    COUNT(CASE WHEN verification_status = 'fraudulent' THEN 1 END) as fraudulent_count,
                    COUNT(CASE WHEN verification_status = 'inconclusive' THEN 1 END) as inconclusive_count,
                    AVG(overall_confidence) as average_confidence,
                    AVG(processing_time) as average_processing_time
                FROM cv_verification_results 
                WHERE created_at >= NOW() - INTERVAL '30 days'
            `;
            
            const statsResult = await fastify.pg.query(statsQuery);
            const stats = statsResult.rows[0];
            
            // System health check
            const systemStatus = {
                status: 'operational',
                models_loaded: true,
                gpu_available: process.env.CUDA_AVAILABLE === 'true',
                blockchain_enabled: process.env.BLOCKCHAIN_VERIFICATION_ENABLED === 'true',
                last_updated: new Date()
            };
            
            return {
                success: true,
                data: {
                    system_status: systemStatus,
                    statistics: {
                        total_verifications: parseInt(stats.total_verifications),
                        status_distribution: {
                            authentic: parseInt(stats.authentic_count || 0),
                            suspicious: parseInt(stats.suspicious_count || 0),
                            fraudulent: parseInt(stats.fraudulent_count || 0),
                            inconclusive: parseInt(stats.inconclusive_count || 0)
                        },
                        average_confidence: parseFloat(stats.average_confidence || 0),
                        average_processing_time: parseFloat(stats.average_processing_time || 0)
                    }
                }
            };
        } catch (error) {
            fastify.log.error('CV verification status error:', error);
            reply.code(500);
            return {
                success: false,
                error: 'Failed to get verification system status',
                details: error.message
            };
        }
    });

    /**
     * POST /verify
     * Verify a single document with optional reference signatures
     */
    fastify.post('/verify', {
        schema: {
            consumes: ['multipart/form-data'],
            body: {
                type: 'object',
                properties: {
                    document: { type: 'object' },
                    references: { type: 'array' },
                    metadata: { type: 'string' },
                    include_blockchain: { type: 'string' },
                    include_details: { type: 'string' }
                },
                required: ['document']
            }
        }
    }, async (request, reply) => {
        try {
            const parts = request.parts();
            let documentFile = null;
            let referenceFiles = [];
            let metadata = {};
            let includeBlockchain = false;
            let includeDetails = false;
            
            // Process multipart form data
            for await (const part of parts) {
                if (part.type === 'file') {
                    const buffer = await part.toBuffer();
                    
                    if (part.fieldname === 'document') {
                        documentFile = {
                            filename: part.filename,
                            buffer: buffer,
                            mimetype: part.mimetype
                        };
                    } else if (part.fieldname === 'references') {
                        referenceFiles.push({
                            filename: part.filename,
                            buffer: buffer,
                            mimetype: part.mimetype
                        });
                    }
                } else if (part.type === 'field') {
                    if (part.fieldname === 'metadata') {
                        try {
                            metadata = JSON.parse(part.value);
                        } catch (e) {
                            // Ignore invalid JSON
                        }
                    } else if (part.fieldname === 'include_blockchain') {
                        includeBlockchain = part.value === 'true';
                    } else if (part.fieldname === 'include_details') {
                        includeDetails = part.value === 'true';
                    }
                }
            }
            
            if (!documentFile) {
                reply.code(400);
                return {
                    success: false,
                    error: 'Document file is required'
                };
            }
            
            // Validate file type
            const allowedMimes = [
                'image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp',
                'application/pdf'
            ];
            
            if (!allowedMimes.includes(documentFile.mimetype)) {
                reply.code(400);
                return {
                    success: false,
                    error: 'Invalid file type. Only images and PDFs are allowed.'
                };
            }
            
            // Save files to disk
            const documentPath = await saveUploadedFile(documentFile, documentFile.filename);
            const referencePaths = [];
            
            for (const refFile of referenceFiles) {
                const refPath = await saveUploadedFile(refFile, refFile.filename);
                referencePaths.push(refPath);
            }
            
            // Add file metadata
            metadata.file_info = {
                original_name: documentFile.filename,
                size: documentFile.buffer.length,
                mime_type: documentFile.mimetype
            };
            
            // Execute CV verification
            const cvResult = await executeCVVerification(documentPath, referencePaths, metadata);
            
            if (!cvResult.success) {
                reply.code(500);
                return {
                    success: false,
                    error: 'CV verification failed',
                    details: cvResult.error
                };
            }
            
            // Generate verification ID
            const verificationId = uuidv4();
            
            // Prepare verification data for storage
            const verificationData = {
                id: verificationId,
                document_hash: cvResult.result.document_hash,
                document_path: documentPath,
                verification_status: cvResult.result.verification_status,
                overall_confidence: cvResult.result.overall_confidence,
                forgery_probability: cvResult.result.forgery_probability,
                signature_authenticity: cvResult.result.signature_authenticity,
                tampering_evidence: cvResult.result.tampering_evidence,
                metadata_analysis: cvResult.result.metadata_analysis,
                image_forensics: cvResult.result.image_forensics,
                blockchain_hash: includeBlockchain ? cvResult.result.blockchain_hash : '',
                verification_timestamp: cvResult.result.verification_timestamp,
                processing_time: cvResult.result.processing_time,
                user_id: request.user?.id || null
            };
            
            // Store results in database
            await storeVerificationResult(fastify, verificationData);
            
            // Prepare response
            const response = {
                success: true,
                verification_id: verificationId,
                data: {
                    verification_status: cvResult.result.verification_status,
                    overall_confidence: cvResult.result.overall_confidence,
                    forgery_probability: cvResult.result.forgery_probability,
                    signature_authenticity: cvResult.result.signature_authenticity,
                    tampering_evidence: cvResult.result.tampering_evidence,
                    summary: {
                        document_hash: cvResult.result.document_hash,
                        processing_time: cvResult.result.processing_time,
                        timestamp: cvResult.result.verification_timestamp
                    }
                }
            };
            
            // Include detailed analysis if requested
            if (includeDetails) {
                response.data.detailed_analysis = {
                    metadata_analysis: cvResult.result.metadata_analysis,
                    image_forensics: cvResult.result.image_forensics
                };
            }
            
            // Include blockchain hash if enabled
            if (includeBlockchain && cvResult.result.blockchain_hash) {
                response.data.blockchain_hash = cvResult.result.blockchain_hash;
            }
            
            return response;
            
        } catch (error) {
            fastify.log.error('CV verification error:', error);
            reply.code(500);
            return {
                success: false,
                error: 'Document verification failed',
                details: error.message
            };
        }
    });

    /**
     * POST /batch-verify
     * Verify multiple documents in batch
     */
    fastify.post('/batch-verify', {
        schema: {
            consumes: ['multipart/form-data'],
            body: {
                type: 'object',
                properties: {
                    documents: { type: 'array' },
                    batch_metadata: { type: 'string' }
                }
            }
        }
    }, async (request, reply) => {
        try {
            const parts = request.parts();
            let documentFiles = [];
            let batchMetadata = {};
            
            // Process multipart form data
            for await (const part of parts) {
                if (part.type === 'file' && part.fieldname === 'documents') {
                    const buffer = await part.toBuffer();
                    documentFiles.push({
                        filename: part.filename,
                        buffer: buffer,
                        mimetype: part.mimetype
                    });
                } else if (part.type === 'field' && part.fieldname === 'batch_metadata') {
                    try {
                        batchMetadata = JSON.parse(part.value);
                    } catch (e) {
                        // Ignore invalid JSON
                    }
                }
            }
            
            if (documentFiles.length === 0) {
                reply.code(400);
                return {
                    success: false,
                    error: 'At least one document file is required'
                };
            }
            
            const batchId = uuidv4();
            const results = [];
            const errors = [];
            
            // Process documents sequentially to avoid resource overload
            for (let i = 0; i < documentFiles.length; i++) {
                const document = documentFiles[i];
                const documentMetadata = batchMetadata[document.filename] || {};
                
                try {
                    // Add file metadata
                    documentMetadata.file_info = {
                        original_name: document.filename,
                        size: document.buffer.length,
                        mime_type: document.mimetype
                    };
                    
                    // Save document to disk
                    const documentPath = await saveUploadedFile(document, document.filename);
                    
                    // Execute CV verification
                    const cvResult = await executeCVVerification(documentPath, [], documentMetadata);
                    
                    if (cvResult.success) {
                        // Store individual result
                        const verificationData = {
                            id: uuidv4(),
                            document_hash: cvResult.result.document_hash,
                            document_path: documentPath,
                            verification_status: cvResult.result.verification_status,
                            overall_confidence: cvResult.result.overall_confidence,
                            forgery_probability: cvResult.result.forgery_probability,
                            signature_authenticity: cvResult.result.signature_authenticity,
                            tampering_evidence: cvResult.result.tampering_evidence,
                            metadata_analysis: cvResult.result.metadata_analysis,
                            image_forensics: cvResult.result.image_forensics,
                            blockchain_hash: '',
                            verification_timestamp: cvResult.result.verification_timestamp,
                            processing_time: cvResult.result.processing_time,
                            user_id: request.user?.id || null
                        };
                        
                        await storeVerificationResult(fastify, verificationData);
                        
                        results.push({
                            filename: document.filename,
                            verification_id: verificationData.id,
                            verification_status: cvResult.result.verification_status,
                            overall_confidence: cvResult.result.overall_confidence,
                            forgery_probability: cvResult.result.forgery_probability,
                            signature_authenticity: cvResult.result.signature_authenticity,
                            processing_time: cvResult.result.processing_time
                        });
                    } else {
                        errors.push({
                            filename: document.filename,
                            error: cvResult.error
                        });
                    }
                    
                } catch (error) {
                    fastify.log.error(`Error verifying document ${document.filename}:`, error);
                    errors.push({
                        filename: document.filename,
                        error: error.message
                    });
                }
            }
            
            // Calculate batch statistics
            const batchStats = {
                total_documents: documentFiles.length,
                successful_verifications: results.length,
                failed_verifications: errors.length,
                average_confidence: results.length > 0 ? 
                    results.reduce((sum, r) => sum + r.overall_confidence, 0) / results.length : 0,
                status_distribution: {
                    authentic: results.filter(r => r.verification_status === 'authentic').length,
                    suspicious: results.filter(r => r.verification_status === 'suspicious').length,
                    fraudulent: results.filter(r => r.verification_status === 'fraudulent').length,
                    inconclusive: results.filter(r => r.verification_status === 'inconclusive').length
                }
            };
            
            return {
                success: true,
                batch_id: batchId,
                data: {
                    batch_statistics: batchStats,
                    verification_results: results,
                    errors: errors
                }
            };
            
        } catch (error) {
            fastify.log.error('Batch CV verification error:', error);
            reply.code(500);
            return {
                success: false,
                error: 'Batch document verification failed',
                details: error.message
            };
        }
    });

    /**
     * GET /result/:verificationId
     * Get detailed verification results by ID
     */
    fastify.get('/result/:verificationId', {
        schema: {
            params: {
                type: 'object',
                properties: {
                    verificationId: { type: 'string', format: 'uuid' }
                },
                required: ['verificationId']
            }
        }
    }, async (request, reply) => {
        try {
            const { verificationId } = request.params;
            
            const query = `
                SELECT * FROM cv_verification_results 
                WHERE id = $1
                ${request.user ? 'AND (user_id = $2 OR user_id IS NULL)' : ''}
            `;
            
            const values = request.user ? [verificationId, request.user.id] : [verificationId];
            const result = await fastify.pg.query(query, values);
            
            if (result.rows.length === 0) {
                reply.code(404);
                return {
                    success: false,
                    error: 'Verification result not found'
                };
            }
            
            const verification = result.rows[0];
            
            return {
                success: true,
                data: {
                    verification_id: verification.id,
                    document_hash: verification.document_hash,
                    verification_status: verification.verification_status,
                    overall_confidence: parseFloat(verification.overall_confidence),
                    forgery_probability: parseFloat(verification.forgery_probability),
                    signature_authenticity: parseFloat(verification.signature_authenticity),
                    tampering_evidence: verification.tampering_evidence,
                    metadata_analysis: verification.metadata_analysis,
                    image_forensics: verification.image_forensics,
                    blockchain_hash: verification.blockchain_hash,
                    verification_timestamp: verification.verification_timestamp,
                    processing_time: parseFloat(verification.processing_time),
                    created_at: verification.created_at
                }
            };
            
        } catch (error) {
            fastify.log.error('Get verification result error:', error);
            reply.code(500);
            return {
                success: false,
                error: 'Failed to retrieve verification result',
                details: error.message
            };
        }
    });

    /**
     * GET /history
     * Get verification history for the current user
     */
    fastify.get('/history', {
        schema: {
            querystring: {
                type: 'object',
                properties: {
                    page: { type: 'integer', minimum: 1, default: 1 },
                    limit: { type: 'integer', minimum: 1, maximum: 100, default: 20 }
                }
            }
        }
    }, async (request, reply) => {
        try {
            const page = parseInt(request.query.page) || 1;
            const limit = Math.min(parseInt(request.query.limit) || 20, 100);
            const offset = (page - 1) * limit;
            
            const whereClause = request.user ? 'WHERE user_id = $1' : '';
            const values = request.user ? [request.user.id] : [];
            
            // Get total count
            const countQuery = `SELECT COUNT(*) FROM cv_verification_results ${whereClause}`;
            const countResult = await fastify.pg.query(countQuery, values);
            const totalCount = parseInt(countResult.rows[0].count);
            
            // Get paginated results
            const query = `
                SELECT 
                    id, document_hash, verification_status, overall_confidence,
                    forgery_probability, signature_authenticity, verification_timestamp,
                    processing_time, created_at
                FROM cv_verification_results 
                ${whereClause}
                ORDER BY created_at DESC
                LIMIT $${values.length + 1} OFFSET $${values.length + 2}
            `;
            
            const result = await fastify.pg.query(query, [...values, limit, offset]);
            
            const verifications = result.rows.map(row => ({
                verification_id: row.id,
                document_hash: row.document_hash,
                verification_status: row.verification_status,
                overall_confidence: parseFloat(row.overall_confidence),
                forgery_probability: parseFloat(row.forgery_probability),
                signature_authenticity: parseFloat(row.signature_authenticity),
                verification_timestamp: row.verification_timestamp,
                processing_time: parseFloat(row.processing_time),
                created_at: row.created_at
            }));
            
            return {
                success: true,
                data: {
                    verifications: verifications,
                    pagination: {
                        page: page,
                        limit: limit,
                        total_count: totalCount,
                        total_pages: Math.ceil(totalCount / limit)
                    }
                }
            };
            
        } catch (error) {
            fastify.log.error('Get verification history error:', error);
            reply.code(500);
            return {
                success: false,
                error: 'Failed to retrieve verification history',
                details: error.message
            };
        }
    });

    /**
     * DELETE /result/:verificationId
     * Delete a verification result (admin or owner only)
     */
    fastify.delete('/result/:verificationId', {
        schema: {
            params: {
                type: 'object',
                properties: {
                    verificationId: { type: 'string', format: 'uuid' }
                },
                required: ['verificationId']
            }
        }
    }, async (request, reply) => {
        try {
            const { verificationId } = request.params;
            
            // Check if user owns the verification or is admin
            const checkQuery = `
                SELECT user_id FROM cv_verification_results WHERE id = $1
            `;
            
            const checkResult = await fastify.pg.query(checkQuery, [verificationId]);
            
            if (checkResult.rows.length === 0) {
                reply.code(404);
                return {
                    success: false,
                    error: 'Verification result not found'
                };
            }
            
            const verification = checkResult.rows[0];
            
            // Allow deletion if user owns the verification or is admin
            if (request.user && (verification.user_id === request.user.id || request.user.role === 'admin')) {
                const deleteQuery = `DELETE FROM cv_verification_results WHERE id = $1`;
                await fastify.pg.query(deleteQuery, [verificationId]);
                
                return {
                    success: true,
                    message: 'Verification result deleted successfully'
                };
            } else {
                reply.code(403);
                return {
                    success: false,
                    error: 'Unauthorized to delete this verification result'
                };
            }
            
        } catch (error) {
            fastify.log.error('Delete verification result error:', error);
            reply.code(500);
            return {
                success: false,
                error: 'Failed to delete verification result',
                details: error.message
            };
        }
    });

    /**
     * GET /analytics
     * Get verification analytics and insights
     */
    fastify.get('/analytics', {
        schema: {
            querystring: {
                type: 'object',
                properties: {
                    range: { 
                        type: 'string', 
                        enum: ['24h', '7d', '30d', '90d'], 
                        default: '30d' 
                    }
                }
            }
        }
    }, async (request, reply) => {
        try {
            const timeRange = request.query.range || '30d';
            let dateCondition = '';
            
            switch (timeRange) {
                case '24h':
                    dateCondition = "WHERE created_at >= NOW() - INTERVAL '1 day'";
                    break;
                case '7d':
                    dateCondition = "WHERE created_at >= NOW() - INTERVAL '7 days'";
                    break;
                case '30d':
                    dateCondition = "WHERE created_at >= NOW() - INTERVAL '30 days'";
                    break;
                case '90d':
                    dateCondition = "WHERE created_at >= NOW() - INTERVAL '90 days'";
                    break;
                default:
                    dateCondition = "WHERE created_at >= NOW() - INTERVAL '30 days'";
            }
            
            // Get verification trends
            const trendsQuery = `
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as total_verifications,
                    COUNT(CASE WHEN verification_status = 'authentic' THEN 1 END) as authentic_count,
                    COUNT(CASE WHEN verification_status = 'suspicious' THEN 1 END) as suspicious_count,
                    COUNT(CASE WHEN verification_status = 'fraudulent' THEN 1 END) as fraudulent_count,
                    AVG(overall_confidence) as avg_confidence,
                    AVG(processing_time) as avg_processing_time
                FROM cv_verification_results 
                ${dateCondition}
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            `;
            
            const trendsResult = await fastify.pg.query(trendsQuery);
            
            // Get confidence distribution
            const confidenceQuery = `
                SELECT 
                    CASE 
                        WHEN overall_confidence >= 0.9 THEN 'very_high'
                        WHEN overall_confidence >= 0.7 THEN 'high'
                        WHEN overall_confidence >= 0.5 THEN 'medium'
                        WHEN overall_confidence >= 0.3 THEN 'low'
                        ELSE 'very_low'
                    END as confidence_range,
                    COUNT(*) as count
                FROM cv_verification_results 
                ${dateCondition}
                GROUP BY confidence_range
                ORDER BY 
                    CASE confidence_range
                        WHEN 'very_high' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                        WHEN 'very_low' THEN 5
                    END
            `;
            
            const confidenceResult = await fastify.pg.query(confidenceQuery);
            
            return {
                success: true,
                data: {
                    time_range: timeRange,
                    verification_trends: trendsResult.rows.map(row => ({
                        date: row.date,
                        total_verifications: parseInt(row.total_verifications),
                        authentic_count: parseInt(row.authentic_count || 0),
                        suspicious_count: parseInt(row.suspicious_count || 0),
                        fraudulent_count: parseInt(row.fraudulent_count || 0),
                        avg_confidence: parseFloat(row.avg_confidence || 0),
                        avg_processing_time: parseFloat(row.avg_processing_time || 0)
                    })),
                    confidence_distribution: confidenceResult.rows.map(row => ({
                        range: row.confidence_range,
                        count: parseInt(row.count)
                    }))
                }
            };
            
        } catch (error) {
            fastify.log.error('Get verification analytics error:', error);
            reply.code(500);
            return {
                success: false,
                error: 'Failed to retrieve verification analytics',
                details: error.message
            };
        }
    });
}

module.exports = cvVerificationRoutes;
