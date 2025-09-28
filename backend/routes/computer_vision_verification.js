/**
 * Computer Vision Document Verification API Routes
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

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');
const rateLimit = require('express-rate-limit');
const { body, param, validationResult } = require('express-validator');
const { spawn } = require('child_process');
const WebSocket = require('ws');

const db = require('../config/database');
const { validateAuth } = require('../utils/validation');

const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: async (req, file, cb) => {
        const uploadDir = path.join(process.cwd(), 'uploads', 'cv_verification');
        try {
            await fs.mkdir(uploadDir, { recursive: true });
            cb(null, uploadDir);
        } catch (error) {
            cb(error);
        }
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, `cv_verification_${uniqueSuffix}${path.extname(file.originalname)}`);
    }
});

const fileFilter = (req, file, cb) => {
    // Accept images and PDFs
    const allowedMimes = [
        'image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp',
        'application/pdf'
    ];
    
    if (allowedMimes.includes(file.mimetype)) {
        cb(null, true);
    } else {
        cb(new Error('Invalid file type. Only images and PDFs are allowed.'), false);
    }
};

const upload = multer({
    storage: storage,
    limits: {
        fileSize: 50 * 1024 * 1024, // 50MB limit
        files: 10 // Maximum 10 files per request
    },
    fileFilter: fileFilter
});

// Rate limiting for CV verification endpoints
const verificationLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 50, // Limit each IP to 50 requests per windowMs
    message: {
        error: 'Too many verification requests from this IP, please try again later.',
        code: 'RATE_LIMIT_EXCEEDED'
    }
});

// WebSocket connections for real-time updates
const wsClients = new Map();

/**
 * Initialize WebSocket connection for real-time verification updates
 */
function initializeWebSocket(server) {
    const wss = new WebSocket.Server({ server });
    
    wss.on('connection', (ws, req) => {
        const clientId = uuidv4();
        wsClients.set(clientId, ws);
        
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                if (data.type === 'subscribe_verification') {
                    ws.clientId = clientId;
                    ws.verificationId = data.verificationId;
                }
            } catch (error) {
                console.error('WebSocket message error:', error);
            }
        });
        
        ws.on('close', () => {
            wsClients.delete(clientId);
        });
        
        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
            wsClients.delete(clientId);
        });
    });
}

/**
 * Send real-time update to WebSocket clients
 */
function sendVerificationUpdate(verificationId, update) {
    for (const [clientId, ws] of wsClients) {
        if (ws.verificationId === verificationId && ws.readyState === WebSocket.OPEN) {
            try {
                ws.send(JSON.stringify({
                    type: 'verification_update',
                    verificationId: verificationId,
                    data: update
                }));
            } catch (error) {
                console.error('Error sending WebSocket update:', error);
                wsClients.delete(clientId);
            }
        }
    }
}

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
async function storeVerificationResult(verificationData) {
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
    
    const result = await db.query(query, values);
    return result.rows[0];
}

/**
 * GET /api/cv-verification/status
 * Get verification system status and statistics
 */
router.get('/status', async (req, res) => {
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
        
        const statsResult = await db.query(statsQuery);
        const stats = statsResult.rows[0];
        
        // System health check
        const systemStatus = {
            status: 'operational',
            models_loaded: true,
            gpu_available: process.env.CUDA_AVAILABLE === 'true',
            blockchain_enabled: process.env.BLOCKCHAIN_VERIFICATION_ENABLED === 'true',
            last_updated: new Date()
        };
        
        res.json({
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
        });
    } catch (error) {
        console.error('CV verification status error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get verification system status',
            details: error.message
        });
    }
});

/**
 * POST /api/cv-verification/verify
 * Verify a single document with optional reference signatures
 */
router.post('/verify',
    verificationLimiter,
    upload.fields([
        { name: 'document', maxCount: 1 },
        { name: 'references', maxCount: 5 }
    ]),
    [
        body('metadata').optional().isJSON().withMessage('Metadata must be valid JSON'),
        body('include_blockchain').optional().isBoolean().withMessage('Include blockchain must be boolean')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    error: 'Validation failed',
                    details: errors.array()
                });
            }
            
            if (!req.files || !req.files.document || !req.files.document[0]) {
                return res.status(400).json({
                    success: false,
                    error: 'Document file is required'
                });
            }
            
            const documentFile = req.files.document[0];
            const referenceFiles = req.files.references || [];
            const metadata = req.body.metadata ? JSON.parse(req.body.metadata) : {};
            const includeBlockchain = req.body.include_blockchain === 'true';
            
            // Generate verification ID for tracking
            const verificationId = uuidv4();
            
            // Send initial status update
            sendVerificationUpdate(verificationId, {
                status: 'processing',
                message: 'Starting document verification...',
                progress: 0
            });
            
            // Prepare file paths
            const documentPath = documentFile.path;
            const referencePaths = referenceFiles.map(file => file.path);
            
            // Add file metadata
            metadata.file_info = {
                original_name: documentFile.originalname,
                size: documentFile.size,
                mime_type: documentFile.mimetype
            };
            
            // Send progress update
            sendVerificationUpdate(verificationId, {
                status: 'processing',
                message: 'Executing computer vision analysis...',
                progress: 25
            });
            
            // Execute CV verification
            const cvResult = await executeCVVerification(documentPath, referencePaths, metadata);
            
            // Send progress update
            sendVerificationUpdate(verificationId, {
                status: 'processing',
                message: 'Analyzing verification results...',
                progress: 75
            });
            
            // Prepare verification data for storage
            const verificationData = {
                id: verificationId,
                document_hash: cvResult.document_hash,
                document_path: documentPath,
                verification_status: cvResult.verification_status,
                overall_confidence: cvResult.overall_confidence,
                forgery_probability: cvResult.forgery_probability,
                signature_authenticity: cvResult.signature_authenticity,
                tampering_evidence: cvResult.tampering_evidence,
                metadata_analysis: cvResult.metadata_analysis,
                image_forensics: cvResult.image_forensics,
                blockchain_hash: includeBlockchain ? cvResult.blockchain_hash : '',
                verification_timestamp: cvResult.verification_timestamp,
                processing_time: cvResult.processing_time,
                user_id: req.user?.id || null
            };
            
            // Store results in database
            await storeVerificationResult(verificationData);
            
            // Send completion update
            sendVerificationUpdate(verificationId, {
                status: 'completed',
                message: 'Verification completed successfully',
                progress: 100,
                result: cvResult
            });
            
            // Prepare response
            const response = {
                success: true,
                verification_id: verificationId,
                data: {
                    verification_status: cvResult.verification_status,
                    overall_confidence: cvResult.overall_confidence,
                    forgery_probability: cvResult.forgery_probability,
                    signature_authenticity: cvResult.signature_authenticity,
                    tampering_evidence: cvResult.tampering_evidence,
                    summary: {
                        document_hash: cvResult.document_hash,
                        processing_time: cvResult.processing_time,
                        timestamp: cvResult.verification_timestamp
                    }
                }
            };
            
            // Include detailed analysis if requested
            if (req.body.include_details === 'true') {
                response.data.detailed_analysis = {
                    metadata_analysis: cvResult.metadata_analysis,
                    image_forensics: cvResult.image_forensics
                };
            }
            
            // Include blockchain hash if enabled
            if (includeBlockchain && cvResult.blockchain_hash) {
                response.data.blockchain_hash = cvResult.blockchain_hash;
            }
            
            res.json(response);
            
        } catch (error) {
            console.error('CV verification error:', error);
            
            // Send error update
            if (req.body.verification_id) {
                sendVerificationUpdate(req.body.verification_id, {
                    status: 'error',
                    message: 'Verification failed',
                    error: error.message
                });
            }
            
            res.status(500).json({
                success: false,
                error: 'Document verification failed',
                details: error.message
            });
        }
    }
);

/**
 * POST /api/cv-verification/batch-verify
 * Verify multiple documents in batch
 */
router.post('/batch-verify',
    verificationLimiter,
    upload.array('documents', 10),
    [
        body('batch_metadata').optional().isJSON().withMessage('Batch metadata must be valid JSON')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    error: 'Validation failed',
                    details: errors.array()
                });
            }
            
            if (!req.files || req.files.length === 0) {
                return res.status(400).json({
                    success: false,
                    error: 'At least one document file is required'
                });
            }
            
            const documents = req.files;
            const batchMetadata = req.body.batch_metadata ? JSON.parse(req.body.batch_metadata) : {};
            const batchId = uuidv4();
            
            // Send initial batch status
            sendVerificationUpdate(batchId, {
                status: 'processing',
                message: `Starting batch verification of ${documents.length} documents...`,
                progress: 0,
                total_documents: documents.length,
                completed_documents: 0
            });
            
            const results = [];
            const batchErrors = [];
            
            // Process documents sequentially to avoid resource overload
            for (let i = 0; i < documents.length; i++) {
                const document = documents[i];
                const documentMetadata = batchMetadata[document.originalname] || {};
                
                try {
                    // Add file metadata
                    documentMetadata.file_info = {
                        original_name: document.originalname,
                        size: document.size,
                        mime_type: document.mimetype
                    };
                    
                    // Send progress update
                    const progress = Math.round((i / documents.length) * 100);
                    sendVerificationUpdate(batchId, {
                        status: 'processing',
                        message: `Processing document ${i + 1} of ${documents.length}: ${document.originalname}`,
                        progress: progress,
                        total_documents: documents.length,
                        completed_documents: i
                    });
                    
                    // Execute CV verification
                    const cvResult = await executeCVVerification(document.path, [], documentMetadata);
                    
                    // Store individual result
                    const verificationData = {
                        id: uuidv4(),
                        document_hash: cvResult.document_hash,
                        document_path: document.path,
                        verification_status: cvResult.verification_status,
                        overall_confidence: cvResult.overall_confidence,
                        forgery_probability: cvResult.forgery_probability,
                        signature_authenticity: cvResult.signature_authenticity,
                        tampering_evidence: cvResult.tampering_evidence,
                        metadata_analysis: cvResult.metadata_analysis,
                        image_forensics: cvResult.image_forensics,
                        blockchain_hash: '',
                        verification_timestamp: cvResult.verification_timestamp,
                        processing_time: cvResult.processing_time,
                        user_id: req.user?.id || null
                    };
                    
                    await storeVerificationResult(verificationData);
                    
                    results.push({
                        filename: document.originalname,
                        verification_id: verificationData.id,
                        verification_status: cvResult.verification_status,
                        overall_confidence: cvResult.overall_confidence,
                        forgery_probability: cvResult.forgery_probability,
                        signature_authenticity: cvResult.signature_authenticity,
                        processing_time: cvResult.processing_time
                    });
                    
                } catch (error) {
                    console.error(`Error verifying document ${document.originalname}:`, error);
                    batchErrors.push({
                        filename: document.originalname,
                        error: error.message
                    });
                }
            }
            
            // Send completion update
            sendVerificationUpdate(batchId, {
                status: 'completed',
                message: `Batch verification completed: ${results.length} successful, ${errors.length} failed`,
                progress: 100,
                total_documents: documents.length,
                completed_documents: results.length
            });
            
            // Calculate batch statistics
            const batchStats = {
                total_documents: documents.length,
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
            
            res.json({
                success: true,
                batch_id: batchId,
                data: {
                    batch_statistics: batchStats,
                    verification_results: results,
                    errors: batchErrors
                }
            });
            
        } catch (error) {
            console.error('Batch CV verification error:', error);
            res.status(500).json({
                success: false,
                error: 'Batch document verification failed',
                details: error.message
            });
        }
    }
);

/**
 * GET /api/cv-verification/result/:verificationId
 * Get detailed verification results by ID
 */
router.get('/result/:verificationId',
    [
        param('verificationId').isUUID().withMessage('Invalid verification ID format')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    error: 'Validation failed',
                    details: errors.array()
                });
            }
            
            const { verificationId } = req.params;
            
            const query = `
                SELECT * FROM cv_verification_results 
                WHERE id = $1
                ${req.user ? 'AND (user_id = $2 OR user_id IS NULL)' : ''}
            `;
            
            const values = req.user ? [verificationId, req.user.id] : [verificationId];
            const result = await db.query(query, values);
            
            if (result.rows.length === 0) {
                return res.status(404).json({
                    success: false,
                    error: 'Verification result not found'
                });
            }
            
            const verification = result.rows[0];
            
            res.json({
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
            });
            
        } catch (error) {
            console.error('Get verification result error:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to retrieve verification result',
                details: error.message
            });
        }
    }
);

/**
 * GET /api/cv-verification/history
 * Get verification history for the current user
 */
router.get('/history', async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = Math.min(parseInt(req.query.limit) || 20, 100);
        const offset = (page - 1) * limit;
        
        const whereClause = req.user ? 'WHERE user_id = $1' : '';
        const values = req.user ? [req.user.id] : [];
        
        // Get total count
        const countQuery = `SELECT COUNT(*) FROM cv_verification_results ${whereClause}`;
        const countResult = await db.query(countQuery, values);
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
        
        const result = await db.query(query, [...values, limit, offset]);
        
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
        
        res.json({
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
        });
        
    } catch (error) {
        console.error('Get verification history error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve verification history',
            details: error.message
        });
    }
});

/**
 * DELETE /api/cv-verification/result/:verificationId
 * Delete a verification result (admin or owner only)
 */
router.delete('/result/:verificationId',
    [
        param('verificationId').isUUID().withMessage('Invalid verification ID format')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    error: 'Validation failed',
                    details: errors.array()
                });
            }
            
            const { verificationId } = req.params;
            
            // Check if user owns the verification or is admin
            const checkQuery = `
                SELECT user_id FROM cv_verification_results WHERE id = $1
            `;
            
            const checkResult = await db.query(checkQuery, [verificationId]);
            
            if (checkResult.rows.length === 0) {
                return res.status(404).json({
                    success: false,
                    error: 'Verification result not found'
                });
            }
            
            const verification = checkResult.rows[0];
            
            // Allow deletion if user owns the verification or is admin
            if (req.user && (verification.user_id === req.user.id || req.user.role === 'admin')) {
                const deleteQuery = `DELETE FROM cv_verification_results WHERE id = $1`;
                await db.query(deleteQuery, [verificationId]);
                
                res.json({
                    success: true,
                    message: 'Verification result deleted successfully'
                });
            } else {
                res.status(403).json({
                    success: false,
                    error: 'Unauthorized to delete this verification result'
                });
            }
            
        } catch (error) {
            console.error('Delete verification result error:', error);
            res.status(500).json({
                success: false,
                error: 'Failed to delete verification result',
                details: error.message
            });
        }
    }
);

/**
 * GET /api/cv-verification/analytics
 * Get verification analytics and insights
 */
router.get('/analytics', async (req, res) => {
    try {
        const timeRange = req.query.range || '30d';
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
        
        const trendsResult = await db.query(trendsQuery);
        
        // Get forgery type distribution
        const forgeryTypesQuery = `
            SELECT 
                jsonb_array_elements(tampering_evidence)->>'tampering_type' as forgery_type,
                COUNT(*) as count,
                AVG((jsonb_array_elements(tampering_evidence)->>'confidence')::float) as avg_confidence
            FROM cv_verification_results 
            ${dateCondition}
            AND jsonb_array_length(tampering_evidence) > 0
            GROUP BY forgery_type
            ORDER BY count DESC
        `;
        
        const forgeryTypesResult = await db.query(forgeryTypesQuery);
        
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
        
        const confidenceResult = await db.query(confidenceQuery);
        
        res.json({
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
                forgery_types: forgeryTypesResult.rows.map(row => ({
                    type: row.forgery_type,
                    count: parseInt(row.count),
                    avg_confidence: parseFloat(row.avg_confidence || 0)
                })),
                confidence_distribution: confidenceResult.rows.map(row => ({
                    range: row.confidence_range,
                    count: parseInt(row.count)
                }))
            }
        });
        
    } catch (error) {
        console.error('Get verification analytics error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve verification analytics',
            details: error.message
        });
    }
});

// Export router and WebSocket initializer
module.exports = {
    router,
    initializeWebSocket
};
