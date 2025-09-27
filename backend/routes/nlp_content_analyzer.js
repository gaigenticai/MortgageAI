/**
 * NLP Content Analyzer API Routes
 * Production-grade Fastify routes for comprehensive natural language processing
 * 
 * Features:
 * - Advanced transformer models for Dutch and English text processing
 * - Named entity recognition for financial, personal, and legal entities
 * - Semantic similarity analysis and content validation
 * - Sentiment analysis and risk indicator detection
 * - Multilingual document processing with automatic language detection
 * - Contextual understanding and relationship extraction
 * - Compliance validation against regulatory requirements
 * - Real-time content monitoring and alerting
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
    documents_analyzed: 0,
    entities_extracted: 0,
    risk_indicators_detected: 0,
    avg_processing_time: 0,
    processing_times: [],
    language_distribution: {},
    content_type_distribution: {}
};

/**
 * Initialize NLP Content Analyzer routes
 */
async function initializeNLPContentAnalyzerRoutes(app, database, redis) {
    dbPool = database;
    redisClient = redis;

    // Initialize WebSocket server
    wsServer = new WebSocket.Server({ port: 8015 });
    
    wsServer.on('connection', (ws) => {
        fastify.log.info('WebSocket client connected to NLP Content Analyzer');
        
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                handleWebSocketMessage(ws, data);
            } catch (error) {
                ws.send(JSON.stringify({ error: 'Invalid message format' }));
            }
        });
        
        ws.on('close', () => {
            fastify.log.info('WebSocket client disconnected from NLP Content Analyzer');
        });
    });

    // Start Python agent
    await startPythonAgent();

    // Register all routes
    await registerContentAnalysisRoutes(app);
    await registerEntityExtractionRoutes(app);
    await registerSemanticAnalysisRoutes(app);
    await registerSentimentAnalysisRoutes(app);
    await registerRiskAnalysisRoutes(app);
    await registerTopicModelingRoutes(app);
    await registerRelationshipExtractionRoutes(app);
    await registerComplianceAnalysisRoutes(app);
    await registerModelManagementRoutes(app);
    await registerAnalyticsRoutes(app);
    await registerAlertManagementRoutes(app);

    fastify.log.info('NLP Content Analyzer routes initialized');
}

/**
 * Start Python agent process
 */
async function startPythonAgent() {
    try {
        const agentPath = path.join(__dirname, '../agents/nlp_content_analyzer/agent.py');
        
        pythonAgent = spawn('python3', [agentPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: {
                ...process.env,
                PYTHONPATH: path.join(__dirname, '../agents'),
                DATABASE_URL: process.env.DATABASE_URL,
                REDIS_URL: process.env.REDIS_URL,
                TRANSFORMERS_CACHE: '/tmp/transformers_cache'
            }
        });

        pythonAgent.stdout.on('data', (data) => {
            fastify.log.info(`NLP Content Analyzer Python Agent: ${data}`);
        });

        pythonAgent.stderr.on('data', (data) => {
            fastify.log.error(`NLP Content Analyzer Python Agent Error: ${data}`);
        });

        pythonAgent.on('close', (code) => {
            fastify.log.warn(`NLP Content Analyzer Python Agent exited with code ${code}`);
            // Auto-restart after 5 seconds
            setTimeout(startPythonAgent, 5000);
        });

        fastify.log.info('NLP Content Analyzer Python agent started');
    } catch (error) {
        fastify.log.error('Failed to start NLP Content Analyzer Python agent:', error);
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
            reject(new Error('NLP Content Analyzer Python agent timeout'));
        }, 300000); // 5 minute timeout for complex NLP operations

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
        case 'subscribe_nlp_analysis':
            ws.nlpSubscription = true;
            break;
        case 'subscribe_entity_extraction':
            ws.entitySubscription = true;
            break;
        case 'subscribe_risk_alerts':
            ws.riskAlertSubscription = true;
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
 * Send real-time metrics
 */
async function sendRealTimeMetrics(ws) {
    try {
        const metrics = await getNLPMetrics();
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
 * Get NLP metrics
 */
async function getNLPMetrics() {
    try {
        const dailyStats = await dbPool.query(`
            SELECT 
                language_detected,
                content_type,
                COUNT(*) as count,
                AVG(confidence_score) as avg_confidence,
                AVG(processing_time_ms) as avg_processing_time
            FROM nlp_content_analysis 
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY language_detected, content_type
        `);

        const entityStats = await dbPool.query(`
            SELECT 
                entity_type,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM nlp_named_entities ne
            JOIN nlp_content_analysis nca ON ne.analysis_id = nca.analysis_id
            WHERE nca.created_at > NOW() - INTERVAL '24 hours'
            GROUP BY entity_type
            ORDER BY count DESC
        `);

        const riskStats = await dbPool.query(`
            SELECT 
                COUNT(*) as total_analyses,
                COUNT(CASE WHEN risk_score > 0.5 THEN 1 END) as high_risk_analyses,
                AVG(risk_score) as avg_risk_score
            FROM nlp_risk_analysis nra
            JOIN nlp_content_analysis nca ON nra.analysis_id = nca.analysis_id
            WHERE nca.created_at > NOW() - INTERVAL '24 hours'
        `);

        return {
            daily_stats: dailyStats.rows,
            entity_stats: entityStats.rows,
            risk_stats: riskStats.rows[0] || {},
            system_metrics: performanceMetrics
        };
    } catch (error) {
        fastify.log.error('Failed to get NLP metrics:', error);
        throw error;
    }
}

/**
 * Register content analysis routes
 */
async function registerContentAnalysisRoutes(app) {
    // Analyze text content
    app.post('/api/nlp-content-analyzer/analyze', async (request, reply) => {
        try {
            const { text, document_id, analysis_options = {} } = request.body;

            if (!text || text.trim().length === 0) {
                reply.status(400).send({
                    success: false,
                    error: 'Text content is required'
                });
                return;
            }

            if (text.length > 100000) {
                reply.status(400).send({
                    success: false,
                    error: 'Text content exceeds maximum length (100,000 characters)'
                });
                return;
            }

            const startTime = Date.now();

            // Call Python agent for comprehensive analysis
            const analysisResult = await callPythonAgent('analyze_content', {
                text,
                document_id: document_id || crypto.randomUUID(),
                options: analysis_options
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.documents_analyzed++;
            performanceMetrics.processing_times.push(processingTime);

            // Update metrics
            if (performanceMetrics.processing_times.length > 1000) {
                performanceMetrics.processing_times = performanceMetrics.processing_times.slice(-1000);
            }
            performanceMetrics.avg_processing_time = 
                performanceMetrics.processing_times.reduce((a, b) => a + b, 0) / 
                performanceMetrics.processing_times.length;

            // Update language and content type distributions
            const language = analysisResult.language_detection;
            const contentType = analysisResult.semantic_analysis.content_type;
            
            performanceMetrics.language_distribution[language] = 
                (performanceMetrics.language_distribution[language] || 0) + 1;
            performanceMetrics.content_type_distribution[contentType] = 
                (performanceMetrics.content_type_distribution[contentType] || 0) + 1;

            performanceMetrics.entities_extracted += analysisResult.named_entities.length;
            performanceMetrics.risk_indicators_detected += analysisResult.risk_indicator_analysis.risk_indicators.length;

            // Generate alerts for high-risk content
            if (analysisResult.risk_indicator_analysis.risk_score > 0.6) {
                await generateNLPAlert(analysisResult);
            }

            // Broadcast to WebSocket clients
            broadcastToClients({
                type: 'nlp_analysis_completed',
                data: {
                    analysis_id: analysisResult.analysis_id,
                    document_id: analysisResult.document_id,
                    language: language,
                    content_type: contentType,
                    entities_found: analysisResult.named_entities.length,
                    risk_score: analysisResult.risk_indicator_analysis.risk_score,
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                analysis: analysisResult,
                processing_time_ms: processingTime
            };

        } catch (error) {
            fastify.log.error('NLP content analysis failed:', error);
            reply.status(500).send({
                success: false,
                error: error.message || 'NLP content analysis failed'
            });
        }
    });

    // Get analysis result by ID
    app.get('/api/nlp-content-analyzer/analysis/:analysisId', async (request, reply) => {
        try {
            const { analysisId } = request.params;

            const result = await dbPool.query(`
                SELECT nca.*, nsa.*, nsent.*, nra.*, ncv.*
                FROM nlp_content_analysis nca
                LEFT JOIN nlp_semantic_analysis nsa ON nca.analysis_id = nsa.analysis_id
                LEFT JOIN nlp_sentiment_analysis nsent ON nca.analysis_id = nsent.analysis_id
                LEFT JOIN nlp_risk_analysis nra ON nca.analysis_id = nra.analysis_id
                LEFT JOIN nlp_content_validation ncv ON nca.analysis_id = ncv.analysis_id
                WHERE nca.analysis_id = $1
            `, [analysisId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Analysis result not found'
                });
                return;
            }

            const analysis = result.rows[0];

            // Get named entities
            const entitiesResult = await dbPool.query(`
                SELECT * FROM nlp_named_entities WHERE analysis_id = $1
                ORDER BY start_position
            `, [analysisId]);

            // Get relationships
            const relationshipsResult = await dbPool.query(`
                SELECT * FROM nlp_relationship_extraction WHERE analysis_id = $1
            `, [analysisId]);

            // Get topic modeling
            const topicsResult = await dbPool.query(`
                SELECT * FROM nlp_topic_modeling WHERE analysis_id = $1
            `, [analysisId]);

            return {
                success: true,
                analysis: {
                    ...analysis,
                    named_entities: entitiesResult.rows,
                    relationships: relationshipsResult.rows,
                    topic_modeling: topicsResult.rows[0] || null
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get analysis result:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve analysis result'
            });
        }
    });

    // Get analysis history
    app.get('/api/nlp-content-analyzer/analysis', async (request, reply) => {
        try {
            const {
                document_id,
                language,
                content_type,
                start_date,
                end_date,
                page = 1,
                limit = 50
            } = request.query;

            let query = `
                SELECT nca.analysis_id, nca.document_id, nca.language_detected,
                       nca.content_type, nca.confidence_score, nca.processing_time_ms,
                       nca.created_at, nra.risk_score, 
                       (SELECT COUNT(*) FROM nlp_named_entities WHERE analysis_id = nca.analysis_id) as entity_count
                FROM nlp_content_analysis nca
                LEFT JOIN nlp_risk_analysis nra ON nca.analysis_id = nra.analysis_id
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (document_id) {
                query += ` AND nca.document_id = $${paramIndex}`;
                params.push(document_id);
                paramIndex++;
            }

            if (language) {
                query += ` AND nca.language_detected = $${paramIndex}`;
                params.push(language);
                paramIndex++;
            }

            if (content_type) {
                query += ` AND nca.content_type = $${paramIndex}`;
                params.push(content_type);
                paramIndex++;
            }

            if (start_date) {
                query += ` AND nca.created_at >= $${paramIndex}`;
                params.push(start_date);
                paramIndex++;
            }

            if (end_date) {
                query += ` AND nca.created_at <= $${paramIndex}`;
                params.push(end_date);
                paramIndex++;
            }

            query += ` ORDER BY nca.created_at DESC`;
            query += ` LIMIT $${paramIndex} OFFSET $${paramIndex + 1}`;
            params.push(parseInt(limit));
            params.push((parseInt(page) - 1) * parseInt(limit));

            const result = await dbPool.query(query, params);

            return {
                success: true,
                analyses: result.rows,
                pagination: {
                    page: parseInt(page),
                    limit: parseInt(limit)
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get analysis history:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve analysis history'
            });
        }
    });

    // Batch analyze multiple texts
    app.post('/api/nlp-content-analyzer/analyze-batch', async (request, reply) => {
        try {
            const { texts, batch_options = {} } = request.body;

            if (!texts || !Array.isArray(texts)) {
                reply.status(400).send({
                    success: false,
                    error: 'Texts array is required'
                });
                return;
            }

            if (texts.length > 100) {
                reply.status(400).send({
                    success: false,
                    error: 'Maximum 100 texts per batch'
                });
                return;
            }

            const startTime = Date.now();
            const results = [];
            const errors = [];

            // Process each text
            for (let i = 0; i < texts.length; i++) {
                try {
                    const textData = texts[i];
                    const analysisResult = await callPythonAgent('analyze_content', {
                        text: textData.text,
                        document_id: textData.document_id || `batch_${i}_${Date.now()}`,
                        options: batch_options
                    });
                    results.push(analysisResult);
                } catch (error) {
                    errors.push({
                        index: i,
                        text_preview: texts[i].text?.substring(0, 100) + '...',
                        error: error.message
                    });
                }
            }

            const processingTime = Date.now() - startTime;
            performanceMetrics.documents_analyzed += results.length;

            return {
                success: true,
                batch_results: {
                    total_texts: texts.length,
                    successful_analyses: results.length,
                    failed_analyses: errors.length,
                    processing_time_ms: processingTime
                },
                analyses: results,
                errors
            };

        } catch (error) {
            fastify.log.error('Batch NLP analysis failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Batch NLP analysis failed'
            });
        }
    });
}

/**
 * Register entity extraction routes
 */
async function registerEntityExtractionRoutes(app) {
    // Extract named entities from text
    app.post('/api/nlp-content-analyzer/extract-entities', async (request, reply) => {
        try {
            const { text, entity_types = [], language = 'auto' } = request.body;

            if (!text) {
                reply.status(400).send({
                    success: false,
                    error: 'Text is required'
                });
                return;
            }

            const startTime = Date.now();

            const entitiesResult = await callPythonAgent('extract_named_entities', {
                text,
                entity_types,
                language
            });

            const processingTime = Date.now() - startTime;

            return {
                success: true,
                entities: entitiesResult,
                processing_time_ms: processingTime,
                entity_count: entitiesResult.length
            };

        } catch (error) {
            fastify.log.error('Entity extraction failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Entity extraction failed'
            });
        }
    });

    // Get entities by analysis ID
    app.get('/api/nlp-content-analyzer/entities/:analysisId', async (request, reply) => {
        try {
            const { analysisId } = request.params;
            const { entity_type, validated_only = false } = request.query;

            let query = `
                SELECT * FROM nlp_named_entities WHERE analysis_id = $1
            `;
            const params = [analysisId];
            let paramIndex = 2;

            if (entity_type) {
                query += ` AND entity_type = $${paramIndex}`;
                params.push(entity_type);
                paramIndex++;
            }

            if (validated_only === 'true') {
                query += ` AND validation_status = true`;
            }

            query += ' ORDER BY start_position';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                entities: result.rows,
                total_entities: result.rows.length
            };

        } catch (error) {
            fastify.log.error('Failed to get entities:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve entities'
            });
        }
    });

    // Validate specific entity
    app.post('/api/nlp-content-analyzer/validate-entity', async (request, reply) => {
        try {
            const { entity_text, entity_type } = request.body;

            if (!entity_text || !entity_type) {
                reply.status(400).send({
                    success: false,
                    error: 'Entity text and type are required'
                });
                return;
            }

            const validationResult = await callPythonAgent('validate_entity', {
                entity_text,
                entity_type
            });

            return {
                success: true,
                validation: validationResult
            };

        } catch (error) {
            fastify.log.error('Entity validation failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Entity validation failed'
            });
        }
    });
}

/**
 * Register semantic analysis routes
 */
async function registerSemanticAnalysisRoutes(app) {
    // Perform semantic analysis
    app.post('/api/nlp-content-analyzer/semantic-analysis', async (request, reply) => {
        try {
            const { text, language = 'auto' } = request.body;

            if (!text) {
                reply.status(400).send({
                    success: false,
                    error: 'Text is required'
                });
                return;
            }

            const semanticResult = await callPythonAgent('analyze_semantic_content', {
                text,
                language
            });

            return {
                success: true,
                semantic_analysis: semanticResult
            };

        } catch (error) {
            fastify.log.error('Semantic analysis failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Semantic analysis failed'
            });
        }
    });

    // Compare semantic similarity
    app.post('/api/nlp-content-analyzer/semantic-similarity', async (request, reply) => {
        try {
            const { text1, text2, similarity_method = 'sentence_transformer' } = request.body;

            if (!text1 || !text2) {
                reply.status(400).send({
                    success: false,
                    error: 'Both text1 and text2 are required'
                });
                return;
            }

            const similarityResult = await callPythonAgent('calculate_semantic_similarity', {
                text1,
                text2,
                method: similarity_method
            });

            return {
                success: true,
                similarity: similarityResult
            };

        } catch (error) {
            fastify.log.error('Semantic similarity calculation failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Semantic similarity calculation failed'
            });
        }
    });
}

/**
 * Register sentiment analysis routes
 */
async function registerSentimentAnalysisRoutes(app) {
    // Analyze sentiment
    app.post('/api/nlp-content-analyzer/sentiment-analysis', async (request, reply) => {
        try {
            const { text, language = 'auto' } = request.body;

            if (!text) {
                reply.status(400).send({
                    success: false,
                    error: 'Text is required'
                });
                return;
            }

            const sentimentResult = await callPythonAgent('analyze_sentiment', {
                text,
                language
            });

            return {
                success: true,
                sentiment_analysis: sentimentResult
            };

        } catch (error) {
            fastify.log.error('Sentiment analysis failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Sentiment analysis failed'
            });
        }
    });

    // Get sentiment trends
    app.get('/api/nlp-content-analyzer/sentiment-trends', async (request, reply) => {
        try {
            const { days = 30, content_type } = request.query;

            let query = `
                SELECT 
                    DATE(nca.created_at) as analysis_date,
                    nsent.overall_sentiment,
                    COUNT(*) as count,
                    AVG(nsent.sentiment_score) as avg_sentiment_score
                FROM nlp_content_analysis nca
                JOIN nlp_sentiment_analysis nsent ON nca.analysis_id = nsent.analysis_id
                WHERE nca.created_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];

            if (content_type) {
                query += ' AND nca.content_type = $1';
                params.push(content_type);
            }

            query += ` GROUP BY DATE(nca.created_at), nsent.overall_sentiment
                      ORDER BY analysis_date DESC, nsent.overall_sentiment`;

            const result = await dbPool.query(query, params);

            return {
                success: true,
                sentiment_trends: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get sentiment trends:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve sentiment trends'
            });
        }
    });
}

/**
 * Register risk analysis routes
 */
async function registerRiskAnalysisRoutes(app) {
    // Analyze risk indicators
    app.post('/api/nlp-content-analyzer/risk-analysis', async (request, reply) => {
        try {
            const { text, risk_categories = [] } = request.body;

            if (!text) {
                reply.status(400).send({
                    success: false,
                    error: 'Text is required'
                });
                return;
            }

            const riskResult = await callPythonAgent('analyze_risk_indicators', {
                text,
                risk_categories
            });

            return {
                success: true,
                risk_analysis: riskResult
            };

        } catch (error) {
            fastify.log.error('Risk analysis failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Risk analysis failed'
            });
        }
    });

    // Get risk statistics
    app.get('/api/nlp-content-analyzer/risk-statistics', async (request, reply) => {
        try {
            const { days = 30 } = request.query;

            const riskStats = await dbPool.query(`
                SELECT 
                    risk_indicator,
                    COUNT(*) as detection_count,
                    AVG(nra.risk_score) as avg_risk_score,
                    COUNT(DISTINCT nra.analysis_id) as documents_affected
                FROM nlp_risk_analysis nra
                CROSS JOIN LATERAL jsonb_array_elements_text(nra.risk_indicators) AS risk_indicator
                JOIN nlp_content_analysis nca ON nra.analysis_id = nca.analysis_id
                WHERE nca.created_at > NOW() - INTERVAL '${days} days'
                GROUP BY risk_indicator
                ORDER BY detection_count DESC
            `);

            const overallStats = await dbPool.query(`
                SELECT 
                    COUNT(*) as total_analyses,
                    COUNT(CASE WHEN risk_score > 0.5 THEN 1 END) as high_risk_analyses,
                    AVG(risk_score) as avg_risk_score,
                    MAX(risk_score) as max_risk_score
                FROM nlp_risk_analysis nra
                JOIN nlp_content_analysis nca ON nra.analysis_id = nca.analysis_id
                WHERE nca.created_at > NOW() - INTERVAL '${days} days'
            `);

            return {
                success: true,
                risk_statistics: {
                    risk_indicators: riskStats.rows,
                    overall_stats: overallStats.rows[0] || {}
                }
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
 * Register topic modeling routes
 */
async function registerTopicModelingRoutes(app) {
    // Perform topic modeling
    app.post('/api/nlp-content-analyzer/topic-modeling', async (request, reply) => {
        try {
            const { text, num_topics = 5, algorithm = 'lda' } = request.body;

            if (!text) {
                reply.status(400).send({
                    success: false,
                    error: 'Text is required'
                });
                return;
            }

            const topicResult = await callPythonAgent('perform_topic_modeling', {
                text,
                num_topics,
                algorithm
            });

            return {
                success: true,
                topic_modeling: topicResult
            };

        } catch (error) {
            fastify.log.error('Topic modeling failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Topic modeling failed'
            });
        }
    });

    // Get topic trends
    app.get('/api/nlp-content-analyzer/topic-trends', async (request, reply) => {
        try {
            const { days = 30, content_type } = request.query;

            let query = `
                SELECT 
                    topic_term,
                    COUNT(*) as frequency,
                    AVG(topic_score) as avg_score
                FROM nlp_topic_modeling ntm
                CROSS JOIN LATERAL jsonb_array_elements(ntm.topics) AS topic_data(topic_info)
                CROSS JOIN LATERAL jsonb_each_text(topic_info) AS topic_details(topic_term, topic_score_text)
                JOIN nlp_content_analysis nca ON ntm.analysis_id = nca.analysis_id
                WHERE nca.created_at > NOW() - INTERVAL '${days} days'
                AND topic_score_text::DECIMAL > 0.1
            `;
            const params = [];

            if (content_type) {
                query += ' AND nca.content_type = $1';
                params.push(content_type);
            }

            query += ` GROUP BY topic_term
                      ORDER BY frequency DESC, avg_score DESC
                      LIMIT 20`;

            const result = await dbPool.query(query, params);

            return {
                success: true,
                topic_trends: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get topic trends:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve topic trends'
            });
        }
    });
}

/**
 * Register relationship extraction routes
 */
async function registerRelationshipExtractionRoutes(app) {
    // Extract relationships
    app.post('/api/nlp-content-analyzer/extract-relationships', async (request, reply) => {
        try {
            const { text, entities = [] } = request.body;

            if (!text) {
                reply.status(400).send({
                    success: false,
                    error: 'Text is required'
                });
                return;
            }

            const relationshipResult = await callPythonAgent('extract_relationships', {
                text,
                entities
            });

            return {
                success: true,
                relationships: relationshipResult
            };

        } catch (error) {
            fastify.log.error('Relationship extraction failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Relationship extraction failed'
            });
        }
    });

    // Get relationships by analysis ID
    app.get('/api/nlp-content-analyzer/relationships/:analysisId', async (request, reply) => {
        try {
            const { analysisId } = request.params;
            const { relationship_type } = request.query;

            let query = `
                SELECT * FROM nlp_relationship_extraction WHERE analysis_id = $1
            `;
            const params = [analysisId];

            if (relationship_type) {
                query += ' AND relationship_type = $2';
                params.push(relationship_type);
            }

            query += ' ORDER BY confidence DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                relationships: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get relationships:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve relationships'
            });
        }
    });
}

/**
 * Register compliance analysis routes
 */
async function registerComplianceAnalysisRoutes(app) {
    // Analyze compliance
    app.post('/api/nlp-content-analyzer/compliance-analysis', async (request, reply) => {
        try {
            const { text, regulations = ['gdpr', 'wft', 'bgfo'] } = request.body;

            if (!text) {
                reply.status(400).send({
                    success: false,
                    error: 'Text is required'
                });
                return;
            }

            const complianceResult = await callPythonAgent('analyze_compliance', {
                text,
                regulations
            });

            return {
                success: true,
                compliance_analysis: complianceResult
            };

        } catch (error) {
            fastify.log.error('Compliance analysis failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Compliance analysis failed'
            });
        }
    });

    // Get compliance statistics
    app.get('/api/nlp-content-analyzer/compliance-statistics', async (request, reply) => {
        try {
            const { days = 30 } = request.query;

            const complianceStats = await dbPool.query(`
                SELECT 
                    compliance_key,
                    compliance_value::BOOLEAN,
                    COUNT(*) as count
                FROM nlp_content_validation ncv
                CROSS JOIN LATERAL jsonb_each(ncv.regulatory_compliance) AS compliance_data(compliance_key, compliance_value)
                JOIN nlp_content_analysis nca ON ncv.analysis_id = nca.analysis_id
                WHERE nca.created_at > NOW() - INTERVAL '${days} days'
                GROUP BY compliance_key, compliance_value::BOOLEAN
                ORDER BY compliance_key, compliance_value::BOOLEAN
            `);

            return {
                success: true,
                compliance_statistics: complianceStats.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get compliance statistics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve compliance statistics'
            });
        }
    });
}

/**
 * Register model management routes
 */
async function registerModelManagementRoutes(app) {
    // Get model performance
    app.get('/api/nlp-content-analyzer/model-performance', async (request, reply) => {
        try {
            const { model_type, language, task_type } = request.query;

            let query = `
                SELECT * FROM nlp_model_performance
                WHERE is_active = true
            `;
            const params = [];
            let paramIndex = 1;

            if (model_type) {
                query += ` AND model_type = $${paramIndex}`;
                params.push(model_type);
                paramIndex++;
            }

            if (language) {
                query += ` AND language = $${paramIndex}`;
                params.push(language);
                paramIndex++;
            }

            if (task_type) {
                query += ` AND task_type = $${paramIndex}`;
                params.push(task_type);
                paramIndex++;
            }

            query += ' ORDER BY evaluation_date DESC';

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

    // Update model performance
    app.post('/api/nlp-content-analyzer/update-model-performance', async (request, reply) => {
        try {
            const performanceData = request.body;

            const requiredFields = ['model_name', 'model_type', 'language', 'task_type'];
            for (const field of requiredFields) {
                if (!performanceData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            const result = await dbPool.query(`
                INSERT INTO nlp_model_performance (
                    model_name, model_type, model_version, language, task_type,
                    accuracy, precision, recall, f1_score, processing_speed_ms,
                    memory_usage_mb, test_samples, performance_notes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING *
            `, [
                performanceData.model_name, performanceData.model_type, performanceData.model_version || '1.0',
                performanceData.language, performanceData.task_type, performanceData.accuracy,
                performanceData.precision, performanceData.recall, performanceData.f1_score,
                performanceData.processing_speed_ms, performanceData.memory_usage_mb,
                performanceData.test_samples, performanceData.performance_notes
            ]);

            return {
                success: true,
                model_performance: result.rows[0],
                message: 'Model performance updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update model performance:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update model performance'
            });
        }
    });
}

/**
 * Register analytics routes
 */
async function registerAnalyticsRoutes(app) {
    // Get NLP analytics dashboard
    app.get('/api/nlp-content-analyzer/analytics/dashboard', async (request, reply) => {
        try {
            const { days = 30 } = request.query;

            const analyticsData = await dbPool.query(`
                SELECT * FROM nlp_analysis_summary
                WHERE analysis_date >= CURRENT_DATE - INTERVAL '${days} days'
            `);

            const entitySummary = await dbPool.query(`
                SELECT 
                    entity_type,
                    COUNT(*) as total_extractions,
                    COUNT(CASE WHEN validation_status = true THEN 1 END) as valid_extractions,
                    AVG(confidence) as avg_confidence
                FROM nlp_named_entities ne
                JOIN nlp_content_analysis nca ON ne.analysis_id = nca.analysis_id
                WHERE nca.created_at > NOW() - INTERVAL '${days} days'
                GROUP BY entity_type
                ORDER BY total_extractions DESC
            `);

            const languageStats = await dbPool.query(`
                SELECT 
                    language_detected,
                    COUNT(*) as document_count,
                    AVG(confidence_score) as avg_confidence
                FROM nlp_content_analysis
                WHERE created_at > NOW() - INTERVAL '${days} days'
                GROUP BY language_detected
                ORDER BY document_count DESC
            `);

            return {
                success: true,
                analytics: {
                    summary: analyticsData.rows,
                    entity_summary: entitySummary.rows,
                    language_stats: languageStats.rows,
                    system_metrics: performanceMetrics
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get NLP analytics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve NLP analytics'
            });
        }
    });

    // Calculate NLP statistics
    app.post('/api/nlp-content-analyzer/calculate-stats', async (request, reply) => {
        try {
            await dbPool.query('SELECT calculate_nlp_analysis_stats()');

            return {
                success: true,
                message: 'NLP statistics calculated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to calculate NLP stats:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to calculate NLP statistics'
            });
        }
    });
}

/**
 * Register alert management routes
 */
async function registerAlertManagementRoutes(app) {
    // Get NLP alerts
    app.get('/api/nlp-content-analyzer/alerts', async (request, reply) => {
        try {
            const { severity, status, days = 7 } = request.query;

            let query = `
                SELECT naa.*, nca.document_id, nca.content_type, nca.language_detected
                FROM nlp_analysis_alerts naa
                JOIN nlp_content_analysis nca ON naa.analysis_id = nca.analysis_id
                WHERE naa.created_at > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (severity) {
                query += ` AND naa.severity = $${paramIndex}`;
                params.push(severity);
                paramIndex++;
            }

            if (status) {
                query += ` AND naa.alert_status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            query += ' ORDER BY naa.created_at DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                alerts: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get NLP alerts:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve NLP alerts'
            });
        }
    });

    // Acknowledge alert
    app.put('/api/nlp-content-analyzer/alerts/:alertId/acknowledge', async (request, reply) => {
        try {
            const { alertId } = request.params;
            const { acknowledgment_notes } = request.body;

            const result = await dbPool.query(`
                UPDATE nlp_analysis_alerts
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

    // Generate NLP alerts
    app.post('/api/nlp-content-analyzer/generate-alerts', async (request, reply) => {
        try {
            const alertCount = await dbPool.query('SELECT generate_nlp_analysis_alerts()');
            const generatedAlerts = alertCount.rows[0].generate_nlp_analysis_alerts;

            return {
                success: true,
                alerts_generated: generatedAlerts,
                message: `Generated ${generatedAlerts} new NLP alerts`
            };

        } catch (error) {
            fastify.log.error('Failed to generate NLP alerts:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to generate NLP alerts'
            });
        }
    });
}

/**
 * Generate NLP alert for high-risk content
 */
async function generateNLPAlert(analysisResult) {
    try {
        const alertId = crypto.randomUUID();
        
        await dbPool.query(`
            INSERT INTO nlp_analysis_alerts (
                alert_id, analysis_id, alert_type, severity, title, description,
                risk_score, entities_involved, recommended_actions
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        `, [
            alertId, analysisResult.analysis_id, 'high_risk_content',
            analysisResult.risk_indicator_analysis.risk_score > 0.8 ? 'high' : 'medium',
            'High Risk Content Detected',
            `NLP analysis detected ${analysisResult.risk_indicator_analysis.risk_indicators.length} risk indicators`,
            analysisResult.risk_indicator_analysis.risk_score,
            JSON.stringify(analysisResult.named_entities.slice(0, 5).map(e => e.text)),
            JSON.stringify(analysisResult.risk_indicator_analysis.mitigation_suggestions.slice(0, 3))
        ]);

        // Broadcast alert
        broadcastToClients({
            type: 'nlp_risk_alert',
            data: {
                alert_id: alertId,
                analysis_id: analysisResult.analysis_id,
                risk_score: analysisResult.risk_indicator_analysis.risk_score,
                risk_indicators: analysisResult.risk_indicator_analysis.risk_indicators.length
            }
        });

    } catch (error) {
        fastify.log.error('Failed to generate NLP alert:', error);
    }
}

module.exports = {
    initializeNLPContentAnalyzerRoutes
};