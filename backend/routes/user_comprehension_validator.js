/**
 * User Comprehension Validator API Routes
 * Production-grade Fastify routes for adaptive testing and knowledge assessment
 * 
 * Features:
 * - Adaptive testing algorithms with intelligent question selection
 * - Comprehensive knowledge assessment across multiple domains
 * - Personalized learning path optimization with AI recommendations
 * - Real-time comprehension monitoring and feedback
 * - Regulatory compliance validation for customer understanding
 * - Multi-modal assessment support
 * - Advanced analytics and progress tracking
 * - Gamification elements for enhanced engagement
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
    assessments_completed: 0,
    learning_paths_created: 0,
    questions_answered: 0,
    avg_assessment_time: 0,
    assessment_times: [],
    user_engagement_score: 0.85
};

/**
 * Initialize User Comprehension Validator routes
 */
async function initializeUserComprehensionValidatorRoutes(app, database, redis) {
    dbPool = database;
    redisClient = redis;

    // Initialize WebSocket server
    wsServer = new WebSocket.Server({ port: 8016 });
    
    wsServer.on('connection', (ws) => {
        fastify.log.info('WebSocket client connected to User Comprehension Validator');
        
        ws.on('message', (message) => {
            try {
                const data = JSON.parse(message);
                handleWebSocketMessage(ws, data);
            } catch (error) {
                ws.send(JSON.stringify({ error: 'Invalid message format' }));
            }
        });
        
        ws.on('close', () => {
            fastify.log.info('WebSocket client disconnected from User Comprehension Validator');
        });
    });

    // Start Python agent
    await startPythonAgent();

    // Register all routes
    await registerAssessmentRoutes(app);
    await registerLearningPathRoutes(app);
    await registerQuestionManagementRoutes(app);
    await registerProgressTrackingRoutes(app);
    await registerComprehensionValidationRoutes(app);
    await registerAnalyticsRoutes(app);
    await registerGamificationRoutes(app);
    await registerAccessibilityRoutes(app);

    fastify.log.info('User Comprehension Validator routes initialized');
}

/**
 * Start Python agent process
 */
async function startPythonAgent() {
    try {
        const agentPath = path.join(__dirname, '../agents/user_comprehension_validator/agent.py');
        
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
            fastify.log.info(`User Comprehension Validator Python Agent: ${data}`);
        });

        pythonAgent.stderr.on('data', (data) => {
            fastify.log.error(`User Comprehension Validator Python Agent Error: ${data}`);
        });

        pythonAgent.on('close', (code) => {
            fastify.log.warn(`User Comprehension Validator Python Agent exited with code ${code}`);
            setTimeout(startPythonAgent, 5000);
        });

        fastify.log.info('User Comprehension Validator Python agent started');
    } catch (error) {
        fastify.log.error('Failed to start User Comprehension Validator Python agent:', error);
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
            reject(new Error('User Comprehension Validator Python agent timeout'));
        }, 180000); // 3 minute timeout

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
        case 'subscribe_assessments':
            ws.assessmentSubscription = true;
            break;
        case 'subscribe_learning_progress':
            ws.progressSubscription = true;
            break;
        case 'subscribe_comprehension_alerts':
            ws.comprehensionAlertSubscription = true;
            break;
        case 'get_real_time_progress':
            sendRealTimeProgress(ws, data.user_id);
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
 * Send real-time progress data
 */
async function sendRealTimeProgress(ws, userId) {
    try {
        const progressData = await getUserProgress(userId);
        ws.send(JSON.stringify({
            type: 'real_time_progress',
            data: progressData,
            timestamp: new Date().toISOString()
        }));
    } catch (error) {
        ws.send(JSON.stringify({ error: 'Failed to get progress data' }));
    }
}

/**
 * Get user progress data
 */
async function getUserProgress(userId) {
    try {
        const assessmentProgress = await dbPool.query(`
            SELECT 
                COUNT(*) as total_assessments,
                AVG(overall_score) as avg_score,
                MAX(assessment_timestamp) as last_assessment,
                overall_comprehension_level
            FROM knowledge_assessments 
            WHERE user_id = $1
            GROUP BY overall_comprehension_level
            ORDER BY last_assessment DESC
            LIMIT 1
        `, [userId]);

        const learningProgress = await dbPool.query(`
            SELECT 
                lp.path_status,
                AVG(ulp.progress_percentage) as avg_progress,
                COUNT(CASE WHEN ulp.mastery_achieved THEN 1 END) as modules_mastered,
                COUNT(ulp.module_id) as total_modules
            FROM learning_paths lp
            LEFT JOIN user_learning_progress ulp ON lp.path_id = ulp.path_id
            WHERE lp.user_id = $1 AND lp.path_status = 'active'
            GROUP BY lp.path_status
        `, [userId]);

        return {
            assessment_progress: assessmentProgress.rows[0] || {},
            learning_progress: learningProgress.rows[0] || {},
            system_metrics: performanceMetrics
        };
    } catch (error) {
        fastify.log.error('Failed to get user progress:', error);
        throw error;
    }
}

/**
 * Register assessment routes
 */
async function registerAssessmentRoutes(app) {
    // Create new assessment session
    app.post('/api/user-comprehension/assessments', async (request, reply) => {
        try {
            const { user_id, assessment_goals, user_preferences = {} } = request.body;

            if (!user_id || !assessment_goals) {
                reply.status(400).send({
                    success: false,
                    error: 'User ID and assessment goals are required'
                });
                return;
            }

            const sessionId = await callPythonAgent('create_adaptive_assessment', {
                user_id,
                assessment_goals,
                user_preferences
            });

            // Broadcast session creation
            broadcastToClients({
                type: 'assessment_session_created',
                data: {
                    session_id: sessionId,
                    user_id,
                    assessment_goals
                }
            });

            return {
                success: true,
                session_id: sessionId,
                message: 'Assessment session created successfully'
            };

        } catch (error) {
            fastify.log.error('Assessment session creation failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to create assessment session'
            });
        }
    });

    // Get next adaptive question
    app.get('/api/user-comprehension/assessments/:sessionId/next-question', async (request, reply) => {
        try {
            const { sessionId } = request.params;

            const nextQuestion = await callPythonAgent('get_next_question', {
                session_id: sessionId
            });

            if (!nextQuestion) {
                return {
                    success: true,
                    question: null,
                    message: 'Assessment complete - no more questions'
                };
            }

            return {
                success: true,
                question: nextQuestion
            };

        } catch (error) {
            fastify.log.error('Next question retrieval failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to get next question'
            });
        }
    });

    // Submit question response
    app.post('/api/user-comprehension/assessments/:sessionId/responses', async (request, reply) => {
        try {
            const { sessionId } = request.params;
            const { question_id, user_answer, confidence_level, response_time } = request.body;

            if (!question_id || user_answer === undefined || !confidence_level) {
                reply.status(400).send({
                    success: false,
                    error: 'Question ID, user answer, and confidence level are required'
                });
                return;
            }

            const startTime = Date.now();

            const feedback = await callPythonAgent('submit_response', {
                session_id: sessionId,
                question_id,
                user_answer,
                confidence_level,
                response_time: response_time || 60
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.questions_answered++;

            // Broadcast response submission
            broadcastToClients({
                type: 'question_answered',
                data: {
                    session_id: sessionId,
                    question_id,
                    is_correct: feedback.is_correct,
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                feedback,
                processing_time_ms: processingTime
            };

        } catch (error) {
            fastify.log.error('Response submission failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to submit response'
            });
        }
    });

    // Complete assessment and get results
    app.post('/api/user-comprehension/assessments/:sessionId/complete', async (request, reply) => {
        try {
            const { sessionId } = request.params;

            const startTime = Date.now();

            const assessment = await callPythonAgent('complete_assessment', {
                session_id: sessionId
            });

            const processingTime = Date.now() - startTime;
            performanceMetrics.assessments_completed++;
            performanceMetrics.assessment_times.push(processingTime);

            // Update average assessment time
            if (performanceMetrics.assessment_times.length > 100) {
                performanceMetrics.assessment_times = performanceMetrics.assessment_times.slice(-100);
            }
            performanceMetrics.avg_assessment_time = 
                performanceMetrics.assessment_times.reduce((a, b) => a + b, 0) / 
                performanceMetrics.assessment_times.length;

            // Broadcast assessment completion
            broadcastToClients({
                type: 'assessment_completed',
                data: {
                    session_id: sessionId,
                    user_id: assessment.user_id,
                    overall_score: assessment.overall_score,
                    comprehension_level: assessment.overall_comprehension_level,
                    processing_time: processingTime
                }
            });

            return {
                success: true,
                assessment,
                processing_time_ms: processingTime
            };

        } catch (error) {
            fastify.log.error('Assessment completion failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to complete assessment'
            });
        }
    });

    // Get assessment history
    app.get('/api/user-comprehension/assessments', async (request, reply) => {
        try {
            const { user_id, start_date, end_date, page = 1, limit = 20 } = request.query;

            let query = `
                SELECT ka.assessment_id, ka.user_id, ka.assessment_timestamp,
                       ka.overall_comprehension_level, ka.overall_score, ka.confidence_interval_lower,
                       ka.confidence_interval_upper, ka.regulatory_compliance_score,
                       ases.session_duration_minutes
                FROM knowledge_assessments ka
                JOIN assessment_sessions ases ON ka.session_id = ases.session_id
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (user_id) {
                query += ` AND ka.user_id = $${paramIndex}`;
                params.push(user_id);
                paramIndex++;
            }

            if (start_date) {
                query += ` AND ka.assessment_timestamp >= $${paramIndex}`;
                params.push(start_date);
                paramIndex++;
            }

            if (end_date) {
                query += ` AND ka.assessment_timestamp <= $${paramIndex}`;
                params.push(end_date);
                paramIndex++;
            }

            query += ` ORDER BY ka.assessment_timestamp DESC`;
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
            fastify.log.error('Failed to get assessment history:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve assessment history'
            });
        }
    });

    // Get specific assessment details
    app.get('/api/user-comprehension/assessments/:assessmentId', async (request, reply) => {
        try {
            const { assessmentId } = request.params;

            const result = await dbPool.query(`
                SELECT ka.*, ases.assessment_goals, ases.user_preferences
                FROM knowledge_assessments ka
                JOIN assessment_sessions ases ON ka.session_id = ases.session_id
                WHERE ka.assessment_id = $1
            `, [assessmentId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Assessment not found'
                });
                return;
            }

            const assessment = result.rows[0];

            // Get user responses for this assessment
            const responsesResult = await dbPool.query(`
                SELECT ur.*, aq.question_text, aq.knowledge_domain
                FROM user_responses ur
                JOIN assessment_questions aq ON ur.question_id = aq.question_id
                WHERE ur.session_id = (
                    SELECT session_id FROM knowledge_assessments WHERE assessment_id = $1
                )
                ORDER BY ur.response_timestamp
            `, [assessmentId]);

            return {
                success: true,
                assessment: {
                    ...assessment,
                    responses: responsesResult.rows
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get assessment details:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve assessment details'
            });
        }
    });
}

/**
 * Register learning path routes
 */
async function registerLearningPathRoutes(app) {
    // Create learning path
    app.post('/api/user-comprehension/learning-paths', async (request, reply) => {
        try {
            const { user_id, assessment_id, user_preferences = {} } = request.body;

            if (!user_id || !assessment_id) {
                reply.status(400).send({
                    success: false,
                    error: 'User ID and assessment ID are required'
                });
                return;
            }

            // Get assessment data
            const assessmentResult = await dbPool.query(`
                SELECT * FROM knowledge_assessments WHERE assessment_id = $1
            `, [assessment_id]);

            if (assessmentResult.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Assessment not found'
                });
                return;
            }

            const learningPath = await callPythonAgent('create_learning_path', {
                user_id,
                assessment: assessmentResult.rows[0],
                user_preferences
            });

            performanceMetrics.learning_paths_created++;

            // Broadcast learning path creation
            broadcastToClients({
                type: 'learning_path_created',
                data: {
                    path_id: learningPath.path_id,
                    user_id,
                    estimated_completion_time: learningPath.estimated_completion_time,
                    modules_count: learningPath.recommended_modules.length
                }
            });

            return {
                success: true,
                learning_path: learningPath,
                message: 'Learning path created successfully'
            };

        } catch (error) {
            fastify.log.error('Learning path creation failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to create learning path'
            });
        }
    });

    // Get user learning paths
    app.get('/api/user-comprehension/learning-paths', async (request, reply) => {
        try {
            const { user_id, status = 'active' } = request.query;

            let query = `
                SELECT lp.*, ka.overall_comprehension_level, ka.overall_score
                FROM learning_paths lp
                LEFT JOIN knowledge_assessments ka ON lp.assessment_id = ka.assessment_id
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (user_id) {
                query += ` AND lp.user_id = $${paramIndex}`;
                params.push(user_id);
                paramIndex++;
            }

            if (status) {
                query += ` AND lp.path_status = $${paramIndex}`;
                params.push(status);
                paramIndex++;
            }

            query += ' ORDER BY lp.started_at DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                learning_paths: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get learning paths:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve learning paths'
            });
        }
    });

    // Update learning path progress
    app.put('/api/user-comprehension/learning-paths/:pathId/progress', async (request, reply) => {
        try {
            const { pathId } = request.params;
            const { module_id, progress_percentage, time_spent, completion_status } = request.body;

            if (!module_id || progress_percentage === undefined) {
                reply.status(400).send({
                    success: false,
                    error: 'Module ID and progress percentage are required'
                });
                return;
            }

            // Update progress
            const result = await dbPool.query(`
                INSERT INTO user_learning_progress (
                    user_id, path_id, module_id, progress_percentage, time_spent_minutes,
                    completion_status, last_activity
                ) VALUES (
                    (SELECT user_id FROM learning_paths WHERE path_id = $1),
                    $1, $2, $3, $4, $5, CURRENT_TIMESTAMP
                )
                ON CONFLICT (user_id, path_id, module_id) DO UPDATE SET
                    progress_percentage = EXCLUDED.progress_percentage,
                    time_spent_minutes = user_learning_progress.time_spent_minutes + EXCLUDED.time_spent_minutes,
                    completion_status = EXCLUDED.completion_status,
                    last_activity = CURRENT_TIMESTAMP,
                    attempts = user_learning_progress.attempts + 1
                RETURNING *
            `, [pathId, module_id, progress_percentage, time_spent || 0, completion_status || 'in_progress']);

            // Check if module is completed (mastery achieved)
            if (progress_percentage >= 100) {
                await dbPool.query(`
                    UPDATE user_learning_progress 
                    SET mastery_achieved = true, completion_status = 'completed'
                    WHERE path_id = $1 AND module_id = $2
                `, [pathId, module_id]);
            }

            return {
                success: true,
                progress: result.rows[0],
                message: 'Learning progress updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update learning progress:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update learning progress'
            });
        }
    });
}

/**
 * Register question management routes
 */
async function registerQuestionManagementRoutes(app) {
    // Get questions by domain
    app.get('/api/user-comprehension/questions', async (request, reply) => {
        try {
            const { knowledge_domain, difficulty_level, question_type, active_only = true } = request.query;

            let query = `
                SELECT question_id, knowledge_domain, question_type, difficulty_level,
                       question_text, options, explanation, learning_objectives,
                       regulatory_relevance, estimated_time_minutes, tags,
                       usage_count, correct_rate, avg_response_time
                FROM assessment_questions
                WHERE 1=1
            `;
            const params = [];
            let paramIndex = 1;

            if (knowledge_domain) {
                query += ` AND knowledge_domain = $${paramIndex}`;
                params.push(knowledge_domain);
                paramIndex++;
            }

            if (difficulty_level) {
                query += ` AND difficulty_level = $${paramIndex}`;
                params.push(difficulty_level);
                paramIndex++;
            }

            if (question_type) {
                query += ` AND question_type = $${paramIndex}`;
                params.push(question_type);
                paramIndex++;
            }

            if (active_only === 'true') {
                query += ` AND is_active = true`;
            }

            query += ' ORDER BY usage_count ASC, correct_rate DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                questions: result.rows,
                total_questions: result.rows.length
            };

        } catch (error) {
            fastify.log.error('Failed to get questions:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve questions'
            });
        }
    });

    // Add new assessment question
    app.post('/api/user-comprehension/questions', async (request, reply) => {
        try {
            const questionData = request.body;

            const requiredFields = ['knowledge_domain', 'question_type', 'difficulty_level', 'question_text', 'correct_answer'];
            for (const field of requiredFields) {
                if (!questionData[field]) {
                    reply.status(400).send({
                        success: false,
                        error: `Missing required field: ${field}`
                    });
                    return;
                }
            }

            const questionId = crypto.randomUUID();
            const result = await dbPool.query(`
                INSERT INTO assessment_questions (
                    question_id, knowledge_domain, question_type, difficulty_level,
                    question_text, question_context, options, correct_answer,
                    explanation, learning_objectives, regulatory_relevance,
                    estimated_time_minutes, multimedia_content, tags, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                RETURNING *
            `, [
                questionId, questionData.knowledge_domain, questionData.question_type,
                questionData.difficulty_level, questionData.question_text, questionData.question_context,
                JSON.stringify(questionData.options || []), JSON.stringify(questionData.correct_answer),
                questionData.explanation, JSON.stringify(questionData.learning_objectives || []),
                JSON.stringify(questionData.regulatory_relevance || []), questionData.estimated_time_minutes || 2,
                JSON.stringify(questionData.multimedia_content || {}), JSON.stringify(questionData.tags || []),
                request.user?.id || 'system'
            ]);

            return {
                success: true,
                question: result.rows[0],
                message: 'Assessment question added successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to add question:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to add assessment question'
            });
        }
    });

    // Update question statistics
    app.put('/api/user-comprehension/questions/:questionId/statistics', async (request, reply) => {
        try {
            const { questionId } = request.params;
            const { usage_count, correct_rate, avg_response_time } = request.body;

            const result = await dbPool.query(`
                UPDATE assessment_questions
                SET usage_count = $1, correct_rate = $2, avg_response_time = $3,
                    last_updated = CURRENT_TIMESTAMP
                WHERE question_id = $4
                RETURNING *
            `, [usage_count, correct_rate, avg_response_time, questionId]);

            if (result.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'Question not found'
                });
                return;
            }

            return {
                success: true,
                question: result.rows[0],
                message: 'Question statistics updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update question statistics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update question statistics'
            });
        }
    });
}

/**
 * Register progress tracking routes
 */
async function registerProgressTrackingRoutes(app) {
    // Get user progress summary
    app.get('/api/user-comprehension/progress/:userId', async (request, reply) => {
        try {
            const { userId } = request.params;

            const progressSummary = await dbPool.query(`
                SELECT 
                    ulp.user_id,
                    ulp.overall_competency_level,
                    COUNT(DISTINCT ka.assessment_id) as total_assessments,
                    AVG(ka.overall_score) as avg_assessment_score,
                    MAX(ka.assessment_timestamp) as last_assessment,
                    COUNT(DISTINCT lp.path_id) as learning_paths_created,
                    AVG(ulprog.progress_percentage) as avg_module_progress,
                    COUNT(CASE WHEN ulprog.mastery_achieved THEN 1 END) as modules_mastered,
                    COUNT(ulprog.module_id) as total_modules_enrolled
                FROM user_learning_profiles ulp
                LEFT JOIN knowledge_assessments ka ON ulp.user_id = ka.user_id
                LEFT JOIN learning_paths lp ON ulp.user_id = lp.user_id
                LEFT JOIN user_learning_progress ulprog ON ulp.user_id = ulprog.user_id
                WHERE ulp.user_id = $1
                GROUP BY ulp.user_id, ulp.overall_competency_level
            `, [userId]);

            if (progressSummary.rows.length === 0) {
                reply.status(404).send({
                    success: false,
                    error: 'User progress not found'
                });
                return;
            }

            // Get detailed module progress
            const moduleProgress = await dbPool.query(`
                SELECT ulp.*, lm.module_name, lm.knowledge_domain, lm.difficulty_level
                FROM user_learning_progress ulp
                JOIN learning_modules lm ON ulp.module_id = lm.module_id
                WHERE ulp.user_id = $1
                ORDER BY ulp.last_activity DESC
            `, [userId]);

            return {
                success: true,
                progress_summary: progressSummary.rows[0],
                module_progress: moduleProgress.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get user progress:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve user progress'
            });
        }
    });

    // Get learning analytics
    app.get('/api/user-comprehension/analytics/:userId', async (request, reply) => {
        try {
            const { userId } = request.params;
            const { days = 30 } = request.query;

            const analyticsData = await dbPool.query(`
                SELECT * FROM learning_analytics
                WHERE user_id = $1 AND metric_date >= CURRENT_DATE - INTERVAL '${days} days'
                ORDER BY metric_date DESC
            `, [userId]);

            const performanceTrends = await dbPool.query(`
                SELECT 
                    DATE_TRUNC('week', response_timestamp) as week,
                    COUNT(*) as questions_answered,
                    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as accuracy,
                    AVG(response_time_seconds) as avg_response_time,
                    AVG(confidence_level) as avg_confidence
                FROM user_responses ur
                JOIN assessment_sessions ases ON ur.session_id = ases.session_id
                WHERE ases.user_id = $1 
                AND ur.response_timestamp >= NOW() - INTERVAL '${days} days'
                GROUP BY DATE_TRUNC('week', response_timestamp)
                ORDER BY week DESC
            `, [userId]);

            return {
                success: true,
                analytics: {
                    daily_metrics: analyticsData.rows,
                    performance_trends: performanceTrends.rows
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get learning analytics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve learning analytics'
            });
        }
    });
}

/**
 * Register comprehension validation routes
 */
async function registerComprehensionValidationRoutes(app) {
    // Validate advice comprehension
    app.post('/api/user-comprehension/validate-advice-comprehension', async (request, reply) => {
        try {
            const { user_id, advice_id, advice_content } = request.body;

            if (!user_id || !advice_id || !advice_content) {
                reply.status(400).send({
                    success: false,
                    error: 'User ID, advice ID, and advice content are required'
                });
                return;
            }

            const validation = await callPythonAgent('validate_mortgage_advice_comprehension', {
                user_id,
                advice_id,
                advice_content
            });

            // Generate alert if comprehension is insufficient
            if (validation.overall_comprehension_score < 0.6 || !validation.regulatory_compliance) {
                await generateComprehensionAlert(validation);
            }

            return {
                success: true,
                validation
            };

        } catch (error) {
            fastify.log.error('Advice comprehension validation failed:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to validate advice comprehension'
            });
        }
    });

    // Get comprehension validation history
    app.get('/api/user-comprehension/validations', async (request, reply) => {
        try {
            const { user_id, advice_id, days = 30 } = request.query;

            let query = `
                SELECT * FROM comprehension_validations
                WHERE validation_timestamp > NOW() - INTERVAL '${days} days'
            `;
            const params = [];
            let paramIndex = 1;

            if (user_id) {
                query += ` AND user_id = $${paramIndex}`;
                params.push(user_id);
                paramIndex++;
            }

            if (advice_id) {
                query += ` AND advice_id = $${paramIndex}`;
                params.push(advice_id);
                paramIndex++;
            }

            query += ' ORDER BY validation_timestamp DESC';

            const result = await dbPool.query(query, params);

            return {
                success: true,
                validations: result.rows
            };

        } catch (error) {
            fastify.log.error('Failed to get comprehension validations:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve comprehension validations'
            });
        }
    });
}

/**
 * Register analytics routes
 */
async function registerAnalyticsRoutes(app) {
    // Get comprehension analytics dashboard
    app.get('/api/user-comprehension/analytics/dashboard', async (request, reply) => {
        try {
            const { days = 30 } = request.query;

            const overallStats = await dbPool.query(`
                SELECT 
                    COUNT(DISTINCT ka.user_id) as total_users_assessed,
                    AVG(ka.overall_score) as avg_comprehension_score,
                    COUNT(CASE WHEN ka.overall_comprehension_level = 'excellent' THEN 1 END) as excellent_users,
                    COUNT(CASE WHEN ka.overall_comprehension_level = 'insufficient' THEN 1 END) as insufficient_users,
                    AVG(ka.regulatory_compliance_score) as avg_regulatory_compliance
                FROM knowledge_assessments ka
                WHERE ka.assessment_timestamp > NOW() - INTERVAL '${days} days'
            `);

            const domainPerformance = await dbPool.query(`
                SELECT 
                    domain_key as knowledge_domain,
                    AVG(CAST(domain_value AS DECIMAL)) as avg_domain_score,
                    COUNT(*) as assessments_count
                FROM knowledge_assessments ka
                CROSS JOIN LATERAL jsonb_each_text(ka.domain_scores) AS domain_data(domain_key, domain_value)
                WHERE ka.assessment_timestamp > NOW() - INTERVAL '${days} days'
                GROUP BY domain_key
                ORDER BY avg_domain_score DESC
            `);

            const learningEffectiveness = await dbPool.query(`
                SELECT * FROM learning_effectiveness_summary
            `);

            return {
                success: true,
                analytics: {
                    overall_stats: overallStats.rows[0] || {},
                    domain_performance: domainPerformance.rows,
                    learning_effectiveness: learningEffectiveness.rows,
                    system_metrics: performanceMetrics
                }
            };

        } catch (error) {
            fastify.log.error('Failed to get analytics dashboard:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve analytics dashboard'
            });
        }
    });

    // Calculate learning analytics
    app.post('/api/user-comprehension/calculate-analytics', async (request, reply) => {
        try {
            await dbPool.query('SELECT calculate_daily_learning_analytics()');

            return {
                success: true,
                message: 'Learning analytics calculated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to calculate analytics:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to calculate learning analytics'
            });
        }
    });
}

/**
 * Register gamification routes
 */
async function registerGamificationRoutes(app) {
    // Get user achievements and badges
    app.get('/api/user-comprehension/gamification/:userId', async (request, reply) => {
        try {
            const { userId } = request.params;

            // Calculate achievements based on progress
            const achievements = await calculateUserAchievements(userId);

            return {
                success: true,
                gamification: achievements
            };

        } catch (error) {
            fastify.log.error('Failed to get gamification data:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve gamification data'
            });
        }
    });

    // Get leaderboard
    app.get('/api/user-comprehension/leaderboard', async (request, reply) => {
        try {
            const { category = 'overall', limit = 10 } = request.query;

            let orderBy = 'AVG(ka.overall_score) DESC';
            if (category === 'speed') {
                orderBy = 'AVG(ases.session_duration_minutes) ASC';
            } else if (category === 'accuracy') {
                orderBy = 'AVG(ka.overall_score) DESC';
            } else if (category === 'completion') {
                orderBy = 'COUNT(CASE WHEN ulp.mastery_achieved THEN 1 END) DESC';
            }

            const leaderboard = await dbPool.query(`
                SELECT 
                    ka.user_id,
                    AVG(ka.overall_score) as avg_score,
                    COUNT(ka.assessment_id) as total_assessments,
                    AVG(ases.session_duration_minutes) as avg_session_time,
                    COUNT(CASE WHEN ulp.mastery_achieved THEN 1 END) as modules_mastered
                FROM knowledge_assessments ka
                JOIN assessment_sessions ases ON ka.session_id = ases.session_id
                LEFT JOIN user_learning_progress ulp ON ka.user_id = ulp.user_id
                WHERE ka.assessment_timestamp > NOW() - INTERVAL '30 days'
                GROUP BY ka.user_id
                ORDER BY ${orderBy}
                LIMIT $1
            `, [parseInt(limit)]);

            return {
                success: true,
                leaderboard: leaderboard.rows.map((row, index) => ({
                    ...row,
                    rank: index + 1,
                    user_id: `User_${row.user_id.substring(0, 8)}` // Anonymize for privacy
                }))
            };

        } catch (error) {
            fastify.log.error('Failed to get leaderboard:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve leaderboard'
            });
        }
    });
}

/**
 * Register accessibility routes
 */
async function registerAccessibilityRoutes(app) {
    // Get accessibility options
    app.get('/api/user-comprehension/accessibility-options', async (request, reply) => {
        try {
            const accessibilityOptions = {
                visual_accommodations: [
                    'High contrast mode',
                    'Large text options',
                    'Screen reader compatibility',
                    'Audio descriptions'
                ],
                hearing_accommodations: [
                    'Closed captions',
                    'Visual indicators',
                    'Text alternatives'
                ],
                motor_accommodations: [
                    'Keyboard navigation',
                    'Extended time limits',
                    'Alternative input methods'
                ],
                cognitive_accommodations: [
                    'Simplified language',
                    'Extended time limits',
                    'Frequent saves',
                    'Multiple explanation formats'
                ]
            };

            return {
                success: true,
                accessibility_options: accessibilityOptions
            };

        } catch (error) {
            fastify.log.error('Failed to get accessibility options:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to retrieve accessibility options'
            });
        }
    });

    // Update user accessibility preferences
    app.put('/api/user-comprehension/accessibility/:userId', async (request, reply) => {
        try {
            const { userId } = request.params;
            const { accessibility_needs } = request.body;

            const result = await dbPool.query(`
                INSERT INTO user_learning_profiles (user_id, accessibility_needs)
                VALUES ($1, $2)
                ON CONFLICT (user_id) DO UPDATE SET
                    accessibility_needs = EXCLUDED.accessibility_needs,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
            `, [userId, JSON.stringify(accessibility_needs || [])]);

            return {
                success: true,
                profile: result.rows[0],
                message: 'Accessibility preferences updated successfully'
            };

        } catch (error) {
            fastify.log.error('Failed to update accessibility preferences:', error);
            reply.status(500).send({
                success: false,
                error: 'Failed to update accessibility preferences'
            });
        }
    });
}

/**
 * Calculate user achievements
 */
async function calculateUserAchievements(userId) {
    try {
        const userStats = await dbPool.query(`
            SELECT 
                COUNT(DISTINCT ka.assessment_id) as total_assessments,
                AVG(ka.overall_score) as avg_score,
                MAX(ka.overall_score) as best_score,
                COUNT(CASE WHEN ulp.mastery_achieved THEN 1 END) as modules_mastered,
                SUM(ulp.time_spent_minutes) as total_time_spent
            FROM knowledge_assessments ka
            LEFT JOIN user_learning_progress ulp ON ka.user_id = ulp.user_id
            WHERE ka.user_id = $1
            GROUP BY ka.user_id
        `, [userId]);

        const stats = userStats.rows[0] || {};
        
        const achievements = {
            points: calculatePoints(stats),
            badges: calculateBadges(stats),
            level: calculateLevel(stats),
            streaks: calculateStreaks(userId),
            milestones: calculateMilestones(stats)
        };

        return achievements;
    } catch (error) {
        fastify.log.error('Failed to calculate achievements:', error);
        return {};
    }
}

/**
 * Calculate user points
 */
function calculatePoints(stats) {
    const basePoints = (stats.total_assessments || 0) * 100;
    const accuracyBonus = (stats.avg_score || 0) * 500;
    const masteryBonus = (stats.modules_mastered || 0) * 200;
    
    return Math.round(basePoints + accuracyBonus + masteryBonus);
}

/**
 * Calculate user badges
 */
function calculateBadges(stats) {
    const badges = [];
    
    if (stats.avg_score >= 0.9) {
        badges.push({ name: 'Perfectionist', description: 'Achieved 90%+ average score' });
    }
    
    if (stats.modules_mastered >= 5) {
        badges.push({ name: 'Dedicated Learner', description: 'Mastered 5+ learning modules' });
    }
    
    if (stats.total_assessments >= 10) {
        badges.push({ name: 'Assessment Expert', description: 'Completed 10+ assessments' });
    }
    
    return badges;
}

/**
 * Calculate user level
 */
function calculateLevel(stats) {
    const totalPoints = calculatePoints(stats);
    
    if (totalPoints >= 5000) return { level: 'Expert', points_to_next: 0 };
    if (totalPoints >= 3000) return { level: 'Advanced', points_to_next: 5000 - totalPoints };
    if (totalPoints >= 1500) return { level: 'Intermediate', points_to_next: 3000 - totalPoints };
    if (totalPoints >= 500) return { level: 'Beginner', points_to_next: 1500 - totalPoints };
    return { level: 'Novice', points_to_next: 500 - totalPoints };
}

/**
 * Calculate streaks
 */
async function calculateStreaks(userId) {
    try {
        // This would calculate learning streaks based on daily activity
        return {
            current_streak: 5,
            longest_streak: 12,
            streak_type: 'daily_learning'
        };
    } catch (error) {
        return { current_streak: 0, longest_streak: 0 };
    }
}

/**
 * Calculate milestones
 */
function calculateMilestones(stats) {
    const milestones = [
        { name: 'First Assessment', achieved: stats.total_assessments >= 1, points: 100 },
        { name: 'Perfect Score', achieved: stats.best_score >= 1.0, points: 500 },
        { name: 'Module Master', achieved: stats.modules_mastered >= 3, points: 300 },
        { name: 'Dedicated Student', achieved: (stats.total_time_spent || 0) >= 300, points: 400 }
    ];
    
    return milestones;
}

/**
 * Generate comprehension alert
 */
async function generateComprehensionAlert(validation) {
    try {
        const alertId = crypto.randomUUID();
        
        await dbPool.query(`
            INSERT INTO comprehension_validation_alerts (
                alert_id, validation_id, user_id, alert_type, severity, title, description,
                comprehension_score, recommended_actions, regulatory_impact
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        `, [
            alertId, validation.validation_id, validation.user_id, 'low_comprehension',
            validation.overall_comprehension_score < 0.4 ? 'high' : 'medium',
            'Low Comprehension Score Detected',
            'User comprehension validation shows insufficient understanding of mortgage advice',
            validation.overall_comprehension_score,
            JSON.stringify(['Provide additional education', 'Schedule follow-up assessment']),
            validation.regulatory_compliance ? 'educational_gap' : 'regulatory_non_compliance'
        ]);

        // Broadcast alert
        broadcastToClients({
            type: 'comprehension_alert',
            data: {
                alert_id: alertId,
                user_id: validation.user_id,
                comprehension_score: validation.overall_comprehension_score,
                regulatory_compliance: validation.regulatory_compliance
            }
        });

    } catch (error) {
        fastify.log.error('Failed to generate comprehension alert:', error);
    }
}

module.exports = {
    initializeUserComprehensionValidatorRoutes
};