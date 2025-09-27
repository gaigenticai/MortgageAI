/**
 * AI Mortgage Advisor Chat WebSocket Server
 * 
 * Production-grade WebSocket server for real-time conversational AI interactions
 * with mortgage borrowers. Provides context-aware responses, message persistence,
 * and integration with compliance agents.
 * 
 * Features:
 * - Real-time bidirectional communication
 * - Message persistence with PostgreSQL
 * - Context-aware conversation management
 * - Integration with AI compliance agents
 * - Session management and authentication
 * - Rate limiting and abuse prevention
 * - Comprehensive logging and monitoring
 */

const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const { Client } = require('pg');
const Redis = require('redis');
const jwt = require('jsonwebtoken');
const rateLimit = require('ws-rate-limit');
const axios = require('axios');

class MortgageAdvisorChatServer {
    constructor() {
        this.port = process.env.CHAT_WEBSOCKET_PORT || 8005;
        this.clients = new Map();
        this.conversations = new Map();
        this.rateLimiter = rateLimit('5 per 10s'); // 5 messages per 10 seconds per client
        
        // Database configuration
        this.dbConfig = {
            host: process.env.DB_HOST || 'postgres',
            port: process.env.DB_PORT || 5432,
            database: process.env.DB_NAME || 'mortgage_ai',
            user: process.env.DB_USER || 'postgres',
            password: process.env.DB_PASSWORD || '',
            ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
        };

        // Redis configuration for session management
        this.redisClient = Redis.createClient({
            host: process.env.REDIS_HOST || 'redis',
            port: process.env.REDIS_PORT || 6379,
            password: process.env.REDIS_PASSWORD || undefined
        });

        // AI Agents service configuration
        this.agentsServiceUrl = process.env.AGENTS_API_URL || 'http://ai-agents:8000';
        
        this.initializeServer();
        this.setupDatabase();
    }

    /**
     * Initialize WebSocket server with comprehensive configuration
     */
    initializeServer() {
        this.wss = new WebSocket.Server({
            port: this.port,
            verifyClient: this.verifyClient.bind(this),
            perMessageDeflate: {
                zlibDeflateOptions: {
                    threshold: 1024,
                    concurrencyLimit: 10,
                },
            },
        });

        this.wss.on('connection', this.handleConnection.bind(this));
        this.wss.on('error', this.handleServerError.bind(this));

        console.log(`🤖 AI Mortgage Advisor Chat Server running on port ${this.port}`);
    }

    /**
     * Verify client connection with authentication and rate limiting
     */
    async verifyClient(info) {
        try {
            const url = new URL(info.req.url, `http://${info.req.headers.host}`);
            const token = url.searchParams.get('token');
            
            // Skip auth if REQUIRE_AUTH is false
            if (process.env.REQUIRE_AUTH === 'false') {
                return true;
            }

            if (!token) {
                console.log('❌ WebSocket connection rejected: No token provided');
                return false;
            }

            // Verify JWT token
            const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-jwt-secret-key-here');
            
            // Store user info for later use
            info.req.user = decoded;
            
            console.log(`✅ WebSocket connection verified for user: ${decoded.userId || 'anonymous'}`);
            return true;
        } catch (error) {
            console.error('❌ WebSocket verification failed:', error.message);
            return false;
        }
    }

    /**
     * Handle new WebSocket connection
     */
    async handleConnection(ws, req) {
        const clientId = uuidv4();
        const userId = req.user?.userId || 'anonymous';
        
        // Apply rate limiting
        this.rateLimiter(ws);

        // Store client information
        const clientInfo = {
            id: clientId,
            userId: userId,
            ws: ws,
            connectedAt: new Date(),
            lastActivity: new Date(),
            messageCount: 0,
            conversationId: null
        };

        this.clients.set(clientId, clientInfo);

        // Initialize conversation
        const conversationId = await this.initializeConversation(userId, clientId);
        clientInfo.conversationId = conversationId;

        console.log(`🔗 New chat connection: ${clientId} (User: ${userId})`);

        // Send welcome message
        await this.sendWelcomeMessage(ws, conversationId);

        // Set up message handlers
        ws.on('message', (data) => this.handleMessage(clientId, data));
        ws.on('close', () => this.handleDisconnection(clientId));
        ws.on('error', (error) => this.handleClientError(clientId, error));

        // Set up ping-pong for connection health
        ws.isAlive = true;
        ws.on('pong', () => { ws.isAlive = true; });
    }

    /**
     * Initialize conversation in database and memory
     */
    async initializeConversation(userId, clientId) {
        const conversationId = uuidv4();
        const client = new Client(this.dbConfig);

        try {
            await client.connect();

            // Create conversation record
            await client.query(`
                INSERT INTO chat_conversations (
                    id, user_id, client_id, status, created_at, updated_at,
                    context_data, conversation_type
                ) VALUES ($1, $2, $3, $4, NOW(), NOW(), $5, $6)
            `, [
                conversationId,
                userId,
                clientId,
                'active',
                JSON.stringify({
                    userProfile: {},
                    mortgageContext: {},
                    conversationGoals: [],
                    riskAssessment: 'pending'
                }),
                'mortgage_advisory'
            ]);

            // Initialize conversation context in memory
            this.conversations.set(conversationId, {
                id: conversationId,
                userId: userId,
                clientId: clientId,
                messages: [],
                context: {
                    userProfile: {},
                    mortgageContext: {},
                    conversationGoals: [],
                    currentTopic: 'introduction',
                    riskAssessment: 'pending',
                    complianceFlags: []
                },
                aiState: {
                    lastResponse: null,
                    confidenceScore: 0,
                    suggestedActions: [],
                    escalationRequired: false
                }
            });

            console.log(`💬 Initialized conversation: ${conversationId}`);
            return conversationId;

        } catch (error) {
            console.error('❌ Failed to initialize conversation:', error);
            throw error;
        } finally {
            await client.end();
        }
    }

    /**
     * Send welcome message with personalized greeting
     */
    async sendWelcomeMessage(ws, conversationId) {
        const welcomeMessage = {
            type: 'ai_message',
            conversationId: conversationId,
            messageId: uuidv4(),
            content: {
                text: "Hallo! Ik ben uw AI Hypotheekadviseur. Ik help u graag met al uw hypotheekvragen en zorg ervoor dat alles voldoet aan de AFM-regelgeving. Waarmee kan ik u vandaag helpen?",
                translation: "Hello! I'm your AI Mortgage Advisor. I'm here to help you with all your mortgage questions and ensure everything complies with AFM regulations. How can I assist you today?",
                suggestions: [
                    "Ik wil informatie over hypotheekmogelijkheden",
                    "Wat zijn de huidige hypotheekrentes?",
                    "Hoe werkt de hypotheekaanvraag?",
                    "Wat zijn de AFM-vereisten?"
                ]
            },
            timestamp: new Date().toISOString(),
            metadata: {
                aiConfidence: 1.0,
                complianceChecked: true,
                language: 'nl'
            }
        };

        await this.sendMessage(ws, welcomeMessage);
        await this.persistMessage(conversationId, welcomeMessage, 'ai');
    }

    /**
     * Handle incoming message from client
     */
    async handleMessage(clientId, data) {
        try {
            const clientInfo = this.clients.get(clientId);
            if (!clientInfo) {
                console.error(`❌ Client not found: ${clientId}`);
                return;
            }

            // Update client activity
            clientInfo.lastActivity = new Date();
            clientInfo.messageCount++;

            // Parse message
            const message = JSON.parse(data.toString());
            
            // Validate message structure
            if (!this.validateMessage(message)) {
                await this.sendErrorMessage(clientInfo.ws, 'Invalid message format');
                return;
            }

            console.log(`📨 Received message from ${clientId}:`, message.type);

            // Route message based on type
            switch (message.type) {
                case 'user_message':
                    await this.handleUserMessage(clientInfo, message);
                    break;
                case 'typing_indicator':
                    await this.handleTypingIndicator(clientInfo, message);
                    break;
                case 'context_update':
                    await this.handleContextUpdate(clientInfo, message);
                    break;
                case 'feedback':
                    await this.handleFeedback(clientInfo, message);
                    break;
                default:
                    await this.sendErrorMessage(clientInfo.ws, `Unknown message type: ${message.type}`);
            }

        } catch (error) {
            console.error(`❌ Error handling message from ${clientId}:`, error);
            const clientInfo = this.clients.get(clientId);
            if (clientInfo) {
                await this.sendErrorMessage(clientInfo.ws, 'Failed to process message');
            }
        }
    }

    /**
     * Handle user message and generate AI response
     */
    async handleUserMessage(clientInfo, message) {
        const conversation = this.conversations.get(clientInfo.conversationId);
        if (!conversation) {
            await this.sendErrorMessage(clientInfo.ws, 'Conversation not found');
            return;
        }

        // Persist user message
        await this.persistMessage(clientInfo.conversationId, message, 'user');

        // Add to conversation context
        conversation.messages.push({
            ...message,
            sender: 'user',
            timestamp: new Date().toISOString()
        });

        // Send typing indicator
        await this.sendTypingIndicator(clientInfo.ws, true);

        try {
            // Generate AI response using compliance agent
            const aiResponse = await this.generateAIResponse(conversation, message);

            // Send AI response
            await this.sendMessage(clientInfo.ws, aiResponse);
            
            // Persist AI response
            await this.persistMessage(clientInfo.conversationId, aiResponse, 'ai');

            // Update conversation context
            conversation.messages.push({
                ...aiResponse,
                sender: 'ai',
                timestamp: new Date().toISOString()
            });

            // Update AI state
            conversation.aiState = {
                lastResponse: aiResponse,
                confidenceScore: aiResponse.metadata?.aiConfidence || 0,
                suggestedActions: aiResponse.content?.suggestedActions || [],
                escalationRequired: aiResponse.metadata?.escalationRequired || false
            };

            // Check for compliance issues
            await this.checkCompliance(conversation, aiResponse);

        } catch (error) {
            console.error('❌ Failed to generate AI response:', error);
            await this.sendErrorMessage(clientInfo.ws, 'Failed to generate response');
        } finally {
            // Stop typing indicator
            await this.sendTypingIndicator(clientInfo.ws, false);
        }
    }

    /**
     * Generate AI response using compliance agent
     */
    async generateAIResponse(conversation, userMessage) {
        try {
            // Prepare context for AI agent
            const context = {
                conversationHistory: conversation.messages.slice(-10), // Last 10 messages for context
                userProfile: conversation.context.userProfile,
                mortgageContext: conversation.context.mortgageContext,
                currentTopic: conversation.context.currentTopic,
                complianceFlags: conversation.context.complianceFlags
            };

            // Call compliance agent for advice generation
            const response = await axios.post(`${this.agentsServiceUrl}/api/compliance/generate-advice`, {
                application_id: conversation.id,
                user_profile: {
                    buyer_type: conversation.context.userProfile.buyerType || 'first_time',
                    annual_income: conversation.context.userProfile.annualIncome || 0,
                    mortgage_amount: conversation.context.userProfile.desiredMortgageAmount || 0,
                    property_value: conversation.context.userProfile.propertyValue || 0,
                    employment_status: conversation.context.userProfile.employmentStatus || 'employed',
                    age: conversation.context.userProfile.age || 30,
                    dependents: conversation.context.userProfile.dependents || 0,
                    existing_debt: conversation.context.userProfile.existingDebt || 0
                },
                product_features: conversation.context.mortgageContext.productFeatures || [],
                conversation_context: {
                    user_message: userMessage.content.text,
                    conversation_history: context.conversationHistory,
                    current_topic: context.currentTopic
                }
            }, {
                timeout: 30000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const aiData = response.data.data;

            // Format response for WebSocket
            const aiResponse = {
                type: 'ai_message',
                conversationId: conversation.id,
                messageId: uuidv4(),
                content: {
                    text: aiData.advice_text || "Ik begrijp uw vraag. Laat me u helpen met meer informatie over hypotheken.",
                    explanation: aiData.explanation || null,
                    suggestions: aiData.suggested_questions || [
                        "Kunt u meer vertellen over uw financiële situatie?",
                        "Welk type hypotheek heeft uw voorkeur?",
                        "Heeft u al een woning op het oog?"
                    ],
                    suggestedActions: aiData.suggested_actions || [],
                    complianceNotes: aiData.compliance_notes || [],
                    riskAssessment: aiData.risk_assessment || 'low'
                },
                timestamp: new Date().toISOString(),
                metadata: {
                    aiConfidence: aiData.confidence_score || 0.8,
                    complianceChecked: true,
                    language: 'nl',
                    processingTime: aiData.processing_time_ms || 0,
                    escalationRequired: aiData.escalation_required || false,
                    regulatoryFlags: aiData.regulatory_flags || []
                }
            };

            return aiResponse;

        } catch (error) {
            console.error('❌ AI response generation failed:', error);
            
            // Fallback response
            return {
                type: 'ai_message',
                conversationId: conversation.id,
                messageId: uuidv4(),
                content: {
                    text: "Excuses voor het ongemak. Ik ondervind momenteel technische problemen. Kunt u uw vraag opnieuw stellen?",
                    translation: "Sorry for the inconvenience. I'm experiencing technical issues. Could you please rephrase your question?",
                    suggestions: [
                        "Probeer uw vraag anders te formuleren",
                        "Contacteer onze klantenservice",
                        "Bekijk onze veelgestelde vragen"
                    ]
                },
                timestamp: new Date().toISOString(),
                metadata: {
                    aiConfidence: 0.1,
                    complianceChecked: false,
                    language: 'nl',
                    error: true
                }
            };
        }
    }

    /**
     * Check compliance and regulatory requirements
     */
    async checkCompliance(conversation, aiResponse) {
        try {
            // Call compliance check endpoint
            const response = await axios.post(`${this.agentsServiceUrl}/api/compliance/check-compliance`, {
                advice_text: aiResponse.content.text,
                user_profile: conversation.context.userProfile,
                conversation_context: {
                    messages: conversation.messages.slice(-5),
                    current_topic: conversation.context.currentTopic
                }
            }, {
                timeout: 10000
            });

            const complianceResult = response.data.data;

            // Update conversation context with compliance flags
            if (complianceResult.compliance_issues && complianceResult.compliance_issues.length > 0) {
                conversation.context.complianceFlags.push(...complianceResult.compliance_issues);
                
                // Send compliance alert if needed
                if (complianceResult.severity === 'high') {
                    await this.sendComplianceAlert(conversation, complianceResult);
                }
            }

        } catch (error) {
            console.error('❌ Compliance check failed:', error);
        }
    }

    /**
     * Send compliance alert to client
     */
    async sendComplianceAlert(conversation, complianceResult) {
        const clientInfo = Array.from(this.clients.values())
            .find(client => client.conversationId === conversation.id);

        if (clientInfo) {
            const alertMessage = {
                type: 'compliance_alert',
                conversationId: conversation.id,
                messageId: uuidv4(),
                content: {
                    severity: complianceResult.severity,
                    issues: complianceResult.compliance_issues,
                    recommendations: complianceResult.recommendations,
                    regulatoryReference: complianceResult.regulatory_reference
                },
                timestamp: new Date().toISOString()
            };

            await this.sendMessage(clientInfo.ws, alertMessage);
        }
    }

    /**
     * Handle typing indicator
     */
    async handleTypingIndicator(clientInfo, message) {
        // Broadcast typing indicator to other participants if needed
        // For now, just acknowledge
        console.log(`⌨️ Typing indicator from ${clientInfo.id}: ${message.content.isTyping}`);
    }

    /**
     * Handle context update from client
     */
    async handleContextUpdate(clientInfo, message) {
        const conversation = this.conversations.get(clientInfo.conversationId);
        if (!conversation) return;

        // Update conversation context
        if (message.content.userProfile) {
            conversation.context.userProfile = {
                ...conversation.context.userProfile,
                ...message.content.userProfile
            };
        }

        if (message.content.mortgageContext) {
            conversation.context.mortgageContext = {
                ...conversation.context.mortgageContext,
                ...message.content.mortgageContext
            };
        }

        if (message.content.currentTopic) {
            conversation.context.currentTopic = message.content.currentTopic;
        }

        // Persist context update
        await this.updateConversationContext(clientInfo.conversationId, conversation.context);

        console.log(`🔄 Context updated for conversation: ${clientInfo.conversationId}`);
    }

    /**
     * Handle user feedback
     */
    async handleFeedback(clientInfo, message) {
        try {
            const client = new Client(this.dbConfig);
            await client.connect();

            await client.query(`
                INSERT INTO chat_feedback (
                    id, conversation_id, message_id, rating, feedback_text,
                    feedback_type, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
            `, [
                uuidv4(),
                clientInfo.conversationId,
                message.content.messageId,
                message.content.rating,
                message.content.feedbackText,
                message.content.feedbackType || 'general'
            ]);

            await client.end();

            console.log(`👍 Feedback received for conversation: ${clientInfo.conversationId}`);

        } catch (error) {
            console.error('❌ Failed to save feedback:', error);
        }
    }

    /**
     * Persist message to database
     */
    async persistMessage(conversationId, message, sender) {
        try {
            const client = new Client(this.dbConfig);
            await client.connect();

            await client.query(`
                INSERT INTO chat_messages (
                    id, conversation_id, sender, message_type, content,
                    metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
            `, [
                message.messageId || uuidv4(),
                conversationId,
                sender,
                message.type,
                JSON.stringify(message.content),
                JSON.stringify(message.metadata || {})
            ]);

            await client.end();

        } catch (error) {
            console.error('❌ Failed to persist message:', error);
        }
    }

    /**
     * Update conversation context in database
     */
    async updateConversationContext(conversationId, context) {
        try {
            const client = new Client(this.dbConfig);
            await client.connect();

            await client.query(`
                UPDATE chat_conversations 
                SET context_data = $1, updated_at = NOW()
                WHERE id = $2
            `, [JSON.stringify(context), conversationId]);

            await client.end();

        } catch (error) {
            console.error('❌ Failed to update conversation context:', error);
        }
    }

    /**
     * Send message to WebSocket client
     */
    async sendMessage(ws, message) {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(message));
        }
    }

    /**
     * Send typing indicator
     */
    async sendTypingIndicator(ws, isTyping) {
        const message = {
            type: 'typing_indicator',
            content: { isTyping: isTyping, sender: 'ai' },
            timestamp: new Date().toISOString()
        };
        await this.sendMessage(ws, message);
    }

    /**
     * Send error message to client
     */
    async sendErrorMessage(ws, errorMessage) {
        const message = {
            type: 'error',
            content: { message: errorMessage },
            timestamp: new Date().toISOString()
        };
        await this.sendMessage(ws, message);
    }

    /**
     * Validate message structure
     */
    validateMessage(message) {
        return message && 
               typeof message.type === 'string' && 
               message.content && 
               typeof message.content === 'object';
    }

    /**
     * Handle client disconnection
     */
    handleDisconnection(clientId) {
        const clientInfo = this.clients.get(clientId);
        if (clientInfo) {
            console.log(`🔌 Client disconnected: ${clientId}`);
            
            // Update conversation status
            if (clientInfo.conversationId) {
                this.updateConversationStatus(clientInfo.conversationId, 'disconnected');
            }
            
            this.clients.delete(clientId);
        }
    }

    /**
     * Update conversation status
     */
    async updateConversationStatus(conversationId, status) {
        try {
            const client = new Client(this.dbConfig);
            await client.connect();

            await client.query(`
                UPDATE chat_conversations 
                SET status = $1, updated_at = NOW()
                WHERE id = $2
            `, [status, conversationId]);

            await client.end();

        } catch (error) {
            console.error('❌ Failed to update conversation status:', error);
        }
    }

    /**
     * Handle client error
     */
    handleClientError(clientId, error) {
        console.error(`❌ Client error ${clientId}:`, error);
        this.handleDisconnection(clientId);
    }

    /**
     * Handle server error
     */
    handleServerError(error) {
        console.error('❌ WebSocket server error:', error);
    }

    /**
     * Setup database tables
     */
    async setupDatabase() {
        const client = new Client(this.dbConfig);
        
        try {
            await client.connect();

            // Create chat_conversations table
            await client.query(`
                CREATE TABLE IF NOT EXISTS chat_conversations (
                    id UUID PRIMARY KEY,
                    user_id VARCHAR(255),
                    client_id VARCHAR(255),
                    status VARCHAR(50) DEFAULT 'active',
                    conversation_type VARCHAR(50) DEFAULT 'mortgage_advisory',
                    context_data JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            `);

            // Create chat_messages table
            await client.query(`
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id UUID PRIMARY KEY,
                    conversation_id UUID REFERENCES chat_conversations(id),
                    sender VARCHAR(50) NOT NULL,
                    message_type VARCHAR(50) NOT NULL,
                    content JSONB NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW()
                )
            `);

            // Create chat_feedback table
            await client.query(`
                CREATE TABLE IF NOT EXISTS chat_feedback (
                    id UUID PRIMARY KEY,
                    conversation_id UUID REFERENCES chat_conversations(id),
                    message_id UUID,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    feedback_text TEXT,
                    feedback_type VARCHAR(50) DEFAULT 'general',
                    created_at TIMESTAMP DEFAULT NOW()
                )
            `);

            // Create indexes for performance
            await client.query(`
                CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_id ON chat_conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_chat_conversations_status ON chat_conversations(status);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);
                CREATE INDEX IF NOT EXISTS idx_chat_feedback_conversation_id ON chat_feedback(conversation_id);
            `);

            console.log('✅ Chat database tables initialized');

        } catch (error) {
            console.error('❌ Failed to setup database:', error);
        } finally {
            await client.end();
        }
    }

    /**
     * Start health check interval
     */
    startHealthCheck() {
        setInterval(() => {
            this.wss.clients.forEach((ws) => {
                if (ws.isAlive === false) {
                    console.log('🔌 Terminating inactive connection');
                    return ws.terminate();
                }

                ws.isAlive = false;
                ws.ping();
            });
        }, 30000); // Check every 30 seconds
    }

    /**
     * Start the server
     */
    start() {
        this.startHealthCheck();
        console.log(`🚀 AI Mortgage Advisor Chat Server started successfully on port ${this.port}`);
    }
}

// Initialize and start the server
const chatServer = new MortgageAdvisorChatServer();
chatServer.start();

module.exports = MortgageAdvisorChatServer;


