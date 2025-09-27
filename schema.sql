-- MortgageAI Database Schema
-- This schema supports the dual-agent framework for mortgage advice and application quality control

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table (for authentication when REQUIRE_AUTH=true)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Regulations table for AFM compliance rules
CREATE TABLE IF NOT EXISTS regulations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    regulation_code VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    effective_date DATE NOT NULL,
    expiry_date DATE,
    source_url VARCHAR(1000),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Mortgage applications table
CREATE TABLE IF NOT EXISTS mortgage_applications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    application_number VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'draft',
    applicant_data JSONB NOT NULL,
    documents JSONB DEFAULT '[]',
    qc_score DECIMAL(5,2),
    compliance_score DECIMAL(5,2),
    advice_draft TEXT,
    final_advice TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP WITH TIME ZONE,
    approved_at TIMESTAMP WITH TIME ZONE
);

-- Application documents table
CREATE TABLE IF NOT EXISTS application_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id UUID REFERENCES mortgage_applications(id) ON DELETE CASCADE,
    document_type VARCHAR(100) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    ocr_text TEXT,
    extracted_data JSONB,
    is_valid BOOLEAN DEFAULT false,
    validation_errors JSONB DEFAULT '[]',
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- QC validation results table
CREATE TABLE IF NOT EXISTS qc_validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id UUID REFERENCES mortgage_applications(id) ON DELETE CASCADE,
    document_id UUID REFERENCES application_documents(id),
    field_name VARCHAR(255) NOT NULL,
    field_value TEXT,
    validation_rule VARCHAR(255),
    is_valid BOOLEAN NOT NULL,
    error_message TEXT,
    severity VARCHAR(20) DEFAULT 'error',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Compliance checks table
CREATE TABLE IF NOT EXISTS compliance_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id UUID REFERENCES mortgage_applications(id) ON DELETE CASCADE,
    regulation_id UUID REFERENCES regulations(id),
    check_type VARCHAR(100) NOT NULL,
    content_analyzed TEXT,
    is_compliant BOOLEAN NOT NULL,
    issues_found JSONB DEFAULT '[]',
    suggestions TEXT,
    readability_score DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent interactions table for audit trail
CREATE TABLE IF NOT EXISTS agent_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id UUID REFERENCES mortgage_applications(id) ON DELETE CASCADE,
    agent_type VARCHAR(50) NOT NULL, -- 'compliance' or 'quality_control'
    interaction_type VARCHAR(100) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    processing_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User feedback table for continuous learning
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id UUID REFERENCES mortgage_applications(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    feedback_type VARCHAR(50) NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comments TEXT,
    improvement_suggestions TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_mortgage_applications_user_id ON mortgage_applications(user_id);
CREATE INDEX IF NOT EXISTS idx_mortgage_applications_status ON mortgage_applications(status);
CREATE INDEX IF NOT EXISTS idx_application_documents_application_id ON application_documents(application_id);
CREATE INDEX IF NOT EXISTS idx_qc_validations_application_id ON qc_validations(application_id);
CREATE INDEX IF NOT EXISTS idx_compliance_checks_application_id ON compliance_checks(application_id);
CREATE INDEX IF NOT EXISTS idx_agent_interactions_application_id ON agent_interactions(application_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_regulations_content_fts ON regulations USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_application_documents_ocr_fts ON application_documents USING gin(to_tsvector('english', ocr_text));

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_mortgage_applications_updated_at BEFORE UPDATE ON mortgage_applications FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_regulations_updated_at BEFORE UPDATE ON regulations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- PHASE 1: DUTCH AFM-COMPLIANT MORTGAGE ADVISORY PLATFORM TABLES
-- =============================================================================

-- Dutch AFM Regulations table (replaces generic regulations for AFM compliance)
CREATE TABLE IF NOT EXISTS afm_regulations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    regulation_code VARCHAR(50) UNIQUE NOT NULL, -- e.g., "Wft_86f", "BGfo_8_1"
    article_reference VARCHAR(100), -- e.g., "Article 86f Wft"
    regulation_type VARCHAR(50) CHECK (regulation_type IN ('disclosure', 'suitability', 'documentation')),
    title_nl VARCHAR(500) NOT NULL,
    title_en VARCHAR(500),
    content_nl TEXT NOT NULL,
    content_en TEXT,
    applicability JSONB, -- mortgage types this applies to
    mandatory_disclosures JSONB, -- required disclosure elements
    compliance_criteria JSONB, -- validation criteria
    effective_date DATE NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Dutch Mortgage Products table
CREATE TABLE IF NOT EXISTS dutch_mortgage_products (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    product_name VARCHAR(200) NOT NULL,
    lender_name VARCHAR(200) NOT NULL,
    product_type VARCHAR(50) CHECK (product_type IN ('fixed', 'variable', 'hybrid')),
    interest_rate_type VARCHAR(50) CHECK (interest_rate_type IN ('fixed_1y', 'fixed_5y', 'fixed_10y', 'fixed_20y', 'fixed_30y', 'variable')),
    nhg_eligible BOOLEAN DEFAULT false,
    max_ltv_percentage DECIMAL(5,2), -- loan-to-value ratio
    max_dti_ratio DECIMAL(5,2), -- debt-to-income ratio
    minimum_income DECIMAL(12,2),
    required_documents JSONB, -- array of required document types
    afm_disclosures JSONB, -- required AFM disclosures for this product
    lender_criteria JSONB, -- specific underwriting criteria
    processing_sla_hours INTEGER, -- expected processing time
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Client Profiles (AFM-compliant)
CREATE TABLE IF NOT EXISTS client_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    advisor_id UUID REFERENCES users(id),
    client_name VARCHAR(200) NOT NULL,
    bsn VARCHAR(9) UNIQUE, -- Dutch social security number (BSN validation)
    date_of_birth DATE NOT NULL,
    marital_status VARCHAR(50) CHECK (marital_status IN ('single', 'married', 'registered_partnership', 'divorced', 'widowed')),
    number_of_dependents INTEGER DEFAULT 0,
    employment_status VARCHAR(100) CHECK (employment_status IN ('employed', 'self_employed', 'unemployed', 'retired', 'student')),
    gross_annual_income DECIMAL(12,2),
    partner_income DECIMAL(12,2),
    existing_debts JSONB DEFAULT '[]', -- array of existing debts with amounts and creditors
    property_purchase_intention VARCHAR(100) CHECK (property_purchase_intention IN ('first_home', 'move_up', 'investment', 'recurring')),
    risk_profile VARCHAR(50) CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive')),
    sustainability_preference VARCHAR(50) CHECK (sustainability_preference IN ('none', 'green_mortgage', 'energy_efficient')),
    afm_questionnaire_completed BOOLEAN DEFAULT false,
    afm_questionnaire_data JSONB, -- AFM-required suitability questions and answers
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AFM Advice Sessions
CREATE TABLE IF NOT EXISTS afm_advice_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
    advisor_id UUID REFERENCES users(id),
    session_type VARCHAR(50) CHECK (session_type IN ('initial', 'follow_up', 'product_selection', 'review')),
    advice_category VARCHAR(50) CHECK (advice_category IN ('mortgage_advice', 'insurance_advice', 'investment_advice')),
    session_status VARCHAR(50) DEFAULT 'draft' CHECK (session_status IN ('draft', 'compliance_check', 'approved', 'delivered', 'archived')),
    advice_content TEXT,
    compliance_validated BOOLEAN DEFAULT false,
    afm_compliance_score DECIMAL(5,2),
    mandatory_disclosures_complete BOOLEAN DEFAULT false,
    client_understanding_confirmed BOOLEAN DEFAULT false,
    explanation_methods JSONB, -- how advice was explained to client (verbal, written, digital)
    session_recording_url VARCHAR(500), -- optional recording for compliance
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE
);

-- Dutch Mortgage Applications (enhanced AFM-compliant version)
CREATE TABLE IF NOT EXISTS dutch_mortgage_applications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
    advisor_id UUID REFERENCES users(id),
    advice_session_id UUID REFERENCES afm_advice_sessions(id),
    application_number VARCHAR(50) UNIQUE NOT NULL,
    lender_name VARCHAR(200) NOT NULL,
    product_id UUID REFERENCES dutch_mortgage_products(id),
    property_address TEXT NOT NULL,
    property_value DECIMAL(12,2) NOT NULL,
    mortgage_amount DECIMAL(12,2) NOT NULL,
    loan_to_value_ratio DECIMAL(5,2),
    debt_to_income_ratio DECIMAL(5,2),
    nhg_application BOOLEAN DEFAULT false,
    application_data JSONB NOT NULL, -- all application fields in structured format
    documents JSONB DEFAULT '[]', -- uploaded documents with metadata
    qc_score DECIMAL(5,2),
    qc_status VARCHAR(50) DEFAULT 'pending' CHECK (qc_status IN ('pending', 'passed', 'failed', 'review_required')),
    lender_validation_status VARCHAR(50) CHECK (lender_validation_status IN ('not_submitted', 'submitted', 'approved', 'rejected', 'conditionally_approved')),
    first_time_right BOOLEAN,
    processing_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP WITH TIME ZONE,
    lender_response_at TIMESTAMP WITH TIME ZONE
);

-- BKR Integration Log (Dutch credit bureau)
CREATE TABLE IF NOT EXISTS bkr_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID REFERENCES client_profiles(id) ON DELETE CASCADE,
    application_id UUID REFERENCES dutch_mortgage_applications(id) ON DELETE CASCADE,
    bkr_reference VARCHAR(100),
    check_type VARCHAR(50) CHECK (check_type IN ('credit_history', 'debt_verification', 'negative_registrations')),
    response_data JSONB, -- full API response from BKR
    credit_score INTEGER CHECK (credit_score >= 0 AND credit_score <= 1000),
    negative_registrations JSONB, -- any negative registrations found
    debt_summary JSONB, -- summary of debts registered with BKR
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AFM Compliance Audit Trail
CREATE TABLE IF NOT EXISTS afm_compliance_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES afm_advice_sessions(id) ON DELETE CASCADE,
    regulation_id UUID REFERENCES afm_regulations(id),
    compliance_check_type VARCHAR(100) CHECK (compliance_check_type IN ('disclosure_validation', 'suitability_check', 'documentation_check', 'advice_quality')),
    check_result VARCHAR(50) CHECK (check_result IN ('passed', 'failed', 'warning')),
    details JSONB, -- specific check results and findings
    remediation_required BOOLEAN DEFAULT false,
    remediation_actions JSONB, -- required actions to achieve compliance
    checked_by VARCHAR(50) CHECK (checked_by IN ('system', 'manual', 'supervisor')),
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for the new Dutch-specific tables
CREATE INDEX IF NOT EXISTS idx_afm_regulations_code ON afm_regulations(regulation_code);
CREATE INDEX IF NOT EXISTS idx_afm_regulations_type ON afm_regulations(regulation_type);
CREATE INDEX IF NOT EXISTS idx_afm_regulations_active ON afm_regulations(is_active) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_products_lender ON dutch_mortgage_products(lender_name);
CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_products_type ON dutch_mortgage_products(product_type);
CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_products_nhg ON dutch_mortgage_products(nhg_eligible) WHERE nhg_eligible = true;

CREATE INDEX IF NOT EXISTS idx_client_profiles_advisor ON client_profiles(advisor_id);
CREATE INDEX IF NOT EXISTS idx_client_profiles_bsn ON client_profiles(bsn);
CREATE INDEX IF NOT EXISTS idx_client_profiles_questionnaire ON client_profiles(afm_questionnaire_completed) WHERE afm_questionnaire_completed = true;

CREATE INDEX IF NOT EXISTS idx_afm_advice_sessions_client ON afm_advice_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_afm_advice_sessions_advisor ON afm_advice_sessions(advisor_id);
CREATE INDEX IF NOT EXISTS idx_afm_advice_sessions_status ON afm_advice_sessions(session_status);
CREATE INDEX IF NOT EXISTS idx_afm_advice_sessions_started ON afm_advice_sessions(started_at);

CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_applications_client ON dutch_mortgage_applications(client_id);
CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_applications_advisor ON dutch_mortgage_applications(advisor_id);
CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_applications_number ON dutch_mortgage_applications(application_number);
CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_applications_qc_status ON dutch_mortgage_applications(qc_status);
CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_applications_lender_status ON dutch_mortgage_applications(lender_validation_status);

CREATE INDEX IF NOT EXISTS idx_bkr_checks_client ON bkr_checks(client_id);
CREATE INDEX IF NOT EXISTS idx_bkr_checks_application ON bkr_checks(application_id);
CREATE INDEX IF NOT EXISTS idx_bkr_checks_type ON bkr_checks(check_type);
CREATE INDEX IF NOT EXISTS idx_bkr_checks_checked_at ON bkr_checks(checked_at);

CREATE INDEX IF NOT EXISTS idx_afm_compliance_logs_session ON afm_compliance_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_logs_regulation ON afm_compliance_logs(regulation_id);

-- =============================================================================
-- PHASE 2: ADVANCED AGENT TABLES
-- =============================================================================

-- AFM Regulation Version Tracking
CREATE TABLE IF NOT EXISTS afm_regulation_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_number VARCHAR(50) UNIQUE NOT NULL,
    release_date DATE NOT NULL,
    changes_summary TEXT,
    affected_articles TEXT[], -- Array of affected regulation codes
    is_current BOOLEAN DEFAULT false,
    requires_advisor_training BOOLEAN DEFAULT false,
    impact_level VARCHAR(20) CHECK (impact_level IN ('low', 'medium', 'high', 'critical')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Advice Generation Audit Trail
CREATE TABLE IF NOT EXISTS advice_generation_audit (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID,
    advice_content_hash VARCHAR(32) NOT NULL, -- SHA256 hash truncated to 32 chars
    products_recommended JSONB, -- Array of recommended product IDs
    afm_compliant BOOLEAN DEFAULT true,
    compliance_score DECIMAL(5,2),
    readability_score DECIMAL(5,2),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    agent_version VARCHAR(20),
    processing_time_ms INTEGER,
    session_metadata JSONB -- Additional processing metadata
);

-- Dutch Mortgage QC Results
CREATE TABLE IF NOT EXISTS dutch_mortgage_qc_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    application_id VARCHAR(100) NOT NULL,
    qc_report JSONB NOT NULL, -- Complete QC analysis results
    ftr_probability DECIMAL(5,2), -- First-time-right probability
    overall_score DECIMAL(5,2), -- Overall QC score
    ready_for_submission BOOLEAN DEFAULT false,
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    qc_agent_version VARCHAR(20),
    analysis_metadata JSONB -- Analysis configuration and versions
);

-- Indexes for new Phase 2 tables
CREATE INDEX IF NOT EXISTS idx_afm_regulation_versions_current ON afm_regulation_versions(is_current) WHERE is_current = true;
CREATE INDEX IF NOT EXISTS idx_afm_regulation_versions_release ON afm_regulation_versions(release_date DESC);

CREATE INDEX IF NOT EXISTS idx_advice_generation_audit_client ON advice_generation_audit(client_id);
CREATE INDEX IF NOT EXISTS idx_advice_generation_audit_hash ON advice_generation_audit(advice_content_hash);
CREATE INDEX IF NOT EXISTS idx_advice_generation_audit_generated ON advice_generation_audit(generated_at DESC);

CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_qc_results_app ON dutch_mortgage_qc_results(application_id);
CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_qc_results_ftr ON dutch_mortgage_qc_results(ftr_probability);
CREATE INDEX IF NOT EXISTS idx_dutch_mortgage_qc_results_analyzed ON dutch_mortgage_qc_results(analyzed_at DESC);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_logs_result ON afm_compliance_logs(check_result);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_logs_checked_at ON afm_compliance_logs(checked_at);

-- Full-text search indexes for Dutch content
CREATE INDEX IF NOT EXISTS idx_afm_regulations_content_nl_fts ON afm_regulations USING gin(to_tsvector('dutch', content_nl));
CREATE INDEX IF NOT EXISTS idx_afm_regulations_content_en_fts ON afm_regulations USING gin(to_tsvector('english', content_en));

-- Triggers for updated_at timestamps on new tables
CREATE TRIGGER update_afm_regulations_last_updated BEFORE UPDATE ON afm_regulations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- AI MORTGAGE ADVISOR CHAT TABLES
-- =====================================================

-- Chat conversations table for AI mortgage advisory sessions
CREATE TABLE IF NOT EXISTS chat_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    client_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'disconnected', 'archived')),
    conversation_type VARCHAR(50) DEFAULT 'mortgage_advisory' CHECK (conversation_type IN ('mortgage_advisory', 'compliance_check', 'document_review', 'general_inquiry')),
    context_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages table for storing conversation history
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES chat_conversations(id) ON DELETE CASCADE,
    sender VARCHAR(50) NOT NULL CHECK (sender IN ('user', 'ai', 'system')),
    message_type VARCHAR(50) NOT NULL CHECK (message_type IN ('user_message', 'ai_message', 'system_message', 'compliance_alert', 'typing_indicator', 'context_update', 'feedback')),
    content JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat feedback table for user ratings and feedback on AI responses
CREATE TABLE IF NOT EXISTS chat_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES chat_conversations(id) ON DELETE CASCADE,
    message_id UUID,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    feedback_type VARCHAR(50) DEFAULT 'general' CHECK (feedback_type IN ('general', 'message_rating', 'conversation_rating', 'feature_request', 'bug_report')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat analytics table for conversation insights and performance metrics
CREATE TABLE IF NOT EXISTS chat_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES chat_conversations(id) ON DELETE CASCADE,
    session_duration_seconds INTEGER,
    message_count INTEGER DEFAULT 0,
    ai_response_count INTEGER DEFAULT 0,
    user_satisfaction_score DECIMAL(3,2),
    topics_discussed TEXT[],
    compliance_alerts_count INTEGER DEFAULT 0,
    escalation_required BOOLEAN DEFAULT false,
    resolution_status VARCHAR(50) DEFAULT 'pending' CHECK (resolution_status IN ('pending', 'resolved', 'escalated', 'abandoned')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chat knowledge base table for AI learning and improvement
CREATE TABLE IF NOT EXISTS chat_knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topic VARCHAR(255) NOT NULL,
    question_pattern TEXT NOT NULL,
    answer_template TEXT NOT NULL,
    confidence_score DECIMAL(3,2) DEFAULT 0.80,
    usage_count INTEGER DEFAULT 0,
    success_rate DECIMAL(3,2) DEFAULT 0.00,
    language VARCHAR(10) DEFAULT 'nl' CHECK (language IN ('nl', 'en')),
    category VARCHAR(100) DEFAULT 'general',
    tags TEXT[],
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for chat tables performance optimization
CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_id ON chat_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_client_id ON chat_conversations(client_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_status ON chat_conversations(status);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_type ON chat_conversations(conversation_type);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_created_at ON chat_conversations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_sender ON chat_messages(sender);
CREATE INDEX IF NOT EXISTS idx_chat_messages_type ON chat_messages(message_type);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_feedback_conversation_id ON chat_feedback(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_feedback_message_id ON chat_feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_chat_feedback_rating ON chat_feedback(rating);
CREATE INDEX IF NOT EXISTS idx_chat_feedback_type ON chat_feedback(feedback_type);

CREATE INDEX IF NOT EXISTS idx_chat_analytics_conversation_id ON chat_analytics(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_analytics_satisfaction ON chat_analytics(user_satisfaction_score);
CREATE INDEX IF NOT EXISTS idx_chat_analytics_resolution ON chat_analytics(resolution_status);

CREATE INDEX IF NOT EXISTS idx_chat_knowledge_base_topic ON chat_knowledge_base(topic);
CREATE INDEX IF NOT EXISTS idx_chat_knowledge_base_category ON chat_knowledge_base(category);
CREATE INDEX IF NOT EXISTS idx_chat_knowledge_base_language ON chat_knowledge_base(language);
CREATE INDEX IF NOT EXISTS idx_chat_knowledge_base_active ON chat_knowledge_base(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_chat_knowledge_base_confidence ON chat_knowledge_base(confidence_score DESC);

-- Full-text search indexes for chat content
CREATE INDEX IF NOT EXISTS idx_chat_messages_content_fts ON chat_messages USING gin(to_tsvector('dutch', content::text));
CREATE INDEX IF NOT EXISTS idx_chat_knowledge_base_question_fts ON chat_knowledge_base USING gin(to_tsvector('dutch', question_pattern));
CREATE INDEX IF NOT EXISTS idx_chat_knowledge_base_answer_fts ON chat_knowledge_base USING gin(to_tsvector('dutch', answer_template));

-- Triggers for updated_at timestamps on chat tables
CREATE TRIGGER update_chat_conversations_updated_at BEFORE UPDATE ON chat_conversations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_chat_knowledge_base_updated_at BEFORE UPDATE ON chat_knowledge_base FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMPUTER VISION DOCUMENT VERIFICATION TABLES
-- ============================================================================
-- These tables support the advanced computer vision document verification system
-- with forgery detection, signature analysis, tampering detection, and authenticity scoring

-- Computer Vision Verification Results
CREATE TABLE IF NOT EXISTS cv_verification_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Document identification
    document_hash VARCHAR(64) NOT NULL, -- SHA-256 hash of document
    document_path VARCHAR(1000), -- Path to original document (may be cleaned up)
    original_filename VARCHAR(255),
    file_size BIGINT,
    mime_type VARCHAR(100),
    
    -- Verification results
    verification_status VARCHAR(20) NOT NULL CHECK (verification_status IN ('authentic', 'suspicious', 'fraudulent', 'inconclusive', 'error')),
    overall_confidence DECIMAL(5,4) NOT NULL CHECK (overall_confidence BETWEEN 0 AND 1),
    forgery_probability DECIMAL(5,4) NOT NULL CHECK (forgery_probability BETWEEN 0 AND 1),
    signature_authenticity DECIMAL(5,4) NOT NULL CHECK (signature_authenticity BETWEEN 0 AND 1),
    
    -- Detailed analysis results
    tampering_evidence JSONB DEFAULT '[]', -- Array of tampering evidence objects
    metadata_analysis JSONB DEFAULT '{}', -- Document metadata analysis results
    image_forensics JSONB DEFAULT '{}', -- Image forensics analysis results
    signature_analysis JSONB DEFAULT '{}', -- Signature analysis details
    
    -- Blockchain and audit
    blockchain_hash VARCHAR(64), -- Hash for blockchain audit trail
    blockchain_transaction_id VARCHAR(100), -- Transaction ID if using blockchain
    
    -- Processing information
    verification_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    processing_time DECIMAL(8,4) NOT NULL, -- Processing time in seconds
    ai_model_version VARCHAR(50) DEFAULT 'v1.0',
    verification_method VARCHAR(100) DEFAULT 'cv_multimodal', -- Method used for verification
    
    -- System metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Computer Vision Verification Batch Jobs
CREATE TABLE IF NOT EXISTS cv_verification_batches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Batch information
    batch_name VARCHAR(255),
    total_documents INTEGER NOT NULL,
    completed_documents INTEGER DEFAULT 0,
    failed_documents INTEGER DEFAULT 0,
    
    -- Batch status and progress
    batch_status VARCHAR(20) DEFAULT 'pending' CHECK (batch_status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    progress_percentage DECIMAL(5,2) DEFAULT 0.00,
    
    -- Processing metrics
    total_processing_time DECIMAL(8,4) DEFAULT 0, -- Total processing time in seconds
    average_processing_time DECIMAL(8,4) DEFAULT 0, -- Average per document
    average_confidence DECIMAL(5,4) DEFAULT 0, -- Average confidence score
    
    -- Batch results summary
    status_distribution JSONB DEFAULT '{}', -- Count of each verification status
    error_summary JSONB DEFAULT '[]', -- Summary of errors encountered
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Reference Signatures for Verification
CREATE TABLE IF NOT EXISTS cv_reference_signatures (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Signature identification
    signature_name VARCHAR(255) NOT NULL,
    signature_description TEXT,
    signature_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash of signature image
    
    -- File information
    file_path VARCHAR(1000) NOT NULL,
    original_filename VARCHAR(255),
    file_size BIGINT,
    mime_type VARCHAR(100),
    
    -- Signature features (extracted and stored for quick comparison)
    signature_features JSONB, -- Extracted signature features for comparison
    feature_version VARCHAR(20) DEFAULT 'v1.0',
    
    -- Usage tracking
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Computer Vision System Configuration
CREATE TABLE IF NOT EXISTS cv_system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Configuration parameters
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    config_type VARCHAR(20) DEFAULT 'string' CHECK (config_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    config_description TEXT,
    
    -- Configuration metadata
    is_sensitive BOOLEAN DEFAULT false, -- Whether config contains sensitive data
    requires_restart BOOLEAN DEFAULT false, -- Whether changing this requires system restart
    config_category VARCHAR(50) DEFAULT 'general', -- Category for grouping configs
    
    -- Versioning
    config_version INTEGER DEFAULT 1,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Computer Vision Audit Log
CREATE TABLE IF NOT EXISTS cv_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Event information
    event_type VARCHAR(50) NOT NULL, -- verification_started, verification_completed, error_occurred, etc.
    event_description TEXT,
    event_category VARCHAR(50) DEFAULT 'verification',
    
    -- Related entities
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    verification_id UUID REFERENCES cv_verification_results(id) ON DELETE CASCADE,
    batch_id UUID REFERENCES cv_verification_batches(id) ON DELETE SET NULL,
    
    -- Event data
    event_data JSONB DEFAULT '{}',
    system_metrics JSONB DEFAULT '{}', -- CPU, memory, etc. at time of event
    
    -- Security and compliance
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(100),
    
    -- Timestamps
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Computer Vision Performance Metrics
CREATE TABLE IF NOT EXISTS cv_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Metric identification
    metric_name VARCHAR(100) NOT NULL,
    metric_category VARCHAR(50) DEFAULT 'performance',
    
    -- Metric values
    metric_value DECIMAL(12,4) NOT NULL,
    metric_unit VARCHAR(20), -- seconds, bytes, percentage, etc.
    
    -- Context
    verification_id UUID REFERENCES cv_verification_results(id) ON DELETE CASCADE,
    batch_id UUID REFERENCES cv_verification_batches(id) ON DELETE CASCADE,
    
    -- Additional metadata
    metric_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR COMPUTER VISION VERIFICATION TABLES
-- ============================================================================

-- Primary lookup indexes
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_user_id ON cv_verification_results(user_id);
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_document_hash ON cv_verification_results(document_hash);
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_status ON cv_verification_results(verification_status);
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_timestamp ON cv_verification_results(verification_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_confidence ON cv_verification_results(overall_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_created_at ON cv_verification_results(created_at DESC);

-- Batch processing indexes
CREATE INDEX IF NOT EXISTS idx_cv_verification_batches_user_id ON cv_verification_batches(user_id);
CREATE INDEX IF NOT EXISTS idx_cv_verification_batches_status ON cv_verification_batches(batch_status);
CREATE INDEX IF NOT EXISTS idx_cv_verification_batches_created_at ON cv_verification_batches(created_at DESC);

-- Reference signatures indexes
CREATE INDEX IF NOT EXISTS idx_cv_reference_signatures_user_id ON cv_reference_signatures(user_id);
CREATE INDEX IF NOT EXISTS idx_cv_reference_signatures_hash ON cv_reference_signatures(signature_hash);
CREATE INDEX IF NOT EXISTS idx_cv_reference_signatures_active ON cv_reference_signatures(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_cv_reference_signatures_usage ON cv_reference_signatures(usage_count DESC);

-- System configuration indexes
CREATE INDEX IF NOT EXISTS idx_cv_system_config_key ON cv_system_config(config_key);
CREATE INDEX IF NOT EXISTS idx_cv_system_config_category ON cv_system_config(config_category);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_cv_audit_log_event_type ON cv_audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_cv_audit_log_user_id ON cv_audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_cv_audit_log_verification_id ON cv_audit_log(verification_id);
CREATE INDEX IF NOT EXISTS idx_cv_audit_log_timestamp ON cv_audit_log(event_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_cv_audit_log_category ON cv_audit_log(event_category);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_cv_performance_metrics_name ON cv_performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_cv_performance_metrics_category ON cv_performance_metrics(metric_category);
CREATE INDEX IF NOT EXISTS idx_cv_performance_metrics_verification_id ON cv_performance_metrics(verification_id);
CREATE INDEX IF NOT EXISTS idx_cv_performance_metrics_recorded_at ON cv_performance_metrics(recorded_at DESC);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_user_status_date ON cv_verification_results(user_id, verification_status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_status_confidence ON cv_verification_results(verification_status, overall_confidence DESC);

-- Full-text search indexes for CV verification
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_filename_fts ON cv_verification_results USING gin(to_tsvector('english', original_filename));

-- JSONB indexes for efficient querying of complex data
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_tampering_evidence_gin ON cv_verification_results USING gin(tampering_evidence);
CREATE INDEX IF NOT EXISTS idx_cv_verification_results_metadata_analysis_gin ON cv_verification_results USING gin(metadata_analysis);

-- ============================================================================
-- TRIGGERS FOR COMPUTER VISION VERIFICATION TABLES
-- ============================================================================

-- Update timestamps
CREATE TRIGGER update_cv_verification_results_updated_at 
    BEFORE UPDATE ON cv_verification_results 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cv_verification_batches_updated_at 
    BEFORE UPDATE ON cv_verification_batches 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cv_reference_signatures_updated_at 
    BEFORE UPDATE ON cv_reference_signatures 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_cv_system_config_updated_at 
    BEFORE UPDATE ON cv_system_config 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- DEFAULT SYSTEM CONFIGURATION FOR COMPUTER VISION VERIFICATION
-- ============================================================================

-- Insert default configuration values
INSERT INTO cv_system_config (config_key, config_value, config_type, config_description, config_category) VALUES
    ('cv_max_file_size', '52428800', 'integer', 'Maximum file size for CV verification in bytes (50MB)', 'file_processing'),
    ('cv_supported_formats', '["jpg", "jpeg", "png", "tiff", "tif", "bmp", "pdf"]', 'json', 'Supported file formats for CV verification', 'file_processing'),
    ('cv_confidence_threshold_authentic', '0.8', 'float', 'Minimum confidence score to mark document as authentic', 'thresholds'),
    ('cv_confidence_threshold_suspicious', '0.6', 'float', 'Minimum confidence score to mark document as suspicious', 'thresholds'),
    ('cv_forgery_threshold_high', '0.7', 'float', 'Forgery probability threshold for high risk classification', 'thresholds'),
    ('cv_forgery_threshold_medium', '0.4', 'float', 'Forgery probability threshold for medium risk classification', 'thresholds'),
    ('cv_signature_match_threshold', '0.75', 'float', 'Minimum similarity score for signature matching', 'signatures'),
    ('cv_enable_blockchain_logging', 'false', 'boolean', 'Enable blockchain logging for audit trails', 'security'),
    ('cv_gpu_acceleration', 'true', 'boolean', 'Enable GPU acceleration for CV processing', 'performance'),
    ('cv_batch_size_limit', '10', 'integer', 'Maximum number of documents in a single batch', 'performance'),
    ('cv_parallel_processing_limit', '4', 'integer', 'Maximum number of parallel CV processing jobs', 'performance'),
    ('cv_model_version', 'v1.0', 'string', 'Current AI model version for CV verification', 'models'),
    ('cv_enable_detailed_logging', 'true', 'boolean', 'Enable detailed logging for CV operations', 'logging'),
    ('cv_cleanup_temp_files', 'true', 'boolean', 'Automatically cleanup temporary files after processing', 'cleanup'),
    ('cv_temp_file_retention_hours', '24', 'integer', 'Hours to retain temporary files before cleanup', 'cleanup')
ON CONFLICT (config_key) DO NOTHING;

-- ============================================================================
-- COMPLIANCE NETWORK GRAPH VISUALIZATION TABLES
-- ============================================================================
-- These tables support the compliance network graph visualization system
-- with risk propagation analysis, relationship mapping, and regulatory impact assessment

-- Network Nodes - represents entities in the compliance network
CREATE TABLE IF NOT EXISTS compliance_network_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Node identification
    node_id VARCHAR(255) NOT NULL, -- Unique identifier within the network
    node_type VARCHAR(50) NOT NULL CHECK (node_type IN ('client', 'advisor', 'regulation', 'mortgage_product', 'lender', 'compliance_rule', 'risk_factor', 'document', 'process', 'audit_event')),
    label VARCHAR(255) NOT NULL,
    
    -- Node properties and metrics
    risk_score DECIMAL(5,4) DEFAULT 0.0 CHECK (risk_score BETWEEN 0 AND 1),
    compliance_status VARCHAR(50) DEFAULT 'unknown',
    
    -- Node positioning and visualization
    x_coordinate DECIMAL(10,6),
    y_coordinate DECIMAL(10,6),
    community_id VARCHAR(100),
    importance_rank INTEGER DEFAULT 0,
    
    -- Centrality measures
    degree_centrality DECIMAL(8,6) DEFAULT 0,
    betweenness_centrality DECIMAL(8,6) DEFAULT 0,
    closeness_centrality DECIMAL(8,6) DEFAULT 0,
    eigenvector_centrality DECIMAL(8,6) DEFAULT 0,
    pagerank_centrality DECIMAL(8,6) DEFAULT 0,
    
    -- Extended properties and metadata
    properties JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Network association
    network_id UUID, -- Reference to the network this node belongs to
    
    -- Temporal tracking
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique node_id within each network
    UNIQUE(network_id, node_id)
);

-- Network Edges - represents relationships in the compliance network
CREATE TABLE IF NOT EXISTS compliance_network_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Edge identification
    edge_id VARCHAR(255) NOT NULL,
    source_node_id VARCHAR(255) NOT NULL,
    target_node_id VARCHAR(255) NOT NULL,
    
    -- Edge properties
    edge_type VARCHAR(50) NOT NULL CHECK (edge_type IN ('advises', 'applies_to', 'complies_with', 'violates', 'depends_on', 'influences', 'requires', 'produces', 'validates', 'triggers', 'escalates')),
    weight DECIMAL(8,4) DEFAULT 1.0 CHECK (weight >= 0),
    confidence DECIMAL(5,4) DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    risk_contribution DECIMAL(5,4) DEFAULT 0.0 CHECK (risk_contribution BETWEEN 0 AND 1),
    
    -- Temporal validity
    validity_start TIMESTAMP WITH TIME ZONE,
    validity_end TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    
    -- Extended properties and metadata
    properties JSONB DEFAULT '{}',
    
    -- Network association
    network_id UUID, -- Reference to the network this edge belongs to
    
    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique edge_id within each network
    UNIQUE(network_id, edge_id)
);

-- Network Definitions - represents complete compliance networks
CREATE TABLE IF NOT EXISTS compliance_networks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Network identification
    network_name VARCHAR(255) NOT NULL,
    network_description TEXT,
    network_type VARCHAR(50) DEFAULT 'compliance' CHECK (network_type IN ('compliance', 'risk_assessment', 'regulatory_impact', 'simulation')),
    
    -- Network metadata
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    density DECIMAL(8,6) DEFAULT 0,
    is_connected BOOLEAN DEFAULT false,
    clustering_coefficient DECIMAL(8,6) DEFAULT 0,
    
    -- Analysis status
    last_analysis_at TIMESTAMP WITH TIME ZONE,
    analysis_status VARCHAR(20) DEFAULT 'pending' CHECK (analysis_status IN ('pending', 'analyzing', 'completed', 'failed')),
    analysis_results JSONB DEFAULT '{}',
    
    -- Network configuration
    layout_algorithm VARCHAR(50) DEFAULT 'spring',
    visualization_config JSONB DEFAULT '{}',
    
    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Risk Propagation Results - stores results of risk propagation analysis
CREATE TABLE IF NOT EXISTS compliance_risk_propagation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    network_id UUID REFERENCES compliance_networks(id) ON DELETE CASCADE,
    
    -- Propagation identification
    propagation_id VARCHAR(255) NOT NULL UNIQUE,
    propagation_type VARCHAR(50) DEFAULT 'linear' CHECK (propagation_type IN ('linear', 'exponential', 'logarithmic', 'threshold', 'cascade')),
    
    -- Source configuration
    source_nodes JSONB NOT NULL, -- Array of source node IDs
    
    -- Propagation parameters
    max_propagation_steps INTEGER DEFAULT 10,
    convergence_tolerance DECIMAL(8,6) DEFAULT 0.001,
    
    -- Results
    affected_nodes JSONB DEFAULT '{}', -- node_id -> risk_increase mapping
    propagation_paths JSONB DEFAULT '[]', -- Array of propagation paths
    total_risk_increase DECIMAL(8,4) DEFAULT 0,
    propagation_time_steps INTEGER DEFAULT 0,
    convergence_achieved BOOLEAN DEFAULT false,
    
    -- Critical paths and recommendations
    critical_paths JSONB DEFAULT '[]',
    mitigation_recommendations JSONB DEFAULT '[]',
    
    -- Processing information
    processing_time DECIMAL(8,4), -- Processing time in seconds
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Network Analysis Sessions - tracks comprehensive network analysis sessions
CREATE TABLE IF NOT EXISTS compliance_network_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    network_id UUID REFERENCES compliance_networks(id) ON DELETE CASCADE,
    
    -- Analysis identification
    analysis_id VARCHAR(255) NOT NULL UNIQUE,
    analysis_type VARCHAR(50) DEFAULT 'comprehensive' CHECK (analysis_type IN ('comprehensive', 'centrality', 'communities', 'anomalies', 'risk_propagation', 'regulatory_impact')),
    
    -- Analysis results
    network_statistics JSONB DEFAULT '{}',
    centrality_analysis JSONB DEFAULT '{}',
    community_structure JSONB DEFAULT '{}',
    risk_assessment JSONB DEFAULT '{}',
    anomaly_detection JSONB DEFAULT '{}',
    
    -- Recommendations and insights
    recommendations JSONB DEFAULT '[]',
    insights JSONB DEFAULT '{}',
    
    -- Visualization data
    visualization_data JSONB DEFAULT '{}',
    layout_used VARCHAR(50),
    
    -- Analysis configuration
    analysis_options JSONB DEFAULT '{}',
    
    -- Performance metrics
    processing_time DECIMAL(8,4), -- Processing time in seconds
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Regulatory Impact Simulations - stores regulatory change impact analysis
CREATE TABLE IF NOT EXISTS compliance_regulatory_simulations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    network_id UUID REFERENCES compliance_networks(id) ON DELETE CASCADE,
    
    -- Simulation identification
    simulation_id VARCHAR(255) NOT NULL UNIQUE,
    simulation_name VARCHAR(255),
    simulation_description TEXT,
    
    -- Baseline network state
    original_network_stats JSONB DEFAULT '{}',
    
    -- Applied changes
    regulatory_changes JSONB NOT NULL,
    changes_applied JSONB DEFAULT '{}',
    
    -- Impact analysis results
    impact_analysis JSONB DEFAULT '{}',
    affected_entities JSONB DEFAULT '{}', -- Entities impacted by the changes
    risk_impact_summary JSONB DEFAULT '{}',
    
    -- Recommendations
    recommendations JSONB DEFAULT '[]',
    mitigation_strategies JSONB DEFAULT '[]',
    
    -- Simulation configuration
    simulation_options JSONB DEFAULT '{}',
    
    -- Performance metrics
    processing_time DECIMAL(8,4), -- Processing time in seconds
    simulation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Temporal tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Network Anomalies - stores detected anomalies in compliance networks
CREATE TABLE IF NOT EXISTS compliance_network_anomalies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    network_id UUID REFERENCES compliance_networks(id) ON DELETE CASCADE,
    
    -- Anomaly identification
    anomaly_type VARCHAR(50) NOT NULL CHECK (anomaly_type IN ('node_anomaly', 'edge_anomaly', 'structural_anomaly', 'temporal_anomaly', 'risk_anomaly')),
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    
    -- Anomaly details
    anomaly_description TEXT NOT NULL,
    affected_entities JSONB DEFAULT '{}', -- Nodes, edges, or other entities affected
    anomaly_score DECIMAL(5,4), -- Anomaly score if applicable
    
    -- Detection information
    detection_method VARCHAR(100),
    detection_confidence DECIMAL(5,4) DEFAULT 0,
    
    -- Resolution status
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'false_positive')),
    resolution_notes TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}',
    
    -- Temporal tracking
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Network Performance Metrics - stores performance metrics for network operations
CREATE TABLE IF NOT EXISTS compliance_network_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    network_id UUID REFERENCES compliance_networks(id) ON DELETE CASCADE,
    
    -- Metric identification
    metric_name VARCHAR(100) NOT NULL,
    metric_category VARCHAR(50) DEFAULT 'performance',
    metric_type VARCHAR(20) DEFAULT 'gauge' CHECK (metric_type IN ('counter', 'gauge', 'histogram', 'summary')),
    
    -- Metric values
    metric_value DECIMAL(12,4) NOT NULL,
    metric_unit VARCHAR(20), -- seconds, bytes, count, percentage, etc.
    
    -- Context
    operation_type VARCHAR(50), -- analysis, visualization, risk_propagation, etc.
    operation_id VARCHAR(255), -- Reference to specific operation
    
    -- Additional metadata
    metric_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR COMPLIANCE NETWORK GRAPH TABLES
-- ============================================================================

-- Primary lookup indexes for nodes
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_user_id ON compliance_network_nodes(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_network_id ON compliance_network_nodes(network_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_node_type ON compliance_network_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_risk_score ON compliance_network_nodes(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_compliance_status ON compliance_network_nodes(compliance_status);
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_community ON compliance_network_nodes(community_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_importance ON compliance_network_nodes(importance_rank);
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_created_at ON compliance_network_nodes(created_at DESC);

-- Primary lookup indexes for edges
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_user_id ON compliance_network_edges(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_network_id ON compliance_network_edges(network_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_source ON compliance_network_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_target ON compliance_network_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_type ON compliance_network_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_active ON compliance_network_edges(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_risk_contrib ON compliance_network_edges(risk_contribution DESC);

-- Network definition indexes
CREATE INDEX IF NOT EXISTS idx_compliance_networks_user_id ON compliance_networks(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_networks_type ON compliance_networks(network_type);
CREATE INDEX IF NOT EXISTS idx_compliance_networks_analysis_status ON compliance_networks(analysis_status);
CREATE INDEX IF NOT EXISTS idx_compliance_networks_created_at ON compliance_networks(created_at DESC);

-- Risk propagation indexes
CREATE INDEX IF NOT EXISTS idx_compliance_risk_propagation_user_id ON compliance_risk_propagation(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_risk_propagation_network_id ON compliance_risk_propagation(network_id);
CREATE INDEX IF NOT EXISTS idx_compliance_risk_propagation_type ON compliance_risk_propagation(propagation_type);
CREATE INDEX IF NOT EXISTS idx_compliance_risk_propagation_timestamp ON compliance_risk_propagation(analysis_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_compliance_risk_propagation_convergence ON compliance_risk_propagation(convergence_achieved);

-- Network analysis indexes
CREATE INDEX IF NOT EXISTS idx_compliance_network_analysis_user_id ON compliance_network_analysis(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_analysis_network_id ON compliance_network_analysis(network_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_analysis_type ON compliance_network_analysis(analysis_type);
CREATE INDEX IF NOT EXISTS idx_compliance_network_analysis_timestamp ON compliance_network_analysis(analysis_timestamp DESC);

-- Regulatory simulation indexes
CREATE INDEX IF NOT EXISTS idx_compliance_regulatory_simulations_user_id ON compliance_regulatory_simulations(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_regulatory_simulations_network_id ON compliance_regulatory_simulations(network_id);
CREATE INDEX IF NOT EXISTS idx_compliance_regulatory_simulations_timestamp ON compliance_regulatory_simulations(simulation_timestamp DESC);

-- Anomaly detection indexes
CREATE INDEX IF NOT EXISTS idx_compliance_network_anomalies_user_id ON compliance_network_anomalies(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_anomalies_network_id ON compliance_network_anomalies(network_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_anomalies_type ON compliance_network_anomalies(anomaly_type);
CREATE INDEX IF NOT EXISTS idx_compliance_network_anomalies_severity ON compliance_network_anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_compliance_network_anomalies_status ON compliance_network_anomalies(status);
CREATE INDEX IF NOT EXISTS idx_compliance_network_anomalies_detected_at ON compliance_network_anomalies(detected_at DESC);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_compliance_network_metrics_network_id ON compliance_network_metrics(network_id);
CREATE INDEX IF NOT EXISTS idx_compliance_network_metrics_name ON compliance_network_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_compliance_network_metrics_category ON compliance_network_metrics(metric_category);
CREATE INDEX IF NOT EXISTS idx_compliance_network_metrics_recorded_at ON compliance_network_metrics(recorded_at DESC);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_network_type_risk ON compliance_network_nodes(network_id, node_type, risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_network_active_type ON compliance_network_edges(network_id, is_active, edge_type);
CREATE INDEX IF NOT EXISTS idx_compliance_network_anomalies_network_status_severity ON compliance_network_anomalies(network_id, status, severity);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_label_fts ON compliance_network_nodes USING gin(to_tsvector('english', label));
CREATE INDEX IF NOT EXISTS idx_compliance_networks_name_description_fts ON compliance_networks USING gin(to_tsvector('english', network_name || ' ' || COALESCE(network_description, '')));

-- JSONB indexes for efficient querying of complex data
CREATE INDEX IF NOT EXISTS idx_compliance_network_nodes_properties_gin ON compliance_network_nodes USING gin(properties);
CREATE INDEX IF NOT EXISTS idx_compliance_network_edges_properties_gin ON compliance_network_edges USING gin(properties);
CREATE INDEX IF NOT EXISTS idx_compliance_risk_propagation_affected_nodes_gin ON compliance_risk_propagation USING gin(affected_nodes);
CREATE INDEX IF NOT EXISTS idx_compliance_network_analysis_recommendations_gin ON compliance_network_analysis USING gin(recommendations);
CREATE INDEX IF NOT EXISTS idx_compliance_regulatory_simulations_changes_gin ON compliance_regulatory_simulations USING gin(regulatory_changes);

-- ============================================================================
-- TRIGGERS FOR COMPLIANCE NETWORK GRAPH TABLES
-- ============================================================================

-- Update timestamps
CREATE TRIGGER update_compliance_network_nodes_updated_at
    BEFORE UPDATE ON compliance_network_nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_network_edges_updated_at
    BEFORE UPDATE ON compliance_network_edges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_networks_updated_at
    BEFORE UPDATE ON compliance_networks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_network_anomalies_updated_at
    BEFORE UPDATE ON compliance_network_anomalies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update network statistics when nodes or edges are modified
CREATE OR REPLACE FUNCTION update_network_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update node and edge counts
    UPDATE compliance_networks 
    SET 
        node_count = (
            SELECT COUNT(*) 
            FROM compliance_network_nodes 
            WHERE network_id = COALESCE(NEW.network_id, OLD.network_id)
        ),
        edge_count = (
            SELECT COUNT(*) 
            FROM compliance_network_edges 
            WHERE network_id = COALESCE(NEW.network_id, OLD.network_id) AND is_active = true
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = COALESCE(NEW.network_id, OLD.network_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_network_stats_on_nodes
    AFTER INSERT OR UPDATE OR DELETE ON compliance_network_nodes
    FOR EACH ROW EXECUTE FUNCTION update_network_statistics();

CREATE TRIGGER update_network_stats_on_edges
    AFTER INSERT OR UPDATE OR DELETE ON compliance_network_edges
    FOR EACH ROW EXECUTE FUNCTION update_network_statistics();

-- ============================================================================
-- AUTONOMOUS WORKFLOW MONITOR TABLES
-- ============================================================================
-- These tables support the autonomous workflow monitoring system with real-time
-- agent decision tracking, learning pattern analysis, and performance optimization

-- Monitoring Sessions - tracks active monitoring sessions
CREATE TABLE IF NOT EXISTS workflow_monitoring_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL UNIQUE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Session details
    session_name VARCHAR(255),
    monitoring_status VARCHAR(20) DEFAULT 'pending' CHECK (monitoring_status IN ('pending', 'active', 'stopped', 'paused', 'error')),
    
    -- Session configuration
    metrics_collection_interval INTEGER DEFAULT 60, -- seconds
    enable_real_time_analysis BOOLEAN DEFAULT true,
    enable_pattern_detection BOOLEAN DEFAULT true,
    enable_optimization_analysis BOOLEAN DEFAULT true,
    
    -- Alert thresholds
    alert_thresholds JSONB DEFAULT '{}',
    
    -- Session lifecycle
    started_at TIMESTAMP WITH TIME ZONE,
    stopped_at TIMESTAMP WITH TIME ZONE,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Session statistics
    total_decisions_logged INTEGER DEFAULT 0,
    total_metrics_collected INTEGER DEFAULT 0,
    total_patterns_detected INTEGER DEFAULT 0,
    total_alerts_triggered INTEGER DEFAULT 0,
    
    -- Configuration and metadata
    configuration JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Decisions - logs all agent decisions for analysis
CREATE TABLE IF NOT EXISTS agent_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_id VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    monitoring_session_id UUID REFERENCES workflow_monitoring_sessions(id) ON DELETE CASCADE,
    
    -- Agent and workflow identification
    agent_id VARCHAR(255) NOT NULL,
    workflow_id VARCHAR(255) DEFAULT 'default_workflow',
    
    -- Decision details
    decision_type VARCHAR(50) NOT NULL CHECK (decision_type IN ('classification', 'recommendation', 'validation', 'escalation', 'approval', 'rejection', 'routing', 'optimization', 'prediction', 'intervention')),
    
    -- Input and output data
    input_data JSONB NOT NULL,
    output_data JSONB NOT NULL,
    
    -- Decision quality metrics
    confidence_score DECIMAL(5,4) NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    correctness_score DECIMAL(5,4) CHECK (correctness_score BETWEEN 0 AND 1),
    user_feedback_score DECIMAL(5,4) CHECK (user_feedback_score BETWEEN 0 AND 1),
    downstream_impact_score DECIMAL(5,4) CHECK (downstream_impact_score BETWEEN 0 AND 1),
    
    -- Performance metrics
    processing_time DECIMAL(10,6) NOT NULL CHECK (processing_time >= 0),
    resource_usage JSONB DEFAULT '{}',
    
    -- Learning indicators
    model_version VARCHAR(50) DEFAULT '1.0',
    training_data_version VARCHAR(50) DEFAULT '1.0',
    feature_importance JSONB DEFAULT '{}',
    
    -- Context and metadata
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    error_details TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Decision timestamp
    decision_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Audit timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique decision_id within each monitoring session
    UNIQUE(monitoring_session_id, decision_id)
);

-- Workflow Executions - tracks complete workflow executions
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    monitoring_session_id UUID REFERENCES workflow_monitoring_sessions(id) ON DELETE CASCADE,
    
    -- Workflow identification
    workflow_id VARCHAR(255) NOT NULL,
    workflow_name VARCHAR(255) NOT NULL,
    
    -- Execution status
    execution_status VARCHAR(20) DEFAULT 'pending' CHECK (execution_status IN ('pending', 'running', 'completed', 'failed', 'paused', 'cancelled', 'retry', 'timeout')),
    
    -- Workflow structure
    steps JSONB NOT NULL, -- Array of step names
    dependencies JSONB DEFAULT '{}', -- Step dependencies
    parallel_branches JSONB DEFAULT '[]', -- Parallel execution branches
    
    -- Execution timeline
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    current_step VARCHAR(255),
    completed_steps JSONB DEFAULT '[]',
    failed_steps JSONB DEFAULT '[]',
    
    -- Performance metrics
    total_processing_time DECIMAL(10,6) DEFAULT 0,
    total_cost DECIMAL(10,4) DEFAULT 0,
    resource_usage JSONB DEFAULT '{}',
    
    -- Quality metrics
    overall_accuracy DECIMAL(5,4) DEFAULT 0 CHECK (overall_accuracy BETWEEN 0 AND 1),
    user_satisfaction DECIMAL(5,4) CHECK (user_satisfaction BETWEEN 0 AND 1),
    business_impact DECIMAL(5,4) CHECK (business_impact BETWEEN 0 AND 1),
    
    -- Input and output
    input_parameters JSONB DEFAULT '{}',
    output_results JSONB DEFAULT '{}',
    
    -- Error handling
    error_log JSONB DEFAULT '[]',
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique execution_id within each monitoring session
    UNIQUE(monitoring_session_id, execution_id)
);

-- Performance Metrics - stores individual performance measurements
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_id VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    monitoring_session_id UUID REFERENCES workflow_monitoring_sessions(id) ON DELETE CASCADE,
    
    -- Agent and context
    agent_id VARCHAR(255) NOT NULL,
    workflow_id VARCHAR(255),
    decision_id VARCHAR(255),
    
    -- Metric details
    metric_type VARCHAR(50) NOT NULL CHECK (metric_type IN ('accuracy', 'precision', 'recall', 'f1_score', 'processing_time', 'throughput', 'error_rate', 'resource_utilization', 'user_satisfaction', 'cost_per_decision')),
    metric_value DECIMAL(12,6) NOT NULL,
    metric_unit VARCHAR(20),
    
    -- Measurement context
    measurement_context JSONB DEFAULT '{}',
    measurement_confidence DECIMAL(5,4) DEFAULT 1.0 CHECK (measurement_confidence BETWEEN 0 AND 1),
    data_quality_score DECIMAL(5,4) DEFAULT 1.0 CHECK (data_quality_score BETWEEN 0 AND 1),
    
    -- Collection method
    collection_method VARCHAR(50) DEFAULT 'automated',
    
    -- Metric timestamp
    metric_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Learning Patterns - detected patterns in agent behavior and performance
CREATE TABLE IF NOT EXISTS learning_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id VARCHAR(255) NOT NULL UNIQUE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    monitoring_session_id UUID REFERENCES workflow_monitoring_sessions(id) ON DELETE CASCADE,
    
    -- Pattern identification
    pattern_type VARCHAR(50) NOT NULL CHECK (pattern_type IN ('improvement_trend', 'performance_plateau', 'degradation', 'seasonal_pattern', 'concept_drift', 'anomalous_behavior', 'adaptation', 'specialization')),
    agent_id VARCHAR(255) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    
    -- Pattern characteristics
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    statistical_significance DECIMAL(5,4) NOT NULL CHECK (statistical_significance BETWEEN 0 AND 1),
    
    -- Pattern details
    trend_direction VARCHAR(50), -- improving, declining, stable, cyclical, anomalous, specializing
    trend_magnitude DECIMAL(8,4),
    seasonality_period INTEGER,
    change_points JSONB DEFAULT '[]', -- Array of change point timestamps
    
    -- Supporting data
    data_points JSONB DEFAULT '[]', -- Array of [timestamp, value] pairs
    statistical_tests JSONB DEFAULT '{}',
    
    -- Insights and recommendations
    insights JSONB DEFAULT '[]', -- Array of insight strings
    recommendations JSONB DEFAULT '[]', -- Array of recommendation strings
    
    -- Detection metadata
    detection_method VARCHAR(100) DEFAULT 'statistical_analysis',
    detection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow Optimization Results - stores optimization analysis results
CREATE TABLE IF NOT EXISTS workflow_optimization_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    optimization_id VARCHAR(255) NOT NULL UNIQUE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    monitoring_session_id UUID REFERENCES workflow_monitoring_sessions(id) ON DELETE CASCADE,
    
    -- Optimization scope
    optimization_scope VARCHAR(50) DEFAULT 'comprehensive' CHECK (optimization_scope IN ('performance', 'cost', 'quality', 'comprehensive')),
    analyzed_workflows JSONB DEFAULT '[]', -- Array of workflow IDs analyzed
    
    -- Analysis results
    bottlenecks_detected JSONB DEFAULT '[]', -- Array of bottleneck objects
    optimization_opportunities JSONB DEFAULT '[]', -- Array of opportunity objects
    predicted_improvements JSONB DEFAULT '{}', -- Predicted improvement metrics
    
    -- Recommendations
    implementation_recommendations JSONB DEFAULT '[]', -- Array of recommendation objects
    priority_ranking JSONB DEFAULT '[]', -- Ordered list of recommendations by priority
    
    -- Analysis configuration
    include_bottleneck_analysis BOOLEAN DEFAULT true,
    include_resource_optimization BOOLEAN DEFAULT true,
    include_predictions BOOLEAN DEFAULT true,
    
    -- Analysis metadata
    analysis_duration DECIMAL(8,4), -- Analysis time in seconds
    confidence_level DECIMAL(5,4) DEFAULT 0.8,
    
    -- Timestamps
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance Alerts - tracks alerts triggered by monitoring system
CREATE TABLE IF NOT EXISTS performance_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(255) NOT NULL UNIQUE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    monitoring_session_id UUID REFERENCES workflow_monitoring_sessions(id) ON DELETE CASCADE,
    
    -- Alert details
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN ('threshold_violation', 'anomaly_detected', 'pattern_change', 'performance_degradation', 'system_error')),
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    
    -- Alert source
    agent_id VARCHAR(255),
    metric_type VARCHAR(50),
    workflow_id VARCHAR(255),
    
    -- Alert content
    alert_title VARCHAR(255) NOT NULL,
    alert_message TEXT NOT NULL,
    alert_data JSONB DEFAULT '{}',
    
    -- Threshold information (for threshold violations)
    threshold_value DECIMAL(12,6),
    actual_value DECIMAL(12,6),
    threshold_type VARCHAR(20), -- 'above', 'below', 'equal'
    
    -- Alert status
    alert_status VARCHAR(20) DEFAULT 'active' CHECK (alert_status IN ('active', 'acknowledged', 'resolved', 'suppressed')),
    acknowledged_by UUID REFERENCES users(id) ON DELETE SET NULL,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    
    -- Alert timestamps
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System Health Snapshots - periodic system health assessments
CREATE TABLE IF NOT EXISTS system_health_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    monitoring_session_id UUID REFERENCES workflow_monitoring_sessions(id) ON DELETE CASCADE,
    
    -- Health assessment
    overall_health_status VARCHAR(20) NOT NULL CHECK (overall_health_status IN ('excellent', 'good', 'fair', 'poor', 'critical', 'error')),
    overall_health_score DECIMAL(5,4) NOT NULL CHECK (overall_health_score BETWEEN 0 AND 1),
    factors_evaluated INTEGER DEFAULT 0,
    
    -- Component health scores
    agent_health_scores JSONB DEFAULT '{}', -- agent_id -> health_score mapping
    system_metrics JSONB DEFAULT '{}', -- System-level metrics
    performance_indicators JSONB DEFAULT '{}', -- Key performance indicators
    
    -- Health factors
    accuracy_factor DECIMAL(5,4),
    performance_factor DECIMAL(5,4),
    error_rate_factor DECIMAL(5,4),
    resource_utilization_factor DECIMAL(5,4),
    
    -- Recommendations
    health_recommendations JSONB DEFAULT '[]',
    action_items JSONB DEFAULT '[]',
    
    -- Snapshot timestamp
    snapshot_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow Monitor Reports - generated reports and exports
CREATE TABLE IF NOT EXISTS workflow_monitor_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_id VARCHAR(255) NOT NULL UNIQUE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    monitoring_session_id UUID REFERENCES workflow_monitoring_sessions(id) ON DELETE CASCADE,
    
    -- Report details
    report_type VARCHAR(50) NOT NULL CHECK (report_type IN ('learning_insights', 'performance_summary', 'optimization_report', 'comprehensive', 'custom')),
    report_name VARCHAR(255) NOT NULL,
    report_description TEXT,
    
    -- Report configuration
    time_period VARCHAR(20) DEFAULT '24h',
    include_visualizations BOOLEAN DEFAULT true,
    include_recommendations BOOLEAN DEFAULT true,
    export_format VARCHAR(20) DEFAULT 'json' CHECK (export_format IN ('json', 'pdf', 'html', 'csv', 'excel')),
    
    -- Report content
    report_data JSONB NOT NULL,
    report_summary JSONB DEFAULT '{}',
    
    -- File information (if exported)
    file_path VARCHAR(500),
    file_size BIGINT,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Report status
    report_status VARCHAR(20) DEFAULT 'generated' CHECK (report_status IN ('generating', 'generated', 'exported', 'expired', 'error')),
    generation_time DECIMAL(8,4), -- Generation time in seconds
    
    -- Access control
    is_public BOOLEAN DEFAULT false,
    access_permissions JSONB DEFAULT '{}',
    
    -- Timestamps
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR AUTONOMOUS WORKFLOW MONITOR TABLES
-- ============================================================================

-- Monitoring sessions indexes
CREATE INDEX IF NOT EXISTS idx_workflow_monitoring_sessions_session_id ON workflow_monitoring_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_monitoring_sessions_user_id ON workflow_monitoring_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_workflow_monitoring_sessions_status ON workflow_monitoring_sessions(monitoring_status);
CREATE INDEX IF NOT EXISTS idx_workflow_monitoring_sessions_created_at ON workflow_monitoring_sessions(created_at DESC);

-- Agent decisions indexes
CREATE INDEX IF NOT EXISTS idx_agent_decisions_session_id ON agent_decisions(monitoring_session_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent_id ON agent_decisions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_workflow_id ON agent_decisions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_decision_type ON agent_decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_timestamp ON agent_decisions(decision_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_confidence ON agent_decisions(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_processing_time ON agent_decisions(processing_time);

-- Workflow executions indexes
CREATE INDEX IF NOT EXISTS idx_workflow_executions_session_id ON workflow_executions(monitoring_session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(execution_status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_start_time ON workflow_executions(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_processing_time ON workflow_executions(total_processing_time);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_session_id ON performance_metrics(monitoring_session_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_agent_id ON performance_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(metric_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_workflow_id ON performance_metrics(workflow_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_decision_id ON performance_metrics(decision_id);

-- Learning patterns indexes
CREATE INDEX IF NOT EXISTS idx_learning_patterns_session_id ON learning_patterns(monitoring_session_id);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_agent_id ON learning_patterns(agent_id);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_type ON learning_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_confidence ON learning_patterns(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_detection_time ON learning_patterns(detection_timestamp DESC);

-- Optimization results indexes
CREATE INDEX IF NOT EXISTS idx_workflow_optimization_results_session_id ON workflow_optimization_results(monitoring_session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_optimization_results_scope ON workflow_optimization_results(optimization_scope);
CREATE INDEX IF NOT EXISTS idx_workflow_optimization_results_timestamp ON workflow_optimization_results(analysis_timestamp DESC);

-- Performance alerts indexes
CREATE INDEX IF NOT EXISTS idx_performance_alerts_session_id ON performance_alerts(monitoring_session_id);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_agent_id ON performance_alerts(agent_id);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_type ON performance_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_severity ON performance_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_status ON performance_alerts(alert_status);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_triggered_at ON performance_alerts(triggered_at DESC);

-- System health snapshots indexes
CREATE INDEX IF NOT EXISTS idx_system_health_snapshots_session_id ON system_health_snapshots(monitoring_session_id);
CREATE INDEX IF NOT EXISTS idx_system_health_snapshots_status ON system_health_snapshots(overall_health_status);
CREATE INDEX IF NOT EXISTS idx_system_health_snapshots_score ON system_health_snapshots(overall_health_score DESC);
CREATE INDEX IF NOT EXISTS idx_system_health_snapshots_timestamp ON system_health_snapshots(snapshot_timestamp DESC);

-- Reports indexes
CREATE INDEX IF NOT EXISTS idx_workflow_monitor_reports_session_id ON workflow_monitor_reports(monitoring_session_id);
CREATE INDEX IF NOT EXISTS idx_workflow_monitor_reports_user_id ON workflow_monitor_reports(user_id);
CREATE INDEX IF NOT EXISTS idx_workflow_monitor_reports_type ON workflow_monitor_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_workflow_monitor_reports_status ON workflow_monitor_reports(report_status);
CREATE INDEX IF NOT EXISTS idx_workflow_monitor_reports_generated_at ON workflow_monitor_reports(generated_at DESC);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent_timestamp ON agent_decisions(agent_id, decision_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_session_agent ON agent_decisions(monitoring_session_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_agent_type_timestamp ON performance_metrics(agent_id, metric_type, metric_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_agent_type ON learning_patterns(agent_id, pattern_type);
CREATE INDEX IF NOT EXISTS idx_performance_alerts_agent_status ON performance_alerts(agent_id, alert_status);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_agent_decisions_input_data_gin ON agent_decisions USING gin(input_data);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_output_data_gin ON agent_decisions USING gin(output_data);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_insights_gin ON learning_patterns USING gin(insights);
CREATE INDEX IF NOT EXISTS idx_workflow_optimization_results_recommendations_gin ON workflow_optimization_results USING gin(implementation_recommendations);

-- ============================================================================
-- TRIGGERS FOR AUTONOMOUS WORKFLOW MONITOR TABLES
-- ============================================================================

-- Update timestamps
CREATE TRIGGER update_workflow_monitoring_sessions_updated_at
    BEFORE UPDATE ON workflow_monitoring_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_decisions_updated_at
    BEFORE UPDATE ON agent_decisions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_executions_updated_at
    BEFORE UPDATE ON workflow_executions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_learning_patterns_updated_at
    BEFORE UPDATE ON learning_patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_performance_alerts_updated_at
    BEFORE UPDATE ON performance_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_monitor_reports_updated_at
    BEFORE UPDATE ON workflow_monitor_reports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update session statistics when decisions are added
CREATE OR REPLACE FUNCTION update_session_statistics()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Update monitoring session statistics
        UPDATE workflow_monitoring_sessions 
        SET 
            total_decisions_logged = total_decisions_logged + 1,
            last_activity_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = NEW.monitoring_session_id;
        
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        -- Decrease count on deletion (cleanup scenarios)
        UPDATE workflow_monitoring_sessions 
        SET 
            total_decisions_logged = GREATEST(0, total_decisions_logged - 1),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = OLD.monitoring_session_id;
        
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_session_stats_on_decisions
    AFTER INSERT OR DELETE ON agent_decisions
    FOR EACH ROW EXECUTE FUNCTION update_session_statistics();

-- Update session statistics when metrics are added
CREATE OR REPLACE FUNCTION update_session_metrics_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE workflow_monitoring_sessions 
        SET 
            total_metrics_collected = total_metrics_collected + 1,
            last_activity_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = NEW.monitoring_session_id;
        
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE workflow_monitoring_sessions 
        SET 
            total_metrics_collected = GREATEST(0, total_metrics_collected - 1),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = OLD.monitoring_session_id;
        
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_session_metrics_count_trigger
    AFTER INSERT OR DELETE ON performance_metrics
    FOR EACH ROW EXECUTE FUNCTION update_session_metrics_count();

-- Update session statistics when patterns are detected
CREATE OR REPLACE FUNCTION update_session_patterns_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE workflow_monitoring_sessions 
        SET 
            total_patterns_detected = total_patterns_detected + 1,
            last_activity_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = NEW.monitoring_session_id;
        
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE workflow_monitoring_sessions 
        SET 
            total_patterns_detected = GREATEST(0, total_patterns_detected - 1),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = OLD.monitoring_session_id;
        
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_session_patterns_count_trigger
    AFTER INSERT OR DELETE ON learning_patterns
    FOR EACH ROW EXECUTE FUNCTION update_session_patterns_count();

-- Update session statistics when alerts are triggered
CREATE OR REPLACE FUNCTION update_session_alerts_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE workflow_monitoring_sessions 
        SET 
            total_alerts_triggered = total_alerts_triggered + 1,
            last_activity_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = NEW.monitoring_session_id;
        
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE workflow_monitoring_sessions 
        SET 
            total_alerts_triggered = GREATEST(0, total_alerts_triggered - 1),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = OLD.monitoring_session_id;
        
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_session_alerts_count_trigger
    AFTER INSERT OR DELETE ON performance_alerts
    FOR EACH ROW EXECUTE FUNCTION update_session_alerts_count();

-- ============================================================================
-- ADVANCED ANALYTICS DASHBOARD TABLES
-- ============================================================================
-- These tables support the advanced analytics dashboard system with Dutch market
-- intelligence, predictive modeling, and comprehensive reporting capabilities

-- Market Data Sources - tracks various Dutch market data sources
CREATE TABLE IF NOT EXISTS market_data_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Source identification
    source_id VARCHAR(50) NOT NULL UNIQUE,
    source_name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('cbs', 'dnb', 'kadaster', 'afm', 'nhg', 'bkr', 'external_api', 'internal')),
    
    -- Source configuration
    api_endpoint VARCHAR(500),
    api_key_required BOOLEAN DEFAULT false,
    update_frequency VARCHAR(20) DEFAULT 'daily' CHECK (update_frequency IN ('real_time', 'hourly', 'daily', 'weekly', 'monthly')),
    data_format VARCHAR(20) DEFAULT 'json' CHECK (data_format IN ('json', 'xml', 'csv', 'api')),
    
    -- Status and reliability
    is_active BOOLEAN DEFAULT true,
    last_successful_update TIMESTAMP WITH TIME ZONE,
    last_error_message TEXT,
    reliability_score DECIMAL(5,4) DEFAULT 1.0 CHECK (reliability_score BETWEEN 0 AND 1),
    
    -- Data quality metrics
    data_quality_score DECIMAL(5,4) DEFAULT 1.0 CHECK (data_quality_score BETWEEN 0 AND 1),
    completeness_score DECIMAL(5,4) DEFAULT 1.0 CHECK (completeness_score BETWEEN 0 AND 1),
    timeliness_score DECIMAL(5,4) DEFAULT 1.0 CHECK (timeliness_score BETWEEN 0 AND 1),
    
    -- Configuration and metadata
    source_config JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Indicators - stores current and historical market indicator values
CREATE TABLE IF NOT EXISTS market_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    source_id VARCHAR(50) REFERENCES market_data_sources(source_id) ON DELETE SET NULL,
    
    -- Indicator identification
    indicator_id VARCHAR(100) NOT NULL,
    indicator_name VARCHAR(255) NOT NULL,
    indicator_category VARCHAR(50) NOT NULL CHECK (indicator_category IN ('interest_rates', 'housing_market', 'lending_market', 'economic_indicators', 'regulatory_environment')),
    
    -- Indicator value and metadata
    indicator_value DECIMAL(15,6) NOT NULL,
    indicator_unit VARCHAR(20),
    measurement_type VARCHAR(50) DEFAULT 'point_in_time' CHECK (measurement_type IN ('point_in_time', 'cumulative', 'average', 'rate_of_change')),
    
    -- Trend and change analysis
    previous_value DECIMAL(15,6),
    change_amount DECIMAL(15,6),
    change_percentage DECIMAL(8,4),
    trend_direction VARCHAR(20) CHECK (trend_direction IN ('increasing', 'decreasing', 'stable', 'volatile')),
    
    -- Data quality and confidence
    data_quality DECIMAL(5,4) DEFAULT 1.0 CHECK (data_quality BETWEEN 0 AND 1),
    confidence_level DECIMAL(5,4) DEFAULT 1.0 CHECK (confidence_level BETWEEN 0 AND 1),
    
    -- Time information
    indicator_date DATE NOT NULL,
    reporting_period VARCHAR(20) DEFAULT 'daily',
    
    -- Additional metadata
    calculation_method VARCHAR(100),
    seasonal_adjustment BOOLEAN DEFAULT false,
    geographic_scope VARCHAR(50) DEFAULT 'netherlands',
    
    -- Extended properties and metadata
    properties JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique indicator per date
    UNIQUE(indicator_id, indicator_date)
);

-- Analytics Models - stores predictive models and their configurations
CREATE TABLE IF NOT EXISTS analytics_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Model identification
    model_id VARCHAR(255) NOT NULL UNIQUE,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('linear', 'random_forest', 'gradient_boosting', 'ensemble', 'neural_network', 'time_series')),
    model_version VARCHAR(20) DEFAULT '1.0',
    
    -- Model purpose and scope
    target_variable VARCHAR(100) NOT NULL,
    model_description TEXT,
    use_case VARCHAR(100) NOT NULL CHECK (use_case IN ('forecasting', 'classification', 'regression', 'clustering', 'anomaly_detection')),
    
    -- Model features and data
    input_features JSONB NOT NULL,
    feature_importance JSONB DEFAULT '{}',
    training_data_size INTEGER DEFAULT 0,
    validation_data_size INTEGER DEFAULT 0,
    
    -- Model performance metrics
    accuracy_metrics JSONB DEFAULT '{}',
    model_parameters JSONB DEFAULT '{}',
    hyperparameters JSONB DEFAULT '{}',
    
    -- Model lifecycle
    model_status VARCHAR(20) DEFAULT 'training' CHECK (model_status IN ('training', 'active', 'deprecated', 'failed', 'archived')),
    training_started_at TIMESTAMP WITH TIME ZONE,
    training_completed_at TIMESTAMP WITH TIME ZONE,
    last_prediction_at TIMESTAMP WITH TIME ZONE,
    
    -- Model validation
    validation_results JSONB DEFAULT '{}',
    cross_validation_score DECIMAL(8,6),
    overfitting_score DECIMAL(8,6),
    
    -- Model storage and deployment
    model_file_path VARCHAR(500),
    deployment_config JSONB DEFAULT '{}',
    
    -- Performance tracking
    prediction_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    average_processing_time DECIMAL(8,4) DEFAULT 0,
    
    -- Extended metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model Predictions - stores predictions made by analytics models
CREATE TABLE IF NOT EXISTS model_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    model_id VARCHAR(255) REFERENCES analytics_models(model_id) ON DELETE CASCADE,
    
    -- Prediction identification
    prediction_id VARCHAR(255) NOT NULL UNIQUE,
    prediction_type VARCHAR(50) DEFAULT 'point_forecast' CHECK (prediction_type IN ('point_forecast', 'interval_forecast', 'probability', 'classification', 'clustering')),
    
    -- Prediction input and output
    input_data JSONB NOT NULL,
    predicted_value DECIMAL(15,6),
    prediction_probability DECIMAL(5,4) CHECK (prediction_probability BETWEEN 0 AND 1),
    confidence_interval_lower DECIMAL(15,6),
    confidence_interval_upper DECIMAL(15,6),
    confidence_level DECIMAL(5,4) DEFAULT 0.95 CHECK (confidence_level BETWEEN 0 AND 1),
    
    -- Prediction metadata
    prediction_horizon INTEGER, -- Number of periods ahead
    forecast_period VARCHAR(20), -- daily, weekly, monthly
    scenario VARCHAR(100), -- baseline, optimistic, pessimistic
    
    -- Model performance at prediction time
    model_accuracy DECIMAL(8,6),
    processing_time DECIMAL(8,4),
    model_version VARCHAR(20),
    
    -- Actual outcome tracking (for model validation)
    actual_value DECIMAL(15,6),
    prediction_error DECIMAL(15,6),
    absolute_error DECIMAL(15,6),
    squared_error DECIMAL(15,6),
    
    -- Validation status
    is_validated BOOLEAN DEFAULT false,
    validation_date TIMESTAMP WITH TIME ZONE,
    
    -- Extended properties
    properties JSONB DEFAULT '{}',
    
    -- Timestamps
    prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Insights - stores generated market insights and analysis
CREATE TABLE IF NOT EXISTS market_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Insight identification
    insight_id VARCHAR(255) NOT NULL UNIQUE,
    insight_type VARCHAR(50) NOT NULL CHECK (insight_type IN ('trend_analysis', 'risk_assessment', 'opportunity_detection', 'regulatory_impact', 'market_anomaly', 'competitive_analysis')),
    
    -- Insight content
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    executive_summary TEXT,
    
    -- Insight classification
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    impact_score DECIMAL(5,4) NOT NULL CHECK (impact_score BETWEEN 0 AND 1),
    urgency VARCHAR(20) DEFAULT 'normal' CHECK (urgency IN ('low', 'normal', 'high', 'urgent')),
    
    -- Temporal context
    time_horizon VARCHAR(20) NOT NULL CHECK (time_horizon IN ('immediate', 'short_term', 'medium_term', 'long_term')),
    relevant_from TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    relevant_until TIMESTAMP WITH TIME ZONE,
    
    -- Supporting data and analysis
    data_points JSONB NOT NULL,
    supporting_data JSONB DEFAULT '{}',
    statistical_significance DECIMAL(5,4),
    correlation_strength DECIMAL(5,4),
    
    -- Recommendations and actions
    recommendations JSONB DEFAULT '[]',
    suggested_actions JSONB DEFAULT '[]',
    potential_risks JSONB DEFAULT '[]',
    potential_opportunities JSONB DEFAULT '[]',
    
    -- Source and methodology
    data_sources JSONB DEFAULT '[]',
    analysis_method VARCHAR(100),
    model_ids JSONB DEFAULT '[]', -- References to models used
    
    -- Insight status and lifecycle
    insight_status VARCHAR(20) DEFAULT 'active' CHECK (insight_status IN ('draft', 'active', 'archived', 'superseded')),
    superseded_by UUID REFERENCES market_insights(id) ON DELETE SET NULL,
    
    -- User interactions
    view_count INTEGER DEFAULT 0,
    action_taken BOOLEAN DEFAULT false,
    user_rating DECIMAL(3,2) CHECK (user_rating BETWEEN 1 AND 5),
    user_feedback TEXT,
    
    -- Extended metadata
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analytics Reports - stores generated analytics reports
CREATE TABLE IF NOT EXISTS analytics_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Report identification
    report_id VARCHAR(255) NOT NULL UNIQUE,
    report_type VARCHAR(50) NOT NULL CHECK (report_type IN ('market_analysis', 'risk_assessment', 'performance', 'compliance', 'custom', 'executive_summary')),
    report_name VARCHAR(255) NOT NULL,
    report_description TEXT,
    
    -- Report content
    executive_summary TEXT,
    key_findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    conclusions TEXT,
    
    -- Report scope and parameters
    time_period VARCHAR(50) NOT NULL,
    analysis_parameters JSONB DEFAULT '{}',
    data_sources JSONB DEFAULT '[]',
    filters_applied JSONB DEFAULT '{}',
    
    -- Report sections and structure
    report_sections JSONB DEFAULT '[]',
    table_of_contents JSONB DEFAULT '[]',
    
    -- Included insights and analysis
    insight_ids JSONB DEFAULT '[]', -- References to market_insights
    metric_summaries JSONB DEFAULT '{}',
    trend_analysis JSONB DEFAULT '{}',
    comparative_analysis JSONB DEFAULT '{}',
    
    -- Visualizations and charts
    visualizations JSONB DEFAULT '{}',
    chart_configurations JSONB DEFAULT '{}',
    
    -- Report quality and metadata
    data_quality_score DECIMAL(5,4) DEFAULT 1.0 CHECK (data_quality_score BETWEEN 0 AND 1),
    completeness_score DECIMAL(5,4) DEFAULT 1.0 CHECK (completeness_score BETWEEN 0 AND 1),
    analysis_depth VARCHAR(20) DEFAULT 'standard' CHECK (analysis_depth IN ('basic', 'standard', 'detailed', 'comprehensive')),
    
    -- Generation and processing
    generation_time DECIMAL(8,4),
    processing_steps JSONB DEFAULT '[]',
    generation_status VARCHAR(20) DEFAULT 'generated' CHECK (generation_status IN ('generating', 'generated', 'failed', 'archived')),
    
    -- Export and distribution
    export_formats JSONB DEFAULT '["json"]',
    file_paths JSONB DEFAULT '{}',
    distribution_list JSONB DEFAULT '[]',
    
    -- Report lifecycle and access
    report_status VARCHAR(20) DEFAULT 'draft' CHECK (report_status IN ('draft', 'published', 'archived', 'superseded')),
    published_at TIMESTAMP WITH TIME ZONE,
    archived_at TIMESTAMP WITH TIME ZONE,
    access_level VARCHAR(20) DEFAULT 'private' CHECK (access_level IN ('private', 'internal', 'public')),
    
    -- User interactions
    view_count INTEGER DEFAULT 0,
    download_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    
    -- Extended metadata
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Dashboard Configurations - stores user dashboard preferences and layouts
CREATE TABLE IF NOT EXISTS dashboard_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Configuration identification
    config_id VARCHAR(255) NOT NULL,
    config_name VARCHAR(255) NOT NULL,
    config_type VARCHAR(50) DEFAULT 'personal' CHECK (config_type IN ('personal', 'team', 'organization', 'template')),
    
    -- Dashboard layout and structure
    layout_config JSONB NOT NULL,
    widget_configurations JSONB DEFAULT '{}',
    visualization_preferences JSONB DEFAULT '{}',
    
    -- Display preferences
    theme VARCHAR(20) DEFAULT 'light' CHECK (theme IN ('light', 'dark', 'auto')),
    refresh_interval INTEGER DEFAULT 30, -- seconds
    auto_refresh BOOLEAN DEFAULT true,
    time_zone VARCHAR(50) DEFAULT 'Europe/Amsterdam',
    
    -- Data preferences
    default_time_period VARCHAR(20) DEFAULT '12m',
    preferred_metrics JSONB DEFAULT '[]',
    alert_thresholds JSONB DEFAULT '{}',
    
    -- Sharing and access
    is_shared BOOLEAN DEFAULT false,
    shared_with JSONB DEFAULT '[]',
    access_permissions JSONB DEFAULT '{}',
    
    -- Configuration metadata
    is_default BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    
    -- Extended metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique config_id per user
    UNIQUE(user_id, config_id)
);

-- Benchmark Comparisons - stores benchmark data and comparisons
CREATE TABLE IF NOT EXISTS benchmark_comparisons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Benchmark identification
    benchmark_id VARCHAR(255) NOT NULL UNIQUE,
    benchmark_name VARCHAR(255) NOT NULL,
    benchmark_type VARCHAR(50) NOT NULL CHECK (benchmark_type IN ('peer_comparison', 'historical_comparison', 'regional_comparison', 'international_comparison', 'industry_standard')),
    
    -- Comparison scope
    comparison_metrics JSONB NOT NULL,
    benchmark_data JSONB NOT NULL,
    our_performance JSONB NOT NULL,
    
    -- Statistical analysis
    comparative_analysis JSONB DEFAULT '{}',
    statistical_significance JSONB DEFAULT '{}',
    percentile_rankings JSONB DEFAULT '{}',
    
    -- Benchmark metadata
    data_sources JSONB DEFAULT '[]',
    methodology VARCHAR(255),
    sample_size INTEGER,
    confidence_level DECIMAL(5,4) DEFAULT 0.95,
    
    -- Time context
    benchmark_period VARCHAR(50) NOT NULL,
    comparison_date DATE NOT NULL,
    data_freshness VARCHAR(20) DEFAULT 'current' CHECK (data_freshness IN ('current', 'recent', 'historical', 'outdated')),
    
    -- Analysis results
    performance_summary JSONB DEFAULT '{}',
    strengths JSONB DEFAULT '[]',
    improvement_areas JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    
    -- Extended metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR ADVANCED ANALYTICS DASHBOARD TABLES
-- ============================================================================

-- Market data sources indexes
CREATE INDEX IF NOT EXISTS idx_market_data_sources_source_id ON market_data_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_market_data_sources_type ON market_data_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_market_data_sources_active ON market_data_sources(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_market_data_sources_last_update ON market_data_sources(last_successful_update DESC);

-- Market indicators indexes
CREATE INDEX IF NOT EXISTS idx_market_indicators_indicator_id ON market_indicators(indicator_id);
CREATE INDEX IF NOT EXISTS idx_market_indicators_category ON market_indicators(indicator_category);
CREATE INDEX IF NOT EXISTS idx_market_indicators_date ON market_indicators(indicator_date DESC);
CREATE INDEX IF NOT EXISTS idx_market_indicators_source ON market_indicators(source_id);
CREATE INDEX IF NOT EXISTS idx_market_indicators_recorded_at ON market_indicators(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_indicators_trend ON market_indicators(trend_direction);

-- Analytics models indexes
CREATE INDEX IF NOT EXISTS idx_analytics_models_model_id ON analytics_models(model_id);
CREATE INDEX IF NOT EXISTS idx_analytics_models_type ON analytics_models(model_type);
CREATE INDEX IF NOT EXISTS idx_analytics_models_status ON analytics_models(model_status);
CREATE INDEX IF NOT EXISTS idx_analytics_models_target_variable ON analytics_models(target_variable);
CREATE INDEX IF NOT EXISTS idx_analytics_models_use_case ON analytics_models(use_case);
CREATE INDEX IF NOT EXISTS idx_analytics_models_created_at ON analytics_models(created_at DESC);

-- Model predictions indexes
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_id ON model_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_prediction_id ON model_predictions(prediction_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_timestamp ON model_predictions(prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_predictions_type ON model_predictions(prediction_type);
CREATE INDEX IF NOT EXISTS idx_model_predictions_validated ON model_predictions(is_validated);

-- Market insights indexes
CREATE INDEX IF NOT EXISTS idx_market_insights_insight_id ON market_insights(insight_id);
CREATE INDEX IF NOT EXISTS idx_market_insights_type ON market_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_market_insights_severity ON market_insights(severity);
CREATE INDEX IF NOT EXISTS idx_market_insights_status ON market_insights(insight_status);
CREATE INDEX IF NOT EXISTS idx_market_insights_time_horizon ON market_insights(time_horizon);
CREATE INDEX IF NOT EXISTS idx_market_insights_generated_at ON market_insights(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_insights_confidence ON market_insights(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_market_insights_impact ON market_insights(impact_score DESC);

-- Analytics reports indexes
CREATE INDEX IF NOT EXISTS idx_analytics_reports_report_id ON analytics_reports(report_id);
CREATE INDEX IF NOT EXISTS idx_analytics_reports_type ON analytics_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_analytics_reports_status ON analytics_reports(report_status);
CREATE INDEX IF NOT EXISTS idx_analytics_reports_generated_at ON analytics_reports(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_reports_user_id ON analytics_reports(user_id);
CREATE INDEX IF NOT EXISTS idx_analytics_reports_time_period ON analytics_reports(time_period);

-- Dashboard configurations indexes
CREATE INDEX IF NOT EXISTS idx_dashboard_configurations_user_id ON dashboard_configurations(user_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_configurations_config_type ON dashboard_configurations(config_type);
CREATE INDEX IF NOT EXISTS idx_dashboard_configurations_active ON dashboard_configurations(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_dashboard_configurations_default ON dashboard_configurations(user_id, is_default) WHERE is_default = true;

-- Benchmark comparisons indexes
CREATE INDEX IF NOT EXISTS idx_benchmark_comparisons_benchmark_id ON benchmark_comparisons(benchmark_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_comparisons_type ON benchmark_comparisons(benchmark_type);
CREATE INDEX IF NOT EXISTS idx_benchmark_comparisons_date ON benchmark_comparisons(comparison_date DESC);
CREATE INDEX IF NOT EXISTS idx_benchmark_comparisons_timestamp ON benchmark_comparisons(analysis_timestamp DESC);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_market_indicators_category_date ON market_indicators(indicator_category, indicator_date DESC);
CREATE INDEX IF NOT EXISTS idx_market_indicators_id_date ON market_indicators(indicator_id, indicator_date DESC);
CREATE INDEX IF NOT EXISTS idx_model_predictions_model_timestamp ON model_predictions(model_id, prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_insights_type_status ON market_insights(insight_type, insight_status);
CREATE INDEX IF NOT EXISTS idx_analytics_reports_type_status ON analytics_reports(report_type, report_status);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_market_insights_title_description_fts ON market_insights USING gin(to_tsvector('english', title || ' ' || description));
CREATE INDEX IF NOT EXISTS idx_analytics_reports_name_description_fts ON analytics_reports USING gin(to_tsvector('english', report_name || ' ' || COALESCE(report_description, '')));

-- JSONB indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_market_indicators_properties_gin ON market_indicators USING gin(properties);
CREATE INDEX IF NOT EXISTS idx_analytics_models_parameters_gin ON analytics_models USING gin(model_parameters);
CREATE INDEX IF NOT EXISTS idx_analytics_models_features_gin ON analytics_models USING gin(input_features);
CREATE INDEX IF NOT EXISTS idx_market_insights_data_points_gin ON market_insights USING gin(data_points);
CREATE INDEX IF NOT EXISTS idx_market_insights_recommendations_gin ON market_insights USING gin(recommendations);
CREATE INDEX IF NOT EXISTS idx_analytics_reports_visualizations_gin ON analytics_reports USING gin(visualizations);

-- ============================================================================
-- TRIGGERS FOR ADVANCED ANALYTICS DASHBOARD TABLES
-- ============================================================================

-- Update timestamps
CREATE TRIGGER update_market_data_sources_updated_at
    BEFORE UPDATE ON market_data_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analytics_models_updated_at
    BEFORE UPDATE ON analytics_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_market_insights_updated_at
    BEFORE UPDATE ON market_insights
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analytics_reports_updated_at
    BEFORE UPDATE ON analytics_reports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_dashboard_configurations_updated_at
    BEFORE UPDATE ON dashboard_configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_benchmark_comparisons_updated_at
    BEFORE UPDATE ON benchmark_comparisons
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update model statistics when predictions are added
CREATE OR REPLACE FUNCTION update_model_prediction_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Update model statistics
        UPDATE analytics_models 
        SET 
            prediction_count = prediction_count + 1,
            last_prediction_at = NEW.prediction_timestamp,
            updated_at = CURRENT_TIMESTAMP
        WHERE model_id = NEW.model_id;
        
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' AND OLD.is_validated = false AND NEW.is_validated = true THEN
        -- Update error statistics when validation is added
        UPDATE analytics_models
        SET
            updated_at = CURRENT_TIMESTAMP
        WHERE model_id = NEW.model_id;
        
        RETURN NEW;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_model_prediction_stats_trigger
    AFTER INSERT OR UPDATE ON model_predictions
    FOR EACH ROW EXECUTE FUNCTION update_model_prediction_stats();

-- Update insight view counts
CREATE OR REPLACE FUNCTION increment_insight_view_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE market_insights 
    SET 
        view_count = view_count + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE insight_id = NEW.insight_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Update report view/download counts
CREATE OR REPLACE FUNCTION update_report_interaction_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        -- Update report statistics based on what changed
        IF NEW.view_count > OLD.view_count THEN
            NEW.updated_at = CURRENT_TIMESTAMP;
        ELSIF NEW.download_count > OLD.download_count THEN
            NEW.updated_at = CURRENT_TIMESTAMP;
        ELSIF NEW.share_count > OLD.share_count THEN
            NEW.updated_at = CURRENT_TIMESTAMP;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_report_interaction_stats_trigger
    BEFORE UPDATE ON analytics_reports
    FOR EACH ROW EXECUTE FUNCTION update_report_interaction_stats();

-- ============================================================================
-- ANOMALY DETECTION INTERFACE TABLES
-- ============================================================================
-- These tables support the anomaly detection interface system with real-time
-- pattern recognition, alert management, and investigation tools

-- Anomaly Detection Records - stores detected anomalies
CREATE TABLE IF NOT EXISTS anomaly_detection_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Anomaly identification
    anomaly_id VARCHAR(255) NOT NULL UNIQUE,
    detection_type VARCHAR(50) NOT NULL CHECK (detection_type IN ('statistical', 'ml_based', 'rule_based', 'hybrid')),
    anomaly_category VARCHAR(100) NOT NULL,

    -- Anomaly classification
    severity VARCHAR(20) NOT NULL DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    confidence_score DECIMAL(5,4) NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    anomaly_score DECIMAL(10,6) NOT NULL,

    -- Anomaly description
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    affected_entities JSONB DEFAULT '[]',

    -- Detection methodology
    detection_method VARCHAR(100) NOT NULL,
    detection_parameters JSONB DEFAULT '{}',
    deviation_metrics JSONB DEFAULT '{}',

    -- Investigation and response
    investigation_priority INTEGER DEFAULT 3 CHECK (investigation_priority BETWEEN 1 AND 5),
    recommended_actions JSONB DEFAULT '[]',
    investigation_hints JSONB DEFAULT '[]',
    related_anomalies JSONB DEFAULT '[]',

    -- Time information
    detection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    time_window VARCHAR(50),

    -- Status and resolution
    status VARCHAR(20) DEFAULT 'detected' CHECK (status IN ('detected', 'investigating', 'resolved', 'false_positive', 'archived')),
    resolution_notes TEXT,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP WITH TIME ZONE,

    -- Source data information
    data_source VARCHAR(100),
    data_sample JSONB DEFAULT '{}',
    feature_importance JSONB DEFAULT '{}',

    -- Additional metadata
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alert Rules - stores anomaly detection rules and configurations
CREATE TABLE IF NOT EXISTS anomaly_alert_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Rule identification
    rule_id VARCHAR(255) NOT NULL UNIQUE,
    rule_name VARCHAR(255) NOT NULL,
    rule_type VARCHAR(50) NOT NULL CHECK (rule_type IN ('threshold', 'statistical', 'pattern', 'composite', 'ml_based')),
    category VARCHAR(100) NOT NULL,

    -- Rule configuration
    conditions JSONB NOT NULL,
    thresholds JSONB DEFAULT '{}',
    parameters JSONB DEFAULT '{}',

    -- Severity and escalation
    severity_mapping JSONB DEFAULT '{}',
    escalation_rules JSONB DEFAULT '[]',
    notification_channels JSONB DEFAULT '["in_app"]',
    suppression_rules JSONB DEFAULT '{}',

    -- Rule status and performance
    is_active BOOLEAN DEFAULT true,
    last_triggered TIMESTAMP WITH TIME ZONE,
    trigger_count INTEGER DEFAULT 0,
    false_positive_count INTEGER DEFAULT 0,
    accuracy_score DECIMAL(5,4) DEFAULT 1.0 CHECK (accuracy_score BETWEEN 0 AND 1),

    -- Rule lifecycle
    created_by VARCHAR(100) DEFAULT 'system',
    last_modified_by VARCHAR(100),
    version INTEGER DEFAULT 1,

    -- Extended metadata
    description TEXT,
    documentation JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Investigation Sessions - tracks anomaly investigation sessions
CREATE TABLE IF NOT EXISTS investigation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Session identification
    session_id VARCHAR(255) NOT NULL UNIQUE,
    anomaly_ids JSONB NOT NULL,
    investigator_id VARCHAR(100) NOT NULL,
    session_name VARCHAR(255) NOT NULL,

    -- Investigation status
    investigation_status VARCHAR(20) DEFAULT 'active' CHECK (investigation_status IN ('active', 'paused', 'completed', 'cancelled')),
    priority_level VARCHAR(20) DEFAULT 'medium' CHECK (priority_level IN ('low', 'medium', 'high', 'urgent')),

    -- Investigation data
    hypothesis JSONB DEFAULT '[]',
    evidence_collected JSONB DEFAULT '[]',
    analysis_results JSONB DEFAULT '{}',
    findings JSONB DEFAULT '[]',

    -- Investigation outcomes
    root_cause TEXT,
    impact_assessment JSONB DEFAULT '{}',
    recommendations JSONB DEFAULT '[]',

    -- Time tracking
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion TIMESTAMP WITH TIME ZONE,

    -- Collaboration
    collaborators JSONB DEFAULT '[]',
    notes JSONB DEFAULT '[]',
    attachments JSONB DEFAULT '[]',

    -- Investigation metrics
    hours_spent DECIMAL(8,2) DEFAULT 0,
    complexity_score INTEGER DEFAULT 3 CHECK (complexity_score BETWEEN 1 AND 5),
    resolution_quality INTEGER CHECK (resolution_quality BETWEEN 1 AND 5),

    -- Extended metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alert Notifications - stores alert notifications and their delivery status
CREATE TABLE IF NOT EXISTS anomaly_alert_notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    anomaly_id VARCHAR(255) REFERENCES anomaly_detection_records(anomaly_id) ON DELETE CASCADE,
    rule_id VARCHAR(255) REFERENCES anomaly_alert_rules(rule_id) ON DELETE SET NULL,

    -- Notification identification
    notification_id VARCHAR(255) NOT NULL UNIQUE,
    notification_type VARCHAR(50) NOT NULL CHECK (notification_type IN ('email', 'sms', 'webhook', 'in_app', 'slack')),

    -- Notification content
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    priority INTEGER DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),

    -- Recipients and delivery
    recipients JSONB NOT NULL,
    delivery_status VARCHAR(20) DEFAULT 'pending' CHECK (delivery_status IN ('pending', 'sent', 'delivered', 'failed', 'suppressed')),
    delivery_attempts INTEGER DEFAULT 0,
    delivery_errors JSONB DEFAULT '[]',

    -- Timing
    scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sent_at TIMESTAMP WITH TIME ZONE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(100),

    -- Suppression and escalation
    is_suppressed BOOLEAN DEFAULT false,
    suppression_reason VARCHAR(255),
    escalation_level INTEGER DEFAULT 1,
    escalated_at TIMESTAMP WITH TIME ZONE,

    -- Extended metadata
    channel_config JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Detection Statistics - tracks anomaly detection performance metrics
CREATE TABLE IF NOT EXISTS anomaly_detection_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Statistics identification
    stats_id VARCHAR(255) NOT NULL UNIQUE,
    measurement_period VARCHAR(20) NOT NULL CHECK (measurement_period IN ('hourly', 'daily', 'weekly', 'monthly')),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Detection metrics
    total_detections INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    precision_score DECIMAL(5,4) CHECK (precision_score BETWEEN 0 AND 1),
    recall_score DECIMAL(5,4) CHECK (recall_score BETWEEN 0 AND 1),
    f1_score DECIMAL(5,4) CHECK (f1_score BETWEEN 0 AND 1),

    -- Performance metrics
    average_processing_time DECIMAL(8,4) DEFAULT 0,
    max_processing_time DECIMAL(8,4) DEFAULT 0,
    min_processing_time DECIMAL(8,4) DEFAULT 0,
    timeout_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,

    -- Detection method breakdown
    statistical_detections INTEGER DEFAULT 0,
    ml_detections INTEGER DEFAULT 0,
    rule_detections INTEGER DEFAULT 0,
    hybrid_detections INTEGER DEFAULT 0,

    -- Severity distribution
    low_severity_count INTEGER DEFAULT 0,
    medium_severity_count INTEGER DEFAULT 0,
    high_severity_count INTEGER DEFAULT 0,
    critical_severity_count INTEGER DEFAULT 0,

    -- Resolution metrics
    resolved_count INTEGER DEFAULT 0,
    average_resolution_time DECIMAL(8,2) DEFAULT 0,
    investigation_count INTEGER DEFAULT 0,
    escalated_count INTEGER DEFAULT 0,

    -- Data quality metrics
    data_quality_score DECIMAL(5,4) DEFAULT 1.0 CHECK (data_quality_score BETWEEN 0 AND 1),
    data_completeness DECIMAL(5,4) DEFAULT 1.0 CHECK (data_completeness BETWEEN 0 AND 1),
    model_drift_score DECIMAL(5,4) DEFAULT 0 CHECK (model_drift_score BETWEEN 0 AND 1),

    -- Extended metrics
    custom_metrics JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Pattern Analysis Results - stores results of pattern analysis investigations
CREATE TABLE IF NOT EXISTS pattern_analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255) REFERENCES investigation_sessions(session_id) ON DELETE CASCADE,

    -- Analysis identification
    analysis_id VARCHAR(255) NOT NULL UNIQUE,
    analysis_type VARCHAR(50) NOT NULL CHECK (analysis_type IN ('temporal', 'correlation', 'clustering', 'trend', 'comprehensive')),

    -- Analysis configuration
    analysis_parameters JSONB DEFAULT '{}',
    data_sources JSONB DEFAULT '[]',
    feature_selection JSONB DEFAULT '[]',

    -- Analysis results
    patterns_found JSONB DEFAULT '[]',
    statistical_summary JSONB DEFAULT '{}',
    correlation_matrix JSONB DEFAULT '{}',
    temporal_patterns JSONB DEFAULT '{}',
    clustering_results JSONB DEFAULT '{}',

    -- Insights and findings
    key_insights JSONB DEFAULT '[]',
    anomaly_explanations JSONB DEFAULT '[]',
    prediction_confidence DECIMAL(5,4) CHECK (prediction_confidence BETWEEN 0 AND 1),

    -- Analysis quality metrics
    analysis_quality_score DECIMAL(5,4) DEFAULT 0 CHECK (analysis_quality_score BETWEEN 0 AND 1),
    data_coverage DECIMAL(5,4) DEFAULT 0 CHECK (data_coverage BETWEEN 0 AND 1),
    pattern_strength DECIMAL(5,4) DEFAULT 0 CHECK (pattern_strength BETWEEN 0 AND 1),

    -- Performance metrics
    processing_time DECIMAL(8,4) DEFAULT 0,
    memory_usage DECIMAL(10,2) DEFAULT 0,
    compute_resources JSONB DEFAULT '{}',

    -- Visualizations and outputs
    visualizations JSONB DEFAULT '{}',
    export_formats JSONB DEFAULT '[]',
    generated_reports JSONB DEFAULT '[]',

    -- Extended metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    analysis_started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    analysis_completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR ANOMALY DETECTION INTERFACE TABLES
-- ============================================================================

-- Anomaly detection records indexes
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_anomaly_id ON anomaly_detection_records(anomaly_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_detection_type ON anomaly_detection_records(detection_type);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_category ON anomaly_detection_records(anomaly_category);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_severity ON anomaly_detection_records(severity);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_status ON anomaly_detection_records(status);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_detection_timestamp ON anomaly_detection_records(detection_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_data_timestamp ON anomaly_detection_records(data_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_confidence ON anomaly_detection_records(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_records_user_id ON anomaly_detection_records(user_id);

-- Alert rules indexes
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_rules_rule_id ON anomaly_alert_rules(rule_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_rules_rule_type ON anomaly_alert_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_rules_category ON anomaly_alert_rules(category);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_rules_active ON anomaly_alert_rules(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_rules_last_triggered ON anomaly_alert_rules(last_triggered DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_rules_created_by ON anomaly_alert_rules(created_by);

-- Investigation sessions indexes
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_session_id ON investigation_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_investigator_id ON investigation_sessions(investigator_id);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_status ON investigation_sessions(investigation_status);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_priority ON investigation_sessions(priority_level);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_started_at ON investigation_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_last_activity ON investigation_sessions(last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_user_id ON investigation_sessions(user_id);

-- Alert notifications indexes
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_notifications_notification_id ON anomaly_alert_notifications(notification_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_notifications_anomaly_id ON anomaly_alert_notifications(anomaly_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_notifications_rule_id ON anomaly_alert_notifications(rule_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_notifications_type ON anomaly_alert_notifications(notification_type);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_notifications_delivery_status ON anomaly_alert_notifications(delivery_status);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_notifications_severity ON anomaly_alert_notifications(severity);
CREATE INDEX IF NOT EXISTS idx_anomaly_alert_notifications_scheduled_at ON anomaly_alert_notifications(scheduled_at DESC);

-- Detection statistics indexes
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_statistics_stats_id ON anomaly_detection_statistics(stats_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_statistics_period ON anomaly_detection_statistics(measurement_period);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_statistics_start ON anomaly_detection_statistics(period_start DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_statistics_end ON anomaly_detection_statistics(period_end DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_detection_statistics_calculated_at ON anomaly_detection_statistics(calculated_at DESC);

-- Pattern analysis results indexes
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_results_analysis_id ON pattern_analysis_results(analysis_id);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_results_session_id ON pattern_analysis_results(session_id);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_results_analysis_type ON pattern_analysis_results(analysis_type);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_results_started_at ON pattern_analysis_results(analysis_started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_results_completed_at ON pattern_analysis_results(analysis_completed_at DESC);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_anomaly_records_severity_timestamp ON anomaly_detection_records(severity, detection_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_records_status_category ON anomaly_detection_records(status, anomaly_category);
CREATE INDEX IF NOT EXISTS idx_anomaly_records_type_confidence ON anomaly_detection_records(detection_type, confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_status_priority ON investigation_sessions(investigation_status, priority_level);
CREATE INDEX IF NOT EXISTS idx_alert_notifications_status_type ON anomaly_alert_notifications(delivery_status, notification_type);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_anomaly_records_title_description_fts ON anomaly_detection_records USING gin(to_tsvector('english', title || ' ' || description));
CREATE INDEX IF NOT EXISTS idx_alert_rules_name_description_fts ON anomaly_alert_rules USING gin(to_tsvector('english', rule_name || ' ' || COALESCE(description, '')));
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_name_fts ON investigation_sessions USING gin(to_tsvector('english', session_name));

-- JSONB indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_anomaly_records_affected_entities_gin ON anomaly_detection_records USING gin(affected_entities);
CREATE INDEX IF NOT EXISTS idx_anomaly_records_metadata_gin ON anomaly_detection_records USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_anomaly_records_tags_gin ON anomaly_detection_records USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_alert_rules_conditions_gin ON anomaly_alert_rules USING gin(conditions);
CREATE INDEX IF NOT EXISTS idx_alert_rules_parameters_gin ON anomaly_alert_rules USING gin(parameters);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_findings_gin ON investigation_sessions USING gin(findings);
CREATE INDEX IF NOT EXISTS idx_investigation_sessions_evidence_gin ON investigation_sessions USING gin(evidence_collected);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_patterns_gin ON pattern_analysis_results USING gin(patterns_found);

-- ============================================================================
-- TRIGGERS FOR ANOMALY DETECTION INTERFACE TABLES
-- ============================================================================

-- Update timestamps
CREATE TRIGGER update_anomaly_detection_records_updated_at
    BEFORE UPDATE ON anomaly_detection_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_anomaly_alert_rules_updated_at
    BEFORE UPDATE ON anomaly_alert_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_investigation_sessions_updated_at
    BEFORE UPDATE ON investigation_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_anomaly_alert_notifications_updated_at
    BEFORE UPDATE ON anomaly_alert_notifications
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pattern_analysis_results_updated_at
    BEFORE UPDATE ON pattern_analysis_results
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update investigation session activity
CREATE OR REPLACE FUNCTION update_investigation_activity()
RETURNS TRIGGER AS $$
BEGIN
    -- Update last_activity_at whenever investigation session is modified
    NEW.last_activity_at = CURRENT_TIMESTAMP;
    
    -- Update hours_spent based on time elapsed
    IF OLD.started_at IS NOT NULL THEN
        NEW.hours_spent = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - OLD.started_at)) / 3600.0;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_investigation_activity_trigger
    BEFORE UPDATE ON investigation_sessions
    FOR EACH ROW EXECUTE FUNCTION update_investigation_activity();

-- Update alert rule statistics when triggered
CREATE OR REPLACE FUNCTION update_alert_rule_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update rule statistics when a notification is created for a rule
    IF TG_OP = 'INSERT' AND NEW.rule_id IS NOT NULL THEN
        UPDATE anomaly_alert_rules
        SET
            last_triggered = NEW.created_at,
            trigger_count = trigger_count + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE rule_id = NEW.rule_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_alert_rule_stats_trigger
    AFTER INSERT ON anomaly_alert_notifications
    FOR EACH ROW EXECUTE FUNCTION update_alert_rule_stats();

-- Update false positive counts when feedback is updated
CREATE OR REPLACE FUNCTION update_false_positive_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update false positive count when anomaly status changes to false_positive
    IF TG_OP = 'UPDATE' AND OLD.status != 'false_positive' AND NEW.status = 'false_positive' THEN
        -- Find associated alert rules and update their false positive counts
        -- This is a simplified version - in practice, you'd need to track the relationship
        -- between anomalies and the rules that detected them
        UPDATE anomaly_alert_rules 
        SET 
            false_positive_count = false_positive_count + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE category = NEW.anomaly_category;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_false_positive_stats_trigger
    AFTER UPDATE ON anomaly_detection_records
    FOR EACH ROW EXECUTE FUNCTION update_false_positive_stats();

-- ============================================================================
-- ADVANCED FIELD VALIDATION ENGINE TABLES
-- ============================================================================
-- These tables support the advanced field validation system with real-time
-- validation, error correction suggestions, and AFM compliance checking

-- Validation Rules - stores field validation rules and configurations
CREATE TABLE IF NOT EXISTS field_validation_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Rule identification
    rule_id VARCHAR(255) NOT NULL UNIQUE,
    rule_name VARCHAR(255) NOT NULL,
    field_path VARCHAR(200) NOT NULL,
    field_type VARCHAR(50) NOT NULL CHECK (field_type IN ('text', 'email', 'phone', 'currency', 'percentage', 'date', 'integer', 'decimal', 'boolean', 'enum', 'bsn', 'iban', 'postcode', 'address', 'name', 'custom')),
    rule_type VARCHAR(50) NOT NULL CHECK (rule_type IN ('required', 'format', 'range', 'length', 'pattern', 'custom', 'afm_compliance')),

    -- Rule configuration
    parameters JSONB DEFAULT '{}',
    conditions JSONB DEFAULT '{}',

    -- Error handling and messaging
    error_message VARCHAR(500) DEFAULT '',
    suggestion_template VARCHAR(500) DEFAULT '',
    severity VARCHAR(20) DEFAULT 'error' CHECK (severity IN ('info', 'warning', 'error', 'critical')),

    -- Rule status and behavior
    is_active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
    dependencies JSONB DEFAULT '[]',

    -- AFM compliance information
    afm_article VARCHAR(50),
    compliance_category VARCHAR(100),

    -- Rule performance tracking
    usage_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    false_positive_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,

    -- Rule lifecycle
    created_by VARCHAR(100) DEFAULT 'system',
    last_modified_by VARCHAR(100),
    version INTEGER DEFAULT 1,

    -- Extended metadata
    description TEXT,
    documentation JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Validation Sessions - tracks validation sessions and their results
CREATE TABLE IF NOT EXISTS field_validation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Session identification
    session_id VARCHAR(255) NOT NULL UNIQUE,
    session_name VARCHAR(255),
    session_type VARCHAR(50) DEFAULT 'field_validation' CHECK (session_type IN ('field_validation', 'bulk_validation', 'compliance_check', 'rule_testing')),

    -- Validation scope and configuration
    validation_scope JSONB NOT NULL,
    validation_config JSONB DEFAULT '{}',
    field_count INTEGER DEFAULT 0,

    -- Session results
    is_valid BOOLEAN,
    total_fields INTEGER DEFAULT 0,
    validated_fields INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    info_count INTEGER DEFAULT 0,

    -- Scoring and quality metrics
    overall_score DECIMAL(5,2) DEFAULT 0,
    compliance_score DECIMAL(5,2) DEFAULT 0,
    quality_score DECIMAL(5,2) DEFAULT 0,
    completeness_score DECIMAL(5,2) DEFAULT 0,

    -- Performance metrics
    processing_time DECIMAL(8,4) DEFAULT 0,
    rules_applied INTEGER DEFAULT 0,
    suggestions_generated INTEGER DEFAULT 0,
    corrections_applied INTEGER DEFAULT 0,

    -- Session status
    session_status VARCHAR(20) DEFAULT 'completed' CHECK (session_status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- AFM compliance tracking
    afm_compliance_checked BOOLEAN DEFAULT false,
    afm_compliance_score DECIMAL(5,2) DEFAULT 0,
    afm_violations INTEGER DEFAULT 0,

    -- Extended metadata
    client_info JSONB DEFAULT '{}',
    application_context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Validation Messages - stores individual validation messages and suggestions
CREATE TABLE IF NOT EXISTS field_validation_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES field_validation_sessions(session_id) ON DELETE CASCADE,
    rule_id VARCHAR(255) REFERENCES field_validation_rules(rule_id) ON DELETE SET NULL,

    -- Message identification
    message_id VARCHAR(255) NOT NULL UNIQUE,
    field_path VARCHAR(200) NOT NULL,
    field_type VARCHAR(50) NOT NULL,

    -- Message content
    message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    message_category VARCHAR(100),

    -- Original and corrected values
    original_value TEXT,
    corrected_value TEXT,
    suggestion TEXT,

    -- Confidence and scoring
    confidence_score DECIMAL(5,4) CHECK (confidence_score BETWEEN 0 AND 1),
    suggestion_confidence DECIMAL(5,4) CHECK (suggestion_confidence BETWEEN 0 AND 1),

    -- AFM compliance information
    afm_reference VARCHAR(100),
    compliance_category VARCHAR(100),

    -- Correction tracking
    correction_method VARCHAR(50),
    correction_applied BOOLEAN DEFAULT false,
    correction_feedback VARCHAR(20) CHECK (correction_feedback IN ('helpful', 'not_helpful', 'incorrect', 'accepted', 'rejected')),

    -- Context and metadata
    validation_context JSONB DEFAULT '{}',
    message_context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    feedback_at TIMESTAMP WITH TIME ZONE
);

-- Field Validation Statistics - tracks validation performance metrics
CREATE TABLE IF NOT EXISTS field_validation_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Statistics identification
    stats_id VARCHAR(255) NOT NULL UNIQUE,
    measurement_period VARCHAR(20) NOT NULL CHECK (measurement_period IN ('hourly', 'daily', 'weekly', 'monthly')),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Validation volume metrics
    total_validations INTEGER DEFAULT 0,
    successful_validations INTEGER DEFAULT 0,
    failed_validations INTEGER DEFAULT 0,
    field_validations INTEGER DEFAULT 0,
    bulk_validations INTEGER DEFAULT 0,

    -- Performance metrics
    average_processing_time DECIMAL(8,4) DEFAULT 0,
    max_processing_time DECIMAL(8,4) DEFAULT 0,
    min_processing_time DECIMAL(8,4) DEFAULT 0,
    total_processing_time DECIMAL(12,4) DEFAULT 0,

    -- Quality metrics
    average_overall_score DECIMAL(5,2) DEFAULT 0,
    average_compliance_score DECIMAL(5,2) DEFAULT 0,
    validation_accuracy DECIMAL(5,4) CHECK (validation_accuracy BETWEEN 0 AND 1),

    -- Error and issue tracking
    total_errors INTEGER DEFAULT 0,
    total_warnings INTEGER DEFAULT 0,
    total_infos INTEGER DEFAULT 0,
    false_positive_count INTEGER DEFAULT 0,

    -- Field type breakdown
    text_field_count INTEGER DEFAULT 0,
    email_field_count INTEGER DEFAULT 0,
    phone_field_count INTEGER DEFAULT 0,
    currency_field_count INTEGER DEFAULT 0,
    date_field_count INTEGER DEFAULT 0,
    bsn_field_count INTEGER DEFAULT 0,
    iban_field_count INTEGER DEFAULT 0,
    other_field_count INTEGER DEFAULT 0,

    -- Rule usage statistics
    rules_applied INTEGER DEFAULT 0,
    custom_rules_used INTEGER DEFAULT 0,
    afm_rules_triggered INTEGER DEFAULT 0,
    most_used_rules JSONB DEFAULT '[]',

    -- Correction and suggestion metrics
    suggestions_generated INTEGER DEFAULT 0,
    corrections_applied INTEGER DEFAULT 0,
    correction_success_rate DECIMAL(5,4) CHECK (correction_success_rate BETWEEN 0 AND 1),

    -- AFM compliance metrics
    afm_compliance_checks INTEGER DEFAULT 0,
    afm_violations INTEGER DEFAULT 0,
    afm_compliance_rate DECIMAL(5,4) CHECK (afm_compliance_rate BETWEEN 0 AND 1),

    -- Extended metrics
    custom_metrics JSONB DEFAULT '{}',
    breakdown_by_severity JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Validation Corrections - tracks correction suggestions and their outcomes
CREATE TABLE IF NOT EXISTS field_validation_corrections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id VARCHAR(255) REFERENCES field_validation_messages(message_id) ON DELETE CASCADE,

    -- Correction identification
    correction_id VARCHAR(255) NOT NULL UNIQUE,
    field_path VARCHAR(200) NOT NULL,
    field_type VARCHAR(50) NOT NULL,

    -- Correction details
    original_value TEXT NOT NULL,
    suggested_value TEXT NOT NULL,
    correction_type VARCHAR(50) NOT NULL CHECK (correction_type IN ('format', 'pattern', 'typo', 'validation', 'completion', 'standardization')),
    correction_method VARCHAR(100) NOT NULL,

    -- Confidence and quality metrics
    confidence_score DECIMAL(5,4) NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    quality_score DECIMAL(5,4) CHECK (quality_score BETWEEN 0 AND 1),
    similarity_score DECIMAL(5,4) CHECK (similarity_score BETWEEN 0 AND 1),

    -- Application and feedback
    was_applied BOOLEAN DEFAULT false,
    applied_at TIMESTAMP WITH TIME ZONE,
    applied_by VARCHAR(100),

    -- User feedback
    user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),
    user_feedback TEXT,
    feedback_category VARCHAR(50) CHECK (feedback_category IN ('helpful', 'not_helpful', 'incorrect', 'partial', 'excellent')),

    -- Learning and improvement data
    improvement_suggestions TEXT,
    algorithm_version VARCHAR(20),
    training_data_source VARCHAR(100),

    -- Extended metadata
    correction_context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    feedback_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AFM Compliance Checks - specific tracking for AFM compliance validations
CREATE TABLE IF NOT EXISTS afm_compliance_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255) REFERENCES field_validation_sessions(session_id) ON DELETE SET NULL,

    -- Compliance check identification
    check_id VARCHAR(255) NOT NULL UNIQUE,
    field_path VARCHAR(200) NOT NULL,
    compliance_category VARCHAR(100) NOT NULL,

    -- AFM regulation details
    afm_article VARCHAR(100) NOT NULL,
    regulation_section VARCHAR(200),
    regulation_description TEXT,

    -- Compliance assessment
    is_compliant BOOLEAN NOT NULL,
    compliance_score DECIMAL(5,2) NOT NULL CHECK (compliance_score BETWEEN 0 AND 100),
    violation_severity VARCHAR(20) CHECK (violation_severity IN ('minor', 'moderate', 'major', 'critical')),

    -- Violation details
    violation_description TEXT,
    violation_category VARCHAR(100),
    potential_penalty VARCHAR(200),
    remediation_required BOOLEAN DEFAULT false,

    -- Compliance requirements
    required_documentation JSONB DEFAULT '[]',
    required_disclosures JSONB DEFAULT '[]',
    required_actions JSONB DEFAULT '[]',

    -- Recommendations and remediation
    recommendations JSONB DEFAULT '[]',
    remediation_steps JSONB DEFAULT '[]',
    follow_up_required BOOLEAN DEFAULT false,

    -- Context and evidence
    field_value TEXT,
    supporting_data JSONB DEFAULT '{}',
    assessment_context JSONB DEFAULT '{}',

    -- Review and approval
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    approval_status VARCHAR(20) CHECK (approval_status IN ('pending', 'approved', 'rejected', 'requires_review')),
    approval_notes TEXT,

    -- Extended metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR ADVANCED FIELD VALIDATION ENGINE TABLES
-- ============================================================================

-- Field validation rules indexes
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_rule_id ON field_validation_rules(rule_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_field_path ON field_validation_rules(field_path);
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_field_type ON field_validation_rules(field_type);
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_rule_type ON field_validation_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_active ON field_validation_rules(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_severity ON field_validation_rules(severity);
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_priority ON field_validation_rules(priority);
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_afm_article ON field_validation_rules(afm_article) WHERE afm_article IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_created_by ON field_validation_rules(created_by);
CREATE INDEX IF NOT EXISTS idx_field_validation_rules_last_used ON field_validation_rules(last_used_at DESC NULLS LAST);

-- Validation sessions indexes
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_session_id ON field_validation_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_user_id ON field_validation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_type ON field_validation_sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_status ON field_validation_sessions(session_status);
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_started_at ON field_validation_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_completed_at ON field_validation_sessions(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_is_valid ON field_validation_sessions(is_valid);
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_overall_score ON field_validation_sessions(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_field_validation_sessions_compliance_score ON field_validation_sessions(compliance_score DESC);

-- Validation messages indexes
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_message_id ON field_validation_messages(message_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_session_id ON field_validation_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_rule_id ON field_validation_messages(rule_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_field_path ON field_validation_messages(field_path);
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_field_type ON field_validation_messages(field_type);
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_severity ON field_validation_messages(severity);
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_afm_reference ON field_validation_messages(afm_reference) WHERE afm_reference IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_created_at ON field_validation_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_field_validation_messages_correction_applied ON field_validation_messages(correction_applied);

-- Validation statistics indexes
CREATE INDEX IF NOT EXISTS idx_field_validation_statistics_stats_id ON field_validation_statistics(stats_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_statistics_period ON field_validation_statistics(measurement_period);
CREATE INDEX IF NOT EXISTS idx_field_validation_statistics_start ON field_validation_statistics(period_start DESC);
CREATE INDEX IF NOT EXISTS idx_field_validation_statistics_end ON field_validation_statistics(period_end DESC);
CREATE INDEX IF NOT EXISTS idx_field_validation_statistics_calculated_at ON field_validation_statistics(calculated_at DESC);

-- Validation corrections indexes
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_correction_id ON field_validation_corrections(correction_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_message_id ON field_validation_corrections(message_id);
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_field_path ON field_validation_corrections(field_path);
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_field_type ON field_validation_corrections(field_type);
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_type ON field_validation_corrections(correction_type);
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_applied ON field_validation_corrections(was_applied);
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_confidence ON field_validation_corrections(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_rating ON field_validation_corrections(user_rating DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_field_validation_corrections_generated_at ON field_validation_corrections(generated_at DESC);

-- AFM compliance checks indexes
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_check_id ON afm_compliance_checks(check_id);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_session_id ON afm_compliance_checks(session_id);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_field_path ON afm_compliance_checks(field_path);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_category ON afm_compliance_checks(compliance_category);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_article ON afm_compliance_checks(afm_article);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_compliant ON afm_compliance_checks(is_compliant);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_severity ON afm_compliance_checks(violation_severity);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_score ON afm_compliance_checks(compliance_score DESC);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_checked_at ON afm_compliance_checks(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_remediation ON afm_compliance_checks(remediation_required) WHERE remediation_required = true;
CREATE INDEX IF NOT EXISTS idx_afm_compliance_checks_follow_up ON afm_compliance_checks(follow_up_required) WHERE follow_up_required = true;

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_validation_rules_field_active ON field_validation_rules(field_path, is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_validation_rules_type_priority ON field_validation_rules(field_type, priority);
CREATE INDEX IF NOT EXISTS idx_validation_sessions_user_status ON field_validation_sessions(user_id, session_status);
CREATE INDEX IF NOT EXISTS idx_validation_messages_session_severity ON field_validation_messages(session_id, severity);
CREATE INDEX IF NOT EXISTS idx_validation_corrections_field_applied ON field_validation_corrections(field_path, was_applied);
CREATE INDEX IF NOT EXISTS idx_afm_checks_compliant_severity ON afm_compliance_checks(is_compliant, violation_severity);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_validation_rules_name_description_fts ON field_validation_rules USING gin(to_tsvector('english', rule_name || ' ' || COALESCE(description, '')));
CREATE INDEX IF NOT EXISTS idx_validation_messages_message_fts ON field_validation_messages USING gin(to_tsvector('english', message));
CREATE INDEX IF NOT EXISTS idx_afm_compliance_description_fts ON afm_compliance_checks USING gin(to_tsvector('english', COALESCE(violation_description, '') || ' ' || COALESCE(regulation_description, '')));

-- JSONB indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_validation_rules_parameters_gin ON field_validation_rules USING gin(parameters);
CREATE INDEX IF NOT EXISTS idx_validation_rules_conditions_gin ON field_validation_rules USING gin(conditions);
CREATE INDEX IF NOT EXISTS idx_validation_rules_dependencies_gin ON field_validation_rules USING gin(dependencies);
CREATE INDEX IF NOT EXISTS idx_validation_sessions_config_gin ON field_validation_sessions USING gin(validation_config);
CREATE INDEX IF NOT EXISTS idx_validation_sessions_scope_gin ON field_validation_sessions USING gin(validation_scope);
CREATE INDEX IF NOT EXISTS idx_validation_messages_context_gin ON field_validation_messages USING gin(validation_context);
CREATE INDEX IF NOT EXISTS idx_validation_corrections_context_gin ON field_validation_corrections USING gin(correction_context);
CREATE INDEX IF NOT EXISTS idx_afm_checks_requirements_gin ON afm_compliance_checks USING gin(required_documentation);
CREATE INDEX IF NOT EXISTS idx_afm_checks_recommendations_gin ON afm_compliance_checks USING gin(recommendations);

-- ============================================================================
-- TRIGGERS FOR ADVANCED FIELD VALIDATION ENGINE TABLES
-- ============================================================================

-- Update timestamps
CREATE TRIGGER update_field_validation_rules_updated_at
    BEFORE UPDATE ON field_validation_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_field_validation_sessions_updated_at
    BEFORE UPDATE ON field_validation_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_afm_compliance_checks_updated_at
    BEFORE UPDATE ON afm_compliance_checks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update rule usage statistics
CREATE OR REPLACE FUNCTION update_rule_usage_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update rule usage statistics when a message is created
    IF TG_OP = 'INSERT' AND NEW.rule_id IS NOT NULL THEN
        UPDATE field_validation_rules
        SET
            usage_count = usage_count + 1,
            last_used_at = NEW.created_at,
            updated_at = CURRENT_TIMESTAMP
        WHERE rule_id = NEW.rule_id;
        
        -- Update success count based on severity
        IF NEW.severity NOT IN ('error', 'critical') THEN
            UPDATE field_validation_rules
            SET success_count = success_count + 1
            WHERE rule_id = NEW.rule_id;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_rule_usage_stats_trigger
    AFTER INSERT ON field_validation_messages
    FOR EACH ROW EXECUTE FUNCTION update_rule_usage_stats();

-- Update false positive counts when feedback is provided
CREATE OR REPLACE FUNCTION update_rule_false_positive_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update false positive count when correction feedback indicates it
    IF TG_OP = 'UPDATE' AND OLD.correction_feedback != NEW.correction_feedback THEN
        IF NEW.correction_feedback IN ('not_helpful', 'incorrect') THEN
            UPDATE field_validation_rules
            SET 
                false_positive_count = false_positive_count + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE rule_id = (
                SELECT rule_id FROM field_validation_messages 
                WHERE message_id = NEW.message_id
            );
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_rule_false_positive_stats_trigger
    AFTER UPDATE ON field_validation_messages
    FOR EACH ROW EXECUTE FUNCTION update_rule_false_positive_stats();

-- Update session completion status and metrics
CREATE OR REPLACE FUNCTION update_validation_session_completion()
RETURNS TRIGGER AS $$
BEGIN
    -- Auto-update session completion when status changes to completed
    IF TG_OP = 'UPDATE' AND OLD.session_status != 'completed' AND NEW.session_status = 'completed' THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
        
        -- Calculate processing time if not set
        IF NEW.processing_time = 0 AND NEW.started_at IS NOT NULL THEN
            NEW.processing_time = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - NEW.started_at));
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_validation_session_completion_trigger
    BEFORE UPDATE ON field_validation_sessions
    FOR EACH ROW EXECUTE FUNCTION update_validation_session_completion();

-- Update correction application tracking
CREATE OR REPLACE FUNCTION track_correction_application()
RETURNS TRIGGER AS $$
BEGIN
    -- Update correction tracking when was_applied changes from false to true
    IF TG_OP = 'UPDATE' AND OLD.was_applied = false AND NEW.was_applied = true THEN
        NEW.applied_at = CURRENT_TIMESTAMP;
        
        -- Update the related validation message
        UPDATE field_validation_messages
        SET 
            correction_applied = true,
            corrected_value = NEW.suggested_value
        WHERE message_id = NEW.message_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER track_correction_application_trigger
    BEFORE UPDATE ON field_validation_corrections
    FOR EACH ROW EXECUTE FUNCTION track_correction_application();

-- ============================================================================
-- AGENT PERFORMANCE METRICS DASHBOARD TABLES
-- ============================================================================
-- These tables support the Agent Performance Metrics Dashboard with comprehensive
-- analytics, success rates tracking, and optimization recommendations

-- Agent Performance Metrics - stores performance metrics data for agents
CREATE TABLE IF NOT EXISTS agent_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Agent identification
    agent_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(255),
    agent_type VARCHAR(100),
    team_id VARCHAR(255),
    
    -- Metrics data and scoring
    metric_data JSONB NOT NULL DEFAULT '{}',
    time_period VARCHAR(50) NOT NULL CHECK (time_period IN ('realtime', 'hourly', 'daily', 'weekly', 'monthly')),
    overall_score DECIMAL(5,2) DEFAULT 0 CHECK (overall_score BETWEEN 0 AND 100),
    
    -- Performance dimensions
    success_rate DECIMAL(5,4) DEFAULT 0 CHECK (success_rate BETWEEN 0 AND 1),
    quality_score DECIMAL(5,4) DEFAULT 0 CHECK (quality_score BETWEEN 0 AND 1),
    efficiency_score DECIMAL(5,4) DEFAULT 0 CHECK (efficiency_score BETWEEN 0 AND 1),
    compliance_score DECIMAL(5,4) DEFAULT 0 CHECK (compliance_score BETWEEN 0 AND 1),
    user_satisfaction_score DECIMAL(5,4) DEFAULT 0 CHECK (user_satisfaction_score BETWEEN 0 AND 1),
    
    -- Task and workload metrics
    total_tasks INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    avg_processing_time DECIMAL(8,4) DEFAULT 0,
    throughput DECIMAL(8,4) DEFAULT 0,
    
    -- Quality and accuracy metrics
    avg_quality_score DECIMAL(5,4) DEFAULT 0,
    accuracy_rate DECIMAL(5,4) DEFAULT 0,
    error_rate DECIMAL(5,4) DEFAULT 0,
    
    -- Resource utilization metrics
    avg_cpu_usage DECIMAL(5,2) DEFAULT 0,
    avg_memory_usage DECIMAL(5,2) DEFAULT 0,
    avg_response_time DECIMAL(8,4) DEFAULT 0,
    
    -- Compliance and risk metrics
    compliance_rate DECIMAL(5,4) DEFAULT 0,
    critical_violations INTEGER DEFAULT 0,
    major_violations INTEGER DEFAULT 0,
    compliance_risk_score DECIMAL(5,4) DEFAULT 0,
    
    -- User interaction metrics
    total_interactions INTEGER DEFAULT 0,
    positive_feedback_rate DECIMAL(5,4) DEFAULT 0,
    negative_feedback_rate DECIMAL(5,4) DEFAULT 0,
    avg_interaction_time DECIMAL(8,4) DEFAULT 0,
    
    -- Derived metrics
    productivity_score DECIMAL(5,4) DEFAULT 0,
    reliability_score DECIMAL(5,4) DEFAULT 0,
    consistency_score DECIMAL(5,4) DEFAULT 0,
    user_experience_score DECIMAL(5,4) DEFAULT 0,
    
    -- Context and metadata
    workload_context JSONB DEFAULT '{}',
    performance_context JSONB DEFAULT '{}',
    collection_metadata JSONB DEFAULT '{}',
    
    -- Processing information
    collection_method VARCHAR(100) DEFAULT 'automated',
    processing_time DECIMAL(8,4) DEFAULT 0,
    data_quality_score DECIMAL(5,4) DEFAULT 1.0,
    
    -- Timestamps
    measurement_start TIMESTAMP WITH TIME ZONE,
    measurement_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Performance Analyses - stores comprehensive performance analysis results
CREATE TABLE IF NOT EXISTS agent_performance_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Analysis identification
    analysis_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255) NOT NULL,
    analysis_type VARCHAR(100) NOT NULL CHECK (analysis_type IN ('basic', 'comprehensive', 'comparative', 'predictive', 'diagnostic')),
    analysis_period VARCHAR(50) NOT NULL,
    
    -- Analysis scope and configuration
    analysis_scope JSONB DEFAULT '{}',
    focus_areas JSONB DEFAULT '[]',
    configuration JSONB DEFAULT '{}',
    
    -- Analysis results
    analysis_data JSONB NOT NULL DEFAULT '{}',
    overall_grade VARCHAR(20) DEFAULT 'N/A',
    performance_summary JSONB DEFAULT '{}',
    
    -- Statistical analysis results
    statistical_summary JSONB DEFAULT '{}',
    trend_analysis JSONB DEFAULT '{}',
    pattern_analysis JSONB DEFAULT '{}',
    benchmark_comparison JSONB DEFAULT '{}',
    
    -- Insights and findings
    key_insights JSONB DEFAULT '[]',
    strengths JSONB DEFAULT '[]',
    improvement_areas JSONB DEFAULT '[]',
    risk_factors JSONB DEFAULT '[]',
    
    -- Recommendations
    recommendations_count INTEGER DEFAULT 0,
    high_priority_recommendations INTEGER DEFAULT 0,
    implementation_recommendations JSONB DEFAULT '[]',
    
    -- Forecasting results
    forecasts JSONB DEFAULT '{}',
    predicted_performance DECIMAL(5,4),
    confidence_level DECIMAL(5,4) DEFAULT 0,
    forecast_horizon_days INTEGER DEFAULT 30,
    
    -- Comparative analysis
    peer_comparison JSONB DEFAULT '{}',
    ranking_position INTEGER,
    percentile_rank DECIMAL(5,4),
    
    -- Quality and validation
    data_quality_score DECIMAL(5,4) DEFAULT 1.0,
    analysis_confidence DECIMAL(5,4) DEFAULT 0,
    validation_passed BOOLEAN DEFAULT true,
    
    -- Processing metadata
    processing_time DECIMAL(8,4) DEFAULT 0,
    data_points_analyzed INTEGER DEFAULT 0,
    model_versions_used JSONB DEFAULT '{}',
    
    -- Extended metadata
    methodology JSONB DEFAULT '{}',
    assumptions JSONB DEFAULT '[]',
    limitations JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    analysis_start TIMESTAMP WITH TIME ZONE,
    analysis_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Performance Logs - tracks performance monitoring activities and events
CREATE TABLE IF NOT EXISTS agent_performance_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Log identification
    log_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    
    -- Activity details
    action VARCHAR(100) NOT NULL,
    action_category VARCHAR(50) DEFAULT 'performance',
    activity_type VARCHAR(100),
    
    -- Request and parameters
    parameters JSONB DEFAULT '{}',
    request_context JSONB DEFAULT '{}',
    user_context JSONB DEFAULT '{}',
    
    -- Results and outcomes
    result_status VARCHAR(20) DEFAULT 'success' CHECK (result_status IN ('success', 'failure', 'partial', 'timeout', 'cancelled')),
    result_summary JSONB DEFAULT '{}',
    result_data JSONB DEFAULT '{}',
    
    -- Error and issue tracking
    error_code VARCHAR(100),
    error_message TEXT,
    error_details JSONB DEFAULT '{}',
    
    -- Performance tracking
    processing_time DECIMAL(8,4) DEFAULT 0,
    cpu_usage DECIMAL(5,2) DEFAULT 0,
    memory_usage DECIMAL(5,2) DEFAULT 0,
    
    -- Impact and metrics
    impact_score DECIMAL(5,4) DEFAULT 0,
    quality_impact DECIMAL(5,4) DEFAULT 0,
    performance_impact DECIMAL(5,4) DEFAULT 0,
    
    -- Tracing and correlation
    correlation_id VARCHAR(255),
    parent_activity_id VARCHAR(255),
    trace_context JSONB DEFAULT '{}',
    
    -- Extended metadata
    client_info JSONB DEFAULT '{}',
    system_info JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    activity_start TIMESTAMP WITH TIME ZONE,
    activity_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Optimization Recommendations - stores optimization recommendations for agents
CREATE TABLE IF NOT EXISTS agent_optimization_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Recommendation identification
    recommendation_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255) NOT NULL,
    analysis_id VARCHAR(255),
    
    -- Recommendation details
    recommendation_type VARCHAR(100) NOT NULL CHECK (recommendation_type IN ('workflow', 'resource', 'training', 'automation', 'quality', 'efficiency', 'compliance')),
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent', 'critical')),
    
    -- Content and description
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    detailed_explanation TEXT,
    
    -- Impact and effort analysis
    expected_impact DECIMAL(5,4) DEFAULT 0 CHECK (expected_impact BETWEEN 0 AND 1),
    implementation_effort VARCHAR(20) CHECK (implementation_effort IN ('low', 'medium', 'high', 'very_high')),
    estimated_timeline VARCHAR(100),
    
    -- Success criteria and metrics
    success_metrics JSONB DEFAULT '[]',
    kpi_targets JSONB DEFAULT '{}',
    measurement_criteria JSONB DEFAULT '[]',
    
    -- Implementation details
    prerequisites JSONB DEFAULT '[]',
    resources_required JSONB DEFAULT '[]',
    implementation_steps JSONB DEFAULT '[]',
    potential_risks JSONB DEFAULT '[]',
    mitigation_strategies JSONB DEFAULT '[]',
    
    -- Financial analysis
    estimated_cost DECIMAL(12,2) DEFAULT 0,
    estimated_savings DECIMAL(12,2) DEFAULT 0,
    estimated_roi DECIMAL(8,4) DEFAULT 0,
    payback_period_months INTEGER,
    
    -- Confidence and validation
    confidence_level DECIMAL(5,4) DEFAULT 0 CHECK (confidence_level BETWEEN 0 AND 1),
    evidence_quality VARCHAR(20) DEFAULT 'medium',
    supporting_data JSONB DEFAULT '{}',
    
    -- Recommendation status and tracking
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'in_progress', 'completed', 'cancelled')),
    approval_status VARCHAR(20) DEFAULT 'pending_review',
    implementation_status VARCHAR(20) DEFAULT 'not_started',
    
    -- Review and feedback
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    review_notes TEXT,
    feedback_score INTEGER CHECK (feedback_score BETWEEN 1 AND 5),
    
    -- Dependencies and relationships
    depends_on JSONB DEFAULT '[]',
    related_recommendations JSONB DEFAULT '[]',
    conflicts_with JSONB DEFAULT '[]',
    
    -- Extended metadata
    generation_method VARCHAR(100) DEFAULT 'automated',
    algorithm_version VARCHAR(50),
    model_confidence DECIMAL(5,4) DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    valid_from TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Performance Alerts - stores performance alerts and notifications
CREATE TABLE IF NOT EXISTS agent_performance_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Alert identification
    alert_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255) NOT NULL,
    alert_rule_id VARCHAR(255),
    
    -- Alert details
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    
    -- Metrics and thresholds
    metric_name VARCHAR(100) NOT NULL,
    metric_values JSONB NOT NULL DEFAULT '{}',
    threshold_values JSONB DEFAULT '{}',
    deviation_percentage DECIMAL(8,4) DEFAULT 0,
    
    -- Alert status and lifecycle
    alert_status VARCHAR(20) DEFAULT 'active' CHECK (alert_status IN ('active', 'acknowledged', 'resolved', 'suppressed', 'expired')),
    first_occurrence TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_occurrence TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    occurrence_count INTEGER DEFAULT 1,
    
    -- Response and handling
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledgment_notes TEXT,
    
    -- Resolution tracking
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_method VARCHAR(100),
    resolution_notes TEXT,
    auto_resolved BOOLEAN DEFAULT false,
    
    -- Impact assessment
    business_impact VARCHAR(20) CHECK (business_impact IN ('low', 'medium', 'high', 'critical')),
    affected_systems JSONB DEFAULT '[]',
    impact_description TEXT,
    
    -- Notification tracking
    notification_sent BOOLEAN DEFAULT false,
    notification_channels JSONB DEFAULT '[]',
    escalation_level INTEGER DEFAULT 0,
    escalated_at TIMESTAMP WITH TIME ZONE,
    
    -- Context and metadata
    alert_context JSONB DEFAULT '{}',
    system_state JSONB DEFAULT '{}',
    related_alerts JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Agent Dashboard Generations - tracks dashboard generation activities
CREATE TABLE IF NOT EXISTS agent_dashboard_generations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Generation identification
    generation_id VARCHAR(255) NOT NULL UNIQUE,
    dashboard_type VARCHAR(100) NOT NULL,
    request_source VARCHAR(100) DEFAULT 'user',
    
    -- Dashboard scope
    agent_ids JSONB DEFAULT '[]',
    agent_count INTEGER DEFAULT 0,
    time_range VARCHAR(100) NOT NULL,
    filters_applied JSONB DEFAULT '{}',
    
    -- Generation details
    dashboard_config JSONB DEFAULT '{}',
    visualization_types JSONB DEFAULT '[]',
    include_forecasts BOOLEAN DEFAULT false,
    include_benchmarks BOOLEAN DEFAULT true,
    include_recommendations BOOLEAN DEFAULT false,
    
    -- Processing metrics
    generation_time DECIMAL(8,4) DEFAULT 0,
    data_points_processed INTEGER DEFAULT 0,
    visualizations_generated INTEGER DEFAULT 0,
    
    -- Caching and storage
    cached BOOLEAN DEFAULT false,
    cache_key VARCHAR(255),
    cached_until TIMESTAMP WITH TIME ZONE,
    storage_location VARCHAR(500),
    
    -- Quality and validation
    generation_quality DECIMAL(5,4) DEFAULT 1.0,
    data_completeness DECIMAL(5,4) DEFAULT 1.0,
    validation_passed BOOLEAN DEFAULT true,
    
    -- Usage and access
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE,
    shared_with JSONB DEFAULT '[]',
    
    -- Extended metadata
    client_info JSONB DEFAULT '{}',
    request_metadata JSONB DEFAULT '{}',
    generation_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    requested_at TIMESTAMP WITH TIME ZONE,
    generation_start TIMESTAMP WITH TIME ZONE,
    generation_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Recommendation Implementations - tracks recommendation implementation status
CREATE TABLE IF NOT EXISTS recommendation_implementations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Implementation identification
    implementation_id VARCHAR(255) NOT NULL UNIQUE,
    recommendation_id VARCHAR(255) NOT NULL REFERENCES agent_optimization_recommendations(recommendation_id) ON DELETE CASCADE,
    agent_id VARCHAR(255) NOT NULL,
    
    -- Implementation details
    implementation_status VARCHAR(20) DEFAULT 'not_started' CHECK (implementation_status IN ('not_started', 'planned', 'in_progress', 'testing', 'completed', 'cancelled', 'failed')),
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage BETWEEN 0 AND 100),
    
    -- Timeline tracking
    planned_start_date DATE,
    actual_start_date DATE,
    planned_completion_date DATE,
    actual_completion_date DATE,
    
    -- Implementation team and resources
    implemented_by VARCHAR(100),
    implementation_team JSONB DEFAULT '[]',
    resources_allocated JSONB DEFAULT '{}',
    budget_allocated DECIMAL(12,2) DEFAULT 0,
    budget_spent DECIMAL(12,2) DEFAULT 0,
    
    -- Progress and milestones
    implementation_milestones JSONB DEFAULT '[]',
    completed_milestones INTEGER DEFAULT 0,
    total_milestones INTEGER DEFAULT 0,
    current_phase VARCHAR(100),
    
    -- Results and outcomes
    results_achieved JSONB DEFAULT '{}',
    performance_impact JSONB DEFAULT '{}',
    measured_benefits JSONB DEFAULT '{}',
    roi_realized DECIMAL(8,4) DEFAULT 0,
    
    -- Issues and challenges
    implementation_challenges JSONB DEFAULT '[]',
    risks_encountered JSONB DEFAULT '[]',
    mitigation_actions JSONB DEFAULT '[]',
    
    -- Quality and validation
    implementation_quality DECIMAL(5,4) DEFAULT 0,
    testing_results JSONB DEFAULT '{}',
    validation_status VARCHAR(20) DEFAULT 'pending',
    
    -- Feedback and assessment
    stakeholder_feedback JSONB DEFAULT '{}',
    success_rating INTEGER CHECK (success_rating BETWEEN 1 AND 5),
    lessons_learned TEXT,
    
    -- Documentation and reporting
    implementation_notes TEXT,
    progress_reports JSONB DEFAULT '[]',
    final_report TEXT,
    documentation_links JSONB DEFAULT '[]',
    
    -- Extended metadata
    implementation_method VARCHAR(100),
    tools_used JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_progress_update TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Task Logs - logs individual agent tasks for performance tracking
CREATE TABLE IF NOT EXISTS agent_task_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Task identification
    task_id VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    task_category VARCHAR(100),
    
    -- Task details
    task_name VARCHAR(500),
    task_description TEXT,
    task_priority VARCHAR(20) CHECK (task_priority IN ('low', 'normal', 'high', 'urgent', 'critical')),
    
    -- Task execution
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout')),
    processing_time DECIMAL(8,4) DEFAULT 0,
    complexity_score DECIMAL(5,4) DEFAULT 0,
    
    -- Task context
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    context_data JSONB DEFAULT '{}',
    
    -- Performance metrics
    cpu_usage DECIMAL(5,2) DEFAULT 0,
    memory_usage DECIMAL(5,2) DEFAULT 0,
    io_operations INTEGER DEFAULT 0,
    network_requests INTEGER DEFAULT 0,
    
    -- Quality assessment
    quality_score DECIMAL(5,4) DEFAULT 0,
    accuracy_score DECIMAL(5,4) DEFAULT 0,
    completeness_score DECIMAL(5,4) DEFAULT 0,
    
    -- Error tracking
    error_code VARCHAR(100),
    error_message TEXT,
    error_category VARCHAR(100),
    
    -- Dependencies and relationships
    parent_task_id VARCHAR(255),
    depends_on JSONB DEFAULT '[]',
    spawned_tasks JSONB DEFAULT '[]',
    
    -- Timestamps
    scheduled_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Quality Assessments - stores quality assessment results for agents
CREATE TABLE IF NOT EXISTS agent_quality_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Assessment identification
    assessment_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255),
    
    -- Assessment details
    assessment_type VARCHAR(100) NOT NULL,
    assessment_method VARCHAR(100) NOT NULL,
    assessor VARCHAR(100),
    
    -- Quality metrics
    quality_score DECIMAL(5,4) NOT NULL CHECK (quality_score BETWEEN 0 AND 1),
    accuracy_score DECIMAL(5,4) DEFAULT 0,
    completeness_score DECIMAL(5,4) DEFAULT 0,
    consistency_score DECIMAL(5,4) DEFAULT 0,
    timeliness_score DECIMAL(5,4) DEFAULT 0,
    
    -- Detailed assessment results
    assessment_criteria JSONB DEFAULT '{}',
    scoring_breakdown JSONB DEFAULT '{}',
    quality_dimensions JSONB DEFAULT '{}',
    
    -- Findings and feedback
    strengths JSONB DEFAULT '[]',
    weaknesses JSONB DEFAULT '[]',
    improvement_suggestions JSONB DEFAULT '[]',
    
    -- Comparison and benchmarking
    benchmark_scores JSONB DEFAULT '{}',
    peer_comparison JSONB DEFAULT '{}',
    historical_comparison JSONB DEFAULT '{}',
    
    -- Context and metadata
    assessment_context JSONB DEFAULT '{}',
    sample_size INTEGER DEFAULT 1,
    confidence_level DECIMAL(5,4) DEFAULT 0,
    
    -- Timestamps
    assessment_start TIMESTAMP WITH TIME ZONE,
    assessment_end TIMESTAMP WITH TIME ZONE,
    assessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Validations - stores validation results for agent outputs
CREATE TABLE IF NOT EXISTS agent_validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Validation identification
    validation_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255),
    
    -- Validation details
    validation_type VARCHAR(100) NOT NULL,
    validation_method VARCHAR(100) NOT NULL,
    validator VARCHAR(100),
    
    -- Validation results
    is_valid BOOLEAN NOT NULL,
    accuracy_score DECIMAL(5,4) DEFAULT 0 CHECK (accuracy_score BETWEEN 0 AND 1),
    confidence_score DECIMAL(5,4) DEFAULT 0 CHECK (confidence_score BETWEEN 0 AND 1),
    
    -- Validation criteria and results
    validation_criteria JSONB DEFAULT '{}',
    validation_results JSONB DEFAULT '{}',
    failed_criteria JSONB DEFAULT '[]',
    
    -- Error and issue tracking
    validation_errors JSONB DEFAULT '[]',
    warning_messages JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    
    -- Context and metadata
    validation_context JSONB DEFAULT '{}',
    reference_data JSONB DEFAULT '{}',
    
    -- Timestamps
    validated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Resource Metrics - stores resource utilization metrics for agents
CREATE TABLE IF NOT EXISTS agent_resource_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Metrics identification
    metric_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255) NOT NULL,
    
    -- Resource utilization
    cpu_usage DECIMAL(5,2) DEFAULT 0 CHECK (cpu_usage BETWEEN 0 AND 100),
    memory_usage DECIMAL(5,2) DEFAULT 0 CHECK (memory_usage BETWEEN 0 AND 100),
    disk_usage DECIMAL(5,2) DEFAULT 0,
    network_usage DECIMAL(10,2) DEFAULT 0,
    
    -- Performance metrics
    response_time DECIMAL(8,4) DEFAULT 0,
    throughput DECIMAL(8,4) DEFAULT 0,
    concurrent_requests INTEGER DEFAULT 0,
    queue_size INTEGER DEFAULT 0,
    
    -- System metrics
    system_load DECIMAL(5,2) DEFAULT 0,
    process_count INTEGER DEFAULT 0,
    thread_count INTEGER DEFAULT 0,
    connection_count INTEGER DEFAULT 0,
    
    -- Application metrics
    cache_hit_rate DECIMAL(5,4) DEFAULT 0,
    error_rate DECIMAL(5,4) DEFAULT 0,
    success_rate DECIMAL(5,4) DEFAULT 1,
    
    -- Context and metadata
    measurement_context JSONB DEFAULT '{}',
    system_info JSONB DEFAULT '{}',
    
    -- Timestamps
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    measurement_start TIMESTAMP WITH TIME ZONE,
    measurement_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Compliance Checks - stores compliance checking results
CREATE TABLE IF NOT EXISTS agent_compliance_checks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Check identification
    check_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(255),
    
    -- Compliance details
    compliance_type VARCHAR(100) NOT NULL,
    regulation VARCHAR(100),
    check_category VARCHAR(100),
    
    -- Check results
    is_compliant BOOLEAN NOT NULL,
    compliance_score DECIMAL(5,2) DEFAULT 0 CHECK (compliance_score BETWEEN 0 AND 100),
    violation_severity VARCHAR(20) CHECK (violation_severity IN ('minor', 'moderate', 'major', 'critical')),
    
    -- Violation details
    violations_found JSONB DEFAULT '[]',
    violation_count INTEGER DEFAULT 0,
    violation_description TEXT,
    
    -- Remediation
    remediation_required BOOLEAN DEFAULT false,
    remediation_actions JSONB DEFAULT '[]',
    remediation_timeline VARCHAR(100),
    
    -- Context and documentation
    check_context JSONB DEFAULT '{}',
    supporting_evidence JSONB DEFAULT '{}',
    documentation_references JSONB DEFAULT '[]',
    
    -- Timestamps
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent User Interactions - stores user interaction and feedback data
CREATE TABLE IF NOT EXISTS agent_user_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Interaction identification
    interaction_id VARCHAR(255) NOT NULL UNIQUE,
    agent_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    
    -- Interaction details
    interaction_type VARCHAR(100) NOT NULL,
    interaction_category VARCHAR(100),
    interaction_channel VARCHAR(100),
    
    -- User satisfaction
    user_satisfaction_score INTEGER CHECK (user_satisfaction_score BETWEEN 1 AND 5),
    satisfaction_category VARCHAR(20) CHECK (satisfaction_category IN ('very_poor', 'poor', 'neutral', 'good', 'excellent')),
    
    -- Interaction metrics
    interaction_duration DECIMAL(8,4) DEFAULT 0,
    response_count INTEGER DEFAULT 0,
    resolution_achieved BOOLEAN DEFAULT false,
    escalation_required BOOLEAN DEFAULT false,
    
    -- Feedback and ratings
    user_feedback TEXT,
    feedback_category VARCHAR(100),
    thumbs_up BOOLEAN DEFAULT false,
    thumbs_down BOOLEAN DEFAULT false,
    
    -- Context and metadata
    interaction_context JSONB DEFAULT '{}',
    user_context JSONB DEFAULT '{}',
    
    -- Timestamps
    interaction_start TIMESTAMP WITH TIME ZONE,
    interaction_end TIMESTAMP WITH TIME ZONE,
    interaction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR AGENT PERFORMANCE METRICS TABLES
-- ============================================================================

-- Agent Performance Metrics indexes
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_agent_id ON agent_performance_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_time_period ON agent_performance_metrics(time_period);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_overall_score ON agent_performance_metrics(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_created_at ON agent_performance_metrics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_measurement_start ON agent_performance_metrics(measurement_start DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_agent_time ON agent_performance_metrics(agent_id, time_period, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_success_rate ON agent_performance_metrics(success_rate DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_quality_score ON agent_performance_metrics(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_efficiency_score ON agent_performance_metrics(efficiency_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_compliance_score ON agent_performance_metrics(compliance_score DESC);

-- Agent Performance Analyses indexes
CREATE INDEX IF NOT EXISTS idx_agent_performance_analyses_analysis_id ON agent_performance_analyses(analysis_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_analyses_agent_id ON agent_performance_analyses(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_analyses_analysis_type ON agent_performance_analyses(analysis_type);
CREATE INDEX IF NOT EXISTS idx_agent_performance_analyses_analysis_period ON agent_performance_analyses(analysis_period);
CREATE INDEX IF NOT EXISTS idx_agent_performance_analyses_overall_grade ON agent_performance_analyses(overall_grade);
CREATE INDEX IF NOT EXISTS idx_agent_performance_analyses_created_at ON agent_performance_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_analyses_agent_type ON agent_performance_analyses(agent_id, analysis_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_analyses_recommendations_count ON agent_performance_analyses(recommendations_count DESC);

-- Agent Performance Logs indexes
CREATE INDEX IF NOT EXISTS idx_agent_performance_logs_log_id ON agent_performance_logs(log_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_logs_agent_id ON agent_performance_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_logs_action ON agent_performance_logs(action);
CREATE INDEX IF NOT EXISTS idx_agent_performance_logs_result_status ON agent_performance_logs(result_status);
CREATE INDEX IF NOT EXISTS idx_agent_performance_logs_created_at ON agent_performance_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_logs_agent_action ON agent_performance_logs(agent_id, action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_logs_processing_time ON agent_performance_logs(processing_time DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_logs_correlation_id ON agent_performance_logs(correlation_id);

-- Agent Optimization Recommendations indexes
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_recommendation_id ON agent_optimization_recommendations(recommendation_id);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_agent_id ON agent_optimization_recommendations(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_type ON agent_optimization_recommendations(recommendation_type);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_priority ON agent_optimization_recommendations(priority);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_status ON agent_optimization_recommendations(status);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_expected_impact ON agent_optimization_recommendations(expected_impact DESC);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_created_at ON agent_optimization_recommendations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_agent_priority ON agent_optimization_recommendations(agent_id, priority, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_confidence_level ON agent_optimization_recommendations(confidence_level DESC);
CREATE INDEX IF NOT EXISTS idx_agent_optimization_recommendations_estimated_roi ON agent_optimization_recommendations(estimated_roi DESC);

-- Agent Performance Alerts indexes
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_alert_id ON agent_performance_alerts(alert_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_agent_id ON agent_performance_alerts(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_alert_type ON agent_performance_alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_severity ON agent_performance_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_alert_status ON agent_performance_alerts(alert_status);
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_created_at ON agent_performance_alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_first_occurrence ON agent_performance_alerts(first_occurrence DESC);
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_agent_severity ON agent_performance_alerts(agent_id, severity, alert_status);
CREATE INDEX IF NOT EXISTS idx_agent_performance_alerts_active ON agent_performance_alerts(alert_status) WHERE alert_status = 'active';

-- Agent Dashboard Generations indexes
CREATE INDEX IF NOT EXISTS idx_agent_dashboard_generations_generation_id ON agent_dashboard_generations(generation_id);
CREATE INDEX IF NOT EXISTS idx_agent_dashboard_generations_dashboard_type ON agent_dashboard_generations(dashboard_type);
CREATE INDEX IF NOT EXISTS idx_agent_dashboard_generations_user_id ON agent_dashboard_generations(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_dashboard_generations_created_at ON agent_dashboard_generations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_dashboard_generations_cached_until ON agent_dashboard_generations(cached_until);
CREATE INDEX IF NOT EXISTS idx_agent_dashboard_generations_cache_key ON agent_dashboard_generations(cache_key);
CREATE INDEX IF NOT EXISTS idx_agent_dashboard_generations_generation_time ON agent_dashboard_generations(generation_time);

-- Recommendation Implementations indexes
CREATE INDEX IF NOT EXISTS idx_recommendation_implementations_implementation_id ON recommendation_implementations(implementation_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_implementations_recommendation_id ON recommendation_implementations(recommendation_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_implementations_agent_id ON recommendation_implementations(agent_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_implementations_status ON recommendation_implementations(implementation_status);
CREATE INDEX IF NOT EXISTS idx_recommendation_implementations_progress ON recommendation_implementations(progress_percentage DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_implementations_created_at ON recommendation_implementations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_implementations_agent_status ON recommendation_implementations(agent_id, implementation_status);

-- Agent Task Logs indexes
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_task_id ON agent_task_logs(task_id);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_agent_id ON agent_task_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_task_type ON agent_task_logs(task_type);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_status ON agent_task_logs(status);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_created_at ON agent_task_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_started_at ON agent_task_logs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_completed_at ON agent_task_logs(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_agent_status ON agent_task_logs(agent_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_processing_time ON agent_task_logs(processing_time DESC);
CREATE INDEX IF NOT EXISTS idx_agent_task_logs_quality_score ON agent_task_logs(quality_score DESC);

-- Agent Quality Assessments indexes
CREATE INDEX IF NOT EXISTS idx_agent_quality_assessments_assessment_id ON agent_quality_assessments(assessment_id);
CREATE INDEX IF NOT EXISTS idx_agent_quality_assessments_agent_id ON agent_quality_assessments(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_quality_assessments_quality_score ON agent_quality_assessments(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_quality_assessments_assessed_at ON agent_quality_assessments(assessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_quality_assessments_agent_assessed ON agent_quality_assessments(agent_id, assessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_quality_assessments_assessment_type ON agent_quality_assessments(assessment_type);

-- Agent Validations indexes
CREATE INDEX IF NOT EXISTS idx_agent_validations_validation_id ON agent_validations(validation_id);
CREATE INDEX IF NOT EXISTS idx_agent_validations_agent_id ON agent_validations(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_validations_is_valid ON agent_validations(is_valid);
CREATE INDEX IF NOT EXISTS idx_agent_validations_accuracy_score ON agent_validations(accuracy_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_validations_validated_at ON agent_validations(validated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_validations_agent_validated ON agent_validations(agent_id, validated_at DESC);

-- Agent Resource Metrics indexes
CREATE INDEX IF NOT EXISTS idx_agent_resource_metrics_metric_id ON agent_resource_metrics(metric_id);
CREATE INDEX IF NOT EXISTS idx_agent_resource_metrics_agent_id ON agent_resource_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_resource_metrics_recorded_at ON agent_resource_metrics(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_resource_metrics_cpu_usage ON agent_resource_metrics(cpu_usage DESC);
CREATE INDEX IF NOT EXISTS idx_agent_resource_metrics_memory_usage ON agent_resource_metrics(memory_usage DESC);
CREATE INDEX IF NOT EXISTS idx_agent_resource_metrics_response_time ON agent_resource_metrics(response_time DESC);
CREATE INDEX IF NOT EXISTS idx_agent_resource_metrics_agent_recorded ON agent_resource_metrics(agent_id, recorded_at DESC);

-- Agent Compliance Checks indexes
CREATE INDEX IF NOT EXISTS idx_agent_compliance_checks_check_id ON agent_compliance_checks(check_id);
CREATE INDEX IF NOT EXISTS idx_agent_compliance_checks_agent_id ON agent_compliance_checks(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_compliance_checks_is_compliant ON agent_compliance_checks(is_compliant);
CREATE INDEX IF NOT EXISTS idx_agent_compliance_checks_compliance_score ON agent_compliance_checks(compliance_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_compliance_checks_violation_severity ON agent_compliance_checks(violation_severity);
CREATE INDEX IF NOT EXISTS idx_agent_compliance_checks_checked_at ON agent_compliance_checks(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_compliance_checks_agent_checked ON agent_compliance_checks(agent_id, checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_compliance_checks_regulation ON agent_compliance_checks(regulation);

-- Agent User Interactions indexes
CREATE INDEX IF NOT EXISTS idx_agent_user_interactions_interaction_id ON agent_user_interactions(interaction_id);
CREATE INDEX IF NOT EXISTS idx_agent_user_interactions_agent_id ON agent_user_interactions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_user_interactions_user_satisfaction_score ON agent_user_interactions(user_satisfaction_score DESC);
CREATE INDEX IF NOT EXISTS idx_agent_user_interactions_interaction_time ON agent_user_interactions(interaction_time DESC);
CREATE INDEX IF NOT EXISTS idx_agent_user_interactions_agent_time ON agent_user_interactions(agent_id, interaction_time DESC);
CREATE INDEX IF NOT EXISTS idx_agent_user_interactions_satisfaction_category ON agent_user_interactions(satisfaction_category);

-- ============================================================================
-- TRIGGERS FOR AGENT PERFORMANCE METRICS TABLES
-- ============================================================================

-- Update timestamp triggers
CREATE TRIGGER update_agent_performance_metrics_updated_at
    BEFORE UPDATE ON agent_performance_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_performance_analyses_updated_at
    BEFORE UPDATE ON agent_performance_analyses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_optimization_recommendations_updated_at
    BEFORE UPDATE ON agent_optimization_recommendations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_performance_alerts_updated_at
    BEFORE UPDATE ON agent_performance_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_dashboard_generations_updated_at
    BEFORE UPDATE ON agent_dashboard_generations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_recommendation_implementations_updated_at
    BEFORE UPDATE ON recommendation_implementations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Performance metrics aggregation trigger
CREATE OR REPLACE FUNCTION aggregate_agent_performance_metrics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update overall score based on component scores
    IF NEW.success_rate IS NOT NULL AND NEW.quality_score IS NOT NULL AND NEW.efficiency_score IS NOT NULL AND NEW.compliance_score IS NOT NULL THEN
        NEW.overall_score = (
            NEW.success_rate * 25 + 
            NEW.quality_score * 25 + 
            NEW.efficiency_score * 20 + 
            NEW.compliance_score * 20 + 
            COALESCE(NEW.user_satisfaction_score, 0.8) * 10
        );
    END IF;
    
    -- Calculate derived metrics
    IF NEW.total_tasks > 0 THEN
        NEW.success_rate = GREATEST(0, LEAST(1, NEW.completed_tasks::DECIMAL / NEW.total_tasks));
        NEW.error_rate = GREATEST(0, LEAST(1, NEW.failed_tasks::DECIMAL / NEW.total_tasks));
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER aggregate_agent_performance_metrics_trigger
    BEFORE INSERT OR UPDATE ON agent_performance_metrics
    FOR EACH ROW EXECUTE FUNCTION aggregate_agent_performance_metrics();

-- Alert status update trigger
CREATE OR REPLACE FUNCTION update_alert_occurrence()
RETURNS TRIGGER AS $$
BEGIN
    -- Update occurrence count and last occurrence timestamp
    IF TG_OP = 'UPDATE' AND OLD.alert_status = NEW.alert_status AND NEW.alert_status = 'active' THEN
        NEW.occurrence_count = COALESCE(OLD.occurrence_count, 0) + 1;
        NEW.last_occurrence = CURRENT_TIMESTAMP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_alert_occurrence_trigger
    BEFORE UPDATE ON agent_performance_alerts
    FOR EACH ROW EXECUTE FUNCTION update_alert_occurrence();

-- Recommendation implementation progress trigger
CREATE OR REPLACE FUNCTION update_implementation_progress()
RETURNS TRIGGER AS $$
BEGIN
    -- Auto-update implementation status based on progress percentage
    IF NEW.progress_percentage = 0 AND NEW.implementation_status = 'not_started' THEN
        -- No change needed
        NULL;
    ELSIF NEW.progress_percentage > 0 AND NEW.progress_percentage < 100 AND NEW.implementation_status = 'not_started' THEN
        NEW.implementation_status = 'in_progress';
    ELSIF NEW.progress_percentage = 100 AND NEW.implementation_status != 'completed' THEN
        NEW.implementation_status = 'completed';
        NEW.actual_completion_date = CURRENT_DATE;
    END IF;
    
    -- Update milestone progress
    IF NEW.total_milestones > 0 THEN
        NEW.progress_percentage = (NEW.completed_milestones * 100) / NEW.total_milestones;
    END IF;
    
    -- Update last progress update timestamp
    NEW.last_progress_update = CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_implementation_progress_trigger
    BEFORE UPDATE ON recommendation_implementations
    FOR EACH ROW EXECUTE FUNCTION update_implementation_progress();

-- ============================================================================
-- DUTCH MARKET INTELLIGENCE INTERFACE TABLES
-- ============================================================================
-- These tables support the Dutch Market Intelligence Interface with comprehensive
-- real-time data feeds, trend analysis, and predictive insights

-- Market Data Sources - stores configuration and metadata for Dutch market data sources
CREATE TABLE IF NOT EXISTS market_data_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Source identification
    source_id VARCHAR(255) NOT NULL UNIQUE,
    source_name VARCHAR(255) NOT NULL,
    source_type VARCHAR(100) NOT NULL CHECK (source_type IN ('cbs', 'dnb', 'kadaster', 'afm', 'nhg', 'bkr', 'ecb', 'eurostat', 'custom')),
    source_category VARCHAR(100) NOT NULL,
    
    -- Source details
    description TEXT,
    data_provider VARCHAR(255),
    official_name VARCHAR(500),
    website_url VARCHAR(500),
    
    -- API configuration
    api_endpoint VARCHAR(500),
    api_key_required BOOLEAN DEFAULT true,
    api_version VARCHAR(50),
    authentication_method VARCHAR(100) DEFAULT 'api_key',
    rate_limit_requests INTEGER DEFAULT 100,
    rate_limit_window INTEGER DEFAULT 3600, -- seconds
    
    -- Data characteristics
    update_frequency VARCHAR(50) DEFAULT 'daily',
    data_quality VARCHAR(20) DEFAULT 'high' CHECK (data_quality IN ('low', 'medium', 'high', 'very_high')),
    historical_data_availability VARCHAR(100),
    real_time_availability BOOLEAN DEFAULT false,
    
    -- Available metrics
    available_metrics JSONB DEFAULT '[]',
    supported_formats JSONB DEFAULT '["json"]',
    supported_filters JSONB DEFAULT '{}',
    
    -- Source status and reliability
    source_status VARCHAR(20) DEFAULT 'active' CHECK (source_status IN ('active', 'inactive', 'maintenance', 'deprecated')),
    reliability_score DECIMAL(5,4) DEFAULT 1.0 CHECK (reliability_score BETWEEN 0 AND 1),
    uptime_percentage DECIMAL(5,4) DEFAULT 1.0,
    last_successful_fetch TIMESTAMP WITH TIME ZONE,
    
    -- Configuration and settings
    fetch_configuration JSONB DEFAULT '{}',
    transformation_rules JSONB DEFAULT '[]',
    validation_rules JSONB DEFAULT '{}',
    
    -- Extended metadata
    contact_information JSONB DEFAULT '{}',
    license_information TEXT,
    usage_restrictions TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_validated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Data Collections - stores collected market data points
CREATE TABLE IF NOT EXISTS market_data_collections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Collection identification
    collection_id VARCHAR(255) NOT NULL UNIQUE,
    source_id VARCHAR(255) NOT NULL REFERENCES market_data_sources(source_id) ON DELETE CASCADE,
    metric_name VARCHAR(255) NOT NULL,
    
    -- Data point details
    data_value DECIMAL(15,6) NOT NULL,
    data_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    data_type VARCHAR(50) DEFAULT 'numeric',
    data_unit VARCHAR(100),
    
    -- Data quality and validation
    quality_score DECIMAL(5,4) DEFAULT 1.0 CHECK (quality_score BETWEEN 0 AND 1),
    confidence_level DECIMAL(5,4) DEFAULT 1.0 CHECK (confidence_level BETWEEN 0 AND 1),
    validation_status VARCHAR(20) DEFAULT 'validated' CHECK (validation_status IN ('pending', 'validated', 'failed', 'flagged')),
    
    -- Data context
    geographical_scope VARCHAR(100) DEFAULT 'national',
    sector_classification VARCHAR(100),
    demographic_filters JSONB DEFAULT '{}',
    
    -- Collection metadata
    collection_method VARCHAR(100) DEFAULT 'automated',
    collection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_time DECIMAL(8,4) DEFAULT 0,
    batch_id VARCHAR(255),
    
    -- Data transformations applied
    raw_value DECIMAL(15,6),
    transformations_applied JSONB DEFAULT '[]',
    normalization_method VARCHAR(100),
    
    -- Extended metadata
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Trend Analyses - stores results of trend analysis operations
CREATE TABLE IF NOT EXISTS market_trend_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Analysis identification
    analysis_id VARCHAR(255) NOT NULL UNIQUE,
    analysis_name VARCHAR(255) NOT NULL,
    analysis_type VARCHAR(100) NOT NULL CHECK (analysis_type IN ('trend_analysis', 'predictive_modeling', 'sentiment_analysis', 'risk_assessment', 'correlation_analysis', 'seasonal_analysis', 'volatility_analysis', 'comparative_analysis')),
    
    -- Analysis scope
    metric_name VARCHAR(255) NOT NULL,
    data_source VARCHAR(100),
    analysis_period VARCHAR(100) NOT NULL,
    data_points_count INTEGER DEFAULT 0,
    
    -- Trend analysis results
    trend_type VARCHAR(50) DEFAULT 'unknown' CHECK (trend_type IN ('upward', 'downward', 'stable', 'volatile', 'cyclical', 'seasonal', 'unknown')),
    trend_direction VARCHAR(50) DEFAULT 'stable',
    trend_strength DECIMAL(5,4) DEFAULT 0 CHECK (trend_strength BETWEEN 0 AND 1),
    confidence_level DECIMAL(5,4) DEFAULT 0 CHECK (confidence_level BETWEEN 0 AND 1),
    
    -- Statistical measures
    statistical_significance DECIMAL(8,6) DEFAULT 0,
    r_squared DECIMAL(8,6) DEFAULT 0,
    trend_equation VARCHAR(500),
    volatility_measure DECIMAL(8,6) DEFAULT 0,
    
    -- Trend components
    trend_component DECIMAL(8,6),
    seasonal_component DECIMAL(8,6),
    cyclical_component DECIMAL(8,6),
    irregular_component DECIMAL(8,6),
    
    -- Analysis parameters and configuration
    analysis_parameters JSONB DEFAULT '{}',
    algorithm_used VARCHAR(100),
    model_configuration JSONB DEFAULT '{}',
    
    -- Key findings and factors
    key_factors JSONB DEFAULT '[]',
    analysis_summary TEXT,
    significant_events JSONB DEFAULT '[]',
    
    -- Analysis quality and validation
    analysis_quality DECIMAL(5,4) DEFAULT 1.0 CHECK (analysis_quality BETWEEN 0 AND 1),
    validation_metrics JSONB DEFAULT '{}',
    peer_review_status VARCHAR(20) DEFAULT 'pending',
    
    -- Time periods and forecasting
    analysis_start_date TIMESTAMP WITH TIME ZONE,
    analysis_end_date TIMESTAMP WITH TIME ZONE,
    forecast_horizon_days INTEGER DEFAULT 0,
    forecast_accuracy DECIMAL(5,4),
    
    -- Extended metadata
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Predictive Models - stores predictive model configurations and results
CREATE TABLE IF NOT EXISTS market_predictive_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Model identification
    model_id VARCHAR(255) NOT NULL UNIQUE,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL CHECK (model_type IN ('random_forest', 'gradient_boosting', 'linear_regression', 'ridge', 'lasso', 'neural_network', 'arima', 'lstm')),
    model_version VARCHAR(50) DEFAULT '1.0',
    
    -- Model target and features
    target_variable VARCHAR(255) NOT NULL,
    feature_variables JSONB NOT NULL DEFAULT '[]',
    feature_importance JSONB DEFAULT '{}',
    
    -- Training data information
    training_data_size INTEGER NOT NULL DEFAULT 0,
    training_start_date TIMESTAMP WITH TIME ZONE,
    training_end_date TIMESTAMP WITH TIME ZONE,
    data_sources_used JSONB DEFAULT '[]',
    
    -- Model performance metrics
    accuracy_score DECIMAL(8,6) DEFAULT 0 CHECK (accuracy_score BETWEEN 0 AND 1),
    mae DECIMAL(12,6) DEFAULT 0, -- Mean Absolute Error
    rmse DECIMAL(12,6) DEFAULT 0, -- Root Mean Square Error
    r2_score DECIMAL(8,6) DEFAULT 0,
    cross_validation_score DECIMAL(8,6) DEFAULT 0,
    
    -- Model configuration and parameters
    model_parameters JSONB DEFAULT '{}',
    hyperparameters JSONB DEFAULT '{}',
    training_configuration JSONB DEFAULT '{}',
    preprocessing_steps JSONB DEFAULT '[]',
    
    -- Predictions and forecasting
    prediction_horizon_days INTEGER DEFAULT 30,
    latest_predictions JSONB DEFAULT '[]',
    prediction_confidence_intervals JSONB DEFAULT '[]',
    prediction_accuracy_history JSONB DEFAULT '[]',
    
    -- Model validation and testing
    validation_method VARCHAR(100) DEFAULT 'train_test_split',
    test_data_size INTEGER DEFAULT 0,
    validation_results JSONB DEFAULT '{}',
    model_stability_score DECIMAL(5,4) DEFAULT 0,
    
    -- Model lifecycle
    model_status VARCHAR(20) DEFAULT 'active' CHECK (model_status IN ('training', 'active', 'deprecated', 'archived', 'failed')),
    last_training_date TIMESTAMP WITH TIME ZONE,
    next_retrain_date TIMESTAMP WITH TIME ZONE,
    retrain_frequency VARCHAR(50) DEFAULT 'monthly',
    
    -- Performance monitoring
    prediction_drift_score DECIMAL(5,4) DEFAULT 0,
    model_degradation_alerts INTEGER DEFAULT 0,
    last_performance_check TIMESTAMP WITH TIME ZONE,
    
    -- Extended metadata
    model_description TEXT,
    business_context TEXT,
    usage_guidelines TEXT,
    limitations TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Intelligence Insights - stores generated market insights and recommendations
CREATE TABLE IF NOT EXISTS market_intelligence_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Insight identification
    insight_id VARCHAR(255) NOT NULL UNIQUE,
    insight_title VARCHAR(500) NOT NULL,
    insight_category VARCHAR(100) NOT NULL,
    insight_type VARCHAR(100) DEFAULT 'trend_based',
    
    -- Insight content
    description TEXT NOT NULL,
    detailed_analysis TEXT,
    key_findings JSONB DEFAULT '[]',
    implications JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    
    -- Insight scoring and prioritization
    importance_score DECIMAL(5,4) NOT NULL CHECK (importance_score BETWEEN 0 AND 1),
    confidence_level DECIMAL(5,4) NOT NULL CHECK (confidence_level BETWEEN 0 AND 1),
    actionability_score DECIMAL(5,4) DEFAULT 0 CHECK (actionability_score BETWEEN 0 AND 1),
    urgency_level VARCHAR(20) DEFAULT 'medium' CHECK (urgency_level IN ('low', 'medium', 'high', 'critical')),
    
    -- Risk assessment
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('very_low', 'low', 'moderate', 'high', 'very_high', 'critical')),
    risk_factors JSONB DEFAULT '[]',
    mitigation_strategies JSONB DEFAULT '[]',
    
    -- Supporting data and analysis
    supporting_data_sources JSONB DEFAULT '[]',
    analysis_references JSONB DEFAULT '[]', -- References to trend_analyses and models
    statistical_evidence JSONB DEFAULT '{}',
    validation_evidence JSONB DEFAULT '{}',
    
    -- Time relevance
    time_horizon VARCHAR(100) DEFAULT 'short_term',
    relevance_period VARCHAR(100),
    expiry_date TIMESTAMP WITH TIME ZONE,
    
    -- Insight lifecycle
    insight_status VARCHAR(20) DEFAULT 'active' CHECK (insight_status IN ('draft', 'active', 'archived', 'superseded', 'rejected')),
    review_status VARCHAR(20) DEFAULT 'pending',
    reviewed_by VARCHAR(100),
    review_notes TEXT,
    
    -- User interactions and feedback
    view_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    implementation_count INTEGER DEFAULT 0,
    user_feedback JSONB DEFAULT '[]',
    
    -- Quality and validation
    insight_quality DECIMAL(5,4) DEFAULT 1.0 CHECK (insight_quality BETWEEN 0 AND 1),
    peer_review_score DECIMAL(5,4),
    validation_status VARCHAR(20) DEFAULT 'validated',
    
    -- Market context
    market_conditions JSONB DEFAULT '{}',
    economic_indicators JSONB DEFAULT '{}',
    regulatory_context JSONB DEFAULT '{}',
    
    -- Extended metadata
    tags JSONB DEFAULT '[]',
    related_insights JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Intelligence Reports - stores comprehensive market intelligence reports
CREATE TABLE IF NOT EXISTS market_intelligence_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Report identification
    report_id VARCHAR(255) NOT NULL UNIQUE,
    report_title VARCHAR(500) NOT NULL,
    report_type VARCHAR(100) NOT NULL CHECK (report_type IN ('comprehensive', 'executive', 'technical', 'regulatory', 'sector_specific', 'custom')),
    report_version VARCHAR(50) DEFAULT '1.0',
    
    -- Report scope and coverage
    coverage_period VARCHAR(100) NOT NULL,
    geographical_scope VARCHAR(100) DEFAULT 'netherlands',
    sector_focus JSONB DEFAULT '[]',
    market_segments JSONB DEFAULT '[]',
    
    -- Report structure and content
    executive_summary TEXT,
    key_findings JSONB DEFAULT '[]',
    market_analysis TEXT,
    trend_analysis TEXT,
    risk_assessment TEXT,
    recommendations JSONB DEFAULT '[]',
    conclusions TEXT,
    
    -- Report components
    included_insights JSONB DEFAULT '[]', -- References to market_intelligence_insights
    data_sources_used JSONB DEFAULT '[]',
    analyses_referenced JSONB DEFAULT '[]',
    models_used JSONB DEFAULT '[]',
    
    -- Report metrics and statistics
    total_insights_count INTEGER DEFAULT 0,
    high_priority_insights_count INTEGER DEFAULT 0,
    critical_risks_identified INTEGER DEFAULT 0,
    recommendations_count INTEGER DEFAULT 0,
    
    -- Visualizations and attachments
    visualizations JSONB DEFAULT '{}',
    chart_configurations JSONB DEFAULT '{}',
    attached_files JSONB DEFAULT '[]',
    
    -- Report quality and validation
    report_quality DECIMAL(5,4) DEFAULT 1.0 CHECK (report_quality BETWEEN 0 AND 1),
    data_completeness DECIMAL(5,4) DEFAULT 1.0,
    analysis_depth VARCHAR(20) DEFAULT 'standard' CHECK (analysis_depth IN ('basic', 'standard', 'detailed', 'comprehensive')),
    
    -- Generation and processing
    generation_method VARCHAR(100) DEFAULT 'automated',
    generation_time DECIMAL(8,4),
    processing_steps JSONB DEFAULT '[]',
    generation_parameters JSONB DEFAULT '{}',
    
    -- Report lifecycle
    report_status VARCHAR(20) DEFAULT 'draft' CHECK (report_status IN ('generating', 'draft', 'review', 'approved', 'published', 'archived')),
    approval_status VARCHAR(20) DEFAULT 'pending',
    approved_by VARCHAR(100),
    approval_date TIMESTAMP WITH TIME ZONE,
    
    -- Distribution and access
    distribution_list JSONB DEFAULT '[]',
    access_level VARCHAR(20) DEFAULT 'internal' CHECK (access_level IN ('restricted', 'internal', 'public')),
    download_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    
    -- Export formats and storage
    available_formats JSONB DEFAULT '["json", "pdf", "html"]',
    file_paths JSONB DEFAULT '{}',
    report_size_kb INTEGER DEFAULT 0,
    
    -- Extended metadata
    report_description TEXT,
    methodology TEXT,
    limitations TEXT,
    disclaimers TEXT,
    tags JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    report_period_start TIMESTAMP WITH TIME ZONE,
    report_period_end TIMESTAMP WITH TIME ZONE,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Intelligence Task Executions - stores task execution history and results for market intelligence
CREATE TABLE IF NOT EXISTS intelligence_task_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Task identification
    task_id VARCHAR(255) NOT NULL UNIQUE,
    execution_id VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL CHECK (task_type IN ('collect_market_data', 'perform_trend_analysis', 'generate_predictive_model', 'generate_market_insights', 'generate_comprehensive_report', 'data_validation', 'cache_management', 'system_health_check', 'performance_optimization')),
    task_name VARCHAR(500) NOT NULL,
    
    -- Task parameters and configuration
    task_parameters JSONB NOT NULL DEFAULT '{}',
    execution_parameters JSONB DEFAULT '{}',
    environment_context JSONB DEFAULT '{}',
    
    -- Task execution details
    execution_status VARCHAR(20) NOT NULL CHECK (execution_status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled', 'timeout', 'retrying')),
    priority_level VARCHAR(20) DEFAULT 'normal' CHECK (priority_level IN ('critical', 'high', 'normal', 'low', 'background')),
    timeout_seconds INTEGER DEFAULT 1800,
    max_retries INTEGER DEFAULT 3,
    retry_count INTEGER DEFAULT 0,
    
    -- Execution timing
    queued_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time DECIMAL(8,4) DEFAULT 0,
    
    -- Task results and outputs
    execution_result JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    generated_artifacts JSONB DEFAULT '[]',
    error_message TEXT,
    error_details JSONB DEFAULT '{}',
    
    -- Resource utilization
    cpu_usage_percent DECIMAL(5,2) DEFAULT 0,
    memory_usage_mb DECIMAL(10,2) DEFAULT 0,
    disk_io_operations INTEGER DEFAULT 0,
    network_requests INTEGER DEFAULT 0,
    
    -- Dependencies and relationships
    dependent_tasks JSONB DEFAULT '[]',
    parent_task_id VARCHAR(255),
    child_task_ids JSONB DEFAULT '[]',
    
    -- Quality and validation
    execution_quality DECIMAL(5,4) DEFAULT 1.0 CHECK (execution_quality BETWEEN 0 AND 1),
    output_validation_status VARCHAR(20) DEFAULT 'pending',
    data_quality_score DECIMAL(5,4) DEFAULT 1.0,
    
    -- Monitoring and alerting
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage BETWEEN 0 AND 100),
    status_updates JSONB DEFAULT '[]',
    alerts_generated JSONB DEFAULT '[]',
    
    -- Extended metadata
    worker_thread VARCHAR(100),
    execution_environment VARCHAR(100),
    system_metrics JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market Data Quality Metrics - stores data quality assessments and metrics
CREATE TABLE IF NOT EXISTS market_data_quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Quality assessment identification
    assessment_id VARCHAR(255) NOT NULL UNIQUE,
    source_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    assessment_period VARCHAR(100) NOT NULL,
    
    -- Quality dimensions
    completeness_score DECIMAL(5,4) DEFAULT 1.0 CHECK (completeness_score BETWEEN 0 AND 1),
    accuracy_score DECIMAL(5,4) DEFAULT 1.0 CHECK (accuracy_score BETWEEN 0 AND 1),
    consistency_score DECIMAL(5,4) DEFAULT 1.0 CHECK (consistency_score BETWEEN 0 AND 1),
    timeliness_score DECIMAL(5,4) DEFAULT 1.0 CHECK (timeliness_score BETWEEN 0 AND 1),
    validity_score DECIMAL(5,4) DEFAULT 1.0 CHECK (validity_score BETWEEN 0 AND 1),
    
    -- Overall quality metrics
    overall_quality_score DECIMAL(5,4) DEFAULT 1.0 CHECK (overall_quality_score BETWEEN 0 AND 1),
    quality_grade VARCHAR(5) DEFAULT 'A' CHECK (quality_grade IN ('A', 'B', 'C', 'D', 'F')),
    quality_trend VARCHAR(20) DEFAULT 'stable' CHECK (quality_trend IN ('improving', 'stable', 'declining')),
    
    -- Data volume and coverage
    total_data_points INTEGER DEFAULT 0,
    missing_data_points INTEGER DEFAULT 0,
    invalid_data_points INTEGER DEFAULT 0,
    duplicate_data_points INTEGER DEFAULT 0,
    outlier_data_points INTEGER DEFAULT 0,
    
    -- Quality issues identified
    quality_issues JSONB DEFAULT '[]',
    data_anomalies JSONB DEFAULT '[]',
    validation_failures JSONB DEFAULT '[]',
    
    -- Assessment details
    assessment_method VARCHAR(100) DEFAULT 'automated',
    quality_rules_applied JSONB DEFAULT '[]',
    validation_criteria JSONB DEFAULT '{}',
    
    -- Improvement recommendations
    quality_recommendations JSONB DEFAULT '[]',
    remediation_actions JSONB DEFAULT '[]',
    priority_improvements JSONB DEFAULT '[]',
    
    -- Historical comparison
    previous_assessment_score DECIMAL(5,4),
    score_change DECIMAL(6,4) DEFAULT 0,
    improvement_percentage DECIMAL(6,2) DEFAULT 0,
    
    -- Extended metadata
    assessment_notes TEXT,
    methodology TEXT,
    data_lineage JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    assessment_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR DUTCH MARKET INTELLIGENCE TABLES
-- ============================================================================

-- Market Data Sources indexes
CREATE INDEX IF NOT EXISTS idx_market_data_sources_source_id ON market_data_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_market_data_sources_source_type ON market_data_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_market_data_sources_status ON market_data_sources(source_status);
CREATE INDEX IF NOT EXISTS idx_market_data_sources_reliability ON market_data_sources(reliability_score);
CREATE INDEX IF NOT EXISTS idx_market_data_sources_last_fetch ON market_data_sources(last_successful_fetch);

-- Market Data Collections indexes
CREATE INDEX IF NOT EXISTS idx_market_data_collections_collection_id ON market_data_collections(collection_id);
CREATE INDEX IF NOT EXISTS idx_market_data_collections_source_metric ON market_data_collections(source_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_market_data_collections_timestamp ON market_data_collections(data_timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_collections_collection_timestamp ON market_data_collections(collection_timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_collections_quality ON market_data_collections(quality_score);
CREATE INDEX IF NOT EXISTS idx_market_data_collections_validation ON market_data_collections(validation_status);
CREATE INDEX IF NOT EXISTS idx_market_data_collections_batch ON market_data_collections(batch_id);

-- Market Trend Analyses indexes
CREATE INDEX IF NOT EXISTS idx_market_trend_analyses_analysis_id ON market_trend_analyses(analysis_id);
CREATE INDEX IF NOT EXISTS idx_market_trend_analyses_type ON market_trend_analyses(analysis_type);
CREATE INDEX IF NOT EXISTS idx_market_trend_analyses_metric ON market_trend_analyses(metric_name);
CREATE INDEX IF NOT EXISTS idx_market_trend_analyses_trend_type ON market_trend_analyses(trend_type);
CREATE INDEX IF NOT EXISTS idx_market_trend_analyses_confidence ON market_trend_analyses(confidence_level);
CREATE INDEX IF NOT EXISTS idx_market_trend_analyses_quality ON market_trend_analyses(analysis_quality);
CREATE INDEX IF NOT EXISTS idx_market_trend_analyses_timestamp ON market_trend_analyses(analysis_timestamp);

-- Market Predictive Models indexes
CREATE INDEX IF NOT EXISTS idx_market_predictive_models_model_id ON market_predictive_models(model_id);
CREATE INDEX IF NOT EXISTS idx_market_predictive_models_type ON market_predictive_models(model_type);
CREATE INDEX IF NOT EXISTS idx_market_predictive_models_target ON market_predictive_models(target_variable);
CREATE INDEX IF NOT EXISTS idx_market_predictive_models_status ON market_predictive_models(model_status);
CREATE INDEX IF NOT EXISTS idx_market_predictive_models_accuracy ON market_predictive_models(accuracy_score);
CREATE INDEX IF NOT EXISTS idx_market_predictive_models_last_used ON market_predictive_models(last_used);
CREATE INDEX IF NOT EXISTS idx_market_predictive_models_retrain_date ON market_predictive_models(next_retrain_date);

-- Market Intelligence Insights indexes
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_insight_id ON market_intelligence_insights(insight_id);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_category ON market_intelligence_insights(insight_category);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_importance ON market_intelligence_insights(importance_score);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_confidence ON market_intelligence_insights(confidence_level);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_risk_level ON market_intelligence_insights(risk_level);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_status ON market_intelligence_insights(insight_status);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_urgency ON market_intelligence_insights(urgency_level);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_generated ON market_intelligence_insights(generated_at);

-- Market Intelligence Reports indexes
CREATE INDEX IF NOT EXISTS idx_market_intelligence_reports_report_id ON market_intelligence_reports(report_id);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_reports_type ON market_intelligence_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_reports_status ON market_intelligence_reports(report_status);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_reports_quality ON market_intelligence_reports(report_quality);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_reports_generated ON market_intelligence_reports(generated_at);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_reports_published ON market_intelligence_reports(published_at);

-- Intelligence Task Executions indexes
CREATE INDEX IF NOT EXISTS idx_intelligence_task_executions_task_id ON intelligence_task_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_intelligence_task_executions_execution_id ON intelligence_task_executions(execution_id);
CREATE INDEX IF NOT EXISTS idx_intelligence_task_executions_type ON intelligence_task_executions(task_type);
CREATE INDEX IF NOT EXISTS idx_intelligence_task_executions_status ON intelligence_task_executions(execution_status);
CREATE INDEX IF NOT EXISTS idx_intelligence_task_executions_priority ON intelligence_task_executions(priority_level);
CREATE INDEX IF NOT EXISTS idx_intelligence_task_executions_started ON intelligence_task_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_intelligence_task_executions_completed ON intelligence_task_executions(completed_at);

-- Market Data Quality Metrics indexes
CREATE INDEX IF NOT EXISTS idx_market_data_quality_metrics_assessment_id ON market_data_quality_metrics(assessment_id);
CREATE INDEX IF NOT EXISTS idx_market_data_quality_metrics_source_metric ON market_data_quality_metrics(source_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_market_data_quality_metrics_overall_score ON market_data_quality_metrics(overall_quality_score);
CREATE INDEX IF NOT EXISTS idx_market_data_quality_metrics_assessment_date ON market_data_quality_metrics(assessment_date);
CREATE INDEX IF NOT EXISTS idx_market_data_quality_metrics_grade ON market_data_quality_metrics(quality_grade);

-- ============================================================================
-- COMPOSITE INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Multi-column indexes for complex queries
CREATE INDEX IF NOT EXISTS idx_market_data_collections_source_metric_time ON market_data_collections(source_id, metric_name, data_timestamp);
CREATE INDEX IF NOT EXISTS idx_market_trend_analyses_metric_type_confidence ON market_trend_analyses(metric_name, analysis_type, confidence_level);
CREATE INDEX IF NOT EXISTS idx_market_intelligence_insights_category_importance_risk ON market_intelligence_insights(insight_category, importance_score, risk_level);
CREATE INDEX IF NOT EXISTS idx_intelligence_task_executions_type_status_started ON intelligence_task_executions(task_type, execution_status, started_at);

-- ============================================================================
-- TRIGGERS FOR DUTCH MARKET INTELLIGENCE TABLES
-- ============================================================================

-- Update timestamps trigger for market_data_sources
CREATE OR REPLACE FUNCTION update_market_data_sources_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_market_data_sources_updated_at
    BEFORE UPDATE ON market_data_sources
    FOR EACH ROW
    EXECUTE FUNCTION update_market_data_sources_updated_at();

-- Update timestamps trigger for market_data_collections
CREATE OR REPLACE FUNCTION update_market_data_collections_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_market_data_collections_updated_at
    BEFORE UPDATE ON market_data_collections
    FOR EACH ROW
    EXECUTE FUNCTION update_market_data_collections_updated_at();

-- Update timestamps trigger for market_trend_analyses
CREATE OR REPLACE FUNCTION update_market_trend_analyses_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_market_trend_analyses_updated_at
    BEFORE UPDATE ON market_trend_analyses
    FOR EACH ROW
    EXECUTE FUNCTION update_market_trend_analyses_updated_at();

-- Update timestamps trigger for market_predictive_models
CREATE OR REPLACE FUNCTION update_market_predictive_models_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_market_predictive_models_updated_at
    BEFORE UPDATE ON market_predictive_models
    FOR EACH ROW
    EXECUTE FUNCTION update_market_predictive_models_updated_at();

-- Update timestamps trigger for market_intelligence_insights
CREATE OR REPLACE FUNCTION update_market_intelligence_insights_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_market_intelligence_insights_updated_at
    BEFORE UPDATE ON market_intelligence_insights
    FOR EACH ROW
    EXECUTE FUNCTION update_market_intelligence_insights_updated_at();

-- Update timestamps trigger for market_intelligence_reports
CREATE OR REPLACE FUNCTION update_market_intelligence_reports_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_market_intelligence_reports_updated_at
    BEFORE UPDATE ON market_intelligence_reports
    FOR EACH ROW
    EXECUTE FUNCTION update_market_intelligence_reports_updated_at();

-- Update timestamps trigger for intelligence_task_executions
CREATE OR REPLACE FUNCTION update_intelligence_task_executions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_intelligence_task_executions_updated_at
    BEFORE UPDATE ON intelligence_task_executions
    FOR EACH ROW
    EXECUTE FUNCTION update_intelligence_task_executions_updated_at();

-- Update timestamps trigger for market_data_quality_metrics
CREATE OR REPLACE FUNCTION update_market_data_quality_metrics_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_market_data_quality_metrics_updated_at
    BEFORE UPDATE ON market_data_quality_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_market_data_quality_metrics_updated_at();
