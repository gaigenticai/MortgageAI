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
