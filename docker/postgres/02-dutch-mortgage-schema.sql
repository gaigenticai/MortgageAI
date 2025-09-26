-- Dutch Mortgage Schema Extensions
-- Additional tables and indexes for Phase 4 and Phase 5 features

-- Lender submissions tracking table
CREATE TABLE IF NOT EXISTS lender_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id VARCHAR(100) NOT NULL,
    lender_name VARCHAR(50) NOT NULL,
    reference_number VARCHAR(100),
    status VARCHAR(50) DEFAULT 'submitted',
    submission_data JSONB,
    error_message TEXT,
    estimated_processing_time VARCHAR(100),
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Compliance alerts table
CREATE TABLE IF NOT EXISTS compliance_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(100),
    alert_type VARCHAR(100) NOT NULL,
    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
    details JSONB,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'dismissed')),
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- AFM regulation updates table
CREATE TABLE IF NOT EXISTS afm_regulation_updates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regulation_code VARCHAR(50) NOT NULL,
    update_type VARCHAR(50) NOT NULL,
    changes JSONB,
    effective_date DATE NOT NULL,
    update_date DATE NOT NULL,
    source_url VARCHAR(1000),
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(regulation_code, effective_date)
);

-- Indexes for new tables
CREATE INDEX IF NOT EXISTS idx_lender_submissions_application ON lender_submissions(application_id);
CREATE INDEX IF NOT EXISTS idx_lender_submissions_lender ON lender_submissions(lender_name);
CREATE INDEX IF NOT EXISTS idx_lender_submissions_status ON lender_submissions(status);
CREATE INDEX IF NOT EXISTS idx_lender_submissions_submitted_at ON lender_submissions(submitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_compliance_alerts_status ON compliance_alerts(status);
CREATE INDEX IF NOT EXISTS idx_compliance_alerts_risk_level ON compliance_alerts(risk_level);
CREATE INDEX IF NOT EXISTS idx_compliance_alerts_session ON compliance_alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_compliance_alerts_created_at ON compliance_alerts(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_afm_regulation_updates_code ON afm_regulation_updates(regulation_code);
CREATE INDEX IF NOT EXISTS idx_afm_regulation_updates_date ON afm_regulation_updates(update_date DESC);
CREATE INDEX IF NOT EXISTS idx_afm_regulation_updates_effective ON afm_regulation_updates(effective_date);

-- Add foreign key constraints (if tables exist)
DO $$
BEGIN
    -- Add FK to dutch_mortgage_applications if it exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'dutch_mortgage_applications') THEN
        ALTER TABLE lender_submissions
        ADD CONSTRAINT fk_lender_submissions_application
        FOREIGN KEY (application_id) REFERENCES dutch_mortgage_applications(id) ON DELETE CASCADE;
    END IF;

    -- Add FK to afm_advice_sessions if it exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'afm_advice_sessions') THEN
        ALTER TABLE compliance_alerts
        ADD CONSTRAINT fk_compliance_alerts_session
        FOREIGN KEY (session_id) REFERENCES afm_advice_sessions(id) ON DELETE SET NULL;
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Foreign key constraints could not be added (may already exist or tables missing)';
END $$;

-- Create views for reporting
CREATE OR REPLACE VIEW lender_submission_summary AS
SELECT
    lender_name,
    COUNT(*) as total_submissions,
    COUNT(CASE WHEN status = 'submitted' THEN 1 END) as active_submissions,
    COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_submissions,
    COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected_submissions,
    AVG(EXTRACT(EPOCH FROM (COALESCE(updated_at, submitted_at) - submitted_at))/86400) as avg_processing_days
FROM lender_submissions
GROUP BY lender_name;

CREATE OR REPLACE VIEW compliance_dashboard AS
SELECT
    DATE_TRUNC('day', created_at) as date,
    COUNT(*) as total_alerts,
    COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_alerts,
    COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_alerts,
    COUNT(CASE WHEN status = 'active' THEN 1 END) as active_alerts,
    COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_alerts
FROM compliance_alerts
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date DESC;

-- Insert sample AFM regulation data (for development/testing)
INSERT INTO afm_regulations (
    regulation_code, title, content, category, effective_date, source_url
) VALUES
    ('Wft-86f', 'Suitability Assessment Requirements', 'Advisors must assess client suitability based on financial situation, knowledge, experience, objectives, and risk tolerance.', 'suitability', '2024-01-01', 'https://www.afm.nl/en/professionals/regelgeving/wft'),
    ('Wft-86c', 'Product Information Disclosure', 'Advisors must provide clear information about product costs, risks, and characteristics.', 'disclosure', '2024-01-01', 'https://www.afm.nl/en/professionals/regelgeving/wft'),
    ('BGfo-8.1', 'Advisor Remuneration Disclosure', 'Advisors must clearly disclose how they are remunerated for their services.', 'remuneration', '2024-01-01', 'https://www.afm.nl/en/professionals/regelgeving/bgfo')
ON CONFLICT (regulation_code) DO NOTHING;

-- Insert sample AFM regulation versions
INSERT INTO afm_regulation_versions (
    version_number, release_date, is_current, changes_summary
) VALUES
    ('2025.1', '2025-01-01', true, 'Updated mortgage advice disclosure requirements and digital communication guidelines'),
    ('2024.2', '2024-07-01', false, 'Enhanced consumer protection measures for variable rate mortgages'),
    ('2024.1', '2024-01-01', false, 'Initial implementation of Wft 2024 requirements')
ON CONFLICT (version_number) DO NOTHING;

-- Create function for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add update triggers to new tables
DROP TRIGGER IF EXISTS update_lender_submissions_updated_at ON lender_submissions;
CREATE TRIGGER update_lender_submissions_updated_at
    BEFORE UPDATE ON lender_submissions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_compliance_alerts_updated_at ON compliance_alerts;
CREATE TRIGGER update_compliance_alerts_updated_at
    BEFORE UPDATE ON compliance_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO mortgage_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO mortgage_user;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Dutch mortgage schema extensions applied successfully';
    RAISE NOTICE 'Added tables: lender_submissions, compliance_alerts, afm_regulation_updates';
    RAISE NOTICE 'Added indexes and views for reporting';
    RAISE NOTICE 'Sample AFM regulation data inserted';
END $$;
