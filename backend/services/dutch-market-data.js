/**
 * Dutch Market Data Service
 * Handles AFM regulations, BKR credit data, NHG validation, and property valuations
 *
 * This service provides real-time integration with Dutch financial systems:
 * - AFM regulation updates and compliance checking
 * - BKR credit bureau data retrieval and analysis
 * - NHG eligibility validation and cost calculations
 * - Property valuation services integration
 */

const express = require('express');
const axios = require('axios');
const { Client } = require('pg');
const Redis = require('redis');
const cron = require('node-cron');
const winston = require('winston');
const Joi = require('joi');

const app = express();

// Environment configuration
const PORT = process.env.DUTCH_DATA_PORT || 8001;
const AFM_API_KEY = process.env.AFM_API_KEY;
const BKR_API_KEY = process.env.BKR_API_KEY;
const NHG_API_KEY = process.env.NHG_API_KEY;
const PROPERTY_VALUATION_API_KEY = process.env.PROPERTY_VALUATION_API_KEY;

// Database and cache clients
let dbClient;
let redisClient;

// Logger configuration
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'dutch-market-data' },
  transports: [
    new winston.transports.File({ filename: '/app/logs/dutch-market-data.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Database connection
async function connectDatabase() {
  try {
    const dbConfig = {
      host: process.env.DB_HOST || 'postgres',
      port: process.env.DB_PORT || 5432,
      database: process.env.DB_NAME || 'mortgage_db',
      user: process.env.DB_USER || 'mortgage_user',
      password: process.env.DB_PASSWORD || 'mortgage_pass',
      ssl: false
    };

    dbClient = new Client(dbConfig);
    await dbClient.connect();
    logger.info('Connected to PostgreSQL database');
  } catch (error) {
    logger.error('Failed to connect to database:', error);
    throw error;
  }
}

// Redis connection
async function connectRedis() {
  try {
    redisClient = Redis.createClient({
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    });

    redisClient.on('error', (err) => logger.error('Redis Client Error', err));
    redisClient.on('connect', () => logger.info('Connected to Redis'));

    await redisClient.connect();
  } catch (error) {
    logger.error('Failed to connect to Redis:', error);
    throw error;
  }
}

// AFM Regulation Management
class AFMRegulationManager {
  constructor() {
    this.regulationCache = new Map();
    this.lastUpdate = null;
    this.updateInterval = 24 * 60 * 60 * 1000; // 24 hours
  }

  async initializeRegulations() {
    try {
      logger.info('Initializing AFM regulations database...');

      // Create AFM regulations table if it doesn't exist
      await dbClient.query(`
        CREATE TABLE IF NOT EXISTS afm_regulations (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          regulation_code VARCHAR(50) UNIQUE NOT NULL,
          title VARCHAR(500) NOT NULL,
          content TEXT NOT NULL,
          category VARCHAR(100),
          effective_date DATE NOT NULL,
          expiry_date DATE,
          source_url VARCHAR(1000),
          is_active BOOLEAN DEFAULT true,
          last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_afm_regulations_code ON afm_regulations(regulation_code);
        CREATE INDEX IF NOT EXISTS idx_afm_regulations_active ON afm_regulations(is_active) WHERE is_active = true;
        CREATE INDEX IF NOT EXISTS idx_afm_regulations_category ON afm_regulations(category);
      `);

      // Fetch initial regulations from AFM API
      await this.fetchAFMRegulations();

      logger.info('AFM regulations initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize AFM regulations:', error);
      throw error;
    }
  }

  async fetchAFMRegulations() {
    try {
      // Check if AFM_REGULATION_FEED_URL is configured
      if (!process.env.AFM_REGULATION_FEED_URL || process.env.AFM_REGULATION_FEED_URL.trim() === '') {
        logger.info('AFM_REGULATION_FEED_URL not configured, using default regulations for development');
        await this.loadDefaultRegulations();
        return;
      }

      const response = await axios.get(process.env.AFM_REGULATION_FEED_URL, {
        headers: {
          'Authorization': `Bearer ${AFM_API_KEY}`,
          'Accept': 'application/json'
        },
        timeout: 30000
      });

      const regulations = response.data.regulations || [];

      for (const reg of regulations) {
        await dbClient.query(`
          INSERT INTO afm_regulations (
            regulation_code, title, content, category,
            effective_date, expiry_date, source_url, last_updated
          ) VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP)
          ON CONFLICT (regulation_code)
          DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            category = EXCLUDED.category,
            effective_date = EXCLUDED.effective_date,
            expiry_date = EXCLUDED.expiry_date,
            source_url = EXCLUDED.source_url,
            last_updated = CURRENT_TIMESTAMP
        `, [
          reg.code,
          reg.title,
          reg.content,
          reg.category,
          reg.effective_date,
          reg.expiry_date,
          reg.source_url
        ]);
      }

      this.lastUpdate = new Date();
      logger.info(`Fetched ${regulations.length} AFM regulations`);

      // Cache active regulations
      await this.cacheActiveRegulations();

    } catch (error) {
      logger.error('Failed to fetch AFM regulations:', error);
      throw error;
    }
  }

  async loadDefaultRegulations() {
    try {
      // Load default AFM regulations for development
      const defaultRegulations = [
        {
          regulation_code: 'Wft_86f',
          title: 'Suitability Assessment (Wft Article 86f)',
          content: 'Financial service providers must assess the suitability of investment services and products for their clients.',
          category: 'suitability',
          effective_date: '2021-01-01'
        },
        {
          regulation_code: 'Wft_86c',
          title: 'Product Information Disclosure (Wft Article 86c)',
          content: 'Adequate product information must be provided to clients in a comprehensible form.',
          category: 'disclosure',
          effective_date: '2021-01-01'
        },
        {
          regulation_code: 'BGfo_8_1',
          title: 'Remuneration Disclosure (BGfo Article 8.1)',
          content: 'All remuneration and incentives must be disclosed to clients.',
          category: 'remuneration',
          effective_date: '2021-01-01'
        },
        {
          regulation_code: 'BGfo_9_1',
          title: 'Risk Warnings (BGfo Article 9.1)',
          content: 'Appropriate risk warnings must be provided for all financial products.',
          category: 'risk_warnings',
          effective_date: '2021-01-01'
        }
      ];

      for (const regulation of defaultRegulations) {
        await dbClient.query(`
          INSERT INTO afm_regulations (regulation_code, title, content, category, effective_date, is_active)
          VALUES ($1, $2, $3, $4, $5, true)
          ON CONFLICT (regulation_code) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            category = EXCLUDED.category,
            last_updated = NOW()
        `, [
          regulation.regulation_code,
          regulation.title,
          regulation.content,
          regulation.category,
          regulation.effective_date
        ]);
      }

      this.lastUpdate = new Date();
      logger.info(`Loaded ${defaultRegulations.length} default AFM regulations for development`);

      // Cache active regulations
      await this.cacheActiveRegulations();

    } catch (error) {
      logger.error('Failed to load default AFM regulations:', error);
      throw error;
    }
  }

  async cacheActiveRegulations() {
    try {
      const result = await dbClient.query(`
        SELECT regulation_code, title, content, category, effective_date
        FROM afm_regulations
        WHERE is_active = true
        ORDER BY last_updated DESC
      `);

      this.regulationCache.clear();
      for (const reg of result.rows) {
        this.regulationCache.set(reg.regulation_code, reg);
      }

      // Cache in Redis for fast access
      await redisClient.setEx('afm_regulations', 3600, JSON.stringify(Object.fromEntries(this.regulationCache)));

      logger.info(`Cached ${this.regulationCache.size} active AFM regulations`);
    } catch (error) {
      logger.error('Failed to cache AFM regulations:', error);
      throw error;
    }
  }

  async getRegulation(code) {
    // Check cache first
    let regulation = this.regulationCache.get(code);

    if (!regulation) {
      // Check Redis cache
      const cached = await redisClient.get('afm_regulations');
      if (cached) {
        const regulations = JSON.parse(cached);
        regulation = regulations[code];
      }
    }

    if (!regulation) {
      // Fetch from database
      const result = await dbClient.query(`
        SELECT * FROM afm_regulations
        WHERE regulation_code = $1 AND is_active = true
      `, [code]);

      if (result.rows.length > 0) {
        regulation = result.rows[0];
        this.regulationCache.set(code, regulation);
      }
    }

    return regulation;
  }

  async validateCompliance(requirement, context) {
    // Comprehensive AFM compliance validation with regulation-specific logic
    const regulation = await this.getRegulation(requirement.regulation_code);

    if (!regulation) {
      return { compliant: false, reason: 'Regulation not found' };
    }

    // Execute comprehensive regulation-specific compliance validation
    const isCompliant = this.checkRegulationCompliance(regulation, requirement, context);

    return {
      compliant: isCompliant,
      regulation: regulation,
      checked_at: new Date().toISOString(),
      requirement: requirement
    };
  }

  checkRegulationCompliance(regulation, requirement, context) {
    // Real AFM compliance validation based on regulation requirements
    const regulationCode = regulation.regulation_code;
    const requirementType = requirement.type || requirement.regulation_code;
    const clientData = context.client_profile || {};
    const adviceData = context.advice_content || '';

    try {
      switch (regulationCode) {
        case 'Wft-86f': // Suitability Assessment
          return this.validateSuitabilityAssessment(clientData);

        case 'Wft-86c': // Product Information Disclosure
          return this.validateProductDisclosure(adviceData, context.product_recommendations || []);

        case 'BGfo-8.1': // Advisor Remuneration
          return this.validateRemunerationDisclosure(adviceData);

        case 'BGfo-9.1': // Risk Warnings
          return this.validateRiskWarnings(adviceData);

        default:
          // For unknown regulations, perform basic content check
          return this.performBasicContentValidation(regulation, adviceData);
      }
    } catch (error) {
      logger.error(`Compliance validation error for ${regulationCode}:`, error);
      return false;
    }
  }

  validateSuitabilityAssessment(clientProfile) {
    // Wft Article 86f - Suitability assessment requirements
    const requiredFactors = [
      'financial_situation',
      'knowledge_experience',
      'investment_objectives',
      'risk_tolerance',
      'debt_capacity'
    ];

    const completedFactors = requiredFactors.filter(factor =>
      clientProfile[factor] !== undefined &&
      clientProfile[factor] !== null &&
      clientProfile[factor] !== ''
    );

    // Must have at least 80% of required factors
    const complianceRatio = completedFactors.length / requiredFactors.length;
    return complianceRatio >= 0.8;
  }

  validateProductDisclosure(adviceContent, products) {
    // Wft Article 86c - Product information requirements
    if (!products || products.length === 0) {
      return false;
    }

    const content = adviceContent.toLowerCase();
    let disclosuresFound = 0;
    const requiredDisclosures = [
      'interest rate',
      'costs',
      'fees',
      'risks',
      'terms',
      'conditions'
    ];

    // Check for disclosure of each required element
    for (const disclosure of requiredDisclosures) {
      if (content.includes(disclosure)) {
        disclosuresFound++;
      }
    }

    // Must disclose at least 80% of required information
    return disclosuresFound / requiredDisclosures.length >= 0.8;
  }

  validateRemunerationDisclosure(adviceContent) {
    // BGfo Article 8.1 - Advisor remuneration disclosure
    const content = adviceContent.toLowerCase();

    // Must mention advisor fees/compensation clearly
    const remunerationIndicators = [
      'fee',
      'commission',
      'remuneration',
      'compensation',
      'cost',
      'payment'
    ];

    return remunerationIndicators.some(indicator => content.includes(indicator));
  }

  validateRiskWarnings(adviceContent) {
    // BGfo Article 9.1 - Risk warnings and consumer protection
    const content = adviceContent.toLowerCase();

    const riskIndicators = [
      'risk',
      'warning',
      'loss',
      'volatility',
      'market',
      'uncertainty'
    ];

    return riskIndicators.some(indicator => content.includes(indicator));
  }

  performBasicContentValidation(regulation, adviceContent) {
    // Fallback validation for unknown regulations
    const content = adviceContent.toLowerCase();
    const regulationKeywords = regulation.content ? regulation.content.toLowerCase() : '';

    // Extract key terms from regulation content
    const keywords = regulationKeywords.match(/\b\w{4,}\b/g) || [];

    // Check if advice content addresses key regulatory requirements
    const relevantKeywords = keywords.filter(keyword =>
      content.includes(keyword) ||
      this.isSynonym(keyword, content)
    );

    // Must address at least 30% of regulatory keywords
    return relevantKeywords.length / Math.max(keywords.length, 1) >= 0.3;
  }

  isSynonym(word, content) {
    // Basic synonym checking for common financial terms
    const synonyms = {
      'client': ['customer', 'borrower', 'applicant'],
      'advice': ['recommendation', 'guidance', 'suggestion'],
      'risk': ['danger', 'hazard', 'exposure'],
      'cost': ['fee', 'expense', 'charge'],
      'interest': ['rate', 'percentage', 'yield'],
      'mortgage': ['loan', 'hypotheek', 'home loan']
    };

    const wordSynonyms = synonyms[word] || [];
    return wordSynonyms.some(synonym => content.includes(synonym));
  }
}

// BKR Credit Bureau Integration
class BKRCreditService {
  constructor() {
    this.apiUrl = process.env.BKR_API_URL;
    this.apiKey = BKR_API_KEY;
    this.cacheTTL = 3600; // 1 hour
  }

  async getCreditReport(bsn) {
    try {
      // Validate BSN format
      if (!this.validateBSN(bsn)) {
        throw new Error('Invalid BSN format');
      }

      // Check cache first
      const cacheKey = `bkr_credit_${bsn}`;
      const cached = await redisClient.get(cacheKey);

      if (cached) {
        logger.info(`Returning cached BKR data for BSN ${bsn}`);
        return JSON.parse(cached);
      }

      // Make API call to BKR
      const response = await axios.post(`${this.apiUrl}/credit-check`, {
        bsn: bsn,
        purpose: 'mortgage_application',
        consent_timestamp: new Date().toISOString()
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: 15000
      });

      const creditData = response.data;

      // Store in database for audit trail
      await this.storeCreditCheck(bsn, creditData);

      // Cache result
      await redisClient.setEx(cacheKey, this.cacheTTL, JSON.stringify(creditData));

      logger.info(`Retrieved BKR credit data for BSN ${bsn}`);
      return creditData;

    } catch (error) {
      logger.error(`BKR credit check failed for BSN ${bsn}:`, error);
      throw error;
    }
  }

  async storeCreditCheck(bsn, creditData) {
    try {
      await dbClient.query(`
        INSERT INTO bkr_checks (
          client_id, bkr_reference, check_type, response_data,
          credit_score, negative_registrations, debt_summary, checked_at
        ) VALUES (
          (SELECT id FROM client_profiles WHERE bsn = $1 LIMIT 1),
          $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP
        )
      `, [
        bsn,
        creditData.reference_number,
        'credit_history',
        JSON.stringify(creditData),
        creditData.credit_score,
        JSON.stringify(creditData.negative_registrations || []),
        JSON.stringify(creditData.debt_summary || [])
      ]);
    } catch (error) {
      logger.error('Failed to store BKR credit check:', error);
      // Don't throw - credit check was successful, just logging failed
    }
  }

  validateBSN(bsn) {
    // Dutch BSN validation - 9 digits, specific checksum algorithm
    if (!/^\d{9}$/.test(bsn)) {
      return false;
    }

    const digits = bsn.split('').map(Number);
    let sum = 0;

    for (let i = 0; i < 8; i++) {
      sum += digits[i] * (9 - i);
    }

    const checkDigit = sum % 11;
    return checkDigit === digits[8] || (checkDigit === 0 && digits[8] === 0);
  }

  analyzeCreditSuitability(creditData, loanAmount, income) {
    // Advanced credit analysis for mortgage suitability
    const creditScore = creditData.credit_score;
    const negativeRegistrations = creditData.negative_registrations?.length || 0;
    const dtiRatio = this.calculateDTI(creditData.debt_summary, income);

    let suitability = 'approved';
    let maxLoanAmount = loanAmount;
    const recommendations = [];

    // Credit score assessment
    if (creditScore < 600) {
      suitability = 'rejected';
      recommendations.push('Credit score too low for mortgage approval');
    } else if (creditScore < 700) {
      maxLoanAmount *= 0.8;
      recommendations.push('Lower loan amount recommended due to credit score');
    }

    // Negative registrations check
    if (negativeRegistrations > 0) {
      suitability = 'conditional';
      recommendations.push('Negative registrations require lender review');
      maxLoanAmount *= 0.9;
    }

    // DTI ratio check (Dutch standard: max 40%)
    if (dtiRatio > 40) {
      suitability = 'rejected';
      recommendations.push('Debt-to-income ratio exceeds Dutch standards');
    } else if (dtiRatio > 30) {
      maxLoanAmount *= 0.95;
      recommendations.push('High debt-to-income ratio - consider debt consolidation');
    }

    return {
      suitability,
      max_loan_amount: Math.round(maxLoanAmount),
      dti_ratio: dtiRatio,
      recommendations,
      risk_level: this.assessRiskLevel(creditScore, negativeRegistrations, dtiRatio)
    };
  }

  calculateDTI(debtSummary, income) {
    if (!debtSummary || !Array.isArray(debtSummary)) {
      return 0;
    }

    const totalMonthlyDebt = debtSummary.reduce((sum, debt) => sum + (debt.monthly_payment || 0), 0);
    const annualDebt = totalMonthlyDebt * 12;

    return income > 0 ? (annualDebt / income) * 100 : 0;
  }

  assessRiskLevel(creditScore, negativeRegistrations, dtiRatio) {
    if (creditScore >= 750 && negativeRegistrations === 0 && dtiRatio <= 30) {
      return 'low';
    } else if (creditScore >= 650 && negativeRegistrations <= 1 && dtiRatio <= 35) {
      return 'medium';
    } else {
      return 'high';
    }
  }
}

// NHG Validation Service
class NHGValidationService {
  constructor() {
    this.apiUrl = process.env.NHG_VALIDATION_URL;
    this.apiKey = NHG_API_KEY;
  }

  async validateEligibility(propertyValue, loanAmount, applicantIncome, existingDebts) {
    try {
      const ltvRatio = (loanAmount / propertyValue) * 100;

      // NHG eligibility criteria
      const eligibility = {
        eligible: false,
        reasons: [],
        maximum_loan: 0,
        premium_amount: 0,
        benefits: []
      };

      // Basic eligibility checks
      if (ltvRatio > 100) {
        eligibility.reasons.push('Loan-to-value ratio exceeds 100%');
        return eligibility;
      }

      if (propertyValue > 405000) { // NHG maximum in 2025
        eligibility.reasons.push('Property value exceeds NHG maximum');
        return eligibility;
      }

      if (loanAmount > 405000) {
        eligibility.reasons.push('Loan amount exceeds NHG maximum');
        return eligibility;
      }

      // Calculate maximum eligible loan based on income
      const maxLoanByIncome = this.calculateMaxLoanByIncome(applicantIncome, existingDebts);

      if (loanAmount > maxLoanByIncome) {
        eligibility.reasons.push(`Loan amount exceeds income-based maximum (€${maxLoanByIncome.toLocaleString()})`);
        eligibility.maximum_loan = maxLoanByIncome;
        return eligibility;
      }

      eligibility.eligible = true;
      eligibility.maximum_loan = Math.min(loanAmount, 405000);
      eligibility.premium_amount = this.calculatePremium(loanAmount);
      eligibility.benefits = [
        'Lower interest rates (0.5-1% reduction)',
        'Guarantee covers up to 100% of loan',
        'No additional security required',
        'Tax benefits for premium payments'
      ];

      return eligibility;

    } catch (error) {
      logger.error('NHG eligibility validation failed:', error);
      throw error;
    }
  }

  calculateMaxLoanByIncome(annualIncome, existingDebts) {
    // Dutch income-based lending limits (simplified)
    const monthlyIncome = annualIncome / 12;
    const existingMonthlyDebts = existingDebts.reduce((sum, debt) => sum + (debt.monthly_payment || 0), 0);

    // Maximum housing costs: 30% of gross income
    const maxHousingCosts = monthlyIncome * 0.3;
    const disposableIncome = monthlyIncome - existingMonthlyDebts;

    // Conservative estimate: 80% of disposable income for housing
    const availableForHousing = disposableIncome * 0.8;

    // Estimate total housing cost (mortgage + taxes + insurance)
    // Assuming mortgage is ~70% of total housing cost
    const estimatedMortgagePayment = availableForHousing * 0.7;

    // Using 30-year mortgage at ~3% interest (Dutch average)
    // Monthly payment formula: P * (r(1+r)^n) / ((1+r)^n - 1)
    const monthlyRate = 0.03 / 12;
    const numPayments = 30 * 12;
    const loanAmount = estimatedMortgagePayment * ((1 - Math.pow(1 + monthlyRate, -numPayments)) / monthlyRate);

    return Math.round(loanAmount);
  }

  calculatePremium(loanAmount) {
    // NHG premium calculation (2025 rates)
    const premiumRate = 0.009; // 0.9%
    return Math.round(loanAmount * premiumRate);
  }
}

// Property Valuation Service
class PropertyValuationService {
  constructor() {
    this.apiUrl = process.env.PROPERTY_VALUATION_API_URL;
    this.apiKey = PROPERTY_VALUATION_API_KEY;
  }

  async getValuation(propertyAddress, propertyType = 'house') {
    try {
      const response = await axios.post(`${this.apiUrl}/valuation`, {
        address: propertyAddress,
        property_type: propertyType,
        valuation_date: new Date().toISOString().split('T')[0]
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: 20000
      });

      return {
        estimated_value: response.data.market_value,
        confidence_level: response.data.confidence_percentage,
        valuation_date: response.data.valuation_date,
        comparables_used: response.data.comparable_properties?.length || 0,
        market_trend: response.data.market_trend || 'stable'
      };

    } catch (error) {
      logger.error('Property valuation failed:', error);
      throw error;
    }
  }
}

// Service instances
const afmManager = new AFMRegulationManager();
const bkrService = new BKRCreditService();
const nhgService = new NHGValidationService();
const valuationService = new PropertyValuationService();

// API Routes

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'dutch-market-data',
    timestamp: new Date().toISOString(),
    services: {
      afm_regulations: afmManager.regulationCache.size,
      database: dbClient ? 'connected' : 'disconnected',
      redis: redisClient?.isOpen ? 'connected' : 'disconnected'
    }
  });
});

// AFM Regulations API
app.get('/api/afm/regulations', async (req, res) => {
  try {
    const { category, active_only = true } = req.query;

    let query = `
      SELECT regulation_code, title, category, effective_date, last_updated
      FROM afm_regulations
      WHERE 1=1
    `;
    const params = [];

    if (active_only === 'true') {
      query += ' AND is_active = true';
    }

    if (category) {
      query += ' AND category = $1';
      params.push(category);
    }

    query += ' ORDER BY last_updated DESC LIMIT 100';

    const result = await dbClient.query(query, params);
    res.json({ regulations: result.rows });

  } catch (error) {
    logger.error('Failed to fetch AFM regulations:', error);
    res.status(500).json({ error: 'Failed to fetch regulations' });
  }
});

app.get('/api/afm/regulations/:code', async (req, res) => {
  try {
    const { code } = req.params;
    const regulation = await afmManager.getRegulation(code);

    if (!regulation) {
      return res.status(404).json({ error: 'Regulation not found' });
    }

    res.json({ regulation });

  } catch (error) {
    logger.error(`Failed to fetch AFM regulation ${req.params.code}:`, error);
    res.status(500).json({ error: 'Failed to fetch regulation' });
  }
});

// BKR Credit Check API
app.post('/api/bkr/credit-check', async (req, res) => {
  try {
    const { bsn, loan_amount, income } = req.body;

    if (!bsn) {
      return res.status(400).json({ error: 'BSN is required' });
    }

    const creditData = await bkrService.getCreditReport(bsn);
    const suitability = bkrService.analyzeCreditSuitability(creditData, loan_amount, income);

    res.json({
      credit_data: creditData,
      suitability_analysis: suitability,
      checked_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('BKR credit check API error:', error);
    res.status(500).json({ error: 'Credit check failed' });
  }
});

// NHG Validation API
app.post('/api/nhg/validate', async (req, res) => {
  try {
    const { property_value, loan_amount, applicant_income, existing_debts } = req.body;

    const eligibility = await nhgService.validateEligibility(
      property_value,
      loan_amount,
      applicant_income,
      existing_debts || []
    );

    res.json({
      eligibility,
      validated_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('NHG validation API error:', error);
    res.status(500).json({ error: 'NHG validation failed' });
  }
});

// Property Valuation API
app.post('/api/property/valuation', async (req, res) => {
  try {
    const { address, property_type } = req.body;

    const valuation = await valuationService.getValuation(address, property_type);

    res.json({
      valuation,
      requested_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Property valuation API error:', error);
    res.status(500).json({ error: 'Property valuation failed' });
  }
});

// AFM Compliance Validation API
app.post('/api/afm/validate-compliance', async (req, res) => {
  try {
    const { requirement, context } = req.body;

    const compliance = await afmManager.validateCompliance(requirement, context);

    res.json({
      compliance,
      validated_at: new Date().toISOString()
    });

  } catch (error) {
    logger.error('AFM compliance validation API error:', error);
    res.status(500).json({ error: 'Compliance validation failed' });
  }
});

// Initialize AFM Regulations
app.post('/api/afm/initialize-regulations', async (req, res) => {
  try {
    await afmManager.initializeRegulations();
    res.json({
      status: 'success',
      message: 'AFM regulations initialized',
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('AFM regulations initialization failed:', error);
    res.status(500).json({ error: 'Failed to initialize regulations' });
  }
});

// Scheduled tasks
async function initializeScheduledTasks() {
  // Update AFM regulations daily at 2 AM
  cron.schedule('0 2 * * *', async () => {
    try {
      logger.info('Starting scheduled AFM regulation update');
      await afmManager.fetchAFMRegulations();
      logger.info('Scheduled AFM regulation update completed');
    } catch (error) {
      logger.error('Scheduled AFM regulation update failed:', error);
    }
  });

  // Clean up old cache entries hourly
  cron.schedule('0 * * * *', async () => {
    try {
      // Clean up expired Redis keys
      const keys = await redisClient.keys('bkr_credit_*');
      for (const key of keys) {
        const ttl = await redisClient.ttl(key);
        if (ttl < 0) {
          await redisClient.del(key);
        }
      }
      logger.info(`Cleaned up ${keys.length} cache entries`);
    } catch (error) {
      logger.error('Cache cleanup failed:', error);
    }
  });
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('Shutting down Dutch Market Data service...');

  if (dbClient) {
    await dbClient.end();
  }

  if (redisClient?.isOpen) {
    await redisClient.quit();
  }

  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('Received SIGINT, shutting down gracefully...');

  if (dbClient) {
    await dbClient.end();
  }

  if (redisClient?.isOpen) {
    await redisClient.quit();
  }

  process.exit(0);
});

// Initialization functions for use by server
async function initializeServices() {
  try {
    // Initialize connections
    await connectDatabase();
    await connectRedis();

    // Initialize AFM regulations
    await afmManager.initializeRegulations();

    // Start scheduled tasks
    await initializeScheduledTasks();

    logger.info(`📊 AFM Regulations: ${afmManager.regulationCache.size} loaded`);
    return true;
  } catch (error) {
    logger.error('Failed to initialize Dutch Market Data services:', error);
    throw error;
  }
}

// Export service classes for use by other modules
class AFMRegulationService {
  constructor() {
    this.afmManager = afmManager;
  }

  async checkRegulationCompliance(text, context) {
    return await this.afmManager.checkRegulationCompliance(text, context);
  }

  async getRegulationByCode(code) {
    return await this.afmManager.getRegulationByCode(code);
  }
}

// Export service classes and initialization function for use by other modules
module.exports = {
  AFMRegulationService,
  BKRCreditService,
  NHGValidationService,
  PropertyValuationService,
  initializeServices
};
