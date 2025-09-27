/**
 * Comprehensive Demo Data Service
 * Provides realistic, production-like data for all MortgageAI features
 * Used when demo mode is enabled to showcase full functionality
 */

export interface DemoClient {
  id: string;
  name: string;
  email: string;
  phone: string;
  bsn: string;
  dateOfBirth: string;
  nationality: string;
  address: {
    street: string;
    houseNumber: string;
    postalCode: string;
    city: string;
    country: string;
  };
  employment: {
    employer: string;
    position: string;
    monthlyIncome: number;
    employmentType: string;
    startDate: string;
  };
  financial: {
    monthlyExpenses: number;
    existingDebts: number;
    savings: number;
    creditScore: number;
  };
}

export interface DemoProperty {
  id: string;
  address: string;
  postalCode: string;
  city: string;
  propertyType: string;
  constructionYear: number;
  energyLabel: string;
  value: number;
  purchasePrice: number;
  livingArea: number;
  plotSize: number;
  rooms: number;
  bedrooms: number;
  bathrooms: number;
}

export interface DemoApplication {
  id: string;
  applicationNumber: string;
  clientId: string;
  propertyId: string;
  loanAmount: number;
  loanToValue: number;
  interestRate: number;
  term: number;
  monthlyPayment: number;
  status: 'draft' | 'under_review' | 'approved' | 'rejected' | 'submitted';
  submissionDate: string;
  lastUpdated: string;
  selectedLenders: string[];
  nhgEligible: boolean;
  complianceScore: number;
  qualityScore: number;
}

export interface DemoBKRReport {
  id: string;
  clientId: string;
  requestId: string;
  creditScore: number;
  riskCategory: 'Low' | 'Medium' | 'High';
  totalDebt: number;
  activeLoans: number;
  paymentHistory: 'Excellent' | 'Good' | 'Fair' | 'Poor';
  reportDate: string;
  loans: Array<{
    type: string;
    amount: number;
    status: string;
    monthlyPayment: number;
    startDate: string;
  }>;
}

export interface DemoMarketData {
  averageHousePrice: number;
  priceChange: number;
  averageInterestRate: number;
  interestRateChange: number;
  mortgageVolume: number;
  volumeChange: number;
  timeOnMarket: number;
  timeOnMarketChange: number;
  regionalData: Array<{
    region: string;
    averagePrice: number;
    priceChange: number;
    volume: number;
  }>;
}

class DemoDataService {
  private clients: DemoClient[] = [
    {
      id: 'client-001',
      name: 'Sarah Johnson',
      email: 'sarah.johnson@email.com',
      phone: '+31 6 1234 5678',
      bsn: '123456789',
      dateOfBirth: '1985-03-15',
      nationality: 'Dutch',
      address: {
        street: 'Prinsengracht',
        houseNumber: '263',
        postalCode: '1016 GV',
        city: 'Amsterdam',
        country: 'Netherlands'
      },
      employment: {
        employer: 'ING Bank',
        position: 'Senior Software Engineer',
        monthlyIncome: 6500,
        employmentType: 'Permanent',
        startDate: '2020-01-15'
      },
      financial: {
        monthlyExpenses: 2800,
        existingDebts: 15000,
        savings: 85000,
        creditScore: 750
      }
    },
    {
      id: 'client-002',
      name: 'Michael Chen',
      email: 'michael.chen@email.com',
      phone: '+31 6 2345 6789',
      bsn: '234567890',
      dateOfBirth: '1990-07-22',
      nationality: 'Dutch',
      address: {
        street: 'Vondelstraat',
        houseNumber: '45',
        postalCode: '1054 GE',
        city: 'Amsterdam',
        country: 'Netherlands'
      },
      employment: {
        employer: 'Philips',
        position: 'Product Manager',
        monthlyIncome: 5800,
        employmentType: 'Permanent',
        startDate: '2019-06-01'
      },
      financial: {
        monthlyExpenses: 2400,
        existingDebts: 8000,
        savings: 65000,
        creditScore: 720
      }
    },
    {
      id: 'client-003',
      name: 'Emma van der Berg',
      email: 'emma.vandenberg@email.com',
      phone: '+31 6 3456 7890',
      bsn: '345678901',
      dateOfBirth: '1988-11-08',
      nationality: 'Dutch',
      address: {
        street: 'Keizersgracht',
        houseNumber: '123',
        postalCode: '1015 CJ',
        city: 'Amsterdam',
        country: 'Netherlands'
      },
      employment: {
        employer: 'ASML',
        position: 'Research Scientist',
        monthlyIncome: 7200,
        employmentType: 'Permanent',
        startDate: '2018-03-01'
      },
      financial: {
        monthlyExpenses: 3200,
        existingDebts: 25000,
        savings: 120000,
        creditScore: 780
      }
    }
  ];

  private properties: DemoProperty[] = [
    {
      id: 'property-001',
      address: 'Herengracht 456, Amsterdam',
      postalCode: '1017 BZ',
      city: 'Amsterdam',
      propertyType: 'Apartment',
      constructionYear: 1920,
      energyLabel: 'C',
      value: 650000,
      purchasePrice: 625000,
      livingArea: 85,
      plotSize: 0,
      rooms: 4,
      bedrooms: 2,
      bathrooms: 1
    },
    {
      id: 'property-002',
      address: 'Jordaan 78, Amsterdam',
      postalCode: '1016 DW',
      city: 'Amsterdam',
      propertyType: 'House',
      constructionYear: 1965,
      energyLabel: 'B',
      value: 750000,
      purchasePrice: 720000,
      livingArea: 120,
      plotSize: 150,
      rooms: 5,
      bedrooms: 3,
      bathrooms: 2
    },
    {
      id: 'property-003',
      address: 'Museumplein 12, Amsterdam',
      postalCode: '1071 DJ',
      city: 'Amsterdam',
      propertyType: 'Apartment',
      constructionYear: 2010,
      energyLabel: 'A',
      value: 850000,
      purchasePrice: 825000,
      livingArea: 110,
      plotSize: 0,
      rooms: 4,
      bedrooms: 3,
      bathrooms: 2
    }
  ];

  private applications: DemoApplication[] = [
    {
      id: 'app-001',
      applicationNumber: 'MA-2025-001',
      clientId: 'client-001',
      propertyId: 'property-001',
      loanAmount: 540000,
      loanToValue: 83.1,
      interestRate: 3.85,
      term: 30,
      monthlyPayment: 2520,
      status: 'approved',
      submissionDate: '2025-09-20T10:00:00Z',
      lastUpdated: '2025-09-26T14:30:00Z',
      selectedLenders: ['ING', 'ABN AMRO'],
      nhgEligible: false,
      complianceScore: 94.2,
      qualityScore: 87.5
    },
    {
      id: 'app-002',
      applicationNumber: 'MA-2025-002',
      clientId: 'client-002',
      propertyId: 'property-002',
      loanAmount: 576000,
      loanToValue: 80.0,
      interestRate: 3.75,
      term: 30,
      monthlyPayment: 2665,
      status: 'under_review',
      submissionDate: '2025-09-25T09:15:00Z',
      lastUpdated: '2025-09-26T16:45:00Z',
      selectedLenders: ['Rabobank', 'SNS Bank'],
      nhgEligible: true,
      complianceScore: 91.8,
      qualityScore: 85.2
    },
    {
      id: 'app-003',
      applicationNumber: 'MA-2025-003',
      clientId: 'client-003',
      propertyId: 'property-003',
      loanAmount: 660000,
      loanToValue: 80.0,
      interestRate: 3.65,
      term: 25,
      monthlyPayment: 3125,
      status: 'draft',
      submissionDate: '2025-09-26T08:00:00Z',
      lastUpdated: '2025-09-26T17:20:00Z',
      selectedLenders: ['ING', 'Volksbank'],
      nhgEligible: false,
      complianceScore: 89.5,
      qualityScore: 82.1
    }
  ];

  private bkrReports: DemoBKRReport[] = [
    {
      id: 'bkr-001',
      clientId: 'client-001',
      requestId: 'BKR-2025-001',
      creditScore: 750,
      riskCategory: 'Low',
      totalDebt: 25000,
      activeLoans: 2,
      paymentHistory: 'Excellent',
      reportDate: '2025-09-26T12:00:00Z',
      loans: [
        {
          type: 'Mortgage',
          amount: 200000,
          status: 'Active',
          monthlyPayment: 850,
          startDate: '2020-01-15'
        },
        {
          type: 'Personal Loan',
          amount: 15000,
          status: 'Active',
          monthlyPayment: 250,
          startDate: '2023-06-01'
        }
      ]
    },
    {
      id: 'bkr-002',
      clientId: 'client-002',
      requestId: 'BKR-2025-002',
      creditScore: 720,
      riskCategory: 'Low',
      totalDebt: 18000,
      activeLoans: 1,
      paymentHistory: 'Good',
      reportDate: '2025-09-25T14:30:00Z',
      loans: [
        {
          type: 'Car Loan',
          amount: 18000,
          status: 'Active',
          monthlyPayment: 320,
          startDate: '2022-03-15'
        }
      ]
    }
  ];

  private marketData: DemoMarketData = {
    averageHousePrice: 425000,
    priceChange: 2.3,
    averageInterestRate: 3.75,
    interestRateChange: -0.15,
    mortgageVolume: 18500,
    volumeChange: 8.2,
    timeOnMarket: 28,
    timeOnMarketChange: -5,
    regionalData: [
      { region: 'Amsterdam', averagePrice: 650000, priceChange: 3.1, volume: 2850 },
      { region: 'Rotterdam', averagePrice: 385000, priceChange: 2.8, volume: 1920 },
      { region: 'The Hague', averagePrice: 485000, priceChange: 2.1, volume: 1450 },
      { region: 'Utrecht', averagePrice: 525000, priceChange: 3.5, volume: 1680 },
      { region: 'Eindhoven', averagePrice: 365000, priceChange: 1.9, volume: 980 }
    ]
  };

  // Dashboard Data
  getDashboardMetrics() {
    return {
      afm_compliance_score: 94.2,
      active_sessions: 12,
      pending_reviews: 3,
      applications_processed_today: 8,
      first_time_right_rate: 87.5,
      avg_processing_time_minutes: 45,
      total_applications: 156,
      approved_applications: 134,
      rejected_applications: 8,
      pending_applications: 14,
      total_clients: 89,
      active_lenders: 8,
      system_uptime: 99.8,
      api_response_time: 245
    };
  }

  getRecentActivity() {
    return {
      activities: [
        {
          type: 'afm_compliance',
          action: 'Client intake validated for mortgage application',
          client_name: 'Sarah Johnson',
          timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
          result: 'approved',
          details: 'All AFM compliance checks passed successfully'
        },
        {
          type: 'dutch_mortgage_qc',
          action: 'BKR credit check completed successfully',
          client_name: 'Michael Chen',
          timestamp: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
          result: 'passed',
          details: 'Credit score: 720, Risk category: Low'
        },
        {
          type: 'afm_compliance',
          action: 'NHG eligibility assessment completed',
          client_name: 'Emma van der Berg',
          timestamp: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
          result: 'eligible',
          details: 'Property value within NHG limits'
        },
        {
          type: 'dutch_mortgage_qc',
          action: 'Property valuation retrieved from Kadaster',
          client_name: 'Jan de Vries',
          timestamp: new Date(Date.now() - 1000 * 60 * 75).toISOString(),
          result: 'completed',
          details: 'WOZ value: €485,000'
        },
        {
          type: 'afm_compliance',
          action: 'Risk assessment completed for high-value loan',
          client_name: 'Lisa Wong',
          timestamp: new Date(Date.now() - 1000 * 60 * 90).toISOString(),
          result: 'approved',
          details: 'Loan amount: €750,000, LTV: 75%'
        }
      ]
    };
  }

  // Client Data
  getClients(): DemoClient[] {
    return this.clients;
  }

  getClient(id: string): DemoClient | undefined {
    return this.clients.find(client => client.id === id);
  }

  // Property Data
  getProperties(): DemoProperty[] {
    return this.properties;
  }

  getProperty(id: string): DemoProperty | undefined {
    return this.properties.find(property => property.id === id);
  }

  // Application Data
  getApplications(): DemoApplication[] {
    return this.applications;
  }

  getApplication(id: string): DemoApplication | undefined {
    return this.applications.find(app => app.id === id);
  }

  // BKR Data
  getBKRReport(clientId: string): DemoBKRReport | undefined {
    return this.bkrReports.find(report => report.clientId === clientId);
  }

  getBKRReports(): DemoBKRReport[] {
    return this.bkrReports;
  }

  // Market Data
  getMarketData(): DemoMarketData {
    return this.marketData;
  }

  // Quality Control Data
  getQualityControlResults(applicationId: string) {
    const application = this.getApplication(applicationId);
    if (!application) return null;

    return {
      applicationId,
      overallScore: application.qualityScore,
      completenessScore: 92.5,
      accuracyScore: application.qualityScore,
      complianceScore: application.complianceScore,
      fieldValidation: {
        personalInfo: { score: 95, issues: [] },
        employment: { score: 88, issues: ['Missing recent payslip'] },
        financial: { score: 92, issues: [] },
        property: { score: 90, issues: ['Energy certificate pending'] },
        documents: { score: 85, issues: ['Bank statement quality low', 'ID document expires soon'] }
      },
      anomalies: [
        {
          type: 'warning',
          field: 'monthly_income',
          description: 'Income variation detected in recent months',
          severity: 'medium',
          recommendation: 'Request additional income verification'
        },
        {
          type: 'info',
          field: 'property_value',
          description: 'Property value above regional average',
          severity: 'low',
          recommendation: 'Consider additional valuation'
        }
      ],
      recommendations: [
        'Request updated bank statements for last 3 months',
        'Verify employment contract renewal date',
        'Obtain energy performance certificate'
      ]
    };
  }

  // Compliance Data
  getComplianceAnalysis(applicationId: string) {
    const application = this.getApplication(applicationId);
    if (!application) return null;

    return {
      applicationId,
      overallScore: application.complianceScore,
      afmCompliance: {
        score: application.complianceScore,
        checks: [
          { name: 'Client Suitability Assessment', status: 'passed', score: 95 },
          { name: 'Affordability Calculation', status: 'passed', score: 92 },
          { name: 'Risk Profile Matching', status: 'passed', score: 88 },
          { name: 'Documentation Requirements', status: 'warning', score: 85 },
          { name: 'Advice Documentation', status: 'passed', score: 96 }
        ]
      },
      riskAssessment: {
        level: 'Low',
        factors: [
          { name: 'Income Stability', risk: 'Low', impact: 'Minimal' },
          { name: 'Debt-to-Income Ratio', risk: 'Low', impact: 'Minimal' },
          { name: 'Property Value Risk', risk: 'Medium', impact: 'Moderate' },
          { name: 'Market Conditions', risk: 'Low', impact: 'Minimal' }
        ]
      },
      recommendations: [
        'Ensure all AFM documentation is complete before submission',
        'Consider additional property valuation for high-value properties',
        'Maintain regular compliance monitoring throughout process'
      ]
    };
  }

  // Document Processing Data
  getDocumentProcessingResults() {
    return {
      totalDocuments: 24,
      processedDocuments: 22,
      pendingDocuments: 2,
      failedDocuments: 0,
      documents: [
        {
          id: 'doc-001',
          name: 'Passport - Sarah Johnson',
          type: 'ID Document',
          status: 'processed',
          confidence: 98.5,
          extractedData: {
            name: 'Sarah Johnson',
            dateOfBirth: '1985-03-15',
            nationality: 'Dutch',
            documentNumber: 'NL123456789'
          }
        },
        {
          id: 'doc-002',
          name: 'Employment Contract - Sarah Johnson',
          type: 'Employment Document',
          status: 'processed',
          confidence: 95.2,
          extractedData: {
            employer: 'ING Bank',
            position: 'Senior Software Engineer',
            salary: '€6,500/month',
            startDate: '2020-01-15'
          }
        },
        {
          id: 'doc-003',
          name: 'Bank Statement - Sarah Johnson',
          type: 'Financial Document',
          status: 'processing',
          confidence: null,
          extractedData: null
        }
      ]
    };
  }

  // Lender Integration Data
  getLenderIntegrations() {
    return {
      connectedLenders: [
        {
          id: 'ing',
          name: 'ING Bank',
          status: 'connected',
          lastSync: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
          apiVersion: 'v2.1',
          successRate: 98.5,
          avgResponseTime: 245,
          supportedProducts: ['Mortgage', 'NHG', 'Investment Property']
        },
        {
          id: 'abn',
          name: 'ABN AMRO',
          status: 'connected',
          lastSync: new Date(Date.now() - 1000 * 60 * 8).toISOString(),
          apiVersion: 'v1.8',
          successRate: 97.2,
          avgResponseTime: 312,
          supportedProducts: ['Mortgage', 'NHG']
        },
        {
          id: 'rabobank',
          name: 'Rabobank',
          status: 'connected',
          lastSync: new Date(Date.now() - 1000 * 60 * 22).toISOString(),
          apiVersion: 'v2.0',
          successRate: 96.8,
          avgResponseTime: 198,
          supportedProducts: ['Mortgage', 'NHG', 'Green Mortgage']
        },
        {
          id: 'sns',
          name: 'SNS Bank',
          status: 'warning',
          lastSync: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
          apiVersion: 'v1.5',
          successRate: 94.1,
          avgResponseTime: 445,
          supportedProducts: ['Mortgage']
        }
      ],
      recentSubmissions: [
        {
          id: 'sub-001',
          applicationId: 'app-001',
          lender: 'ING Bank',
          status: 'approved',
          submittedAt: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
          responseAt: new Date(Date.now() - 1000 * 60 * 30).toISOString()
        },
        {
          id: 'sub-002',
          applicationId: 'app-002',
          lender: 'Rabobank',
          status: 'pending',
          submittedAt: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
          responseAt: null
        }
      ]
    };
  }

  // NHG Eligibility Data
  getNHGEligibility(applicationId: string) {
    const application = this.getApplication(applicationId);
    if (!application) return null;

    return {
      applicationId,
      eligible: application.nhgEligible,
      maxLoanAmount: 435000, // 2025 NHG limit
      propertyValue: application.propertyId ? this.getProperty(application.propertyId)?.value : 0,
      loanAmount: application.loanAmount,
      checks: [
        {
          name: 'Property Value Limit',
          status: application.loanAmount <= 435000 ? 'passed' : 'failed',
          details: `Property value: €${application.propertyId ? this.getProperty(application.propertyId)?.value.toLocaleString() : 0}`
        },
        {
          name: 'First-Time Buyer',
          status: 'passed',
          details: 'Client qualifies as first-time buyer'
        },
        {
          name: 'Income Requirements',
          status: 'passed',
          details: 'Income meets NHG criteria'
        },
        {
          name: 'Loan-to-Value Ratio',
          status: application.loanToValue <= 100 ? 'passed' : 'failed',
          details: `LTV: ${application.loanToValue}%`
        }
      ],
      benefits: [
        'Lower interest rate (typically 0.1-0.2% reduction)',
        'Protection against residual debt',
        'Government backing for lender security',
        'Favorable terms for early repayment'
      ]
    };
  }

  // Settings Data
  getSystemSettings() {
    return {
      general: {
        companyName: 'MortgageAI',
        language: 'Dutch',
        timezone: 'Europe/Amsterdam',
        currency: 'EUR',
        dateFormat: 'DD-MM-YYYY'
      },
      afm: {
        complianceLevel: 'Full',
        autoValidation: true,
        requireDigitalSignature: true,
        documentRetention: 7, // years
        auditTrailEnabled: true
      },
      integrations: {
        bkrEnabled: true,
        kadasterEnabled: true,
        nhgEnabled: true,
        lenderApiTimeout: 30, // seconds
        maxRetries: 3
      },
      notifications: {
        emailNotifications: true,
        smsNotifications: false,
        pushNotifications: true,
        complianceAlerts: true,
        systemAlerts: true
      },
      security: {
        sessionTimeout: 30, // minutes
        passwordPolicy: 'Strong',
        twoFactorAuth: true,
        ipWhitelist: ['192.168.1.0/24'],
        encryptionLevel: 'AES-256'
      }
    };
  }

  // Audit Trail Data
  getAuditTrail() {
    return {
      events: [
        {
          id: 'audit-001',
          timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
          userId: 'user-001',
          userName: 'Sarah Johnson',
          action: 'APPLICATION_SUBMITTED',
          resource: 'Application MA-2025-001',
          details: 'Mortgage application submitted to ING Bank',
          ipAddress: '192.168.1.100',
          userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
          result: 'SUCCESS'
        },
        {
          id: 'audit-002',
          timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
          userId: 'system',
          userName: 'System',
          action: 'BKR_CHECK_COMPLETED',
          resource: 'Client client-001',
          details: 'BKR credit check completed successfully',
          ipAddress: '10.0.0.1',
          userAgent: 'MortgageAI-System/1.0',
          result: 'SUCCESS'
        },
        {
          id: 'audit-003',
          timestamp: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
          userId: 'user-002',
          userName: 'Michael Chen',
          action: 'DOCUMENT_UPLOADED',
          resource: 'Document doc-003',
          details: 'Bank statement uploaded for verification',
          ipAddress: '192.168.1.101',
          userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
          result: 'SUCCESS'
        }
      ]
    };
  }
}

export const demoDataService = new DemoDataService();
