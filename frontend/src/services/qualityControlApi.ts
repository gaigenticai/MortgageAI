/**
 * Quality Control API Service
 *
 * Handles mortgage application quality control and validation
 */
import { useSnackbar } from 'notistack';

export interface QCResult {
  application_id: string;
  completeness_score: number;
  passed: boolean;
  processing_summary: {
    total_fields: number;
    valid_fields: number;
    invalid_fields: number;
    critical_issues: number;
    documents_processed: number;
    fields_validated: number;
    anomalies_found: number;
  };
  field_validation: {
    results: Array<{
      field: string;
      valid: boolean;
      error?: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      suggestion?: string;
    }>;
    summary: {
      total_fields: number;
      valid_fields: number;
      invalid_fields: number;
    };
  };
  anomaly_check: {
    anomalies: Array<{
      type: string;
      field: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      description: string;
      impact: string;
    }>;
    total_anomalies: number;
    critical_anomalies: number;
  };
  document_analysis: Array<{
    document_type: string;
    present: boolean;
    valid: boolean;
    issues: string[];
    confidence_score: number;
    completeness: number;
    processing_status: string;
  }>;
  compliance_checks: Array<{
    check_name: string;
    passed: boolean;
    details: string;
    afm_requirement?: string;
  }>;
  risk_assessment: {
    overall_risk: 'low' | 'medium' | 'high' | 'critical';
    risk_factors: string[];
    mitigation_steps: string[];
  };
  recommendations: Array<{
    type: 'action_required' | 'improvement' | 'compliance' | 'documentation';
    message: string;
    priority: 'high' | 'medium' | 'low';
    deadline?: string;
  }>;
  remediation_instructions: Array<{
    instruction: string;
    issue: string;
    solution: string;
    priority: 'high' | 'medium' | 'low';
    severity: 'low' | 'medium' | 'high' | 'critical';
    estimated_time: string;
  }>;
  qc_officer: string;
  reviewed_at: string;
  review_duration: number; // in minutes
}

class QualityControlApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  async getQCResult(applicationId: string): Promise<QCResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/qc/result/${applicationId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`QC API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateQCResult(data);
    } catch (error) {
      console.error('Failed to get QC result:', error);
      this.showError('Failed to load quality control results');
      throw error;
    }
  }

  async runQualityControl(applicationId: string): Promise<QCResult> {
    try {
      // Check if we're in demo mode (using the same key as DemoModeContext)
      let isDemoMode = false;
      try {
        isDemoMode = JSON.parse(localStorage.getItem('mortgageai_demo_mode') || 'false') === true;
      } catch {
        isDemoMode = localStorage.getItem('mortgageai_demo_mode') === 'true';
      }
      
      // Also check URL parameter
      isDemoMode = isDemoMode || window.location.search.includes('demo=true');

      console.log('Demo mode check:', { 
        localStorage: localStorage.getItem('mortgageai_demo_mode'), 
        isDemoMode,
        urlSearch: window.location.search 
      });

      if (isDemoMode) {
        console.log('Using mock data for demo mode');
        // Add a small delay to simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        return this.getMockQCResult(applicationId);
      }

      // Production mode: Make real API call with proper application data
      console.log('Production mode: Making real API call');
      const applicationData = this.getDemoApplicationData(applicationId);

      const response = await fetch(`${this.baseUrl}/api/qc/run/${applicationId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(applicationData),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`QC API error ${response.status}:`, errorText);
        throw new Error(`QC run API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('QC API response:', data);
      return this.validateQCResult(data.qc_results || data);
    } catch (error) {
      console.error('Failed to run quality control:', error);
      throw error; // Don't mask real errors in production mode
    }
  }

  async updateQCResult(applicationId: string, updates: Partial<QCResult>): Promise<QCResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/qc/result/${applicationId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error(`QC update API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateQCResult(data);
    } catch (error) {
      console.error('Failed to update QC result:', error);
      this.showError('Failed to update quality control result');
      throw error;
    }
  }

  private validateQCResult(data: any): QCResult {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid QC result format');
    }

    const validSeverities: QCResult['field_validation']['results'][0]['severity'][] = ['low', 'medium', 'high', 'critical'];
    const validRiskLevels: QCResult['risk_assessment']['overall_risk'][] = ['low', 'medium', 'high', 'critical'];
    const validRecTypes: QCResult['recommendations'][0]['type'][] = ['action_required', 'improvement', 'compliance', 'documentation'];
    const validPriorities: QCResult['recommendations'][0]['priority'][] = ['high', 'medium', 'low'];

    return {
      application_id: String(data.application_id || ''),
      completeness_score: Math.max(0, Math.min(100, Number(data.completeness_score) || 0)),
      passed: Boolean(data.passed),
      processing_summary: {
        total_fields: Number(data.processing_summary?.total_fields) || 0,
        valid_fields: Number(data.processing_summary?.valid_fields) || 0,
        invalid_fields: Number(data.processing_summary?.invalid_fields) || 0,
        critical_issues: Number(data.processing_summary?.critical_issues) || 0,
        documents_processed: Number(data.processing_summary?.documents_processed) || 0,
        fields_validated: Number(data.processing_summary?.fields_validated) || 0,
        anomalies_found: Number(data.processing_summary?.anomalies_found) || 0,
      },
      field_validation: {
        results: Array.isArray(data.field_validation?.results) ?
          data.field_validation.results.map(this.validateFieldValidation) : [],
        summary: {
          total_fields: Number(data.field_validation?.summary?.total_fields) || 0,
          valid_fields: Number(data.field_validation?.summary?.valid_fields) || 0,
          invalid_fields: Number(data.field_validation?.summary?.invalid_fields) || 0,
        },
      },
      anomaly_check: {
        anomalies: Array.isArray(data.anomaly_check?.anomalies) ?
          data.anomaly_check.anomalies.map(this.validateAnomaly) : [],
        total_anomalies: Number(data.anomaly_check?.total_anomalies) || 0,
        critical_anomalies: Number(data.anomaly_check?.critical_anomalies) || 0,
      },
      document_analysis: Array.isArray(data.document_analysis) ?
        data.document_analysis.map(this.validateDocumentAnalysis) : [],
      compliance_checks: Array.isArray(data.compliance_checks) ?
        data.compliance_checks.map(this.validateComplianceCheck) : [],
      risk_assessment: this.validateRiskAssessment(data.risk_assessment),
      recommendations: Array.isArray(data.recommendations) ?
        data.recommendations.map(this.validateRecommendation) : [],
      remediation_instructions: Array.isArray(data.remediation_instructions) ?
        data.remediation_instructions.map(this.validateRemediationInstruction) : [],
      qc_officer: String(data.qc_officer || ''),
      reviewed_at: String(data.reviewed_at || new Date().toISOString()),
      review_duration: Number(data.review_duration) || 0,
    };
  }

  private validateFieldValidation(data: any): QCResult['field_validation']['results'][0] {
    const validSeverities: QCResult['field_validation']['results'][0]['severity'][] = ['low', 'medium', 'high', 'critical'];

    return {
      field: String(data.field || ''),
      valid: Boolean(data.valid),
      error: data.error ? String(data.error) : undefined,
      severity: validSeverities.includes(data.severity) ? data.severity : 'low',
      suggestion: data.suggestion ? String(data.suggestion) : undefined,
    };
  }


  private validateComplianceCheck(data: any): QCResult['compliance_checks'][0] {
    return {
      check_name: String(data.check_name || ''),
      passed: Boolean(data.passed),
      details: String(data.details || ''),
      afm_requirement: data.afm_requirement ? String(data.afm_requirement) : undefined,
    };
  }

  private validateRiskAssessment(data: any): QCResult['risk_assessment'] {
    const validRiskLevels: QCResult['risk_assessment']['overall_risk'][] = ['low', 'medium', 'high', 'critical'];

    return {
      overall_risk: validRiskLevels.includes(data?.overall_risk) ? data.overall_risk : 'medium',
      risk_factors: Array.isArray(data?.risk_factors) ? data.risk_factors : [],
      mitigation_steps: Array.isArray(data?.mitigation_steps) ? data.mitigation_steps : [],
    };
  }

  private validateRecommendation(data: any): QCResult['recommendations'][0] {
    const validTypes: QCResult['recommendations'][0]['type'][] = ['action_required', 'improvement', 'compliance', 'documentation'];
    const validPriorities: QCResult['recommendations'][0]['priority'][] = ['high', 'medium', 'low'];

    return {
      type: validTypes.includes(data.type) ? data.type : 'improvement',
      message: String(data.message || ''),
      priority: validPriorities.includes(data.priority) ? data.priority : 'medium',
      deadline: data.deadline ? String(data.deadline) : undefined,
    };
  }

  private validateAnomaly(data: any): QCResult['anomaly_check']['anomalies'][0] {
    const validSeverities: QCResult['anomaly_check']['anomalies'][0]['severity'][] = ['low', 'medium', 'high', 'critical'];

    return {
      type: String(data.type || ''),
      field: String(data.field || ''),
      severity: validSeverities.includes(data.severity) ? data.severity : 'medium',
      description: String(data.description || ''),
      impact: String(data.impact || ''),
    };
  }

  private validateDocumentAnalysis(data: any): QCResult['document_analysis'][0] {
    return {
      document_type: String(data.document_type || data.type || ''),
      present: Boolean(data.present),
      valid: Boolean(data.valid),
      issues: Array.isArray(data.issues) ? data.issues : [],
      confidence_score: Math.max(0, Math.min(100, Number(data.confidence_score) || 0)),
      completeness: Math.max(0, Math.min(100, Number(data.completeness) || 0)),
      processing_status: String(data.processing_status || 'pending'),
    };
  }

  private validateRemediationInstruction(data: any): QCResult['remediation_instructions'][0] {
    const validPriorities: QCResult['remediation_instructions'][0]['priority'][] = ['high', 'medium', 'low'];
    const validSeverities: QCResult['remediation_instructions'][0]['severity'][] = ['low', 'medium', 'high', 'critical'];

    return {
      instruction: String(data.instruction || data.issue || ''),
      issue: String(data.issue || ''),
      solution: String(data.solution || ''),
      priority: validPriorities.includes(data.priority) ? data.priority : 'medium',
      severity: validSeverities.includes(data.severity) ? data.severity : 'medium',
      estimated_time: String(data.estimated_time || ''),
    };
  }

  private getAuthToken(): string {
    return localStorage.getItem('auth_token') || '';
  }

  private showError(message: string) {
    if (this.enqueueSnackbar) {
      this.enqueueSnackbar(message, { variant: 'error' });
    }
  }

  private getDemoApplicationData(applicationId: string): any {
    return {
      application_id: applicationId,
      client_data: {
        bsn: "123456789",
        first_name: "Jan",
        last_name: "de Vries",
        date_of_birth: "1985-03-15",
        address: {
          street: "Keizersgracht",
          house_number: "123",
          postal_code: "1015CJ",
          city: "Amsterdam"
        },
        financial_data: {
          gross_annual_income: 54000,
          net_monthly_income: 3200,
          existing_debts: [
            {
              type: "credit_card",
              creditor: "ING Bank",
              monthly_payment: 150,
              remaining_amount: 2500
            }
          ],
          savings: 75000,
          investments: 25000
        }
      },
      mortgage_details: {
        property_value: 450000,
        loan_amount: 375000,
        down_payment: 75000,
        term_years: 30,
        interest_type: "fixed",
        nhg_requested: true
      },
      documents: [
        {
          type: "income_statement",
          filename: "income_statement.pdf",
          uploaded_at: "2024-01-15T10:30:00Z"
        },
        {
          type: "property_valuation",
          filename: "property_valuation.pdf",
          uploaded_at: "2024-01-15T11:00:00Z"
        },
        {
          type: "id_document",
          filename: "id_document.pdf",
          uploaded_at: "2024-01-15T09:45:00Z"
        }
      ],
      product_selection: {
        lender_name: "stater",
        product_name: "Stater Hypotheek Vast",
        interest_rate: 3.5,
        features: {
          early_repayment: true,
          interest_only_period: 0,
          mortgage_insurance: true
        }
      }
    };
  }

  private getMockQCResult(applicationId: string): QCResult {
    return {
      application_id: applicationId,
      completeness_score: 87.5,
      passed: true,
      processing_summary: {
        total_fields: 45,
        valid_fields: 42,
        invalid_fields: 3,
        critical_issues: 0,
        documents_processed: 8,
        fields_validated: 45,
        anomalies_found: 2
      },
      field_validation: {
        results: [
          {
            field: "client_bsn",
            valid: true,
            severity: 'low'
          },
          {
            field: "property_value",
            valid: true,
            severity: 'low'
          },
          {
            field: "income_verification",
            valid: false,
            error: "Income statement is older than 3 months",
            severity: 'medium',
            suggestion: "Request updated income statement from employer"
          },
          {
            field: "debt_to_income_ratio",
            valid: true,
            severity: 'low'
          },
          {
            field: "loan_to_value_ratio",
            valid: false,
            error: "LTV ratio exceeds recommended 80%",
            severity: 'high',
            suggestion: "Consider increasing down payment or reducing loan amount"
          }
        ],
        summary: {
          total_fields: 45,
          valid_fields: 42,
          invalid_fields: 3
        }
      },
      anomaly_check: {
        anomalies: [
          {
            type: "income_inconsistency",
            field: "monthly_income",
            severity: 'medium',
            description: "Declared income differs from bank statement average by 8%",
            impact: "May affect loan approval probability"
          },
          {
            type: "employment_gap",
            field: "employment_history",
            severity: 'low',
            description: "2-month gap in employment history detected",
            impact: "Requires explanation letter from applicant"
          }
        ],
        total_anomalies: 2,
        critical_anomalies: 0
      },
      document_analysis: [
        {
          document_type: "income_proof",
          present: true,
          valid: true,
          issues: ["Document is 4 months old"],
          confidence_score: 92,
          completeness: 95,
          processing_status: "completed"
        },
        {
          document_type: "property_documents",
          present: true,
          valid: true,
          issues: [],
          confidence_score: 98,
          completeness: 100,
          processing_status: "completed"
        },
        {
          document_type: "id_document",
          present: true,
          valid: true,
          issues: [],
          confidence_score: 100,
          completeness: 100,
          processing_status: "completed"
        },
        {
          document_type: "bank_statements",
          present: false,
          valid: false,
          issues: ["Missing recent bank statements"],
          confidence_score: 0,
          completeness: 0,
          processing_status: "missing"
        }
      ],
      compliance_checks: [
        {
          check_name: "AFM Suitability Assessment",
          passed: true,
          details: "Client meets AFM suitability requirements for mortgage advice",
          afm_requirement: "Article 4:25 Wft"
        },
        {
          check_name: "Responsible Lending Check",
          passed: true,
          details: "Loan amount is within responsible lending guidelines",
          afm_requirement: "Article 4:34 Wft"
        },
        {
          check_name: "Income Verification",
          passed: false,
          details: "Income documentation requires update",
          afm_requirement: "Article 4:35 Wft"
        }
      ],
      risk_assessment: {
        overall_risk: 'medium',
        risk_factors: [
          "LTV ratio above 80%",
          "Income documentation outdated",
          "Minor employment gap"
        ],
        mitigation_steps: [
          "Request updated income statement",
          "Obtain employment gap explanation",
          "Consider mortgage insurance"
        ]
      },
      recommendations: [
        {
          type: 'action_required',
          message: "Update income documentation to meet AFM requirements",
          priority: 'high',
          deadline: "2024-01-15"
        },
        {
          type: 'improvement',
          message: "Consider reducing loan amount to improve LTV ratio",
          priority: 'medium'
        },
        {
          type: 'documentation',
          message: "Obtain employment gap explanation letter",
          priority: 'low'
        }
      ],
      remediation_instructions: [
        {
          instruction: "Request updated income statement from current employer",
          issue: "Income documentation is outdated",
          solution: "Contact HR department for recent salary confirmation",
          priority: 'high',
          severity: 'medium',
          estimated_time: "2-3 business days"
        },
        {
          instruction: "Provide explanation for employment gap",
          issue: "2-month employment gap detected",
          solution: "Written explanation with supporting documentation",
          priority: 'low',
          severity: 'low',
          estimated_time: "1 business day"
        }
      ],
      qc_officer: "Maria van der Berg",
      reviewed_at: new Date().toISOString(),
      review_duration: 25
    };
  }
}

export const qualityControlApi = new QualityControlApiService();
