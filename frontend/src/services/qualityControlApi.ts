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
      const response = await fetch(`${this.baseUrl}/api/qc/run/${applicationId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`QC run API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateQCResult(data);
    } catch (error) {
      console.error('Failed to run quality control:', error);
      this.showError('Failed to run quality control');
      throw error;
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
}

export const qualityControlApi = new QualityControlApiService();
