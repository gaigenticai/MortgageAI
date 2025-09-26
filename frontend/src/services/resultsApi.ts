/**
 * Results Display API Service
 *
 * Handles final mortgage application results combining AFM compliance and QC
 */
import { useSnackbar } from 'notistack';

export interface FinalResults {
  application_id: string;
  overall_status: 'approved' | 'conditional' | 'rejected' | 'pending_review';
  overall_score: number;
  afm_compliance: {
    compliance_score: number;
    risk_profile: 'low' | 'medium' | 'high';
    afm_status: 'compliant' | 'conditional' | 'non_compliant';
    recommendations: string[];
  };
  quality_control: {
    completeness_score: number;
    passed: boolean;
    critical_issues: number;
    field_validation: Array<{
      field: string;
      valid: boolean;
      error?: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
    }>;
    recommendations: string[];
  };
  financial_analysis: {
    dti_ratio: number;
    ltv_ratio: number;
    affordability_score: number;
    risk_factors: string[];
  };
  lender_matches: Array<{
    lender: string;
    product: string;
    interest_rate: number;
    max_ltv: number;
    eligibility_score: number;
    conditions: string[];
  }>;
  next_steps: Array<{
    step: string;
    priority: 'high' | 'medium' | 'low';
    deadline?: string;
    responsible_party: string;
  }>;
  compliance: {
    compliance_score: number;
    readability_level: string;
    advice_generated: boolean;
    understanding_confirmed: boolean;
  };
  final_advice: {
    summary: string;
    key_recommendations: string[];
    next_steps: string[];
    risk_assessment: string;
  };
  processing_timeline: Array<{
    step: string;
    status: 'completed' | 'in_progress' | 'pending';
    timestamp: string | null;
    details: string;
  }>;
  generated_at: string;
  valid_until: string;
}

class ResultsApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  async getFinalResults(applicationId: string): Promise<FinalResults> {
    try {
      const response = await fetch(`${this.baseUrl}/api/results/final/${applicationId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Results API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateFinalResults(data);
    } catch (error) {
      console.error('Failed to get final results:', error);
      this.showError('Failed to load final results');
      throw error;
    }
  }

  async generateFinalResults(applicationId: string): Promise<FinalResults> {
    try {
      const response = await fetch(`${this.baseUrl}/api/results/generate/${applicationId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Results generation API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateFinalResults(data);
    } catch (error) {
      console.error('Failed to generate final results:', error);
      this.showError('Failed to generate final results');
      throw error;
    }
  }

  async exportResults(applicationId: string, format: 'pdf' | 'json' = 'pdf'): Promise<Blob> {
    try {
      const response = await fetch(`${this.baseUrl}/api/results/export/${applicationId}?format=${format}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Results export API error: ${response.status} ${response.statusText}`);
      }

      return await response.blob();
    } catch (error) {
      console.error('Failed to export results:', error);
      this.showError('Failed to export results');
      throw error;
    }
  }

  private validateFinalResults(data: any): FinalResults {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid final results format');
    }

    const validStatuses: FinalResults['overall_status'][] = ['approved', 'conditional', 'rejected', 'pending_review'];
    const validRiskProfiles: FinalResults['afm_compliance']['risk_profile'][] = ['low', 'medium', 'high'];
    const validAFMStatuses: FinalResults['afm_compliance']['afm_status'][] = ['compliant', 'conditional', 'non_compliant'];
    const validSeverities: FinalResults['quality_control']['field_validation'][0]['severity'][] = ['low', 'medium', 'high', 'critical'];
    const validPriorities: FinalResults['next_steps'][0]['priority'][] = ['high', 'medium', 'low'];

    return {
      application_id: String(data.application_id || ''),
      overall_status: validStatuses.includes(data.overall_status) ? data.overall_status : 'pending_review',
      overall_score: Math.max(0, Math.min(100, Number(data.overall_score) || 0)),
      afm_compliance: {
        compliance_score: Math.max(0, Math.min(100, Number(data.afm_compliance?.compliance_score) || 0)),
        risk_profile: validRiskProfiles.includes(data.afm_compliance?.risk_profile) ? data.afm_compliance.risk_profile : 'medium',
        afm_status: validAFMStatuses.includes(data.afm_compliance?.afm_status) ? data.afm_compliance.afm_status : 'non_compliant',
        recommendations: Array.isArray(data.afm_compliance?.recommendations) ? data.afm_compliance.recommendations : [],
      },
      quality_control: {
        completeness_score: Math.max(0, Math.min(100, Number(data.quality_control?.completeness_score) || 0)),
        passed: Boolean(data.quality_control?.passed),
        critical_issues: Number(data.quality_control?.critical_issues) || 0,
        field_validation: Array.isArray(data.quality_control?.field_validation) ?
          data.quality_control.field_validation.map(this.validateFieldValidation) : [],
        recommendations: Array.isArray(data.quality_control?.recommendations) ? data.quality_control.recommendations : [],
      },
      financial_analysis: {
        dti_ratio: Number(data.financial_analysis?.dti_ratio) || 0,
        ltv_ratio: Number(data.financial_analysis?.ltv_ratio) || 0,
        affordability_score: Math.max(0, Math.min(100, Number(data.financial_analysis?.affordability_score) || 0)),
        risk_factors: Array.isArray(data.financial_analysis?.risk_factors) ? data.financial_analysis.risk_factors : [],
      },
      compliance: {
        compliance_score: Math.max(0, Math.min(100, Number(data.compliance?.compliance_score) || 0)),
        readability_level: String(data.compliance?.readability_level || 'Unknown'),
        advice_generated: Boolean(data.compliance?.advice_generated),
        understanding_confirmed: Boolean(data.compliance?.understanding_confirmed),
      },
      final_advice: {
        summary: String(data.final_advice?.summary || ''),
        key_recommendations: Array.isArray(data.final_advice?.key_recommendations) ? data.final_advice.key_recommendations : [],
        next_steps: Array.isArray(data.final_advice?.next_steps) ? data.final_advice.next_steps : [],
        risk_assessment: String(data.final_advice?.risk_assessment || ''),
      },
      lender_matches: Array.isArray(data.lender_matches) ? data.lender_matches.map(this.validateLenderMatch) : [],
      next_steps: Array.isArray(data.next_steps) ? data.next_steps.map(this.validateNextStep) : [],
      processing_timeline: Array.isArray(data.processing_timeline) ? data.processing_timeline.map(this.validateTimelineStep) : [],
      generated_at: String(data.generated_at || new Date().toISOString()),
      valid_until: String(data.valid_until || ''),
    };
  }

  private validateFieldValidation(data: any): FinalResults['quality_control']['field_validation'][0] {
    const validSeverities: FinalResults['quality_control']['field_validation'][0]['severity'][] = ['low', 'medium', 'high', 'critical'];

    return {
      field: String(data.field || ''),
      valid: Boolean(data.valid),
      error: data.error ? String(data.error) : undefined,
      severity: validSeverities.includes(data.severity) ? data.severity : 'low',
    };
  }

  private validateLenderMatch(data: any): FinalResults['lender_matches'][0] {
    return {
      lender: String(data.lender || ''),
      product: String(data.product || ''),
      interest_rate: Number(data.interest_rate) || 0,
      max_ltv: Math.max(0, Math.min(100, Number(data.max_ltv) || 0)),
      eligibility_score: Math.max(0, Math.min(100, Number(data.eligibility_score) || 0)),
      conditions: Array.isArray(data.conditions) ? data.conditions : [],
    };
  }

  private validateNextStep(data: any): FinalResults['next_steps'][0] {
    const validPriorities: FinalResults['next_steps'][0]['priority'][] = ['high', 'medium', 'low'];

    return {
      step: String(data.step || ''),
      priority: validPriorities.includes(data.priority) ? data.priority : 'medium',
      deadline: data.deadline ? String(data.deadline) : undefined,
      responsible_party: String(data.responsible_party || ''),
    };
  }

  private validateTimelineStep(data: any): FinalResults['processing_timeline'][0] {
    const validStatuses: FinalResults['processing_timeline'][0]['status'][] = ['completed', 'in_progress', 'pending'];

    return {
      step: String(data.step || ''),
      status: validStatuses.includes(data.status) ? data.status : 'pending',
      timestamp: data.timestamp ? String(data.timestamp) : null,
      details: String(data.details || ''),
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

export const resultsApi = new ResultsApiService();
