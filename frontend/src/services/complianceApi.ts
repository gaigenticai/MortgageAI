/**
 * AFM Compliance Assessment API Service
 *
 * Provides comprehensive AFM compliance analysis and assessment
 * for mortgage advice and product recommendations
 */
import { useSnackbar } from 'notistack';

export interface ComplianceRecommendation {
  id: string;
  type: 'approved' | 'conditional' | 'rejected' | 'review_required';
  title: string;
  description: string;
  risk_level: 'low' | 'medium' | 'high';
  afm_requirements: string[];
  recommended_actions: string[];
  deadline?: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export interface ProductRecommendation {
  id: string;
  lender: string;
  lender_id: string;
  product_name: string;
  product_type: 'fixed_rate' | 'variable_rate' | 'green_mortgage' | 'investment_mortgage';
  interest_rate: number;
  max_ltv: number;
  suitability_score: number;
  afm_compliant: boolean;
  conditions?: string[];
  term_years: number;
  nhg_required?: boolean;
  energy_efficiency_bonus?: number;
  estimated_monthly_payment?: number;
  total_costs?: number;
}

export interface ComplianceAnalysis {
  id: string;
  client_id: string;
  client_name: string;
  assessment_date: string;
  compliance_score: number;
  risk_profile: 'low' | 'medium' | 'high';
  afm_status: 'compliant' | 'conditional' | 'non_compliant';
  overall_status: 'passed' | 'conditional_approval' | 'requires_review' | 'rejected';
  recommendations: ComplianceRecommendation[];
  product_recommendations: ProductRecommendation[];
  review_deadline?: string;
  advisor_notes?: string;
  next_review_date?: string;
  compliance_flags: string[];
  regulatory_requirements: {
    wft_article_86f: boolean;
    suitability_assessment: boolean;
    product_governance: boolean;
    client_categorization: boolean;
  };
}

export interface ComplianceAssessmentRequest {
  client_id: string;
  assessment_type: 'initial' | 'annual_review' | 'product_change' | 'complaint';
  include_product_recommendations: boolean;
  priority: 'normal' | 'urgent' | 'critical';
}

export interface AssessmentHistory {
  id: string;
  client_id: string;
  assessment_date: string;
  score: number;
  status: string;
  assessor: string;
  changes_from_previous?: string[];
}

class ComplianceApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  /**
   * Set snackbar function for error notifications
   */
  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  /**
   * Request AFM compliance assessment
   */
  async requestComplianceAssessment(request: ComplianceAssessmentRequest): Promise<{ assessment_id: string; estimated_completion: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance/assessment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Compliance assessment API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        assessment_id: String(data.assessment_id || ''),
        estimated_completion: String(data.estimated_completion || ''),
      };
    } catch (error) {
      console.error('Failed to request compliance assessment:', error);
      this.showError('Failed to request AFM compliance assessment');
      throw error;
    }
  }

  /**
   * Get compliance assessment by ID
   */
  async getComplianceAssessment(assessmentId: string): Promise<ComplianceAnalysis> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance/assessment/${assessmentId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Compliance assessment API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateComplianceAnalysis(data);
    } catch (error) {
      console.error('Failed to get compliance assessment:', error);
      this.showError('Failed to load AFM compliance assessment');
      throw error;
    }
  }

  /**
   * Get compliance assessment by client ID
   */
  async getClientComplianceAssessment(clientId: string): Promise<ComplianceAnalysis> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance/client/${clientId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Client compliance API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateComplianceAnalysis(data);
    } catch (error) {
      console.error('Failed to get client compliance assessment:', error);
      this.showError('Failed to load client AFM compliance assessment');
      throw error;
    }
  }

  /**
   * Update compliance assessment
   */
  async updateComplianceAssessment(assessmentId: string, updates: Partial<ComplianceAnalysis>): Promise<ComplianceAnalysis> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance/assessment/${assessmentId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error(`Update compliance assessment API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateComplianceAnalysis(data);
    } catch (error) {
      console.error('Failed to update compliance assessment:', error);
      this.showError('Failed to update AFM compliance assessment');
      throw error;
    }
  }

  /**
   * Approve compliance assessment
   */
  async approveComplianceAssessment(assessmentId: string, notes?: string): Promise<{ success: boolean; approval_date: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance/assessment/${assessmentId}/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify({ notes }),
      });

      if (!response.ok) {
        throw new Error(`Approve compliance assessment API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: Boolean(data.success),
        approval_date: String(data.approval_date || new Date().toISOString()),
      };
    } catch (error) {
      console.error('Failed to approve compliance assessment:', error);
      this.showError('Failed to approve AFM compliance assessment');
      throw error;
    }
  }

  /**
   * Reject compliance assessment
   */
  async rejectComplianceAssessment(assessmentId: string, reason: string, notes?: string): Promise<{ success: boolean; rejection_date: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance/assessment/${assessmentId}/reject`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify({ reason, notes }),
      });

      if (!response.ok) {
        throw new Error(`Reject compliance assessment API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: Boolean(data.success),
        rejection_date: String(data.rejection_date || new Date().toISOString()),
      };
    } catch (error) {
      console.error('Failed to reject compliance assessment:', error);
      this.showError('Failed to reject AFM compliance assessment');
      throw error;
    }
  }

  /**
   * Get assessment history for client
   */
  async getAssessmentHistory(clientId: string, limit: number = 10): Promise<AssessmentHistory[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance/history/${clientId}?limit=${limit}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Assessment history API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.history?.map(this.validateAssessmentHistory) || [];
    } catch (error) {
      console.error('Failed to get assessment history:', error);
      this.showError('Failed to load assessment history');
      throw error;
    }
  }

  /**
   * Generate compliance report
   */
  async generateComplianceReport(assessmentId: string, format: 'pdf' | 'html' | 'json' = 'pdf'): Promise<Blob> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance/report/${assessmentId}?format=${format}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Compliance report API error: ${response.status} ${response.statusText}`);
      }

      return await response.blob();
    } catch (error) {
      console.error('Failed to generate compliance report:', error);
      this.showError('Failed to generate AFM compliance report');
      throw error;
    }
  }

  /**
   * Validate compliance analysis data structure
   */
  private validateComplianceAnalysis(data: any): ComplianceAnalysis {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid compliance analysis format');
    }

    const validStatuses: ComplianceAnalysis['afm_status'][] = ['compliant', 'conditional', 'non_compliant'];
    const validOverallStatuses: ComplianceAnalysis['overall_status'][] = ['passed', 'conditional_approval', 'requires_review', 'rejected'];
    const validRiskProfiles: ComplianceAnalysis['risk_profile'][] = ['low', 'medium', 'high'];

    return {
      id: String(data.id || ''),
      client_id: String(data.client_id || ''),
      client_name: String(data.client_name || ''),
      assessment_date: String(data.assessment_date || new Date().toISOString()),
      compliance_score: Math.max(0, Math.min(100, Number(data.compliance_score) || 0)),
      risk_profile: validRiskProfiles.includes(data.risk_profile) ? data.risk_profile : 'medium',
      afm_status: validStatuses.includes(data.afm_status) ? data.afm_status : 'non_compliant',
      overall_status: validOverallStatuses.includes(data.overall_status) ? data.overall_status : 'requires_review',
      recommendations: Array.isArray(data.recommendations) ? data.recommendations.map(this.validateComplianceRecommendation) : [],
      product_recommendations: Array.isArray(data.product_recommendations) ? data.product_recommendations.map(this.validateProductRecommendation) : [],
      review_deadline: data.review_deadline,
      advisor_notes: data.advisor_notes,
      next_review_date: data.next_review_date,
      compliance_flags: Array.isArray(data.compliance_flags) ? data.compliance_flags : [],
      regulatory_requirements: {
        wft_article_86f: Boolean(data.regulatory_requirements?.wft_article_86f),
        suitability_assessment: Boolean(data.regulatory_requirements?.suitability_assessment),
        product_governance: Boolean(data.regulatory_requirements?.product_governance),
        client_categorization: Boolean(data.regulatory_requirements?.client_categorization),
      },
    };
  }

  /**
   * Validate compliance recommendation
   */
  private validateComplianceRecommendation(data: any): ComplianceRecommendation {
    const validTypes: ComplianceRecommendation['type'][] = ['approved', 'conditional', 'rejected', 'review_required'];
    const validRiskLevels: ComplianceRecommendation['risk_level'][] = ['low', 'medium', 'high'];
    const validPriorities: ComplianceRecommendation['priority'][] = ['low', 'medium', 'high', 'critical'];

    return {
      id: String(data.id || ''),
      type: validTypes.includes(data.type) ? data.type : 'review_required',
      title: String(data.title || ''),
      description: String(data.description || ''),
      risk_level: validRiskLevels.includes(data.risk_level) ? data.risk_level : 'medium',
      afm_requirements: Array.isArray(data.afm_requirements) ? data.afm_requirements : [],
      recommended_actions: Array.isArray(data.recommended_actions) ? data.recommended_actions : [],
      deadline: data.deadline,
      priority: validPriorities.includes(data.priority) ? data.priority : 'medium',
    };
  }

  /**
   * Validate product recommendation
   */
  private validateProductRecommendation(data: any): ProductRecommendation {
    const validTypes: ProductRecommendation['product_type'][] = ['fixed_rate', 'variable_rate', 'green_mortgage', 'investment_mortgage'];

    return {
      id: String(data.id || ''),
      lender: String(data.lender || ''),
      lender_id: String(data.lender_id || ''),
      product_name: String(data.product_name || ''),
      product_type: validTypes.includes(data.product_type) ? data.product_type : 'fixed_rate',
      interest_rate: Number(data.interest_rate) || 0,
      max_ltv: Math.max(0, Math.min(100, Number(data.max_ltv) || 0)),
      suitability_score: Math.max(0, Math.min(100, Number(data.suitability_score) || 0)),
      afm_compliant: Boolean(data.afm_compliant),
      conditions: Array.isArray(data.conditions) ? data.conditions : undefined,
      term_years: Number(data.term_years) || 30,
      nhg_required: data.nhg_required ? Boolean(data.nhg_required) : undefined,
      energy_efficiency_bonus: data.energy_efficiency_bonus ? Number(data.energy_efficiency_bonus) : undefined,
      estimated_monthly_payment: data.estimated_monthly_payment ? Number(data.estimated_monthly_payment) : undefined,
      total_costs: data.total_costs ? Number(data.total_costs) : undefined,
    };
  }

  /**
   * Validate assessment history
   */
  private validateAssessmentHistory(data: any): AssessmentHistory {
    return {
      id: String(data.id || ''),
      client_id: String(data.client_id || ''),
      assessment_date: String(data.assessment_date || ''),
      score: Math.max(0, Math.min(100, Number(data.score) || 0)),
      status: String(data.status || ''),
      assessor: String(data.assessor || ''),
      changes_from_previous: Array.isArray(data.changes_from_previous) ? data.changes_from_previous : undefined,
    };
  }

  /**
   * Get authentication token
   */
  private getAuthToken(): string {
    return localStorage.getItem('auth_token') || '';
  }

  /**
   * Show error notification
   */
  private showError(message: string) {
    if (this.enqueueSnackbar) {
      this.enqueueSnackbar(message, { variant: 'error' });
    }
  }

  /**
   * Show success notification
   */
  private showSuccess(message: string) {
    if (this.enqueueSnackbar) {
      this.enqueueSnackbar(message, { variant: 'success' });
    }
  }
}

export const complianceApi = new ComplianceApiService();
