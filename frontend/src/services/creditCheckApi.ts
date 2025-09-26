/**
 * BKR Credit Check API Service
 *
 * Integrates with Dutch BKR (Bureau Krediet Registratie) for comprehensive
 * credit scoring and mortgage eligibility assessment
 */
import { useSnackbar } from 'notistack';

export interface CreditReport {
  id: string;
  bsn: string;
  credit_score: number;
  score_range: 'excellent' | 'good' | 'fair' | 'poor' | 'very_poor';
  report_date: string;
  last_updated: string;
  active_loans: Array<{
    id: string;
    type: 'mortgage' | 'personal_loan' | 'credit_card' | 'student_loan' | 'car_loan' | 'other';
    creditor: string;
    creditor_code: string;
    outstanding_amount: number;
    monthly_payment: number;
    original_amount: number;
    interest_rate: number;
    status: 'current' | 'delinquent' | 'paid_off' | 'defaulted';
    start_date: string;
    end_date?: string;
    remaining_term_months: number;
  }>;
  payment_history: {
    total_accounts: number;
    on_time_payments: number;
    late_payments: number;
    missed_payments: number;
    defaulted_loans: number;
    payment_history_score: number;
    oldest_account_age_months: number;
    average_account_age_months: number;
  };
  credit_utilization: {
    total_credit_limit: number;
    total_balance: number;
    utilization_percentage: number;
    revolving_utilization: number;
  };
  inquiries: Array<{
    date: string;
    type: 'hard' | 'soft';
    purpose: string;
    creditor: string;
  }>;
  recommendations: Array<{
    type: 'positive' | 'caution' | 'warning' | 'critical';
    message: string;
    impact_on_mortgage: 'positive' | 'neutral' | 'negative' | 'blocking';
  }>;
  mortgage_eligibility: {
    eligible: boolean;
    max_mortgage_amount: number;
    risk_category: 'low' | 'medium' | 'high' | 'very_high';
    conditions: string[];
    nhg_required: boolean;
    minimum_down_payment_percentage: number;
    estimated_interest_rate_premium: number;
  };
  risk_indicators: {
    bankruptcy_history: boolean;
    foreclosure_history: boolean;
    debt_collection_actions: number;
    high_balance_utilization: boolean;
    recent_hard_inquiries: number;
    thin_credit_file: boolean;
  };
}

export interface CreditCheckRequest {
  client_id: string;
  bsn: string;
  purpose: 'mortgage_pre_approval' | 'mortgage_application' | 'credit_review' | 'annual_review';
  include_payment_history: boolean;
  include_inquiries: boolean;
  consent_given: boolean;
  consent_timestamp: string;
}

export interface CreditCheckResponse {
  request_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  estimated_completion_time: string;
  report?: CreditReport;
  error?: string;
}

class CreditCheckApiService {
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
   * Request credit check from BKR
   */
  async requestCreditCheck(request: CreditCheckRequest): Promise<CreditCheckResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/bkr/credit-check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Credit check API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateCreditCheckResponse(data);
    } catch (error) {
      console.error('Failed to request credit check:', error);
      this.showError('Failed to request BKR credit check');
      throw error;
    }
  }

  /**
   * Get credit check status
   */
  async getCreditCheckStatus(requestId: string): Promise<CreditCheckResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/bkr/credit-check/${requestId}/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Credit check status API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateCreditCheckResponse(data);
    } catch (error) {
      console.error('Failed to get credit check status:', error);
      this.showError('Failed to get BKR credit check status');
      throw error;
    }
  }

  /**
   * Get credit report by ID
   */
  async getCreditReport(reportId: string): Promise<CreditReport> {
    try {
      const response = await fetch(`${this.baseUrl}/api/bkr/credit-report/${reportId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Credit report API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateCreditReport(data);
    } catch (error) {
      console.error('Failed to get credit report:', error);
      this.showError('Failed to load BKR credit report');
      throw error;
    }
  }

  /**
   * Get credit report by client ID
   */
  async getClientCreditReport(clientId: string): Promise<CreditReport> {
    try {
      const response = await fetch(`${this.baseUrl}/api/bkr/client/${clientId}/credit-report`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Client credit report API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateCreditReport(data);
    } catch (error) {
      console.error('Failed to get client credit report:', error);
      this.showError('Failed to load client BKR credit report');
      throw error;
    }
  }

  /**
   * Refresh credit report
   */
  async refreshCreditReport(reportId: string): Promise<CreditCheckResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/bkr/credit-report/${reportId}/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Refresh credit report API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateCreditCheckResponse(data);
    } catch (error) {
      console.error('Failed to refresh credit report:', error);
      this.showError('Failed to refresh BKR credit report');
      throw error;
    }
  }

  /**
   * Get credit score history for client
   */
  async getCreditScoreHistory(clientId: string, months: number = 24): Promise<Array<{
    date: string;
    score: number;
    change: number;
  }>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/bkr/client/${clientId}/score-history?months=${months}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Credit score history API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return Array.isArray(data.history) ? data.history.map(this.validateScoreHistoryEntry) : [];
    } catch (error) {
      console.error('Failed to get credit score history:', error);
      this.showError('Failed to load credit score history');
      throw error;
    }
  }

  /**
   * Validate BSN format (Dutch Social Security Number)
   */
  async validateBSN(bsn: string): Promise<{ isValid: boolean; message?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/validation/bsn`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify({ bsn }),
      });

      if (!response.ok) {
        throw new Error(`BSN validation API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        isValid: Boolean(data.is_valid),
        message: data.message,
      };
    } catch (error) {
      console.error('Failed to validate BSN:', error);
      return {
        isValid: false,
        message: 'Unable to validate BSN at this time',
      };
    }
  }

  /**
   * Validate credit check response
   */
  private validateCreditCheckResponse(data: any): CreditCheckResponse {
    const validStatuses: CreditCheckResponse['status'][] = ['pending', 'processing', 'completed', 'failed'];

    return {
      request_id: String(data.request_id || ''),
      status: validStatuses.includes(data.status) ? data.status : 'pending',
      estimated_completion_time: String(data.estimated_completion_time || ''),
      report: data.report ? this.validateCreditReport(data.report) : undefined,
      error: data.error ? String(data.error) : undefined,
    };
  }

  /**
   * Validate credit report data structure
   */
  private validateCreditReport(data: any): CreditReport {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid credit report format');
    }

    const validScoreRanges: CreditReport['score_range'][] = ['excellent', 'good', 'fair', 'poor', 'very_poor'];
    const validLoanTypes: CreditReport['active_loans'][0]['type'][] = [
      'mortgage', 'personal_loan', 'credit_card', 'student_loan', 'car_loan', 'other'
    ];
    const validStatuses: CreditReport['active_loans'][0]['status'][] = [
      'current', 'delinquent', 'paid_off', 'defaulted'
    ];
    const validRiskCategories: CreditReport['mortgage_eligibility']['risk_category'][] = [
      'low', 'medium', 'high', 'very_high'
    ];

    return {
      id: String(data.id || ''),
      bsn: String(data.bsn || ''),
      credit_score: Math.max(0, Math.min(1000, Number(data.credit_score) || 0)),
      score_range: validScoreRanges.includes(data.score_range) ? data.score_range : 'fair',
      report_date: String(data.report_date || new Date().toISOString()),
      last_updated: String(data.last_updated || new Date().toISOString()),
      active_loans: Array.isArray(data.active_loans) ? data.active_loans.map(this.validateLoan) : [],
      payment_history: this.validatePaymentHistory(data.payment_history),
      credit_utilization: this.validateCreditUtilization(data.credit_utilization),
      inquiries: Array.isArray(data.inquiries) ? data.inquiries.map(this.validateInquiry) : [],
      recommendations: Array.isArray(data.recommendations) ? data.recommendations.map(this.validateRecommendation) : [],
      mortgage_eligibility: this.validateMortgageEligibility(data.mortgage_eligibility),
      risk_indicators: this.validateRiskIndicators(data.risk_indicators),
    };
  }

  /**
   * Validate loan data
   */
  private validateLoan(data: any): CreditReport['active_loans'][0] {
    const validTypes: CreditReport['active_loans'][0]['type'][] = [
      'mortgage', 'personal_loan', 'credit_card', 'student_loan', 'car_loan', 'other'
    ];
    const validStatuses: CreditReport['active_loans'][0]['status'][] = [
      'current', 'delinquent', 'paid_off', 'defaulted'
    ];

    return {
      id: String(data.id || ''),
      type: validTypes.includes(data.type) ? data.type : 'other',
      creditor: String(data.creditor || ''),
      creditor_code: String(data.creditor_code || ''),
      outstanding_amount: Number(data.outstanding_amount) || 0,
      monthly_payment: Number(data.monthly_payment) || 0,
      original_amount: Number(data.original_amount) || 0,
      interest_rate: Number(data.interest_rate) || 0,
      status: validStatuses.includes(data.status) ? data.status : 'current',
      start_date: String(data.start_date || ''),
      end_date: data.end_date ? String(data.end_date) : undefined,
      remaining_term_months: Number(data.remaining_term_months) || 0,
    };
  }

  /**
   * Validate payment history
   */
  private validatePaymentHistory(data: any): CreditReport['payment_history'] {
    return {
      total_accounts: Number(data?.total_accounts) || 0,
      on_time_payments: Number(data?.on_time_payments) || 0,
      late_payments: Number(data?.late_payments) || 0,
      missed_payments: Number(data?.missed_payments) || 0,
      defaulted_loans: Number(data?.defaulted_loans) || 0,
      payment_history_score: Math.max(0, Math.min(100, Number(data?.payment_history_score) || 0)),
      oldest_account_age_months: Number(data?.oldest_account_age_months) || 0,
      average_account_age_months: Number(data?.average_account_age_months) || 0,
    };
  }

  /**
   * Validate credit utilization
   */
  private validateCreditUtilization(data: any): CreditReport['credit_utilization'] {
    const utilization = Math.max(0, Math.min(100, Number(data?.utilization_percentage) || 0));

    return {
      total_credit_limit: Number(data?.total_credit_limit) || 0,
      total_balance: Number(data?.total_balance) || 0,
      utilization_percentage: utilization,
      revolving_utilization: Math.max(0, Math.min(100, Number(data?.revolving_utilization) || 0)),
    };
  }

  /**
   * Validate inquiry data
   */
  private validateInquiry(data: any): CreditReport['inquiries'][0] {
    return {
      date: String(data.date || ''),
      type: data.type === 'hard' ? 'hard' : 'soft',
      purpose: String(data.purpose || ''),
      creditor: String(data.creditor || ''),
    };
  }

  /**
   * Validate recommendation
   */
  private validateRecommendation(data: any): CreditReport['recommendations'][0] {
    const validTypes: CreditReport['recommendations'][0]['type'][] = ['positive', 'caution', 'warning', 'critical'];
    const validImpacts: CreditReport['recommendations'][0]['impact_on_mortgage'][] = ['positive', 'neutral', 'negative', 'blocking'];

    return {
      type: validTypes.includes(data.type) ? data.type : 'caution',
      message: String(data.message || ''),
      impact_on_mortgage: validImpacts.includes(data.impact_on_mortgage) ? data.impact_on_mortgage : 'neutral',
    };
  }

  /**
   * Validate mortgage eligibility
   */
  private validateMortgageEligibility(data: any): CreditReport['mortgage_eligibility'] {
    const validRiskCategories: CreditReport['mortgage_eligibility']['risk_category'][] = [
      'low', 'medium', 'high', 'very_high'
    ];

    return {
      eligible: Boolean(data?.eligible),
      max_mortgage_amount: Number(data?.max_mortgage_amount) || 0,
      risk_category: validRiskCategories.includes(data?.risk_category) ? data.risk_category : 'medium',
      conditions: Array.isArray(data?.conditions) ? data.conditions : [],
      nhg_required: Boolean(data?.nhg_required),
      minimum_down_payment_percentage: Math.max(0, Math.min(100, Number(data?.minimum_down_payment_percentage) || 0)),
      estimated_interest_rate_premium: Number(data?.estimated_interest_rate_premium) || 0,
    };
  }

  /**
   * Validate risk indicators
   */
  private validateRiskIndicators(data: any): CreditReport['risk_indicators'] {
    return {
      bankruptcy_history: Boolean(data?.bankruptcy_history),
      foreclosure_history: Boolean(data?.foreclosure_history),
      debt_collection_actions: Number(data?.debt_collection_actions) || 0,
      high_balance_utilization: Boolean(data?.high_balance_utilization),
      recent_hard_inquiries: Number(data?.recent_hard_inquiries) || 0,
      thin_credit_file: Boolean(data?.thin_credit_file),
    };
  }

  /**
   * Validate score history entry
   */
  private validateScoreHistoryEntry(data: any): { date: string; score: number; change: number } {
    return {
      date: String(data.date || ''),
      score: Math.max(0, Math.min(1000, Number(data.score) || 0)),
      change: Number(data.change) || 0,
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

export const creditCheckApi = new CreditCheckApiService();
