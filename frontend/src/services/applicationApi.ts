/**
 * Mortgage Application API Service
 *
 * Manages Dutch mortgage applications across multiple lenders
 * with AFM compliance and application lifecycle management
 */
import { useSnackbar } from 'notistack';

export interface ApplicationDocument {
  id: string;
  type: 'income_proof' | 'property_deed' | 'id_document' | 'bank_statement' | 'tax_return' | 'other';
  filename: string;
  uploaded_at: string;
  status: 'pending' | 'approved' | 'rejected' | 'requires_revision';
  url?: string;
  comments?: string;
}

export interface ApplicationStatus {
  id: string;
  lender_id: string;
  lender_name: string;
  application_number?: string;
  status: 'draft' | 'submitted' | 'under_review' | 'approved' | 'rejected' | 'offer_received' | 'completed' | 'cancelled';
  submitted_at?: string;
  last_updated: string;
  estimated_completion?: string;
  current_stage: string;
  progress_percentage: number;
  assigned_officer?: string;
  next_action_required?: string;
  next_action_deadline?: string;
  rejection_reason?: string;
  offer_details?: {
    amount: number;
    interest_rate: number;
    term_years: number;
    conditions: string[];
  };
}

export interface ApplicationData {
  id: string;
  client_id: string;
  client_name: string;
  property_details: {
    address: string;
    postal_code: string;
    city: string;
    property_type: 'apartment' | 'house' | 'townhouse' | 'condo';
    construction_year: number;
    energy_label?: 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G';
    value: number;
    purchase_price?: number;
  };
  mortgage_requirements: {
    amount: number;
    term_years: number;
    interest_type: 'fixed' | 'variable' | 'flexible';
    nhg_required: boolean;
    down_payment: number;
    purpose: 'purchase' | 'refinance' | 'additional_borrowing';
  };
  financial_info: {
    gross_income: number;
    partner_income?: number;
    other_income: number;
    monthly_expenses: number;
    existing_debts: number;
    savings: number;
    investments: number;
  };
  applications: ApplicationStatus[];
  documents: ApplicationDocument[];
  overall_status: 'draft' | 'in_progress' | 'offers_received' | 'completed' | 'cancelled';
  created_at: string;
  updated_at: string;
  afm_compliance_score: number;
  risk_profile: 'low' | 'medium' | 'high';
}

export interface ApplicationCreateRequest {
  client_id: string;
  property_details: ApplicationData['property_details'];
  mortgage_requirements: ApplicationData['mortgage_requirements'];
  financial_info: ApplicationData['financial_info'];
  selected_lenders: string[];
}

export interface ApplicationUpdateRequest {
  property_details?: Partial<ApplicationData['property_details']>;
  mortgage_requirements?: Partial<ApplicationData['mortgage_requirements']>;
  financial_info?: Partial<ApplicationData['financial_info']>;
}

class ApplicationApiService {
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
   * Create new mortgage application
   */
  async createApplication(request: ApplicationCreateRequest): Promise<ApplicationData> {
    try {
      const response = await fetch(`${this.baseUrl}/api/applications`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Application creation API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateApplicationData(data);
    } catch (error) {
      console.error('Failed to create application:', error);
      this.showError('Failed to create mortgage application');
      throw error;
    }
  }

  /**
   * Get application by ID
   */
  async getApplication(applicationId: string): Promise<ApplicationData> {
    try {
      const response = await fetch(`${this.baseUrl}/api/applications/${applicationId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Application API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateApplicationData(data);
    } catch (error) {
      console.error('Failed to get application:', error);
      this.showError('Failed to load mortgage application');
      throw error;
    }
  }

  /**
   * Get applications by client ID
   */
  async getClientApplications(clientId: string): Promise<ApplicationData[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/applications/client/${clientId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Client applications API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return Array.isArray(data.applications) ? data.applications.map(this.validateApplicationData) : [];
    } catch (error) {
      console.error('Failed to get client applications:', error);
      this.showError('Failed to load client applications');
      throw error;
    }
  }

  /**
   * Update application
   */
  async updateApplication(applicationId: string, updates: ApplicationUpdateRequest): Promise<ApplicationData> {
    try {
      const response = await fetch(`${this.baseUrl}/api/applications/${applicationId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(updates),
      });

      if (!response.ok) {
        throw new Error(`Application update API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateApplicationData(data);
    } catch (error) {
      console.error('Failed to update application:', error);
      this.showError('Failed to update mortgage application');
      throw error;
    }
  }

  /**
   * Submit application to lenders
   */
  async submitApplication(applicationId: string, lenderIds: string[]): Promise<{ success: boolean; submissions: ApplicationStatus[] }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/applications/${applicationId}/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify({ lender_ids: lenderIds }),
      });

      if (!response.ok) {
        throw new Error(`Application submission API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: Boolean(data.success),
        submissions: Array.isArray(data.submissions) ? data.submissions.map(this.validateApplicationStatus) : [],
      };
    } catch (error) {
      console.error('Failed to submit application:', error);
      this.showError('Failed to submit mortgage application');
      throw error;
    }
  }

  /**
   * Upload document to application
   */
  async uploadDocument(applicationId: string, file: File, documentType: ApplicationDocument['type'], comments?: string): Promise<ApplicationDocument> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('document_type', documentType);
      if (comments) {
        formData.append('comments', comments);
      }

      const response = await fetch(`${this.baseUrl}/api/applications/${applicationId}/documents`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Document upload API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateApplicationDocument(data);
    } catch (error) {
      console.error('Failed to upload document:', error);
      this.showError('Failed to upload document');
      throw error;
    }
  }

  /**
   * Get application status updates
   */
  async getApplicationStatusUpdates(applicationId: string, since?: string): Promise<ApplicationStatus[]> {
    try {
      const url = new URL(`${this.baseUrl}/api/applications/${applicationId}/status-updates`);
      if (since) {
        url.searchParams.append('since', since);
      }

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Status updates API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return Array.isArray(data.updates) ? data.updates.map(this.validateApplicationStatus) : [];
    } catch (error) {
      console.error('Failed to get status updates:', error);
      this.showError('Failed to load application status updates');
      throw error;
    }
  }

  /**
   * Cancel application
   */
  async cancelApplication(applicationId: string, reason: string): Promise<{ success: boolean; cancelled_at: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/applications/${applicationId}/cancel`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify({ reason }),
      });

      if (!response.ok) {
        throw new Error(`Application cancellation API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: Boolean(data.success),
        cancelled_at: String(data.cancelled_at || new Date().toISOString()),
      };
    } catch (error) {
      console.error('Failed to cancel application:', error);
      this.showError('Failed to cancel mortgage application');
      throw error;
    }
  }

  /**
   * Validate application data structure
   */
  private validateApplicationData(data: any): ApplicationData {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid application data format');
    }

    const validStatuses: ApplicationData['overall_status'][] = ['draft', 'in_progress', 'offers_received', 'completed', 'cancelled'];
    const validRiskProfiles: ApplicationData['risk_profile'][] = ['low', 'medium', 'high'];
    const validPropertyTypes: ApplicationData['property_details']['property_type'][] = ['apartment', 'house', 'townhouse', 'condo'];
    const validInterestTypes: ApplicationData['mortgage_requirements']['interest_type'][] = ['fixed', 'variable', 'flexible'];
    const validPurposes: ApplicationData['mortgage_requirements']['purpose'][] = ['purchase', 'refinance', 'additional_borrowing'];

    return {
      id: String(data.id || ''),
      client_id: String(data.client_id || ''),
      client_name: String(data.client_name || ''),
      property_details: {
        address: String(data.property_details?.address || ''),
        postal_code: String(data.property_details?.postal_code || ''),
        city: String(data.property_details?.city || ''),
        property_type: validPropertyTypes.includes(data.property_details?.property_type) ? data.property_details.property_type : 'house',
        construction_year: Number(data.property_details?.construction_year) || new Date().getFullYear(),
        energy_label: data.property_details?.energy_label,
        value: Number(data.property_details?.value) || 0,
        purchase_price: data.property_details?.purchase_price ? Number(data.property_details.purchase_price) : undefined,
      },
      mortgage_requirements: {
        amount: Number(data.mortgage_requirements?.amount) || 0,
        term_years: Number(data.mortgage_requirements?.term_years) || 30,
        interest_type: validInterestTypes.includes(data.mortgage_requirements?.interest_type) ? data.mortgage_requirements.interest_type : 'fixed',
        nhg_required: Boolean(data.mortgage_requirements?.nhg_required),
        down_payment: Number(data.mortgage_requirements?.down_payment) || 0,
        purpose: validPurposes.includes(data.mortgage_requirements?.purpose) ? data.mortgage_requirements.purpose : 'purchase',
      },
      financial_info: {
        gross_income: Number(data.financial_info?.gross_income) || 0,
        partner_income: data.financial_info?.partner_income ? Number(data.financial_info.partner_income) : undefined,
        other_income: Number(data.financial_info?.other_income) || 0,
        monthly_expenses: Number(data.financial_info?.monthly_expenses) || 0,
        existing_debts: Number(data.financial_info?.existing_debts) || 0,
        savings: Number(data.financial_info?.savings) || 0,
        investments: Number(data.financial_info?.investments) || 0,
      },
      applications: Array.isArray(data.applications) ? data.applications.map(this.validateApplicationStatus) : [],
      documents: Array.isArray(data.documents) ? data.documents.map(this.validateApplicationDocument) : [],
      overall_status: validStatuses.includes(data.overall_status) ? data.overall_status : 'draft',
      created_at: String(data.created_at || new Date().toISOString()),
      updated_at: String(data.updated_at || new Date().toISOString()),
      afm_compliance_score: Math.max(0, Math.min(100, Number(data.afm_compliance_score) || 0)),
      risk_profile: validRiskProfiles.includes(data.risk_profile) ? data.risk_profile : 'medium',
    };
  }

  /**
   * Validate application status
   */
  private validateApplicationStatus(data: any): ApplicationStatus {
    const validStatuses: ApplicationStatus['status'][] = [
      'draft', 'submitted', 'under_review', 'approved', 'rejected', 'offer_received', 'completed', 'cancelled'
    ];

    return {
      id: String(data.id || ''),
      lender_id: String(data.lender_id || ''),
      lender_name: String(data.lender_name || ''),
      application_number: data.application_number ? String(data.application_number) : undefined,
      status: validStatuses.includes(data.status) ? data.status : 'draft',
      submitted_at: data.submitted_at,
      last_updated: String(data.last_updated || new Date().toISOString()),
      estimated_completion: data.estimated_completion,
      current_stage: String(data.current_stage || ''),
      progress_percentage: Math.max(0, Math.min(100, Number(data.progress_percentage) || 0)),
      assigned_officer: data.assigned_officer,
      next_action_required: data.next_action_required,
      next_action_deadline: data.next_action_deadline,
      rejection_reason: data.rejection_reason,
      offer_details: data.offer_details ? {
        amount: Number(data.offer_details.amount) || 0,
        interest_rate: Number(data.offer_details.interest_rate) || 0,
        term_years: Number(data.offer_details.term_years) || 0,
        conditions: Array.isArray(data.offer_details.conditions) ? data.offer_details.conditions : [],
      } : undefined,
    };
  }

  /**
   * Validate application document
   */
  private validateApplicationDocument(data: any): ApplicationDocument {
    const validTypes: ApplicationDocument['type'][] = [
      'income_proof', 'property_deed', 'id_document', 'bank_statement', 'tax_return', 'other'
    ];
    const validStatuses: ApplicationDocument['status'][] = [
      'pending', 'approved', 'rejected', 'requires_revision'
    ];

    return {
      id: String(data.id || ''),
      type: validTypes.includes(data.type) ? data.type : 'other',
      filename: String(data.filename || ''),
      uploaded_at: String(data.uploaded_at || new Date().toISOString()),
      status: validStatuses.includes(data.status) ? data.status : 'pending',
      url: data.url,
      comments: data.comments,
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

export const applicationApi = new ApplicationApiService();
