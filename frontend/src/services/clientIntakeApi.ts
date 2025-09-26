/**
 * AFM Client Intake API Service
 *
 * Handles AFM-compliant client assessment and intake process
 * with comprehensive validation and regulatory compliance checks.
 */
import { useSnackbar } from 'notistack';

export interface PersonalInfo {
  full_name: string;
  bsn: string;
  date_of_birth: string;
  marital_status: 'single' | 'married' | 'registered_partnership' | 'divorced' | 'widowed';
  number_of_dependents: number;
  email: string;
  phone: string;
}

export interface EmploymentInfo {
  employment_status: 'employed' | 'self_employed' | 'unemployed' | 'retired' | 'student' | 'other';
  employer_name: string;
  job_title: string;
  employment_duration_months: number;
  gross_annual_income: number;
  partner_income: number;
  other_income_sources: string[];
  other_income_amount?: number;
}

export interface FinancialSituation {
  existing_debts: DebtInfo[];
  monthly_expenses: number;
  savings_amount: number;
  investments: InvestmentInfo[];
  other_properties: boolean;
  other_properties_value?: number;
}

export interface DebtInfo {
  type: 'mortgage' | 'personal_loan' | 'credit_card' | 'student_loan' | 'other';
  creditor_name: string;
  outstanding_amount: number;
  monthly_payment: number;
  interest_rate?: number;
  remaining_term_months?: number;
}

export interface InvestmentInfo {
  type: 'savings' | 'stocks' | 'bonds' | 'pension' | 'real_estate' | 'other';
  institution_name: string;
  current_value: number;
  monthly_contribution?: number;
}

export interface MortgageRequirements {
  property_type: 'apartment' | 'house' | 'townhouse' | 'condo' | 'other';
  property_location: string;
  estimated_property_value: number;
  desired_mortgage_amount: number;
  preferred_mortgage_term: number;
  interest_rate_preference: 'fixed' | 'variable' | 'flexible';
  down_payment_amount?: number;
  nhg_required: boolean;
}

export interface AFMSuitabilityAssessment {
  mortgage_experience: 'first_time' | 'experienced' | 'very_experienced';
  financial_knowledge_level: 'basic' | 'intermediate' | 'advanced';
  risk_tolerance: 'conservative' | 'moderate' | 'aggressive';
  investment_objectives: string[];
  sustainability_preferences: 'not_important' | 'somewhat_important' | 'very_important';
  advice_needs: string[];
  expected_advice_frequency: 'one_time' | 'regular' | 'ongoing';
}

export interface ClientProfile {
  id?: string;
  personal_info: PersonalInfo;
  employment_info: EmploymentInfo;
  financial_situation: FinancialSituation;
  mortgage_requirements: MortgageRequirements;
  afm_suitability: AFMSuitabilityAssessment;
  created_at?: string;
  updated_at?: string;
  status: 'draft' | 'submitted' | 'under_review' | 'approved' | 'rejected';
  compliance_score?: number;
  risk_profile?: 'low' | 'medium' | 'high';
}

export interface ValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
  warnings: Record<string, string>;
  afm_compliance_score: number;
  risk_assessment: 'low' | 'medium' | 'high';
}

export interface IntakeProgress {
  completed_sections: string[];
  total_sections: number;
  completion_percentage: number;
  next_required_section?: string;
  can_submit: boolean;
}

class ClientIntakeApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  /**
   * Set snackbar function for notifications
   */
  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  /**
   * Create new client intake profile
   */
  async createClientProfile(profile: Omit<ClientProfile, 'id' | 'created_at' | 'updated_at'>): Promise<ClientProfile> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/client-intake`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(profile),
      });

      if (!response.ok) {
        throw new Error(`Client intake API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      this.showSuccess('Client profile created successfully');
      return this.parseClientProfile(data);
    } catch (error) {
      console.error('Failed to create client profile:', error);
      this.showError('Failed to create client profile. Please try again.');
      throw error;
    }
  }

  /**
   * Update existing client profile
   */
  async updateClientProfile(id: string, profile: Partial<ClientProfile>): Promise<ClientProfile> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/client-intake/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(profile),
      });

      if (!response.ok) {
        throw new Error(`Client update API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      this.showSuccess('Client profile updated successfully');
      return this.parseClientProfile(data);
    } catch (error) {
      console.error('Failed to update client profile:', error);
      this.showError('Failed to update client profile. Please try again.');
      throw error;
    }
  }

  /**
   * Get client profile by ID
   */
  async getClientProfile(id: string): Promise<ClientProfile> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/client-intake/${id}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Client profile API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.parseClientProfile(data);
    } catch (error) {
      console.error('Failed to fetch client profile:', error);
      this.showError('Failed to load client profile.');
      throw error;
    }
  }

  /**
   * Submit client profile for AFM compliance review
   */
  async submitClientProfile(id: string): Promise<{ success: boolean; compliance_score: number; review_id: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/client-intake/${id}/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Client submission API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      this.showSuccess('Client profile submitted for AFM compliance review');
      return {
        success: true,
        compliance_score: Number(data.compliance_score) || 0,
        review_id: String(data.review_id || ''),
      };
    } catch (error) {
      console.error('Failed to submit client profile:', error);
      this.showError('Failed to submit client profile for review.');
      throw error;
    }
  }

  /**
   * Validate client profile data
   */
  async validateClientProfile(profile: Partial<ClientProfile>): Promise<ValidationResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/client-intake/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(profile),
      });

      if (!response.ok) {
        throw new Error(`Validation API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        isValid: Boolean(data.is_valid),
        errors: data.errors || {},
        warnings: data.warnings || {},
        afm_compliance_score: Number(data.afm_compliance_score) || 0,
        risk_assessment: this.validateRiskAssessment(data.risk_assessment),
      };
    } catch (error) {
      console.error('Failed to validate client profile:', error);
      this.showError('Failed to validate client profile.');
      throw error;
    }
  }

  /**
   * Get intake progress for a profile
   */
  async getIntakeProgress(id: string): Promise<IntakeProgress> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/client-intake/${id}/progress`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Progress API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        completed_sections: Array.isArray(data.completed_sections) ? data.completed_sections : [],
        total_sections: Number(data.total_sections) || 5,
        completion_percentage: Math.max(0, Math.min(100, Number(data.completion_percentage) || 0)),
        next_required_section: data.next_required_section,
        can_submit: Boolean(data.can_submit),
      };
    } catch (error) {
      console.error('Failed to fetch intake progress:', error);
      // Return default progress if API fails
      return {
        completed_sections: [],
        total_sections: 5,
        completion_percentage: 0,
        can_submit: false,
      };
    }
  }

  /**
   * Validate BSN (Dutch Social Security Number)
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
   * Calculate debt-to-income ratio
   */
  async calculateDTI(profile: Partial<ClientProfile>): Promise<{ dti_ratio: number; max_mortgage_amount: number }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/calculations/dti`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(profile),
      });

      if (!response.ok) {
        throw new Error(`DTI calculation API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        dti_ratio: Number(data.dti_ratio) || 0,
        max_mortgage_amount: Number(data.max_mortgage_amount) || 0,
      };
    } catch (error) {
      console.error('Failed to calculate DTI:', error);
      return {
        dti_ratio: 0,
        max_mortgage_amount: 0,
      };
    }
  }

  /**
   * Validate client profile data structure
   */
  private parseClientProfile(data: any): ClientProfile {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid client profile format');
    }

    const validStatuses: ClientProfile['status'][] = ['draft', 'submitted', 'under_review', 'approved', 'rejected'];
    const validRiskProfiles: ClientProfile['risk_profile'][] = ['low', 'medium', 'high'];

    return {
      id: String(data.id || ''),
      personal_info: this.validatePersonalInfo(data.personal_info),
      employment_info: this.validateEmploymentInfo(data.employment_info),
      financial_situation: this.validateFinancialSituation(data.financial_situation),
      mortgage_requirements: this.validateMortgageRequirements(data.mortgage_requirements),
      afm_suitability: this.validateAFMSuitability(data.afm_suitability),
      created_at: data.created_at,
      updated_at: data.updated_at,
      status: validStatuses.includes(data.status) ? data.status : 'draft',
      compliance_score: data.compliance_score ? Number(data.compliance_score) : undefined,
      risk_profile: validRiskProfiles.includes(data.risk_profile) ? data.risk_profile : undefined,
    };
  }

  private validatePersonalInfo(data: any): PersonalInfo {
    const validMaritalStatuses: PersonalInfo['marital_status'][] = [
      'single', 'married', 'registered_partnership', 'divorced', 'widowed'
    ];

    return {
      full_name: String(data?.full_name || ''),
      bsn: String(data?.bsn || ''),
      date_of_birth: String(data?.date_of_birth || ''),
      marital_status: validMaritalStatuses.includes(data?.marital_status) ? data.marital_status : 'single',
      number_of_dependents: Number(data?.number_of_dependents) || 0,
      email: String(data?.email || ''),
      phone: String(data?.phone || ''),
    };
  }

  private validateEmploymentInfo(data: any): EmploymentInfo {
    const validStatuses: EmploymentInfo['employment_status'][] = [
      'employed', 'self_employed', 'unemployed', 'retired', 'student', 'other'
    ];

    return {
      employment_status: validStatuses.includes(data?.employment_status) ? data.employment_status : 'employed',
      employer_name: String(data?.employer_name || ''),
      job_title: String(data?.job_title || ''),
      employment_duration_months: Number(data?.employment_duration_months) || 0,
      gross_annual_income: Number(data?.gross_annual_income) || 0,
      partner_income: Number(data?.partner_income) || 0,
      other_income_sources: Array.isArray(data?.other_income_sources) ? data.other_income_sources : [],
      other_income_amount: data?.other_income_amount ? Number(data.other_income_amount) : undefined,
    };
  }

  private validateFinancialSituation(data: any): FinancialSituation {
    return {
      existing_debts: Array.isArray(data?.existing_debts) ? data.existing_debts.map(this.validateDebtInfo) : [],
      monthly_expenses: Number(data?.monthly_expenses) || 0,
      savings_amount: Number(data?.savings_amount) || 0,
      investments: Array.isArray(data?.investments) ? data.investments.map(this.validateInvestmentInfo) : [],
      other_properties: Boolean(data?.other_properties),
      other_properties_value: data?.other_properties_value ? Number(data.other_properties_value) : undefined,
    };
  }

  private validateDebtInfo(data: any): DebtInfo {
    const validTypes: DebtInfo['type'][] = ['mortgage', 'personal_loan', 'credit_card', 'student_loan', 'other'];

    return {
      type: validTypes.includes(data?.type) ? data.type : 'other',
      creditor_name: String(data?.creditor_name || ''),
      outstanding_amount: Number(data?.outstanding_amount) || 0,
      monthly_payment: Number(data?.monthly_payment) || 0,
      interest_rate: data?.interest_rate ? Number(data.interest_rate) : undefined,
      remaining_term_months: data?.remaining_term_months ? Number(data.remaining_term_months) : undefined,
    };
  }

  private validateInvestmentInfo(data: any): InvestmentInfo {
    const validTypes: InvestmentInfo['type'][] = ['savings', 'stocks', 'bonds', 'pension', 'real_estate', 'other'];

    return {
      type: validTypes.includes(data?.type) ? data.type : 'other',
      institution_name: String(data?.institution_name || ''),
      current_value: Number(data?.current_value) || 0,
      monthly_contribution: data?.monthly_contribution ? Number(data.monthly_contribution) : undefined,
    };
  }

  private validateMortgageRequirements(data: any): MortgageRequirements {
    const validPropertyTypes: MortgageRequirements['property_type'][] = [
      'apartment', 'house', 'townhouse', 'condo', 'other'
    ];
    const validRatePreferences: MortgageRequirements['interest_rate_preference'][] = [
      'fixed', 'variable', 'flexible'
    ];

    return {
      property_type: validPropertyTypes.includes(data?.property_type) ? data.property_type : 'house',
      property_location: String(data?.property_location || ''),
      estimated_property_value: Number(data?.estimated_property_value) || 0,
      desired_mortgage_amount: Number(data?.desired_mortgage_amount) || 0,
      preferred_mortgage_term: Number(data?.preferred_mortgage_term) || 30,
      interest_rate_preference: validRatePreferences.includes(data?.interest_rate_preference)
        ? data.interest_rate_preference
        : 'fixed',
      down_payment_amount: data?.down_payment_amount ? Number(data.down_payment_amount) : undefined,
      nhg_required: Boolean(data?.nhg_required),
    };
  }

  private validateAFMSuitability(data: any): AFMSuitabilityAssessment {
    const validExperience: AFMSuitabilityAssessment['mortgage_experience'][] = [
      'first_time', 'experienced', 'very_experienced'
    ];
    const validKnowledge: AFMSuitabilityAssessment['financial_knowledge_level'][] = [
      'basic', 'intermediate', 'advanced'
    ];
    const validRiskTolerance: AFMSuitabilityAssessment['risk_tolerance'][] = [
      'conservative', 'moderate', 'aggressive'
    ];
    const validSustainability: AFMSuitabilityAssessment['sustainability_preferences'][] = [
      'not_important', 'somewhat_important', 'very_important'
    ];
    const validFrequency: AFMSuitabilityAssessment['expected_advice_frequency'][] = [
      'one_time', 'regular', 'ongoing'
    ];

    return {
      mortgage_experience: validExperience.includes(data?.mortgage_experience) ? data.mortgage_experience : 'first_time',
      financial_knowledge_level: validKnowledge.includes(data?.financial_knowledge_level)
        ? data.financial_knowledge_level
        : 'basic',
      risk_tolerance: validRiskTolerance.includes(data?.risk_tolerance) ? data.risk_tolerance : 'moderate',
      investment_objectives: Array.isArray(data?.investment_objectives) ? data.investment_objectives : [],
      sustainability_preferences: validSustainability.includes(data?.sustainability_preferences)
        ? data.sustainability_preferences
        : 'not_important',
      advice_needs: Array.isArray(data?.advice_needs) ? data.advice_needs : [],
      expected_advice_frequency: validFrequency.includes(data?.expected_advice_frequency)
        ? data.expected_advice_frequency
        : 'one_time',
    };
  }

  private validateRiskAssessment(risk: any): 'low' | 'medium' | 'high' {
    const validRisks: Array<'low' | 'medium' | 'high'> = ['low', 'medium', 'high'];
    return validRisks.includes(risk) ? risk : 'medium';
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

export const clientIntakeApi = new ClientIntakeApiService();
