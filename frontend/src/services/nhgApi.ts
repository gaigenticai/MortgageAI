/**
 * NHG (National Mortgage Guarantee) API Service
 *
 * Integrates with Dutch NHG system for mortgage guarantee assessment
 */
import { useSnackbar } from 'notistack';

export interface NHGEligibility {
  id: string;
  client_id: string;
  assessment_date: string;
  eligible: boolean;
  mortgage_amount: number;
  nhg_costs: number;
  benefits: Array<{
    type: string;
    description: string;
    value: string;
  }>;
  requirements: Array<{
    requirement: string;
    met: boolean;
    details?: string;
  }>;
  risk_assessment: 'low' | 'medium' | 'high';
  recommendations: Array<{
    type: string;
    message: string;
  }>;
}

class NHGApiService {
  private baseUrl: string;
  private enqueueSnackbar: any = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  setSnackbar(enqueueSnackbar: any) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  async getClientNHGEligibility(clientId: string): Promise<NHGEligibility> {
    try {
      const response = await fetch(`${this.baseUrl}/api/nhg/client/${clientId}/eligibility`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${this.getAuthToken()}` },
      });
      if (!response.ok) throw new Error(`NHG eligibility API error: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Failed to get NHG eligibility:', error);
      this.showError('Failed to load NHG eligibility assessment');
      throw error;
    }
  }

  private getAuthToken(): string {
    return localStorage.getItem('auth_token') || '';
  }

  private showError(message: string) {
    if (this.enqueueSnackbar) this.enqueueSnackbar(message, { variant: 'error' });
  }
}

export const nhgApi = new NHGApiService();