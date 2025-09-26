/**
 * Lender Integration API Service
 *
 * Manages integration with Dutch mortgage lenders
 */
import { useSnackbar } from 'notistack';

export interface Lender {
  id: string;
  name: string;
  logo: string;
  status: 'active' | 'inactive' | 'maintenance';
  api_status: 'connected' | 'disconnected' | 'error';
  last_sync: string;
  supported_products: string[];
  processing_time: string;
  success_rate: number;
  enabled: boolean;
}

class LenderApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  async getLenders(): Promise<Lender[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/lenders`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${this.getAuthToken()}` },
      });
      if (!response.ok) throw new Error(`Lenders API error: ${response.status}`);
      const data = await response.json();
      return Array.isArray(data.lenders) ? data.lenders : [];
    } catch (error) {
      this.showError('Failed to load lenders');
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

export const lenderApi = new LenderApiService();
