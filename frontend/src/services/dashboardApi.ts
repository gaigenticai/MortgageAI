/**
 * Dutch Mortgage Dashboard API Service
 *
 * Provides real-time data for AFM compliance status, lender integration,
 * and activity monitoring for the Dutch mortgage advisory platform.
 */
import { useSnackbar } from 'notistack';

export interface AFMComplianceStatus {
  overall_score: number;
  active_sessions: number;
  pending_reviews: number;
  audit_ready: number;
  last_updated: string;
  compliance_trends: {
    score_change: number;
    sessions_trend: number;
    reviews_trend: number;
  };
}

export interface LenderStatus {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'maintenance' | 'error';
  processing_time: string;
  success_rate: number;
  active_applications: number;
  last_successful_submission: string | null;
  api_endpoint: string;
  authentication_status: 'valid' | 'expired' | 'invalid';
}

export interface ActivityItem {
  id: string;
  type: 'compliance_check' | 'application_submitted' | 'bkr_check' | 'nhg_eligibility' | 'client_intake' | 'audit_review';
  client_name: string;
  client_id: string;
  lender_name?: string;
  timestamp: string;
  status: 'completed' | 'pending' | 'failed' | 'in_progress';
  details: Record<string, any>;
}

export interface DashboardData {
  afm_status: AFMComplianceStatus;
  lender_statuses: LenderStatus[];
  recent_activity: ActivityItem[];
  market_indicators: {
    average_processing_time: number;
    total_applications_this_month: number;
    compliance_rate: number;
    lender_availability_score: number;
  };
}

class DashboardApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    // Use environment variable or default to localhost for development
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  /**
   * Set snackbar function for error notifications
   */
  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  /**
   * Fetch complete dashboard data
   */
  async getDashboardData(): Promise<DashboardData> {
    try {
      const response = await fetch(`${this.baseUrl}/api/dashboard/overview`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // Add authentication headers if needed
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Dashboard API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateDashboardData(data);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      this.showError('Failed to load dashboard data. Please try again.');
      throw error;
    }
  }

  /**
   * Fetch AFM compliance status
   */
  async getAFMComplianceStatus(): Promise<AFMComplianceStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/api/afm/compliance-status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`AFM compliance API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateAFMStatus(data);
    } catch (error) {
      console.error('Failed to fetch AFM compliance status:', error);
      this.showError('Failed to load AFM compliance status.');
      throw error;
    }
  }

  /**
   * Fetch lender integration statuses
   */
  async getLenderStatuses(): Promise<LenderStatus[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/lenders/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Lender status API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.lenders?.map(this.validateLenderStatus) || [];
    } catch (error) {
      console.error('Failed to fetch lender statuses:', error);
      this.showError('Failed to load lender integration status.');
      throw error;
    }
  }

  /**
   * Fetch recent activity data
   */
  async getRecentActivity(limit: number = 10): Promise<ActivityItem[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/activity/recent?limit=${limit}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Activity API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return data.activities?.map(this.validateActivityItem) || [];
    } catch (error) {
      console.error('Failed to fetch recent activity:', error);
      this.showError('Failed to load recent activity.');
      throw error;
    }
  }

  /**
   * Refresh lender status for a specific lender
   */
  async refreshLenderStatus(lenderId: string): Promise<LenderStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/api/lenders/${lenderId}/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Lender refresh API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateLenderStatus(data);
    } catch (error) {
      console.error(`Failed to refresh lender ${lenderId} status:`, error);
      this.showError(`Failed to refresh ${lenderId} status.`);
      throw error;
    }
  }

  /**
   * Get market indicators
   */
  async getMarketIndicators(): Promise<DashboardData['market_indicators']> {
    try {
      const response = await fetch(`${this.baseUrl}/api/market/indicators`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Market indicators API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return this.validateMarketIndicators(data);
    } catch (error) {
      console.error('Failed to fetch market indicators:', error);
      this.showError('Failed to load market indicators.');
      throw error;
    }
  }

  /**
   * Validate dashboard data structure
   */
  private validateDashboardData(data: any): DashboardData {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid dashboard data format');
    }

    return {
      afm_status: this.validateAFMStatus(data.afm_status),
      lender_statuses: data.lender_statuses?.map(this.validateLenderStatus) || [],
      recent_activity: data.recent_activity?.map(this.validateActivityItem) || [],
      market_indicators: this.validateMarketIndicators(data.market_indicators),
    };
  }

  /**
   * Validate AFM compliance status
   */
  private validateAFMStatus(data: any): AFMComplianceStatus {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid AFM status format');
    }

    return {
      overall_score: Number(data.overall_score) || 0,
      active_sessions: Number(data.active_sessions) || 0,
      pending_reviews: Number(data.pending_reviews) || 0,
      audit_ready: Number(data.audit_ready) || 0,
      last_updated: data.last_updated || new Date().toISOString(),
      compliance_trends: {
        score_change: Number(data.compliance_trends?.score_change) || 0,
        sessions_trend: Number(data.compliance_trends?.sessions_trend) || 0,
        reviews_trend: Number(data.compliance_trends?.reviews_trend) || 0,
      },
    };
  }

  /**
   * Validate lender status
   */
  private validateLenderStatus(data: any): LenderStatus {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid lender status format');
    }

    const validStatuses: LenderStatus['status'][] = ['online', 'offline', 'maintenance', 'error'];
    const validAuthStatuses: LenderStatus['authentication_status'][] = ['valid', 'expired', 'invalid'];

    return {
      id: String(data.id || ''),
      name: String(data.name || ''),
      status: validStatuses.includes(data.status) ? data.status : 'offline',
      processing_time: String(data.processing_time || 'Unknown'),
      success_rate: Math.max(0, Math.min(100, Number(data.success_rate) || 0)),
      active_applications: Number(data.active_applications) || 0,
      last_successful_submission: data.last_successful_submission || null,
      api_endpoint: String(data.api_endpoint || ''),
      authentication_status: validAuthStatuses.includes(data.authentication_status)
        ? data.authentication_status
        : 'invalid',
    };
  }

  /**
   * Validate activity item
   */
  private validateActivityItem(data: any): ActivityItem {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid activity item format');
    }

    const validTypes: ActivityItem['type'][] = [
      'compliance_check', 'application_submitted', 'bkr_check',
      'nhg_eligibility', 'client_intake', 'audit_review'
    ];
    const validStatuses: ActivityItem['status'][] = [
      'completed', 'pending', 'failed', 'in_progress'
    ];

    return {
      id: String(data.id || ''),
      type: validTypes.includes(data.type) ? data.type : 'client_intake',
      client_name: String(data.client_name || ''),
      client_id: String(data.client_id || ''),
      lender_name: data.lender_name ? String(data.lender_name) : undefined,
      timestamp: String(data.timestamp || new Date().toISOString()),
      status: validStatuses.includes(data.status) ? data.status : 'pending',
      details: data.details || {},
    };
  }

  /**
   * Validate market indicators
   */
  private validateMarketIndicators(data: any): DashboardData['market_indicators'] {
    return {
      average_processing_time: Number(data?.average_processing_time) || 0,
      total_applications_this_month: Number(data?.total_applications_this_month) || 0,
      compliance_rate: Math.max(0, Math.min(100, Number(data?.compliance_rate) || 0)),
      lender_availability_score: Math.max(0, Math.min(100, Number(data?.lender_availability_score) || 0)),
    };
  }

  /**
   * Get authentication token from storage
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

export const dashboardApi = new DashboardApiService();
