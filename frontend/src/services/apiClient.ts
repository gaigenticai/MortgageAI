import axios, { AxiosInstance, AxiosResponse } from 'axios';

export class MortgageAIApiClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:3000';
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add demo mode interceptor
    this.client.interceptors.request.use((config) => {
      const isDemoMode = this.isDemoMode();
      if (isDemoMode) {
        // Intercept and return demo data instead of making real API calls
        return this.handleDemoRequest(config);
      }
      return config;
    });
  }

  /**
   * Check if demo mode is enabled
   */
  private isDemoMode(): boolean {
    try {
      const demoMode = localStorage.getItem('mortgageai_demo_mode');
      return demoMode ? JSON.parse(demoMode) : false;
    } catch {
      return false;
    }
  }

  /**
   * Handle demo mode requests by returning local JSON fixtures
   */
  private async handleDemoRequest(config: any): Promise<any> {
    const url = config.url;

    try {
      if (url.includes('/api/dashboard/metrics')) {
        const demoData = await import('../demo-data/dashboardMetrics.json');
        return {
          data: demoData.default,
          status: 200,
          statusText: 'OK',
          headers: {},
          config
        };
      }

      if (url.includes('/api/dashboard/agent-status')) {
        const demoData = await import('../demo-data/recentActivity.json');
        return {
          data: { agents: demoData.default.activities.slice(0, 3).map((activity: any, index: number) => ({
            agent_type: index === 0 ? 'afm_compliance' : index === 1 ? 'dutch_mortgage_qc' : 'afm_compliance',
            status: 'online',
            processed_today: Math.floor(Math.random() * 20) + 10,
            success_rate: 95 + Math.random() * 5,
            last_activity: new Date().toISOString()
          })) },
          status: 200,
          statusText: 'OK',
          headers: {},
          config
        };
      }

      if (url.includes('/api/dashboard/lender-status')) {
        return {
          data: {
            lenders: [
              { lender_name: 'Stater', status: 'online', api_response_time_ms: 245, success_rate: 97.1, last_sync: new Date().toISOString() },
              { lender_name: 'Quion', status: 'online', api_response_time_ms: 189, success_rate: 98.3, last_sync: new Date().toISOString() },
              { lender_name: 'ING', status: 'online', api_response_time_ms: 156, success_rate: 96.8, last_sync: new Date().toISOString() }
            ]
          },
          status: 200,
          statusText: 'OK',
          headers: {},
          config
        };
      }

      if (url.includes('/api/dashboard/recent-activity')) {
        const demoData = await import('../demo-data/recentActivity.json');
        return {
          data: demoData.default,
          status: 200,
          statusText: 'OK',
          headers: {},
          config
        };
      }

      // For other endpoints, proceed with normal API call
      return config;
    } catch (error) {
      console.error('Demo mode error:', error);
      // Fall back to normal API call on error
      return config;
    }
  }

    // Request interceptor for authentication
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        // Production-grade error logging and handling
        const errorDetails = {
          message: error.message,
          status: error.response?.status,
          data: error.response?.data,
          url: error.config?.url,
          method: error.config?.method,
          timestamp: new Date().toISOString()
        };
        
        // Send to error tracking service in production
        if (process.env.NODE_ENV === 'production' && process.env.REACT_APP_ERROR_REPORTING_DSN) {
          // Integration with error tracking service (Sentry, LogRocket, etc.)
          this.reportError(errorDetails);
        }
        
        return Promise.reject(error);
      }
    );
  }

  // Production-grade error reporting
  private reportError(errorDetails: any): void {
    // In production, this would integrate with error tracking services
    // like Sentry, LogRocket, Bugsnag, etc.
    if (typeof window !== 'undefined' && (window as any).Sentry) {
      (window as any).Sentry.captureException(new Error(errorDetails.message), {
        extra: errorDetails
      });
    }
    
    // Could also send to custom analytics endpoint
    if (process.env.REACT_APP_ANALYTICS_ENDPOINT) {
      fetch(process.env.REACT_APP_ANALYTICS_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'error',
          ...errorDetails
        })
      }).catch(() => {
        // Silently fail if analytics endpoint is unavailable
      });
    }
  }

  // AFM Compliance Agent Integration
  async createClientIntake(clientData: any) {
    const response = await this.client.post('/api/afm/client-intake', clientData);
    return response.data;
  }

  async updateClientIntake(id: string, clientData: any) {
    const response = await this.client.put(`/api/afm/client-intake/${id}`, clientData);
    return response.data;
  }

  async getClientIntake(id: string) {
    const response = await this.client.get(`/api/afm/client-intake/${id}`);
    return response.data;
  }

  async submitClientIntake(id: string) {
    const response = await this.client.post(`/api/afm/client-intake/${id}/submit`);
    return response.data;
  }

  async validateClientProfile(clientData: any) {
    const response = await this.client.post('/api/afm/client-intake/validate', {
      client_profile: clientData,
      validation_type: 'suitability_assessment'
    });
    return response.data;
  }

  async getClientIntakeProgress(id: string) {
    const response = await this.client.get(`/api/afm/client-intake/${id}/progress`);
    return response.data;
  }

  async getClientStatus(clientId: string) {
    const response = await this.client.get(`/api/afm/client-status/${clientId}`);
    return response.data;
  }

  async createComplianceAssessment(assessmentData: any) {
    const response = await this.client.post('/api/afm/compliance/assessment', assessmentData);
    return response.data;
  }

  async getComplianceAssessment(assessmentId: string) {
    const response = await this.client.get(`/api/afm/compliance/assessment/${assessmentId}`);
    return response.data;
  }

  async getClientComplianceAssessment(clientId: string) {
    const response = await this.client.get(`/api/afm/compliance/client/${clientId}`);
    return response.data;
  }

  async updateComplianceAssessment(assessmentId: string, updates: any) {
    const response = await this.client.put(`/api/afm/compliance/assessment/${assessmentId}`, updates);
    return response.data;
  }

  async approveComplianceAssessment(assessmentId: string) {
    const response = await this.client.post(`/api/afm/compliance/assessment/${assessmentId}/approve`);
    return response.data;
  }

  async rejectComplianceAssessment(assessmentId: string, reason: string) {
    const response = await this.client.post(`/api/afm/compliance/assessment/${assessmentId}/reject`, { reason });
    return response.data;
  }

  async getComplianceHistory(clientId: string) {
    const response = await this.client.get(`/api/afm/compliance/history/${clientId}`);
    return response.data;
  }

  async getComplianceReport(assessmentId: string) {
    const response = await this.client.get(`/api/afm/compliance/report/${assessmentId}`);
    return response.data;
  }

  // Dutch Mortgage QC Agent Integration
  async runQCAnalysis(applicationId: string) {
    const response = await this.client.post(`/api/qc/run/${applicationId}`);
    return response.data;
  }

  async getQCResults(applicationId: string) {
    const response = await this.client.get(`/api/qc/result/${applicationId}`);
    return response.data;
  }

  async getSupportedLenders() {
    const response = await this.client.get('/api/lenders');
    return response.data;
  }

  async initiateBKRCheck(clientId: string, bsn: string, consentGiven: boolean) {
    const response = await this.client.post('/api/bkr-check', {
      client_id: clientId,
      bsn,
      consent_given: consentGiven
    });
    return response.data;
  }

  async getBKRCheckStatus(requestId: string) {
    const response = await this.client.get(`/api/bkr-check/${requestId}/status`);
    return response.data;
  }

  async getBKRReport(reportId: string) {
    const response = await this.client.get(`/api/bkr-report/${reportId}`);
    return response.data;
  }

  async getBKRReportByClient(clientId: string) {
    const response = await this.client.get(`/api/bkr-client/${clientId}/credit-report`);
    return response.data;
  }

  async refreshBKRReport(reportId: string) {
    const response = await this.client.post(`/api/bkr-report/${reportId}/refresh`);
    return response.data;
  }

  async getBKRScoreHistory(clientId: string) {
    const response = await this.client.get(`/api/bkr-client/${clientId}/score-history`);
    return response.data;
  }

  async getNHGEligibility(clientId: string) {
    const response = await this.client.get(`/api/nhg/client/${clientId}/eligibility`);
    return response.data;
  }

  async getMarketInsights() {
    const response = await this.client.get('/api/market/insights');
    return response.data;
  }

  async refreshMarketData() {
    const response = await this.client.post('/api/market/refresh');
    return response.data;
  }

  async validateBSN(bsn: string) {
    const response = await this.client.post('/api/validation/bsn', { bsn });
    return response.data;
  }

  // Application Management
  async createApplication(applicationData: any) {
    const response = await this.client.post('/api/applications', applicationData);
    return response.data;
  }

  async getApplication(applicationId: string) {
    const response = await this.client.get(`/api/applications/${applicationId}`);
    return response.data;
  }

  async updateApplication(applicationId: string, updates: any) {
    const response = await this.client.put(`/api/applications/${applicationId}`, updates);
    return response.data;
  }

  async submitApplication(applicationId: string, lenderIds: string[]) {
    const response = await this.client.post(`/api/applications/${applicationId}/submit`, {
      lender_ids: lenderIds
    });
    return response.data;
  }

  async getApplicationStatusUpdates(applicationId: string) {
    const response = await this.client.get(`/api/applications/${applicationId}/status-updates`);
    return response.data;
  }

  async uploadApplicationDocuments(applicationId: string, documents: FormData) {
    const response = await this.client.post(`/api/applications/${applicationId}/documents/upload`, documents, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  // Real-time updates using WebSocket or polling
  async subscribeToApplicationUpdates(applicationId: string, callback: (update: any) => void) {
    // Implement polling for real-time updates
    const poll = async () => {
      try {
        const response = await this.getApplication(applicationId);
        callback(response);
      } catch (error) {
        // Production-grade error handling for polling
        const errorDetails = {
          message: (error as Error).message || 'Unknown polling error',
          context: 'application_polling',
          applicationId,
          timestamp: new Date().toISOString()
        };
        
        if (process.env.NODE_ENV === 'production' && process.env.REACT_APP_ERROR_REPORTING_DSN) {
          this.reportError(errorDetails);
        }
      }
    };

    const intervalId = setInterval(poll, 5000); // Poll every 5 seconds
    return () => clearInterval(intervalId);
  }

  // Dashboard and Analytics
  async getDashboardMetrics() {
    const response = await this.client.get('/api/dashboard/metrics');
    return response.data;
  }

  async getAgentStatus() {
    const response = await this.client.get('/api/dashboard/agent-status');
    return response.data;
  }

  async getLenderStatus() {
    const response = await this.client.get('/api/dashboard/lender-status');
    return response.data;
  }

  async getRecentActivity() {
    const response = await this.client.get('/api/dashboard/recent-activity');
    return response.data;
  }

  // Agent Operations
  async runBatchComplianceCheck() {
    const response = await this.client.post('/api/afm/compliance/run-batch-check');
    return response.data;
  }

  async runBatchQCAnalysis() {
    const response = await this.client.post('/api/qc/run-batch-analysis');
    return response.data;
  }

  // Health checks
  async checkSystemHealth() {
    const response = await this.client.get('/api/health');
    return response.data;
  }

  async checkAgentHealth(agentType: 'afm' | 'qc') {
    const response = await this.client.get(`/api/${agentType}/health`);
    return response.data;
  }
}

export const apiClient = new MortgageAIApiClient();
