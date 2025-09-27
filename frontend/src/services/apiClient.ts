import axios, { AxiosInstance } from 'axios';
import { demoDataService } from './demoDataService';

export class MortgageAIApiClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:3000';
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: { 'Content-Type': 'application/json' }
    });

    this.client.interceptors.request.use(async (config) => {
      if (this.isDemoMode()) {
        return this.handleDemoRequest(config);
      }

      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        const errorDetails = {
          message: error.message,
          status: error.response?.status,
          data: error.response?.data,
          url: error.config?.url,
          method: error.config?.method,
          timestamp: new Date().toISOString()
        };

        if (process.env.NODE_ENV === 'production' && process.env.REACT_APP_ERROR_REPORTING_DSN) {
          this.reportError(errorDetails);
        }
        return Promise.reject(error);
      }
    );
  }

  private isDemoMode(): boolean {
    try {
      return JSON.parse(localStorage.getItem('mortgageai_demo_mode') || 'false');
    } catch {
      return false;
    }
  }

  private async handleDemoRequest(config: any): Promise<any> {
    const url = config.url || '';
    
    try {
      // Dashboard endpoints
      if (url.includes('/api/dashboard/metrics')) {
        const data = demoDataService.getDashboardMetrics();
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      if (url.includes('/api/dashboard/recent-activity')) {
        const data = demoDataService.getRecentActivity();
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      if (url.includes('/api/dashboard/agent-status')) {
        const data = {
          agents: [
            { agent_type: 'afm_compliance', status: 'online', processed_today: 15, success_rate: 97.2, last_activity: new Date().toISOString() },
            { agent_type: 'dutch_mortgage_qc', status: 'online', processed_today: 12, success_rate: 95.8, last_activity: new Date().toISOString() }
          ]
        };
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      if (url.includes('/api/dashboard/lender-status')) {
        const lenderData = demoDataService.getLenderIntegrations();
        const data = {
          lenders: lenderData.connectedLenders.map(lender => ({
            lender_name: lender.name,
            status: lender.status,
            api_response_time_ms: lender.avgResponseTime,
            success_rate: lender.successRate,
            last_sync: lender.lastSync
          }))
        };
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Application endpoints
      if (url.includes('/api/applications')) {
        if (config.method === 'GET') {
          const data = { applications: demoDataService.getApplications() };
          return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
        }
        if (config.method === 'POST') {
          const newApp = { id: `app-${Date.now()}`, ...config.data, status: 'draft' };
          return { ...config, adapter: () => Promise.resolve({ data: newApp, status: 201, statusText: 'Created', config }) };
        }
      }

      // BKR endpoints
      if (url.includes('/api/bkr-check')) {
        if (url.includes('/status')) {
          const data = { status: 'completed', progress: 100 };
          return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
        }
        const bsnMatch = url.match(/bsn\/(\d+)/);
        if (bsnMatch) {
          const bsn = bsnMatch[1];
          const client = demoDataService.getClients().find(c => c.bsn === bsn);
          if (client) {
            const report = demoDataService.getBKRReport(client.id);
            return { ...config, adapter: () => Promise.resolve({ data: report, status: 200, statusText: 'OK', config }) };
          }
        }
        // Default BKR response
        const data = demoDataService.getBKRReports()[0];
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Quality Control endpoints
      if (url.includes('/api/quality-control') || url.includes('/api/qc')) {
        const appIdMatch = url.match(/\/([^\/]+)$/);
        const applicationId = appIdMatch ? appIdMatch[1] : 'app-001';
        const data = demoDataService.getQualityControlResults(applicationId);
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Compliance endpoints
      if (url.includes('/api/compliance')) {
        const appIdMatch = url.match(/\/([^\/]+)$/);
        const applicationId = appIdMatch ? appIdMatch[1] : 'app-001';
        const data = demoDataService.getComplianceAnalysis(applicationId);
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Document endpoints
      if (url.includes('/api/documents') || url.includes('/api/upload')) {
        const data = demoDataService.getDocumentProcessingResults();
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // NHG endpoints
      if (url.includes('/api/nhg')) {
        const appIdMatch = url.match(/\/([^\/]+)$/);
        const applicationId = appIdMatch ? appIdMatch[1] : 'app-001';
        const data = demoDataService.getNHGEligibility(applicationId);
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Market data endpoints
      if (url.includes('/api/market')) {
        const data = demoDataService.getMarketData();
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Lender integration endpoints
      if (url.includes('/api/lenders')) {
        const data = demoDataService.getLenderIntegrations();
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Settings endpoints
      if (url.includes('/api/settings')) {
        const data = demoDataService.getSystemSettings();
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Audit trail endpoints
      if (url.includes('/api/audit')) {
        const data = demoDataService.getAuditTrail();
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Client endpoints
      if (url.includes('/api/clients')) {
        const data = { clients: demoDataService.getClients() };
        return { ...config, adapter: () => Promise.resolve({ data, status: 200, statusText: 'OK', config }) };
      }

      // Default: pass through to real API (for production mode)
      return config;
    } catch (error) {
      console.error('Demo mode error', error);
      return config;
    }
  }

  private reportError(errorDetails: any): void {
    if (typeof window !== 'undefined' && (window as any).Sentry) {
      (window as any).Sentry.captureException(new Error(errorDetails.message), { extra: errorDetails });
    }

    if (process.env.REACT_APP_ANALYTICS_ENDPOINT) {
      fetch(process.env.REACT_APP_ANALYTICS_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'error', ...errorDetails })
      }).catch(() => {});
    }
  }

  async createClientIntake(payload: any) {
    const response = await this.client.post('/api/afm/client-intake', payload);
    return response.data;
  }

  async updateClientIntake(id: string, payload: any) {
    const response = await this.client.put(`/api/afm/client-intake/${id}`, payload);
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

  async validateClientProfile(payload: any) {
    const response = await this.client.post('/api/afm/client-intake/validate', { client_profile: payload, validation_type: 'suitability_assessment' });
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

  async performBKRCheck(bsn: string) {
    const response = await this.client.post('/api/bkr-check', { bsn });
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

  async createComplianceAssessment(payload: any) {
    const response = await this.client.post('/api/afm/compliance/assessment', payload);
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

  async runBatchComplianceCheck() {
    const response = await this.client.post('/api/afm/compliance/run-batch-check');
    return response.data;
  }

  async runBatchQCAnalysis() {
    const response = await this.client.post('/api/qc/run-batch-analysis');
    return response.data;
  }

  async analyzeDutchMortgageApplication(payload: any) {
    const response = await this.client.post('/api/qc/run', payload);
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

  async performBKRCreditCheck(payload: any) {
    const response = await this.client.post('/api/bkr-check', payload);
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

  async createApplication(payload: any) {
    const response = await this.client.post('/api/applications', payload);
    return response.data;
  }

  async getApplication(applicationId: string) {
    const response = await this.client.get(`/api/applications/${applicationId}`);
    return response.data;
  }

  async updateApplication(applicationId: string, payload: any) {
    const response = await this.client.put(`/api/applications/${applicationId}`, payload);
    return response.data;
  }

  async submitApplication(applicationId: string, lenderIds: string[]) {
    const response = await this.client.post(`/api/applications/${applicationId}/submit`, { lender_ids: lenderIds });
    return response.data;
  }

  async getApplicationStatusUpdates(applicationId: string) {
    const response = await this.client.get(`/api/applications/${applicationId}/status-updates`);
    return response.data;
  }

  async uploadApplicationDocuments(applicationId: string, documents: FormData) {
    const response = await this.client.post(`/api/applications/${applicationId}/documents/upload`, documents, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  }

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

  async checkSystemHealth() {
    const response = await this.client.get('/api/health');
    return response.data;
  }

  async checkAgentHealth(agent: 'afm' | 'qc') {
    const response = await this.client.get(`/api/${agent}/health`);
    return response.data;
  }
}

export const apiClient = new MortgageAIApiClient();
