/**
 * Audit Trail API Service
 *
 * Provides comprehensive audit trail for AFM compliance activities
 */
import { useSnackbar } from 'notistack';

export interface AuditEntry {
  id: string;
  timestamp: string;
  action: string;
  actor: string;
  client_id?: string;
  client_name?: string;
  details: Record<string, any>;
  compliance_score?: number;
  risk_level: 'low' | 'medium' | 'high';
  status: 'success' | 'warning' | 'error';
  category: 'client_intake' | 'compliance_check' | 'application' | 'audit' | 'system';
}

export interface ComplianceStats {
  total_entries: number;
  compliance_checks: number;
  high_risk_actions: number;
  audit_alerts: number;
  average_score: number;
  last_audit: string;
}

class AuditApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  async getAuditEntries(limit: number = 50): Promise<AuditEntry[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/audit/entries?limit=${limit}`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${this.getAuthToken()}` },
      });
      if (!response.ok) throw new Error(`Audit API error: ${response.status}`);
      const data = await response.json();
      return Array.isArray(data.entries) ? data.entries : [];
    } catch (error) {
      this.showError('Failed to load audit entries');
      throw error;
    }
  }

  async getComplianceStats(): Promise<ComplianceStats> {
    try {
      const response = await fetch(`${this.baseUrl}/api/audit/stats`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${this.getAuthToken()}` },
      });
      if (!response.ok) throw new Error(`Stats API error: ${response.status}`);
      return await response.json();
    } catch (error) {
      this.showError('Failed to load compliance stats');
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

export const auditApi = new AuditApiService();
