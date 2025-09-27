/**
 * Settings API Service
 *
 * Handles API key validation and settings management
 */
import { useSnackbar } from 'notistack';

export interface APIKeyValidationResult {
  provider: string;
  isValid: boolean;
  message?: string;
  rateLimit?: {
    remaining: number;
    resetTime: string;
  };
  capabilities?: string[];
}

class SettingsApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  async validateAPIKey(provider: string, apiKey: string): Promise<APIKeyValidationResult> {
    try {
      const response = await fetch(`${this.baseUrl}/api/settings/validate-api-key`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify({ provider, api_key: apiKey }),
      });

      if (!response.ok) {
        throw new Error(`API key validation error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        provider,
        isValid: Boolean(data.is_valid),
        message: data.message,
        rateLimit: data.rate_limit ? {
          remaining: Number(data.rate_limit.remaining) || 0,
          resetTime: String(data.rate_limit.reset_time || ''),
        } : undefined,
        capabilities: Array.isArray(data.capabilities) ? data.capabilities : undefined,
      };
    } catch (error) {
      console.error('Failed to validate API key:', error);
      this.showError('Failed to validate API key');
      throw error;
    }
  }

  async testAPIConnection(provider: string, apiKey: string): Promise<{ success: boolean; responseTime?: number; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/settings/test-connection`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify({ provider, api_key: apiKey }),
      });

      if (!response.ok) {
        throw new Error(`API connection test error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: Boolean(data.success),
        responseTime: data.response_time ? Number(data.response_time) : undefined,
        error: data.error,
      };
    } catch (error) {
      console.error('Failed to test API connection:', error);
      this.showError('Failed to test API connection');
      throw error;
    }
  }

  async saveSettings(settings: Record<string, any>): Promise<{ success: boolean }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify(settings),
      });

      if (!response.ok) {
        throw new Error(`Settings save error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return {
        success: Boolean(data.success),
      };
    } catch (error) {
      console.error('Failed to save settings:', error);
      this.showError('Failed to save settings');
      throw error;
    }
  }

  async getSystemConfig(): Promise<Record<string, any>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/settings/system-config`, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
      });

      if (!response.ok) {
        throw new Error(`System config error: ${response.status} ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Failed to load system config:', error);
      this.showError('Failed to load system configuration');
      throw error;
    }
  }

  async validateApiKey(provider: string, apiKey: string) {
    return this.validateAPIKey(provider, apiKey);
  }

  async updateAuthRequirement(requireAuth: boolean): Promise<{ success: boolean }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/settings/authentication`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`,
        },
        body: JSON.stringify({ require_auth: requireAuth })
      });

      if (!response.ok) {
        throw new Error(`Auth update error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      return { success: Boolean(data.success) };
    } catch (error) {
      console.error('Failed to update authentication requirement:', error);
      this.showError('Failed to update authentication requirement');
      throw error;
    }
  }

  private getAuthToken(): string {
    return localStorage.getItem('auth_token') || '';
  }

  private showError(message: string) {
    if (this.enqueueSnackbar) {
      this.enqueueSnackbar(message, { variant: 'error' });
    }
  }
}

export const settingsApi = new SettingsApiService();
