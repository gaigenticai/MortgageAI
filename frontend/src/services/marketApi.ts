/**
 * Market Insights API Service
 *
 * Provides real-time market intelligence for Dutch mortgage market
 */
import { useSnackbar } from 'notistack';

export interface MarketIndicator {
  id: string;
  name: string;
  value: number;
  change: number;
  trend: 'up' | 'down' | 'stable';
  unit: string;
  last_updated: string;
}

export interface LenderRate {
  lender: string;
  fixed_10yr: number;
  fixed_20yr: number;
  variable: number;
  last_updated: string;
}

export interface RegulatoryUpdate {
  id: string;
  title: string;
  summary: string;
  impact: 'high' | 'medium' | 'low';
  date: string;
  category: 'afm' | 'ecb' | 'government' | 'industry';
}

export interface MarketInsights {
  indicators: MarketIndicator[];
  lender_rates: LenderRate[];
  regulatory_updates: RegulatoryUpdate[];
  market_summary: {
    overall_trend: 'bullish' | 'bearish' | 'neutral';
    key_drivers: string[];
    forecast_3m: string;
    risk_factors: string[];
  };
}

class MarketApiService {
  private baseUrl: string;
  private enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar'] | null = null;

  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  }

  setSnackbar(enqueueSnackbar: ReturnType<typeof useSnackbar>['enqueueSnackbar']) {
    this.enqueueSnackbar = enqueueSnackbar;
  }

  async getMarketInsights(): Promise<MarketInsights> {
    try {
      const response = await fetch(`${this.baseUrl}/api/market/insights`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${this.getAuthToken()}` },
      });
      if (!response.ok) throw new Error(`Market API error: ${response.status}`);
      return await response.json();
    } catch (error) {
      this.showError('Failed to load market insights');
      throw error;
    }
  }

  async refreshMarketData(): Promise<MarketInsights> {
    try {
      const response = await fetch(`${this.baseUrl}/api/market/refresh`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${this.getAuthToken()}` },
      });
      if (!response.ok) throw new Error(`Refresh API error: ${response.status}`);
      return await response.json();
    } catch (error) {
      this.showError('Failed to refresh market data');
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

export const marketApi = new MarketApiService();
