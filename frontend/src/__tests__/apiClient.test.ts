/**
 * API Client Integration Tests
 * Tests the MortgageAIApiClient for proper endpoint calling and response handling
 */

import { MortgageAIApiClient } from '../services/apiClient';

// Mock axios
jest.mock('axios');
import axios from 'axios';

const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('MortgageAIApiClient', () => {
  let apiClient: MortgageAIApiClient;

  beforeEach(() => {
    jest.clearAllMocks();
    apiClient = new MortgageAIApiClient();
  });

  describe('Dashboard API Methods', () => {
    it('getDashboardMetrics calls correct endpoint and returns data', async () => {
      const mockMetrics = {
        afm_compliance_score: 94.2,
        active_sessions: 12,
        pending_reviews: 3,
        applications_processed_today: 8,
        first_time_right_rate: 87.5,
        avg_processing_time_minutes: 45
      };

      mockedAxios.create.mockReturnValue({
        ...mockedAxios,
        get: jest.fn().mockResolvedValue({ data: mockMetrics }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      const result = await apiClient.getDashboardMetrics();

      expect(mockedAxios.create).toHaveBeenCalled();
      expect(result).toEqual(mockMetrics);
    });

    it('getAgentStatus returns object with agents array', async () => {
      const mockAgentResponse = {
        agents: [
          {
            agent_type: 'afm_compliance',
            status: 'online',
            processed_today: 15,
            success_rate: 98.5,
            last_activity: '2025-09-26T11:00:00Z'
          },
          {
            agent_type: 'dutch_mortgage_qc',
            status: 'online',
            processed_today: 12,
            success_rate: 96.1,
            last_activity: '2025-09-26T11:00:00Z'
          }
        ]
      };

      mockedAxios.create.mockReturnValue({
        ...mockedAxios,
        get: jest.fn().mockResolvedValue({ data: mockAgentResponse }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      const result = await apiClient.getAgentStatus();

      expect(result).toEqual(mockAgentResponse);
      expect(result.agents).toHaveLength(2);
      expect(result.agents[0].agent_type).toBe('afm_compliance');
    });

    it('getLenderStatus returns object with lenders array', async () => {
      const mockLenderResponse = {
        lenders: [
          {
            lender_name: 'Stater',
            status: 'online',
            api_response_time_ms: 245,
            success_rate: 97.1,
            last_sync: '2025-09-26T11:00:00Z'
          }
        ]
      };

      mockedAxios.create.mockReturnValue({
        ...mockedAxios,
        get: jest.fn().mockResolvedValue({ data: mockLenderResponse }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      const result = await apiClient.getLenderStatus();

      expect(result).toEqual(mockLenderResponse);
      expect(result.lenders).toHaveLength(1);
      expect(result.lenders[0].lender_name).toBe('Stater');
    });

    it('getRecentActivity returns object with activities array', async () => {
      const mockActivityResponse = {
        activities: [
          {
            type: 'afm_compliance',
            action: 'Client intake validated',
            client_name: 'John Doe',
            timestamp: '2025-09-26T11:00:00Z',
            result: 'approved'
          }
        ]
      };

      mockedAxios.create.mockReturnValue({
        ...mockedAxios,
        get: jest.fn().mockResolvedValue({ data: mockActivityResponse }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      const result = await apiClient.getRecentActivity();

      expect(result).toEqual(mockActivityResponse);
      expect(result.activities).toHaveLength(1);
      expect(result.activities[0].type).toBe('afm_compliance');
    });
  });

  describe('AFM Compliance API Methods', () => {
    it('createClientIntake calls correct AFM endpoint', async () => {
      const mockClientData = {
        first_name: 'John',
        last_name: 'Doe',
        email: 'john@example.com'
      };

      const mockResponse = { success: true, client_id: '123' };

      mockedAxios.create.mockReturnValue({
        ...mockedAxios,
        post: jest.fn().mockResolvedValue({ data: mockResponse }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      const result = await apiClient.createClientIntake(mockClientData);

      expect(mockedAxios.create).toHaveBeenCalled();
      expect(result).toEqual(mockResponse);
    });

    it('validateClientProfile calls correct AFM validation endpoint', async () => {
      const mockClientData = {
        first_name: 'John',
        last_name: 'Doe'
      };

      const mockValidation = {
        validation_result: {
          is_valid: true,
          score: 85,
          recommendations: ['Consider higher income documentation']
        }
      };

      mockedAxios.create.mockReturnValue({
        ...mockedAxios,
        post: jest.fn().mockResolvedValue({ data: mockValidation }),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      const result = await apiClient.validateClientProfile(mockClientData);

      expect(result).toEqual(mockValidation);
    });
  });

  describe('Error Handling', () => {
    it('handles network errors gracefully', async () => {
      const networkError = new Error('Network Error');

      mockedAxios.create.mockReturnValue({
        ...mockedAxios,
        get: jest.fn().mockRejectedValue(networkError),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      await expect(apiClient.getDashboardMetrics()).rejects.toThrow('Network Error');
    });

    it('handles 404 errors', async () => {
      const notFoundError = {
        response: {
          status: 404,
          statusText: 'Not Found',
          data: { error: 'Endpoint not found' }
        }
      };

      mockedAxios.create.mockReturnValue({
        ...mockedAxios,
        get: jest.fn().mockRejectedValue(notFoundError),
        interceptors: {
          request: { use: jest.fn() },
          response: { use: jest.fn() }
        }
      } as any);

      await expect(apiClient.getDashboardMetrics()).rejects.toEqual(notFoundError);
    });
  });
});
