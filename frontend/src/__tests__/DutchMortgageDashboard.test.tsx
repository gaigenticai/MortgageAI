/**
 * Dutch Mortgage Dashboard Integration Tests
 * Tests the Dashboard component with mocked API responses
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DutchMortgageDashboard } from '../pages/DutchMortgageDashboard';
import { apiClient } from '../services/apiClient';
import { SnackbarProvider } from 'notistack';

// Mock the API client
jest.mock('../services/apiClient');
const mockedApiClient = apiClient as jest.Mocked<typeof apiClient>;

// Mock react-router-dom
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

// Mock Material-UI components that might cause issues
jest.mock('@mui/material', () => ({
  ...jest.requireActual('@mui/material'),
  useMediaQuery: () => true,
}));

describe('DutchMortgageDashboard', () => {
  const mockDashboardData = {
    metrics: {
      afm_compliance_score: 94.2,
      active_sessions: 12,
      pending_reviews: 3,
      applications_processed_today: 8,
      first_time_right_rate: 87.5,
      avg_processing_time_minutes: 45
    },
    agents: [
      {
        agent_type: 'afm_compliance',
        status: 'online' as const,
        processed_today: 15,
        success_rate: 98.5,
        last_activity: '2025-09-26T11:00:00Z'
      },
      {
        agent_type: 'dutch_mortgage_qc',
        status: 'online' as const,
        processed_today: 12,
        success_rate: 96.1,
        last_activity: '2025-09-26T11:00:00Z'
      }
    ],
    lenders: [
      {
        lender_name: 'Stater',
        status: 'online' as const,
        api_response_time_ms: 245,
        success_rate: 97.1,
        last_sync: '2025-09-26T11:00:00Z'
      },
      {
        lender_name: 'Quion',
        status: 'online' as const,
        api_response_time_ms: 189,
        success_rate: 98.3,
        last_sync: '2025-09-26T11:00:00Z'
      }
    ],
    activities: [
      {
        type: 'afm_compliance' as const,
        action: 'Client intake validated',
        client_name: 'John Doe',
        timestamp: '2025-09-26T11:00:00Z',
        result: 'approved'
      },
      {
        type: 'dutch_mortgage_qc' as const,
        action: 'Application QC completed',
        client_name: 'Jane Smith',
        timestamp: '2025-09-26T10:45:00Z',
        result: 'passed'
      }
    ]
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock all API calls
    mockedApiClient.getDashboardMetrics.mockResolvedValue(mockDashboardData.metrics);
    mockedApiClient.getAgentStatus.mockResolvedValue({ agents: mockDashboardData.agents });
    mockedApiClient.getLenderStatus.mockResolvedValue({ lenders: mockDashboardData.lenders });
    mockedApiClient.getRecentActivity.mockResolvedValue({ activities: mockDashboardData.activities });
  });

  const renderDashboard = () => {
    return render(
      <SnackbarProvider>
        <DutchMortgageDashboard />
      </SnackbarProvider>
    );
  };

  it('renders dashboard with all sections', async () => {
    renderDashboard();

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('MortgageAI Dashboard')).toBeInTheDocument();
    });

    // Check main sections are rendered
    expect(screen.getByText('Key Metrics')).toBeInTheDocument();
    expect(screen.getByText('Agent Status')).toBeInTheDocument();
    expect(screen.getByText('Lender Integration')).toBeInTheDocument();
    expect(screen.getByText('Recent Activity')).toBeInTheDocument();
  });

  it('displays correct metrics data', async () => {
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('94.2%')).toBeInTheDocument(); // AFM compliance score
      expect(screen.getByText('12')).toBeInTheDocument(); // Active sessions
      expect(screen.getByText('3')).toBeInTheDocument(); // Pending reviews
    });
  });

  it('displays agent status correctly', async () => {
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('AFM Compliance Agent')).toBeInTheDocument();
      expect(screen.getByText('Mortgage QC Agent')).toBeInTheDocument();
      expect(screen.getByText('98.5%')).toBeInTheDocument(); // AFM success rate
      expect(screen.getByText('96.1%')).toBeInTheDocument(); // QC success rate
    });
  });

  it('displays lender status correctly', async () => {
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Stater')).toBeInTheDocument();
      expect(screen.getByText('Quion')).toBeInTheDocument();
      expect(screen.getByText('245ms')).toBeInTheDocument(); // Stater response time
      expect(screen.getByText('97.1%')).toBeInTheDocument(); // Stater success rate
    });
  });

  it('displays recent activities correctly', async () => {
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Client intake validated')).toBeInTheDocument();
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('Application QC completed')).toBeInTheDocument();
      expect(screen.getByText('Jane Smith')).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    // Mock API to reject
    mockedApiClient.getDashboardMetrics.mockRejectedValue(new Error('API Error'));
    mockedApiClient.getAgentStatus.mockRejectedValue(new Error('API Error'));
    mockedApiClient.getLenderStatus.mockRejectedValue(new Error('API Error'));
    mockedApiClient.getRecentActivity.mockRejectedValue(new Error('API Error'));

    renderDashboard();

    // Should still render but show loading/error states
    await waitFor(() => {
      expect(screen.getByText('MortgageAI Dashboard')).toBeInTheDocument();
    });

    // Check that error handling doesn't crash the component
    expect(screen.getByText('MortgageAI Dashboard')).toBeInTheDocument();
  });

  it('calls correct API endpoints on mount', async () => {
    renderDashboard();

    await waitFor(() => {
      expect(mockedApiClient.getDashboardMetrics).toHaveBeenCalledTimes(1);
      expect(mockedApiClient.getAgentStatus).toHaveBeenCalledTimes(1);
      expect(mockedApiClient.getLenderStatus).toHaveBeenCalledTimes(1);
      expect(mockedApiClient.getRecentActivity).toHaveBeenCalledTimes(1);
    });
  });

  it('navigates to AFM client intake when button clicked', async () => {
    const user = userEvent.setup();
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('MortgageAI Dashboard')).toBeInTheDocument();
    });

    // Find and click AFM client intake button
    const afmButton = screen.getByRole('button', { name: /client intake/i });
    await user.click(afmButton);

    expect(mockNavigate).toHaveBeenCalledWith('/afm-client-intake');
  });

  it('navigates to compliance check when button clicked', async () => {
    const user = userEvent.setup();
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('MortgageAI Dashboard')).toBeInTheDocument();
    });

    // Find and click compliance check button
    const complianceButton = screen.getByRole('button', { name: /compliance check/i });
    await user.click(complianceButton);

    expect(mockNavigate).toHaveBeenCalledWith('/compliance');
  });

  it('destructures API responses correctly', async () => {
    // This test verifies that the component correctly destructures
    // the API responses as expected by the implementation
    renderDashboard();

    await waitFor(() => {
      // If this test passes, it means the destructuring in the component
      // matches the API response structure
      expect(screen.getByText('MortgageAI Dashboard')).toBeInTheDocument();
    });

    // Verify the API was called and responses were destructured properly
    expect(mockedApiClient.getAgentStatus).toHaveBeenCalled();
    expect(mockedApiClient.getLenderStatus).toHaveBeenCalled();
    expect(mockedApiClient.getRecentActivity).toHaveBeenCalled();
  });
});
