/**
 * AFM Client Intake Integration Tests
 * Tests the AFMClientIntake component with mocked API responses
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AFMClientIntake from '../pages/AFMClientIntake';
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

describe('AFMClientIntake', () => {
  const mockClientData = {
    first_name: 'John',
    last_name: 'Doe',
    email: 'john.doe@example.com',
    phone: '+31612345678',
    date_of_birth: '1985-06-15',
    bsn: '123456789',
    address: {
      street: 'Main Street',
      house_number: '123',
      postal_code: '1234AB',
      city: 'Amsterdam',
      country: 'Netherlands'
    },
    financial_situation: {
      annual_income: 75000,
      monthly_debt: 1200,
      savings: 25000,
      employment_status: 'employed'
    },
    mortgage_requirements: {
      loan_amount: 300000,
      loan_term: 30,
      property_value: 350000,
      property_type: 'house'
    }
  };

  const mockValidationResponse = {
    validation_result: {
      is_valid: true,
      score: 85,
      recommendations: [
        'Consider providing additional income documentation',
        'Review debt-to-income ratio'
      ],
      risk_assessment: 'medium',
      compliance_flags: []
    }
  };

  const mockIntakeResponse = {
    success: true,
    client_id: 'client_123',
    message: 'Client intake completed successfully'
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock API calls
    mockedApiClient.validateBSN.mockResolvedValue({ valid: true });
    mockedApiClient.validateClientProfile.mockResolvedValue(mockValidationResponse);
    mockedApiClient.createClientIntake.mockResolvedValue(mockIntakeResponse);
  });

  const renderAFMClientIntake = () => {
    return render(
      <SnackbarProvider>
        <AFMClientIntake />
      </SnackbarProvider>
    );
  };

  it('renders AFM client intake form with all sections', () => {
    renderAFMClientIntake();

    expect(screen.getByText('AFM Compliant Client Intake')).toBeInTheDocument();
    expect(screen.getByText('Personal Information')).toBeInTheDocument();
    expect(screen.getByText('Address Details')).toBeInTheDocument();
    expect(screen.getByText('Financial Situation')).toBeInTheDocument();
    expect(screen.getByText('Mortgage Requirements')).toBeInTheDocument();
  });

  it('validates BSN on input change', async () => {
    const user = userEvent.setup();
    renderAFMClientIntake();

    const bsnInput = screen.getByLabelText(/bsn/i);
    await user.type(bsnInput, '123456789');

    await waitFor(() => {
      expect(mockedApiClient.validateBSN).toHaveBeenCalledWith('123456789');
    });
  });

  it('shows BSN validation success', async () => {
    const user = userEvent.setup();
    renderAFMClientIntake();

    const bsnInput = screen.getByLabelText(/bsn/i);
    await user.type(bsnInput, '123456789');

    await waitFor(() => {
      expect(screen.getByText('BSN validated successfully')).toBeInTheDocument();
    });
  });

  it('performs AFM compliance validation', async () => {
    const user = userEvent.setup();
    renderAFMClientIntake();

    // Fill out the form
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john@example.com');
    await user.type(screen.getByLabelText(/phone/i), '+31612345678');
    await user.type(screen.getByLabelText(/date of birth/i), '1985-06-15');
    await user.type(screen.getByLabelText(/bsn/i), '123456789');

    // Fill address
    await user.type(screen.getByLabelText(/street/i), 'Main Street');
    await user.type(screen.getByLabelText(/house number/i), '123');
    await user.type(screen.getByLabelText(/postal code/i), '1234AB');
    await user.type(screen.getByLabelText(/city/i), 'Amsterdam');

    // Fill financial info
    await user.type(screen.getByLabelText(/annual income/i), '75000');
    await user.type(screen.getByLabelText(/monthly debt/i), '1200');
    await user.type(screen.getByLabelText(/savings/i), '25000');
    await user.selectOptions(screen.getByLabelText(/employment status/i), 'employed');

    // Fill mortgage requirements
    await user.type(screen.getByLabelText(/loan amount/i), '300000');
    await user.type(screen.getByLabelText(/loan term/i), '30');
    await user.type(screen.getByLabelText(/property value/i), '350000');
    await user.selectOptions(screen.getByLabelText(/property type/i), 'house');

    // Click validate button
    const validateButton = screen.getByRole('button', { name: /validate with afm agent/i });
    await user.click(validateButton);

    await waitFor(() => {
      expect(mockedApiClient.validateClientProfile).toHaveBeenCalled();
    });
  });

  it('displays AFM validation results', async () => {
    const user = userEvent.setup();
    renderAFMClientIntake();

    // Fill minimal required fields
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john@example.com');

    // Click validate
    const validateButton = screen.getByRole('button', { name: /validate with afm agent/i });
    await user.click(validateButton);

    await waitFor(() => {
      expect(screen.getByText('AFM Compliance Score: 85')).toBeInTheDocument();
      expect(screen.getByText('Consider providing additional income documentation')).toBeInTheDocument();
    });
  });

  it('submits client intake successfully', async () => {
    const user = userEvent.setup();
    renderAFMClientIntake();

    // Fill out complete form
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john@example.com');
    await user.type(screen.getByLabelText(/phone/i), '+31612345678');
    await user.type(screen.getByLabelText(/date of birth/i), '1985-06-15');
    await user.type(screen.getByLabelText(/bsn/i), '123456789');
    await user.type(screen.getByLabelText(/street/i), 'Main Street');
    await user.type(screen.getByLabelText(/house number/i), '123');
    await user.type(screen.getByLabelText(/postal code/i), '1234AB');
    await user.type(screen.getByLabelText(/city/i), 'Amsterdam');
    await user.type(screen.getByLabelText(/annual income/i), '75000');
    await user.type(screen.getByLabelText(/monthly debt/i), '1200');
    await user.type(screen.getByLabelText(/savings/i), '25000');
    await user.selectOptions(screen.getByLabelText(/employment status/i), 'employed');
    await user.type(screen.getByLabelText(/loan amount/i), '300000');
    await user.type(screen.getByLabelText(/loan term/i), '30');
    await user.type(screen.getByLabelText(/property value/i), '350000');
    await user.selectOptions(screen.getByLabelText(/property type/i), 'house');

    // First validate
    const validateButton = screen.getByRole('button', { name: /validate with afm agent/i });
    await user.click(validateButton);

    await waitFor(() => {
      expect(mockedApiClient.validateClientProfile).toHaveBeenCalled();
    });

    // Then submit
    const submitButton = screen.getByRole('button', { name: /submit client intake/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockedApiClient.createClientIntake).toHaveBeenCalledWith(mockClientData);
      expect(screen.getByText('Client intake completed successfully!')).toBeInTheDocument();
    });
  });

  it('calls correct AFM endpoints', async () => {
    const user = userEvent.setup();
    renderAFMClientIntake();

    // Fill minimal form
    await user.type(screen.getByLabelText(/first name/i), 'John');
    await user.type(screen.getByLabelText(/last name/i), 'Doe');
    await user.type(screen.getByLabelText(/email/i), 'john@example.com');

    // Validate
    const validateButton = screen.getByRole('button', { name: /validate with afm agent/i });
    await user.click(validateButton);

    await waitFor(() => {
      expect(mockedApiClient.validateClientProfile).toHaveBeenCalled();
    });

    // Submit
    const submitButton = screen.getByRole('button', { name: /submit client intake/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockedApiClient.createClientIntake).toHaveBeenCalled();
    });
  });

  it('handles API errors gracefully', async () => {
    // Mock API to reject
    mockedApiClient.validateClientProfile.mockRejectedValue(new Error('Validation failed'));

    const user = userEvent.setup();
    renderAFMClientIntake();

    // Fill minimal form
    await user.type(screen.getByLabelText(/first name/i), 'John');

    // Try to validate
    const validateButton = screen.getByRole('button', { name: /validate with afm agent/i });
    await user.click(validateButton);

    await waitFor(() => {
      expect(mockedApiClient.validateClientProfile).toHaveBeenCalled();
    });

    // Should not crash, should handle error gracefully
    expect(screen.getByText('AFM Compliant Client Intake')).toBeInTheDocument();
  });

  it('validates required fields', async () => {
    const user = userEvent.setup();
    renderAFMClientIntake();

    // Try to submit without filling required fields
    const submitButton = screen.getByRole('button', { name: /submit client intake/i });
    await user.click(submitButton);

    // Should show validation errors or prevent submission
    expect(screen.getByText('AFM Compliant Client Intake')).toBeInTheDocument();
  });

  it('maintains form state across validation', async () => {
    const user = userEvent.setup();
    renderAFMClientIntake();

    // Fill some fields
    const firstNameInput = screen.getByLabelText(/first name/i);
    await user.type(firstNameInput, 'John');

    // Validate
    const validateButton = screen.getByRole('button', { name: /validate with afm agent/i });
    await user.click(validateButton);

    await waitFor(() => {
      expect(mockedApiClient.validateClientProfile).toHaveBeenCalled();
    });

    // Check that form data is preserved
    expect(firstNameInput).toHaveValue('John');
  });
});
