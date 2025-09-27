/**
 * AFM Client Intake - Full Mantine Implementation
 */

import React, { useState } from 'react';
import {
  Container,
  Card,
  TextInput,
  NumberInput,
  Select,
  Checkbox,
  Stepper,
  Title,
  Group,
  Stack,
  Grid,
  Button,
  Alert,
  Badge,
  Progress,
  RingProgress,
  ThemeIcon,
  Text,
} from '@mantine/core';
import {
  IconUser,
  IconBriefcase,
  IconHome,
  IconShield,
  IconCheck,
  IconArrowRight,
  IconArrowLeft,
  IconDeviceFloppy,
  IconSend,
  IconInfoCircle,
} from '@tabler/icons-react';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import { useNavigate } from 'react-router-dom';
import { clientIntakeApi, ClientProfile } from '../services/clientIntakeApi';
import { useDemoMode } from '../contexts/DemoModeContext';

interface FormData {
  personal_info: {
    full_name: string;
    bsn: string;
    date_of_birth: string;
    marital_status: string;
    number_of_dependents: number;
    email: string;
    phone: string;
  };
  employment_info: {
    employment_status: string;
    employer_name: string;
    job_title: string;
    employment_duration_months: number;
    gross_annual_income: number;
    partner_income: number;
    other_income_amount: number;
  };
  mortgage_requirements: {
    property_type: string;
    property_location: string;
    estimated_property_value: number;
    desired_mortgage_amount: number;
    preferred_mortgage_term: number;
    interest_rate_preference: string;
    down_payment_amount: number;
    nhg_required: boolean;
  };
  afm_suitability: {
    mortgage_experience: string;
    financial_knowledge_level: string;
    risk_tolerance: string;
    sustainability_preferences: string;
    expected_advice_frequency: string;
  };
}

const AFMClientIntake: React.FC = () => {
  const navigate = useNavigate();
  const { isDemoMode } = useDemoMode();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [clientId, setClientId] = useState<string | null>(null);
  const [complianceScore, setComplianceScore] = useState(75);
  const [riskProfile, setRiskProfile] = useState<'low' | 'medium' | 'high'>('medium');

  const form = useForm<FormData>({
    initialValues: {
      personal_info: {
        full_name: '',
        bsn: '',
        date_of_birth: '',
        marital_status: 'single',
        number_of_dependents: 0,
        email: '',
        phone: '',
      },
      employment_info: {
        employment_status: 'employed',
        employer_name: '',
        job_title: '',
        employment_duration_months: 0,
        gross_annual_income: 0,
        partner_income: 0,
        other_income_amount: 0,
      },
      mortgage_requirements: {
        property_type: 'house',
        property_location: '',
        estimated_property_value: 0,
        desired_mortgage_amount: 0,
        preferred_mortgage_term: 30,
        interest_rate_preference: 'fixed',
        down_payment_amount: 0,
        nhg_required: false,
      },
      afm_suitability: {
        mortgage_experience: 'first_time',
        financial_knowledge_level: 'basic',
        risk_tolerance: 'moderate',
        sustainability_preferences: 'somewhat_important',
        expected_advice_frequency: 'one_time',
      },
    },
  });

  const saveDraft = async () => {
    setSaving(true);
    try {
      if (isDemoMode) {
        // Demo mode: simulate save with delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        notifications.show({
          title: 'Draft Saved',
          message: 'Your client profile has been saved as draft (Demo)',
          color: 'green',
          icon: <IconDeviceFloppy size={16} />,
        });
      } else {
        // Production mode: save to real API
        const clientProfile = {
          personal_info: {
            ...form.values.personal_info,
            marital_status: form.values.personal_info.marital_status as 'single' | 'married' | 'registered_partnership' | 'divorced' | 'widowed',
          },
          employment_info: {
            ...form.values.employment_info,
            employment_status: form.values.employment_info.employment_status as 'employed' | 'self_employed' | 'unemployed' | 'retired' | 'student' | 'other',
            other_income_sources: [], // Add default empty array
          },
          financial_situation: {
            existing_debts: [],
            monthly_expenses: 0,
            savings_amount: 0,
            investments: [],
            other_properties: false,
          },
          mortgage_requirements: {
            ...form.values.mortgage_requirements,
            property_type: form.values.mortgage_requirements.property_type as 'house' | 'apartment' | 'townhouse' | 'condo' | 'other',
            interest_rate_preference: form.values.mortgage_requirements.interest_rate_preference as 'fixed' | 'variable' | 'flexible',
          },
          afm_suitability: {
            ...form.values.afm_suitability,
            mortgage_experience: form.values.afm_suitability.mortgage_experience as 'first_time' | 'experienced' | 'very_experienced',
            financial_knowledge_level: form.values.afm_suitability.financial_knowledge_level as 'basic' | 'intermediate' | 'advanced',
            risk_tolerance: form.values.afm_suitability.risk_tolerance as 'conservative' | 'moderate' | 'aggressive',
            expected_advice_frequency: form.values.afm_suitability.expected_advice_frequency as 'one_time' | 'regular' | 'ongoing',
            sustainability_preferences: form.values.afm_suitability.sustainability_preferences as 'not_important' | 'somewhat_important' | 'very_important',
            investment_objectives: [], // Add default empty array
            advice_needs: [], // Add default empty array
          },
          status: 'draft' as const,
          compliance_score: complianceScore,
          risk_profile: riskProfile,
        };

        let savedProfile: ClientProfile;
        if (clientId) {
          // Update existing profile
          savedProfile = await clientIntakeApi.updateClientProfile(clientId, clientProfile);
        } else {
          // Create new profile
          savedProfile = await clientIntakeApi.createClientProfile(clientProfile);
          setClientId(savedProfile.id || null);
        }
        
        setComplianceScore(savedProfile.compliance_score || 75);
        setRiskProfile(savedProfile.risk_profile || 'medium');
        
        notifications.show({
          title: 'Draft Saved',
          message: 'Your client profile has been saved as draft',
          color: 'green',
          icon: <IconDeviceFloppy size={16} />,
        });
      }
    } catch (error) {
      console.error('Failed to save draft:', error);
      notifications.show({
        title: 'Error',
        message: 'Failed to save application draft',
        color: 'red',
        icon: <IconDeviceFloppy size={16} />,
      });
    } finally {
      setSaving(false);
    }
  };

  const submitProfile = async () => {
    setLoading(true);
    try {
      if (isDemoMode) {
        // Demo mode: simulate submission
        await saveDraft();
        notifications.show({
          title: 'Profile Submitted',
          message: 'Your client profile has been submitted for review (Demo)',
          color: 'green',
          icon: <IconCheck size={16} />,
        });
        navigate('/compliance');
      } else {
        // Production mode: real API submission
        // First save as draft to ensure we have a client ID
        await saveDraft();
        
        if (!clientId) {
          throw new Error('No client ID available for submission');
        }

        // Submit the profile for AFM compliance review
        const result = await clientIntakeApi.submitClientProfile(clientId);
        
        notifications.show({
          title: 'Profile Submitted',
          message: `Your client profile has been submitted for AFM compliance review. Compliance Score: ${result.compliance_score}%`,
          color: 'green',
          icon: <IconCheck size={16} />,
        });
        
        navigate('/compliance');
      }
    } catch (error) {
      console.error('Failed to submit profile:', error);
      notifications.show({
        title: 'Submission Failed',
        message: 'Failed to save application. Please try again.',
        color: 'red',
        icon: <IconSend size={16} />,
      });
    } finally {
      setLoading(false);
    }
  };

  const nextStep = () => setActiveStep((current) => Math.min(current + 1, 4));
  const prevStep = () => setActiveStep((current) => Math.max(current - 1, 0));

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        {/* Header */}
        <Group justify="space-between">
          <Group>
            <ThemeIcon size="xl" radius={0} color="indigo">
              <IconUser size={32} />
            </ThemeIcon>
            <div>
              <Title order={1}>AFM Client Intake</Title>
              <Text c="dimmed">Comprehensive client assessment and suitability analysis</Text>
            </div>
          </Group>
          <Group>
            <Badge color={riskProfile === 'low' ? 'green' : riskProfile === 'medium' ? 'yellow' : 'red'} size="lg" radius={0}>
              {riskProfile.toUpperCase()} RISK
            </Badge>
            <RingProgress
              size={60}
              thickness={6}
              sections={[{ value: complianceScore, color: complianceScore >= 80 ? 'green' : complianceScore >= 60 ? 'yellow' : 'red' }]}
              label={<Text size="xs" ta="center">{complianceScore}%</Text>}
            />
          </Group>
        </Group>

        {/* Progress Stepper */}
        <Card radius={0} shadow="sm" padding="lg">
          <Stepper active={activeStep} radius={0}>
            <Stepper.Step 
              label="Personal Information" 
              description="Basic client details"
              icon={<IconUser size={18} />}
            />
            <Stepper.Step 
              label="Employment & Income" 
              description="Employment status and income"
              icon={<IconBriefcase size={18} />}
            />
            <Stepper.Step 
              label="Mortgage Requirements" 
              description="Property and loan preferences"
              icon={<IconHome size={18} />}
            />
            <Stepper.Step 
              label="AFM Suitability" 
              description="Risk assessment and compliance"
              icon={<IconShield size={18} />}
            />
            <Stepper.Step 
              label="Review & Submit" 
              description="Final review and submission"
              icon={<IconCheck size={18} />}
            />
          </Stepper>
        </Card>

        <form onSubmit={form.onSubmit(() => {})}>
          {/* Step 1: Personal Information */}
          {activeStep === 0 && (
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Personal Information</Title>
              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Full Name"
                    placeholder="Enter full legal name"
                    required
                    radius={0}
                    {...form.getInputProps('personal_info.full_name')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="BSN (Dutch Social Security Number)"
                    placeholder="123456789"
                    required
                    radius={0}
                    {...form.getInputProps('personal_info.bsn')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Date of Birth"
                    placeholder="YYYY-MM-DD"
                    type="date"
                    required
                    radius={0}
                    {...form.getInputProps('personal_info.date_of_birth')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Marital Status"
                    placeholder="Select marital status"
                    data={[
                      { value: 'single', label: 'Single' },
                      { value: 'married', label: 'Married' },
                      { value: 'registered_partnership', label: 'Registered Partnership' },
                      { value: 'divorced', label: 'Divorced' },
                      { value: 'widowed', label: 'Widowed' },
                    ]}
                    radius={0}
                    {...form.getInputProps('personal_info.marital_status')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Number of Dependents"
                    placeholder="0"
                    min={0}
                    radius={0}
                    {...form.getInputProps('personal_info.number_of_dependents')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Email Address"
                    placeholder="your.email@example.com"
                    type="email"
                    required
                    radius={0}
                    {...form.getInputProps('personal_info.email')}
                  />
                </Grid.Col>
                <Grid.Col span={12}>
                  <TextInput
                    label="Phone Number"
                    placeholder="+31 6 12345678"
                    required
                    radius={0}
                    {...form.getInputProps('personal_info.phone')}
                  />
                </Grid.Col>
              </Grid>
            </Card>
          )}

          {/* Step 2: Employment & Income */}
          {activeStep === 1 && (
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Employment & Income Information</Title>
              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Employment Status"
                    data={[
                      { value: 'employed', label: 'Employed' },
                      { value: 'self_employed', label: 'Self-Employed' },
                      { value: 'unemployed', label: 'Unemployed' },
                      { value: 'retired', label: 'Retired' },
                      { value: 'student', label: 'Student' },
                      { value: 'other', label: 'Other' },
                    ]}
                    radius={0}
                    {...form.getInputProps('employment_info.employment_status')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Employer Name"
                    placeholder="Company Name"
                    radius={0}
                    {...form.getInputProps('employment_info.employer_name')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Job Title"
                    placeholder="Your job title"
                    radius={0}
                    {...form.getInputProps('employment_info.job_title')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Employment Duration (Months)"
                    placeholder="24"
                    min={0}
                    radius={0}
                    {...form.getInputProps('employment_info.employment_duration_months')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Gross Annual Income (€)"
                    placeholder="75000"
                    min={0}
                    step={1000}
                    thousandSeparator=","
                    required
                    radius={0}
                    {...form.getInputProps('employment_info.gross_annual_income')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Partner Income (€)"
                    placeholder="45000"
                    min={0}
                    step={1000}
                    thousandSeparator=","
                    radius={0}
                    {...form.getInputProps('employment_info.partner_income')}
                  />
                </Grid.Col>
                <Grid.Col span={12}>
                  <NumberInput
                    label="Other Income Amount (€)"
                    placeholder="5000"
                    min={0}
                    step={1000}
                    thousandSeparator=","
                    radius={0}
                    {...form.getInputProps('employment_info.other_income_amount')}
                  />
                </Grid.Col>
              </Grid>
            </Card>
          )}

          {/* Step 3: Mortgage Requirements */}
          {activeStep === 2 && (
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Mortgage Requirements</Title>
              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Property Type"
                    data={[
                      { value: 'apartment', label: 'Apartment' },
                      { value: 'house', label: 'House' },
                      { value: 'townhouse', label: 'Townhouse' },
                      { value: 'condo', label: 'Condominium' },
                      { value: 'other', label: 'Other' },
                    ]}
                    radius={0}
                    {...form.getInputProps('mortgage_requirements.property_type')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <TextInput
                    label="Property Location"
                    placeholder="Amsterdam, Netherlands"
                    radius={0}
                    {...form.getInputProps('mortgage_requirements.property_location')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Estimated Property Value (€)"
                    placeholder="400000"
                    min={0}
                    step={10000}
                    thousandSeparator=","
                    required
                    radius={0}
                    {...form.getInputProps('mortgage_requirements.estimated_property_value')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Desired Mortgage Amount (€)"
                    placeholder="320000"
                    min={0}
                    step={10000}
                    thousandSeparator=","
                    required
                    radius={0}
                    {...form.getInputProps('mortgage_requirements.desired_mortgage_amount')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Preferred Mortgage Term (Years)"
                    placeholder="30"
                    min={5}
                    max={35}
                    radius={0}
                    {...form.getInputProps('mortgage_requirements.preferred_mortgage_term')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Interest Rate Preference"
                    data={[
                      { value: 'fixed', label: 'Fixed Rate' },
                      { value: 'variable', label: 'Variable Rate' },
                      { value: 'flexible', label: 'Flexible Rate' },
                    ]}
                    radius={0}
                    {...form.getInputProps('mortgage_requirements.interest_rate_preference')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <NumberInput
                    label="Down Payment Amount (€)"
                    placeholder="80000"
                    min={0}
                    step={5000}
                    thousandSeparator=","
                    radius={0}
                    {...form.getInputProps('mortgage_requirements.down_payment_amount')}
                  />
                </Grid.Col>
                <Grid.Col span={12}>
                  <Checkbox
                    label="NHG (National Mortgage Guarantee) Required"
                    description="Lower interest rates but property value limits apply"
                    {...form.getInputProps('mortgage_requirements.nhg_required', { type: 'checkbox' })}
                  />
                </Grid.Col>
              </Grid>
            </Card>
          )}

          {/* Step 4: AFM Suitability Assessment */}
          {activeStep === 3 && (
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">AFM Suitability Assessment</Title>
              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Mortgage Experience"
                    data={[
                      { value: 'first_time', label: 'First-time buyer' },
                      { value: 'experienced', label: 'Experienced (2-3 mortgages)' },
                      { value: 'very_experienced', label: 'Very experienced (4+ mortgages)' },
                    ]}
                    radius={0}
                    {...form.getInputProps('afm_suitability.mortgage_experience')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Financial Knowledge Level"
                    data={[
                      { value: 'basic', label: 'Basic' },
                      { value: 'intermediate', label: 'Intermediate' },
                      { value: 'advanced', label: 'Advanced' },
                    ]}
                    radius={0}
                    {...form.getInputProps('afm_suitability.financial_knowledge_level')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Risk Tolerance"
                    data={[
                      { value: 'conservative', label: 'Conservative' },
                      { value: 'moderate', label: 'Moderate' },
                      { value: 'aggressive', label: 'Aggressive' },
                    ]}
                    radius={0}
                    {...form.getInputProps('afm_suitability.risk_tolerance')}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Select
                    label="Sustainability Preferences"
                    data={[
                      { value: 'not_important', label: 'Not Important' },
                      { value: 'somewhat_important', label: 'Somewhat Important' },
                      { value: 'very_important', label: 'Very Important' },
                    ]}
                    radius={0}
                    {...form.getInputProps('afm_suitability.sustainability_preferences')}
                  />
                </Grid.Col>
                <Grid.Col span={12}>
                  <Select
                    label="Expected Advice Frequency"
                    data={[
                      { value: 'one_time', label: 'One-time consultation' },
                      { value: 'regular', label: 'Regular check-ins' },
                      { value: 'ongoing', label: 'Ongoing relationship' },
                    ]}
                    radius={0}
                    {...form.getInputProps('afm_suitability.expected_advice_frequency')}
                  />
                </Grid.Col>
              </Grid>
            </Card>
          )}

          {/* Step 5: Review & Submit */}
          {activeStep === 4 && (
            <Card radius={0} shadow="sm" padding="lg">
              <Title order={3} mb="md">Review Your Information</Title>
              
              <Grid mb="xl">
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Card radius={0} withBorder padding="md">
                    <Group justify="space-between" mb="sm">
                      <Text fw={500}>AFM Compliance Score</Text>
                      <Badge color={complianceScore >= 80 ? 'green' : complianceScore >= 60 ? 'yellow' : 'red'} size="lg" radius={0}>
                        {complianceScore}%
                      </Badge>
                    </Group>
                    <Progress value={complianceScore} color={complianceScore >= 80 ? 'green' : complianceScore >= 60 ? 'yellow' : 'red'} radius={0} />
                  </Card>
                </Grid.Col>
                
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Card radius={0} withBorder padding="md">
                    <Group justify="space-between" mb="sm">
                      <Text fw={500}>Risk Profile</Text>
                      <Badge color={riskProfile === 'low' ? 'green' : riskProfile === 'medium' ? 'yellow' : 'red'} size="lg" radius={0}>
                        {riskProfile.toUpperCase()}
                      </Badge>
                    </Group>
                    <Text size="sm" c="dimmed">Based on financial assessment</Text>
                  </Card>
                </Grid.Col>
              </Grid>

              <Alert color="blue" icon={<IconInfoCircle size={16} />} radius={0}>
                <Text size="sm">
                  By submitting this client intake form, you confirm that all information provided is accurate and complete. 
                  This information will be used for AFM compliance assessment and mortgage suitability analysis.
                </Text>
              </Alert>
            </Card>
          )}

          {/* Navigation Buttons */}
          <Group justify="space-between">
            <Group>
              {activeStep > 0 && (
                <Button variant="outline" leftSection={<IconArrowLeft size={16} />} onClick={prevStep} radius={0}>
                  Previous
                </Button>
              )}
            </Group>
            
            <Group>
              <Button 
                variant="outline" 
                leftSection={<IconDeviceFloppy size={16} />} 
                onClick={saveDraft}
                loading={saving}
                radius={0}
              >
                Save Draft
              </Button>
              
              {activeStep < 4 ? (
                <Button leftSection={<IconArrowRight size={16} />} onClick={nextStep} radius={0}>
                  Next
                </Button>
              ) : (
                <Button 
                  leftSection={<IconSend size={16} />} 
                  onClick={submitProfile}
                  loading={loading}
                  color="green"
                  radius={0}
                >
                  Submit for Review
                </Button>
              )}
            </Group>
          </Group>
        </form>
      </Stack>
    </Container>
  );
};

export default AFMClientIntake;
