/**
 * Dutch Mortgage Application - Full Mantine Implementation
 */

import React, { useState } from 'react';
import {
  Container,
  Card,
  TextInput,
  Button,
  Title,
  Group,
  Stack,
  Grid,
  Stepper,
  NumberInput,
  Select,
  Checkbox,
  Textarea,
  ThemeIcon,
  Text,
  Badge,
  Alert,
  Divider,
} from '@mantine/core';
import {
  IconHome,
  IconCurrencyEuro,
  IconFileText,
  IconCheck,
  IconInfoCircle,
  IconCalculator,
} from '@tabler/icons-react';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';

const DutchMortgageApplication: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [calculationResults, setCalculationResults] = useState<any>(null);

  const form = useForm({
    initialValues: {
      firstName: '',
      lastName: '',
      bsn: '',
      dateOfBirth: '',
      nationality: '',
      maritalStatus: '',
      hasPartner: false,
      partnerFirstName: '',
      partnerLastName: '',
      partnerBsn: '',
      partnerDateOfBirth: '',
      employmentType: '',
      employer: '',
      grossAnnualIncome: 0,
      netMonthlyIncome: 0,
      contractType: '',
      partnerEmploymentType: '',
      partnerGrossAnnualIncome: 0,
      propertyAddress: '',
      propertyPostalCode: '',
      propertyCity: '',
      propertyType: '',
      constructionYear: 0,
      livingArea: 0,
      plotSize: 0,
      energyLabel: '',
      purchasePrice: 0,
      appraisalValue: 0,
      loanAmount: 0,
      loanPurpose: '',
      preferredLender: '',
      interestRateType: '',
      loanTerm: 30,
      monthlyPaymentBudget: 0,
      hasNHG: false,
      hasOtherLoans: false,
      otherLoansAmount: 0,
      additionalIncome: 0,
      specialCircumstances: '',
    },
    validate: (values) => {
      const errors: any = {};
      
      if (activeStep === 0) {
        if (!values.firstName) errors.firstName = 'First name is required';
        if (!values.lastName) errors.lastName = 'Last name is required';
        if (!values.bsn) errors.bsn = 'BSN is required';
        if (!values.dateOfBirth) errors.dateOfBirth = 'Date of birth is required';
      }
      
      if (activeStep === 1) {
        if (!values.employmentType) errors.employmentType = 'Employment type is required';
        if (!values.grossAnnualIncome) errors.grossAnnualIncome = 'Annual income is required';
      }
      
      if (activeStep === 2) {
        if (!values.propertyAddress) errors.propertyAddress = 'Property address is required';
        if (!values.purchasePrice) errors.purchasePrice = 'Purchase price is required';
      }
      
      if (activeStep === 3) {
        if (!values.loanAmount) errors.loanAmount = 'Loan amount is required';
        if (!values.preferredLender) errors.preferredLender = 'Preferred lender is required';
      }
      
      return errors;
    },
  });

  const nextStep = () => {
    const validation = form.validate();
    if (!validation.hasErrors) {
      if (activeStep === 3) {
        calculateMortgage();
      }
      setActiveStep((current) => (current < 4 ? current + 1 : current));
    }
  };

  const prevStep = () => setActiveStep((current) => (current > 0 ? current - 1 : current));

  const calculateMortgage = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const totalIncome = form.values.grossAnnualIncome + 
        (form.values.hasPartner ? form.values.partnerGrossAnnualIncome : 0);
      const maxLoanAmount = Math.min(totalIncome * 4.5, form.values.purchasePrice * 1.06);
      const monthlyPayment = (form.values.loanAmount * 0.04) / 12;
      const ltvRatio = (form.values.loanAmount / form.values.purchasePrice) * 100;
      
      setCalculationResults({
        maxLoanAmount,
        requestedAmount: form.values.loanAmount,
        monthlyPayment,
        ltvRatio,
        totalIncome,
        affordabilityRatio: (monthlyPayment * 12) / totalIncome,
        nhgEligible: form.values.purchasePrice <= 435000,
        estimatedInterestRate: 3.8,
      });

      notifications.show({
        title: 'Calculation Complete',
        message: 'Mortgage calculation has been completed',
        color: 'green',
        icon: <IconCheck size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Calculation Failed',
        message: 'Failed to calculate mortgage details',
        color: 'red',
      });
    } finally {
      setLoading(false);
    }
  };

  const submitApplication = async () => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      notifications.show({
        title: 'Application Submitted',
        message: 'Your mortgage application has been submitted successfully',
        color: 'green',
        icon: <IconCheck size={16} />,
      });
    } catch (error) {
      notifications.show({
        title: 'Submission Failed',
        message: 'Failed to submit mortgage application',
        color: 'red',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Group>
          <ThemeIcon size="xl" radius={0} color="orange">
            <IconHome size={32} />
          </ThemeIcon>
          <div>
            <Title order={1}>Dutch Mortgage Application</Title>
            <Text c="dimmed">Complete mortgage application for Dutch properties</Text>
          </div>
        </Group>

        <Card radius={0} shadow="sm" padding="lg">
          <Stepper active={activeStep} radius={0}>
            <Stepper.Step label="Personal Info" description="Basic information">
              <Stack gap="md" mt="xl">
                <Title order={3}>Personal Information</Title>
                <Grid>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <TextInput
                      label="First Name"
                      placeholder="John"
                      {...form.getInputProps('firstName')}
                      radius={0}
                      required
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <TextInput
                      label="Last Name"
                      placeholder="Doe"
                      {...form.getInputProps('lastName')}
                      radius={0}
                      required
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <TextInput
                      label="BSN (Dutch Social Security Number)"
                      placeholder="123456789"
                      {...form.getInputProps('bsn')}
                      radius={0}
                      required
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <TextInput
                      label="Date of Birth"
                      placeholder="1990-01-01"
                      type="date"
                      {...form.getInputProps('dateOfBirth')}
                      radius={0}
                      required
                    />
                  </Grid.Col>
                </Grid>
                
                <Checkbox
                  label="I have a partner applying jointly"
                  {...form.getInputProps('hasPartner', { type: 'checkbox' })}
                  radius={0}
                />
                
                {form.values.hasPartner && (
                  <>
                    <Divider label="Partner Information" />
                    <Grid>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <TextInput
                          label="Partner's First Name"
                          placeholder="Jane"
                          {...form.getInputProps('partnerFirstName')}
                          radius={0}
                        />
                      </Grid.Col>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <TextInput
                          label="Partner's Last Name"
                          placeholder="Doe"
                          {...form.getInputProps('partnerLastName')}
                          radius={0}
                        />
                      </Grid.Col>
                    </Grid>
                  </>
                )}
              </Stack>
            </Stepper.Step>

            <Stepper.Step label="Employment" description="Income details">
              <Stack gap="md" mt="xl">
                <Title order={3}>Employment & Income</Title>
                <Grid>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <Select
                      label="Employment Type"
                      placeholder="Select employment type"
                      data={[
                        { value: 'employee', label: 'Employee' },
                        { value: 'self-employed', label: 'Self-employed' },
                        { value: 'freelancer', label: 'Freelancer' },
                        { value: 'retired', label: 'Retired' },
                      ]}
                      {...form.getInputProps('employmentType')}
                      radius={0}
                      required
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <NumberInput
                      label="Gross Annual Income"
                      placeholder="50000"
                      {...form.getInputProps('grossAnnualIncome')}
                      leftSection={<IconCurrencyEuro size={16} />}
                      thousandSeparator=","
                      radius={0}
                      required
                    />
                  </Grid.Col>
                </Grid>
              </Stack>
            </Stepper.Step>

            <Stepper.Step label="Property" description="Property details">
              <Stack gap="md" mt="xl">
                <Title order={3}>Property Information</Title>
                <Grid>
                  <Grid.Col span={12}>
                    <TextInput
                      label="Property Address"
                      placeholder="Damrak 1"
                      {...form.getInputProps('propertyAddress')}
                      radius={0}
                      required
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <NumberInput
                      label="Purchase Price"
                      placeholder="400000"
                      {...form.getInputProps('purchasePrice')}
                      leftSection={<IconCurrencyEuro size={16} />}
                      thousandSeparator=","
                      radius={0}
                      required
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <Select
                      label="Property Type"
                      placeholder="Select property type"
                      data={[
                        { value: 'apartment', label: 'Apartment' },
                        { value: 'house', label: 'House' },
                        { value: 'townhouse', label: 'Townhouse' },
                        { value: 'villa', label: 'Villa' },
                      ]}
                      {...form.getInputProps('propertyType')}
                      radius={0}
                    />
                  </Grid.Col>
                </Grid>
              </Stack>
            </Stepper.Step>

            <Stepper.Step label="Mortgage" description="Loan requirements">
              <Stack gap="md" mt="xl">
                <Title order={3}>Mortgage Requirements</Title>
                <Grid>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <NumberInput
                      label="Requested Loan Amount"
                      placeholder="350000"
                      {...form.getInputProps('loanAmount')}
                      leftSection={<IconCurrencyEuro size={16} />}
                      thousandSeparator=","
                      radius={0}
                      required
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <Select
                      label="Preferred Lender"
                      placeholder="Select lender"
                      data={[
                        { value: 'ing', label: 'ING Bank' },
                        { value: 'abn', label: 'ABN AMRO' },
                        { value: 'rabobank', label: 'Rabobank' },
                        { value: 'sns', label: 'SNS Bank' },
                        { value: 'no-preference', label: 'No Preference' },
                      ]}
                      {...form.getInputProps('preferredLender')}
                      radius={0}
                      required
                    />
                  </Grid.Col>
                </Grid>
                
                <Checkbox
                  label="I want to apply for NHG (National Mortgage Guarantee)"
                  {...form.getInputProps('hasNHG', { type: 'checkbox' })}
                  radius={0}
                />
              </Stack>
            </Stepper.Step>

            <Stepper.Step label="Review" description="Final review">
              <Stack gap="md" mt="xl">
                <Title order={3}>Application Review</Title>
                
                {calculationResults && (
                  <Card radius={0} withBorder padding="md">
                    <Title order={4} mb="md">Mortgage Calculation Results</Title>
                    <Grid>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <Text size="sm" c="dimmed">Maximum Loan Amount</Text>
                        <Text fw={600} size="lg">€{calculationResults.maxLoanAmount.toLocaleString()}</Text>
                      </Grid.Col>
                      <Grid.Col span={{ base: 12, md: 6 }}>
                        <Text size="sm" c="dimmed">Estimated Monthly Payment</Text>
                        <Text fw={600} size="lg">€{Math.round(calculationResults.monthlyPayment).toLocaleString()}</Text>
                      </Grid.Col>
                    </Grid>
                    
                    <Divider my="md" />
                    
                    <Group justify="space-between">
                      <Text size="sm">NHG Eligible:</Text>
                      <Badge color={calculationResults.nhgEligible ? 'green' : 'red'} radius={0}>
                        {calculationResults.nhgEligible ? 'Yes' : 'No'}
                      </Badge>
                    </Group>
                  </Card>
                )}
                
                <Alert color="blue" icon={<IconInfoCircle size={16} />} radius={0}>
                  <Text size="sm">
                    Please review all information carefully before submitting your application.
                  </Text>
                </Alert>
                
                <Button
                  fullWidth
                  size="lg"
                  leftSection={<IconFileText size={20} />}
                  onClick={submitApplication}
                  loading={loading}
                  radius={0}
                >
                  Submit Mortgage Application
                </Button>
              </Stack>
            </Stepper.Step>
          </Stepper>

          <Group justify="space-between" mt="xl">
            <Button variant="default" onClick={prevStep} disabled={activeStep === 0} radius={0}>
              Back
            </Button>
            <Button 
              onClick={nextStep} 
              disabled={activeStep === 4}
              loading={loading && activeStep === 3}
              leftSection={activeStep === 3 ? <IconCalculator size={16} /> : undefined}
              radius={0}
            >
              {activeStep === 3 ? 'Calculate & Continue' : 'Next Step'}
            </Button>
          </Group>
        </Card>
      </Stack>
    </Container>
  );
};

export default DutchMortgageApplication;
