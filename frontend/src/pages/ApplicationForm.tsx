/**
 * Mortgage Application Form
 *
 * Professional mortgage application form with:
 * - Comprehensive field validation
 * - Real-time feedback
 * - Professional UI design
 * - Integration with backend APIs
 * - Progress tracking
 */

import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Box,
  Stepper,
  Step,
  StepLabel,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Divider
} from '@mui/material';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import {
  Person,
  Home,
  AttachMoney,
  Work,
  Send
} from '@mui/icons-material';

// Form steps
const steps = ['Personal Information', 'Property Details', 'Financial Information', 'Review & Submit'];

interface FormData {
  // Personal Information
  firstName: string;
  lastName: string;
  dateOfBirth: string;
  email: string;
  phone: string;
  address: string;

  // Property Details
  propertyValue: number;
  mortgageAmount: number;
  loanTerm: number;
  interestPreference: string;

  // Financial Information
  annualIncome: number;
  employerName: string;
  employmentStatus: string;
}

const validationSchemas = [
  // Step 1: Personal Information
  Yup.object({
    firstName: Yup.string().required('First name is required').min(2, 'Too short'),
    lastName: Yup.string().required('Last name is required').min(2, 'Too short'),
    dateOfBirth: Yup.date()
      .required('Date of birth is required')
      .max(new Date(Date.now() - 18 * 365 * 24 * 60 * 60 * 1000), 'Must be at least 18 years old'),
    email: Yup.string().email('Invalid email').required('Email is required'),
    phone: Yup.string().required('Phone number is required'),
    address: Yup.string().required('Address is required').min(10, 'Please provide complete address'),
  }),

  // Step 2: Property Details
  Yup.object({
    propertyValue: Yup.number()
      .required('Property value is required')
      .min(50000, 'Property value must be at least €50,000')
      .max(10000000, 'Property value seems too high'),
    mortgageAmount: Yup.number()
      .required('Mortgage amount is required')
      .min(25000, 'Mortgage amount must be at least €25,000')
      .max(2000000, 'Mortgage amount seems too high'),
    loanTerm: Yup.number()
      .required('Loan term is required')
      .min(5, 'Minimum term is 5 years')
      .max(40, 'Maximum term is 40 years'),
    interestPreference: Yup.string().required('Interest preference is required'),
  }),

  // Step 3: Financial Information
  Yup.object({
    annualIncome: Yup.number()
      .required('Annual income is required')
      .min(15000, 'Annual income must be at least €15,000'),
    employerName: Yup.string().required('Employer name is required'),
    employmentStatus: Yup.string().required('Employment status is required'),
  }),

  // Step 4: Review (no additional validation)
  Yup.object({})
];

const initialValues: FormData = {
  firstName: '',
  lastName: '',
  dateOfBirth: '',
  email: '',
  phone: '',
  address: '',
  propertyValue: 0,
  mortgageAmount: 0,
  loanTerm: 25,
  interestPreference: '',
  annualIncome: 0,
  employerName: '',
  employmentStatus: '',
};

const ApplicationForm: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [activeStep, setActiveStep] = useState(0);
  const [completed, setCompleted] = useState<{ [k: number]: boolean }>({});

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleStepClick = (step: number) => {
    setActiveStep(step);
  };

  const handleSubmit = async (values: FormData) => {
    try {
      // TODO: Submit to backend API
      console.log('Submitting application:', values);

      enqueueSnackbar('Application submitted successfully!', { variant: 'success' });
      navigate('/results');
    } catch (error) {
      enqueueSnackbar('Failed to submit application', { variant: 'error' });
    }
  };

  const isStepCompleted = (step: number) => completed[step];

  const renderStepContent = (step: number, formik: any) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Person />
                Personal Information
              </Typography>
              <Divider sx={{ mb: 2 }} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="firstName"
                label="First Name"
                variant="outlined"
                error={formik.touched.firstName && Boolean(formik.errors.firstName)}
                helperText={formik.touched.firstName && formik.errors.firstName}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="lastName"
                label="Last Name"
                variant="outlined"
                error={formik.touched.lastName && Boolean(formik.errors.lastName)}
                helperText={formik.touched.lastName && formik.errors.lastName}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="dateOfBirth"
                label="Date of Birth"
                type="date"
                InputLabelProps={{ shrink: true }}
                variant="outlined"
                error={formik.touched.dateOfBirth && Boolean(formik.errors.dateOfBirth)}
                helperText={formik.touched.dateOfBirth && formik.errors.dateOfBirth}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="email"
                label="Email Address"
                type="email"
                variant="outlined"
                error={formik.touched.email && Boolean(formik.errors.email)}
                helperText={formik.touched.email && formik.errors.email}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="phone"
                label="Phone Number"
                variant="outlined"
                error={formik.touched.phone && Boolean(formik.errors.phone)}
                helperText={formik.touched.phone && formik.errors.phone}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              {/* Placeholder for address autocomplete */}
            </Grid>
            <Grid item xs={12}>
              <Field
                as={TextField}
                fullWidth
                name="address"
                label="Home Address"
                multiline
                rows={3}
                variant="outlined"
                error={formik.touched.address && Boolean(formik.errors.address)}
                helperText={formik.touched.address && formik.errors.address}
              />
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Home />
                Property Details
              </Typography>
              <Divider sx={{ mb: 2 }} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="propertyValue"
                label="Property Value (€)"
                type="number"
                variant="outlined"
                error={formik.touched.propertyValue && Boolean(formik.errors.propertyValue)}
                helperText={formik.touched.propertyValue && formik.errors.propertyValue}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="mortgageAmount"
                label="Mortgage Amount (€)"
                type="number"
                variant="outlined"
                error={formik.touched.mortgageAmount && Boolean(formik.errors.mortgageAmount)}
                helperText={formik.touched.mortgageAmount && formik.errors.mortgageAmount}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="loanTerm"
                label="Loan Term (Years)"
                type="number"
                variant="outlined"
                error={formik.touched.loanTerm && Boolean(formik.errors.loanTerm)}
                helperText={formik.touched.loanTerm && formik.errors.loanTerm}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Interest Rate Preference</InputLabel>
                <Field
                  as={Select}
                  name="interestPreference"
                  label="Interest Rate Preference"
                  error={formik.touched.interestPreference && Boolean(formik.errors.interestPreference)}
                >
                  <MenuItem value="fixed">Fixed Rate</MenuItem>
                  <MenuItem value="variable">Variable Rate</MenuItem>
                  <MenuItem value="mixed">Mixed/Other</MenuItem>
                </Field>
                {formik.touched.interestPreference && formik.errors.interestPreference && (
                  <FormHelperText error>{formik.errors.interestPreference}</FormHelperText>
                )}
              </FormControl>
            </Grid>
            {/* LTV Ratio Display */}
            <Grid item xs={12}>
              <Alert severity="info">
                Estimated Loan-to-Value Ratio: {formik.values.propertyValue > 0
                  ? `${((formik.values.mortgageAmount / formik.values.propertyValue) * 100).toFixed(1)}%`
                  : 'N/A'
                }
              </Alert>
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Work />
                Financial Information
              </Typography>
              <Divider sx={{ mb: 2 }} />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Field
                as={TextField}
                fullWidth
                name="annualIncome"
                label="Annual Income (€)"
                type="number"
                variant="outlined"
                error={formik.touched.annualIncome && Boolean(formik.errors.annualIncome)}
                helperText={formik.touched.annualIncome && formik.errors.annualIncome}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Employment Status</InputLabel>
                <Field
                  as={Select}
                  name="employmentStatus"
                  label="Employment Status"
                  error={formik.touched.employmentStatus && Boolean(formik.errors.employmentStatus)}
                >
                  <MenuItem value="employed">Employed</MenuItem>
                  <MenuItem value="self-employed">Self-Employed</MenuItem>
                  <MenuItem value="contractor">Contractor</MenuItem>
                  <MenuItem value="retired">Retired</MenuItem>
                  <MenuItem value="other">Other</MenuItem>
                </Field>
                {formik.touched.employmentStatus && formik.errors.employmentStatus && (
                  <FormHelperText error>{formik.errors.employmentStatus}</FormHelperText>
                )}
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Field
                as={TextField}
                fullWidth
                name="employerName"
                label="Employer Name"
                variant="outlined"
                error={formik.touched.employerName && Boolean(formik.errors.employerName)}
                helperText={formik.touched.employerName && formik.errors.employerName}
              />
            </Grid>
            {/* DTI Ratio Display */}
            <Grid item xs={12}>
              <Alert severity="info">
                Estimated Debt-to-Income Ratio: {formik.values.annualIncome > 0
                  ? `${((formik.values.mortgageAmount / formik.values.annualIncome) * 100).toFixed(1)}%`
                  : 'N/A'
                }
              </Alert>
            </Grid>
          </Grid>
        );

      case 3:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Review Your Application
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Alert severity="info" sx={{ mb: 3 }}>
                Please review your information carefully before submitting. Once submitted, the application will be processed by our AI agents for compliance and quality control.
              </Alert>
            </Grid>

            {/* Summary Cards */}
            <Grid item xs={12} md={4}>
              <Paper elevation={2} sx={{ p: 2, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                  Personal Details
                </Typography>
                <Typography variant="body2">
                  {formik.values.firstName} {formik.values.lastName}<br/>
                  Born: {formik.values.dateOfBirth}<br/>
                  Email: {formik.values.email}
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper elevation={2} sx={{ p: 2, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                  Property & Loan
                </Typography>
                <Typography variant="body2">
                  Property: €{formik.values.propertyValue.toLocaleString()}<br/>
                  Mortgage: €{formik.values.mortgageAmount.toLocaleString()}<br/>
                  Term: {formik.values.loanTerm} years<br/>
                  Rate: {formik.values.interestPreference}
                </Typography>
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper elevation={2} sx={{ p: 2, borderRadius: 2 }}>
                <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                  Financial Details
                </Typography>
                <Typography variant="body2">
                  Income: €{formik.values.annualIncome.toLocaleString()}<br/>
                  Employer: {formik.values.employerName}<br/>
                  Status: {formik.values.employmentStatus}
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        );

      default:
        return null;
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ fontWeight: 'bold', mb: 3 }}>
          Mortgage Application
        </Typography>

        {/* Stepper */}
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label, index) => (
            <Step key={label} completed={isStepCompleted(index)}>
              <StepLabel
                sx={{
                  cursor: index <= activeStep ? 'pointer' : 'default',
                  '& .MuiStepLabel-label': {
                    fontSize: { xs: '0.75rem', sm: '0.875rem' }
                  }
                }}
                onClick={() => index <= activeStep && handleStepClick(index)}
              >
                {label}
              </StepLabel>
            </Step>
          ))}
        </Stepper>

        {/* Form */}
        <Formik
          initialValues={initialValues}
          validationSchema={validationSchemas[activeStep]}
          onSubmit={handleSubmit}
        >
          {(formik) => (
            <Form>
              {renderStepContent(activeStep, formik)}

              {/* Navigation Buttons */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
                <Button
                  disabled={activeStep === 0}
                  onClick={handleBack}
                  variant="outlined"
                  sx={{ borderRadius: 2 }}
                >
                  Back
                </Button>

                <Box sx={{ display: 'flex', gap: 2 }}>
                  {activeStep === steps.length - 1 ? (
                    <Button
                      type="submit"
                      variant="contained"
                      startIcon={<Send />}
                      disabled={!formik.isValid || formik.isSubmitting}
                      sx={{ borderRadius: 2, px: 4 }}
                    >
                      {formik.isSubmitting ? 'Submitting...' : 'Submit Application'}
                    </Button>
                  ) : (
                    <Button
                      variant="contained"
                      onClick={handleNext}
                      disabled={!formik.isValid}
                      sx={{ borderRadius: 2, px: 4 }}
                    >
                      Next
                    </Button>
                  )}
                </Box>
              </Box>
            </Form>
          )}
        </Formik>
      </Paper>
    </Container>
  );
};

export default ApplicationForm;
