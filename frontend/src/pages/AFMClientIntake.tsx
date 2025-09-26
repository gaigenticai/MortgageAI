/**
 * AFM-Compliant Client Intake Form
 *
 * Implements Dutch AFM requirements for mortgage advice client assessment.
 * Includes mandatory suitability questionnaire and risk profiling.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  Container,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Box,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  Radio,
  RadioGroup,
  FormLabel,
  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
  Divider,
  Paper,
} from '@mui/material';
import {
  Person,
  Work,
  Home,
  Assessment,
  CheckCircle,
  Error as ErrorIcon,
  Info as InfoIcon,
  ArrowBack,
  ArrowForward,
  Save as SaveIcon,
  Send as SendIcon,
  Calculate as CalculateIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import {
  clientIntakeApi,
  ClientProfile,
  PersonalInfo,
  EmploymentInfo,
  FinancialSituation,
  MortgageRequirements,
  AFMSuitabilityAssessment,
  ValidationResult,
  IntakeProgress,
} from '../services/clientIntakeApi';

const AFMClientIntake: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const [activeStep, setActiveStep] = useState(0);
  const [clientProfile, setClientProfile] = useState<ClientProfile>({
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
      other_income_sources: [],
    },
    financial_situation: {
      existing_debts: [],
      monthly_expenses: 0,
      savings_amount: 0,
      investments: [],
      other_properties: false,
    },
    mortgage_requirements: {
      property_type: 'house',
      property_location: '',
      estimated_property_value: 0,
      desired_mortgage_amount: 0,
      preferred_mortgage_term: 30,
      interest_rate_preference: 'fixed',
      nhg_required: false,
    },
    afm_suitability: {
      mortgage_experience: 'first_time',
      financial_knowledge_level: 'basic',
      risk_tolerance: 'moderate',
      investment_objectives: [],
      sustainability_preferences: 'not_important',
      advice_needs: [],
      expected_advice_frequency: 'one_time',
    },
    status: 'draft',
  });

  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [intakeProgress, setIntakeProgress] = useState<IntakeProgress | null>(null);
  const [saving, setSaving] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [bsnValidating, setBsnValidating] = useState(false);
  const [dtiCalculating, setDtiCalculating] = useState(false);
  const [dtiResult, setDtiResult] = useState<{ dti_ratio: number; max_mortgage_amount: number } | null>(null);

  // Initialize API service with snackbar
  useEffect(() => {
    clientIntakeApi.setSnackbar(enqueueSnackbar);
  }, [enqueueSnackbar]);

  // Load intake progress
  const loadIntakeProgress = useCallback(async () => {
    if (clientProfile.id) {
      try {
        const progress = await clientIntakeApi.getIntakeProgress(clientProfile.id);
        setIntakeProgress(progress);
      } catch (error) {
        console.error('Failed to load intake progress:', error);
      }
    }
  }, [clientProfile.id]);

  // Validate current profile
  const validateCurrentProfile = useCallback(async () => {
    try {
      const result = await clientIntakeApi.validateClientProfile(clientProfile);
      setValidationResult(result);
      return result;
    } catch (error) {
      console.error('Validation failed:', error);
      return null;
    }
  }, [clientProfile]);

  // Update intake progress when profile changes
  useEffect(() => {
    loadIntakeProgress();
  }, [loadIntakeProgress]);

  // Steps configuration
  const steps = [
    {
      label: 'Personal Information',
      icon: Person,
      description: 'Basic personal and contact details',
      completed: intakeProgress?.completed_sections.includes('personal_info') || false,
    },
    {
      label: 'Employment & Income',
      icon: Work,
      description: 'Employment status and income verification',
      completed: intakeProgress?.completed_sections.includes('employment_info') || false,
    },
    {
      label: 'Financial Situation',
      icon: Assessment,
      description: 'Current financial position and obligations',
      completed: intakeProgress?.completed_sections.includes('financial_situation') || false,
    },
    {
      label: 'Mortgage Requirements',
      icon: Home,
      description: 'Property and mortgage preferences',
      completed: intakeProgress?.completed_sections.includes('mortgage_requirements') || false,
    },
    {
      label: 'AFM Suitability Assessment',
      icon: CheckCircle,
      description: 'Regulatory compliance questionnaire',
      completed: intakeProgress?.completed_sections.includes('afm_suitability') || false,
    },
  ];

  // Update client profile
  const updateClientProfile = useCallback((section: keyof ClientProfile, field: string, value: any) => {
    setClientProfile(prev => ({
      ...prev,
      [section]: {
        ...(prev[section] as any),
        [field]: value
      }
    }));
  }, []);

  // Handle next step
  const handleNext = async () => {
    const validation = await validateCurrentProfile();
    if (validation && !validation.isValid && Object.keys(validation.errors).length > 0) {
      enqueueSnackbar('Please correct the errors before proceeding', { variant: 'warning' });
      return;
    }

    if (activeStep < steps.length - 1) {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  // Handle back step
  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  // Save draft
  const handleSaveDraft = async () => {
    setSaving(true);
    try {
      let profile: ClientProfile;
      if (clientProfile.id) {
        profile = await clientIntakeApi.updateClientProfile(clientProfile.id, clientProfile);
      } else {
        profile = await clientIntakeApi.createClientProfile(clientProfile);
      }
      setClientProfile(profile);
      enqueueSnackbar('Draft saved successfully', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar('Failed to save draft', { variant: 'error' });
    } finally {
      setSaving(false);
    }
  };

  // Submit for AFM review
  const handleSubmit = async () => {
    const validation = await validateCurrentProfile();
    if (!validation?.isValid) {
      enqueueSnackbar('Please complete all required fields before submission', { variant: 'error' });
      return;
    }

    setSubmitting(true);
    try {
      if (!clientProfile.id) {
        const profile = await clientIntakeApi.createClientProfile(clientProfile);
        setClientProfile(profile);
        const result = await clientIntakeApi.submitClientProfile(profile.id!);
        navigate('/afm-compliance-advisor', {
          state: {
            clientId: profile.id,
            complianceScore: result.compliance_score,
            reviewId: result.review_id
          }
        });
      } else {
        const result = await clientIntakeApi.submitClientProfile(clientProfile.id);
        navigate('/afm-compliance-advisor', {
          state: {
            clientId: clientProfile.id,
            complianceScore: result.compliance_score,
            reviewId: result.review_id
          }
        });
      }
    } catch (error) {
      enqueueSnackbar('Failed to submit for AFM review', { variant: 'error' });
    } finally {
      setSubmitting(false);
    }
  };

  // Validate BSN
  const validateBSN = async (bsn: string) => {
    if (bsn.length === 9) {
      setBsnValidating(true);
      try {
        const result = await clientIntakeApi.validateBSN(bsn);
        if (!result.isValid) {
          enqueueSnackbar(result.message || 'Invalid BSN', { variant: 'error' });
        }
      } catch (error) {
        enqueueSnackbar('Failed to validate BSN', { variant: 'error' });
      } finally {
        setBsnValidating(false);
      }
    }
  };

  // Calculate DTI ratio
  const calculateDTI = async () => {
    setDtiCalculating(true);
    try {
      const result = await clientIntakeApi.calculateDTI(clientProfile);
      setDtiResult(result);
    } catch (error) {
      enqueueSnackbar('Failed to calculate DTI ratio', { variant: 'error' });
    } finally {
      setDtiCalculating(false);
    }
  };

  // Render personal information step
  const renderPersonalInformation = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          label="Full Name"
          value={clientProfile.personal_info.full_name}
          onChange={(e) => updateClientProfile('personal_info', 'full_name', e.target.value)}
          required
          error={validationResult?.errors.full_name ? true : false}
          helperText={validationResult?.errors.full_name}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          label="BSN (Dutch Social Security Number)"
          value={clientProfile.personal_info.bsn}
          onChange={(e) => {
            updateClientProfile('personal_info', 'bsn', e.target.value);
            if (e.target.value.length === 9) validateBSN(e.target.value);
          }}
          required
          inputProps={{ pattern: '[0-9]{9}', maxLength: 9 }}
          helperText="9-digit Dutch social security number"
          error={validationResult?.errors.bsn ? true : false}
          InputProps={{
            endAdornment: bsnValidating ? <CircularProgress size={20} /> : null,
          }}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="date"
          label="Date of Birth"
          value={clientProfile.personal_info.date_of_birth}
          onChange={(e) => updateClientProfile('personal_info', 'date_of_birth', e.target.value)}
          InputLabelProps={{ shrink: true }}
          required
          error={validationResult?.errors.date_of_birth ? true : false}
          helperText={validationResult?.errors.date_of_birth}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <FormControl fullWidth required error={validationResult?.errors.marital_status ? true : false}>
          <InputLabel>Marital Status</InputLabel>
          <Select
            value={clientProfile.personal_info.marital_status}
            onChange={(e) => updateClientProfile('personal_info', 'marital_status', e.target.value)}
          >
            <MenuItem value="single">Single</MenuItem>
            <MenuItem value="married">Married</MenuItem>
            <MenuItem value="registered_partnership">Registered Partnership</MenuItem>
            <MenuItem value="divorced">Divorced</MenuItem>
            <MenuItem value="widowed">Widowed</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Number of Dependents"
          value={clientProfile.personal_info.number_of_dependents}
          onChange={(e) => updateClientProfile('personal_info', 'number_of_dependents', parseInt(e.target.value))}
          inputProps={{ min: 0, max: 10 }}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="email"
          label="Email Address"
          value={clientProfile.personal_info.email}
          onChange={(e) => updateClientProfile('personal_info', 'email', e.target.value)}
          required
          error={validationResult?.errors.email ? true : false}
          helperText={validationResult?.errors.email}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="tel"
          label="Phone Number"
          value={clientProfile.personal_info.phone}
          onChange={(e) => updateClientProfile('personal_info', 'phone', e.target.value)}
          required
          error={validationResult?.errors.phone ? true : false}
          helperText={validationResult?.errors.phone}
        />
      </Grid>
    </Grid>
  );

  // Render employment information step
  const renderEmploymentInformation = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <FormControl fullWidth required>
          <InputLabel>Employment Status</InputLabel>
          <Select
            value={clientProfile.employment_info.employment_status}
            onChange={(e) => updateClientProfile('employment_info', 'employment_status', e.target.value)}
          >
            <MenuItem value="employed">Employed</MenuItem>
            <MenuItem value="self_employed">Self Employed</MenuItem>
            <MenuItem value="unemployed">Unemployed</MenuItem>
            <MenuItem value="retired">Retired</MenuItem>
            <MenuItem value="student">Student</MenuItem>
            <MenuItem value="other">Other</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          label="Employer Name"
          value={clientProfile.employment_info.employer_name}
          onChange={(e) => updateClientProfile('employment_info', 'employer_name', e.target.value)}
          required={clientProfile.employment_info.employment_status === 'employed'}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          label="Job Title"
          value={clientProfile.employment_info.job_title}
          onChange={(e) => updateClientProfile('employment_info', 'job_title', e.target.value)}
          required={clientProfile.employment_info.employment_status === 'employed'}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Employment Duration (Months)"
          value={clientProfile.employment_info.employment_duration_months}
          onChange={(e) => updateClientProfile('employment_info', 'employment_duration_months', parseInt(e.target.value))}
          inputProps={{ min: 0 }}
          required={clientProfile.employment_info.employment_status === 'employed'}
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Gross Annual Income (€)"
          value={clientProfile.employment_info.gross_annual_income}
          onChange={(e) => updateClientProfile('employment_info', 'gross_annual_income', parseFloat(e.target.value))}
          inputProps={{ min: 0 }}
          required
        />
      </Grid>
      <Grid item xs={12} md={6}>
        <TextField
          fullWidth
          type="number"
          label="Partner Income (€)"
          value={clientProfile.employment_info.partner_income}
          onChange={(e) => updateClientProfile('employment_info', 'partner_income', parseFloat(e.target.value))}
          inputProps={{ min: 0 }}
        />
      </Grid>
    </Grid>
  );

  // Render AFM suitability assessment step
  const renderAFMSuitabilityAssessment = () => (
    <Box>
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
          AFM Suitability Requirements
        </Typography>
        <Typography variant="body2">
          Under Dutch AFM regulations (Wft Article 86f), we must assess your financial situation,
          knowledge, experience, and objectives to provide suitable mortgage advice.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <FormControl component="fieldset" fullWidth>
            <FormLabel component="legend" sx={{ fontWeight: 600, mb: 2 }}>
              Previous Mortgage Experience
            </FormLabel>
            <RadioGroup
              value={clientProfile.afm_suitability.mortgage_experience}
              onChange={(e) => updateClientProfile('afm_suitability', 'mortgage_experience', e.target.value)}
            >
              <FormControlLabel
                value="first_time"
                control={<Radio />}
                label="First-time homebuyer (no previous mortgage experience)"
              />
              <FormControlLabel
                value="experienced"
                control={<Radio />}
                label="Experienced (had mortgage in past 5 years)"
              />
              <FormControlLabel
                value="very_experienced"
                control={<Radio />}
                label="Very experienced (multiple mortgages, investment properties)"
              />
            </RadioGroup>
          </FormControl>
        </Grid>

        <Grid item xs={12}>
          <FormControl component="fieldset" fullWidth>
            <FormLabel component="legend" sx={{ fontWeight: 600, mb: 2 }}>
              Financial Knowledge Level
            </FormLabel>
            <RadioGroup
              value={clientProfile.afm_suitability.financial_knowledge_level}
              onChange={(e) => updateClientProfile('afm_suitability', 'financial_knowledge_level', e.target.value)}
            >
              <FormControlLabel
                value="basic"
                control={<Radio />}
                label="Basic (understand simple financial concepts)"
              />
              <FormControlLabel
                value="intermediate"
                control={<Radio />}
                label="Intermediate (comfortable with interest rates, terms, basic investments)"
              />
              <FormControlLabel
                value="advanced"
                control={<Radio />}
                label="Advanced (experienced with complex financial products and risks)"
              />
            </RadioGroup>
          </FormControl>
        </Grid>

        <Grid item xs={12}>
          <FormControl component="fieldset" fullWidth>
            <FormLabel component="legend" sx={{ fontWeight: 600, mb: 2 }}>
              Risk Tolerance
            </FormLabel>
            <RadioGroup
              value={clientProfile.afm_suitability.risk_tolerance}
              onChange={(e) => updateClientProfile('afm_suitability', 'risk_tolerance', e.target.value)}
            >
              <FormControlLabel
                value="conservative"
                control={<Radio />}
                label="Conservative (prefer certainty, avoid payment fluctuations)"
              />
              <FormControlLabel
                value="moderate"
                control={<Radio />}
                label="Moderate (accept some risk for potential benefits)"
              />
              <FormControlLabel
                value="aggressive"
                control={<Radio />}
                label="Aggressive (comfortable with significant payment variations)"
              />
            </RadioGroup>
          </FormControl>
        </Grid>

        <Grid item xs={12}>
          <FormControl fullWidth>
            <InputLabel>Sustainability Preferences</InputLabel>
            <Select
              value={clientProfile.afm_suitability.sustainability_preferences}
              onChange={(e) => updateClientProfile('afm_suitability', 'sustainability_preferences', e.target.value)}
            >
              <MenuItem value="not_important">Not a priority</MenuItem>
              <MenuItem value="somewhat_important">Somewhat important</MenuItem>
              <MenuItem value="very_important">Very important (prefer green mortgage)</MenuItem>
            </Select>
          </FormControl>
        </Grid>

        {/* AFM Compliance Score Display */}
        {validationResult && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2, bgcolor: 'background.paper' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                AFM Compliance Assessment
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
                <Typography variant="body2">
                  Compliance Score: {validationResult.afm_compliance_score}%
                </Typography>
                <Chip
                  label={validationResult.risk_assessment.toUpperCase()}
                  color={validationResult.risk_assessment === 'low' ? 'success' :
                         validationResult.risk_assessment === 'medium' ? 'warning' : 'error'}
                  size="small"
                />
              </Box>
              <LinearProgress
                variant="determinate"
                value={validationResult.afm_compliance_score}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 700 }}>
            AFM-Compliant Client Intake
          </Typography>
          {clientProfile.id && (
            <Chip label={`ID: ${clientProfile.id}`} size="small" variant="outlined" />
          )}
        </Box>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          Complete client assessment following Dutch AFM regulatory requirements
        </Typography>

        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Completion Progress: {intakeProgress?.completion_percentage || 0}%
          </Typography>
          <LinearProgress
            variant="determinate"
            value={intakeProgress?.completion_percentage || 0}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>
      </Box>

      <Card>
        <CardContent>
          <Stepper activeStep={activeStep} orientation="vertical">
            {steps.map((step, index) => (
              <Step key={step.label} completed={step.completed}>
                <StepLabel
                  icon={<step.icon />}
                  optional={
                    <Typography variant="caption">{step.description}</Typography>
                  }
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      {step.label}
                    </Typography>
                    {step.completed && <CheckCircle sx={{ color: 'success.main', fontSize: 16 }} />}
                  </Box>
                </StepLabel>
                <StepContent>
                  <Box sx={{ py: 2 }}>
                    {index === 0 && renderPersonalInformation()}
                    {index === 1 && renderEmploymentInformation()}
                    {index === 4 && renderAFMSuitabilityAssessment()}
                    {/* Add other step content renderers */}

                    {/* Action buttons */}
                    <Box sx={{ mt: 4, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                      <Button
                        disabled={activeStep === 0}
                        onClick={handleBack}
                        variant="outlined"
                        startIcon={<ArrowBack />}
                      >
                        Back
                      </Button>

                      <Button
                        variant="outlined"
                        onClick={handleSaveDraft}
                        disabled={saving}
                        startIcon={saving ? <CircularProgress size={16} /> : <SaveIcon />}
                      >
                        {saving ? 'Saving...' : 'Save Draft'}
                      </Button>

                      {activeStep === steps.length - 1 ? (
                        <Button
                          variant="contained"
                          onClick={handleSubmit}
                          disabled={submitting || !intakeProgress?.can_submit}
                          startIcon={submitting ? <CircularProgress size={16} /> : <SendIcon />}
                          sx={{
                            background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                          }}
                        >
                          {submitting ? 'Submitting...' : 'Complete AFM Assessment'}
                        </Button>
                      ) : (
                        <Button
                          variant="contained"
                          onClick={handleNext}
                          endIcon={<ArrowForward />}
                        >
                          Continue
                        </Button>
                      )}
                    </Box>

                    {/* Validation errors */}
                    {validationResult && !validationResult.isValid && (
                      <Alert severity="error" sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                          Please correct the following errors:
                        </Typography>
                        <ul style={{ margin: 0, paddingLeft: '20px' }}>
                          {Object.entries(validationResult.errors).map(([field, error]) => (
                            <li key={field}>{error}</li>
                          ))}
                        </ul>
                      </Alert>
                    )}

                    {/* Validation warnings */}
                    {validationResult && validationResult.warnings && Object.keys(validationResult.warnings).length > 0 && (
                      <Alert severity="warning" sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                          Warnings:
                        </Typography>
                        <ul style={{ margin: 0, paddingLeft: '20px' }}>
                          {Object.entries(validationResult.warnings).map(([field, warning]) => (
                            <li key={field}>{warning}</li>
                          ))}
                        </ul>
                      </Alert>
                    )}
                  </Box>
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>
    </Container>
  );
};

export default AFMClientIntake;
