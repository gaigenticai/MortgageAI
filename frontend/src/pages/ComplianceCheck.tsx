/**
 * Compliance & Plain-Language Advisor Interface
 *
 * Interface for interacting with the Compliance & Plain-Language Advisor Agent with:
 * - Mortgage advice generation with AFM compliance
 * - Plain-language transformation and simplification
 * - Explain-back validation for user understanding
 * - Real-time compliance checking and suggestions
 * - Integration with backend compliance agent
 */

import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Alert,
  Grid,
  Card,
  CardContent,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress
} from '@mui/material';
import {
  Gavel,
  CheckCircle,
  Help,
  ExpandMore,
  Lightbulb,
  Warning,
  NavigateNext,
  NavigateBefore,
  Send,
  ThumbUp,
  ThumbDown,
  Psychology
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { useNavigate } from 'react-router-dom';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';

// Mock compliance advice results - in production this would come from the backend agent
interface ComplianceResult {
  advice_id: string;
  original_advice: string;
  compliant_advice: string;
  compliance_score: number;
  readability_level: string;
  key_requirements: string[];
  missing_disclosures: string[];
  simplification_suggestions: Array<{
    original: string;
    simplified: string;
    improvement: string;
  }>;
  explain_back_questions: Array<{
    question: string;
    expected_answer: string;
    user_answer?: string;
    understood: boolean;
  }>;
  generated_at: string;
}

interface UserProfile {
  firstTimeBuyer: boolean;
  age: number;
  income: number;
  mortgageAmount: number;
  propertyValue: number;
  employmentStatus: string;
  riskTolerance: string;
}

const ComplianceCheck: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [generating, setGenerating] = useState(false);
  const [complianceResult, setComplianceResult] = useState<ComplianceResult | null>(null);
  const [explainBackDialog, setExplainBackDialog] = useState(false);
  const [userAnswers, setUserAnswers] = useState<Record<number, string>>({});

  const userProfile: UserProfile = {
    firstTimeBuyer: true,
    age: 35,
    income: 75000,
    mortgageAmount: 340000,
    propertyValue: 400000,
    employmentStatus: 'employed',
    riskTolerance: 'moderate'
  };

  const handleGenerateAdvice = async (values: { topic: string; complexity: string }) => {
    setGenerating(true);

    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 4000));

    // Mock compliance result
    const mockResult: ComplianceResult = {
      advice_id: "ADV-2025-001",
      original_advice: values.topic,
      compliant_advice: `Based on your profile as a ${userProfile.firstTimeBuyer ? 'first-time' : 'experienced'} buyer with an annual income of €${userProfile.income.toLocaleString()}, we recommend considering a fixed-rate mortgage for the next 10-15 years.

Key points to consider:
• Your debt-to-income ratio is ${((userProfile.mortgageAmount / userProfile.income) * 100).toFixed(1)}%, which is within acceptable limits
• Interest rates are currently competitive, but consider that rates may rise in the future
• You should maintain an emergency fund covering 3-6 months of mortgage payments

Important disclosures:
• Mortgage interest rates can change over time
• Early repayment penalties may apply if you sell or refinance within the first 10 years
• Additional costs include property taxes, insurance, and maintenance

This advice is based on current market conditions and your financial profile. We recommend consulting with a mortgage advisor for personalized guidance.`,
      compliance_score: 95,
      readability_level: "B1 (Intermediate)",
      key_requirements: [
        "Complete cost disclosures",
        "Risk warnings included",
        "Plain language used",
        "Personalized to user profile",
        "No misleading statements"
      ],
      missing_disclosures: [],
      simplification_suggestions: [
        {
          original: "debt-to-income ratio",
          simplified: "how much of your income goes to debt payments",
          improvement: "Uses everyday language instead of financial jargon"
        },
        {
          original: "competitive interest rates",
          simplified: "good interest rates available",
          improvement: "Simpler vocabulary maintains meaning"
        }
      ],
      explain_back_questions: [
        {
          question: "What does it mean that interest rates can change over time?",
          expected_answer: "Mortgage rates are not guaranteed to stay the same",
          understood: false
        },
        {
          question: "Why is an emergency fund important for homeownership?",
          expected_answer: "To cover unexpected expenses like repairs or job loss",
          understood: false
        },
        {
          question: "What are the main costs of owning a home besides the mortgage payment?",
          expected_answer: "Property taxes, insurance, maintenance, and utilities",
          understood: false
        }
      ],
      generated_at: new Date().toISOString()
    };

    setComplianceResult(mockResult);
    setGenerating(false);
    enqueueSnackbar('Compliance advice generated successfully!', { variant: 'success' });
  };

  const handleExplainBack = () => {
    setExplainBackDialog(true);
  };

  const handleAnswerSubmit = (questionIndex: number, answer: string) => {
    setUserAnswers(prev => ({ ...prev, [questionIndex]: answer }));

    // Simulate understanding check
    const question = complianceResult?.explain_back_questions[questionIndex];
    const understood = answer.toLowerCase().includes(question?.expected_answer.toLowerCase().split(' ')[0] || '');

    if (complianceResult) {
      const updatedQuestions = [...complianceResult.explain_back_questions];
      updatedQuestions[questionIndex] = {
        ...updatedQuestions[questionIndex],
        user_answer: answer,
        understood
      };

      setComplianceResult({
        ...complianceResult,
        explain_back_questions: updatedQuestions
      });
    }
  };

  const handleContinue = () => {
    const allUnderstood = complianceResult?.explain_back_questions.every(q => q.understood) ?? false;

    if (allUnderstood) {
      enqueueSnackbar('All compliance checks completed successfully!', { variant: 'success' });
      navigate('/results');
    } else {
      enqueueSnackbar('Please answer all understanding questions before proceeding', { variant: 'warning' });
    }
  };

  const handleBack = () => {
    navigate('/quality-control');
  };

  if (generating) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 4, borderRadius: 2, textAlign: 'center' }}>
          <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
            Generating Compliance Advice
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
            <Psychology sx={{ fontSize: 64, color: 'primary.main' }} />
            <Typography variant="h6">
              Our AI advisor is preparing personalized mortgage guidance...
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Ensuring AFM compliance and plain-language requirements
            </Typography>
            <CircularProgress size={60} />
          </Box>
        </Paper>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 8, mb: 8 }}>
      <Paper elevation={0} sx={{
        p: 6,
        borderRadius: 4,
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(226, 232, 240, 0.8)',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1), 0 4px 10px rgba(0, 0, 0, 0.05)',
      }}>
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Box sx={{
            width: 80,
            height: 80,
            borderRadius: 4,
            background: 'linear-gradient(135deg, #EC4899 0%, #F472B6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mx: 'auto',
            mb: 4,
            boxShadow: '0 8px 24px rgba(236, 72, 153, 0.25), 0 4px 12px rgba(0, 0, 0, 0.1)',
          }}>
            <Gavel sx={{ color: 'white', fontSize: 40 }} />
          </Box>
          <Typography
            variant="h2"
            component="h1"
            gutterBottom
            align="center"
            sx={{
              fontWeight: 700,
              mb: 3,
              background: 'linear-gradient(135deg, #EC4899 0%, #F472B6 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Compliance & Plain-Language Advisor
          </Typography>
          <Typography
            variant="h6"
            align="center"
            color="text.secondary"
            sx={{ fontWeight: 400, maxWidth: 600, mx: 'auto', lineHeight: 1.6 }}
          >
            Get AFM-compliant mortgage advice with AI-powered plain-language explanations
          </Typography>
        </Box>

        {/* User Profile Summary */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Your Profile Summary
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'bold' }}>
                    €{userProfile.income.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Annual Income
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'bold' }}>
                    €{userProfile.mortgageAmount.toLocaleString()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Mortgage Amount
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'bold' }}>
                    {userProfile.age} years
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Age
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h5" color="primary.main" sx={{ fontWeight: 'bold' }}>
                    {userProfile.firstTimeBuyer ? 'First-time' : 'Experienced'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Buyer Status
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>

        {/* Advice Generation Form */}
        {!complianceResult && (
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Request Personalized Advice
            </Typography>
            <Formik
              initialValues={{
                topic: 'mortgage options for first-time buyers',
                complexity: 'intermediate'
              }}
              validationSchema={Yup.object({
                topic: Yup.string().required('Please specify what advice you need'),
                complexity: Yup.string().required('Please select complexity level')
              })}
              onSubmit={handleGenerateAdvice}
            >
              {(formik) => (
                <Form>
                  <Grid container spacing={3}>
                    <Grid item xs={12}>
                      <Field
                        as={TextField}
                        fullWidth
                        name="topic"
                        label="What mortgage advice do you need?"
                        placeholder="e.g., mortgage options for first-time buyers, fixed vs variable rates, etc."
                        variant="outlined"
                        multiline
                        rows={3}
                        error={formik.touched.topic && Boolean(formik.errors.topic)}
                        helperText={formik.touched.topic && formik.errors.topic}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth variant="outlined">
                        <InputLabel>Complexity Level</InputLabel>
                        <Field
                          as={Select}
                          name="complexity"
                          label="Complexity Level"
                          error={formik.touched.complexity && Boolean(formik.errors.complexity)}
                        >
                          <MenuItem value="basic">Basic - Simple explanations</MenuItem>
                          <MenuItem value="intermediate">Intermediate - Some detail</MenuItem>
                          <MenuItem value="advanced">Advanced - Full details</MenuItem>
                        </Field>
                        {formik.touched.complexity && formik.errors.complexity && (
                          <FormHelperText error>{formik.errors.complexity}</FormHelperText>
                        )}
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6} sx={{ display: 'flex', alignItems: 'center' }}>
                      <Button
                        type="submit"
                        variant="contained"
                        startIcon={<Send />}
                        disabled={formik.isSubmitting}
                        sx={{ borderRadius: 2, px: 4 }}
                        fullWidth
                      >
                        {formik.isSubmitting ? 'Generating...' : 'Generate Advice'}
                      </Button>
                    </Grid>
                  </Grid>
                </Form>
              )}
            </Formik>
          </Box>
        )}

        {/* Compliance Results */}
        {complianceResult && (
          <>
            {/* Compliance Score */}
            <Box sx={{ mb: 4, textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'success.main' }}>
                {complianceResult.compliance_score}%
              </Typography>
              <Typography variant="h6" gutterBottom>
                AFM Compliance Score
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Readability: {complianceResult.readability_level}
              </Typography>
              <Chip
                label="AFM COMPLIANT"
                color="success"
                icon={<CheckCircle />}
                sx={{ fontSize: '1rem', py: 1, px: 2 }}
              />
            </Box>

            <Divider sx={{ my: 4 }} />

            {/* Compliant Advice */}
            <Box sx={{ mb: 4 }}>
              <Typography variant="h6" gutterBottom>
                Your Personalized Mortgage Advice
              </Typography>
              <Paper variant="outlined" sx={{ p: 3, backgroundColor: 'grey.50' }}>
                <Typography variant="body1" sx={{ whiteSpace: 'pre-line', lineHeight: 1.6 }}>
                  {complianceResult.compliant_advice}
                </Typography>
              </Paper>
            </Box>

            {/* Detailed Analysis */}
            <Box sx={{ mb: 4 }}>
              <Typography variant="h6" gutterBottom>
                Compliance Analysis
              </Typography>

              {/* Requirements Met */}
              <Accordion defaultExpanded sx={{ mb: 2 }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Gavel color="success" />
                    <Typography variant="h6">
                      AFM Requirements Met ({complianceResult.key_requirements.length})
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <List>
                    {complianceResult.key_requirements.map((req, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <CheckCircle color="success" />
                        </ListItemIcon>
                        <ListItemText primary={req} />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>

              {/* Simplifications Made */}
              <Accordion sx={{ mb: 2 }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Lightbulb color="info" />
                    <Typography variant="h6">
                      Language Simplifications ({complianceResult.simplification_suggestions.length})
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <List>
                    {complianceResult.simplification_suggestions.map((suggestion, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <Lightbulb color="info" />
                        </ListItemIcon>
                        <ListItemText
                          primary={`"${suggestion.original}" → "${suggestion.simplified}"`}
                          secondary={suggestion.improvement}
                        />
                      </ListItem>
                    ))}
                  </List>
                </AccordionDetails>
              </Accordion>
            </Box>

            {/* Explain-Back Validation */}
            <Box sx={{ mb: 4 }}>
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  To ensure you understand this advice, please answer the following questions.
                  This is required by AFM regulations to confirm consumer comprehension.
                </Typography>
              </Alert>

              <List>
                {complianceResult.explain_back_questions.map((question, index) => (
                  <ListItem key={index} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, mb: 1 }}>
                    <ListItemIcon>
                      <Help color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary={question.question}
                      secondary={question.user_answer ? (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                            Your answer: {question.user_answer}
                          </Typography>
                          <Chip
                            size="small"
                            label={question.understood ? "UNDERSTOOD" : "NEEDS CLARIFICATION"}
                            color={question.understood ? "success" : "warning"}
                            sx={{ mt: 0.5 }}
                          />
                        </Box>
                      ) : "Click to answer"}
                    />
                    {!question.user_answer && (
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={() => {
                          const answer = prompt(question.question);
                          if (answer) {
                            handleAnswerSubmit(index, answer);
                          }
                        }}
                      >
                        Answer
                      </Button>
                    )}
                  </ListItem>
                ))}
              </List>
            </Box>
          </>
        )}

        {/* Action Buttons */}
        {complianceResult && (
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
            <Button
              variant="outlined"
              onClick={handleBack}
              startIcon={<NavigateBefore />}
              sx={{ borderRadius: 2 }}
            >
              Back to Quality Control
            </Button>
            <Button
              variant="contained"
              onClick={handleContinue}
              endIcon={<NavigateNext />}
              disabled={!complianceResult.explain_back_questions.every(q => q.understood)}
              sx={{ borderRadius: 2, px: 4 }}
            >
              View Final Results
            </Button>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ComplianceCheck;
