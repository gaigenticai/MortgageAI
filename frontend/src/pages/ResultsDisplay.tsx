/**
 * Final Results Display
 *
 * Comprehensive results display showing analysis from both:
 * - Quality Control Agent (document validation and completeness)
 * - Compliance & Plain-Language Advisor Agent (mortgage advice)
 * - Final application status and next steps
 * - Professional summary with all findings
 */

import React from 'react';
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
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stepper,
  Step,
  StepLabel
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Info,
  ExpandMore,
  Assignment,
  Work,
  Home,
  AccountCircle,
  Analytics,
  Gavel,
  Celebration,
  Download,
  Print,
  Share,
  NavigateBefore
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

// Mock final results combining both agents
interface FinalResults {
  application_id: string;
  overall_status: 'approved' | 'conditional' | 'rejected' | 'pending_review';
  overall_score: number;

  quality_control: {
    completeness_score: number;
    passed: boolean;
    critical_issues: number;
    recommendations: string[];
  };

  compliance: {
    compliance_score: number;
    readability_level: string;
    advice_generated: boolean;
    understanding_confirmed: boolean;
  };

  final_advice: {
    summary: string;
    key_recommendations: string[];
    next_steps: string[];
    risk_assessment: string;
  };

  processing_timeline: Array<{
    step: string;
    status: 'completed' | 'in_progress' | 'pending';
    timestamp: string | null;
    details: string;
  }>;

  generated_at: string;
}

const ResultsDisplay: React.FC = () => {
  const navigate = useNavigate();

  // Mock final results
  const finalResults: FinalResults = {
    application_id: "APP-2025-001-FINAL",
    overall_status: 'approved',
    overall_score: 92.5,

    quality_control: {
      completeness_score: 87.5,
      passed: true,
      critical_issues: 0,
      recommendations: [
        "Verify property valuation with additional documentation",
        "Consider debt-to-income ratio optimization"
      ]
    },

    compliance: {
      compliance_score: 95,
      readability_level: "B1 (Intermediate)",
      advice_generated: true,
      understanding_confirmed: true
    },

    final_advice: {
      summary: `Your mortgage application has been successfully processed through our comprehensive AI-powered system. Both quality control and compliance checks have passed with excellent scores. You are well-positioned for mortgage approval with the recommended fixed-rate option.`,
      key_recommendations: [
        "Proceed with a 25-year fixed-rate mortgage at current competitive rates",
        "Maintain emergency fund covering 3-6 months of mortgage payments",
        "Consider additional property valuation to strengthen application",
        "Schedule consultation with mortgage advisor within 7 days"
      ],
      next_steps: [
        "Application submitted to lender for underwriting review",
        "Lender will contact you within 2-3 business days",
        "Prepare additional documentation if requested",
        "Monitor application status through lender portal"
      ],
      risk_assessment: "Low risk - Strong financial profile with good debt-to-income ratio and stable employment"
    },

    processing_timeline: [
      {
        step: "Application Submission",
        status: "completed",
        timestamp: "2025-09-25T10:00:00Z",
        details: "Mortgage application form completed and validated"
      },
      {
        step: "Document Upload",
        status: "completed",
        timestamp: "2025-09-25T10:15:00Z",
        details: "All required documents uploaded and categorized"
      },
      {
        step: "Quality Control Analysis",
        status: "completed",
        timestamp: "2025-09-25T10:20:00Z",
        details: "AI-powered document validation and anomaly detection completed"
      },
      {
        step: "Compliance Review",
        status: "completed",
        timestamp: "2025-09-25T10:25:00Z",
        details: "AFM-compliant advice generated and understanding confirmed"
      },
      {
        step: "Final Processing",
        status: "completed",
        timestamp: "2025-09-25T10:30:00Z",
        details: "Application prepared for lender submission"
      },
      {
        step: "Lender Review",
        status: "pending",
        timestamp: null,
        details: "Underwriting and final approval process"
      }
    ],

    generated_at: new Date().toISOString()
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'success';
      case 'conditional': return 'warning';
      case 'rejected': return 'error';
      case 'pending_review': return 'info';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'approved': return <CheckCircle />;
      case 'conditional': return <Warning />;
      case 'rejected': return <Error />;
      case 'pending_review': return <Info />;
      default: return <Info />;
    }
  };

  const getTimelineIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle color="success" />;
      case 'in_progress': return <Info color="primary" />;
      case 'pending': return <Info color="disabled" />;
      default: return <Info />;
    }
  };

  const handleDownloadReport = () => {
    // In a real app, this would generate and download a PDF report
    alert('Report download feature would be implemented here');
  };

  const handlePrint = () => {
    window.print();
  };

  const handleShare = () => {
    // In a real app, this would share the results
    alert('Share feature would be implemented here');
  };

  const handleBack = () => {
    navigate('/compliance');
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        {/* Header */}
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
            Application Processing Complete
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
            Application ID: {finalResults.application_id}
          </Typography>

          {/* Overall Status */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="h2" sx={{ fontWeight: 'bold', color: `${getStatusColor(finalResults.overall_status)}.main`, mb: 1 }}>
              {finalResults.overall_score.toFixed(1)}%
            </Typography>
            <Typography variant="h6" gutterBottom>
              Overall Application Score
            </Typography>
            <Chip
              label={finalResults.overall_status.replace('_', ' ').toUpperCase()}
              color={getStatusColor(finalResults.overall_status) as any}
              icon={getStatusIcon(finalResults.overall_status)}
              sx={{ fontSize: '1.1rem', py: 1.5, px: 3 }}
            />
          </Box>

          {/* Success Message */}
          {finalResults.overall_status === 'approved' && (
            <Alert severity="success" sx={{ mb: 3, py: 2 }}>
              <Typography variant="h6" gutterBottom>
                🎉 Congratulations! Your Application is Approved
              </Typography>
              <Typography variant="body2">
                Your mortgage application has passed all automated quality control and compliance checks.
                It has been forwarded to your lender for final underwriting review.
              </Typography>
            </Alert>
          )}
        </Box>

        <Divider sx={{ my: 4 }} />

        {/* Processing Timeline */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Processing Timeline
          </Typography>
          <Stepper orientation="vertical">
            {finalResults.processing_timeline.map((step, index) => (
              <Step key={step.step} active={step.status === 'in_progress'} completed={step.status === 'completed'}>
                <StepLabel
                  icon={getTimelineIcon(step.status)}
                  sx={{
                    '& .MuiStepLabel-label': {
                      fontSize: '1rem',
                      fontWeight: step.status === 'completed' ? 'bold' : 'normal'
                    }
                  }}
                >
                  <Box>
                    <Typography variant="subtitle1">{step.step}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {step.details}
                    </Typography>
                    {step.timestamp && (
                      <Typography variant="caption" color="text.secondary">
                        {new Date(step.timestamp).toLocaleString()}
                      </Typography>
                    )}
                  </Box>
                </StepLabel>
              </Step>
            ))}
          </Stepper>
        </Box>

        <Divider sx={{ my: 4 }} />

        {/* Detailed Scores */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Detailed Analysis Scores
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Analytics color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">Quality Control</Typography>
                  </Box>
                  <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'primary.main', mb: 1 }}>
                    {finalResults.quality_control.completeness_score.toFixed(1)}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Completeness Score
                  </Typography>
                  <Chip
                    label={finalResults.quality_control.passed ? "PASSED" : "ISSUES FOUND"}
                    color={finalResults.quality_control.passed ? "success" : "warning"}
                    size="small"
                  />
                  <Typography variant="body2" sx={{ mt: 2 }}>
                    Critical Issues: {finalResults.quality_control.critical_issues}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Gavel color="success" sx={{ mr: 1 }} />
                    <Typography variant="h6">Compliance & Advice</Typography>
                  </Box>
                  <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'success.main', mb: 1 }}>
                    {finalResults.compliance.compliance_score}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    AFM Compliance Score
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Readability: {finalResults.compliance.readability_level}
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Chip
                      label="ADVICE GENERATED"
                      color="success"
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    <Chip
                      label="UNDERSTANDING CONFIRMED"
                      color="success"
                      size="small"
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>

        {/* Final Advice */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Final Recommendations
          </Typography>

          <Accordion defaultExpanded sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Summary</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body1" sx={{ lineHeight: 1.6 }}>
                {finalResults.final_advice.summary}
              </Typography>
            </AccordionDetails>
          </Accordion>

          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Key Recommendations</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List>
                {finalResults.final_advice.key_recommendations.map((rec, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText primary={rec} />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>

          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Next Steps</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List>
                {finalResults.final_advice.next_steps.map((step, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <Info color="primary" />
                    </ListItemIcon>
                    <ListItemText primary={step} />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">Risk Assessment</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Alert severity="info">
                <Typography variant="body2">
                  {finalResults.final_advice.risk_assessment}
                </Typography>
              </Alert>
            </AccordionDetails>
          </Accordion>
        </Box>

        {/* Quality Control Recommendations */}
        {finalResults.quality_control.recommendations.length > 0 && (
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Quality Control Recommendations
            </Typography>
            <Alert severity="info">
              <Typography variant="body2" gutterBottom>
                The following recommendations were identified during quality control analysis:
              </Typography>
              <List dense>
                {finalResults.quality_control.recommendations.map((rec, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <Warning color="warning" />
                    </ListItemIcon>
                    <ListItemText primary={rec} />
                  </ListItem>
                ))}
              </List>
            </Alert>
          </Box>
        )}

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 4 }}>
          <Button
            variant="outlined"
            onClick={handleBack}
            startIcon={<NavigateBefore />}
            sx={{ borderRadius: 2 }}
          >
            Back to Compliance Check
          </Button>

          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={handleDownloadReport}
              sx={{ borderRadius: 2 }}
            >
              Download Report
            </Button>
            <Button
              variant="outlined"
              startIcon={<Print />}
              onClick={handlePrint}
              sx={{ borderRadius: 2 }}
            >
              Print
            </Button>
            <Button
              variant="outlined"
              startIcon={<Share />}
              onClick={handleShare}
              sx={{ borderRadius: 2 }}
            >
              Share
            </Button>
          </Box>
        </Box>

        {/* Footer */}
        <Box sx={{ mt: 4, pt: 3, borderTop: 1, borderColor: 'divider', textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            Report generated on {new Date(finalResults.generated_at).toLocaleString()}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Powered by MortgageAI - Agentic AI Solution for Mortgage Advice
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default ResultsDisplay;
