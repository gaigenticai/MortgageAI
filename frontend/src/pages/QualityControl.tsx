/**
 * Quality Control Interface
 *
 * Interface for interacting with the Mortgage Application Quality Control Agent with:
 * - Real-time document analysis and validation
 * - Field-level validation results
 * - Anomaly detection and consistency checks
 * - Completeness scoring and remediation suggestions
 * - Integration with backend QC agent
 */

import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Alert,
  LinearProgress,
  Grid,
  Card,
  CardContent,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  CircularProgress
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
  NavigateNext,
  NavigateBefore
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { useNavigate } from 'react-router-dom';

// Mock QC results - in production this would come from the backend agent
interface QCResult {
  application_id: string;
  completeness_score: number;
  passed: boolean;
  field_validation: {
    results: Array<{
      field: string;
      valid: boolean;
      error?: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
    }>;
    cross_validation_passed: boolean;
    summary: {
      total_fields: number;
      valid_fields: number;
      invalid_fields: number;
    };
  };
  anomaly_check: {
    anomalies: Array<{
      field: string;
      type: string;
      description: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
    }>;
    severity_score: number;
    requires_review: boolean;
  };
  document_analysis: Array<{
    document_type: string;
    processing_status: 'success' | 'error';
    extracted_data: Record<string, any>;
    required_fields: string[];
    completeness: number;
  }>;
  remediation_instructions: Array<{
    type: string;
    field?: string;
    issue: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    instruction: string;
  }>;
  analyzed_at: string;
  processing_summary: {
    documents_processed: number;
    fields_validated: number;
    anomalies_found: number;
    critical_issues: number;
  };
}

const QualityControl: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [analyzing, setAnalyzing] = useState(true);
  const [qcResult, setQcResult] = useState<QCResult | null>(null);

  useEffect(() => {
    // Simulate QC analysis
    const analyzeApplication = async () => {
      setAnalyzing(true);

      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Mock QC results
      const mockResult: QCResult = {
        application_id: "APP-2025-001",
        completeness_score: 87.5,
        passed: true,
        field_validation: {
          results: [
            {
              field: "applicant_name",
              valid: true,
              severity: "low"
            },
            {
              field: "date_of_birth",
              valid: true,
              severity: "low"
            },
            {
              field: "annual_income",
              valid: true,
              severity: "low"
            },
            {
              field: "mortgage_amount",
              valid: true,
              severity: "low"
            },
            {
              field: "property_value",
              valid: false,
              error: "Property value appears unusually high for the area",
              severity: "medium"
            },
            {
              field: "loan_term",
              valid: true,
              severity: "low"
            }
          ],
          cross_validation_passed: true,
          summary: {
            total_fields: 6,
            valid_fields: 5,
            invalid_fields: 1
          }
        },
        anomaly_check: {
          anomalies: [
            {
              field: "loan_to_value_ratio",
              type: "unusual_ratio",
              description: "Loan-to-value ratio of 85% is above typical thresholds",
              severity: "medium"
            }
          ],
          severity_score: 2.5,
          requires_review: true
        },
        document_analysis: [
          {
            document_type: "application_form",
            processing_status: "success",
            extracted_data: {
              applicant_name: "John Doe",
              date_of_birth: "1985-03-15",
              mortgage_amount: 340000,
              property_value: 400000
            },
            required_fields: ["applicant_name", "date_of_birth", "mortgage_amount", "property_value"],
            completeness: 100
          },
          {
            document_type: "income_proof",
            processing_status: "success",
            extracted_data: {
              annual_income: 75000,
              employer_name: "Tech Corp",
              tax_year: "2024"
            },
            required_fields: ["employer_name", "annual_income", "tax_year"],
            completeness: 100
          }
        ],
        remediation_instructions: [
          {
            type: "field_correction",
            field: "property_value",
            issue: "Property value validation warning",
            severity: "medium",
            instruction: "Please verify the property valuation amount. Consider providing additional documentation if the value seems correct."
          },
          {
            type: "anomaly_resolution",
            field: "loan_to_value_ratio",
            issue: "High loan-to-value ratio detected",
            severity: "medium",
            instruction: "Consider increasing down payment or exploring lower loan amounts to reduce LTV ratio."
          }
        ],
        analyzed_at: new Date().toISOString(),
        processing_summary: {
          documents_processed: 4,
          fields_validated: 12,
          anomalies_found: 1,
          critical_issues: 0
        }
      };

      setQcResult(mockResult);
      setAnalyzing(false);

      if (mockResult.passed) {
        enqueueSnackbar('Quality control analysis completed successfully!', { variant: 'success' });
      } else {
        enqueueSnackbar('Quality control issues found - please review remediation suggestions', { variant: 'warning' });
      }
    };

    analyzeApplication();
  }, [enqueueSnackbar]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'error';
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'high': return <Error />;
      case 'medium': return <Warning />;
      case 'low': return <Info />;
      default: return <Info />;
    }
  };

  const getDocumentIcon = (type: string) => {
    switch (type) {
      case 'application_form': return <Assignment />;
      case 'income_proof': return <Work />;
      case 'property_documents': return <Home />;
      case 'id_document': return <AccountCircle />;
      default: return <Assignment />;
    }
  };

  const handleContinue = () => {
    navigate('/compliance');
  };

  const handleBack = () => {
    navigate('/documents');
  };

  if (analyzing) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={3} sx={{ p: 4, borderRadius: 2, textAlign: 'center' }}>
          <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
            Quality Control Analysis
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
            <Analytics sx={{ fontSize: 64, color: 'primary.main' }} />
            <Typography variant="h6">
              Analyzing your application documents...
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Our AI agents are performing comprehensive validation checks
            </Typography>
            <CircularProgress size={60} />
            <LinearProgress sx={{ width: '100%', maxWidth: 400, height: 8, borderRadius: 4 }} />
          </Box>
        </Paper>
      </Container>
    );
  }

  if (!qcResult) return null;

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ fontWeight: 'bold', mb: 1 }}>
          Quality Control Results
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 4 }}>
          Comprehensive analysis of your mortgage application documents
        </Typography>

        {/* Overall Score */}
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" sx={{ fontWeight: 'bold', color: qcResult.passed ? 'success.main' : 'warning.main' }}>
            {qcResult.completeness_score.toFixed(1)}%
          </Typography>
          <Typography variant="h6" gutterBottom>
            Completeness Score
          </Typography>
          <Chip
            label={qcResult.passed ? "PASSED" : "REQUIRES ATTENTION"}
            color={qcResult.passed ? "success" : "warning"}
            icon={qcResult.passed ? <CheckCircle /> : <Warning />}
            sx={{ fontSize: '1rem', py: 1, px: 2 }}
          />
        </Box>

        {/* Processing Summary */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Processing Summary
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h4" color="primary.main" sx={{ fontWeight: 'bold' }}>
                    {qcResult.processing_summary.documents_processed}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Documents Processed
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h4" color="success.main" sx={{ fontWeight: 'bold' }}>
                    {qcResult.processing_summary.fields_validated}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Fields Validated
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h4" color="warning.main" sx={{ fontWeight: 'bold' }}>
                    {qcResult.processing_summary.anomalies_found}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Anomalies Found
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="h4" color="error.main" sx={{ fontWeight: 'bold' }}>
                    {qcResult.processing_summary.critical_issues}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Critical Issues
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ my: 4 }} />

        {/* Detailed Results */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Detailed Analysis
          </Typography>

          {/* Field Validation */}
          <Accordion defaultExpanded sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CheckCircle color="success" />
                <Typography variant="h6">
                  Field Validation ({qcResult.field_validation.summary.valid_fields}/{qcResult.field_validation.summary.total_fields} valid)
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <List>
                {qcResult.field_validation.results.map((result, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      {result.valid ? (
                        <CheckCircle color="success" />
                      ) : (
                        getSeverityIcon(result.severity)
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={result.field.replace(/_/g, ' ').toUpperCase()}
                      secondary={result.error || 'Valid'}
                    />
                    <Chip
                      label={result.valid ? 'Valid' : result.severity.toUpperCase()}
                      color={result.valid ? 'success' : getSeverityColor(result.severity) as any}
                      size="small"
                    />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>

          {/* Anomaly Detection */}
          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Warning color="warning" />
                <Typography variant="h6">
                  Anomaly Detection ({qcResult.anomaly_check.anomalies.length} issues found)
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              {qcResult.anomaly_check.anomalies.length > 0 ? (
                <List>
                  {qcResult.anomaly_check.anomalies.map((anomaly, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        {getSeverityIcon(anomaly.severity)}
                      </ListItemIcon>
                      <ListItemText
                        primary={anomaly.field.replace(/_/g, ' ').toUpperCase()}
                        secondary={anomaly.description}
                      />
                      <Chip
                        label={anomaly.severity.toUpperCase()}
                        color={getSeverityColor(anomaly.severity) as any}
                        size="small"
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography color="text.secondary">No anomalies detected</Typography>
              )}
            </AccordionDetails>
          </Accordion>

          {/* Document Analysis */}
          <Accordion sx={{ mb: 2 }}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Assignment color="primary" />
                <Typography variant="h6">
                  Document Analysis ({qcResult.document_analysis.length} documents processed)
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <List>
                {qcResult.document_analysis.map((doc, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      {getDocumentIcon(doc.document_type)}
                    </ListItemIcon>
                    <ListItemText
                      primary={doc.document_type.replace(/_/g, ' ').toUpperCase()}
                      secondary={`Completeness: ${doc.completeness}% • Status: ${doc.processing_status}`}
                    />
                    <Chip
                      label={`${doc.completeness}%`}
                      color={doc.completeness === 100 ? 'success' : 'warning'}
                      size="small"
                    />
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        </Box>

        {/* Remediation Instructions */}
        {qcResult.remediation_instructions.length > 0 && (
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Remediation Instructions
            </Typography>
            <Alert severity="info" sx={{ mb: 2 }}>
              The following issues were identified and require attention before proceeding:
            </Alert>
            <List>
              {qcResult.remediation_instructions.map((instruction, index) => (
                <ListItem key={index} sx={{ borderLeft: 4, borderColor: `${getSeverityColor(instruction.severity)}.main`, pl: 2, mb: 1 }}>
                  <ListItemIcon>
                    {getSeverityIcon(instruction.severity)}
                  </ListItemIcon>
                  <ListItemText
                    primary={instruction.issue}
                    secondary={instruction.instruction}
                  />
                  <Chip
                    label={instruction.severity.toUpperCase()}
                    color={getSeverityColor(instruction.severity) as any}
                    size="small"
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
          <Button
            variant="outlined"
            onClick={handleBack}
            startIcon={<NavigateBefore />}
            sx={{ borderRadius: 2 }}
          >
            Back to Documents
          </Button>
          <Button
            variant="contained"
            onClick={handleContinue}
            endIcon={<NavigateNext />}
            sx={{ borderRadius: 2, px: 4 }}
          >
            Continue to Compliance Check
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default QualityControl;
