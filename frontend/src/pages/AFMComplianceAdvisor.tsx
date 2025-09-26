/**
 * AFM Compliance Advisor
 *
 * Provides AFM-compliant mortgage advice based on client profile analysis
 * Includes risk assessment, product recommendations, and regulatory compliance checks.
 */
import React, { useState, useEffect } from 'react';
import {
  Container,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Grid,
  Chip,
  LinearProgress,
  Alert,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Paper,
  CircularProgress,
} from '@mui/material';
import {
  Gavel,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Assessment,
  TrendingUp,
  Security,
  Business,
  ArrowBack,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { complianceApi, ComplianceAnalysis } from '../services/complianceApi';


const AFMComplianceAdvisor: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { enqueueSnackbar } = useSnackbar();

  const [analysis, setAnalysis] = useState<ComplianceAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [generatingAdvice, setGeneratingAdvice] = useState(false);

  // Initialize API service with snackbar
  useEffect(() => {
    complianceApi.setSnackbar(enqueueSnackbar);
  }, [enqueueSnackbar]);

  // Get client data from navigation state
  const clientId = location.state?.clientId;

  useEffect(() => {
    if (clientId) {
      loadComplianceAnalysis(clientId);
    } else {
      // If no client ID, redirect to intake
      navigate('/afm-client-intake');
    }
  }, [clientId, navigate]);

  const loadComplianceAnalysis = async (id: string) => {
    try {
      setLoading(true);
      const analysisData = await complianceApi.getClientComplianceAssessment(id);
      setAnalysis(analysisData);
    } catch (error) {
      console.error('Failed to load compliance analysis:', error);
      enqueueSnackbar('Failed to load AFM compliance analysis', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const generateAdviceReport = async () => {
    if (!analysis) return;

    setGeneratingAdvice(true);
    try {
      const reportBlob = await complianceApi.generateComplianceReport(analysis.id, 'pdf');
      // Create download link for the report
      const url = window.URL.createObjectURL(reportBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `afm-compliance-report-${analysis.client_id}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      enqueueSnackbar('AFM compliance report generated successfully', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar('Failed to generate AFM compliance report', { variant: 'error' });
    } finally {
      setGeneratingAdvice(false);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
          flexDirection: 'column',
          gap: 2
        }}>
          <CircularProgress size={60} />
          <Typography variant="h6" color="text.secondary">
            Analyzing AFM Compliance...
          </Typography>
        </Box>
      </Container>
    );
  }

  if (!analysis) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4 }}>
          <Alert severity="error">
            Failed to load compliance analysis. Please try again.
          </Alert>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/afm-client-intake')}
          sx={{ mb: 2 }}
        >
          Back to Client Intake
        </Button>

        <Typography variant="h4" component="h1" sx={{ fontWeight: 700, mb: 2 }}>
          AFM Compliance Advisor
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          AFM-compliant mortgage advice and product recommendations
        </Typography>

        {/* Compliance Score Overview */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                  <Avatar sx={{
                    bgcolor: analysis.afm_status === 'compliant' ? 'success.main' :
                             analysis.afm_status === 'conditional' ? 'warning.main' : 'error.main',
                    mr: 2,
                    width: 48,
                    height: 48
                  }}>
                    <Gavel />
                  </Avatar>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" sx={{ fontWeight: 600 }}>
                      AFM Compliance Assessment
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Client ID: {analysis.client_id}
                    </Typography>
                  </Box>
                  <Chip
                    label={analysis.afm_status.toUpperCase()}
                    color={analysis.afm_status === 'compliant' ? 'success' :
                           analysis.afm_status === 'conditional' ? 'warning' : 'error'}
                    sx={{ fontWeight: 600 }}
                  />
                </Box>

                <Grid container spacing={3}>
                  <Grid item xs={12} sm={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h3" sx={{
                        fontWeight: 700,
                        color: 'primary.main',
                        mb: 1
                      }}>
                        {analysis.compliance_score}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Compliance Score
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h3" sx={{
                        fontWeight: 700,
                        color: analysis.risk_profile === 'low' ? 'success.main' :
                               analysis.risk_profile === 'medium' ? 'warning.main' : 'error.main',
                        mb: 1
                      }}>
                        {analysis.risk_profile.toUpperCase()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Risk Profile
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h3" sx={{
                        fontWeight: 700,
                        color: 'info.main',
                        mb: 1
                      }}>
                        {analysis.product_recommendations.length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Product Matches
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Compliance Recommendations */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                  Compliance Recommendations
                </Typography>

                <List>
                  {analysis.recommendations.map((rec, index) => (
                    <React.Fragment key={rec.id}>
                      <ListItem>
                        <ListItemIcon>
                          <Avatar sx={{
                            bgcolor: rec.type === 'approved' ? 'success.main' :
                                     rec.type === 'conditional' ? 'warning.main' : 'error.main',
                            width: 32,
                            height: 32
                          }}>
                            {rec.type === 'approved' ? <CheckCircle /> :
                             rec.type === 'conditional' ? <Warning /> : <ErrorIcon />}
                          </Avatar>
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {rec.title}
                              </Typography>
                              <Chip
                                label={rec.risk_level}
                                size="small"
                                color={rec.risk_level === 'low' ? 'success' :
                                       rec.risk_level === 'medium' ? 'warning' : 'error'}
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                {rec.description}
                              </Typography>
                              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                                AFM Requirements: {rec.afm_requirements.join(', ')}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < analysis.recommendations.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Product Recommendations */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                  Recommended Products
                </Typography>

                <List>
                  {analysis.product_recommendations.map((product, index) => (
                    <React.Fragment key={product.id}>
                      <ListItem>
                        <ListItemIcon>
                          <Avatar sx={{
                            bgcolor: product.afm_compliant ? 'success.main' : 'warning.main',
                            width: 32,
                            height: 32
                          }}>
                            <Business />
                          </Avatar>
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {product.lender} - {product.product_name}
                              </Typography>
                              {product.afm_compliant && (
                                <Chip label="AFM Compliant" size="small" color="success" />
                              )}
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Interest Rate: {product.interest_rate}% • Max LTV: {product.max_ltv}%
                              </Typography>
                              <Typography variant="caption" sx={{ color: 'primary.main', fontWeight: 600 }}>
                                Suitability Score: {product.suitability_score}%
                              </Typography>
                              {product.conditions && product.conditions.length > 0 && (
                                <Box sx={{ mt: 1 }}>
                                  <Typography variant="caption" sx={{ fontWeight: 600, color: 'warning.main' }}>
                                    Conditions:
                                  </Typography>
                                  <ul style={{ margin: 0, paddingLeft: '20px' }}>
                                    {product.conditions.map((condition, idx) => (
                                      <li key={idx}>
                                        <Typography variant="caption">{condition}</Typography>
                                      </li>
                                    ))}
                                  </ul>
                                </Box>
                              )}
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < analysis.product_recommendations.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button
            variant="outlined"
            onClick={() => navigate('/mortgage-application')}
            startIcon={<Business />}
          >
            Proceed to Application
          </Button>
          <Button
            variant="contained"
            onClick={generateAdviceReport}
            disabled={generatingAdvice}
            startIcon={generatingAdvice ? <CircularProgress size={16} /> : <Assessment />}
            sx={{
              background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
            }}
          >
            {generatingAdvice ? 'Generating Report...' : 'Generate Advice Report'}
          </Button>
        </Box>

        {/* Review Deadline Notice */}
        {analysis.review_deadline && (
          <Alert severity="info" sx={{ mt: 3 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
              Review Deadline
            </Typography>
            <Typography variant="body2">
              This AFM compliance assessment is valid until {new Date(analysis.review_deadline).toLocaleDateString('nl-NL')}.
              Please ensure all recommendations are addressed before this date.
            </Typography>
          </Alert>
        )}
      </Box>
    </Container>
  );
};

export default AFMComplianceAdvisor;
