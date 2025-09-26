/**
 * BKR Credit Check
 *
 * Integrates with Dutch BKR (Bureau Krediet Registratie) for credit scoring
 * Provides comprehensive credit assessment for mortgage applications
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
  Alert,
  Avatar,
  CircularProgress,
  Paper,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  Assessment,
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  CreditScore,
  ArrowBack,
  Refresh,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { creditCheckApi, CreditReport } from '../services/creditCheckApi';
import { useClient } from '../contexts/ClientContext';


const BKRCreditCheck: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const { currentClientId } = useClient();

  const [creditReport, setCreditReport] = useState<CreditReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [checking, setChecking] = useState(false);

  // Initialize API service with snackbar
  useEffect(() => {
    creditCheckApi.setSnackbar(enqueueSnackbar);
  }, [enqueueSnackbar]);

  useEffect(() => {
    loadCreditReport();
  }, [currentClientId]);

  const loadCreditReport = async () => {
    if (!currentClientId) {
      enqueueSnackbar('No client selected', { variant: 'warning' });
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const report = await creditCheckApi.getClientCreditReport(currentClientId);
      setCreditReport(report);
    } catch (error) {
      console.error('Failed to load credit report:', error);
      enqueueSnackbar('Failed to load BKR credit report', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const performCreditCheck = async () => {
    if (!creditReport) return;

    setChecking(true);
    try {
      const response = await creditCheckApi.refreshCreditReport(creditReport.id);
      if (response.status === 'completed' && response.report) {
        setCreditReport(response.report);
        enqueueSnackbar('BKR credit check refreshed successfully', { variant: 'success' });
      } else {
        enqueueSnackbar('Credit check is processing. Please check back later.', { variant: 'info' });
      }
    } catch (error) {
      enqueueSnackbar('Failed to refresh BKR credit check', { variant: 'error' });
    } finally {
      setChecking(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 800) return 'success';
    if (score >= 700) return 'warning';
    if (score >= 600) return 'warning';
    return 'error';
  };

  const getScoreLabel = (range: string) => {
    switch (range) {
      case 'excellent': return 'Excellent (800+)';
      case 'good': return 'Good (700-799)';
      case 'fair': return 'Fair (600-699)';
      case 'poor': return 'Poor (500-599)';
      case 'very_poor': return 'Very Poor (<500)';
      default: return 'Unknown';
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
            Loading BKR Credit Report...
          </Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 4 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          sx={{ mb: 2 }}
        >
          Back to Dashboard
        </Button>

        <Typography variant="h4" component="h1" sx={{ fontWeight: 700, mb: 2 }}>
          BKR Credit Check
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          Dutch credit scoring and mortgage eligibility assessment
        </Typography>

        {creditReport && (
          <>
            {/* Credit Score Overview */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Avatar sx={{
                        bgcolor: getScoreColor(creditReport.credit_score) + '.main',
                        mr: 2,
                        width: 48,
                        height: 48
                      }}>
                        <CreditScore />
                      </Avatar>
                      <Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          Credit Score
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          BKR Report • {new Date(creditReport.report_date).toLocaleDateString('nl-NL')}
                        </Typography>
                      </Box>
                    </Box>

                    <Box sx={{ textAlign: 'center', mb: 2 }}>
                      <Typography variant="h2" sx={{
                        fontWeight: 700,
                        color: getScoreColor(creditReport.credit_score) + '.main',
                        mb: 1
                      }}>
                        {creditReport.credit_score}
                      </Typography>
                      <Chip
                        label={getScoreLabel(creditReport.score_range)}
                        color={getScoreColor(creditReport.credit_score)}
                        sx={{ fontWeight: 600 }}
                      />
                    </Box>

                    <Button
                      fullWidth
                      variant="outlined"
                      onClick={performCreditCheck}
                      disabled={checking}
                      startIcon={checking ? <CircularProgress size={16} /> : <Refresh />}
                    >
                      {checking ? 'Checking...' : 'Refresh Credit Report'}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                      Mortgage Eligibility
                    </Typography>

                    <Box sx={{ textAlign: 'center', mb: 2 }}>
                      <Avatar sx={{
                        bgcolor: creditReport.mortgage_eligibility.eligible ? 'success.main' : 'error.main',
                        width: 48,
                        height: 48,
                        mx: 'auto',
                        mb: 2
                      }}>
                        {creditReport.mortgage_eligibility.eligible ? <CheckCircle /> : <ErrorIcon />}
                      </Avatar>

                      <Typography variant="h4" sx={{
                        fontWeight: 700,
                        color: creditReport.mortgage_eligibility.eligible ? 'success.main' : 'error.main',
                        mb: 1
                      }}>
                        {creditReport.mortgage_eligibility.eligible ? 'Eligible' : 'Not Eligible'}
                      </Typography>

                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Max Amount: €{creditReport.mortgage_eligibility.max_mortgage_amount.toLocaleString()}
                      </Typography>

                      {creditReport.mortgage_eligibility.conditions && (
                        <Box>
                          <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            Conditions:
                          </Typography>
                          <List dense>
                            {creditReport.mortgage_eligibility.conditions.map((condition, index) => (
                              <ListItem key={index}>
                                <ListItemIcon>
                                  <Warning sx={{ color: 'warning.main', fontSize: 16 }} />
                                </ListItemIcon>
                                <ListItemText
                                  primary={<Typography variant="body2">{condition}</Typography>}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Payment History */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                      Payment History
                    </Typography>

                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={4}>
                        <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                          <Typography variant="h4" sx={{ fontWeight: 700, color: 'success.dark' }}>
                            {creditReport.payment_history.on_time_payments}
                          </Typography>
                          <Typography variant="body2" color="success.dark">
                            On-Time Payments
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light' }}>
                          <Typography variant="h4" sx={{ fontWeight: 700, color: 'warning.dark' }}>
                            {creditReport.payment_history.late_payments}
                          </Typography>
                          <Typography variant="body2" color="warning.dark">
                            Late Payments
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'error.light' }}>
                          <Typography variant="h4" sx={{ fontWeight: 700, color: 'error.dark' }}>
                            {creditReport.payment_history.defaulted_loans}
                          </Typography>
                          <Typography variant="body2" color="error.dark">
                            Defaulted Loans
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Active Loans */}
            {creditReport.active_loans.length > 0 && (
              <Card sx={{ mb: 4 }}>
                <CardContent>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                    Active Loans
                  </Typography>

                  <List>
                    {creditReport.active_loans.map((loan, index) => (
                      <React.Fragment key={index}>
                        <ListItem>
                          <ListItemIcon>
                            <Assessment />
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {loan.type} - {loan.creditor}
                              </Typography>
                            }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Amount: €{loan.outstanding_amount.toLocaleString()} • Monthly: €{loan.monthly_payment}
                              </Typography>
                              <Chip
                                label={loan.status}
                                size="small"
                                color={loan.status === 'current' ? 'success' : 'warning'}
                                sx={{ mt: 1 }}
                              />
                            </Box>
                          }
                          />
                        </ListItem>
                        {index < creditReport.active_loans.length - 1 && <Divider />}
                      </React.Fragment>
                    ))}
                  </List>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            <Alert severity="info" sx={{ mb: 4 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                BKR Recommendations
              </Typography>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                {creditReport.recommendations.map((rec, index) => (
                  <li key={index}>
                    <Typography variant="body2">{rec.message}</Typography>
                  </li>
                ))}
              </ul>
            </Alert>

            {/* Action Buttons */}
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                variant="outlined"
                onClick={() => navigate('/nhg-eligibility')}
              >
                Check NHG Eligibility
              </Button>
              <Button
                variant="contained"
                onClick={() => navigate('/mortgage-application')}
                sx={{
                  background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                }}
              >
                Proceed to Application
              </Button>
            </Box>
          </>
        )}
      </Box>
    </Container>
  );
};

export default BKRCreditCheck;
