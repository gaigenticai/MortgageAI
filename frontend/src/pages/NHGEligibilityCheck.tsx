/**
 * NHG Eligibility Check
 *
 * Assesses eligibility for Dutch National Mortgage Guarantee (NHG)
 * Provides NHG benefits and requirements analysis
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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  Security,
  CheckCircle,
  Error as ErrorIcon,
  Euro,
  ArrowBack,
  Calculate,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { nhgApi, NHGEligibility } from '../services/nhgApi';
import { useClient } from '../contexts/ClientContext';


const NHGEligibilityCheck: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const { currentClientId } = useClient();

  const [nhgEligibility, setNhgEligibility] = useState<NHGEligibility | null>(null);
  const [loading, setLoading] = useState(true);
  const [calculating, setCalculating] = useState(false);

  // Initialize API service with snackbar
  useEffect(() => {
    nhgApi.setSnackbar(enqueueSnackbar);
  }, [enqueueSnackbar]);

  useEffect(() => {
    checkNHGEligibility();
  }, [currentClientId]);

  const checkNHGEligibility = async () => {
    if (!currentClientId) {
      enqueueSnackbar('No client selected', { variant: 'warning' });
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const eligibility = await nhgApi.getClientNHGEligibility(currentClientId);
      setNhgEligibility(eligibility);
    } catch (error) {
      console.error('Failed to check NHG eligibility:', error);
      enqueueSnackbar('Failed to check NHG eligibility', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const calculateNHG = async () => {
    if (!nhgEligibility) return;

    setCalculating(true);
    try {
      // In production, this would use updated client data for recalculation
      // For now, just refresh the current eligibility
      await new Promise(resolve => setTimeout(resolve, 1500));
      await checkNHGEligibility();
      enqueueSnackbar('NHG calculation updated', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar('NHG calculation failed', { variant: 'error' });
    } finally {
      setCalculating(false);
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
            Checking NHG Eligibility...
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
          NHG Eligibility Check
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          Dutch National Mortgage Guarantee assessment and benefits analysis
        </Typography>

        {nhgEligibility && (
          <>
            {/* Eligibility Overview */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Avatar sx={{
                        bgcolor: nhgEligibility.eligible ? 'success.main' : 'error.main',
                        mr: 2,
                        width: 48,
                        height: 48
                      }}>
                        {nhgEligibility.eligible ? <CheckCircle /> : <ErrorIcon />}
                      </Avatar>
                      <Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          NHG Eligibility
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          National Mortgage Guarantee
                        </Typography>
                      </Box>
                    </Box>

                    <Box sx={{ textAlign: 'center', mb: 2 }}>
                      <Typography variant="h3" sx={{
                        fontWeight: 700,
                        color: nhgEligibility.eligible ? 'success.main' : 'error.main',
                        mb: 1
                      }}>
                        {nhgEligibility.eligible ? 'Eligible' : 'Not Eligible'}
                      </Typography>

                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        Mortgage Amount: €{nhgEligibility.mortgage_amount.toLocaleString()}
                      </Typography>

                      <Typography variant="body2" color="text.secondary">
                        NHG Costs: €{nhgEligibility.nhg_costs.toLocaleString()}
                      </Typography>
                    </Box>

                    <Button
                      fullWidth
                      variant="outlined"
                      onClick={calculateNHG}
                      disabled={calculating}
                      startIcon={calculating ? <CircularProgress size={16} /> : <Calculate />}
                    >
                      {calculating ? 'Calculating...' : 'Recalculate NHG'}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                      NHG Benefits
                    </Typography>

                    <List>
                      {nhgEligibility.benefits.map((benefit, index) => (
                        <ListItem key={index}>
                          <ListItemIcon>
                            <Euro sx={{ color: 'success.main' }} />
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {benefit.type}
                              </Typography>
                            }
                            secondary={
                              <Box>
                                <Typography variant="body2" color="text.secondary">
                                  {benefit.description}
                                </Typography>
                                {benefit.value && (
                                  <Typography variant="caption" sx={{ color: 'success.main', fontWeight: 600 }}>
                                    {benefit.value}
                                  </Typography>
                                )}
                              </Box>
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Requirements Check */}
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                  NHG Requirements Assessment
                </Typography>

                <List>
                  {nhgEligibility.requirements.map((req, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <Avatar sx={{
                          bgcolor: req.met ? 'success.main' : 'error.main',
                          width: 24,
                          height: 24
                        }}>
                          {req.met ? <CheckCircle sx={{ fontSize: 16 }} /> : <ErrorIcon sx={{ fontSize: 16 }} />}
                        </Avatar>
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                              {req.requirement}
                            </Typography>
                            <Chip
                              label={req.met ? 'Met' : 'Not Met'}
                              size="small"
                              color={req.met ? 'success' : 'error'}
                            />
                          </Box>
                        }
                        secondary={
                          req.details && (
                            <Typography variant="body2" color="text.secondary">
                              {req.details}
                            </Typography>
                          )
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>

            {/* Recommendations */}
            <Alert severity="info" sx={{ mb: 4 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                NHG Recommendations
              </Typography>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                {nhgEligibility.recommendations.map((rec, index) => (
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
                onClick={() => navigate('/bkr-credit-check')}
              >
                Back to Credit Check
              </Button>
              <Button
                variant="contained"
                onClick={() => navigate('/mortgage-application')}
                disabled={!nhgEligibility.eligible}
                sx={{
                  background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                }}
              >
                Proceed with NHG
              </Button>
            </Box>
          </>
        )}
      </Box>
    </Container>
  );
};

export default NHGEligibilityCheck;
