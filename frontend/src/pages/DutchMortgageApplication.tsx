/**
 * Dutch Mortgage Application
 *
 * Processes Dutch mortgage applications with AFM compliance
 * Integrates with multiple lenders and NHG requirements
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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Divider,
} from '@mui/material';
import {
  Business,
  AccountBalance,
  CheckCircle,
  ArrowBack,
  Send,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { applicationApi, ApplicationData } from '../services/applicationApi';
import { useClient } from '../contexts/ClientContext';


const DutchMortgageApplication: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const { currentApplicationId } = useClient();

  const [applicationData, setApplicationData] = useState<ApplicationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  // Initialize API service with snackbar
  useEffect(() => {
    applicationApi.setSnackbar(enqueueSnackbar);
  }, [enqueueSnackbar]);

  useEffect(() => {
    loadApplicationData();
  }, [currentApplicationId]);

  const loadApplicationData = async () => {
    if (!currentApplicationId) {
      enqueueSnackbar('No application selected', { variant: 'warning' });
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const data = await applicationApi.getApplication(currentApplicationId);
      setApplicationData(data);
    } catch (error) {
      console.error('Failed to load application data:', error);
      enqueueSnackbar('Failed to load application data', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const submitToLenders = async () => {
    if (!applicationData) return;

    const enabledLenders = applicationData.applications.filter(app => app.status !== 'cancelled');
    if (enabledLenders.length === 0) {
      enqueueSnackbar('No active applications found', { variant: 'warning' });
      return;
    }

    setSubmitting(true);
    try {
      const result = await applicationApi.submitApplication(
        applicationData.id,
        enabledLenders.map(app => app.lender_id)
      );
      if (result.success) {
        enqueueSnackbar(`Applications submitted to ${result.submissions.length} lenders`, { variant: 'success' });
        // Refresh data
        loadApplicationData();
      }
    } catch (error) {
      enqueueSnackbar('Failed to submit applications', { variant: 'error' });
    } finally {
      setSubmitting(false);
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
            Loading Mortgage Application...
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
          Dutch Mortgage Application
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          Process mortgage applications across multiple Dutch lenders
        </Typography>

        {applicationData && (
          <>
            {/* Application Overview */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                      Property Details
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {applicationData.property_details.address}, {applicationData.property_details.postal_code} {applicationData.property_details.city}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Value: €{applicationData.property_details.value.toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Type: {applicationData.property_details.property_type}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                      Mortgage Requirements
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Amount: €{applicationData.mortgage_requirements.amount.toLocaleString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Term: {applicationData.mortgage_requirements.term_years} years
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Type: {applicationData.mortgage_requirements.interest_type}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Application Status */}
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                  Application Status
                </Typography>

                <List>
                  {applicationData.applications.map((app, index) => (
                    <React.Fragment key={app.id}>
                      <ListItem>
                        <ListItemIcon>
                          <Avatar sx={{
                            bgcolor: app.status === 'approved' ? 'success.main' :
                                     app.status === 'submitted' || app.status === 'under_review' ? 'warning.main' :
                                     app.status === 'offer_received' ? 'info.main' : 'error.main',
                            width: 32,
                            height: 32
                          }}>
                            <AccountBalance />
                          </Avatar>
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {app.lender_name}
                              </Typography>
                              <Chip
                                label={app.status.replace('_', ' ')}
                                size="small"
                                color={app.status === 'approved' ? 'success' :
                                       app.status === 'submitted' || app.status === 'under_review' ? 'warning' :
                                       app.status === 'offer_received' ? 'info' : 'error'}
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Current Step: {app.current_stage}
                              </Typography>
                              {app.estimated_completion && (
                                <Typography variant="caption">
                                  Estimated completion: {new Date(app.estimated_completion).toLocaleDateString('nl-NL')}
                                </Typography>
                              )}
                              <Box sx={{ mt: 1 }}>
                                <Typography variant="caption" sx={{ mr: 1 }}>
                                  Progress: {app.progress_percentage}%
                                </Typography>
                                <Box sx={{ width: '100%', bgcolor: 'grey.200', borderRadius: 1, height: 4 }}>
                                  <Box
                                    sx={{
                                      width: `${app.progress_percentage}%`,
                                      bgcolor: 'primary.main',
                                      borderRadius: 1,
                                      height: 4
                                    }}
                                  />
                                </Box>
                              </Box>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < applicationData.applications.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button
                variant="outlined"
                onClick={() => navigate('/lender-integration')}
                startIcon={<AccountBalance />}
              >
                Manage Lenders
              </Button>
              <Button
                variant="contained"
                onClick={submitToLenders}
                disabled={submitting}
                startIcon={submitting ? <CircularProgress size={16} /> : <Send />}
                sx={{
                  background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
                }}
              >
                {submitting ? 'Submitting...' : 'Submit to Lenders'}
              </Button>
            </Box>
          </>
        )}
      </Box>
    </Container>
  );
};

export default DutchMortgageApplication;
