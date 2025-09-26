/**
 * Lender Integration
 *
 * Manages integration with Dutch mortgage lenders (Stater, Quion, ING, ABN AMRO)
 * Provides lender selection, API status, and application submission
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
  Switch,
  FormControlLabel,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  AccountBalance,
  CheckCircle,
  Error as ErrorIcon,
  Settings,
  ArrowBack,
  Send,
  Refresh,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { lenderApi, Lender } from '../services/lenderApi';

interface IntegrationSettings {
  auto_submit: boolean;
  preferred_lenders: string[];
  notification_settings: {
    email: boolean;
    dashboard: boolean;
  };
}

const LenderIntegration: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const [lenders, setLenders] = useState<Lender[]>([]);
  const [settings, setSettings] = useState<IntegrationSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [settingsDialog, setSettingsDialog] = useState(false);

  useEffect(() => {
    lenderApi.setSnackbar(enqueueSnackbar);
    loadLenderData();
  }, [enqueueSnackbar]);

  const loadLenderData = async () => {
    try {
      setLoading(true);
      const lendersData = await lenderApi.getLenders();
      setLenders(lendersData);

      // Integration settings - in production this would come from a separate API
      const integrationSettings: IntegrationSettings = {
        auto_submit: true,
        preferred_lenders: lendersData.filter(l => l.enabled).map(l => l.id),
        notification_settings: { email: true, dashboard: true },
      };
      setSettings(integrationSettings);
    } catch (error) {
      console.error('Failed to load lender data:', error);
      enqueueSnackbar('Failed to load lender integration data', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const toggleLender = async (lenderId: string, enabled: boolean) => {
    try {
      // In production, this would update lender settings via API
      setLenders(prev => prev.map(lender =>
        lender.id === lenderId ? { ...lender, enabled } : lender
      ));
      enqueueSnackbar(`${enabled ? 'Enabled' : 'Disabled'} integration with lender`, { variant: 'success' });
    } catch (error) {
      enqueueSnackbar('Failed to update lender settings', { variant: 'error' });
    }
  };

  const syncLender = async (lenderId: string) => {
    setSyncing(true);
    try {
      // In production, this would sync with lender API
      await new Promise(resolve => setTimeout(resolve, 2000));
      setLenders(prev => prev.map(lender =>
        lender.id === lenderId
          ? { ...lender, last_sync: new Date().toISOString(), api_status: 'connected' as const }
          : lender
      ));
      enqueueSnackbar('Lender sync completed', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar('Lender sync failed', { variant: 'error' });
    } finally {
      setSyncing(false);
    }
  };

  const submitToLenders = async () => {
    const enabledLenders = lenders.filter(l => l.enabled);
    if (enabledLenders.length === 0) {
      enqueueSnackbar('Please enable at least one lender', { variant: 'warning' });
      return;
    }

    try {
      // In production, this would submit applications to enabled lenders
      await new Promise(resolve => setTimeout(resolve, 3000));
      enqueueSnackbar(`Applications submitted to ${enabledLenders.length} lenders`, { variant: 'success' });
      navigate('/mortgage-application');
    } catch (error) {
      enqueueSnackbar('Failed to submit applications', { variant: 'error' });
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
            Loading Lender Integrations...
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

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 700 }}>
              Lender Integration
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Manage connections with Dutch mortgage lenders
            </Typography>
          </Box>
          <Button
            variant="outlined"
            startIcon={<Settings />}
            onClick={() => setSettingsDialog(true)}
          >
            Settings
          </Button>
        </Box>

        {/* Lender Status Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {lenders.map((lender) => (
            <Grid item xs={12} md={6} key={lender.id}>
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Avatar sx={{ mr: 2, fontSize: 24 }}>
                        {lender.logo}
                      </Avatar>
                      <Box>
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
                          {lender.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {lender.processing_time} processing • {lender.success_rate}% success rate
                        </Typography>
                      </Box>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip
                        label={lender.status}
                        size="small"
                        color={lender.status === 'active' ? 'success' :
                               lender.status === 'maintenance' ? 'warning' : 'error'}
                      />
                      <Chip
                        label={lender.api_status}
                        size="small"
                        variant="outlined"
                        color={lender.api_status === 'connected' ? 'success' : 'error'}
                      />
                    </Box>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      Supported Products:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      {lender.supported_products.map((product, index) => (
                        <Chip key={index} label={product} size="small" variant="outlined" />
                      ))}
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={lender.enabled}
                          onChange={(e) => toggleLender(lender.id, e.target.checked)}
                          disabled={lender.status === 'maintenance'}
                        />
                      }
                      label="Enable Integration"
                    />
                    <Button
                      size="small"
                      onClick={() => syncLender(lender.id)}
                      disabled={syncing}
                      startIcon={syncing ? <CircularProgress size={16} /> : <Refresh />}
                    >
                      Sync
                    </Button>
                  </Box>

                  <Typography variant="caption" color="text.secondary">
                    Last sync: {new Date(lender.last_sync).toLocaleString('nl-NL')}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Integration Summary */}
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
              Integration Summary
            </Typography>

            <Grid container spacing={2}>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: 'success.dark' }}>
                    {lenders.filter(l => l.enabled).length}
                  </Typography>
                  <Typography variant="body2" color="success.dark">
                    Active Lenders
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.light' }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: 'info.dark' }}>
                    {lenders.filter(l => l.api_status === 'connected').length}
                  </Typography>
                  <Typography variant="body2" color="info.dark">
                    Connected APIs
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light' }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: 'warning.dark' }}>
                    {Math.round(lenders.reduce((acc, l) => acc + l.success_rate, 0) / lenders.length)}%
                  </Typography>
                  <Typography variant="body2" color="warning.dark">
                    Avg Success Rate
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'primary.light' }}>
                  <Typography variant="h4" sx={{ fontWeight: 700, color: 'primary.dark' }}>
                    {settings?.auto_submit ? 'ON' : 'OFF'}
                  </Typography>
                  <Typography variant="body2" color="primary.dark">
                    Auto Submit
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button
            variant="outlined"
            onClick={() => navigate('/mortgage-application')}
          >
            View Applications
          </Button>
          <Button
            variant="contained"
            onClick={submitToLenders}
            startIcon={<Send />}
            sx={{
              background: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
            }}
          >
            Submit to Enabled Lenders
          </Button>
        </Box>

        {/* Settings Dialog */}
        <Dialog open={settingsDialog} onClose={() => setSettingsDialog(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Integration Settings</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings?.auto_submit || false}
                    onChange={(e) => setSettings(prev => prev ? { ...prev, auto_submit: e.target.checked } : null)}
                  />
                }
                label="Auto-submit applications to enabled lenders"
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1, mb: 2 }}>
                When enabled, applications will be automatically submitted to all enabled lenders after AFM compliance approval.
              </Typography>

              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Notification Settings
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings?.notification_settings.email || false}
                    onChange={(e) => setSettings(prev => prev ? {
                      ...prev,
                      notification_settings: { ...prev.notification_settings, email: !prev.notification_settings.email }
                    } : null)}
                  />
                }
                label="Email notifications for lender updates"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings?.notification_settings.dashboard || false}
                    onChange={(e) => setSettings(prev => prev ? {
                      ...prev,
                      notification_settings: { ...prev.notification_settings, dashboard: !prev.notification_settings.dashboard }
                    } : null)}
                  />
                }
                label="Dashboard notifications for lender status changes"
              />
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSettingsDialog(false)}>Cancel</Button>
            <Button onClick={() => setSettingsDialog(false)} variant="contained">
              Save Settings
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Container>
  );
};

export default LenderIntegration;
