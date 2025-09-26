/**
 * Dutch Mortgage Advisory Dashboard
 *
 * Specialized dashboard for Dutch mortgage advisors showing:
 * - AFM compliance status
 * - Active client sessions
 * - Lender integration status
 * - Market insights and regulatory updates
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Chip,
  LinearProgress,
  Alert,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  IconButton,
  Tooltip,
  CircularProgress,
  Paper,
} from '@mui/material';
import {
  Gavel,
  AccountBalance,
  Assessment,
  Security,
  TrendingUp,
  CheckCircle,
  Warning,
  Person,
  Business,
  Schedule,
  Verified,
  Refresh,
  TrendingDown,
  TrendingFlat,
  Error as ErrorIcon,
  Settings as SettingsIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';
import { dashboardApi, AFMComplianceStatus, LenderStatus, ActivityItem, DashboardData } from '../services/dashboardApi';

const DutchMortgageDashboard: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize API service with snackbar
  useEffect(() => {
    dashboardApi.setSnackbar(enqueueSnackbar);
  }, [enqueueSnackbar]);

  // Load dashboard data
  const loadDashboardData = useCallback(async () => {
    try {
      setError(null);
      const data = await dashboardApi.getDashboardData();
      setDashboardData(data);
    } catch (err) {
      setError('Failed to load dashboard data. Please check your connection and try again.');
      console.error('Dashboard data loading error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Refresh dashboard data
  const refreshDashboard = useCallback(async () => {
    setRefreshing(true);
    try {
      await loadDashboardData();
      enqueueSnackbar('Dashboard refreshed successfully', { variant: 'success' });
    } catch (err) {
      // Error already handled in loadDashboardData
    } finally {
      setRefreshing(false);
    }
  }, [loadDashboardData, enqueueSnackbar]);

  // Refresh specific lender status
  const refreshLenderStatus = useCallback(async (lenderId: string) => {
    try {
      const updatedLender = await dashboardApi.refreshLenderStatus(lenderId);
      setDashboardData(prev => prev ? {
        ...prev,
        lender_statuses: prev.lender_statuses.map(lender =>
          lender.id === lenderId ? updatedLender : lender
        )
      } : null);
      enqueueSnackbar(`${updatedLender.name} status refreshed`, { variant: 'success' });
    } catch (err) {
      enqueueSnackbar(`Failed to refresh lender status`, { variant: 'error' });
    }
  }, [enqueueSnackbar]);

  // Load data on component mount
  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Quick actions configuration
  const quickActions = [
    {
      title: 'New Client Intake',
      description: 'Start AFM-compliant client assessment',
      icon: Person,
      path: '/afm-client-intake',
      color: 'primary' as const,
      gradient: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
    },
    {
      title: 'AFM Compliance Check',
      description: 'Validate advice for regulatory compliance',
      icon: Gavel,
      path: '/afm-compliance-advisor',
      color: 'success' as const,
      gradient: 'linear-gradient(135deg, #10B981 0%, #34D399 100%)',
    },
    {
      title: 'Mortgage Application',
      description: 'Process Dutch mortgage application',
      icon: Business,
      path: '/mortgage-application',
      color: 'warning' as const,
      gradient: 'linear-gradient(135deg, #F59E0B 0%, #FCD34D 100%)',
    },
    {
      title: 'Lender Integration',
      description: 'Submit to Stater, Quion, and other lenders',
      icon: AccountBalance,
      path: '/lender-integration',
      color: 'info' as const,
      gradient: 'linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%)',
    },
  ];

  // Render loading state
  if (loading) {
    return (
      <Container maxWidth="xl">
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
            Loading Dutch Mortgage Dashboard...
          </Typography>
        </Box>
      </Container>
    );
  }

  // Render error state
  if (error) {
    return (
      <Container maxWidth="xl">
        <Box sx={{ mt: 4 }}>
          <Alert
            severity="error"
            action={
              <Button color="inherit" size="small" onClick={loadDashboardData}>
                Retry
              </Button>
            }
          >
            {error}
          </Alert>
        </Box>
      </Container>
    );
  }

  const afmStatus = dashboardData?.afm_status;
  const lenderStatuses = dashboardData?.lender_statuses || [];
  const recentActivity = dashboardData?.recent_activity || [];

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 4 }}>
        <Box sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2
        }}>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 700 }}>
            Dutch Mortgage Advisory Dashboard
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Tooltip title="Refresh Dashboard">
              <IconButton
                onClick={refreshDashboard}
                disabled={refreshing}
                sx={{
                  bgcolor: 'background.paper',
                  '&:hover': { bgcolor: 'action.hover' }
                }}
              >
                {refreshing ? <CircularProgress size={20} /> : <Refresh />}
              </IconButton>
            </Tooltip>
            <Button
              variant="outlined"
              startIcon={<SettingsIcon />}
              onClick={() => navigate('/settings')}
            >
              Settings
            </Button>
          </Box>
        </Box>
        <Typography variant="body1" color="text.secondary">
          AFM-compliant mortgage advisory platform with integrated lender processing
        </Typography>
      </Box>

      {/* AFM Compliance Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Avatar sx={{
                  bgcolor: 'success.main',
                  mr: 2,
                  width: 48,
                  height: 48
                }}>
                  <Gavel />
                </Avatar>
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    AFM Compliance Status
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Real-time regulatory compliance monitoring • Last updated: {afmStatus?.last_updated ?
                      new Date(afmStatus.last_updated).toLocaleString('nl-NL') : 'Unknown'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Chip
                    label="Audit Ready"
                    color="success"
                    icon={<Verified />}
                    sx={{ fontWeight: 600 }}
                  />
                  {afmStatus?.compliance_trends && (
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {afmStatus.compliance_trends.score_change > 0 && <TrendingUp sx={{ color: 'success.main', fontSize: 16 }} />}
                      {afmStatus.compliance_trends.score_change < 0 && <TrendingDown sx={{ color: 'error.main', fontSize: 16 }} />}
                      {afmStatus.compliance_trends.score_change === 0 && <TrendingFlat sx={{ color: 'text.secondary', fontSize: 16 }} />}
                    </Box>
                  )}
                </Box>
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h3" sx={{
                      fontWeight: 700,
                      background: 'linear-gradient(135deg, #10B981 0%, #34D399 100%)',
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      mb: 1
                    }}>
                      {afmStatus?.overall_score.toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Overall Compliance
                    </Typography>
                    {afmStatus?.compliance_trends && (
                      <Typography
                        variant="caption"
                        sx={{
                          color: afmStatus.compliance_trends.score_change >= 0 ? 'success.main' : 'error.main',
                          fontWeight: 600
                        }}
                      >
                        {afmStatus.compliance_trends.score_change > 0 ? '+' : ''}
                        {afmStatus.compliance_trends.score_change.toFixed(1)}% this month
                      </Typography>
                    )}
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h3" sx={{
                      fontWeight: 700,
                      color: 'primary.main',
                      mb: 1
                    }}>
                      {afmStatus?.active_sessions}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Active Sessions
                    </Typography>
                    {afmStatus?.compliance_trends && (
                      <Typography
                        variant="caption"
                        sx={{
                          color: afmStatus.compliance_trends.sessions_trend >= 0 ? 'success.main' : 'warning.main',
                          fontWeight: 600
                        }}
                      >
                        {afmStatus.compliance_trends.sessions_trend > 0 ? '+' : ''}
                        {afmStatus.compliance_trends.sessions_trend} this week
                      </Typography>
                    )}
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h3" sx={{
                      fontWeight: 700,
                      color: 'warning.main',
                      mb: 1
                    }}>
                      {afmStatus?.pending_reviews}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Pending Reviews
                    </Typography>
                    {afmStatus?.compliance_trends && (
                      <Typography
                        variant="caption"
                        sx={{
                          color: afmStatus.compliance_trends.reviews_trend <= 0 ? 'success.main' : 'warning.main',
                          fontWeight: 600
                        }}
                      >
                        {afmStatus.compliance_trends.reviews_trend > 0 ? '+' : ''}
                        {afmStatus.compliance_trends.reviews_trend} this week
                      </Typography>
                    )}
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h3" sx={{
                      fontWeight: 700,
                      color: 'success.main',
                      mb: 1
                    }}>
                      {afmStatus?.audit_ready}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Audit Ready
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quick Actions */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12}>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 3 }}>
            Quick Actions
          </Typography>
        </Grid>

        {quickActions.map((action, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{
              cursor: 'pointer',
              height: '100%',
              background: `linear-gradient(135deg, ${action.gradient})`,
              color: 'white',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              '&:hover': {
                transform: 'translateY(-4px)',
                boxShadow: '0 20px 40px rgba(0,0,0,0.15)',
              }
            }} onClick={() => navigate(action.path)}>
              <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <action.icon sx={{ fontSize: 48, mr: 2 }} />
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  {action.title}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9, flexGrow: 1 }}>
                  {action.description}
                </Typography>
                <Button
                  variant="outlined"
                  sx={{
                    mt: 2,
                    borderColor: 'rgba(255,255,255,0.5)',
                    color: 'white',
                    '&:hover': {
                      borderColor: 'white',
                      backgroundColor: 'rgba(255,255,255,0.1)',
                    }
                  }}
                >
                  Start Process
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container spacing={3}>
        {/* Lender Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  Lender Integration Status
                </Typography>
                <Tooltip title="Refresh all lender statuses">
                  <IconButton
                    onClick={refreshDashboard}
                    disabled={refreshing}
                    size="small"
                  >
                    <Refresh />
                  </IconButton>
                </Tooltip>
              </Box>

              <List>
                {lenderStatuses.map((lender, index) => (
                  <React.Fragment key={lender.id}>
                    <ListItem>
                      <ListItemIcon>
                        <Avatar sx={{
                          bgcolor: lender.status === 'online' ? 'success.main' :
                                  lender.status === 'maintenance' ? 'warning.main' : 'error.main',
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
                              {lender.name}
                            </Typography>
                            <Chip
                              label={lender.status}
                              size="small"
                              color={lender.status === 'online' ? 'success' :
                                     lender.status === 'maintenance' ? 'warning' : 'error'}
                            />
                            {lender.authentication_status !== 'valid' && (
                              <Tooltip title={`Authentication: ${lender.authentication_status}`}>
                                <ErrorIcon sx={{ fontSize: 16, color: 'error.main' }} />
                              </Tooltip>
                            )}
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              Processing: {lender.processing_time} • Success: {lender.success_rate.toFixed(1)}%
                            </Typography>
                            {lender.active_applications > 0 && (
                              <Typography variant="caption" sx={{ color: 'primary.main', fontWeight: 500 }}>
                                {lender.active_applications} active application{lender.active_applications !== 1 ? 's' : ''}
                              </Typography>
                            )}
                          </Box>
                        }
                      />
                      <Tooltip title={`Last successful submission: ${lender.last_successful_submission ?
                        new Date(lender.last_successful_submission).toLocaleString('nl-NL') : 'Never'}`}>
                        <IconButton
                          size="small"
                          onClick={() => refreshLenderStatus(lender.id)}
                        >
                          <InfoIcon />
                        </IconButton>
                      </Tooltip>
                    </ListItem>
                    {index < lenderStatuses.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>

              <Button
                fullWidth
                variant="outlined"
                sx={{ mt: 2 }}
                onClick={() => navigate('/lender-integration')}
              >
                Manage Lender Integrations
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>
                Recent Activity
              </Typography>

              <List>
                {recentActivity.map((activity, index) => (
                  <React.Fragment key={activity.id}>
                    <ListItem>
                      <ListItemIcon>
                        <Avatar sx={{
                          bgcolor: activity.status === 'completed' ? 'success.main' : 'primary.main',
                          width: 32,
                          height: 32
                        }}>
                          {activity.type === 'compliance_check' && <Gavel />}
                          {activity.type === 'application_submitted' && <Business />}
                          {activity.type === 'bkr_check' && <Assessment />}
                          {activity.type === 'nhg_eligibility' && <Security />}
                          {activity.type === 'client_intake' && <Person />}
                        </Avatar>
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                            {activity.client_name}
                            {activity.lender_name && ` → ${activity.lender_name}`}
                          </Typography>
                        }
                        secondary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                            <Typography variant="body2" color="text.secondary">
                              {activity.type.replace('_', ' ')} • {new Date(activity.timestamp).toLocaleString('nl-NL')}
                            </Typography>
                            <Chip
                              label={activity.status}
                              size="small"
                              color="success"
                            />
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < recentActivity.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>

              <Button
                fullWidth
                variant="outlined"
                sx={{ mt: 2 }}
                onClick={() => navigate('/compliance-audit')}
              >
                View Full Activity Log
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Market Insights */}
      {dashboardData?.market_indicators && (
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3, bgcolor: 'background.paper' }}>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Market Indicators
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: 'primary.main' }}>
                      {dashboardData.market_indicators.total_applications_this_month}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Applications This Month
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: 'success.main' }}>
                      {dashboardData.market_indicators.compliance_rate.toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Compliance Rate
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: 'warning.main' }}>
                      {Math.round(dashboardData.market_indicators.average_processing_time)} days
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Avg. Processing Time
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 700, color: 'info.main' }}>
                      {dashboardData.market_indicators.lender_availability_score.toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Lender Availability
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default DutchMortgageDashboard;
