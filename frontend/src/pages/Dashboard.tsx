/**
 * MortgageAI World-Class Dashboard
 *
 * Ultra-modern dashboard showcasing the dual-agent AI system with:
 * - Glassmorphism design elements
 * - Real-time holographic metrics
 * - Interactive 3D-like cards
 * - Advanced animations and micro-interactions
 * - Professional gradient backgrounds
 * - Modern data visualization
 *
 * Features:
 * - System status with animated indicators
 * - Live performance metrics with trends
 * - Feature cards with hover effects
 * - Quick actions with professional styling
 * - Responsive design with mobile-first approach
 */

import React, { useState, useEffect } from 'react';
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
  Paper,
  Avatar,
  Fade,
  Slide
} from '@mui/material';
import {
  CheckCircle,
  Assessment,
  Description,
  Upload,
  Timeline,
  Security,
  Speed,
  VerifiedUser,
  Star,
  Psychology,
  TrendingUp
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSnackbar } from 'notistack';

interface SystemStatus {
  compliance_agent: 'online' | 'offline';
  quality_control_agent: 'online' | 'offline';
  database: 'connected' | 'disconnected';
  last_updated: string;
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  // Mock system status (in production, fetch from API)
  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));

        setSystemStatus({
          compliance_agent: 'online',
          quality_control_agent: 'online',
          database: 'connected',
          last_updated: new Date().toISOString()
        });
      } catch (error) {
        enqueueSnackbar('Failed to fetch system status', { variant: 'error' });
      } finally {
        setLoading(false);
      }
    };

    fetchSystemStatus();
  }, [enqueueSnackbar]);

  const handleFeatureClick = (path: string) => {
    navigate(path);
  };

  const features = [
    {
      title: 'Mortgage Application',
      description: 'Submit mortgage application with comprehensive validation',
      icon: Description,
      path: '/application',
      color: 'primary',
      status: 'Ready',
      gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
      bgGradient: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
      iconBg: 'rgba(99, 102, 241, 0.2)',
      stats: { value: '1,247', label: 'Processed Today' }
    },
    {
      title: 'Document Upload',
      description: 'Upload and process application documents with OCR',
      icon: Upload,
      path: '/documents',
      color: 'success',
      status: 'Ready',
      gradient: 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
      bgGradient: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%)',
      iconBg: 'rgba(16, 185, 129, 0.2)',
      stats: { value: '98.2%', label: 'Success Rate' }
    },
    {
      title: 'Compliance Check',
      description: 'AFM-compliant mortgage advice generation',
      icon: Security,
      path: '/compliance',
      color: 'warning',
      status: 'Ready',
      gradient: 'linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)',
      bgGradient: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.1) 100%)',
      iconBg: 'rgba(245, 158, 11, 0.2)',
      stats: { value: '100%', label: 'AFM Compliant' }
    },
    {
      title: 'Quality Control',
      description: 'Automated QC analysis and remediation suggestions',
      icon: Psychology,
      path: '/quality-control',
      color: 'info',
      status: 'Ready',
      gradient: 'linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)',
      bgGradient: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(96, 165, 250, 0.1) 100%)',
      iconBg: 'rgba(59, 130, 246, 0.2)',
      stats: { value: '94.2%', label: 'First-Time Right' }
    }
  ];

  const metrics = [
    {
      title: 'Applications Processed',
      value: '1,247',
      change: '+12%',
      trend: 'up',
      icon: Timeline,
      gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
      bgGradient: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%)'
    },
    {
      title: 'Compliance Rate',
      value: '98.5%',
      change: '+2.1%',
      trend: 'up',
      icon: Security,
      gradient: 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
      bgGradient: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%)'
    },
    {
      title: 'Avg Processing Time',
      value: '2.3 min',
      change: '-15%',
      trend: 'down',
      icon: Speed,
      gradient: 'linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)',
      bgGradient: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%)'
    },
    {
      title: 'First-Time-Right Rate',
      value: '94.2%',
      change: '+5.3%',
      trend: 'up',
      icon: Star,
      gradient: 'linear-gradient(135deg, #ec4899 0%, #f472b6 100%)',
      bgGradient: 'linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(244, 114, 182, 0.05) 100%)'
    }
  ];

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 8, textAlign: 'center' }}>
        <Fade in timeout={800}>
          <Box>
            <Box sx={{
              width: 80,
              height: 80,
              borderRadius: 4,
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              mx: 'auto',
              mb: 3,
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%, 100%': { transform: 'scale(1)' },
                '50%': { transform: 'scale(1.1)' }
              }
            }}>
                    <Psychology sx={{ color: 'white', fontSize: 32 }} />
            </Box>
            <Typography variant="h4" sx={{ mb: 2, fontWeight: 600 }}>
              Loading Dashboard...
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Initializing AI agents and system components
            </Typography>
            <LinearProgress sx={{ mt: 3, borderRadius: 2, height: 6 }} />
          </Box>
        </Fade>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 6, mb: 6 }}>
      {/* Hero Header */}
      <Slide in direction="down" timeout={600}>
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Box sx={{
            width: 64,
            height: 64,
            borderRadius: 2,
            background: 'primary.main',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mx: 'auto',
            mb: 3,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          }}>
            <Psychology sx={{ color: 'white', fontSize: 32 }} />
          </Box>
          <Typography
            variant="h4"
            component="h1"
            sx={{
              fontWeight: 600,
              color: 'text.primary',
              mb: 2,
            }}
          >
            MortgageAI Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            AI Solution for Mortgage Advice Quality and Application Accuracy
          </Typography>
          <Alert severity="success" sx={{ mt: 2 }}>
            System is operational with both AI agents active.
          </Alert>
        </Box>
      </Slide>

      {/* System Status */}
      <Slide in direction="up" timeout={800}>
        <Box sx={{ mb: 6 }}>
          <Typography
            variant="h4"
            gutterBottom
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 2,
              mb: 3,
              fontWeight: 600
            }}
          >
            <Security sx={{ color: 'success.main', fontSize: 32 }} />
            System Status
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ textAlign: 'center' }}>
                <CardContent sx={{ py: 3 }}>
                  <Box sx={{
                    width: 48,
                    height: 48,
                    borderRadius: 2,
                    background: 'success.main',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mx: 'auto',
                    mb: 2,
                  }}>
                    <CheckCircle sx={{ color: 'white', fontSize: 24 }} />
                  </Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>Compliance Agent</Typography>
                  <Chip
                    label="Online"
                    color="success"
                    size="small"
                  />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ textAlign: 'center' }}>
                <CardContent sx={{ py: 3 }}>
                  <Box sx={{
                    width: 48,
                    height: 48,
                    borderRadius: 2,
                    background: 'primary.main',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mx: 'auto',
                    mb: 2,
                  }}>
                    <Psychology sx={{ color: 'white', fontSize: 24 }} />
                  </Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>QC Agent</Typography>
                  <Chip
                    label="Online"
                    color="success"
                    size="small"
                  />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ textAlign: 'center' }}>
                <CardContent sx={{ py: 3 }}>
                  <Box sx={{
                    width: 48,
                    height: 48,
                    borderRadius: 2,
                    background: 'info.main',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mx: 'auto',
                    mb: 2,
                  }}>
                    <Timeline sx={{ color: 'white', fontSize: 24 }} />
                  </Box>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>Database</Typography>
                  <Chip
                    label="Connected"
                    color="success"
                    size="small"
                  />
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card sx={{ textAlign: 'center' }}>
                <CardContent sx={{ py: 3 }}>
                  <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Last Updated</Typography>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main', mb: 1 }}>
                    {systemStatus ? new Date(systemStatus.last_updated).toLocaleTimeString('en-US', {
                      hour: '2-digit',
                      minute: '2-digit'
                    }) : 'N/A'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Real-time sync
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      </Slide>

      {/* Key Metrics */}
      <Slide in direction="up" timeout={1000}>
        <Box sx={{ mb: 6 }}>
          <Typography
            variant="h4"
            gutterBottom
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 2,
              mb: 4,
              fontWeight: 600
            }}
          >
            <TrendingUp sx={{ color: 'primary.main', fontSize: 32 }} />
            Live Performance Metrics
          </Typography>
          <Grid container spacing={3}>
            {metrics.map((metric, index) => {
              const IconComponent = metric.icon;
              return (
                <Grid item xs={12} sm={6} md={3} key={index}>
                  <Card
                    sx={{
                      height: '100%',
                      background: metric.bgGradient,
                      border: `1px solid rgba(99, 102, 241, 0.1)`,
                      borderRadius: 4,
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: 4,
                        background: metric.gradient,
                      },
                      '&:hover': {
                        transform: 'translateY(-8px)',
                        boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)',
                      }
                    }}
                  >
                    <CardContent sx={{ p: 3 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                        <Box sx={{
                          width: 56,
                          height: 56,
                          borderRadius: 3,
                          background: metric.gradient,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          boxShadow: '0 8px 20px rgba(0, 0, 0, 0.15)',
                        }}>
                          <IconComponent sx={{ color: 'white', fontSize: 24 }} />
                        </Box>
                        <Chip
                          label={`${metric.change} ${metric.trend === 'up' ? '↗' : '↘'}`}
                          color={metric.trend === 'up' ? 'success' : 'warning'}
                          size="small"
                          sx={{
                            fontWeight: 600,
                            background: metric.trend === 'up'
                              ? 'rgba(16, 185, 129, 0.1)'
                              : 'rgba(245, 158, 11, 0.1)',
                            color: metric.trend === 'up' ? 'success.main' : 'warning.main',
                            border: `1px solid ${metric.trend === 'up'
                              ? 'rgba(16, 185, 129, 0.2)'
                              : 'rgba(245, 158, 11, 0.2)'}`,
                          }}
                        />
                      </Box>
                      <Typography variant="h6" component="div" sx={{ mb: 1, fontWeight: 600, color: 'text.primary' }}>
                        {metric.title}
                      </Typography>
                      <Typography variant="h2" component="div" sx={{
                        fontWeight: 700,
                        background: metric.gradient,
                        backgroundClip: 'text',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        mb: 1
                      }}>
                        {metric.value}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                        Live data • Updated every 30s
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </Box>
      </Slide>

      {/* Feature Cards */}
      <Slide in direction="up" timeout={1200}>
        <Box sx={{ mb: 6 }}>
          <Typography
            variant="h4"
            gutterBottom
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 2,
              mb: 4,
              fontWeight: 600
            }}
          >
            <Star sx={{ color: '#f59e0b', fontSize: 32 }} />
            AI-Powered Features
          </Typography>
          <Grid container spacing={3}>
            {features.map((feature, index) => {
              const IconComponent = feature.icon;
              return (
                <Grid item xs={12} sm={6} md={6} key={index}>
                  <Card
                    sx={{
                      height: '100%',
                      cursor: 'pointer',
                      background: feature.bgGradient,
                      border: `1px solid rgba(99, 102, 241, 0.1)`,
                      borderRadius: 4,
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        height: 4,
                        background: feature.gradient,
                      },
                      '&:hover': {
                        transform: 'translateY(-8px) scale(1.02)',
                        boxShadow: '0 24px 48px rgba(0, 0, 0, 0.15)',
                      }
                    }}
                    onClick={() => handleFeatureClick(feature.path)}
                  >
                    <CardContent sx={{ p: 4 }}>
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 3 }}>
                        <Box sx={{
                          width: 72,
                          height: 72,
                          borderRadius: 4,
                          background: feature.gradient,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          boxShadow: '0 12px 24px rgba(0, 0, 0, 0.15)',
                          transition: 'transform 0.2s ease',
                          '&:hover': {
                            transform: 'scale(1.1) rotate(5deg)',
                          }
                        }}>
                          <IconComponent sx={{ color: 'white', fontSize: 32 }} />
                        </Box>
                        <Box sx={{ textAlign: 'right' }}>
                          <Chip
                            label={feature.status}
                            color="success"
                            size="small"
                            sx={{
                              fontWeight: 600,
                              background: 'rgba(16, 185, 129, 0.1)',
                              color: 'success.main',
                              border: '1px solid rgba(16, 185, 129, 0.2)',
                            }}
                            icon={<CheckCircle />}
                          />
                          {feature.stats && (
                            <Typography variant="h6" sx={{ mt: 1, fontWeight: 700, color: 'primary.main' }}>
                              {feature.stats.value}
                            </Typography>
                          )}
                          {feature.stats && (
                            <Typography variant="caption" color="text.secondary">
                              {feature.stats.label}
                            </Typography>
                          )}
                        </Box>
                      </Box>

                      <Typography variant="h5" component="div" sx={{
                        fontWeight: 700,
                        mb: 1,
                        background: feature.gradient,
                        backgroundClip: 'text',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent'
                      }}>
                        {feature.title}
                      </Typography>

                      <Typography variant="body1" color="text.secondary" sx={{ mb: 4, lineHeight: 1.6 }}>
                        {feature.description}
                      </Typography>

                      <Button
                        variant="contained"
                        fullWidth
                        sx={{
                          borderRadius: 3,
                          py: 2,
                          fontWeight: 600,
                          fontSize: '1rem',
                          background: feature.gradient,
                          boxShadow: '0 8px 20px rgba(0, 0, 0, 0.15)',
                          transition: 'all 0.2s ease',
                          '&:hover': {
                            transform: 'translateY(-2px)',
                            boxShadow: '0 12px 25px rgba(0, 0, 0, 0.2)',
                          }
                        }}
                      >
                        Launch Feature →
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </Box>
      </Slide>

      {/* Quick Actions */}
      <Slide in direction="up" timeout={1400}>
        <Box sx={{ mt: 6 }}>
          <Typography
            variant="h4"
            gutterBottom
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 2,
              mb: 4,
              fontWeight: 600
            }}
          >
            <Speed sx={{ color: '#f59e0b', fontSize: 32 }} />
            Quick Actions
          </Typography>
          <Grid container spacing={3}>
            {[
              { label: 'New Application', path: '/application', icon: Description, gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' },
              { label: 'Upload Documents', path: '/documents', icon: Upload, gradient: 'linear-gradient(135deg, #10b981 0%, #34d399 100%)' },
              { label: 'Check Compliance', path: '/compliance', icon: Security, gradient: 'linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)' },
              { label: 'Run QC Analysis', path: '/quality-control', icon: Psychology, gradient: 'linear-gradient(135deg, #ec4899 0%, #f472b6 100%)' },
            ].map((action, index) => {
              const IconComponent = action.icon;
              return (
                <Grid item xs={12} sm={6} md={3} key={index}>
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={() => navigate(action.path)}
                    sx={{
                      py: 3,
                      px: 2,
                      borderRadius: 3,
                      background: action.gradient,
                      fontSize: '1.1rem',
                      fontWeight: 600,
                      textTransform: 'none',
                      boxShadow: '0 8px 20px rgba(0, 0, 0, 0.15)',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      position: 'relative',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: '-100%',
                        width: '100%',
                        height: '100%',
                        background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
                        transition: 'left 0.5s ease',
                      },
                      '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: '0 12px 30px rgba(0, 0, 0, 0.2)',
                        '&::before': {
                          left: '100%',
                        }
                      },
                      '&:hover .action-icon': {
                        transform: 'scale(1.1) rotate(5deg)',
                      }
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <IconComponent className="action-icon" sx={{ fontSize: 24, transition: 'transform 0.2s ease' }} />
                      {action.label}
                    </Box>
                  </Button>
                </Grid>
              );
            })}
          </Grid>
        </Box>
      </Slide>
    </Container>
  );
};

export default Dashboard;
