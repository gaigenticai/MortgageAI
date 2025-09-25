/**
 * MortgageAI Dashboard
 *
 * Professional dashboard providing overview of the dual-agent system:
 * - Compliance & Plain-Language Advisor Agent
 * - Mortgage Application Quality Control Agent
 *
 * Features:
 * - System status overview
 * - Quick access to features
 * - Real-time metrics
 * - Professional UI design
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
  Paper
} from '@mui/material';
import {
  CheckCircle,
  Assessment,
  Description,
  Upload,
  Timeline,
  Security,
  Speed,
  VerifiedUser
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
      status: 'Ready'
    },
    {
      title: 'Document Upload',
      description: 'Upload and process application documents with OCR',
      icon: Upload,
      path: '/documents',
      color: 'secondary',
      status: 'Ready'
    },
    {
      title: 'Compliance Check',
      description: 'AFM-compliant mortgage advice generation',
      icon: CheckCircle,
      path: '/compliance',
      color: 'success',
      status: 'Ready'
    },
    {
      title: 'Quality Control',
      description: 'Automated QC analysis and remediation suggestions',
      icon: Assessment,
      path: '/quality-control',
      color: 'warning',
      status: 'Ready'
    }
  ];

  const metrics = [
    {
      title: 'Applications Processed',
      value: '1,247',
      change: '+12%',
      trend: 'up',
      icon: Timeline
    },
    {
      title: 'Compliance Rate',
      value: '98.5%',
      change: '+2.1%',
      trend: 'up',
      icon: Security
    },
    {
      title: 'Avg Processing Time',
      value: '2.3 min',
      change: '-15%',
      trend: 'down',
      icon: Speed
    },
    {
      title: 'First-Time-Right Rate',
      value: '94.2%',
      change: '+5.3%',
      trend: 'up',
      icon: VerifiedUser
    }
  ];

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Loading Dashboard...
        </Typography>
        <LinearProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          MortgageAI Dashboard
        </Typography>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          Agentic AI Solution for Enhancing Mortgage Advice Quality and Application Accuracy
        </Typography>
        <Alert severity="success" sx={{ mt: 2 }}>
          System is fully operational with both Compliance and Quality Control agents active.
        </Alert>
      </Box>

      {/* System Status */}
      <Paper elevation={2} sx={{ p: 3, mb: 4, borderRadius: 2 }}>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Security />
          System Status
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 1 }} />
              <Typography variant="h6">Compliance Agent</Typography>
              <Chip
                label="Online"
                color="success"
                size="small"
                sx={{ mt: 1 }}
              />
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Assessment sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6">QC Agent</Typography>
              <Chip
                label="Online"
                color="success"
                size="small"
                sx={{ mt: 1 }}
              />
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Timeline sx={{ fontSize: 48, color: 'info.main', mb: 1 }} />
              <Typography variant="h6">Database</Typography>
              <Chip
                label="Connected"
                color="success"
                size="small"
                sx={{ mt: 1 }}
              />
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" sx={{ mb: 1 }}>Last Updated</Typography>
              <Typography variant="body2" color="text.secondary">
                {systemStatus ? new Date(systemStatus.last_updated).toLocaleTimeString() : 'N/A'}
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Key Metrics */}
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Key Performance Metrics
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {metrics.map((metric, index) => {
          const IconComponent = metric.icon;
          return (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card sx={{ height: '100%', borderRadius: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <IconComponent sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6" component="div">
                      {metric.title}
                    </Typography>
                  </Box>
                  <Typography variant="h3" component="div" sx={{ mb: 1, fontWeight: 'bold' }}>
                    {metric.value}
                  </Typography>
                  <Chip
                    label={`${metric.change} ${metric.trend === 'up' ? '↑' : '↓'}`}
                    color={metric.trend === 'up' ? 'success' : 'error'}
                    size="small"
                  />
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Feature Cards */}
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Available Features
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
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: (theme) => theme.shadows[8],
                  },
                  borderRadius: 2
                }}
                onClick={() => handleFeatureClick(feature.path)}
              >
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <IconComponent
                      sx={{
                        fontSize: 40,
                        color: `${feature.color}.main`,
                        mr: 2
                      }}
                    />
                    <Box>
                      <Typography variant="h5" component="div" sx={{ fontWeight: 'bold' }}>
                        {feature.title}
                      </Typography>
                      <Chip
                        label={feature.status}
                        color="success"
                        size="small"
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  </Box>
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                    {feature.description}
                  </Typography>
                  <Button
                    variant="contained"
                    color={feature.color as any}
                    fullWidth
                    sx={{ borderRadius: 2 }}
                  >
                    Access Feature
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Quick Actions */}
      <Paper elevation={2} sx={{ p: 3, mt: 4, borderRadius: 2 }}>
        <Typography variant="h5" gutterBottom>
          Quick Actions
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              variant="outlined"
              fullWidth
              onClick={() => navigate('/application')}
              sx={{ borderRadius: 2, py: 2 }}
            >
              New Application
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              variant="outlined"
              fullWidth
              onClick={() => navigate('/documents')}
              sx={{ borderRadius: 2, py: 2 }}
            >
              Upload Documents
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              variant="outlined"
              fullWidth
              onClick={() => navigate('/compliance')}
              sx={{ borderRadius: 2, py: 2 }}
            >
              Check Compliance
            </Button>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Button
              variant="outlined"
              fullWidth
              onClick={() => navigate('/quality-control')}
              sx={{ borderRadius: 2, py: 2 }}
            >
              Run QC Analysis
            </Button>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default Dashboard;
