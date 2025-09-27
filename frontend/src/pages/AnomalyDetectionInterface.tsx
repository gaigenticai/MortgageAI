/**
 * Anomaly Detection Interface - React Component
 * Created: 2024-01-15
 * Author: MortgageAI Development Team
 * 
 * Professional React interface for the anomaly detection system with real-time
 * pattern recognition, alert management, and investigation tools.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Alert,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Divider,
  IconButton,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Badge,
  Snackbar
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Search as SearchIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  TrendingUp as TrendingUpIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
  Visibility as VisibilityIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
  FilterList as FilterListIcon,
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  NotificationsActive as NotificationsActiveIcon,
  Security as SecurityIcon,
  Insights as InsightsIcon
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  TimeScale,
  Filler
} from 'chart.js';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';

// Chart.js registration
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  TimeScale,
  Title,
  ChartTooltip,
  Legend,
  Filler
);

// Type definitions
interface AnomalyRecord {
  anomaly_id: string;
  detection_type: string;
  anomaly_category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence_score: number;
  anomaly_score: number;
  title: string;
  description: string;
  affected_entities: string[];
  detection_method: string;
  detection_timestamp: string;
  data_timestamp: string;
  status: 'detected' | 'investigating' | 'resolved' | 'false_positive' | 'archived';
  investigation_priority: number;
  recommended_actions: string[];
  investigation_hints: string[];
  related_anomalies: string[];
  metadata: Record<string, any>;
  tags: string[];
}

interface AlertRule {
  rule_id: string;
  rule_name: string;
  rule_type: string;
  category: string;
  conditions: Record<string, any>;
  thresholds: Record<string, any>;
  severity_mapping: Record<string, any>;
  is_active: boolean;
  trigger_count: number;
  false_positive_count: number;
  accuracy_score: number;
  created_at: string;
  last_triggered: string | null;
}

interface InvestigationSession {
  session_id: string;
  anomaly_ids: string[];
  investigator_id: string;
  session_name: string;
  investigation_status: 'active' | 'paused' | 'completed' | 'cancelled';
  priority_level: 'low' | 'medium' | 'high' | 'urgent';
  hypothesis: string[];
  evidence_collected: any[];
  findings: string[];
  started_at: string;
  last_activity_at: string;
  completed_at: string | null;
  hours_spent: number;
}

interface DetectionSummary {
  total_anomalies: number;
  severity_distribution: Record<string, number>;
  method_distribution: Record<string, number>;
  confidence_stats: {
    mean_confidence: number;
    min_confidence: number;
    max_confidence: number;
  };
  data_processed: {
    records: number;
    features: number;
  };
}

const AnomalyDetectionInterface: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [anomalies, setAnomalies] = useState<AnomalyRecord[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [investigations, setInvestigations] = useState<InvestigationSession[]>([]);
  const [detectionSummary, setDetectionSummary] = useState<DetectionSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [realTimeDetection, setRealTimeDetection] = useState(true);
  
  // Dialog states
  const [selectedAnomaly, setSelectedAnomaly] = useState<AnomalyRecord | null>(null);
  const [anomalyDetailsDialog, setAnomalyDetailsDialog] = useState(false);
  const [ruleDialog, setRuleDialog] = useState(false);
  const [investigationDialog, setInvestigationDialog] = useState(false);
  const [newRuleDialog, setNewRuleDialog] = useState(false);
  
  // Filter states
  const [severityFilter, setSeverityFilter] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [categoryFilter, setCategoryFilter] = useState<string>('');
  const [dateFilter, setDateFilter] = useState<string>('24h');
  
  // Form states
  const [detectionData, setDetectionData] = useState<string>('');
  const [detectionMethods, setDetectionMethods] = useState<string[]>(['statistical', 'ml_based']);
  const [severityThreshold, setSeverityThreshold] = useState<string>('medium');
  
  // New rule form state
  const [newRule, setNewRule] = useState({
    rule_name: '',
    rule_type: 'threshold',
    category: '',
    conditions: '{}',
    thresholds: '{}',
    severity_mapping: '{}',
    notification_channels: ['in_app']
  });

  // API calls
  const apiCall = useCallback(async (endpoint: string, options: RequestInit = {}) => {
    const baseUrl = process.env.REACT_APP_API_URL || '';
    const response = await fetch(`${baseUrl}/api/anomaly-detection/${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.statusText}`);
    }

    return response.json();
  }, []);

  // Load initial data
  useEffect(() => {
    const loadInitialData = async () => {
      setLoading(true);
      try {
        await Promise.all([
          loadAnomalySummary(),
          loadAlertRules(),
          loadInvestigations()
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, []);

  // Load anomaly summary
  const loadAnomalySummary = useCallback(async () => {
    try {
      const response = await apiCall(`summary?time_period=${dateFilter}&severity=${severityFilter}&category=${categoryFilter}`);
      if (response.success) {
        setDetectionSummary(response.summary);
      } else {
        throw new Error(response.error || 'Failed to load anomaly summary');
      }
    } catch (err) {
      console.error('Failed to load anomaly summary:', err);
      setError(err instanceof Error ? err.message : 'Failed to load anomaly summary');
    }
  }, [dateFilter, severityFilter, categoryFilter, apiCall]);

  // Load alert rules
  const loadAlertRules = useCallback(async () => {
    try {
      const response = await apiCall('rules');
      if (response.success) {
        setAlertRules(response.rules || []);
      } else {
        throw new Error(response.error || 'Failed to load alert rules');
      }
    } catch (err) {
      console.error('Failed to load alert rules:', err);
      setError(err instanceof Error ? err.message : 'Failed to load alert rules');
    }
  }, [apiCall]);

  // Load investigations
  const loadInvestigations = useCallback(async () => {
    try {
      const response = await apiCall('investigations');
      if (response.success) {
        setInvestigations(response.investigations || []);
      } else {
        throw new Error(response.error || 'Failed to load investigations');
      }
    } catch (err) {
      console.error('Failed to load investigations:', err);
      setError(err instanceof Error ? err.message : 'Failed to load investigations');
    }
  }, [apiCall]);

  // Detect anomalies
  const detectAnomalies = useCallback(async () => {
    if (!detectionData.trim()) {
      setError('Please provide data for anomaly detection');
      return;
    }

    setLoading(true);
    try {
      const response = await apiCall('detect', {
        method: 'POST',
        body: JSON.stringify({
          detection_id: `detection_${Date.now()}`,
          data: detectionData,
          methods: detectionMethods.join(','),
          severity_threshold: severityThreshold,
          detection_options: JSON.stringify({
            real_time: realTimeDetection,
            include_recommendations: true,
            include_investigation_hints: true
          })
        })
      });

      if (response.success) {
        setAnomalies(response.anomalies || []);
        setDetectionSummary(response.detection_summary);
        setSuccess('Anomaly detection completed successfully');
      } else {
        throw new Error(response.error || 'Failed to detect anomalies');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to detect anomalies');
    } finally {
      setLoading(false);
    }
  }, [detectionData, detectionMethods, severityThreshold, realTimeDetection, apiCall]);

  // Create alert rule
  const createAlertRule = useCallback(async () => {
    setLoading(true);
    try {
      const ruleData = {
        ...newRule,
        conditions: JSON.parse(newRule.conditions || '{}'),
        thresholds: JSON.parse(newRule.thresholds || '{}'),
        severity_mapping: JSON.parse(newRule.severity_mapping || '{}')
      };

      const response = await apiCall('rules', {
        method: 'POST',
        body: JSON.stringify({
          rule_id: `rule_${Date.now()}`,
          rule_data: JSON.stringify(ruleData)
        })
      });

      if (response.success) {
        await loadAlertRules();
        setNewRuleDialog(false);
        setNewRule({
          rule_name: '',
          rule_type: 'threshold',
          category: '',
          conditions: '{}',
          thresholds: '{}',
          severity_mapping: '{}',
          notification_channels: ['in_app']
        });
        setSuccess('Alert rule created successfully');
      } else {
        throw new Error(response.error || 'Failed to create alert rule');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create alert rule');
    } finally {
      setLoading(false);
    }
  }, [newRule, loadAlertRules, apiCall]);

  // Start investigation
  const startInvestigation = useCallback(async (anomalyIds: string[]) => {
    setLoading(true);
    try {
      const response = await apiCall('investigations/start', {
        method: 'POST',
        body: JSON.stringify({
          session_id: `investigation_${Date.now()}`,
          anomaly_ids: anomalyIds.join(','),
          session_name: `Investigation ${new Date().toLocaleDateString()}`,
          investigator_id: 'current_user',
          priority_level: 'medium'
        })
      });

      if (response.success) {
        await loadInvestigations();
        setSuccess('Investigation started successfully');
      } else {
        throw new Error(response.error || 'Failed to start investigation');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start investigation');
    } finally {
      setLoading(false);
    }
  }, [loadInvestigations, apiCall]);

  // Health check
  const performHealthCheck = useCallback(async () => {
    setLoading(true);
    try {
      const response = await apiCall('health');
      if (response.success) {
        setSuccess(`System health: ${response.status}. All services operational.`);
      } else {
        throw new Error(response.error || 'Health check failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Health check failed');
    } finally {
      setLoading(false);
    }
  }, [apiCall]);

  // Get severity color
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#f44336';
      case 'high': return '#ff9800';
      case 'medium': return '#2196f3';
      case 'low': return '#4caf50';
      default: return '#9e9e9e';
    }
  };

  // Get severity chip
  const getSeverityChip = (severity: string) => (
    <Chip
      label={severity.toUpperCase()}
      size="small"
      style={{
        backgroundColor: getSeverityColor(severity),
        color: 'white',
        fontWeight: 'bold'
      }}
    />
  );

  // Chart configurations
  const severityChartData = useMemo(() => ({
    labels: Object.keys(detectionSummary?.severity_distribution || {}),
    datasets: [
      {
        data: Object.values(detectionSummary?.severity_distribution || {}),
        backgroundColor: [
          getSeverityColor('low'),
          getSeverityColor('medium'),
          getSeverityColor('high'),
          getSeverityColor('critical')
        ],
        borderWidth: 2
      }
    ]
  }), [detectionSummary]);

  const methodChartData = useMemo(() => ({
    labels: Object.keys(detectionSummary?.method_distribution || {}),
    datasets: [
      {
        label: 'Detection Method Usage',
        data: Object.values(detectionSummary?.method_distribution || {}),
        backgroundColor: 'rgba(54, 162, 235, 0.8)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }
    ]
  }), [detectionSummary]);

  // Tab panels
  const TabPanel: React.FC<{ children?: React.ReactNode; value: number; index: number }> = ({ 
    children, value, index 
  }) => (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
        <SecurityIcon fontSize="large" color="primary" />
        Anomaly Detection Interface
        <Badge badgeContent={anomalies.length} color="error" sx={{ ml: 2 }}>
          <NotificationsActiveIcon />
        </Badge>
      </Typography>

      {/* Loading and Error States */}
      {loading && <LinearProgress sx={{ mb: 2 }} />}
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert severity="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      </Snackbar>

      {/* Main Controls */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Detection Data (JSON)"
                placeholder='{"field1": [1,2,3,100], "field2": [10,20,30,300]}'
                value={detectionData}
                onChange={(e) => setDetectionData(e.target.value)}
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Detection Methods</InputLabel>
                <Select
                  multiple
                  value={detectionMethods}
                  onChange={(e) => setDetectionMethods(e.target.value as string[])}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  <MenuItem value="statistical">Statistical</MenuItem>
                  <MenuItem value="ml_based">ML-Based</MenuItem>
                  <MenuItem value="rule_based">Rule-Based</MenuItem>
                  <MenuItem value="hybrid">Hybrid</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <Box display="flex" flexDirection="column" gap={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>Severity Threshold</InputLabel>
                  <Select
                    value={severityThreshold}
                    onChange={(e) => setSeverityThreshold(e.target.value)}
                  >
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                    <MenuItem value="critical">Critical</MenuItem>
                  </Select>
                </FormControl>
                <FormControlLabel
                  control={
                    <Switch
                      checked={realTimeDetection}
                      onChange={(e) => setRealTimeDetection(e.target.checked)}
                    />
                  }
                  label="Real-time Detection"
                />
              </Box>
            </Grid>
            <Grid item xs={12}>
              <Box display="flex" gap={2} flexWrap="wrap">
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={detectAnomalies}
                  disabled={loading}
                  size="large"
                >
                  Detect Anomalies
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={loadAnomalySummary}
                  disabled={loading}
                >
                  Refresh Summary
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<SettingsIcon />}
                  onClick={performHealthCheck}
                  disabled={loading}
                >
                  Health Check
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={() => setNewRuleDialog(true)}
                  disabled={loading}
                >
                  New Alert Rule
                </Button>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Summary Dashboard */}
      {detectionSummary && (
        <Grid container spacing={4} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Total Anomalies
                </Typography>
                <Typography variant="h4" color="error">
                  {detectionSummary.total_anomalies}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Mean Confidence
                </Typography>
                <Typography variant="h4" color="primary">
                  {(detectionSummary.confidence_stats.mean_confidence * 100).toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent style={{ height: '200px' }}>
                <Typography variant="h6" gutterBottom>
                  Severity Distribution
                </Typography>
                <Doughnut data={severityChartData} options={{ maintainAspectRatio: false }} />
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent style={{ height: '200px' }}>
                <Typography variant="h6" gutterBottom>
                  Method Distribution
                </Typography>
                <Bar data={methodChartData} options={{ maintainAspectRatio: false }} />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Main Content Tabs */}
      <Card>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab label="Detected Anomalies" icon={<WarningIcon />} />
          <Tab label="Alert Rules" icon={<SettingsIcon />} />
          <Tab label="Investigations" icon={<SearchIcon />} />
          <Tab label="Analytics" icon={<AssessmentIcon />} />
        </Tabs>

        {/* Detected Anomalies Tab */}
        <TabPanel value={activeTab} index={0}>
          <Box display="flex" gap={2} mb={3} flexWrap="wrap">
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Severity</InputLabel>
              <Select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}>
                <MenuItem value="">All</MenuItem>
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Status</InputLabel>
              <Select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                <MenuItem value="">All</MenuItem>
                <MenuItem value="detected">Detected</MenuItem>
                <MenuItem value="investigating">Investigating</MenuItem>
                <MenuItem value="resolved">Resolved</MenuItem>
                <MenuItem value="false_positive">False Positive</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Category</InputLabel>
              <Select value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
                <MenuItem value="">All</MenuItem>
                <MenuItem value="statistical">Statistical</MenuItem>
                <MenuItem value="behavioral">Behavioral</MenuItem>
                <MenuItem value="pattern">Pattern</MenuItem>
                <MenuItem value="temporal">Temporal</MenuItem>
              </Select>
            </FormControl>
          </Box>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Title</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Detection Time</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {anomalies.map((anomaly) => (
                  <TableRow key={anomaly.anomaly_id}>
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {anomaly.anomaly_id.substring(0, 8)}...
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" fontWeight="bold">
                        {anomaly.title}
                      </Typography>
                    </TableCell>
                    <TableCell>{getSeverityChip(anomaly.severity)}</TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={1}>
                        <LinearProgress
                          variant="determinate"
                          value={anomaly.confidence_score * 100}
                          sx={{ width: 50, height: 6 }}
                        />
                        <Typography variant="body2">
                          {(anomaly.confidence_score * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip label={anomaly.anomaly_category} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={anomaly.status}
                        size="small"
                        color={anomaly.status === 'resolved' ? 'success' : 'default'}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {new Date(anomaly.detection_timestamp).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            onClick={() => {
                              setSelectedAnomaly(anomaly);
                              setAnomalyDetailsDialog(true);
                            }}
                          >
                            <VisibilityIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Start Investigation">
                          <IconButton
                            size="small"
                            onClick={() => startInvestigation([anomaly.anomaly_id])}
                          >
                            <SearchIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
                {anomalies.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body1" color="textSecondary" py={4}>
                        No anomalies detected. Run anomaly detection to see results.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Alert Rules Tab */}
        <TabPanel value={activeTab} index={1}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <Typography variant="h6">Alert Rules Management</Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setNewRuleDialog(true)}
            >
              Create Rule
            </Button>
          </Box>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Rule Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Triggers</TableCell>
                  <TableCell>Accuracy</TableCell>
                  <TableCell>Last Triggered</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {alertRules.map((rule) => (
                  <TableRow key={rule.rule_id}>
                    <TableCell>
                      <Typography variant="body2" fontWeight="bold">
                        {rule.rule_name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip label={rule.rule_type} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell>{rule.category}</TableCell>
                    <TableCell>
                      <Chip
                        label={rule.is_active ? 'Active' : 'Inactive'}
                        size="small"
                        color={rule.is_active ? 'success' : 'default'}
                      />
                    </TableCell>
                    <TableCell>{rule.trigger_count}</TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={1}>
                        <LinearProgress
                          variant="determinate"
                          value={rule.accuracy_score * 100}
                          sx={{ width: 50, height: 6 }}
                        />
                        <Typography variant="body2">
                          {(rule.accuracy_score * 100).toFixed(0)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      {rule.last_triggered 
                        ? new Date(rule.last_triggered).toLocaleDateString()
                        : 'Never'
                      }
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <Tooltip title="Edit Rule">
                          <IconButton size="small">
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete Rule">
                          <IconButton size="small" color="error">
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
                {alertRules.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body1" color="textSecondary" py={4}>
                        No alert rules configured. Create your first alert rule.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Investigations Tab */}
        <TabPanel value={activeTab} index={2}>
          <Typography variant="h6" gutterBottom>
            Active Investigations
          </Typography>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Session Name</TableCell>
                  <TableCell>Investigator</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Priority</TableCell>
                  <TableCell>Anomalies</TableCell>
                  <TableCell>Hours Spent</TableCell>
                  <TableCell>Started</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {investigations.map((investigation) => (
                  <TableRow key={investigation.session_id}>
                    <TableCell>
                      <Typography variant="body2" fontWeight="bold">
                        {investigation.session_name}
                      </Typography>
                    </TableCell>
                    <TableCell>{investigation.investigator_id}</TableCell>
                    <TableCell>
                      <Chip
                        label={investigation.investigation_status}
                        size="small"
                        color={investigation.investigation_status === 'completed' ? 'success' : 'primary'}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={investigation.priority_level}
                        size="small"
                        color={investigation.priority_level === 'urgent' ? 'error' : 'default'}
                      />
                    </TableCell>
                    <TableCell>{investigation.anomaly_ids.length}</TableCell>
                    <TableCell>{investigation.hours_spent.toFixed(1)}h</TableCell>
                    <TableCell>
                      {new Date(investigation.started_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <Tooltip title="View Investigation">
                          <IconButton size="small">
                            <VisibilityIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Add Evidence">
                          <IconButton size="small">
                            <AddIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
                {investigations.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      <Typography variant="body1" color="textSecondary" py={4}>
                        No active investigations. Start investigating anomalies.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Analytics Tab */}
        <TabPanel value={activeTab} index={3}>
          <Typography variant="h6" gutterBottom>
            Detection Analytics & Performance
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Detection Performance
                  </Typography>
                  <Box display="flex" flexDirection="column" gap={2}>
                    <Box display="flex" justifyContent="space-between">
                      <Typography>Total Detections:</Typography>
                      <Typography fontWeight="bold">{anomalies.length}</Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography>Active Rules:</Typography>
                      <Typography fontWeight="bold">
                        {alertRules.filter(rule => rule.is_active).length}
                      </Typography>
                    </Box>
                    <Box display="flex" justifyContent="space-between">
                      <Typography>Active Investigations:</Typography>
                      <Typography fontWeight="bold">
                        {investigations.filter(inv => inv.investigation_status === 'active').length}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    System Health
                  </Typography>
                  <Box display="flex" flexDirection="column" gap={2}>
                    <Box display="flex" alignItems="center" gap={2}>
                      <CheckCircleIcon color="success" />
                      <Typography>Detection Services: Online</Typography>
                    </Box>
                    <Box display="flex" alignItems="center" gap={2}>
                      <CheckCircleIcon color="success" />
                      <Typography>Alert Manager: Online</Typography>
                    </Box>
                    <Box display="flex" alignItems="center" gap={2}>
                      <CheckCircleIcon color="success" />
                      <Typography>Investigation Tools: Online</Typography>
                    </Box>
                    <Button
                      variant="outlined"
                      onClick={performHealthCheck}
                      startIcon={<RefreshIcon />}
                      size="small"
                    >
                      Run Health Check
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Card>

      {/* Anomaly Details Dialog */}
      <Dialog
        open={anomalyDetailsDialog}
        onClose={() => setAnomalyDetailsDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Anomaly Details: {selectedAnomaly?.title}
        </DialogTitle>
        <DialogContent>
          {selectedAnomaly && (
            <Box display="flex" flexDirection="column" gap={3} mt={2}>
              <Box display="flex" gap={2} alignItems="center">
                {getSeverityChip(selectedAnomaly.severity)}
                <Chip label={selectedAnomaly.detection_type} variant="outlined" />
                <Typography variant="body2" color="textSecondary">
                  Confidence: {(selectedAnomaly.confidence_score * 100).toFixed(1)}%
                </Typography>
              </Box>
              
              <Typography variant="body1">{selectedAnomaly.description}</Typography>
              
              <Box>
                <Typography variant="h6" gutterBottom>Recommended Actions</Typography>
                {selectedAnomaly.recommended_actions.map((action, index) => (
                  <Typography key={index} variant="body2" sx={{ ml: 2 }}>
                    • {action}
                  </Typography>
                ))}
              </Box>
              
              {selectedAnomaly.investigation_hints.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>Investigation Hints</Typography>
                  {selectedAnomaly.investigation_hints.map((hint, index) => (
                    <Typography key={index} variant="body2" sx={{ ml: 2 }}>
                      • {hint}
                    </Typography>
                  ))}
                </Box>
              )}
              
              <Box>
                <Typography variant="h6" gutterBottom>Affected Entities</Typography>
                <Box display="flex" gap={1} flexWrap="wrap">
                  {selectedAnomaly.affected_entities.map((entity, index) => (
                    <Chip key={index} label={entity} size="small" />
                  ))}
                </Box>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAnomalyDetailsDialog(false)}>Close</Button>
          <Button
            variant="contained"
            onClick={() => {
              if (selectedAnomaly) {
                startInvestigation([selectedAnomaly.anomaly_id]);
                setAnomalyDetailsDialog(false);
              }
            }}
          >
            Start Investigation
          </Button>
        </DialogActions>
      </Dialog>

      {/* New Rule Dialog */}
      <Dialog
        open={newRuleDialog}
        onClose={() => setNewRuleDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New Alert Rule</DialogTitle>
        <DialogContent>
          <Box display="flex" flexDirection="column" gap={3} mt={2}>
            <TextField
              fullWidth
              label="Rule Name"
              value={newRule.rule_name}
              onChange={(e) => setNewRule({ ...newRule, rule_name: e.target.value })}
            />
            
            <FormControl fullWidth>
              <InputLabel>Rule Type</InputLabel>
              <Select
                value={newRule.rule_type}
                onChange={(e) => setNewRule({ ...newRule, rule_type: e.target.value })}
              >
                <MenuItem value="threshold">Threshold</MenuItem>
                <MenuItem value="statistical">Statistical</MenuItem>
                <MenuItem value="pattern">Pattern</MenuItem>
                <MenuItem value="composite">Composite</MenuItem>
                <MenuItem value="ml_based">ML-Based</MenuItem>
              </Select>
            </FormControl>
            
            <TextField
              fullWidth
              label="Category"
              value={newRule.category}
              onChange={(e) => setNewRule({ ...newRule, category: e.target.value })}
              placeholder="e.g., financial, behavioral, system"
            />
            
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Conditions (JSON)"
              value={newRule.conditions}
              onChange={(e) => setNewRule({ ...newRule, conditions: e.target.value })}
              placeholder='{"field": "value", "operator": "gt", "threshold": 100}'
            />
            
            <TextField
              fullWidth
              multiline
              rows={2}
              label="Thresholds (JSON)"
              value={newRule.thresholds}
              onChange={(e) => setNewRule({ ...newRule, thresholds: e.target.value })}
              placeholder='{"warning": 50, "critical": 100}'
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewRuleDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={createAlertRule}
            disabled={!newRule.rule_name || !newRule.category}
          >
            Create Rule
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default AnomalyDetectionInterface;
