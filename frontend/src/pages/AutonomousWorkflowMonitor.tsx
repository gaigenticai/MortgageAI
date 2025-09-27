/**
 * Autonomous Workflow Monitor Dashboard
 * 
 * This component provides a comprehensive real-time monitoring dashboard for agent decisions,
 * learning patterns, performance analytics, and workflow optimization in the MortgageAI system.
 * 
 * Features:
 * - Real-time workflow monitoring and tracking
 * - Agent decision analytics and performance metrics
 * - Learning pattern visualization and insights
 * - Performance optimization recommendations
 * - Bottleneck detection and resolution
 * - Comprehensive reporting and data export
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Paper,
  Tabs,
  Tab,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  AlertTitle,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Tooltip,
  IconButton,
  Badge,
  LinearProgress,
  CircularProgress,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Slider,
  ButtonGroup
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayArrowIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Settings as SettingsIcon,
  Analytics as AnalyticsIcon,
  Insights as InsightsIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Timeline as TimelineIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Psychology as PsychologyIcon,
  Build as BuildIcon,
  Visibility as VisibilityIcon,
  NotificationsActive as AlertIcon,
  Assessment as AssessmentIcon,
  Memory as MemoryIcon,
  NetworkCheck as NetworkCheckIcon,
  AutoGraph as AutoGraphIcon
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter
} from 'recharts';

interface DashboardData {
  timestamp: string;
  performance_summary: { [key: string]: any };
  learning_patterns: any[];
  workflow_statistics: any;
  optimization_recommendations: any[];
  active_alerts: number;
  system_health: {
    status: string;
    score: number;
    factors_evaluated: number;
    last_update: string;
  };
  predictions?: { [key: string]: any };
  optimization_summary?: {
    total_opportunities: number;
    total_bottlenecks: number;
    predicted_savings: number;
  };
}

interface MonitoringSession {
  session_id: string;
  monitoring_status: 'active' | 'stopped';
  start_time: string;
  configuration: any;
}

interface LearningPattern {
  pattern_id: string;
  pattern_type: string;
  agent_id: string;
  confidence: number;
  trend_direction: string;
  insights: string[];
  recommendations: string[];
  start_time: string;
  end_time: string;
}

interface OptimizationRecommendation {
  priority: number;
  title: string;
  type: string;
  implementation_complexity: string;
  estimated_effort: string;
  expected_impact: number;
  detailed_recommendations: string[];
  affected_components: string[];
}

const AutonomousWorkflowMonitor: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [monitoringSession, setMonitoringSession] = useState<MonitoringSession | null>(null);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [learningPatterns, setLearningPatterns] = useState<LearningPattern[]>([]);
  const [optimizationRecommendations, setOptimizationRecommendations] = useState<OptimizationRecommendation[]>([]);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [showSettings, setShowSettings] = useState(false);
  const [showPatternDetails, setShowPatternDetails] = useState(false);
  const [selectedPattern, setSelectedPattern] = useState<LearningPattern | null>(null);
  
  // Filter states
  const [timeRange, setTimeRange] = useState('24h');
  const [agentFilter, setAgentFilter] = useState<string>('');
  const [patternTypeFilter, setPatternTypeFilter] = useState<string[]>([]);
  const [minimumConfidence, setMinimumConfidence] = useState(0.5);
  
  // Refs
  const refreshTimerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Constants
  const TIME_RANGES = [
    { value: '1h', label: '1 Hour' },
    { value: '24h', label: '24 Hours' },
    { value: '7d', label: '7 Days' },
    { value: '30d', label: '30 Days' }
  ];

  const PATTERN_TYPES = [
    'improvement_trend',
    'degradation',
    'seasonal_pattern',
    'concept_drift',
    'anomalous_behavior',
    'specialization'
  ];

  const HEALTH_COLORS = {
    excellent: '#4caf50',
    good: '#8bc34a',
    fair: '#ffc107',
    poor: '#ff9800',
    critical: '#f44336',
    error: '#d32f2f'
  };

  const TREND_COLORS = {
    improving: '#4caf50',
    declining: '#f44336',
    stable: '#2196f3',
    anomalous: '#ff9800'
  };

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh && monitoringSession?.monitoring_status === 'active') {
      refreshTimerRef.current = setInterval(() => {
        refreshDashboard();
      }, refreshInterval * 1000);
    }
    
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
      }
    };
  }, [autoRefresh, refreshInterval, monitoringSession]);

  // API functions
  const startMonitoring = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/workflow-monitor/start-monitoring', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Failed to start monitoring');
      }

      setMonitoringSession(result);
      
      // Initial dashboard refresh
      await refreshDashboard();
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const stopMonitoring = useCallback(async () => {
    if (!monitoringSession) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/workflow-monitor/stop-monitoring', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setMonitoringSession(prev => prev ? { ...prev, monitoring_status: 'stopped' } : null);
      }
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [monitoringSession]);

  const refreshDashboard = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        time_range: timeRange,
        include_predictions: 'true',
        include_optimization: 'true'
      });
      
      if (agentFilter) {
        params.append('agent_filter', agentFilter);
      }

      const response = await fetch(`/api/workflow-monitor/dashboard?${params}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Failed to refresh dashboard');
      }

      setDashboardData(result.dashboard_data);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      console.warn(`Dashboard refresh failed: ${errorMessage}`);
      // Don't set error state for refresh failures to avoid disrupting UI
      return null;
    }
  }, [timeRange, agentFilter]);

  const analyzeLearningPatterns = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/workflow-monitor/analyze-patterns', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          time_range: timeRange,
          agent_ids: agentFilter ? [agentFilter] : [],
          pattern_types: patternTypeFilter,
          minimum_confidence: minimumConfidence,
          include_recommendations: true
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Pattern analysis failed');
      }

      setLearningPatterns(result.patterns_detected);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [timeRange, agentFilter, patternTypeFilter, minimumConfidence]);

  const getOptimizationRecommendations = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/workflow-monitor/optimize-workflow', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          optimization_scope: 'comprehensive',
          include_bottleneck_analysis: true,
          include_resource_optimization: true,
          include_predictions: true
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Optimization analysis failed');
      }

      setOptimizationRecommendations(result.implementation_recommendations);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    refreshDashboard();
  }, [refreshDashboard]);

  // Computed values
  const systemHealthColor = useMemo(() => {
    if (!dashboardData?.system_health) return HEALTH_COLORS.error;
    return HEALTH_COLORS[dashboardData.system_health.status as keyof typeof HEALTH_COLORS] || HEALTH_COLORS.error;
  }, [dashboardData?.system_health]);

  const performanceTrends = useMemo(() => {
    if (!dashboardData?.performance_summary) return [];
    
    return Object.entries(dashboardData.performance_summary).map(([agentId, data]) => ({
      agent: agentId,
      accuracy: data.accuracy?.current || 0,
      processing_time: data.processing_time?.current || 0,
      trend: data.accuracy?.trend || 'stable'
    }));
  }, [dashboardData?.performance_summary]);

  // Event handlers
  const handleStartMonitoring = () => {
    startMonitoring();
  };

  const handleStopMonitoring = () => {
    stopMonitoring();
  };

  const handleRefreshDashboard = () => {
    refreshDashboard();
  };

  const handleAnalyzePatterns = () => {
    analyzeLearningPatterns();
  };

  const handleGetOptimizations = () => {
    getOptimizationRecommendations();
  };

  const handlePatternClick = (pattern: LearningPattern) => {
    setSelectedPattern(pattern);
    setShowPatternDetails(true);
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  const getPatternIcon = (patternType: string) => {
    switch (patternType) {
      case 'improvement_trend':
        return <TrendingUpIcon color="success" />;
      case 'degradation':
        return <TrendingDownIcon color="error" />;
      case 'anomalous_behavior':
        return <WarningIcon color="warning" />;
      case 'specialization':
        return <PsychologyIcon color="primary" />;
      default:
        return <TimelineIcon />;
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low':
        return 'success';
      case 'medium':
        return 'warning';
      case 'high':
        return 'error';
      default:
        return 'default';
    }
  };

  // Tab content components
  const OverviewTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">System Performance Overview</Typography>
              <Box>
                <Tooltip title="Refresh Dashboard">
                  <IconButton onClick={handleRefreshDashboard} disabled={loading}>
                    <RefreshIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Settings">
                  <IconButton onClick={() => setShowSettings(true)}>
                    <SettingsIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
            
            {dashboardData && (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="agent" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip />
                  <Legend />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#2196f3" 
                    strokeWidth={2}
                    name="Accuracy"
                  />
                  <Line 
                    yAxisId="right"
                    type="monotone" 
                    dataKey="processing_time" 
                    stroke="#ff9800" 
                    strokeWidth={2}
                    name="Processing Time (s)"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <NetworkCheckIcon sx={{ mr: 1, color: systemHealthColor }} />
                  <Typography variant="h6">System Health</Typography>
                </Box>
                
                {dashboardData?.system_health && (
                  <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Box sx={{ width: '100%', mr: 1 }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={dashboardData.system_health.score * 100}
                          sx={{ 
                            height: 8, 
                            borderRadius: 4,
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: systemHealthColor
                            }
                          }}
                        />
                      </Box>
                      <Typography variant="body2" sx={{ minWidth: 35 }}>
                        {Math.round(dashboardData.system_health.score * 100)}%
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="textSecondary" sx={{ textTransform: 'capitalize' }}>
                      Status: {dashboardData.system_health.status}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Factors: {dashboardData.system_health.factors_evaluated}
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <AlertIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Active Alerts
                </Typography>
                
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 60 }}>
                  <Badge badgeContent={dashboardData?.active_alerts || 0} color="error" max={99}>
                    <ErrorIcon sx={{ fontSize: 48, color: dashboardData?.active_alerts ? '#f44336' : '#ccc' }} />
                  </Badge>
                </Box>
                
                {dashboardData?.active_alerts ? (
                  <Typography variant="body2" color="error" align="center">
                    {dashboardData.active_alerts} alert{dashboardData.active_alerts !== 1 ? 's' : ''} require attention
                  </Typography>
                ) : (
                  <Typography variant="body2" color="success.main" align="center">
                    All systems operating normally
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {dashboardData?.workflow_statistics && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Workflow Statistics
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Total Executions:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {dashboardData.workflow_statistics.total_executions}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Success Rate:</Typography>
                      <Typography variant="body2" fontWeight="bold" color="success.main">
                        {dashboardData.workflow_statistics.total_executions > 0 
                          ? Math.round((dashboardData.workflow_statistics.successful_executions / dashboardData.workflow_statistics.total_executions) * 100)
                          : 0}%
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Avg Processing:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {formatDuration(dashboardData.workflow_statistics.average_processing_time)}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Grid>
    </Grid>
  );

  const LearningPatternsTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Learning Patterns Analysis</Typography>
              <Button 
                variant="outlined" 
                startIcon={<PsychologyIcon />}
                onClick={handleAnalyzePatterns}
                disabled={loading}
              >
                Analyze Patterns
              </Button>
            </Box>
            
            {learningPatterns.length > 0 ? (
              <List>
                {learningPatterns.slice(0, 10).map((pattern) => (
                  <ListItem 
                    key={pattern.pattern_id}
                    button
                    onClick={() => handlePatternClick(pattern)}
                    sx={{ border: '1px solid #e0e0e0', borderRadius: 1, mb: 1 }}
                  >
                    <ListItemIcon>
                      {getPatternIcon(pattern.pattern_type)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="subtitle2">
                            {pattern.agent_id}
                          </Typography>
                          <Chip 
                            label={pattern.pattern_type.replace('_', ' ')} 
                            size="small"
                            color={pattern.trend_direction === 'improving' ? 'success' : 
                                   pattern.trend_direction === 'declining' ? 'error' : 'default'}
                          />
                          <Typography variant="caption" color="textSecondary">
                            Confidence: {Math.round(pattern.confidence * 100)}%
                          </Typography>
                        </Box>
                      }
                      secondary={
                        <Typography variant="body2" color="textSecondary">
                          {pattern.insights[0] || 'No insights available'}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <PsychologyIcon sx={{ fontSize: 64, color: 'text.secondary' }} />
                <Typography variant="h6" color="textSecondary">
                  No Learning Patterns Detected
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Run pattern analysis to identify learning trends
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Pattern Filters
            </Typography>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Agent Filter</InputLabel>
              <Select
                value={agentFilter}
                onChange={(e) => setAgentFilter(e.target.value)}
                label="Agent Filter"
              >
                <MenuItem value="">All Agents</MenuItem>
                <MenuItem value="compliance_agent">Compliance Agent</MenuItem>
                <MenuItem value="quality_control_agent">Quality Control Agent</MenuItem>
                <MenuItem value="network_analyzer">Network Analyzer</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Pattern Types</InputLabel>
              <Select
                multiple
                value={patternTypeFilter}
                onChange={(e) => setPatternTypeFilter(typeof e.target.value === 'string' ? e.target.value.split(',') : e.target.value)}
                label="Pattern Types"
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value.replace('_', ' ')} size="small" />
                    ))}
                  </Box>
                )}
              >
                {PATTERN_TYPES.map((type) => (
                  <MenuItem key={type} value={type}>
                    {type.replace('_', ' ')}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Typography gutterBottom>
              Minimum Confidence: {Math.round(minimumConfidence * 100)}%
            </Typography>
            <Slider
              value={minimumConfidence}
              onChange={(_, value) => setMinimumConfidence(value as number)}
              min={0}
              max={1}
              step={0.05}
              marks={[
                { value: 0, label: '0%' },
                { value: 0.5, label: '50%' },
                { value: 1, label: '100%' }
              ]}
              sx={{ mb: 2 }}
            />
          </CardContent>
        </Card>

        {dashboardData?.predictions && (
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Predictions
              </Typography>
              
              {Object.entries(dashboardData.predictions).map(([agentId, predictions]) => (
                <Accordion key={agentId} sx={{ mb: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle2">{agentId}</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    {Object.entries(predictions as Record<string, any>).map(([metric, prediction]) => (
                      <Box key={metric} sx={{ mb: 1 }}>
                        <Typography variant="caption" color="textSecondary">
                          {metric}:
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {prediction.trend === 'improving' ? (
                            <TrendingUpIcon fontSize="small" color="success" />
                          ) : prediction.trend === 'declining' ? (
                            <TrendingDownIcon fontSize="small" color="error" />
                          ) : (
                            <TimelineIcon fontSize="small" />
                          )}
                          <Typography variant="body2">
                            {prediction.trend}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            ({Math.round(prediction.confidence * 100)}% confidence)
                          </Typography>
                        </Box>
                      </Box>
                    ))}
                  </AccordionDetails>
                </Accordion>
              ))}
            </CardContent>
          </Card>
        )}
      </Grid>
    </Grid>
  );

  const OptimizationTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Workflow Optimization</Typography>
              <Button 
                variant="outlined" 
                startIcon={<BuildIcon />}
                onClick={handleGetOptimizations}
                disabled={loading}
              >
                Get Recommendations
              </Button>
            </Box>

            {dashboardData?.optimization_summary && (
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <SpeedIcon sx={{ fontSize: 48, color: 'primary.main' }} />
                      <Typography variant="h4">
                        {dashboardData.optimization_summary.total_opportunities}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Optimization Opportunities
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <WarningIcon sx={{ fontSize: 48, color: 'warning.main' }} />
                      <Typography variant="h4">
                        {dashboardData.optimization_summary.total_bottlenecks}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Detected Bottlenecks
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center' }}>
                      <TrendingUpIcon sx={{ fontSize: 48, color: 'success.main' }} />
                      <Typography variant="h4">
                        {formatDuration(dashboardData.optimization_summary.predicted_savings)}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Predicted Time Savings
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}

            {optimizationRecommendations.length > 0 ? (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Priority</TableCell>
                      <TableCell>Recommendation</TableCell>
                      <TableCell>Complexity</TableCell>
                      <TableCell>Effort</TableCell>
                      <TableCell>Expected Impact</TableCell>
                      <TableCell>Components</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {optimizationRecommendations.slice(0, 10).map((rec) => (
                      <TableRow key={`${rec.priority}-${rec.title}`}>
                        <TableCell>
                          <Chip 
                            label={rec.priority} 
                            color={rec.priority <= 3 ? 'error' : rec.priority <= 6 ? 'warning' : 'default'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {rec.title}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            {rec.type}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={rec.implementation_complexity} 
                            color={getComplexityColor(rec.implementation_complexity)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {rec.estimated_effort}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {rec.expected_impact > 0 ? formatDuration(rec.expected_impact) : 'TBD'}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {rec.affected_components.slice(0, 2).map((component) => (
                              <Chip key={component} label={component} size="small" variant="outlined" />
                            ))}
                            {rec.affected_components.length > 2 && (
                              <Typography variant="caption" color="textSecondary">
                                +{rec.affected_components.length - 2} more
                              </Typography>
                            )}
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <BuildIcon sx={{ fontSize: 64, color: 'text.secondary' }} />
                <Typography variant="h6" color="textSecondary">
                  No Optimization Data Available
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Run optimization analysis to get recommendations
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const AnalyticsTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Performance Trends
            </Typography>
            
            {performanceTrends.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={performanceTrends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="agent" />
                  <YAxis />
                  <RechartsTooltip />
                  <Bar dataKey="accuracy" fill="#2196f3" name="Accuracy" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <AssessmentIcon sx={{ fontSize: 64, color: 'text.secondary' }} />
                <Typography variant="body1" color="textSecondary">
                  No performance data available
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              System Resource Usage
            </Typography>
            
            {/* Placeholder for resource usage visualization */}
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <MemoryIcon sx={{ fontSize: 64, color: 'text.secondary' }} />
              <Typography variant="body1" color="textSecondary">
                Resource usage data will be displayed here
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Recent Recommendations
            </Typography>
            
            {dashboardData?.optimization_recommendations && dashboardData.optimization_recommendations.length > 0 ? (
              <List>
                {dashboardData.optimization_recommendations.slice(0, 5).map((rec, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckCircleIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={rec}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="textSecondary">
                No recommendations available
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  // Settings Dialog
  const SettingsDialog = () => (
    <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="sm" fullWidth>
      <DialogTitle>Monitoring Settings</DialogTitle>
      <DialogContent>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                label="Time Range"
              >
                {TIME_RANGES.map((range) => (
                  <MenuItem key={range.value} value={range.value}>
                    {range.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
              }
              label="Auto Refresh"
            />
          </Grid>

          {autoRefresh && (
            <Grid item xs={12}>
              <Typography gutterBottom>
                Refresh Interval: {refreshInterval} seconds
              </Typography>
              <Slider
                value={refreshInterval}
                onChange={(_, value) => setRefreshInterval(value as number)}
                min={10}
                max={300}
                step={10}
                marks={[
                  { value: 10, label: '10s' },
                  { value: 60, label: '1m' },
                  { value: 300, label: '5m' }
                ]}
              />
            </Grid>
          )}
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowSettings(false)}>Cancel</Button>
        <Button onClick={() => setShowSettings(false)} variant="contained">
          Apply
        </Button>
      </DialogActions>
    </Dialog>
  );

  // Pattern Details Dialog
  const PatternDetailsDialog = () => (
    <Dialog 
      open={showPatternDetails} 
      onClose={() => setShowPatternDetails(false)} 
      maxWidth="md" 
      fullWidth
    >
      <DialogTitle>
        Learning Pattern Details
        {selectedPattern && (
          <Chip 
            label={selectedPattern.pattern_type.replace('_', ' ')}
            color={selectedPattern.trend_direction === 'improving' ? 'success' : 
                   selectedPattern.trend_direction === 'declining' ? 'error' : 'default'}
            sx={{ ml: 2 }}
          />
        )}
      </DialogTitle>
      <DialogContent>
        {selectedPattern && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>Pattern Information</Typography>
              <Typography variant="body2"><strong>Agent:</strong> {selectedPattern.agent_id}</Typography>
              <Typography variant="body2"><strong>Type:</strong> {selectedPattern.pattern_type}</Typography>
              <Typography variant="body2"><strong>Confidence:</strong> {Math.round(selectedPattern.confidence * 100)}%</Typography>
              <Typography variant="body2"><strong>Trend:</strong> {selectedPattern.trend_direction}</Typography>
              <Typography variant="body2"><strong>Duration:</strong> {new Date(selectedPattern.start_time).toLocaleDateString()} - {new Date(selectedPattern.end_time).toLocaleDateString()}</Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>Insights</Typography>
              <List dense>
                {selectedPattern.insights.map((insight, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <InfoIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={insight}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Grid>
            {selectedPattern.recommendations.length > 0 && (
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>Recommendations</Typography>
                <List dense>
                  {selectedPattern.recommendations.map((rec, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <CheckCircleIcon fontSize="small" color="primary" />
                      </ListItemIcon>
                      <ListItemText 
                        primary={rec}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Grid>
            )}
          </Grid>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowPatternDetails(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            🤖 Autonomous Workflow Monitor
          </Typography>
          <Typography variant="body1" color="textSecondary">
            Real-time agent decision tracking, learning pattern analysis, and performance optimization
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          {monitoringSession?.monitoring_status === 'active' ? (
            <Button
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
              onClick={handleStopMonitoring}
              disabled={loading}
            >
              Stop Monitoring
            </Button>
          ) : (
            <Button
              variant="contained"
              color="success"
              startIcon={<PlayArrowIcon />}
              onClick={handleStartMonitoring}
              disabled={loading}
            >
              Start Monitoring
            </Button>
          )}
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            disabled={!dashboardData}
          >
            Export Data
          </Button>
        </Box>
      </Box>

      {/* Monitoring Status */}
      {monitoringSession && (
        <Alert 
          severity={monitoringSession.monitoring_status === 'active' ? 'success' : 'info'}
          sx={{ mb: 3 }}
        >
          <AlertTitle>Monitoring Status</AlertTitle>
          Session {monitoringSession.monitoring_status} since {new Date(monitoringSession.start_time).toLocaleString()}
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <AlertTitle>Error</AlertTitle>
          {error}
        </Alert>
      )}

      {loading && (
        <Box sx={{ mb: 3 }}>
          <LinearProgress />
          <Typography variant="body2" sx={{ mt: 1 }}>
            Processing workflow monitoring data...
          </Typography>
        </Box>
      )}

      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, value) => setActiveTab(value)}>
          <Tab icon={<AutoGraphIcon />} label="Overview" />
          <Tab icon={<PsychologyIcon />} label="Learning Patterns" />
          <Tab icon={<BuildIcon />} label="Optimization" />
          <Tab icon={<AnalyticsIcon />} label="Analytics" />
        </Tabs>
      </Paper>

      {activeTab === 0 && <OverviewTab />}
      {activeTab === 1 && <LearningPatternsTab />}
      {activeTab === 2 && <OptimizationTab />}
      {activeTab === 3 && <AnalyticsTab />}

      <SettingsDialog />
      <PatternDetailsDialog />
    </Box>
  );
};

export default AutonomousWorkflowMonitor;
