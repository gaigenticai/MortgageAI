/**
 * Agent Performance Metrics Dashboard - React Component
 * Created: 2024-01-15
 * Author: MortgageAI Development Team
 * 
 * Comprehensive React interface for the Agent Performance Metrics Dashboard with
 * detailed analytics, success rates tracking, optimization recommendations,
 * and real-time monitoring capabilities.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
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
  Snackbar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  CircularProgress,
  Slider,
  TextField,
  Autocomplete,
  Avatar,
  AvatarGroup
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Analytics as AnalyticsIcon,
  Speed as SpeedIcon,
  Assignment as AssignmentIcon,
  Star as StarIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Settings as SettingsIcon,
  Compare as CompareIcon,
  Timeline as TimelineIcon,
  Insights as InsightsIcon,
  Recommend as RecommendIcon,
  Notifications as NotificationsIcon,
  Dashboard as DashboardIcon,
  Assessment as AssessmentIcon,
  ExpandMore as ExpandMoreIcon,
  Close as CloseIcon,
  FilterList as FilterListIcon,
  Sort as SortIcon,
  Visibility as VisibilityIcon,
  Edit as EditIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  CloudDownload as CloudDownloadIcon,
  Share as ShareIcon,
  Bookmark as BookmarkIcon,
  EmojiObjects as EmojiObjectsIcon,
  Psychology as PsychologyIcon,
  Engineering as EngineeringIcon,
  Security as SecurityIcon,
  Groups as GroupsIcon,
  Timeline2 as Timeline2Icon
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip as ChartTooltip,
  Legend
} from 'chart.js';
import { Line, Bar, Doughnut, Pie, Radar, PolarArea, Scatter } from 'react-chartjs-2';

// Chart.js registration
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  ChartTooltip,
  Legend
);

// Type definitions
interface AgentMetric {
  agent_id: string;
  agent_name: string;
  overall_score: number;
  success_rate: number;
  quality_score: number;
  efficiency_score: number;
  compliance_score: number;
  user_satisfaction_score: number;
  total_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  avg_processing_time: number;
  last_updated: string;
  trend: 'improving' | 'declining' | 'stable';
  performance_grade: string;
  rank: number;
  team_id?: string;
}

interface PerformanceAnalysis {
  analysis_id: string;
  agent_id: string;
  overall_grade: string;
  key_insights: string[];
  strengths: string[];
  improvement_areas: string[];
  recommendations: OptimizationRecommendation[];
  forecasts: Record<string, any>;
  trend_analysis: Record<string, any>;
  benchmark_comparison: Record<string, any>;
  processing_time: number;
  created_at: string;
}

interface OptimizationRecommendation {
  recommendation_id: string;
  title: string;
  description: string;
  recommendation_type: string;
  priority: 'low' | 'normal' | 'high' | 'urgent' | 'critical';
  expected_impact: number;
  implementation_effort: string;
  estimated_roi: number;
  confidence_level: number;
  implementation_steps: string[];
  created_at: string;
}

interface PerformanceAlert {
  alert_id: string;
  agent_id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  alert_status: string;
  metric_name: string;
  created_at: string;
  acknowledged_by?: string;
}

interface DashboardData {
  dashboard_id: string;
  agents: Record<string, AgentMetric>;
  aggregates: {
    total_agents: number;
    avg_performance: number;
    top_performer: string;
    improvement_needed: number;
  };
  rankings: AgentMetric[];
  kpis: Record<string, number>;
  alerts: PerformanceAlert[];
  generated_at: string;
}

const AgentPerformanceMetrics: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [performanceAnalyses, setPerformanceAnalyses] = useState<PerformanceAnalysis[]>([]);
  const [optimizationRecommendations, setOptimizationRecommendations] = useState<OptimizationRecommendation[]>([]);
  const [performanceAlerts, setPerformanceAlerts] = useState<PerformanceAlert[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Dialog states
  const [selectedAgent, setSelectedAgent] = useState<AgentMetric | null>(null);
  const [agentDetailsDialog, setAgentDetailsDialog] = useState(false);
  const [analysisDialog, setAnalysisDialog] = useState(false);
  const [recommendationDialog, setRecommendationDialog] = useState(false);
  const [compareDialog, setCompareDialog] = useState(false);
  
  // Filter and settings states
  const [timeRange, setTimeRange] = useState('last_30_days');
  const [dashboardType, setDashboardType] = useState('comprehensive');
  const [sortBy, setSortBy] = useState('overall_score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterTeam, setFilterTeam] = useState<string>('all');
  const [filterGrade, setFilterGrade] = useState<string>('all');
  const [autoRefresh, setAutoRefresh] = useState(false);
  
  // Real-time monitoring states
  const [realTimeMode, setRealTimeMode] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(30);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  // API calls
  const apiCall = useCallback(async (endpoint: string, options: RequestInit = {}) => {
    const baseUrl = process.env.REACT_APP_API_URL || '';
    const response = await fetch(`${baseUrl}/api/performance/${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`,
        ...options.headers
      },
      ...options
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.statusText}`);
    }

    return response.json();
  }, []);

  // Load dashboard data
  const loadDashboardData = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        time_range: timeRange,
        dashboard_type: dashboardType,
        include_forecasts: 'true',
        include_benchmarks: 'true'
      });

      if (selectedAgents.length > 0) {
        params.append('agent_ids', selectedAgents.join(','));
      }

      const response = await apiCall(`dashboard/generate?${params}`);
      
      if (response.success) {
        setDashboardData(response.data.dashboard);
        setLastRefresh(new Date());
        setSuccess('Dashboard data updated successfully');
      } else {
        throw new Error(response.error || 'Failed to load dashboard data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  }, [timeRange, dashboardType, selectedAgents, apiCall]);

  // Load performance analyses
  const loadPerformanceAnalyses = useCallback(async (agentId?: string) => {
    try {
      const response = await apiCall('analysis/performance', {
        method: 'POST',
        body: JSON.stringify({
          agent_id: agentId,
          analysis_period: 'monthly',
          include_forecasting: true,
          include_recommendations: true
        })
      });
      
      if (response.success && response.data.analysis) {
        setPerformanceAnalyses(prev => [
          response.data.analysis,
          ...prev.filter(a => a.agent_id !== agentId)
        ]);
      }
    } catch (err) {
      console.error('Failed to load performance analysis:', err);
    }
  }, [apiCall]);

  // Load optimization recommendations
  const loadOptimizationRecommendations = useCallback(async (agentId: string) => {
    try {
      const response = await apiCall('recommendations/optimize', {
        method: 'POST',
        body: JSON.stringify({
          agent_id: agentId,
          include_implementation_plan: true,
          include_roi_analysis: true
        })
      });
      
      if (response.success) {
        setOptimizationRecommendations(prev => [
          ...response.data.recommendations,
          ...prev.filter(r => r.recommendation_id !== response.data.recommendations[0]?.recommendation_id)
        ]);
      }
    } catch (err) {
      console.error('Failed to load optimization recommendations:', err);
    }
  }, [apiCall]);

  // Load performance alerts
  const loadPerformanceAlerts = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        time_range: 'last_day',
        status: 'active',
        limit: '50'
      });

      const response = await apiCall(`alerts/performance?${params}`);
      
      if (response.success) {
        setPerformanceAlerts(response.data.alerts);
      }
    } catch (err) {
      console.error('Failed to load performance alerts:', err);
    }
  }, [apiCall]);

  // Auto-refresh mechanism
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (autoRefresh && realTimeMode) {
      intervalId = setInterval(() => {
        loadDashboardData();
        loadPerformanceAlerts();
      }, refreshInterval * 1000);
    }
    
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [autoRefresh, realTimeMode, refreshInterval, loadDashboardData, loadPerformanceAlerts]);

  // Initial data load
  useEffect(() => {
    loadDashboardData();
    loadPerformanceAlerts();
  }, [loadDashboardData, loadPerformanceAlerts]);

  // Collect metrics for selected agent
  const collectAgentMetrics = useCallback(async (agentId: string) => {
    setLoading(true);
    try {
      const response = await apiCall('metrics/collect', {
        method: 'POST',
        body: JSON.stringify({
          agent_id: agentId,
          time_period: 'daily',
          include_context: true,
          metrics_types: ['basic', 'quality', 'efficiency', 'compliance', 'interaction']
        })
      });

      if (response.success) {
        await loadDashboardData(); // Refresh dashboard with new metrics
        setSuccess(`Metrics collected successfully for agent ${agentId}`);
      } else {
        throw new Error(response.error || 'Failed to collect metrics');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to collect metrics');
    } finally {
      setLoading(false);
    }
  }, [apiCall, loadDashboardData]);

  // Generate analysis for agent
  const generateAnalysis = useCallback(async (agentId: string) => {
    setLoading(true);
    try {
      await loadPerformanceAnalyses(agentId);
      await loadOptimizationRecommendations(agentId);
      setSuccess(`Performance analysis generated for agent ${agentId}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate analysis');
    } finally {
      setLoading(false);
    }
  }, [loadPerformanceAnalyses, loadOptimizationRecommendations]);

  // Export dashboard data
  const exportDashboard = useCallback(async (format: string) => {
    setLoading(true);
    try {
      const response = await apiCall('export', {
        method: 'POST',
        body: JSON.stringify({
          export_type: 'dashboard',
          agent_ids: selectedAgents,
          time_range: timeRange,
          format: format,
          include_charts: true
        })
      });

      if (response.success) {
        // Handle file download
        setSuccess(`Dashboard exported successfully as ${format.toUpperCase()}`);
      } else {
        throw new Error(response.error || 'Failed to export dashboard');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export dashboard');
    } finally {
      setLoading(false);
    }
  }, [apiCall, selectedAgents, timeRange]);

  // Utility functions
  const getPerformanceGrade = (score: number): string => {
    if (score >= 90) return 'A+';
    if (score >= 85) return 'A';
    if (score >= 80) return 'B+';
    if (score >= 75) return 'B';
    if (score >= 70) return 'C+';
    if (score >= 65) return 'C';
    if (score >= 60) return 'D+';
    if (score >= 55) return 'D';
    return 'F';
  };

  const getGradeColor = (grade: string): string => {
    switch (grade) {
      case 'A+':
      case 'A': return '#4caf50';
      case 'B+':
      case 'B': return '#8bc34a';
      case 'C+':
      case 'C': return '#ffc107';
      case 'D+':
      case 'D': return '#ff9800';
      default: return '#f44336';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <TrendingUpIcon style={{ color: '#4caf50' }} />;
      case 'declining': return <TrendingDownIcon style={{ color: '#f44336' }} />;
      default: return <TimelineIcon style={{ color: '#9e9e9e' }} />;
    }
  };

  const getSeverityColor = (severity: string): string => {
    switch (severity) {
      case 'critical': return '#f44336';
      case 'high': return '#ff5722';
      case 'medium': return '#ff9800';
      case 'low': return '#2196f3';
      default: return '#9e9e9e';
    }
  };

  const getPriorityChip = (priority: string) => (
    <Chip
      label={priority.toUpperCase()}
      size="small"
      style={{
        backgroundColor: priority === 'critical' ? '#f44336' : 
                        priority === 'high' ? '#ff5722' :
                        priority === 'urgent' ? '#ff9800' :
                        priority === 'normal' ? '#2196f3' : '#9e9e9e',
        color: 'white',
        fontWeight: 'bold'
      }}
    />
  );

  // Sorted and filtered agents
  const sortedAgents = useMemo(() => {
    if (!dashboardData) return [];
    
    let agents = Object.values(dashboardData.agents);
    
    // Apply filters
    if (filterTeam !== 'all') {
      agents = agents.filter(agent => agent.team_id === filterTeam);
    }
    
    if (filterGrade !== 'all') {
      agents = agents.filter(agent => getPerformanceGrade(agent.overall_score) === filterGrade);
    }
    
    // Apply sorting
    agents.sort((a, b) => {
      const aValue = a[sortBy as keyof AgentMetric] as number;
      const bValue = b[sortBy as keyof AgentMetric] as number;
      
      if (sortOrder === 'asc') {
        return aValue - bValue;
      } else {
        return bValue - aValue;
      }
    });
    
    return agents;
  }, [dashboardData, filterTeam, filterGrade, sortBy, sortOrder]);

  // Performance trends chart data
  const performanceTrendsData = useMemo(() => {
    if (!dashboardData) return null;
    
    const agents = Object.values(dashboardData.agents).slice(0, 10);
    
    return {
      labels: agents.map(agent => agent.agent_name),
      datasets: [
        {
          label: 'Overall Score',
          data: agents.map(agent => agent.overall_score),
          backgroundColor: 'rgba(54, 162, 235, 0.8)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        },
        {
          label: 'Success Rate',
          data: agents.map(agent => agent.success_rate * 100),
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1
        },
        {
          label: 'Quality Score',
          data: agents.map(agent => agent.quality_score * 100),
          backgroundColor: 'rgba(255, 206, 86, 0.8)',
          borderColor: 'rgba(255, 206, 86, 1)',
          borderWidth: 1
        }
      ]
    };
  }, [dashboardData]);

  // Performance distribution chart
  const performanceDistributionData = useMemo(() => {
    if (!dashboardData) return null;
    
    const grades = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F'];
    const gradeDistribution = grades.map(grade => 
      Object.values(dashboardData.agents).filter(agent => 
        getPerformanceGrade(agent.overall_score) === grade
      ).length
    );
    
    return {
      labels: grades,
      datasets: [{
        data: gradeDistribution,
        backgroundColor: grades.map(grade => getGradeColor(grade)),
        borderWidth: 2
      }]
    };
  }, [dashboardData]);

  // Team performance radar chart
  const teamPerformanceData = useMemo(() => {
    if (!dashboardData) return null;
    
    const sampleAgent = Object.values(dashboardData.agents)[0];
    if (!sampleAgent) return null;
    
    return {
      labels: ['Success Rate', 'Quality Score', 'Efficiency', 'Compliance', 'User Satisfaction'],
      datasets: [{
        label: sampleAgent.agent_name,
        data: [
          sampleAgent.success_rate * 100,
          sampleAgent.quality_score * 100,
          sampleAgent.efficiency_score * 100,
          sampleAgent.compliance_score * 100,
          sampleAgent.user_satisfaction_score * 20 // Convert to 0-100 scale
        ],
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(54, 162, 235, 1)'
      }]
    };
  }, [dashboardData]);

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
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <DashboardIcon fontSize="large" color="primary" />
          Agent Performance Metrics Dashboard
          {dashboardData && (
            <Badge badgeContent={dashboardData.aggregates.total_agents} color="primary">
              <GroupsIcon />
            </Badge>
          )}
        </Typography>

        <Box display="flex" gap={2} alignItems="center">
          {/* Real-time controls */}
          <FormControlLabel
            control={
              <Switch
                checked={realTimeMode}
                onChange={(e) => setRealTimeMode(e.target.checked)}
                color="primary"
              />
            }
            label="Real-time"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                disabled={!realTimeMode}
                color="secondary"
              />
            }
            label="Auto-refresh"
          />
          
          <IconButton onClick={loadDashboardData} disabled={loading}>
            <RefreshIcon />
          </IconButton>
          
          <Button
            variant="outlined"
            startIcon={<CloudDownloadIcon />}
            onClick={() => exportDashboard('pdf')}
            disabled={loading}
          >
            Export
          </Button>
        </Box>
      </Box>

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

      {/* Dashboard Controls */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                >
                  <MenuItem value="last_7_days">Last 7 Days</MenuItem>
                  <MenuItem value="last_30_days">Last 30 Days</MenuItem>
                  <MenuItem value="last_90_days">Last 90 Days</MenuItem>
                  <MenuItem value="last_year">Last Year</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Dashboard Type</InputLabel>
                <Select
                  value={dashboardType}
                  onChange={(e) => setDashboardType(e.target.value)}
                >
                  <MenuItem value="comprehensive">Comprehensive</MenuItem>
                  <MenuItem value="summary">Summary</MenuItem>
                  <MenuItem value="comparative">Comparative</MenuItem>
                  <MenuItem value="executive">Executive</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Sort By</InputLabel>
                <Select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                >
                  <MenuItem value="overall_score">Overall Score</MenuItem>
                  <MenuItem value="success_rate">Success Rate</MenuItem>
                  <MenuItem value="quality_score">Quality</MenuItem>
                  <MenuItem value="efficiency_score">Efficiency</MenuItem>
                  <MenuItem value="compliance_score">Compliance</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Filter Grade</InputLabel>
                <Select
                  value={filterGrade}
                  onChange={(e) => setFilterGrade(e.target.value)}
                >
                  <MenuItem value="all">All Grades</MenuItem>
                  <MenuItem value="A+">A+</MenuItem>
                  <MenuItem value="A">A</MenuItem>
                  <MenuItem value="B+">B+</MenuItem>
                  <MenuItem value="B">B</MenuItem>
                  <MenuItem value="C+">C+</MenuItem>
                  <MenuItem value="C">C</MenuItem>
                  <MenuItem value="D+">D+</MenuItem>
                  <MenuItem value="D">D</MenuItem>
                  <MenuItem value="F">F</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={2}>
              <Box display="flex" gap={1}>
                <IconButton 
                  size="small" 
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                >
                  <SortIcon />
                </IconButton>
                <IconButton size="small" onClick={() => setCompareDialog(true)}>
                  <CompareIcon />
                </IconButton>
                <IconButton size="small">
                  <FilterListIcon />
                </IconButton>
              </Box>
            </Grid>
          </Grid>
          
          {lastRefresh && (
            <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
              Last updated: {lastRefresh.toLocaleTimeString()}
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* KPI Summary Cards */}
      {dashboardData && (
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Total Agents
                    </Typography>
                    <Typography variant="h4" color="primary">
                      {dashboardData.aggregates.total_agents}
                    </Typography>
                  </Box>
                  <GroupsIcon fontSize="large" color="primary" />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Avg Performance
                    </Typography>
                    <Typography variant="h4" color="success.main">
                      {dashboardData.aggregates.avg_performance?.toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Grade: {getPerformanceGrade(dashboardData.aggregates.avg_performance || 0)}
                    </Typography>
                  </Box>
                  <SpeedIcon fontSize="large" color="success" />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Top Performer
                    </Typography>
                    <Typography variant="h6" color="warning.main">
                      {dashboardData.aggregates.top_performer || 'N/A'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Leading the team
                    </Typography>
                  </Box>
                  <StarIcon fontSize="large" color="warning" />
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Need Improvement
                    </Typography>
                    <Typography variant="h4" color="error.main">
                      {dashboardData.aggregates.improvement_needed || 0}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Agents below 75%
                    </Typography>
                  </Box>
                  <AssessmentIcon fontSize="large" color="error" />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Performance Alerts */}
      {performanceAlerts.length > 0 && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                <NotificationsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Active Performance Alerts
              </Typography>
              <Badge badgeContent={performanceAlerts.length} color="error">
                <NotificationsIcon />
              </Badge>
            </Box>
            
            <List dense>
              {performanceAlerts.slice(0, 5).map((alert) => (
                <ListItem key={alert.alert_id}>
                  <ListItemIcon>
                    <ErrorIcon style={{ color: getSeverityColor(alert.severity) }} />
                  </ListItemIcon>
                  <ListItemText
                    primary={alert.title}
                    secondary={
                      <Box>
                        <Typography variant="body2">{alert.description}</Typography>
                        <Typography variant="caption" color="textSecondary">
                          {new Date(alert.created_at).toLocaleString()}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <Chip 
                      label={alert.severity} 
                      size="small" 
                      style={{ backgroundColor: getSeverityColor(alert.severity), color: 'white' }}
                    />
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
            
            {performanceAlerts.length > 5 && (
              <Button size="small" onClick={() => setActiveTab(3)}>
                View All {performanceAlerts.length} Alerts
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {/* Performance Charts */}
      <Grid container spacing={4} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Agent Performance Comparison
              </Typography>
              {performanceTrendsData && (
                <Bar 
                  data={performanceTrendsData} 
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { position: 'top' as const },
                      title: { display: false }
                    },
                    scales: {
                      y: { beginAtZero: true, max: 100 }
                    }
                  }}
                  height={300}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Distribution
              </Typography>
              {performanceDistributionData && (
                <Doughnut 
                  data={performanceDistributionData} 
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { position: 'bottom' as const }
                    }
                  }}
                  height={300}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Content Tabs */}
      <Card>
        <Tabs 
          value={activeTab} 
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Agents Overview" icon={<GroupsIcon />} />
          <Tab label="Performance Analysis" icon={<AnalyticsIcon />} />
          <Tab label="Recommendations" icon={<RecommendIcon />} />
          <Tab label="Alerts & Monitoring" icon={<NotificationsIcon />} />
          <Tab label="Settings" icon={<SettingsIcon />} />
        </Tabs>

        {/* Agents Overview Tab */}
        <TabPanel value={activeTab} index={0}>
          <Typography variant="h6" gutterBottom>
            Agent Performance Overview
          </Typography>
          
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Agent</TableCell>
                  <TableCell>Overall Score</TableCell>
                  <TableCell>Grade</TableCell>
                  <TableCell>Success Rate</TableCell>
                  <TableCell>Quality</TableCell>
                  <TableCell>Efficiency</TableCell>
                  <TableCell>Compliance</TableCell>
                  <TableCell>Trend</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sortedAgents.map((agent) => (
                  <TableRow key={agent.agent_id}>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Avatar sx={{ width: 32, height: 32, fontSize: '0.875rem' }}>
                          {agent.agent_name.substring(0, 2).toUpperCase()}
                        </Avatar>
                        <Box>
                          <Typography variant="body2" fontWeight="bold">
                            {agent.agent_name}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            #{agent.rank}
                          </Typography>
                        </Box>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={1}>
                        <CircularProgress 
                          variant="determinate" 
                          value={agent.overall_score} 
                          size={24}
                          color={agent.overall_score >= 80 ? 'success' : 
                                agent.overall_score >= 60 ? 'warning' : 'error'}
                        />
                        <Typography variant="body2" fontWeight="bold">
                          {agent.overall_score.toFixed(1)}%
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={getPerformanceGrade(agent.overall_score)}
                        size="small"
                        style={{ 
                          backgroundColor: getGradeColor(getPerformanceGrade(agent.overall_score)),
                          color: 'white',
                          fontWeight: 'bold'
                        }}
                      />
                    </TableCell>
                    <TableCell>{(agent.success_rate * 100).toFixed(1)}%</TableCell>
                    <TableCell>{(agent.quality_score * 100).toFixed(1)}%</TableCell>
                    <TableCell>{(agent.efficiency_score * 100).toFixed(1)}%</TableCell>
                    <TableCell>{(agent.compliance_score * 100).toFixed(1)}%</TableCell>
                    <TableCell>{getTrendIcon(agent.trend)}</TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <Tooltip title="View Details">
                          <IconButton
                            size="small"
                            onClick={() => {
                              setSelectedAgent(agent);
                              setAgentDetailsDialog(true);
                            }}
                          >
                            <VisibilityIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Collect Metrics">
                          <IconButton
                            size="small"
                            onClick={() => collectAgentMetrics(agent.agent_id)}
                            disabled={loading}
                          >
                            <RefreshIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Generate Analysis">
                          <IconButton
                            size="small"
                            onClick={() => generateAnalysis(agent.agent_id)}
                            disabled={loading}
                          >
                            <AnalyticsIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
                {sortedAgents.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={9} align="center">
                      <Typography variant="body1" color="textSecondary" py={4}>
                        No agent data available. Load dashboard data to see performance metrics.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Performance Analysis Tab */}
        <TabPanel value={activeTab} index={1}>
          <Typography variant="h6" gutterBottom>
            Performance Analysis Results
          </Typography>
          
          {performanceAnalyses.length > 0 ? (
            <Grid container spacing={3}>
              {performanceAnalyses.map((analysis) => (
                <Grid item xs={12} md={6} key={analysis.analysis_id}>
                  <Card>
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                        <Typography variant="h6">
                          Agent Analysis
                        </Typography>
                        <Chip 
                          label={analysis.overall_grade} 
                          color={analysis.overall_grade.startsWith('A') ? 'success' : 
                                analysis.overall_grade.startsWith('B') ? 'warning' : 'error'}
                        />
                      </Box>
                      
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        Analysis ID: {analysis.analysis_id}
                      </Typography>
                      
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        Generated: {new Date(analysis.created_at).toLocaleString()}
                      </Typography>
                      
                      <Divider sx={{ my: 2 }} />
                      
                      <Typography variant="subtitle2" gutterBottom>
                        Key Insights:
                      </Typography>
                      <List dense>
                        {analysis.key_insights?.slice(0, 3).map((insight, idx) => (
                          <ListItem key={idx}>
                            <ListItemIcon>
                              <InsightsIcon fontSize="small" />
                            </ListItemIcon>
                            <ListItemText primary={insight} />
                          </ListItem>
                        ))}
                      </List>
                      
                      <Box display="flex" justifyContent="space-between" mt={2}>
                        <Typography variant="caption" color="textSecondary">
                          Processing time: {analysis.processing_time?.toFixed(2)}s
                        </Typography>
                        <Button 
                          size="small" 
                          onClick={() => {
                            setSelectedAgent({ agent_id: analysis.agent_id } as AgentMetric);
                            setAnalysisDialog(true);
                          }}
                        >
                          View Details
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box textAlign="center" py={4}>
              <PsychologyIcon fontSize="large" color="disabled" />
              <Typography variant="body1" color="textSecondary" mt={2}>
                No performance analyses available. Generate analysis for agents to see detailed insights.
              </Typography>
            </Box>
          )}
        </TabPanel>

        {/* Recommendations Tab */}
        <TabPanel value={activeTab} index={2}>
          <Typography variant="h6" gutterBottom>
            Optimization Recommendations
          </Typography>
          
          {optimizationRecommendations.length > 0 ? (
            <Grid container spacing={3}>
              {optimizationRecommendations.map((recommendation) => (
                <Grid item xs={12} md={6} key={recommendation.recommendation_id}>
                  <Card>
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                        <Typography variant="h6">
                          {recommendation.title}
                        </Typography>
                        {getPriorityChip(recommendation.priority)}
                      </Box>
                      
                      <Typography variant="body2" color="textSecondary" paragraph>
                        {recommendation.description}
                      </Typography>
                      
                      <Box display="flex" gap={2} mb={2}>
                        <Chip 
                          icon={<TrendingUpIcon />}
                          label={`${(recommendation.expected_impact * 100).toFixed(1)}% Impact`}
                          size="small" 
                          color="success"
                        />
                        <Chip 
                          icon={<Timeline2Icon />}
                          label={recommendation.implementation_effort}
                          size="small" 
                          variant="outlined"
                        />
                        <Chip 
                          icon={<EngineeringIcon />}
                          label={`${(recommendation.estimated_roi * 100).toFixed(0)}% ROI`}
                          size="small" 
                          color="warning"
                        />
                      </Box>
                      
                      <LinearProgress 
                        variant="determinate" 
                        value={recommendation.confidence_level * 100}
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="caption" color="textSecondary">
                        Confidence: {(recommendation.confidence_level * 100).toFixed(1)}%
                      </Typography>
                      
                      <Box display="flex" justifyContent="space-between" mt={2}>
                        <Typography variant="caption" color="textSecondary">
                          Created: {new Date(recommendation.created_at).toLocaleDateString()}
                        </Typography>
                        <Button 
                          size="small" 
                          startIcon={<VisibilityIcon />}
                          onClick={() => setRecommendationDialog(true)}
                        >
                          View Details
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box textAlign="center" py={4}>
              <EmojiObjectsIcon fontSize="large" color="disabled" />
              <Typography variant="body1" color="textSecondary" mt={2}>
                No optimization recommendations available. Generate recommendations for agents to see improvement suggestions.
              </Typography>
            </Box>
          )}
        </TabPanel>

        {/* Alerts & Monitoring Tab */}
        <TabPanel value={activeTab} index={3}>
          <Typography variant="h6" gutterBottom>
            Performance Alerts & Monitoring
          </Typography>
          
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Alert</TableCell>
                  <TableCell>Agent</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Metric</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {performanceAlerts.map((alert) => (
                  <TableRow key={alert.alert_id}>
                    <TableCell>
                      <Typography variant="body2" fontWeight="bold">
                        {alert.title}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {alert.description}
                      </Typography>
                    </TableCell>
                    <TableCell>{alert.agent_id}</TableCell>
                    <TableCell>
                      <Chip 
                        label={alert.severity} 
                        size="small" 
                        style={{ 
                          backgroundColor: getSeverityColor(alert.severity),
                          color: 'white'
                        }}
                      />
                    </TableCell>
                    <TableCell>{alert.metric_name}</TableCell>
                    <TableCell>
                      <Chip 
                        label={alert.alert_status} 
                        size="small" 
                        color={alert.alert_status === 'active' ? 'error' : 'default'}
                      />
                    </TableCell>
                    <TableCell>
                      {new Date(alert.created_at).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <Tooltip title="Acknowledge">
                          <IconButton size="small">
                            <CheckCircleIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="View Details">
                          <IconButton size="small">
                            <VisibilityIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
                {performanceAlerts.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={7} align="center">
                      <Box py={4}>
                        <CheckCircleIcon fontSize="large" color="success" />
                        <Typography variant="body1" color="textSecondary" mt={2}>
                          No active performance alerts. All agents are performing within expected parameters.
                        </Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Settings Tab */}
        <TabPanel value={activeTab} index={4}>
          <Typography variant="h6" gutterBottom>
            Dashboard Settings & Configuration
          </Typography>
          
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Real-time Monitoring
                  </Typography>
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={realTimeMode}
                        onChange={(e) => setRealTimeMode(e.target.checked)}
                      />
                    }
                    label="Enable Real-time Mode"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={autoRefresh}
                        onChange={(e) => setAutoRefresh(e.target.checked)}
                        disabled={!realTimeMode}
                      />
                    }
                    label="Auto-refresh Dashboard"
                  />
                  
                  <Box mt={2}>
                    <Typography gutterBottom>
                      Refresh Interval (seconds): {refreshInterval}
                    </Typography>
                    <Slider
                      value={refreshInterval}
                      onChange={(_, value) => setRefreshInterval(value as number)}
                      min={10}
                      max={300}
                      step={10}
                      marks={[
                        { value: 30, label: '30s' },
                        { value: 60, label: '1m' },
                        { value: 300, label: '5m' }
                      ]}
                      disabled={!realTimeMode}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Export & Reporting
                  </Typography>
                  
                  <Box display="flex" flexDirection="column" gap={2}>
                    <Button 
                      variant="outlined" 
                      startIcon={<CloudDownloadIcon />}
                      onClick={() => exportDashboard('pdf')}
                      disabled={loading}
                    >
                      Export as PDF
                    </Button>
                    <Button 
                      variant="outlined" 
                      startIcon={<CloudDownloadIcon />}
                      onClick={() => exportDashboard('xlsx')}
                      disabled={loading}
                    >
                      Export as Excel
                    </Button>
                    <Button 
                      variant="outlined" 
                      startIcon={<CloudDownloadIcon />}
                      onClick={() => exportDashboard('csv')}
                      disabled={loading}
                    >
                      Export as CSV
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Card>

      {/* Agent Details Dialog */}
      <Dialog
        open={agentDetailsDialog}
        onClose={() => setAgentDetailsDialog(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Agent Performance Details
        </DialogTitle>
        <DialogContent>
          {selectedAgent && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  {selectedAgent.agent_name}
                </Typography>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Agent ID: {selectedAgent.agent_id}
                </Typography>
                
                <Box mt={2}>
                  <Typography variant="subtitle2">Performance Metrics:</Typography>
                  <Box display="flex" flexWrap="wrap" gap={1} mt={1}>
                    <Chip 
                      label={`Overall: ${selectedAgent.overall_score?.toFixed(1)}%`}
                      color="primary"
                    />
                    <Chip 
                      label={`Success: ${(selectedAgent.success_rate * 100).toFixed(1)}%`}
                      color="success"
                    />
                    <Chip 
                      label={`Quality: ${(selectedAgent.quality_score * 100).toFixed(1)}%`}
                      color="info"
                    />
                    <Chip 
                      label={`Efficiency: ${(selectedAgent.efficiency_score * 100).toFixed(1)}%`}
                      color="warning"
                    />
                  </Box>
                </Box>
                
                <Box mt={2}>
                  <Typography variant="subtitle2">Task Statistics:</Typography>
                  <Typography variant="body2">
                    Total Tasks: {selectedAgent.total_tasks}
                  </Typography>
                  <Typography variant="body2">
                    Completed: {selectedAgent.completed_tasks}
                  </Typography>
                  <Typography variant="body2">
                    Failed: {selectedAgent.failed_tasks}
                  </Typography>
                  <Typography variant="body2">
                    Avg Processing Time: {selectedAgent.avg_processing_time?.toFixed(2)}s
                  </Typography>
                </Box>
              </Grid>
              
              <Grid item xs={12} md={6}>
                {teamPerformanceData && (
                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Performance Radar:
                    </Typography>
                    <Radar 
                      data={teamPerformanceData}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          r: {
                            beginAtZero: true,
                            max: 100
                          }
                        }
                      }}
                      height={250}
                    />
                  </Box>
                )}
              </Grid>
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAgentDetailsDialog(false)}>Close</Button>
          <Button 
            variant="contained" 
            startIcon={<AnalyticsIcon />}
            onClick={() => {
              if (selectedAgent) {
                generateAnalysis(selectedAgent.agent_id);
                setAgentDetailsDialog(false);
              }
            }}
          >
            Generate Analysis
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default AgentPerformanceMetrics;
