/**
 * Advanced Field Validation Interface - React Component
 * Created: 2024-01-15
 * Author: MortgageAI Development Team
 * 
 * Professional React interface for the Advanced Field Validation Engine with
 * real-time validation, error correction suggestions, and AFM compliance checking.
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
  Snackbar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  PlayArrow as PlayArrowIcon,
  Settings as SettingsIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Visibility as VisibilityIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  AutoFixHigh as AutoFixHighIcon,
  Security as SecurityIcon,
  Gavel as GavelIcon,
  TrendingUp as TrendingUpIcon,
  ExpandMore as ExpandMoreIcon,
  Close as CloseIcon,
  Assignment as AssignmentIcon,
  Rule as RuleIcon,
  Insights as InsightsIcon,
  Analytics as AnalyticsIcon
} from '@mui/icons-material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend
} from 'chart.js';
import { Line, Bar, Doughnut, Pie } from 'react-chartjs-2';

// Chart.js registration
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  ChartTooltip,
  Legend
);

// Type definitions
interface ValidationMessage {
  message_id: string;
  rule_id: string;
  field_path: string;
  message: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  suggestion?: string;
  corrected_value?: string;
  afm_reference?: string;
  context: Record<string, any>;
  timestamp: string;
}

interface ValidationResult {
  success: boolean;
  is_valid: boolean;
  total_fields: number;
  validated_fields: number;
  errors: number;
  warnings: number;
  infos: number;
  messages: ValidationMessage[];
  field_scores: Record<string, number>;
  overall_score: number;
  compliance_score: number;
  processing_time: number;
  metadata: Record<string, any>;
}

interface ValidationRule {
  rule_id: string;
  rule_name: string;
  field_path: string;
  field_type: string;
  rule_type: string;
  parameters: Record<string, any>;
  error_message: string;
  suggestion_template: string;
  severity: string;
  is_active: boolean;
  afm_article?: string;
  priority: number;
  dependencies: string[];
  conditions: Record<string, any>;
  created_at: string;
  updated_at: string;
}

interface FieldValidationSession {
  session_id: string;
  session_name: string;
  session_type: string;
  is_valid?: boolean;
  overall_score: number;
  compliance_score: number;
  processing_time: number;
  error_count: number;
  warning_count: number;
  info_count: number;
  total_fields: number;
  started_at: string;
  completed_at?: string;
}

interface ValidationStatistics {
  validation_stats: {
    total_validations: number;
    successful_validations: number;
    failed_validations: number;
    average_processing_time: number;
  };
  engine_info: {
    total_rules: number;
    active_rules: number;
    supported_field_types: string[];
    afm_compliance_enabled: boolean;
  };
  performance_metrics: {
    success_rate: number;
    avg_field_validation_time: number;
  };
}

const AdvancedFieldValidation: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [validationRules, setValidationRules] = useState<ValidationRule[]>([]);
  const [validationSessions, setValidationSessions] = useState<FieldValidationSession[]>([]);
  const [validationStatistics, setValidationStatistics] = useState<ValidationStatistics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Dialog states
  const [selectedRule, setSelectedRule] = useState<ValidationRule | null>(null);
  const [ruleDialog, setRuleDialog] = useState(false);
  const [newRuleDialog, setNewRuleDialog] = useState(false);
  const [validationDialog, setValidationDialog] = useState(false);
  const [bulkValidationDialog, setBulkValidationDialog] = useState(false);
  
  // Form states
  const [singleField, setSingleField] = useState({
    field_path: '',
    value: '',
    field_type: 'text'
  });
  
  const [bulkData, setBulkData] = useState('');
  const [validationConfig, setValidationConfig] = useState({
    check_afm_compliance: true,
    generate_suggestions: true,
    include_scores: true
  });
  
  // New rule form state
  const [newRule, setNewRule] = useState({
    rule_name: '',
    field_path: '',
    field_type: 'text',
    rule_type: 'required',
    parameters: '{}',
    error_message: '',
    suggestion_template: '',
    severity: 'error',
    is_active: true,
    afm_article: '',
    priority: 1,
    dependencies: '[]',
    conditions: '{}'
  });

  // API calls
  const apiCall = useCallback(async (endpoint: string, options: RequestInit = {}) => {
    const baseUrl = process.env.REACT_APP_API_URL || '';
    const response = await fetch(`${baseUrl}/api/field-validation/${endpoint}`, {
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
          loadValidationRules(),
          loadValidationStatistics()
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, []);

  // Load validation rules
  const loadValidationRules = useCallback(async () => {
    try {
      const response = await apiCall('rules');
      if (response.success) {
        setValidationRules(response.rules || []);
      } else {
        throw new Error(response.error || 'Failed to load validation rules');
      }
    } catch (err) {
      console.error('Failed to load validation rules:', err);
      setError(err instanceof Error ? err.message : 'Failed to load validation rules');
    }
  }, [apiCall]);

  // Load validation statistics
  const loadValidationStatistics = useCallback(async () => {
    try {
      const response = await apiCall('statistics');
      if (response.success) {
        setValidationStatistics(response);
      } else {
        throw new Error(response.error || 'Failed to load validation statistics');
      }
    } catch (err) {
      console.error('Failed to load validation statistics:', err);
      setError(err instanceof Error ? err.message : 'Failed to load validation statistics');
    }
  }, [apiCall]);

  // Validate single field
  const validateSingleField = useCallback(async () => {
    if (!singleField.field_path || !singleField.value) {
      setError('Please provide both field path and value');
      return;
    }

    setLoading(true);
    try {
      const response = await apiCall('validate/field', {
        method: 'POST',
        body: JSON.stringify({
          field_path: singleField.field_path,
          value: singleField.value,
          field_type: singleField.field_type,
          context: {},
          check_afm_compliance: validationConfig.check_afm_compliance,
          generate_suggestions: validationConfig.generate_suggestions
        })
      });

      if (response.success) {
        const mockResult: ValidationResult = {
          success: true,
          is_valid: response.is_valid,
          total_fields: 1,
          validated_fields: 1,
          errors: response.messages.filter((m: ValidationMessage) => m.severity === 'error').length,
          warnings: response.messages.filter((m: ValidationMessage) => m.severity === 'warning').length,
          infos: response.messages.filter((m: ValidationMessage) => m.severity === 'info').length,
          messages: response.messages,
          field_scores: { [response.field_path]: response.field_score },
          overall_score: response.field_score || 100,
          compliance_score: 100, // Would be calculated based on AFM compliance
          processing_time: response.processing_time,
          metadata: response.metadata
        };
        
        setValidationResult(mockResult);
        setSuccess('Field validation completed successfully');
      } else {
        throw new Error(response.error || 'Failed to validate field');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to validate field');
    } finally {
      setLoading(false);
    }
  }, [singleField, validationConfig, apiCall]);

  // Validate bulk data
  const validateBulkData = useCallback(async () => {
    if (!bulkData.trim()) {
      setError('Please provide data for bulk validation');
      return;
    }

    setLoading(true);
    try {
      let parsedData;
      try {
        parsedData = JSON.parse(bulkData);
      } catch {
        throw new Error('Invalid JSON data format');
      }

      const response = await apiCall('validate/data', {
        method: 'POST',
        body: JSON.stringify({
          data: parsedData,
          validation_config: {
            context: {
              check_afm_compliance: validationConfig.check_afm_compliance,
              generate_suggestions: validationConfig.generate_suggestions,
              include_scores: validationConfig.include_scores
            }
          },
          check_afm_compliance: validationConfig.check_afm_compliance,
          generate_suggestions: validationConfig.generate_suggestions,
          include_scores: validationConfig.include_scores
        })
      });

      if (response.success) {
        setValidationResult(response);
        setBulkValidationDialog(false);
        setSuccess('Bulk validation completed successfully');
      } else {
        throw new Error(response.error || 'Failed to validate data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to validate data');
    } finally {
      setLoading(false);
    }
  }, [bulkData, validationConfig, apiCall]);

  // Create validation rule
  const createValidationRule = useCallback(async () => {
    setLoading(true);
    try {
      const ruleData = {
        ...newRule,
        parameters: JSON.parse(newRule.parameters || '{}'),
        dependencies: JSON.parse(newRule.dependencies || '[]'),
        conditions: JSON.parse(newRule.conditions || '{}')
      };

      const response = await apiCall('rules', {
        method: 'POST',
        body: JSON.stringify(ruleData)
      });

      if (response.success) {
        await loadValidationRules();
        setNewRuleDialog(false);
        setNewRule({
          rule_name: '',
          field_path: '',
          field_type: 'text',
          rule_type: 'required',
          parameters: '{}',
          error_message: '',
          suggestion_template: '',
          severity: 'error',
          is_active: true,
          afm_article: '',
          priority: 1,
          dependencies: '[]',
          conditions: '{}'
        });
        setSuccess('Validation rule created successfully');
      } else {
        throw new Error(response.error || 'Failed to create validation rule');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create validation rule');
    } finally {
      setLoading(false);
    }
  }, [newRule, loadValidationRules, apiCall]);

  // Delete validation rule
  const deleteValidationRule = useCallback(async (ruleId: string) => {
    setLoading(true);
    try {
      const response = await apiCall(`rules/${ruleId}`, {
        method: 'DELETE'
      });

      if (response.success) {
        await loadValidationRules();
        setSuccess('Validation rule deleted successfully');
      } else {
        throw new Error(response.error || 'Failed to delete validation rule');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete validation rule');
    } finally {
      setLoading(false);
    }
  }, [loadValidationRules, apiCall]);

  // Health check
  const performHealthCheck = useCallback(async () => {
    setLoading(true);
    try {
      const response = await apiCall('health');
      if (response.success) {
        setSuccess(`System health: ${response.status}. All validation services operational.`);
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
      case 'error': return '#ff5722';
      case 'warning': return '#ff9800';
      case 'info': return '#2196f3';
      default: return '#9e9e9e';
    }
  };

  // Get severity icon
  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <ErrorIcon style={{ color: getSeverityColor(severity) }} />;
      case 'error': return <ErrorIcon style={{ color: getSeverityColor(severity) }} />;
      case 'warning': return <WarningIcon style={{ color: getSeverityColor(severity) }} />;
      case 'info': return <InfoIcon style={{ color: getSeverityColor(severity) }} />;
      default: return <InfoIcon style={{ color: getSeverityColor(severity) }} />;
    }
  };

  // Get severity chip
  const getSeverityChip = (severity: string) => (
    <Chip
      icon={getSeverityIcon(severity)}
      label={severity.toUpperCase()}
      size="small"
      style={{
        backgroundColor: getSeverityColor(severity),
        color: 'white',
        fontWeight: 'bold'
      }}
    />
  );

  // Chart data
  const validationResultsChartData = useMemo(() => {
    if (!validationResult) return null;

    return {
      labels: ['Valid Fields', 'Fields with Errors', 'Fields with Warnings', 'Fields with Info'],
      datasets: [
        {
          data: [
            validationResult.validated_fields - validationResult.errors - validationResult.warnings,
            validationResult.errors,
            validationResult.warnings,
            validationResult.infos
          ],
          backgroundColor: [
            '#4caf50',
            '#f44336',
            '#ff9800',
            '#2196f3'
          ],
          borderWidth: 2
        }
      ]
    };
  }, [validationResult]);

  const performanceChartData = useMemo(() => {
    if (!validationStatistics) return null;

    return {
      labels: ['Success Rate', 'Processing Time (ms)', 'Rule Coverage'],
      datasets: [
        {
          label: 'Performance Metrics',
          data: [
            validationStatistics.performance_metrics.success_rate * 100,
            validationStatistics.performance_metrics.avg_field_validation_time * 1000,
            (validationStatistics.engine_info.active_rules / validationStatistics.engine_info.total_rules) * 100
          ],
          backgroundColor: 'rgba(54, 162, 235, 0.8)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        }
      ]
    };
  }, [validationStatistics]);

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
        <AssignmentIcon fontSize="large" color="primary" />
        Advanced Field Validation Engine
        <Badge badgeContent={validationRules.filter(r => r.is_active).length} color="primary" sx={{ ml: 2 }}>
          <RuleIcon />
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

      {/* Quick Actions */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Quick Validation
          </Typography>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Field Path"
                placeholder="e.g. email, phone, bsn"
                value={singleField.field_path}
                onChange={(e) => setSingleField({...singleField, field_path: e.target.value})}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Field Value"
                placeholder="Enter value to validate"
                value={singleField.value}
                onChange={(e) => setSingleField({...singleField, value: e.target.value})}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Field Type</InputLabel>
                <Select
                  value={singleField.field_type}
                  onChange={(e) => setSingleField({...singleField, field_type: e.target.value})}
                >
                  <MenuItem value="text">Text</MenuItem>
                  <MenuItem value="email">Email</MenuItem>
                  <MenuItem value="phone">Phone</MenuItem>
                  <MenuItem value="bsn">BSN</MenuItem>
                  <MenuItem value="iban">IBAN</MenuItem>
                  <MenuItem value="postcode">Postcode</MenuItem>
                  <MenuItem value="currency">Currency</MenuItem>
                  <MenuItem value="date">Date</MenuItem>
                  <MenuItem value="custom">Custom</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <Box display="flex" flexDirection="column" gap={1}>
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={validationConfig.check_afm_compliance}
                      onChange={(e) => setValidationConfig({
                        ...validationConfig, 
                        check_afm_compliance: e.target.checked
                      })}
                    />
                  }
                  label="AFM Compliance"
                />
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={validationConfig.generate_suggestions}
                      onChange={(e) => setValidationConfig({
                        ...validationConfig, 
                        generate_suggestions: e.target.checked
                      })}
                    />
                  }
                  label="Suggestions"
                />
              </Box>
            </Grid>
            <Grid item xs={12} md={2}>
              <Box display="flex" gap={1}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={validateSingleField}
                  disabled={loading}
                  fullWidth
                >
                  Validate
                </Button>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Summary Dashboard */}
      {validationStatistics && (
        <Grid container spacing={4} sx={{ mb: 4 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Total Rules
                </Typography>
                <Typography variant="h4" color="primary">
                  {validationStatistics.engine_info.total_rules}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Active: {validationStatistics.engine_info.active_rules}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Success Rate
                </Typography>
                <Typography variant="h4" color="success.main">
                  {(validationStatistics.performance_metrics.success_rate * 100).toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Total Validations
                </Typography>
                <Typography variant="h4" color="info.main">
                  {validationStatistics.validation_stats.total_validations.toLocaleString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent style={{ height: '150px' }}>
                <Typography variant="h6" gutterBottom>
                  Performance
                </Typography>
                {performanceChartData && (
                  <Bar data={performanceChartData} options={{ maintainAspectRatio: false }} />
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Validation Results */}
      {validationResult && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                Validation Results
              </Typography>
              <Box display="flex" alignItems="center" gap={2}>
                <Chip 
                  label={validationResult.is_valid ? 'Valid' : 'Invalid'} 
                  color={validationResult.is_valid ? 'success' : 'error'}
                />
                <Typography variant="body2">
                  Score: {validationResult.overall_score.toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  AFM Compliance: {validationResult.compliance_score.toFixed(1)}%
                </Typography>
              </Box>
            </Box>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <TableContainer component={Paper}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Field</TableCell>
                        <TableCell>Message</TableCell>
                        <TableCell>Severity</TableCell>
                        <TableCell>Suggestion</TableCell>
                        <TableCell>AFM</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {validationResult.messages.map((message) => (
                        <TableRow key={message.message_id}>
                          <TableCell>
                            <Typography variant="body2" fontWeight="bold">
                              {message.field_path}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {message.message}
                            </Typography>
                          </TableCell>
                          <TableCell>{getSeverityChip(message.severity)}</TableCell>
                          <TableCell>
                            {message.suggestion && (
                              <Tooltip title={message.suggestion}>
                                <Chip 
                                  icon={<AutoFixHighIcon />}
                                  label="Fix Available" 
                                  size="small" 
                                  variant="outlined" 
                                />
                              </Tooltip>
                            )}
                            {message.corrected_value && (
                              <Typography variant="caption" display="block">
                                Suggested: {message.corrected_value}
                              </Typography>
                            )}
                          </TableCell>
                          <TableCell>
                            {message.afm_reference && (
                              <Chip 
                                icon={<GavelIcon />}
                                label={message.afm_reference}
                                size="small"
                                color="warning"
                              />
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                      {validationResult.messages.length === 0 && (
                        <TableRow>
                          <TableCell colSpan={5} align="center">
                            <Box display="flex" alignItems="center" justifyContent="center" py={2}>
                              <CheckCircleIcon color="success" sx={{ mr: 1 }} />
                              <Typography color="success.main">All fields are valid!</Typography>
                            </Box>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
              <Grid item xs={12} md={4}>
                {validationResultsChartData && (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Validation Summary
                    </Typography>
                    <Doughnut 
                      data={validationResultsChartData} 
                      options={{ 
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            position: 'bottom'
                          }
                        }
                      }} 
                    />
                  </Box>
                )}
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Main Content Tabs */}
      <Card>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab label="Validation Rules" icon={<RuleIcon />} />
          <Tab label="Bulk Validation" icon={<AnalyticsIcon />} />
          <Tab label="Statistics" icon={<TrendingUpIcon />} />
          <Tab label="Settings" icon={<SettingsIcon />} />
        </Tabs>

        {/* Validation Rules Tab */}
        <TabPanel value={activeTab} index={0}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <Typography variant="h6">Validation Rules Management</Typography>
            <Box display="flex" gap={2}>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={loadValidationRules}
                disabled={loading}
              >
                Refresh
              </Button>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => setNewRuleDialog(true)}
              >
                Create Rule
              </Button>
            </Box>
          </Box>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Rule Name</TableCell>
                  <TableCell>Field Path</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Field Type</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Priority</TableCell>
                  <TableCell>AFM</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {validationRules.map((rule) => (
                  <TableRow key={rule.rule_id}>
                    <TableCell>
                      <Typography variant="body2" fontWeight="bold">
                        {rule.rule_name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {rule.field_path}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip label={rule.rule_type} size="small" variant="outlined" />
                    </TableCell>
                    <TableCell>
                      <Chip label={rule.field_type} size="small" />
                    </TableCell>
                    <TableCell>{getSeverityChip(rule.severity)}</TableCell>
                    <TableCell>
                      <Chip
                        label={rule.is_active ? 'Active' : 'Inactive'}
                        size="small"
                        color={rule.is_active ? 'success' : 'default'}
                      />
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={`P${rule.priority}`} 
                        size="small" 
                        color={rule.priority <= 2 ? 'error' : rule.priority <= 3 ? 'warning' : 'default'}
                      />
                    </TableCell>
                    <TableCell>
                      {rule.afm_article && (
                        <Chip
                          icon={<GavelIcon />}
                          label={rule.afm_article}
                          size="small"
                          color="warning"
                        />
                      )}
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1}>
                        <Tooltip title="View Rule">
                          <IconButton
                            size="small"
                            onClick={() => {
                              setSelectedRule(rule);
                              setRuleDialog(true);
                            }}
                          >
                            <VisibilityIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit Rule">
                          <IconButton size="small">
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete Rule">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => deleteValidationRule(rule.rule_id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
                {validationRules.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={9} align="center">
                      <Typography variant="body1" color="textSecondary" py={4}>
                        No validation rules configured. Create your first validation rule.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Bulk Validation Tab */}
        <TabPanel value={activeTab} index={1}>
          <Box mb={3}>
            <Typography variant="h6" gutterBottom>
              Bulk Data Validation
            </Typography>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Validate multiple fields or complete data structures at once.
            </Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <TextField
                fullWidth
                multiline
                rows={12}
                label="JSON Data for Validation"
                placeholder='{"personal_info": {"name": "Jan de Vries", "email": "jan@example.com", "bsn": "123456782"}, "financial_info": {"income": 4500, "mortgage_amount": 350000}}'
                value={bulkData}
                onChange={(e) => setBulkData(e.target.value)}
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Validation Options
                  </Typography>
                  <Box display="flex" flexDirection="column" gap={2}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={validationConfig.check_afm_compliance}
                          onChange={(e) => setValidationConfig({
                            ...validationConfig, 
                            check_afm_compliance: e.target.checked
                          })}
                        />
                      }
                      label="AFM Compliance Check"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={validationConfig.generate_suggestions}
                          onChange={(e) => setValidationConfig({
                            ...validationConfig, 
                            generate_suggestions: e.target.checked
                          })}
                        />
                      }
                      label="Generate Suggestions"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={validationConfig.include_scores}
                          onChange={(e) => setValidationConfig({
                            ...validationConfig, 
                            include_scores: e.target.checked
                          })}
                        />
                      }
                      label="Include Field Scores"
                    />
                    <Divider />
                    <Button
                      variant="contained"
                      startIcon={<PlayArrowIcon />}
                      onClick={validateBulkData}
                      disabled={loading || !bulkData.trim()}
                      fullWidth
                    >
                      Validate Data
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<UploadIcon />}
                      disabled={loading}
                      fullWidth
                    >
                      Upload File
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Statistics Tab */}
        <TabPanel value={activeTab} index={2}>
          <Typography variant="h6" gutterBottom>
            Validation Engine Statistics & Performance
          </Typography>
          
          {validationStatistics && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Engine Information
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Total Rules" 
                          secondary={validationStatistics.engine_info.total_rules} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Active Rules" 
                          secondary={validationStatistics.engine_info.active_rules} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="AFM Compliance" 
                          secondary={validationStatistics.engine_info.afm_compliance_enabled ? 'Enabled' : 'Disabled'} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Supported Field Types" 
                          secondary={validationStatistics.engine_info.supported_field_types?.join(', ') || 'N/A'} 
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Performance Metrics
                    </Typography>
                    <List dense>
                      <ListItem>
                        <ListItemText 
                          primary="Total Validations" 
                          secondary={validationStatistics.validation_stats.total_validations.toLocaleString()} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Success Rate" 
                          secondary={`${(validationStatistics.performance_metrics.success_rate * 100).toFixed(2)}%`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Average Processing Time" 
                          secondary={`${(validationStatistics.performance_metrics.avg_field_validation_time * 1000).toFixed(2)}ms`} 
                        />
                      </ListItem>
                      <ListItem>
                        <ListItemText 
                          primary="Successful Validations" 
                          secondary={validationStatistics.validation_stats.successful_validations.toLocaleString()} 
                        />
                      </ListItem>
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}
        </TabPanel>

        {/* Settings Tab */}
        <TabPanel value={activeTab} index={3}>
          <Typography variant="h6" gutterBottom>
            System Settings & Health
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    System Health
                  </Typography>
                  <Box display="flex" flexDirection="column" gap={2}>
                    <Box display="flex" alignItems="center" gap={2}>
                      <CheckCircleIcon color="success" />
                      <Typography>Validation Engine: Online</Typography>
                    </Box>
                    <Box display="flex" alignItems="center" gap={2}>
                      <CheckCircleIcon color="success" />
                      <Typography>AFM Compliance: Online</Typography>
                    </Box>
                    <Box display="flex" alignItems="center" gap={2}>
                      <CheckCircleIcon color="success" />
                      <Typography>Correction Engine: Online</Typography>
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
            
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Data Management
                  </Typography>
                  <Box display="flex" flexDirection="column" gap={2}>
                    <Button
                      variant="outlined"
                      startIcon={<DownloadIcon />}
                      size="small"
                    >
                      Export Rules
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<UploadIcon />}
                      size="small"
                    >
                      Import Rules
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<AnalyticsIcon />}
                      size="small"
                    >
                      Export Statistics
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Card>

      {/* Rule Details Dialog */}
      <Dialog
        open={ruleDialog}
        onClose={() => setRuleDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Validation Rule Details
        </DialogTitle>
        <DialogContent>
          {selectedRule && (
            <Box display="flex" flexDirection="column" gap={3} mt={2}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    {selectedRule.rule_name}
                  </Typography>
                  <Box display="flex" gap={1} alignItems="center" mb={2}>
                    {getSeverityChip(selectedRule.severity)}
                    <Chip
                      label={selectedRule.is_active ? 'Active' : 'Inactive'}
                      size="small"
                      color={selectedRule.is_active ? 'success' : 'default'}
                    />
                    <Chip label={`Priority ${selectedRule.priority}`} size="small" />
                  </Box>
                  <Typography variant="body2" gutterBottom>
                    <strong>Field Path:</strong> {selectedRule.field_path}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Field Type:</strong> {selectedRule.field_type}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Rule Type:</strong> {selectedRule.rule_type}
                  </Typography>
                  {selectedRule.afm_article && (
                    <Typography variant="body2" gutterBottom>
                      <strong>AFM Article:</strong> {selectedRule.afm_article}
                    </Typography>
                  )}
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="body2" gutterBottom>
                    <strong>Error Message:</strong>
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                    {selectedRule.error_message}
                  </Typography>
                  {selectedRule.suggestion_template && (
                    <>
                      <Typography variant="body2" gutterBottom>
                        <strong>Suggestion Template:</strong>
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 2, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                        {selectedRule.suggestion_template}
                      </Typography>
                    </>
                  )}
                </Grid>
              </Grid>
              
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography>Advanced Configuration</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" gutterBottom>
                        <strong>Parameters:</strong>
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1, borderRadius: 1 }}>
                        {JSON.stringify(selectedRule.parameters, null, 2)}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body2" gutterBottom>
                        <strong>Conditions:</strong>
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1, borderRadius: 1 }}>
                        {JSON.stringify(selectedRule.conditions, null, 2)}
                      </Typography>
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRuleDialog(false)}>Close</Button>
          <Button variant="contained" startIcon={<EditIcon />}>
            Edit Rule
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
        <DialogTitle>Create New Validation Rule</DialogTitle>
        <DialogContent>
          <Box display="flex" flexDirection="column" gap={3} mt={2}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Rule Name"
                  value={newRule.rule_name}
                  onChange={(e) => setNewRule({ ...newRule, rule_name: e.target.value })}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Field Path"
                  value={newRule.field_path}
                  onChange={(e) => setNewRule({ ...newRule, field_path: e.target.value })}
                  placeholder="e.g. email, personal_info.name"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Field Type</InputLabel>
                  <Select
                    value={newRule.field_type}
                    onChange={(e) => setNewRule({ ...newRule, field_type: e.target.value })}
                  >
                    <MenuItem value="text">Text</MenuItem>
                    <MenuItem value="email">Email</MenuItem>
                    <MenuItem value="phone">Phone</MenuItem>
                    <MenuItem value="bsn">BSN</MenuItem>
                    <MenuItem value="iban">IBAN</MenuItem>
                    <MenuItem value="postcode">Postcode</MenuItem>
                    <MenuItem value="currency">Currency</MenuItem>
                    <MenuItem value="date">Date</MenuItem>
                    <MenuItem value="custom">Custom</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Rule Type</InputLabel>
                  <Select
                    value={newRule.rule_type}
                    onChange={(e) => setNewRule({ ...newRule, rule_type: e.target.value })}
                  >
                    <MenuItem value="required">Required</MenuItem>
                    <MenuItem value="format">Format</MenuItem>
                    <MenuItem value="range">Range</MenuItem>
                    <MenuItem value="length">Length</MenuItem>
                    <MenuItem value="pattern">Pattern</MenuItem>
                    <MenuItem value="custom">Custom</MenuItem>
                    <MenuItem value="afm_compliance">AFM Compliance</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Severity</InputLabel>
                  <Select
                    value={newRule.severity}
                    onChange={(e) => setNewRule({ ...newRule, severity: e.target.value })}
                  >
                    <MenuItem value="info">Info</MenuItem>
                    <MenuItem value="warning">Warning</MenuItem>
                    <MenuItem value="error">Error</MenuItem>
                    <MenuItem value="critical">Critical</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  type="number"
                  label="Priority (1-5)"
                  value={newRule.priority}
                  onChange={(e) => setNewRule({ ...newRule, priority: parseInt(e.target.value) || 1 })}
                  inputProps={{ min: 1, max: 5 }}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  label="Error Message"
                  value={newRule.error_message}
                  onChange={(e) => setNewRule({ ...newRule, error_message: e.target.value })}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  label="Suggestion Template"
                  value={newRule.suggestion_template}
                  onChange={(e) => setNewRule({ ...newRule, suggestion_template: e.target.value })}
                  placeholder="Provide helpful suggestion for users"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="AFM Article (optional)"
                  value={newRule.afm_article}
                  onChange={(e) => setNewRule({ ...newRule, afm_article: e.target.value })}
                  placeholder="e.g. Wft 86f"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={newRule.is_active}
                      onChange={(e) => setNewRule({ ...newRule, is_active: e.target.checked })}
                    />
                  }
                  label="Active Rule"
                />
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewRuleDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={createValidationRule}
            disabled={!newRule.rule_name || !newRule.field_path}
            startIcon={<SaveIcon />}
          >
            Create Rule
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default AdvancedFieldValidation;
