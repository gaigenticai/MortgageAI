import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent
} from './ui/card';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger
} from './ui/tabs';
import {
  Button
} from './ui/button';
import {
  Badge
} from './ui/badge';
import {
  Progress
} from './ui/progress';
import {
  Alert,
  AlertDescription,
  AlertTitle
} from './ui/alert';
import {
  Input
} from './ui/input';
import {
  Label
} from './ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from './ui/select';
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Shield,
  Target,
  BarChart3,
  Activity,
  Bell,
  Zap,
  Brain,
  AlertTriangle,
  RefreshCw,
  Eye,
  Settings,
  Download,
  Calculator,
  Gauge,
  LineChart,
  PieChart,
  Users,
  DollarSign,
  Percent,
  Clock,
  Database,
  Lightbulb,
  Layers,
  Crosshair
} from 'lucide-react';

/**
 * Advanced Risk Assessment Engine Component
 * Comprehensive UI for multi-factor risk analysis and mitigation
 */
const RiskAssessmentEngine = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [riskAssessments, setRiskAssessments] = useState([]);
  const [stressTestResults, setStressTestResults] = useState([]);
  const [mitigationStrategies, setMitigationStrategies] = useState([]);
  const [riskAlerts, setRiskAlerts] = useState([]);
  const [portfolioMetrics, setPortfolioMetrics] = useState({});
  const [modelPerformance, setModelPerformance] = useState([]);
  const [benchmarks, setBenchmarks] = useState([]);
  const [analytics, setAnalytics] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [realTimeData, setRealTimeData] = useState({});

  // Assessment form state
  const [assessmentForm, setAssessmentForm] = useState({
    entity_id: '',
    entity_type: 'mortgage_application',
    credit_score: 720,
    loan_amount: 300000,
    property_value: 400000,
    net_monthly_income: 5000,
    monthly_debt_payments: 800,
    employment_years: 5
  });

  // WebSocket connection
  const wsRef = useRef(null);

  // Initialize component
  useEffect(() => {
    initializeComponent();
    setupWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  /**
   * Initialize component data
   */
  const initializeComponent = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadRiskAssessments(),
        loadRiskAlerts(),
        loadModelPerformance(),
        loadAnalytics(),
        loadBenchmarks()
      ]);
    } catch (err) {
      setError('Failed to initialize Risk Assessment Engine');
      console.error('Initialization error:', err);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Setup WebSocket connection for real-time updates
   */
  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket(`ws://${window.location.hostname}:8013`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected to Risk Assessment Engine');
        // Subscribe to updates
        wsRef.current.send(JSON.stringify({ type: 'subscribe_risk_assessments' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_alerts' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_stress_tests' }));
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (err) {
          console.error('WebSocket message error:', err);
        }
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected, attempting to reconnect...');
        setTimeout(setupWebSocket, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (err) {
      console.error('WebSocket setup error:', err);
    }
  };

  /**
   * Handle WebSocket messages
   */
  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'real_time_dashboard':
        setRealTimeData(data.data);
        break;
      case 'risk_assessment_completed':
        loadRiskAssessments();
        showNotification(`Risk assessment completed for ${data.data.entity_id}`, 'success');
        break;
      case 'risk_alert_generated':
        loadRiskAlerts();
        showNotification(`High risk alert generated`, 'warning');
        break;
      case 'stress_test_completed':
        loadStressTestResults();
        showNotification('Stress test completed', 'success');
        break;
      default:
        console.log('Unknown WebSocket message:', data);
    }
  };

  /**
   * Load risk assessments
   */
  const loadRiskAssessments = async () => {
    try {
      const response = await fetch('/api/risk-assessment/assessments?limit=20');
      const data = await response.json();
      
      if (data.success) {
        setRiskAssessments(data.assessments);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load risk assessments:', err);
      throw err;
    }
  };

  /**
   * Load risk alerts
   */
  const loadRiskAlerts = async () => {
    try {
      const response = await fetch('/api/risk-assessment/alerts');
      const data = await response.json();
      
      if (data.success) {
        setRiskAlerts(data.alerts);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load risk alerts:', err);
      throw err;
    }
  };

  /**
   * Load model performance
   */
  const loadModelPerformance = async () => {
    try {
      const response = await fetch('/api/risk-assessment/model-performance');
      const data = await response.json();
      
      if (data.success) {
        setModelPerformance(data.model_performance);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load model performance:', err);
      throw err;
    }
  };

  /**
   * Load analytics
   */
  const loadAnalytics = async () => {
    try {
      const response = await fetch('/api/risk-assessment/analytics/dashboard');
      const data = await response.json();
      
      if (data.success) {
        setAnalytics(data.analytics);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load analytics:', err);
      throw err;
    }
  };

  /**
   * Load benchmarks
   */
  const loadBenchmarks = async () => {
    try {
      const response = await fetch('/api/risk-assessment/benchmarks');
      const data = await response.json();
      
      if (data.success) {
        setBenchmarks(data.benchmarks);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load benchmarks:', err);
      throw err;
    }
  };

  /**
   * Perform risk assessment
   */
  const performRiskAssessment = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/risk-assessment/assess', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          entity_id: assessmentForm.entity_id || crypto.randomUUID(),
          entity_type: assessmentForm.entity_type,
          input_data: assessmentForm
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setRiskAssessments(prev => [data.assessment, ...prev.slice(0, 9)]);
        showNotification('Risk assessment completed successfully', 'success');
        return data.assessment;
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Risk assessment failed: ${err.message}`, 'error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Execute stress test
   */
  const executeStressTest = async (assessmentId, scenarioIds) => {
    setLoading(true);
    try {
      const response = await fetch('/api/risk-assessment/stress-test', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          assessment_id: assessmentId,
          scenario_ids: scenarioIds
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setStressTestResults(prev => [data.stress_test_results, ...prev.slice(0, 9)]);
        showNotification('Stress test completed successfully', 'success');
        return data.stress_test_results;
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Stress test failed: ${err.message}`, 'error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Show notification
   */
  const showNotification = (message, type = 'info') => {
    console.log(`${type.toUpperCase()}: ${message}`);
  };

  /**
   * Get risk level color
   */
  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'very_low': return 'text-green-600';
      case 'low': return 'text-blue-600';
      case 'medium': return 'text-yellow-600';
      case 'high': return 'text-orange-600';
      case 'very_high': return 'text-red-600';
      case 'critical': return 'text-red-800';
      default: return 'text-gray-600';
    }
  };

  /**
   * Get risk level badge variant
   */
  const getRiskLevelBadgeVariant = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'very_low':
      case 'low': return 'default';
      case 'medium': return 'secondary';
      case 'high':
      case 'very_high': return 'destructive';
      case 'critical': return 'destructive';
      default: return 'outline';
    }
  };

  /**
   * Format currency
   */
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('nl-NL', {
      style: 'currency',
      currency: 'EUR'
    }).format(amount || 0);
  };

  /**
   * Format percentage
   */
  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  /**
   * Render Dashboard Tab
   */
  const renderDashboard = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Calculator className="h-4 w-4 text-blue-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Assessments (24h)</p>
                <p className="text-2xl font-bold">{realTimeData.system_metrics?.assessments_performed || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Active Alerts</p>
                <p className="text-2xl font-bold">{riskAlerts.filter(a => a.alert_status === 'active').length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-purple-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Stress Tests</p>
                <p className="text-2xl font-bold">{realTimeData.system_metrics?.stress_tests_executed || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-green-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Avg Processing</p>
                <p className="text-2xl font-bold">
                  {realTimeData.system_metrics?.avg_assessment_time 
                    ? `${Math.round(realTimeData.system_metrics.avg_assessment_time)}ms`
                    : '0ms'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk Level Distribution */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <PieChart className="h-5 w-5" />
              <span>Risk Level Distribution</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { level: 'low', count: 45, color: 'bg-green-500' },
                { level: 'medium', count: 28, color: 'bg-yellow-500' },
                { level: 'high', count: 12, color: 'bg-orange-500' },
                { level: 'critical', count: 3, color: 'bg-red-500' }
              ].map((item) => (
                <div key={item.level} className="flex items-center space-x-3">
                  <div className={`w-4 h-4 rounded ${item.color}`} />
                  <div className="flex-1">
                    <div className="flex justify-between text-sm">
                      <span className="capitalize">{item.level}</span>
                      <span>{item.count}</span>
                    </div>
                    <Progress value={(item.count / 88) * 100} className="h-2 mt-1" />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Gauge className="h-5 w-5" />
              <span>Portfolio Risk Metrics</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">Portfolio VaR (99%)</span>
                <span className="font-medium">{formatCurrency(2450000)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Expected Loss</span>
                <span className="font-medium">{formatCurrency(890000)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Concentration Risk</span>
                <span className="font-medium">12.5%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Capital Requirement</span>
                <span className="font-medium">{formatCurrency(1850000)}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Risk Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Bell className="h-5 w-5" />
            <span>Recent Risk Alerts</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {riskAlerts.slice(0, 5).map((alert) => (
              <div key={alert.alert_id} className="flex items-start space-x-3 p-3 border rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  alert.severity === 'critical' ? 'bg-red-500' :
                  alert.severity === 'high' ? 'bg-orange-500' :
                  alert.severity === 'medium' ? 'bg-yellow-500' :
                  'bg-blue-500'
                }`} />
                <div className="flex-1">
                  <p className="text-sm font-medium">{alert.title}</p>
                  <p className="text-xs text-gray-500">{alert.description}</p>
                  <div className="flex items-center space-x-4 mt-1">
                    <span className="text-xs text-gray-500">
                      Entity: {alert.entity_id}
                    </span>
                    <span className="text-xs text-gray-500">
                      Risk Score: {(alert.current_value * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(alert.created_at).toLocaleString()}
                    </span>
                  </div>
                </div>
                <Badge variant={getRiskLevelBadgeVariant(alert.severity)}>
                  {alert.severity}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Risk Assessment Tab
   */
  const renderRiskAssessment = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Risk Assessment</h3>
        <Button onClick={performRiskAssessment} disabled={loading}>
          {loading ? <RefreshCw className="h-4 w-4 animate-spin mr-2" /> : <Calculator className="h-4 w-4 mr-2" />}
          Perform Assessment
        </Button>
      </div>

      {/* Assessment Form */}
      <Card>
        <CardHeader>
          <CardTitle>New Risk Assessment</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="entity-id">Entity ID</Label>
              <Input
                id="entity-id"
                placeholder="customer_123"
                value={assessmentForm.entity_id}
                onChange={(e) => setAssessmentForm({...assessmentForm, entity_id: e.target.value})}
              />
            </div>
            <div>
              <Label htmlFor="entity-type">Entity Type</Label>
              <Select value={assessmentForm.entity_type} onValueChange={(value) => setAssessmentForm({...assessmentForm, entity_type: value})}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mortgage_application">Mortgage Application</SelectItem>
                  <SelectItem value="customer">Customer</SelectItem>
                  <SelectItem value="portfolio">Portfolio</SelectItem>
                  <SelectItem value="product">Product</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="credit-score">Credit Score</Label>
              <Input
                id="credit-score"
                type="number"
                min="300"
                max="850"
                value={assessmentForm.credit_score}
                onChange={(e) => setAssessmentForm({...assessmentForm, credit_score: parseInt(e.target.value)})}
              />
            </div>
            <div>
              <Label htmlFor="loan-amount">Loan Amount (€)</Label>
              <Input
                id="loan-amount"
                type="number"
                value={assessmentForm.loan_amount}
                onChange={(e) => setAssessmentForm({...assessmentForm, loan_amount: parseInt(e.target.value)})}
              />
            </div>
            <div>
              <Label htmlFor="property-value">Property Value (€)</Label>
              <Input
                id="property-value"
                type="number"
                value={assessmentForm.property_value}
                onChange={(e) => setAssessmentForm({...assessmentForm, property_value: parseInt(e.target.value)})}
              />
            </div>
            <div>
              <Label htmlFor="monthly-income">Monthly Income (€)</Label>
              <Input
                id="monthly-income"
                type="number"
                value={assessmentForm.net_monthly_income}
                onChange={(e) => setAssessmentForm({...assessmentForm, net_monthly_income: parseInt(e.target.value)})}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recent Assessments */}
      <div className="space-y-4">
        {riskAssessments.map((assessment) => (
          <Card key={assessment.assessment_id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <Shield className="h-5 w-5 text-blue-500" />
                  <div>
                    <h4 className="font-semibold">{assessment.entity_id}</h4>
                    <p className="text-sm text-gray-500">{assessment.entity_type}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Badge variant={getRiskLevelBadgeVariant(assessment.risk_level)}>
                    {assessment.risk_level}
                  </Badge>
                  <span className="text-lg font-bold">{(assessment.overall_risk_score * 100).toFixed(1)}%</span>
                </div>
              </div>
              
              <div className="grid md:grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-sm text-gray-500">Default Probability</p>
                  <p className="text-lg font-bold">{formatPercentage(assessment.predicted_default_probability)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Expected Loss</p>
                  <p className="text-lg font-bold">{formatCurrency(assessment.expected_loss)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Value at Risk</p>
                  <p className="text-lg font-bold">{formatCurrency(assessment.value_at_risk)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Data Quality</p>
                  <p className="text-lg font-bold">{(assessment.data_quality_score * 100).toFixed(0)}%</p>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Risk Score</span>
                  <span>{(assessment.overall_risk_score * 100).toFixed(1)}%</span>
                </div>
                <Progress value={assessment.overall_risk_score * 100} className="h-2" />
              </div>

              <div className="flex justify-between items-center mt-4">
                <span className="text-sm text-gray-500">
                  Assessed: {new Date(assessment.created_at).toLocaleString()}
                </span>
                <div className="flex space-x-2">
                  <Button variant="outline" size="sm">
                    <Target className="h-4 w-4 mr-2" />
                    Stress Test
                  </Button>
                  <Button variant="outline" size="sm">
                    <Eye className="h-4 w-4 mr-2" />
                    Details
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  /**
   * Render Stress Testing Tab
   */
  const renderStressTesting = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Stress Testing</h3>
        <Button onClick={() => {
          // Demo stress test
          if (riskAssessments.length > 0) {
            executeStressTest(riskAssessments[0].assessment_id, ['scenario_1', 'scenario_2']);
          }
        }}>
          <Target className="h-4 w-4 mr-2" />
          Run Stress Test
        </Button>
      </div>

      {/* Stress Test Scenarios */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-red-500" />
              <h4 className="font-semibold mb-2">Economic Downturn</h4>
              <p className="text-sm text-gray-500 mb-4">GDP decline of 5%, unemployment to 12%</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Severity:</span>
                  <Badge variant="destructive">Severe</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Probability:</span>
                  <span>5%</span>
                </div>
                <div className="flex justify-between">
                  <span>Duration:</span>
                  <span>18 months</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <TrendingUp className="h-8 w-8 mx-auto mb-2 text-orange-500" />
              <h4 className="font-semibold mb-2">Interest Rate Shock</h4>
              <p className="text-sm text-gray-500 mb-4">Interest rates increase by 300 basis points</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Severity:</span>
                  <Badge variant="secondary">Moderate</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Probability:</span>
                  <span>15%</span>
                </div>
                <div className="flex justify-between">
                  <span>Duration:</span>
                  <span>12 months</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="text-center">
              <TrendingDown className="h-8 w-8 mx-auto mb-2 text-purple-500" />
              <h4 className="font-semibold mb-2">Property Market Crash</h4>
              <p className="text-sm text-gray-500 mb-4">Property values decline by 25%</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Severity:</span>
                  <Badge variant="destructive">Severe</Badge>
                </div>
                <div className="flex justify-between">
                  <span>Probability:</span>
                  <span>8%</span>
                </div>
                <div className="flex justify-between">
                  <span>Duration:</span>
                  <span>24 months</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Stress Test Results */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Stress Test Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[
              { scenario: 'Economic Downturn', baseline: 0.15, stressed: 0.28, impact: 0.13 },
              { scenario: 'Interest Rate Shock', baseline: 0.15, stressed: 0.22, impact: 0.07 },
              { scenario: 'Property Market Crash', baseline: 0.15, stressed: 0.35, impact: 0.20 }
            ].map((result, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <p className="font-medium">{result.scenario}</p>
                  <p className="text-sm text-gray-500">
                    Baseline: {formatPercentage(result.baseline)} → Stressed: {formatPercentage(result.stressed)}
                  </p>
                </div>
                <div className="text-right">
                  <p className="font-bold text-red-600">+{formatPercentage(result.impact)}</p>
                  <p className="text-xs text-gray-500">Impact</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Mitigation Tab
   */
  const renderMitigation = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Risk Mitigation Strategies</h3>
        <Button variant="outline">
          <Lightbulb className="h-4 w-4 mr-2" />
          Generate Recommendations
        </Button>
      </div>

      {/* Mitigation Strategy Categories */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <Shield className="h-8 w-8 mx-auto mb-2 text-blue-500" />
            <p className="text-lg font-bold">8</p>
            <p className="text-sm text-gray-500">Credit Risk Strategies</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <BarChart3 className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <p className="text-lg font-bold">5</p>
            <p className="text-sm text-gray-500">Market Risk Strategies</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Settings className="h-8 w-8 mx-auto mb-2 text-orange-500" />
            <p className="text-lg font-bold">6</p>
            <p className="text-sm text-gray-500">Operational Strategies</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <CheckCircle className="h-8 w-8 mx-auto mb-2 text-purple-500" />
            <p className="text-lg font-bold">4</p>
            <p className="text-sm text-gray-500">Compliance Strategies</p>
          </CardContent>
        </Card>
      </div>

      {/* Recommended Strategies */}
      <Card>
        <CardHeader>
          <CardTitle>Recommended Mitigation Strategies</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[
              {
                name: 'Credit Enhancement Program',
                category: 'Credit Risk',
                cost: 5000,
                reduction: 0.15,
                effectiveness: 0.8,
                timeframe: '12 weeks'
              },
              {
                name: 'Interest Rate Hedging',
                category: 'Market Risk',
                cost: 10000,
                reduction: 0.30,
                effectiveness: 0.85,
                timeframe: '8 weeks'
              },
              {
                name: 'Process Automation',
                category: 'Operational Risk',
                cost: 25000,
                reduction: 0.35,
                effectiveness: 0.88,
                timeframe: '20 weeks'
              }
            ].map((strategy, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h4 className="font-semibold">{strategy.name}</h4>
                    <p className="text-sm text-gray-500">{strategy.category}</p>
                  </div>
                  <Badge variant="outline">
                    {(strategy.effectiveness * 100).toFixed(0)}% Effective
                  </Badge>
                </div>
                
                <div className="grid md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Implementation Cost:</span>
                    <span className="ml-2 font-medium">{formatCurrency(strategy.cost)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Risk Reduction:</span>
                    <span className="ml-2 font-medium">{formatPercentage(strategy.reduction)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Timeframe:</span>
                    <span className="ml-2 font-medium">{strategy.timeframe}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">ROI:</span>
                    <span className="ml-2 font-medium text-green-600">
                      {((strategy.reduction * 100000 - strategy.cost) / strategy.cost * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                
                <div className="mt-3 flex space-x-2">
                  <Button variant="outline" size="sm">
                    <Eye className="h-4 w-4 mr-2" />
                    Details
                  </Button>
                  <Button size="sm">
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Implement
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Analytics Tab
   */
  const renderAnalytics = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Risk Analytics</h3>
      
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <LineChart className="h-5 w-5" />
              <span>Risk Trends</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">7-Day Trend</span>
                <div className="flex items-center space-x-2">
                  <TrendingDown className="h-4 w-4 text-green-500" />
                  <span className="text-green-600 font-medium">-2.3%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">30-Day Trend</span>
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-red-500" />
                  <span className="text-red-600 font-medium">+1.8%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">90-Day Trend</span>
                <div className="flex items-center space-x-2">
                  <TrendingDown className="h-4 w-4 text-green-500" />
                  <span className="text-green-600 font-medium">-5.1%</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>Model Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {modelPerformance.slice(0, 3).map((model, index) => (
                <div key={index} className="flex justify-between items-center">
                  <div>
                    <p className="font-medium">{model.model_type}</p>
                    <p className="text-sm text-gray-500">R² Score: {(model.r2_score * 100).toFixed(1)}%</p>
                  </div>
                  <Badge variant={model.r2_score > 0.8 ? 'default' : 'secondary'}>
                    {model.r2_score > 0.8 ? 'Excellent' : 'Good'}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Risk Factor Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-blue-600">15</p>
              <p className="text-sm text-gray-500">Risk Factors Tracked</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-green-600">94.2%</p>
              <p className="text-sm text-gray-500">Avg Data Quality</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-orange-600">0.87</p>
              <p className="text-sm text-gray-500">Model Accuracy</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  if (loading && riskAssessments.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading Advanced Risk Assessment Engine...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center space-x-3">
          <Shield className="h-8 w-8 text-blue-600" />
          <span>Advanced Risk Assessment Engine</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Sophisticated multi-factor risk analysis with predictive modeling and comprehensive mitigation strategies
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="dashboard" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="assessment" className="flex items-center space-x-2">
            <Calculator className="h-4 w-4" />
            <span>Assessment</span>
          </TabsTrigger>
          <TabsTrigger value="stress-testing" className="flex items-center space-x-2">
            <Target className="h-4 w-4" />
            <span>Stress Testing</span>
          </TabsTrigger>
          <TabsTrigger value="mitigation" className="flex items-center space-x-2">
            <Lightbulb className="h-4 w-4" />
            <span>Mitigation</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Analytics</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard">
          {renderDashboard()}
        </TabsContent>

        <TabsContent value="assessment">
          {renderRiskAssessment()}
        </TabsContent>

        <TabsContent value="stress-testing">
          {renderStressTesting()}
        </TabsContent>

        <TabsContent value="mitigation">
          {renderMitigation()}
        </TabsContent>

        <TabsContent value="analytics">
          {renderAnalytics()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RiskAssessmentEngine;