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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from './ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from './ui/select';
import {
  Input
} from './ui/input';
import {
  Label
} from './ui/label';
import {
  Textarea
} from './ui/textarea';
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  Clock,
  TrendingUp,
  TrendingDown,
  Zap,
  Shield,
  Target,
  BarChart3,
  Settings,
  Send,
  RefreshCw,
  Eye,
  Brain,
  Activity,
  Bell,
  Users,
  FileText,
  DollarSign,
  Percent,
  Calendar,
  Gauge
} from 'lucide-react';

/**
 * Advanced Lender Integration Manager Component
 * Comprehensive UI for managing lender integrations with AI-powered features
 */
const LenderIntegrationAdvanced = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [lenders, setLenders] = useState([]);
  const [submissions, setSubmissions] = useState([]);
  const [healthMetrics, setHealthMetrics] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [validationRules, setValidationRules] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedLender, setSelectedLender] = useState(null);
  const [applicationData, setApplicationData] = useState({});
  const [realTimeMetrics, setRealTimeMetrics] = useState({});

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
        loadLenders(),
        loadHealthMetrics(),
        loadRecentSubmissions(),
        loadNotifications()
      ]);
    } catch (err) {
      setError('Failed to initialize Advanced Lender Integration Manager');
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
      wsRef.current = new WebSocket(`ws://${window.location.hostname}:8010`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected to Advanced Lender Integration');
        // Subscribe to real-time updates
        wsRef.current.send(JSON.stringify({ type: 'subscribe_health' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_submissions' }));
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
      case 'real_time_metrics':
        setRealTimeMetrics(data.data);
        break;
      case 'application_submitted':
        loadRecentSubmissions();
        showNotification('Application submitted successfully', 'success');
        break;
      case 'lender_health_alert':
        loadHealthMetrics();
        showNotification(`Health alert for ${data.data.lender}`, 'warning');
        break;
      default:
        console.log('Unknown WebSocket message:', data);
    }
  };

  /**
   * Load lender configurations
   */
  const loadLenders = async () => {
    try {
      const response = await fetch('/api/lender-integration-advanced/lenders');
      const data = await response.json();
      
      if (data.success) {
        setLenders(data.lenders);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load lenders:', err);
      throw err;
    }
  };

  /**
   * Load health metrics
   */
  const loadHealthMetrics = async () => {
    try {
      const response = await fetch('/api/lender-integration-advanced/health/lenders');
      const data = await response.json();
      
      if (data.success) {
        setHealthMetrics(data.metrics);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load health metrics:', err);
      throw err;
    }
  };

  /**
   * Load recent submissions
   */
  const loadRecentSubmissions = async () => {
    try {
      const response = await fetch('/api/lender-integration-advanced/reports/submissions?days=7');
      const data = await response.json();
      
      if (data.success) {
        setSubmissions(data.statistics);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load submissions:', err);
      throw err;
    }
  };

  /**
   * Load notifications
   */
  const loadNotifications = async () => {
    try {
      const response = await fetch('/api/lender-integration-advanced/notifications');
      const data = await response.json();
      
      if (data.success) {
        setNotifications(data.notifications);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load notifications:', err);
      throw err;
    }
  };

  /**
   * Submit application to lender
   */
  const submitApplication = async (lenderName, appData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/lender-integration-advanced/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          lender_name: lenderName,
          application_data: appData
        })
      });

      const data = await response.json();
      
      if (data.success) {
        showNotification('Application submitted successfully', 'success');
        loadRecentSubmissions();
        return data;
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Submission failed: ${err.message}`, 'error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Get approval predictions
   */
  const getApprovalPredictions = async (appData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/lender-integration-advanced/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          application_data: appData
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setPredictions(data.predictions);
        return data.predictions;
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Prediction failed: ${err.message}`, 'error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Get optimized recommendations
   */
  const getOptimizedRecommendations = async (appData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/lender-integration-advanced/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          application_data: appData
        })
      });

      const data = await response.json();
      
      if (data.success) {
        return data.recommendations;
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Optimization failed: ${err.message}`, 'error');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  /**
   * Show notification
   */
  const showNotification = (message, type = 'info') => {
    // Add to notifications state
    const newNotification = {
      id: Date.now(),
      message,
      type,
      timestamp: new Date().toISOString()
    };
    setNotifications(prev => [newNotification, ...prev.slice(0, 9)]);
  };

  /**
   * Get status badge color
   */
  const getStatusBadgeVariant = (status) => {
    switch (status?.toLowerCase()) {
      case 'active': return 'default';
      case 'degraded': return 'secondary';
      case 'offline': return 'destructive';
      case 'approved': return 'default';
      case 'submitted': return 'secondary';
      case 'rejected': return 'destructive';
      case 'pending': return 'outline';
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
    return `${(value * 100).toFixed(1)}%`;
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
              <Users className="h-4 w-4 text-blue-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Active Lenders</p>
                <p className="text-2xl font-bold">{lenders.filter(l => l.is_active).length}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Send className="h-4 w-4 text-green-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Submissions (7d)</p>
                <p className="text-2xl font-bold">
                  {submissions.reduce((sum, s) => sum + s.count, 0)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-orange-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Avg Response Time</p>
                <p className="text-2xl font-bold">
                  {healthMetrics.length > 0 
                    ? `${Math.round(healthMetrics.reduce((sum, h) => sum + (h.response_time_ms || 0), 0) / healthMetrics.length)}ms`
                    : '0ms'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Success Rate</p>
                <p className="text-2xl font-bold">
                  {healthMetrics.length > 0 
                    ? formatPercentage(healthMetrics.reduce((sum, h) => sum + (h.success_rate || 0), 0) / healthMetrics.length)
                    : '0%'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Health Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Lender Health Status</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {healthMetrics.map((metric) => (
              <div key={metric.lender_name} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    metric.status === 'active' ? 'bg-green-500' :
                    metric.status === 'degraded' ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`} />
                  <div>
                    <p className="font-medium">{metric.lender_name}</p>
                    <p className="text-sm text-gray-500">
                      Last check: {new Date(metric.last_check).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-6 text-sm">
                  <div>
                    <span className="text-gray-500">Response:</span>
                    <span className="ml-1 font-medium">{Math.round(metric.response_time_ms || 0)}ms</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Success:</span>
                    <span className="ml-1 font-medium">{formatPercentage(metric.success_rate || 0)}</span>
                  </div>
                  <Badge variant={getStatusBadgeVariant(metric.status)}>
                    {metric.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Bell className="h-5 w-5" />
            <span>Recent Notifications</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {notifications.slice(0, 5).map((notification) => (
              <div key={notification.id} className="flex items-start space-x-3 p-3 border rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  notification.type === 'success' ? 'bg-green-500' :
                  notification.type === 'warning' ? 'bg-yellow-500' :
                  notification.type === 'error' ? 'bg-red-500' :
                  'bg-blue-500'
                }`} />
                <div className="flex-1">
                  <p className="text-sm">{notification.message}</p>
                  <p className="text-xs text-gray-500">
                    {new Date(notification.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Lenders Tab
   */
  const renderLenders = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Lender Configurations</h3>
        <Button onClick={() => setSelectedLender({})}>
          <Settings className="h-4 w-4 mr-2" />
          Add Lender
        </Button>
      </div>

      <div className="grid gap-4">
        {lenders.map((lender) => (
          <Card key={lender.id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div>
                    <h4 className="font-semibold">{lender.name}</h4>
                    <p className="text-sm text-gray-500">
                      {lender.supported_products?.length || 0} products • 
                      Processing: {lender.processing_time_days} days
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="text-right text-sm">
                    <p>Max LTV: {lender.max_ltv}%</p>
                    <p>Min Income: {formatCurrency(lender.min_income)}</p>
                  </div>
                  
                  <div className="text-right text-sm">
                    <p>Loan Range:</p>
                    <p>{formatCurrency(lender.min_loan_amount)} - {formatCurrency(lender.max_loan_amount)}</p>
                  </div>
                  
                  <Badge variant={lender.is_active ? 'default' : 'secondary'}>
                    {lender.is_active ? 'Active' : 'Inactive'}
                  </Badge>
                  
                  <Button variant="outline" size="sm" onClick={() => setSelectedLender(lender)}>
                    <Eye className="h-4 w-4" />
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
   * Render Predictions Tab
   */
  const renderPredictions = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">AI Approval Predictions</h3>
        <Button onClick={() => {
          // Demo application data
          const demoData = {
            loan_amount: 300000,
            property_value: 400000,
            gross_annual_income: 60000,
            employment_type: 'permanent',
            employment_years: 5
          };
          getApprovalPredictions(demoData);
        }}>
          <Brain className="h-4 w-4 mr-2" />
          Get Predictions
        </Button>
      </div>

      {predictions.length > 0 && (
        <div className="grid gap-4">
          {predictions.map((prediction) => (
            <Card key={prediction.lender_name}>
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-semibold">{prediction.lender_name}</h4>
                  <Badge variant={
                    prediction.probability > 0.8 ? 'default' :
                    prediction.probability > 0.6 ? 'secondary' :
                    'destructive'
                  }>
                    {formatPercentage(prediction.probability)} Approval Chance
                  </Badge>
                </div>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Confidence</span>
                      <span>{formatPercentage((prediction.confidence_interval[1] - prediction.confidence_interval[0]) / 2)}</span>
                    </div>
                    <Progress value={prediction.probability * 100} className="h-2" />
                  </div>
                  
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-green-600 mb-2">Positive Factors</p>
                      <ul className="text-sm space-y-1">
                        {prediction.positive_factors?.map((factor, idx) => (
                          <li key={idx} className="flex items-center space-x-2">
                            <CheckCircle className="h-3 w-3 text-green-500" />
                            <span>{factor}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <p className="text-sm font-medium text-red-600 mb-2">Risk Factors</p>
                      <ul className="text-sm space-y-1">
                        {prediction.risk_factors?.map((factor, idx) => (
                          <li key={idx} className="flex items-center space-x-2">
                            <AlertCircle className="h-3 w-3 text-red-500" />
                            <span>{factor}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm font-medium mb-1">Recommendation</p>
                    <p className="text-sm">{prediction.recommendation}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );

  /**
   * Render Submissions Tab
   */
  const renderSubmissions = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Application Submissions</h3>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={loadRecentSubmissions}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={() => {
            // Demo submission
            const demoData = {
              loan_amount: 250000,
              property_value: 350000,
              first_name: 'Jan',
              last_name: 'de Vries',
              email: 'jan@example.com'
            };
            if (lenders.length > 0) {
              submitApplication(lenders[0].name, demoData);
            }
          }}>
            <Send className="h-4 w-4 mr-2" />
            Test Submission
          </Button>
        </div>
      </div>

      <div className="grid gap-4">
        {submissions.map((submission, idx) => (
          <Card key={idx}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-semibold">{submission.lender_name}</h4>
                  <p className="text-sm text-gray-500">
                    {new Date(submission.date).toLocaleDateString()} • 
                    {submission.count} submissions • 
                    Avg response: {Math.round(submission.avg_response_time || 0)}ms
                  </p>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Badge variant={getStatusBadgeVariant(submission.status)}>
                    {submission.status}
                  </Badge>
                  <span className="text-lg font-bold">{submission.count}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  /**
   * Render Analytics Tab
   */
  const renderAnalytics = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Performance Analytics</h3>
      
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Success Rates</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {healthMetrics.map((metric) => (
                <div key={metric.lender_name} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>{metric.lender_name}</span>
                    <span>{formatPercentage(metric.success_rate || 0)}</span>
                  </div>
                  <Progress value={(metric.success_rate || 0) * 100} className="h-2" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Gauge className="h-5 w-5" />
              <span>Response Times</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {healthMetrics.map((metric) => (
                <div key={metric.lender_name} className="flex justify-between items-center">
                  <span className="text-sm">{metric.lender_name}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-medium">{Math.round(metric.response_time_ms || 0)}ms</span>
                    {metric.response_time_ms < 1000 ? (
                      <TrendingUp className="h-4 w-4 text-green-500" />
                    ) : (
                      <TrendingDown className="h-4 w-4 text-red-500" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Real-Time Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          {Object.keys(realTimeMetrics).length > 0 ? (
            <div className="grid md:grid-cols-3 gap-4">
              <div className="text-center p-4 border rounded-lg">
                <p className="text-2xl font-bold text-blue-600">
                  {realTimeMetrics.health_metrics?.length || 0}
                </p>
                <p className="text-sm text-gray-500">Active Connections</p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <p className="text-2xl font-bold text-green-600">
                  {realTimeMetrics.submission_stats?.reduce((sum, s) => sum + s.count, 0) || 0}
                </p>
                <p className="text-sm text-gray-500">Today's Submissions</p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <p className="text-2xl font-bold text-orange-600">
                  {realTimeMetrics.timestamp ? new Date(realTimeMetrics.timestamp).toLocaleTimeString() : '--'}
                </p>
                <p className="text-sm text-gray-500">Last Update</p>
              </div>
            </div>
          ) : (
            <p className="text-center text-gray-500 py-8">
              Connecting to real-time metrics...
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );

  if (loading && lenders.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading Advanced Lender Integration Manager...</p>
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
          <Zap className="h-8 w-8 text-blue-600" />
          <span>Advanced Lender Integration Manager</span>
        </h1>
        <p className="text-gray-600 mt-2">
          AI-powered lender integration with real-time monitoring, approval predictions, and advanced validation
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="dashboard" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="lenders" className="flex items-center space-x-2">
            <Users className="h-4 w-4" />
            <span>Lenders</span>
          </TabsTrigger>
          <TabsTrigger value="predictions" className="flex items-center space-x-2">
            <Brain className="h-4 w-4" />
            <span>AI Predictions</span>
          </TabsTrigger>
          <TabsTrigger value="submissions" className="flex items-center space-x-2">
            <Send className="h-4 w-4" />
            <span>Submissions</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Analytics</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard">
          {renderDashboard()}
        </TabsContent>

        <TabsContent value="lenders">
          {renderLenders()}
        </TabsContent>

        <TabsContent value="predictions">
          {renderPredictions()}
        </TabsContent>

        <TabsContent value="submissions">
          {renderSubmissions()}
        </TabsContent>

        <TabsContent value="analytics">
          {renderAnalytics()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default LenderIntegrationAdvanced;