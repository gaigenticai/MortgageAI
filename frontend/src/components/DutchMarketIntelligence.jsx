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
  TrendingUp,
  TrendingDown,
  Activity,
  BarChart3,
  LineChart,
  PieChart,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Database,
  Wifi,
  WifiOff,
  RefreshCw,
  Download,
  Settings,
  Eye,
  Target,
  Zap,
  Globe,
  Building,
  DollarSign,
  Percent,
  Calendar,
  Filter,
  Search,
  Bell,
  BellRing,
  Info,
  ArrowUp,
  ArrowDown,
  Minus,
  Home,
  Banknote,
  Calculator,
  Users
} from 'lucide-react';

/**
 * Dutch Market Intelligence Interface Component
 * Comprehensive UI for real-time market data feeds, trend analysis, and predictive insights
 */
const DutchMarketIntelligence = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [marketData, setMarketData] = useState([]);
  const [trendAnalyses, setTrendAnalyses] = useState([]);
  const [predictiveInsights, setPredictiveInsights] = useState([]);
  const [intelligenceReports, setIntelligenceReports] = useState([]);
  const [marketAlerts, setMarketAlerts] = useState([]);
  const [dataSources, setDataSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [realTimeConnected, setRealTimeConnected] = useState(false);

  // Dashboard metrics
  const [dashboardMetrics, setDashboardMetrics] = useState({
    data_points_collected: 0,
    trends_analyzed: 0,
    predictions_generated: 0,
    reports_created: 0,
    avg_processing_time: 0,
    market_coverage_score: 0.95
  });

  // Filters and settings
  const [selectedSegment, setSelectedSegment] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // WebSocket connection
  const wsRef = useRef(null);

  // Market segments and sources
  const marketSegments = [
    { value: 'all', label: 'All Segments' },
    { value: 'residential_mortgage', label: 'Residential Mortgage' },
    { value: 'property_prices', label: 'Property Prices' },
    { value: 'interest_rates', label: 'Interest Rates' },
    { value: 'economic_indicators', label: 'Economic Indicators' }
  ];

  const dataSources = [
    { value: 'cbs', label: 'CBS (Statistics Netherlands)' },
    { value: 'dnb', label: 'DNB (Dutch Central Bank)' },
    { value: 'kadaster', label: 'Kadaster (Land Registry)' },
    { value: 'nhg', label: 'NHG (Mortgage Guarantee)' }
  ];

  // Initialize component
  useEffect(() => {
    initializeComponent();
    setupWebSocket();
    
    // Auto-refresh interval
    let refreshInterval;
    if (autoRefresh) {
      refreshInterval = setInterval(() => {
        refreshDashboardData();
      }, 30000); // 30 seconds
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, [autoRefresh]);

  /**
   * Initialize component data
   */
  const initializeComponent = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadDashboardData(),
        loadMarketData(),
        loadTrendAnalyses(),
        loadPredictiveInsights(),
        loadIntelligenceReports(),
        loadMarketAlerts(),
        loadDataSources()
      ]);
    } catch (err) {
      setError('Failed to initialize Dutch Market Intelligence');
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
      wsRef.current = new WebSocket(`ws://${window.location.hostname}:8017`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected to Dutch Market Intelligence');
        setRealTimeConnected(true);
        wsRef.current.send(JSON.stringify({ type: 'subscribe_market_data' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_trend_analysis' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_market_alerts' }));
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
        setRealTimeConnected(false);
        setTimeout(setupWebSocket, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setRealTimeConnected(false);
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
      case 'market_data_updated':
        loadMarketData();
        updateDashboardMetrics(data.data);
        showNotification('Market data updated', 'info');
        break;
      case 'trend_analysis_completed':
        loadTrendAnalyses();
        showNotification(`Trend analysis completed for ${data.data.segment}`, 'success');
        break;
      case 'prediction_generated':
        loadPredictiveInsights();
        showNotification(`New prediction generated for ${data.data.segment}`, 'success');
        break;
      case 'market_alert':
        loadMarketAlerts();
        showNotification(`Market alert: ${data.data.title}`, 'warning');
        break;
      case 'real_time_market_data':
        updateRealTimeData(data);
        break;
      default:
        console.log('Unknown WebSocket message:', data);
    }
  };

  /**
   * Load dashboard data
   */
  const loadDashboardData = async () => {
    try {
      const response = await fetch('/api/dutch-market-intelligence/dashboard');
      const data = await response.json();
      
      if (data.success) {
        setDashboardMetrics(data.dashboard.system_metrics);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    }
  };

  /**
   * Load market data
   */
  const loadMarketData = async () => {
    try {
      const params = new URLSearchParams();
      if (selectedSegment !== 'all') {
        params.append('segment', selectedSegment);
      }
      params.append('limit', '100');

      const response = await fetch(`/api/dutch-market-intelligence/market-data?${params}`);
      const data = await response.json();
      
      if (data.success) {
        setMarketData(data.market_data);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load market data:', err);
    }
  };

  /**
   * Load trend analyses
   */
  const loadTrendAnalyses = async () => {
    try {
      const params = new URLSearchParams();
      if (selectedSegment !== 'all') {
        params.append('segment', selectedSegment);
      }
      params.append('days', '30');

      const response = await fetch(`/api/dutch-market-intelligence/trend-analyses?${params}`);
      const data = await response.json();
      
      if (data.success) {
        setTrendAnalyses(data.trend_analyses);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load trend analyses:', err);
    }
  };

  /**
   * Load predictive insights
   */
  const loadPredictiveInsights = async () => {
    try {
      const params = new URLSearchParams();
      if (selectedSegment !== 'all') {
        params.append('segment', selectedSegment);
      }
      params.append('min_confidence', '0.7');

      const response = await fetch(`/api/dutch-market-intelligence/predictive-insights?${params}`);
      const data = await response.json();
      
      if (data.success) {
        setPredictiveInsights(data.predictive_insights);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load predictive insights:', err);
    }
  };

  /**
   * Load intelligence reports
   */
  const loadIntelligenceReports = async () => {
    try {
      const response = await fetch('/api/dutch-market-intelligence/reports?limit=10');
      const data = await response.json();
      
      if (data.success) {
        setIntelligenceReports(data.reports);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load intelligence reports:', err);
    }
  };

  /**
   * Load market alerts
   */
  const loadMarketAlerts = async () => {
    try {
      const response = await fetch('/api/dutch-market-intelligence/alerts?status=active&limit=20');
      const data = await response.json();
      
      if (data.success) {
        setMarketAlerts(data.alerts);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load market alerts:', err);
    }
  };

  /**
   * Load data sources
   */
  const loadDataSources = async () => {
    try {
      const response = await fetch('/api/dutch-market-intelligence/data-sources');
      const data = await response.json();
      
      if (data.success) {
        setDataSources(data.data_sources);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load data sources:', err);
    }
  };

  /**
   * Generate trend analysis
   */
  const generateTrendAnalysis = async (segment) => {
    setLoading(true);
    try {
      const response = await fetch('/api/dutch-market-intelligence/analyze-trends', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          segment: segment,
          analysis_period: '3_months',
          analysis_method: 'comprehensive'
        })
      });

      const data = await response.json();
      
      if (data.success) {
        showNotification('Trend analysis completed successfully', 'success');
        loadTrendAnalyses();
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Failed to generate trend analysis: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Generate predictive insights
   */
  const generatePredictiveInsights = async (segment) => {
    setLoading(true);
    try {
      const response = await fetch('/api/dutch-market-intelligence/generate-predictions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          segment: segment,
          prediction_horizon: 'medium_term',
          model_type: 'ensemble'
        })
      });

      const data = await response.json();
      
      if (data.success) {
        showNotification('Predictive insights generated successfully', 'success');
        loadPredictiveInsights();
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Failed to generate predictions: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Generate intelligence report
   */
  const generateIntelligenceReport = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/dutch-market-intelligence/generate-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          segments: ['residential_mortgage', 'property_prices', 'interest_rates'],
          sources: ['cbs', 'dnb', 'kadaster'],
          reporting_period: 'monthly',
          include_predictions: true,
          include_risk_assessment: true
        })
      });

      const data = await response.json();
      
      if (data.success) {
        showNotification('Intelligence report generated successfully', 'success');
        loadIntelligenceReports();
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Failed to generate report: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Refresh dashboard data
   */
  const refreshDashboardData = async () => {
    try {
      await loadDashboardData();
      await loadMarketData();
    } catch (err) {
      console.error('Failed to refresh dashboard data:', err);
    }
  };

  /**
   * Update dashboard metrics
   */
  const updateDashboardMetrics = (data) => {
    setDashboardMetrics(prev => ({
      ...prev,
      data_points_collected: data.data_points_collected || prev.data_points_collected
    }));
  };

  /**
   * Update real-time data
   */
  const updateRealTimeData = (data) => {
    if (data.segment === selectedSegment || selectedSegment === 'all') {
      setMarketData(prev => [...data.data, ...prev.slice(0, 99)]);
    }
  };

  /**
   * Show notification
   */
  const showNotification = (message, type = 'info') => {
    console.log(`${type.toUpperCase()}: ${message}`);
  };

  /**
   * Get trend direction icon
   */
  const getTrendDirectionIcon = (direction) => {
    switch (direction?.toLowerCase()) {
      case 'strongly_upward':
      case 'upward':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'strongly_downward':
      case 'downward':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      case 'stable':
        return <Minus className="h-4 w-4 text-blue-600" />;
      default:
        return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  /**
   * Get trend direction color
   */
  const getTrendDirectionColor = (direction) => {
    switch (direction?.toLowerCase()) {
      case 'strongly_upward':
      case 'upward':
        return 'text-green-600';
      case 'strongly_downward':
      case 'downward':
        return 'text-red-600';
      case 'stable':
        return 'text-blue-600';
      default:
        return 'text-gray-600';
    }
  };

  /**
   * Get alert severity badge variant
   */
  const getAlertSeverityVariant = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  /**
   * Format timestamp
   */
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  /**
   * Format currency
   */
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('nl-NL', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
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
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {realTimeConnected ? (
            <>
              <Wifi className="h-5 w-5 text-green-500" />
              <span className="text-sm text-green-600">Real-time connected</span>
            </>
          ) : (
            <>
              <WifiOff className="h-5 w-5 text-red-500" />
              <span className="text-sm text-red-600">Disconnected</span>
            </>
          )}
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Label>Auto-refresh</Label>
            <Button
              variant={autoRefresh ? "default" : "outline"}
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              {autoRefresh ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
            </Button>
          </div>
          
          <Button onClick={refreshDashboardData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4 text-blue-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Data Points</p>
                <p className="text-2xl font-bold">{dashboardMetrics.data_points_collected.toLocaleString()}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Trends Analyzed</p>
                <p className="text-2xl font-bold">{dashboardMetrics.trends_analyzed}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-purple-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Predictions</p>
                <p className="text-2xl font-bold">{dashboardMetrics.predictions_generated}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4 text-orange-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Reports</p>
                <p className="text-2xl font-bold">{dashboardMetrics.reports_created}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Market Overview */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Activity className="h-5 w-5" />
              <span>Latest Market Trends</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {trendAnalyses.slice(0, 5).map((trend, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    {getTrendDirectionIcon(trend.trend_direction)}
                    <div>
                      <p className="font-medium">{trend.market_segment.replace('_', ' ').toUpperCase()}</p>
                      <p className="text-sm text-gray-500">
                        Strength: {(trend.trend_strength * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  <Badge variant="outline" className={getTrendDirectionColor(trend.trend_direction)}>
                    {trend.trend_direction.replace('_', ' ')}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5" />
              <span>Market Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {marketAlerts.slice(0, 5).map((alert, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 border rounded-lg">
                  <div className="mt-0.5">
                    {alert.severity === 'high' ? (
                      <AlertTriangle className="h-4 w-4 text-red-500" />
                    ) : alert.severity === 'medium' ? (
                      <Info className="h-4 w-4 text-yellow-500" />
                    ) : (
                      <Bell className="h-4 w-4 text-blue-500" />
                    )}
                  </div>
                  <div className="flex-1">
                    <p className="font-medium text-sm">{alert.title}</p>
                    <p className="text-xs text-gray-500">{alert.market_segment.replace('_', ' ')}</p>
                    <p className="text-xs text-gray-400">{formatTimestamp(alert.created_at)}</p>
                  </div>
                  <Badge variant={getAlertSeverityVariant(alert.severity)}>
                    {alert.severity}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-4">
            <Button 
              onClick={() => generateTrendAnalysis('residential_mortgage')} 
              disabled={loading}
              className="h-20"
            >
              <div className="text-center">
                <TrendingUp className="h-6 w-6 mx-auto mb-1" />
                <div className="text-sm">Analyze Trends</div>
              </div>
            </Button>
            
            <Button 
              onClick={() => generatePredictiveInsights('property_prices')} 
              disabled={loading}
              className="h-20"
            >
              <div className="text-center">
                <Target className="h-6 w-6 mx-auto mb-1" />
                <div className="text-sm">Generate Predictions</div>
              </div>
            </Button>
            
            <Button 
              onClick={generateIntelligenceReport} 
              disabled={loading}
              className="h-20"
            >
              <div className="text-center">
                <BarChart3 className="h-6 w-6 mx-auto mb-1" />
                <div className="text-sm">Create Report</div>
              </div>
            </Button>
            
            <Button 
              onClick={loadMarketAlerts} 
              disabled={loading}
              className="h-20"
            >
              <div className="text-center">
                <Bell className="h-6 w-6 mx-auto mb-1" />
                <div className="text-sm">Check Alerts</div>
              </div>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Market Data Tab
   */
  const renderMarketData = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Market Data</h3>
        <div className="flex items-center space-x-4">
          <Select value={selectedSegment} onValueChange={setSelectedSegment}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Select segment" />
            </SelectTrigger>
            <SelectContent>
              {marketSegments.map((segment) => (
                <SelectItem key={segment.value} value={segment.value}>
                  {segment.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Button onClick={loadMarketData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Market Data Table */}
      <Card>
        <CardHeader>
          <CardTitle>Latest Market Data Points</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {marketData.slice(0, 20).map((dataPoint, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="w-3 h-3 rounded-full bg-blue-500" />
                  <div>
                    <p className="font-medium">{dataPoint.market_segment.replace('_', ' ').toUpperCase()}</p>
                    <p className="text-sm text-gray-500">
                      Source: {dataPoint.data_source.toUpperCase()} • 
                      Region: {dataPoint.region}
                    </p>
                    <p className="text-xs text-gray-400">
                      {formatTimestamp(dataPoint.timestamp)}
                    </p>
                  </div>
                </div>
                
                <div className="text-right">
                  <p className="text-lg font-bold">
                    {dataPoint.unit === 'EUR' ? formatCurrency(dataPoint.value) :
                     dataPoint.unit === 'percentage' ? formatPercentage(dataPoint.value / 100) :
                     dataPoint.value.toLocaleString()}
                  </p>
                  <p className="text-sm text-gray-500">
                    Confidence: {formatPercentage(dataPoint.confidence_score)}
                  </p>
                  <Badge variant="outline">
                    {dataPoint.data_quality}
                  </Badge>
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
      <h3 className="text-lg font-semibold">Market Analytics</h3>
      
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <LineChart className="h-5 w-5" />
              <span>Trend Analysis Results</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {trendAnalyses.map((analysis, index) => (
                <div key={index} className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">{analysis.market_segment.replace('_', ' ').toUpperCase()}</h4>
                    {getTrendDirectionIcon(analysis.trend_direction)}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Trend Strength</p>
                      <div className="flex items-center space-x-2">
                        <Progress value={analysis.trend_strength * 100} className="flex-1 h-2" />
                        <span>{(analysis.trend_strength * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-gray-500">Volatility Index</p>
                      <div className="flex items-center space-x-2">
                        <Progress value={analysis.volatility_index * 100} className="flex-1 h-2" />
                        <span>{(analysis.volatility_index * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-xs text-gray-400 mt-2">
                    Analysis Period: {analysis.analysis_period} • 
                    {formatTimestamp(analysis.analysis_timestamp)}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="h-5 w-5" />
              <span>Predictive Insights</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {predictiveInsights.map((insight, index) => (
                <div key={index} className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">{insight.market_segment.replace('_', ' ').toUpperCase()}</h4>
                    <Badge variant="default">
                      {insight.prediction_horizon.replace('_', ' ')}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Predicted Value:</span>
                      <span className="font-medium">
                        {insight.market_segment === 'property_prices' ? formatCurrency(insight.predicted_value) :
                         insight.market_segment === 'interest_rates' ? formatPercentage(insight.predicted_value / 100) :
                         insight.predicted_value.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Confidence:</span>
                      <span className="font-medium">{formatPercentage(insight.confidence_score)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Model Used:</span>
                      <span className="text-sm">{insight.model_used}</span>
                    </div>
                  </div>
                  
                  <p className="text-xs text-gray-400 mt-2">
                    {formatTimestamp(insight.insight_timestamp)}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-blue-600">{dashboardMetrics.avg_processing_time.toFixed(0)}ms</p>
              <p className="text-sm text-gray-500">Avg Processing Time</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-green-600">{formatPercentage(dashboardMetrics.market_coverage_score)}</p>
              <p className="text-sm text-gray-500">Market Coverage</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-purple-600">{dataSources.filter(s => s.is_active).length}</p>
              <p className="text-sm text-gray-500">Active Data Sources</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-orange-600">{marketAlerts.filter(a => a.alert_status === 'active').length}</p>
              <p className="text-sm text-gray-500">Active Alerts</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Reports Tab
   */
  const renderReports = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Intelligence Reports</h3>
        <Button onClick={generateIntelligenceReport} disabled={loading}>
          <BarChart3 className="h-4 w-4 mr-2" />
          Generate New Report
        </Button>
      </div>

      <div className="space-y-4">
        {intelligenceReports.map((report, index) => (
          <Card key={index}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-semibold">Market Intelligence Report</h4>
                  <p className="text-sm text-gray-500">
                    Period: {report.reporting_period} • 
                    Confidence: {formatPercentage(report.report_confidence)}
                  </p>
                  <p className="text-xs text-gray-400">
                    Generated: {formatTimestamp(report.report_timestamp)} • 
                    Processing: {report.processing_time_ms}ms
                  </p>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Badge variant="outline">
                    {report.report_status}
                  </Badge>
                  <Button variant="outline" size="sm">
                    <Eye className="h-4 w-4 mr-2" />
                    View
                  </Button>
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                </div>
              </div>
              
              <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">Data Sources</p>
                  <p className="font-medium">{report.data_sources_used?.length || 0} sources</p>
                </div>
                <div>
                  <p className="text-gray-500">Market Overview</p>
                  <p className="font-medium">
                    {Object.keys(report.market_overview || {}).length} segments
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Recommendations</p>
                  <p className="font-medium">{report.recommendations?.length || 0} items</p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  if (loading && marketData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading Dutch Market Intelligence...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center space-x-3">
          <Globe className="h-8 w-8 text-blue-600" />
          <span>Dutch Market Intelligence</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Real-time market data feeds, trend analysis, and predictive insights for the Dutch mortgage market
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="dashboard" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="market-data" className="flex items-center space-x-2">
            <Database className="h-4 w-4" />
            <span>Market Data</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Analytics</span>
          </TabsTrigger>
          <TabsTrigger value="reports" className="flex items-center space-x-2">
            <LineChart className="h-4 w-4" />
            <span>Reports</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard">
          {renderDashboard()}
        </TabsContent>

        <TabsContent value="market-data">
          {renderMarketData()}
        </TabsContent>

        <TabsContent value="analytics">
          {renderAnalytics()}
        </TabsContent>

        <TabsContent value="reports">
          {renderReports()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DutchMarketIntelligence;