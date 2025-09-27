import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  Input
} from './ui/input';
import {
  Label
} from './ui/label';
import {
  CheckCircle,
  XCircle,
  AlertCircle,
  Upload,
  FileText,
  Shield,
  Search,
  Eye,
  Download,
  RefreshCw,
  AlertTriangle,
  Activity,
  BarChart3,
  Brain,
  Fingerprint,
  Lock,
  Zap,
  Target,
  Database,
  Clock,
  Users,
  TrendingUp,
  TrendingDown,
  Camera,
  Scan,
  Hash,
  Link,
  Award,
  Flag,
  Settings,
  Bell
} from 'lucide-react';

/**
 * Document Authenticity Checker Component
 * Advanced UI for document verification, fraud detection, and blockchain validation
 */
const DocumentAuthenticityChecker = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [verifications, setVerifications] = useState([]);
  const [fraudAnalysis, setFraudAnalysis] = useState([]);
  const [templates, setTemplates] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [analytics, setAnalytics] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [realTimeData, setRealTimeData] = useState({});

  // Upload states
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [verificationProgress, setVerificationProgress] = useState({});
  const [selectedVerification, setSelectedVerification] = useState(null);

  // WebSocket connection
  const wsRef = useRef(null);
  const fileInputRef = useRef(null);

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
        loadVerifications(),
        loadAlerts(),
        loadTemplates(),
        loadAnalytics()
      ]);
    } catch (err) {
      setError('Failed to initialize Document Authenticity Checker');
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
      wsRef.current = new WebSocket(`ws://${window.location.hostname}:8014`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected to Document Authenticity Checker');
        // Subscribe to updates
        wsRef.current.send(JSON.stringify({ type: 'subscribe_verifications' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_fraud_alerts' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_blockchain_events' }));
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
      case 'real_time_stats':
        setRealTimeData(data.data);
        break;
      case 'document_verified':
        loadVerifications();
        showNotification(`Document verified: ${data.data.authenticity_status}`, 
          data.data.authenticity_status === 'authentic' ? 'success' : 'warning');
        break;
      case 'fraud_alert_generated':
        loadAlerts();
        showNotification('Fraud alert generated', 'error');
        break;
      case 'batch_verification_completed':
        loadVerifications();
        showNotification(`Batch verification completed: ${data.data.successful_verifications} documents`, 'success');
        break;
      case 'blockchain_registration':
        showNotification('Document registered on blockchain', 'success');
        break;
      default:
        console.log('Unknown WebSocket message:', data);
    }
  };

  /**
   * Load verifications
   */
  const loadVerifications = async () => {
    try {
      const response = await fetch('/api/document-authenticity/verifications?limit=20');
      const data = await response.json();
      
      if (data.success) {
        setVerifications(data.verifications);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load verifications:', err);
      throw err;
    }
  };

  /**
   * Load alerts
   */
  const loadAlerts = async () => {
    try {
      const response = await fetch('/api/document-authenticity/alerts');
      const data = await response.json();
      
      if (data.success) {
        setAlerts(data.alerts);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load alerts:', err);
      throw err;
    }
  };

  /**
   * Load templates
   */
  const loadTemplates = async () => {
    try {
      const response = await fetch('/api/document-authenticity/templates');
      const data = await response.json();
      
      if (data.success) {
        setTemplates(data.templates);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load templates:', err);
      throw err;
    }
  };

  /**
   * Load analytics
   */
  const loadAnalytics = async () => {
    try {
      const response = await fetch('/api/document-authenticity/analytics/dashboard');
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
   * Handle file upload
   */
  const handleFileUpload = async (files) => {
    const formData = new FormData();
    
    if (files.length === 1) {
      // Single file verification
      formData.append('file', files[0]);
      
      try {
        setLoading(true);
        const response = await fetch('/api/document-authenticity/verify', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();
        
        if (data.success) {
          setVerifications(prev => [data.verification, ...prev.slice(0, 19)]);
          showNotification('Document verification completed', 'success');
        } else {
          throw new Error(data.error);
        }
      } catch (err) {
        showNotification(`Verification failed: ${err.message}`, 'error');
      } finally {
        setLoading(false);
      }
    } else {
      // Batch verification
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }
      
      try {
        setLoading(true);
        const response = await fetch('/api/document-authenticity/verify-batch', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();
        
        if (data.success) {
          showNotification(`Batch verification completed: ${data.batch_results.successful_verifications} documents`, 'success');
          loadVerifications();
        } else {
          throw new Error(data.error);
        }
      } catch (err) {
        showNotification(`Batch verification failed: ${err.message}`, 'error');
      } finally {
        setLoading(false);
      }
    }
  };

  /**
   * Handle drag and drop
   */
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const files = Array.from(e.dataTransfer.files);
      handleFileUpload(files);
    }
  }, []);

  /**
   * Show notification
   */
  const showNotification = (message, type = 'info') => {
    console.log(`${type.toUpperCase()}: ${message}`);
  };

  /**
   * Get authenticity status color
   */
  const getAuthenticityStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'authentic': return 'text-green-600';
      case 'suspicious': return 'text-yellow-600';
      case 'fraudulent': return 'text-red-600';
      case 'inconclusive': return 'text-gray-600';
      case 'error': return 'text-purple-600';
      default: return 'text-gray-600';
    }
  };

  /**
   * Get authenticity badge variant
   */
  const getAuthenticityBadgeVariant = (status) => {
    switch (status?.toLowerCase()) {
      case 'authentic': return 'default';
      case 'suspicious': return 'secondary';
      case 'fraudulent': return 'destructive';
      case 'inconclusive': return 'outline';
      case 'error': return 'outline';
      default: return 'outline';
    }
  };

  /**
   * Format file size
   */
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  /**
   * Format timestamp
   */
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
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
              <FileText className="h-4 w-4 text-blue-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Verified Today</p>
                <p className="text-2xl font-bold">{realTimeData.daily_stats?.total_verifications || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Authentic</p>
                <p className="text-2xl font-bold">{realTimeData.daily_stats?.authentic_count || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Fraud Detected</p>
                <p className="text-2xl font-bold">{realTimeData.daily_stats?.fraudulent_count || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-purple-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Avg Processing</p>
                <p className="text-2xl font-bold">
                  {realTimeData.daily_stats?.avg_processing_time 
                    ? `${Math.round(realTimeData.daily_stats.avg_processing_time)}ms`
                    : '0ms'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Document Upload Area */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Upload className="h-5 w-5" />
            <span>Document Verification</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
            <p className="text-lg font-medium mb-2">
              Drop documents here or click to upload
            </p>
            <p className="text-sm text-gray-500 mb-4">
              Supports PDF, JPEG, PNG, TIFF files up to 50MB
            </p>
            <div className="flex justify-center space-x-4">
              <Button 
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
              >
                <FileText className="h-4 w-4 mr-2" />
                Select Files
              </Button>
              <Button variant="outline" disabled={loading}>
                <Camera className="h-4 w-4 mr-2" />
                Take Photo
              </Button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.jpg,.jpeg,.png,.tiff"
              onChange={(e) => {
                if (e.target.files) {
                  handleFileUpload(Array.from(e.target.files));
                }
              }}
              className="hidden"
            />
          </div>
        </CardContent>
      </Card>

      {/* Verification Status Overview */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="h-5 w-5" />
              <span>Verification Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Authenticity Rate</span>
                <div className="flex items-center space-x-2">
                  <span className="font-medium">94.2%</span>
                  <TrendingUp className="h-4 w-4 text-green-500" />
                </div>
              </div>
              <Progress value={94.2} className="h-2" />
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Authentic:</span>
                  <span className="ml-2 font-medium text-green-600">847</span>
                </div>
                <div>
                  <span className="text-gray-500">Suspicious:</span>
                  <span className="ml-2 font-medium text-yellow-600">32</span>
                </div>
                <div>
                  <span className="text-gray-500">Fraudulent:</span>
                  <span className="ml-2 font-medium text-red-600">18</span>
                </div>
                <div>
                  <span className="text-gray-500">Inconclusive:</span>
                  <span className="ml-2 font-medium text-gray-600">3</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>AI Detection Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">Fraud Detection Rate</span>
                <span className="font-medium">98.5%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">False Positive Rate</span>
                <span className="font-medium">1.2%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Model Accuracy</span>
                <span className="font-medium">97.8%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Confidence Level</span>
                <span className="font-medium">96.1%</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Bell className="h-5 w-5" />
            <span>Recent Fraud Alerts</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {alerts.slice(0, 5).map((alert) => (
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
                      Fraud Probability: {(alert.fraud_probability * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-gray-500">
                      {formatTimestamp(alert.created_at)}
                    </span>
                  </div>
                </div>
                <Badge variant={getAuthenticityBadgeVariant(alert.severity)}>
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
   * Render Verification Tab
   */
  const renderVerification = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Document Verifications</h3>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={loadVerifications}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={() => fileInputRef.current?.click()}>
            <Upload className="h-4 w-4 mr-2" />
            Upload Document
          </Button>
        </div>
      </div>

      {/* Verification Results */}
      <div className="space-y-4">
        {verifications.map((verification) => (
          <Card key={verification.verification_id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <FileText className="h-5 w-5 text-blue-500" />
                  <div>
                    <h4 className="font-semibold">{verification.file_name || 'Unknown Document'}</h4>
                    <p className="text-sm text-gray-500">
                      {verification.document_type} • {formatFileSize(verification.file_size)}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Badge variant={getAuthenticityBadgeVariant(verification.authenticity_status)}>
                    {verification.authenticity_status}
                  </Badge>
                  <span className="text-lg font-bold">
                    {(verification.confidence_score * 100).toFixed(1)}%
                  </span>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedVerification(verification)}>
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              <div className="grid md:grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-sm text-gray-500">Processing Time</p>
                  <p className="text-lg font-bold">{verification.processing_time_ms}ms</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Fraud Probability</p>
                  <p className="text-lg font-bold text-red-600">
                    {verification.overall_fraud_probability 
                      ? `${(verification.overall_fraud_probability * 100).toFixed(1)}%`
                      : 'N/A'
                    }
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Verification Methods</p>
                  <p className="text-sm font-medium">{verification.verification_methods?.length || 0} methods</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Verified</p>
                  <p className="text-sm">{formatTimestamp(verification.verification_timestamp)}</p>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Confidence Score</span>
                  <span>{(verification.confidence_score * 100).toFixed(1)}%</span>
                </div>
                <Progress value={verification.confidence_score * 100} className="h-2" />
              </div>

              {verification.fraud_indicators && verification.fraud_indicators.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm font-medium text-red-600 mb-2">Fraud Indicators:</p>
                  <div className="flex flex-wrap gap-1">
                    {verification.fraud_indicators.map((indicator, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs text-red-600">
                        {indicator}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  /**
   * Render Forensics Tab
   */
  const renderForensics = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Forensics Analysis</h3>
        <Button variant="outline">
          <Scan className="h-4 w-4 mr-2" />
          Advanced Analysis
        </Button>
      </div>

      {/* Forensics Methods */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <Camera className="h-8 w-8 mx-auto mb-2 text-blue-500" />
            <p className="text-lg font-bold">ELA Analysis</p>
            <p className="text-sm text-gray-500">Error Level Analysis</p>
            <Button variant="outline" size="sm" className="mt-2">
              <Eye className="h-4 w-4 mr-2" />
              View Results
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Target className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <p className="text-lg font-bold">Copy-Move</p>
            <p className="text-sm text-gray-500">Forgery Detection</p>
            <Button variant="outline" size="sm" className="mt-2">
              <Eye className="h-4 w-4 mr-2" />
              View Results
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Activity className="h-8 w-8 mx-auto mb-2 text-orange-500" />
            <p className="text-lg font-bold">Noise Analysis</p>
            <p className="text-sm text-gray-500">Pattern Detection</p>
            <Button variant="outline" size="sm" className="mt-2">
              <Eye className="h-4 w-4 mr-2" />
              View Results
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Fingerprint className="h-8 w-8 mx-auto mb-2 text-purple-500" />
            <p className="text-lg font-bold">Compression</p>
            <p className="text-sm text-gray-500">Artifact Analysis</p>
            <Button variant="outline" size="sm" className="mt-2">
              <Eye className="h-4 w-4 mr-2" />
              View Results
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Forensics Results */}
      <Card>
        <CardHeader>
          <CardTitle>Latest Forensics Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[
              { method: 'ELA Analysis', score: 0.92, status: 'authentic', indicators: 0 },
              { method: 'Copy-Move Detection', score: 0.88, status: 'authentic', indicators: 0 },
              { method: 'Noise Analysis', score: 0.75, status: 'suspicious', indicators: 2 },
              { method: 'Compression Analysis', score: 0.95, status: 'authentic', indicators: 0 }
            ].map((result, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    result.status === 'authentic' ? 'bg-green-500' :
                    result.status === 'suspicious' ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`} />
                  <div>
                    <p className="font-medium">{result.method}</p>
                    <p className="text-sm text-gray-500">
                      {result.indicators} indicators detected
                    </p>
                  </div>
                </div>
                
                <div className="text-right">
                  <p className="font-bold">{(result.score * 100).toFixed(1)}%</p>
                  <Badge variant={getAuthenticityBadgeVariant(result.status)} className="text-xs">
                    {result.status}
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
   * Render Blockchain Tab
   */
  const renderBlockchain = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Blockchain Verification</h3>
        <Button variant="outline">
          <Link className="h-4 w-4 mr-2" />
          Register Document
        </Button>
      </div>

      {/* Blockchain Status */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <Lock className="h-8 w-8 mx-auto mb-2 text-blue-500" />
            <p className="text-2xl font-bold text-blue-600">156</p>
            <p className="text-sm text-gray-500">Blockchain Registered</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Hash className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <p className="text-2xl font-bold text-green-600">100%</p>
            <p className="text-sm text-gray-500">Verification Success</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Database className="h-8 w-8 mx-auto mb-2 text-purple-500" />
            <p className="text-2xl font-bold text-purple-600">1.2 GB</p>
            <p className="text-sm text-gray-500">IPFS Storage</p>
          </CardContent>
        </Card>
      </div>

      {/* Blockchain Features */}
      <Card>
        <CardHeader>
          <CardTitle>Blockchain Verification Features</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">Immutable Registration</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>SHA-256 document hashing</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Smart contract verification</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>IPFS metadata storage</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Multi-signature validation</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-3">Verification Process</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-blue-500" />
                  <span>Real-time blockchain queries</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-blue-500" />
                  <span>Confirmation tracking</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-blue-500" />
                  <span>Revocation status checking</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-blue-500" />
                  <span>Audit trail integration</span>
                </li>
              </ul>
            </div>
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
      <h3 className="text-lg font-semibold">Verification Analytics</h3>
      
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Verification Trends</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">This Week</span>
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <span className="text-green-600 font-medium">+12.5%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">This Month</span>
                <div className="flex items-center space-x-2">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                  <span className="text-green-600 font-medium">+8.2%</span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Fraud Detection</span>
                <div className="flex items-center space-x-2">
                  <TrendingDown className="h-4 w-4 text-red-500" />
                  <span className="text-red-600 font-medium">-2.1%</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>AI Model Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">Accuracy</span>
                <span className="font-medium">97.8%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Precision</span>
                <span className="font-medium">96.5%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Recall</span>
                <span className="font-medium">98.2%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">F1 Score</span>
                <span className="font-medium">97.3%</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Document Type Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-blue-600">342</p>
              <p className="text-sm text-gray-500">Identity Documents</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-green-600">289</p>
              <p className="text-sm text-gray-500">Financial Statements</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-orange-600">156</p>
              <p className="text-sm text-gray-500">Employment Documents</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-purple-600">98</p>
              <p className="text-sm text-gray-500">Property Documents</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  if (loading && verifications.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading Document Authenticity Checker...</p>
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
          <span>Document Authenticity Checker</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Advanced document verification with computer vision, blockchain validation, and fraud detection
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="dashboard" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="verification" className="flex items-center space-x-2">
            <FileText className="h-4 w-4" />
            <span>Verification</span>
          </TabsTrigger>
          <TabsTrigger value="forensics" className="flex items-center space-x-2">
            <Scan className="h-4 w-4" />
            <span>Forensics</span>
          </TabsTrigger>
          <TabsTrigger value="blockchain" className="flex items-center space-x-2">
            <Link className="h-4 w-4" />
            <span>Blockchain</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Analytics</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard">
          {renderDashboard()}
        </TabsContent>

        <TabsContent value="verification">
          {renderVerification()}
        </TabsContent>

        <TabsContent value="forensics">
          {renderForensics()}
        </TabsContent>

        <TabsContent value="blockchain">
          {renderBlockchain()}
        </TabsContent>

        <TabsContent value="analytics">
          {renderAnalytics()}
        </TabsContent>
      </Tabs>

      {/* Verification Details Modal */}
      {selectedVerification && (
        <Dialog open={!!selectedVerification} onOpenChange={() => setSelectedVerification(null)}>
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle>Document Verification Details</DialogTitle>
              <DialogDescription>
                Verification ID: {selectedVerification.verification_id}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <Label>Document Type</Label>
                  <p className="font-medium">{selectedVerification.document_type}</p>
                </div>
                <div>
                  <Label>Authenticity Status</Label>
                  <Badge variant={getAuthenticityBadgeVariant(selectedVerification.authenticity_status)}>
                    {selectedVerification.authenticity_status}
                  </Badge>
                </div>
                <div>
                  <Label>Confidence Score</Label>
                  <p className="font-medium">{(selectedVerification.confidence_score * 100).toFixed(1)}%</p>
                </div>
                <div>
                  <Label>Processing Time</Label>
                  <p className="font-medium">{selectedVerification.processing_time_ms}ms</p>
                </div>
              </div>
              
              {selectedVerification.fraud_indicators && selectedVerification.fraud_indicators.length > 0 && (
                <div>
                  <Label>Fraud Indicators</Label>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {selectedVerification.fraud_indicators.map((indicator, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {indicator}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              {selectedVerification.analyst_notes && (
                <div>
                  <Label>Analyst Notes</Label>
                  <p className="text-sm bg-gray-100 p-3 rounded-lg">
                    {selectedVerification.analyst_notes}
                  </p>
                </div>
              )}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setSelectedVerification(null)}>
                Close
              </Button>
              <Button>
                <Download className="h-4 w-4 mr-2" />
                Download Report
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default DocumentAuthenticityChecker;