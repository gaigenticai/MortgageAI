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
  Textarea
} from './ui/textarea';
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
  Brain,
  Search,
  Eye,
  RefreshCw,
  BarChart3,
  Activity,
  Users,
  MessageSquare,
  Hash,
  Target,
  TrendingUp,
  TrendingDown,
  Languages,
  FileText,
  Shield,
  AlertTriangle,
  Lightbulb,
  Network,
  Zap,
  Database,
  Clock,
  Gauge,
  BookOpen,
  Flag,
  Settings
} from 'lucide-react';

/**
 * NLP Content Analyzer Component
 * Advanced UI for natural language processing, entity extraction, and semantic analysis
 */
const NLPContentAnalyzer = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [analyses, setAnalyses] = useState([]);
  const [entities, setEntities] = useState([]);
  const [relationships, setRelationships] = useState([]);
  const [sentimentResults, setSentimentResults] = useState([]);
  const [riskAnalysis, setRiskAnalysis] = useState([]);
  const [topicModeling, setTopicModeling] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [analytics, setAnalytics] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [realTimeData, setRealTimeData] = useState({});

  // Analysis form state
  const [analysisText, setAnalysisText] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('auto');
  const [analysisOptions, setAnalysisOptions] = useState({
    extract_entities: true,
    analyze_sentiment: true,
    detect_risks: true,
    extract_relationships: true,
    perform_topic_modeling: true
  });

  // Selected analysis result
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);

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
        loadAnalyses(),
        loadAlerts(),
        loadAnalytics()
      ]);
    } catch (err) {
      setError('Failed to initialize NLP Content Analyzer');
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
      wsRef.current = new WebSocket(`ws://${window.location.hostname}:8015`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected to NLP Content Analyzer');
        // Subscribe to updates
        wsRef.current.send(JSON.stringify({ type: 'subscribe_nlp_analysis' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_entity_extraction' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_risk_alerts' }));
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
        setRealTimeData(data.data);
        break;
      case 'nlp_analysis_completed':
        loadAnalyses();
        showNotification(`NLP analysis completed for ${data.data.content_type}`, 'success');
        break;
      case 'nlp_risk_alert':
        loadAlerts();
        showNotification(`High risk content detected (${data.data.risk_indicators} indicators)`, 'warning');
        break;
      default:
        console.log('Unknown WebSocket message:', data);
    }
  };

  /**
   * Load analyses
   */
  const loadAnalyses = async () => {
    try {
      const response = await fetch('/api/nlp-content-analyzer/analysis?limit=20');
      const data = await response.json();
      
      if (data.success) {
        setAnalyses(data.analyses);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load analyses:', err);
      throw err;
    }
  };

  /**
   * Load alerts
   */
  const loadAlerts = async () => {
    try {
      const response = await fetch('/api/nlp-content-analyzer/alerts');
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
   * Load analytics
   */
  const loadAnalytics = async () => {
    try {
      const response = await fetch('/api/nlp-content-analyzer/analytics/dashboard');
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
   * Perform NLP analysis
   */
  const performNLPAnalysis = async () => {
    if (!analysisText.trim()) {
      showNotification('Please enter text to analyze', 'warning');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/nlp-content-analyzer/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: analysisText,
          document_id: `manual_${Date.now()}`,
          analysis_options: analysisOptions
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setAnalyses(prev => [data.analysis, ...prev.slice(0, 19)]);
        setSelectedAnalysis(data.analysis);
        showNotification('NLP analysis completed successfully', 'success');
        
        // Clear the text area
        setAnalysisText('');
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`NLP analysis failed: ${err.message}`, 'error');
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
   * Get sentiment color
   */
  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'very_positive':
      case 'positive': return 'text-green-600';
      case 'neutral': return 'text-gray-600';
      case 'negative':
      case 'very_negative': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  /**
   * Get entity type color
   */
  const getEntityTypeColor = (entityType) => {
    const colorMap = {
      'person': 'bg-blue-100 text-blue-800',
      'organization': 'bg-purple-100 text-purple-800',
      'location': 'bg-green-100 text-green-800',
      'money': 'bg-yellow-100 text-yellow-800',
      'date': 'bg-gray-100 text-gray-800',
      'bsn': 'bg-red-100 text-red-800',
      'iban': 'bg-indigo-100 text-indigo-800',
      'phone': 'bg-pink-100 text-pink-800',
      'email': 'bg-cyan-100 text-cyan-800'
    };
    return colorMap[entityType] || 'bg-gray-100 text-gray-800';
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
              <Brain className="h-4 w-4 text-blue-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Analyzed Today</p>
                <p className="text-2xl font-bold">{realTimeData.system_metrics?.documents_analyzed || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Hash className="h-4 w-4 text-green-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Entities Extracted</p>
                <p className="text-2xl font-bold">{realTimeData.system_metrics?.entities_extracted || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Risk Indicators</p>
                <p className="text-2xl font-bold">{realTimeData.system_metrics?.risk_indicators_detected || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4 text-purple-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Avg Processing</p>
                <p className="text-2xl font-bold">
                  {realTimeData.system_metrics?.avg_processing_time 
                    ? `${Math.round(realTimeData.system_metrics.avg_processing_time)}ms`
                    : '0ms'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Language Distribution */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Languages className="h-5 w-5" />
              <span>Language Distribution</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {analytics.language_stats?.map((lang) => (
                <div key={lang.language_detected} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="capitalize">{lang.language_detected}</span>
                    <span>{lang.document_count} docs ({(lang.avg_confidence * 100).toFixed(1)}% conf)</span>
                  </div>
                  <Progress value={(lang.document_count / 100) * 100} className="h-2" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="h-5 w-5" />
              <span>Content Types</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { type: 'financial_statement', count: 45, confidence: 0.94 },
                { type: 'identity_document', count: 32, confidence: 0.96 },
                { type: 'employment_document', count: 28, confidence: 0.89 },
                { type: 'property_document', count: 15, confidence: 0.91 }
              ].map((item) => (
                <div key={item.type} className="flex items-center justify-between">
                  <span className="text-sm capitalize">{item.type.replace('_', ' ')}</span>
                  <div className="text-sm">
                    <span className="font-medium">{item.count}</span>
                    <span className="text-gray-500 ml-2">({(item.confidence * 100).toFixed(1)}%)</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Recent NLP Analysis</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {analyses.slice(0, 5).map((analysis) => (
              <div key={analysis.analysis_id} className="flex items-start space-x-3 p-3 border rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  analysis.risk_score > 0.7 ? 'bg-red-500' :
                  analysis.risk_score > 0.4 ? 'bg-yellow-500' :
                  'bg-green-500'
                }`} />
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <p className="text-sm font-medium">{analysis.content_type}</p>
                    <Badge variant="outline" className="text-xs">
                      {analysis.language_detected}
                    </Badge>
                  </div>
                  <p className="text-xs text-gray-500">
                    {analysis.entity_count} entities • 
                    Risk: {(analysis.risk_score * 100).toFixed(0)}% • 
                    Confidence: {(analysis.confidence_score * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-gray-500">
                    {formatTimestamp(analysis.created_at)}
                  </p>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setSelectedAnalysis(analysis)}>
                  <Eye className="h-3 w-3" />
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Analysis Tab
   */
  const renderAnalysis = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Text Analysis</h3>
        <Button onClick={performNLPAnalysis} disabled={loading || !analysisText.trim()}>
          {loading ? <RefreshCw className="h-4 w-4 animate-spin mr-2" /> : <Brain className="h-4 w-4 mr-2" />}
          Analyze Text
        </Button>
      </div>

      {/* Analysis Input */}
      <Card>
        <CardHeader>
          <CardTitle>Text Input</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="analysis-text">Text Content</Label>
            <Textarea
              id="analysis-text"
              placeholder="Enter text content for NLP analysis..."
              value={analysisText}
              onChange={(e) => setAnalysisText(e.target.value)}
              rows={8}
              className="resize-none"
            />
            <p className="text-sm text-gray-500 mt-1">
              {analysisText.length}/100,000 characters
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="language-select">Language</Label>
              <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto-detect</SelectItem>
                  <SelectItem value="nl">Dutch</SelectItem>
                  <SelectItem value="en">English</SelectItem>
                  <SelectItem value="de">German</SelectItem>
                  <SelectItem value="fr">French</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Analysis Options</Label>
              <div className="flex flex-wrap gap-2 mt-2">
                {Object.entries(analysisOptions).map(([key, value]) => (
                  <Badge 
                    key={key}
                    variant={value ? 'default' : 'outline'}
                    className="cursor-pointer"
                    onClick={() => setAnalysisOptions({
                      ...analysisOptions,
                      [key]: !value
                    })}
                  >
                    {key.replace('_', ' ')}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Analysis Examples */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Examples</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <Button 
              variant="outline" 
              onClick={() => setAnalysisText(`Geachte heer De Vries,

Betreft: Hypotheekaanvraag voor woning aan de Damrak 123, 1012 LP Amsterdam

Wij hebben uw hypotheekaanvraag van €300.000 voor de aankoop van bovengenoemd pand ontvangen. Uw BSN 123456782 is geverifieerd en uw bankrekening NL91ABNA0417164300 is gecontroleerd.

Op basis van uw bruto jaarinkomen van €60.000 en uw huidige maandelijkse lasten van €800, kunnen wij u een hypotheek aanbieden tegen een rente van 3,2% voor een periode van 30 jaar.

Met vriendelijke groet,
ING Bank Nederland`)}
              className="text-left h-auto"
            >
              <div>
                <p className="font-medium">Dutch Mortgage Letter</p>
                <p className="text-sm text-gray-500">Financial document with entities</p>
              </div>
            </Button>
            
            <Button 
              variant="outline" 
              onClick={() => setAnalysisText(`Dear Mr. Johnson,

I am writing to express my concern about my current financial situation. Due to recent unemployment and mounting debts, I am struggling to meet my monthly obligations. 

My previous employer terminated my contract last month, and I have been unable to find new employment. The stress is overwhelming, and I fear I may lose my home if I cannot secure a mortgage modification.

I have been a loyal customer for 15 years and hope we can work together to find a solution.

Sincerely,
John Johnson`)}
              className="text-left h-auto"
            >
              <div>
                <p className="font-medium">Customer Correspondence</p>
                <p className="text-sm text-gray-500">High-risk content with sentiment</p>
              </div>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {selectedAnalysis && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-500">Language</p>
                <p className="font-medium capitalize">{selectedAnalysis.language_detection}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Content Type</p>
                <p className="font-medium capitalize">{selectedAnalysis.semantic_analysis?.content_type?.replace('_', ' ')}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Entities Found</p>
                <p className="font-medium">{selectedAnalysis.named_entities?.length || 0}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Risk Score</p>
                <p className={`font-medium ${
                  selectedAnalysis.risk_indicator_analysis?.risk_score > 0.7 ? 'text-red-600' :
                  selectedAnalysis.risk_indicator_analysis?.risk_score > 0.4 ? 'text-yellow-600' :
                  'text-green-600'
                }`}>
                  {(selectedAnalysis.risk_indicator_analysis?.risk_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Confidence Score</span>
                <span>{(selectedAnalysis.confidence_score * 100).toFixed(1)}%</span>
              </div>
              <Progress value={selectedAnalysis.confidence_score * 100} className="h-2" />
            </div>

            {selectedAnalysis.named_entities && selectedAnalysis.named_entities.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Extracted Entities:</p>
                <div className="flex flex-wrap gap-2">
                  {selectedAnalysis.named_entities.slice(0, 10).map((entity, idx) => (
                    <Badge key={idx} className={getEntityTypeColor(entity.entity_type)}>
                      {entity.text} ({entity.entity_type})
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {selectedAnalysis.risk_indicator_analysis?.risk_indicators && 
             selectedAnalysis.risk_indicator_analysis.risk_indicators.length > 0 && (
              <div>
                <p className="text-sm font-medium text-red-600 mb-2">Risk Indicators:</p>
                <div className="flex flex-wrap gap-2">
                  {selectedAnalysis.risk_indicator_analysis.risk_indicators.map((indicator, idx) => (
                    <Badge key={idx} variant="destructive" className="text-xs">
                      {indicator.replace('_', ' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );

  /**
   * Render Entities Tab
   */
  const renderEntities = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Named Entity Recognition</h3>
        <Button variant="outline">
          <Hash className="h-4 w-4 mr-2" />
          Extract Entities
        </Button>
      </div>

      {/* Entity Statistics */}
      <div className="grid md:grid-cols-5 gap-4">
        {analytics.entity_summary?.slice(0, 5).map((entity) => (
          <Card key={entity.entity_type}>
            <CardContent className="p-6 text-center">
              <p className="text-2xl font-bold">{entity.total_extractions}</p>
              <p className="text-sm text-gray-500 capitalize">{entity.entity_type.replace('_', ' ')}</p>
              <p className="text-xs text-gray-400">
                {(entity.avg_confidence * 100).toFixed(0)}% confidence
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Entity Types Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Supported Entity Types</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-semibold mb-3 text-blue-600">Personal Information</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  <span>Person Names</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-red-500 rounded-full" />
                  <span>BSN Numbers</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-pink-500 rounded-full" />
                  <span>Phone Numbers</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-cyan-500 rounded-full" />
                  <span>Email Addresses</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <span>Addresses</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3 text-green-600">Financial Information</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full" />
                  <span>Money Amounts</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full" />
                  <span>IBAN Numbers</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-orange-500 rounded-full" />
                  <span>Interest Rates</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full" />
                  <span>Bank Names</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-teal-500 rounded-full" />
                  <span>Account Numbers</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3 text-purple-600">Organizational</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full" />
                  <span>Organizations</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <span>Locations</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-gray-500 rounded-full" />
                  <span>Dates</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  <span>Legal Entities</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-red-500 rounded-full" />
                  <span>Regulations</span>
                </li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  /**
   * Render Sentiment Tab
   */
  const renderSentiment = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Sentiment Analysis</h3>
        <Button variant="outline">
          <MessageSquare className="h-4 w-4 mr-2" />
          Analyze Sentiment
        </Button>
      </div>

      {/* Sentiment Overview */}
      <div className="grid md:grid-cols-5 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-green-600">42</div>
            <p className="text-sm text-gray-500">Very Positive</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-blue-600">128</div>
            <p className="text-sm text-gray-500">Positive</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-gray-600">89</div>
            <p className="text-sm text-gray-500">Neutral</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-orange-600">23</div>
            <p className="text-sm text-gray-500">Negative</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <div className="text-2xl font-bold text-red-600">8</div>
            <p className="text-sm text-gray-500">Very Negative</p>
          </CardContent>
        </Card>
      </div>

      {/* Sentiment Trends */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5" />
            <span>Sentiment Trends</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm">Overall Sentiment Trend</span>
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-green-500" />
                <span className="text-green-600 font-medium">+5.2%</span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Customer Satisfaction</span>
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-green-500" />
                <span className="text-green-600 font-medium">+3.8%</span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm">Stress Indicators</span>
              <div className="flex items-center space-x-2">
                <TrendingDown className="h-4 w-4 text-red-500" />
                <span className="text-red-600 font-medium">-2.1%</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Emotional Indicators */}
      <Card>
        <CardHeader>
          <CardTitle>Emotional Indicators Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold mb-3 text-green-600">Positive Indicators</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Satisfied, confident, optimistic</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Stable employment, good income</span>
                </li>
                <li className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span>Positive financial outlook</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3 text-yellow-600">Neutral Indicators</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 text-yellow-500" />
                  <span>Standard business language</span>
                </li>
                <li className="flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 text-yellow-500" />
                  <span>Formal communication tone</span>
                </li>
                <li className="flex items-center space-x-2">
                  <AlertCircle className="h-4 w-4 text-yellow-500" />
                  <span>Factual information only</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-3 text-red-600">Risk Indicators</h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center space-x-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span>Financial distress, debt problems</span>
                </li>
                <li className="flex items-center space-x-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span>Employment instability</span>
                </li>
                <li className="flex items-center space-x-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span>Legal issues, stress indicators</span>
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
      <h3 className="text-lg font-semibold">NLP Analytics</h3>
      
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Processing Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">Average Processing Time</span>
                <span className="font-medium">2.8s</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Entity Extraction Accuracy</span>
                <span className="font-medium">96.5%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Language Detection Accuracy</span>
                <span className="font-medium">99.2%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Risk Detection Accuracy</span>
                <span className="font-medium">94.8%</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="h-5 w-5" />
              <span>Model Performance</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">NER Model F1-Score</span>
                <span className="font-medium">0.94</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Sentiment Model Accuracy</span>
                <span className="font-medium">0.91</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Topic Model Coherence</span>
                <span className="font-medium">0.87</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Risk Model Precision</span>
                <span className="font-medium">0.89</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Analysis Volume</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-blue-600">1,247</p>
              <p className="text-sm text-gray-500">Total Analyses</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-green-600">15,892</p>
              <p className="text-sm text-gray-500">Entities Extracted</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-orange-600">156</p>
              <p className="text-sm text-gray-500">Risk Indicators</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-purple-600">892</p>
              <p className="text-sm text-gray-500">Relationships Found</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  if (loading && analyses.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading NLP Content Analyzer...</p>
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
          <Brain className="h-8 w-8 text-blue-600" />
          <span>NLP Content Analyzer</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Advanced natural language processing with semantic analysis, entity extraction, and contextual understanding
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="dashboard" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="analysis" className="flex items-center space-x-2">
            <Brain className="h-4 w-4" />
            <span>Analysis</span>
          </TabsTrigger>
          <TabsTrigger value="entities" className="flex items-center space-x-2">
            <Hash className="h-4 w-4" />
            <span>Entities</span>
          </TabsTrigger>
          <TabsTrigger value="sentiment" className="flex items-center space-x-2">
            <MessageSquare className="h-4 w-4" />
            <span>Sentiment</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Analytics</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard">
          {renderDashboard()}
        </TabsContent>

        <TabsContent value="analysis">
          {renderAnalysis()}
        </TabsContent>

        <TabsContent value="entities">
          {renderEntities()}
        </TabsContent>

        <TabsContent value="sentiment">
          {renderSentiment()}
        </TabsContent>

        <TabsContent value="analytics">
          {renderAnalytics()}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default NLPContentAnalyzer;