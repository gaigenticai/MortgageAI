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
  Input
} from './ui/input';
import {
  Label
} from './ui/label';
import {
  Textarea
} from './ui/textarea';
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
  Clock,
  Shield,
  FileText,
  Search,
  Filter,
  Download,
  Eye,
  AlertTriangle,
  Activity,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Users,
  Bell,
  Lock,
  Database,
  Zap,
  Target,
  Settings,
  RefreshCw,
  Calendar,
  Hash,
  Fingerprint,
  Microscope,
  Scale,
  BookOpen,
  Flag,
  UserCheck,
  ClipboardList
} from 'lucide-react';

/**
 * Comprehensive Compliance Audit Trail Component
 * Advanced UI for compliance logging, investigation, and regulatory reporting
 */
const ComplianceAuditTrail = () => {
  // State management
  const [activeTab, setActiveTab] = useState('dashboard');
  const [auditEvents, setAuditEvents] = useState([]);
  const [violations, setViolations] = useState([]);
  const [investigations, setInvestigations] = useState([]);
  const [complianceReports, setComplianceReports] = useState([]);
  const [patterns, setPatterns] = useState([]);
  const [stakeholders, setStakeholders] = useState([]);
  const [integrityStatus, setIntegrityStatus] = useState(null);
  const [metrics, setMetrics] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [realTimeData, setRealTimeData] = useState({});

  // Search and filter states
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [filters, setFilters] = useState({});
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [selectedViolation, setSelectedViolation] = useState(null);
  const [selectedInvestigation, setSelectedInvestigation] = useState(null);

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
        loadAuditEvents(),
        loadViolations(),
        loadInvestigations(),
        loadMetrics(),
        loadStakeholders(),
        loadPatterns()
      ]);
    } catch (err) {
      setError('Failed to initialize Compliance Audit Trail');
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
      wsRef.current = new WebSocket(`ws://${window.location.hostname}:8012`);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected to Compliance Audit Trail');
        // Subscribe to updates
        wsRef.current.send(JSON.stringify({ type: 'subscribe_audit_events' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_violations' }));
        wsRef.current.send(JSON.stringify({ type: 'subscribe_investigations' }));
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
      case 'audit_event_logged':
        loadAuditEvents();
        showNotification('New audit event logged', 'info');
        break;
      case 'violation_detected':
        loadViolations();
        showNotification('Compliance violation detected', 'warning');
        break;
      case 'investigation_created':
        loadInvestigations();
        showNotification('New investigation created', 'info');
        break;
      case 'violation_updated':
        loadViolations();
        showNotification('Violation status updated', 'success');
        break;
      default:
        console.log('Unknown WebSocket message:', data);
    }
  };

  /**
   * Load audit events
   */
  const loadAuditEvents = async () => {
    try {
      const response = await fetch('/api/compliance-audit/events?limit=20');
      const data = await response.json();
      
      if (data.success) {
        setAuditEvents(data.events);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load audit events:', err);
      throw err;
    }
  };

  /**
   * Load violations
   */
  const loadViolations = async () => {
    try {
      const response = await fetch('/api/compliance-audit/violations?limit=20');
      const data = await response.json();
      
      if (data.success) {
        setViolations(data.violations);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load violations:', err);
      throw err;
    }
  };

  /**
   * Load investigations
   */
  const loadInvestigations = async () => {
    try {
      const response = await fetch('/api/compliance-audit/investigations');
      const data = await response.json();
      
      if (data.success) {
        setInvestigations(data.investigations);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load investigations:', err);
      throw err;
    }
  };

  /**
   * Load metrics
   */
  const loadMetrics = async () => {
    try {
      const response = await fetch('/api/compliance-audit/metrics');
      const data = await response.json();
      
      if (data.success) {
        setMetrics(data);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load metrics:', err);
      throw err;
    }
  };

  /**
   * Load stakeholders
   */
  const loadStakeholders = async () => {
    try {
      const response = await fetch('/api/compliance-audit/stakeholders');
      const data = await response.json();
      
      if (data.success) {
        setStakeholders(data.stakeholders);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load stakeholders:', err);
      throw err;
    }
  };

  /**
   * Load patterns
   */
  const loadPatterns = async () => {
    try {
      const response = await fetch('/api/compliance-audit/patterns');
      const data = await response.json();
      
      if (data.success) {
        setPatterns(data.patterns);
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      console.error('Failed to load patterns:', err);
      throw err;
    }
  };

  /**
   * Perform search
   */
  const performSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('/api/compliance-audit/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: searchQuery,
          filters
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setSearchResults(data.results);
        showNotification(`Found ${data.results.length} results`, 'success');
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Search failed: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Verify integrity
   */
  const verifyIntegrity = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/compliance-audit/verify-integrity', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          start_date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          end_date: new Date().toISOString()
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setIntegrityStatus(data.verification);
        showNotification('Integrity verification completed', 'success');
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Integrity verification failed: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Generate report
   */
  const generateReport = async (reportType, regulation, startDate, endDate) => {
    setLoading(true);
    try {
      const response = await fetch('/api/compliance-audit/reports', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          report_type: reportType,
          regulation,
          start_date: startDate,
          end_date: endDate
        })
      });

      const data = await response.json();
      
      if (data.success) {
        setComplianceReports(prev => [data.report, ...prev.slice(0, 9)]);
        showNotification('Compliance report generated successfully', 'success');
        return data.report;
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      showNotification(`Report generation failed: ${err.message}`, 'error');
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
   * Get status badge variant
   */
  const getStatusBadgeVariant = (status) => {
    switch (status?.toLowerCase()) {
      case 'compliant':
      case 'resolved':
      case 'completed': return 'default';
      case 'pending':
      case 'in_progress':
      case 'open': return 'secondary';
      case 'non_compliant':
      case 'failed':
      case 'critical': return 'destructive';
      case 'warning':
      case 'degraded': return 'outline';
      default: return 'outline';
    }
  };

  /**
   * Get severity color
   */
  const getSeverityColor = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'text-red-600';
      case 'high': return 'text-orange-600';
      case 'medium': return 'text-yellow-600';
      case 'low': return 'text-blue-600';
      case 'info': return 'text-gray-600';
      default: return 'text-gray-600';
    }
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
                <p className="text-sm font-medium leading-none">Events Today</p>
                <p className="text-2xl font-bold">{realTimeData.events?.total_events || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Violations</p>
                <p className="text-2xl font-bold">{realTimeData.violations?.total_violations || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Microscope className="h-4 w-4 text-purple-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Active Investigations</p>
                <p className="text-2xl font-bold">{realTimeData.investigations?.open_investigations || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-green-500" />
              <div className="space-y-1">
                <p className="text-sm font-medium leading-none">Avg Risk Score</p>
                <p className="text-2xl font-bold">
                  {realTimeData.events?.avg_risk_score 
                    ? (realTimeData.events.avg_risk_score * 100).toFixed(1) + '%'
                    : '0%'
                  }
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Compliance Overview */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Shield className="h-5 w-5" />
              <span>Compliance Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Overall Compliance Rate</span>
                <div className="flex items-center space-x-2">
                  <span className="font-medium">95.2%</span>
                  <TrendingUp className="h-4 w-4 text-green-500" />
                </div>
              </div>
              <Progress value={95.2} className="h-2" />
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Wft Compliance:</span>
                  <span className="ml-2 font-medium">98.1%</span>
                </div>
                <div>
                  <span className="text-gray-500">BGfo Compliance:</span>
                  <span className="ml-2 font-medium">96.5%</span>
                </div>
                <div>
                  <span className="text-gray-500">GDPR Compliance:</span>
                  <span className="ml-2 font-medium">94.8%</span>
                </div>
                <div>
                  <span className="text-gray-500">NHG Compliance:</span>
                  <span className="ml-2 font-medium">99.2%</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Lock className="h-5 w-5" />
              <span>Audit Trail Integrity</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Hash Chain Integrity</span>
                <Badge variant={integrityStatus?.integrity_status === 'verified' ? 'default' : 'destructive'}>
                  {integrityStatus?.integrity_status || 'Unknown'}
                </Badge>
              </div>
              
              {integrityStatus && (
                <>
                  <div className="text-sm space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Events Verified:</span>
                      <span className="font-medium">{integrityStatus.total_events}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Integrity Score:</span>
                      <span className="font-medium">{(integrityStatus.integrity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Last Verified:</span>
                      <span className="text-xs">{formatTimestamp(integrityStatus.verification_timestamp)}</span>
                    </div>
                  </div>
                  
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={verifyIntegrity}
                    disabled={loading}
                    className="w-full"
                  >
                    <Fingerprint className="h-4 w-4 mr-2" />
                    Verify Integrity
                  </Button>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Recent Compliance Activity</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {auditEvents.slice(0, 5).map((event) => (
              <div key={event.event_id} className="flex items-start space-x-3 p-3 border rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 ${getSeverityColor(event.severity).replace('text-', 'bg-')}`} />
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <p className="text-sm font-medium">{event.action}</p>
                    <Badge variant={getStatusBadgeVariant(event.compliance_status)} className="text-xs">
                      {event.compliance_status}
                    </Badge>
                  </div>
                  <p className="text-xs text-gray-500">
                    {event.entity_type} • {event.regulation || 'No regulation'} • 
                    Risk: {(event.risk_score * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-gray-500">
                    {formatTimestamp(event.timestamp)}
                  </p>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setSelectedEvent(event)}>
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
   * Render Audit Events Tab
   */
  const renderAuditEvents = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Audit Events</h3>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={loadAuditEvents}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          <Button onClick={() => {
            // Demo event logging
            const demoEvent = {
              event_type: 'data_access',
              entity_type: 'customer_data',
              entity_id: 'customer_123',
              action: 'view_sensitive_data',
              details: { fields_accessed: ['bsn', 'income'] },
              regulation: 'GDPR',
              severity: 'medium'
            };
            // Would call API to log event
            showNotification('Demo event logged', 'success');
          }}>
            <FileText className="h-4 w-4 mr-2" />
            Log Demo Event
          </Button>
        </div>
      </div>

      {/* Search and Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex space-x-4">
            <div className="flex-1">
              <Input
                placeholder="Search audit events..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && performSearch()}
              />
            </div>
            <Button onClick={performSearch} disabled={loading}>
              <Search className="h-4 w-4 mr-2" />
              Search
            </Button>
            <Button variant="outline">
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Events List */}
      <div className="space-y-4">
        {(searchResults.length > 0 ? searchResults : auditEvents).map((event) => (
          <Card key={event.event_id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${getSeverityColor(event.severity).replace('text-', 'bg-')}`} />
                  <div>
                    <h4 className="font-semibold">{event.action}</h4>
                    <p className="text-sm text-gray-500">
                      {event.entity_type} • {event.entity_id}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Badge variant={getStatusBadgeVariant(event.compliance_status)}>
                    {event.compliance_status}
                  </Badge>
                  <Badge variant="outline" className={getSeverityColor(event.severity)}>
                    {event.severity}
                  </Badge>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedEvent(event)}>
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              <div className="grid md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Event Type:</span>
                  <span className="ml-2 font-medium">{event.event_type}</span>
                </div>
                <div>
                  <span className="text-gray-500">Regulation:</span>
                  <span className="ml-2 font-medium">{event.regulation || 'None'}</span>
                </div>
                <div>
                  <span className="text-gray-500">Risk Score:</span>
                  <span className="ml-2 font-medium">{(event.risk_score * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-gray-500">Timestamp:</span>
                  <span className="ml-2 text-xs">{formatTimestamp(event.timestamp)}</span>
                </div>
              </div>

              {event.tags && event.tags.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1">
                  {event.tags.map((tag, idx) => (
                    <Badge key={idx} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  /**
   * Render Violations Tab
   */
  const renderViolations = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Compliance Violations</h3>
        <Button variant="outline">
          <AlertTriangle className="h-4 w-4 mr-2" />
          Create Manual Violation
        </Button>
      </div>

      {/* Violation Statistics */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <AlertCircle className="h-8 w-8 mx-auto mb-2 text-red-500" />
            <p className="text-2xl font-bold text-red-600">
              {violations.filter(v => v.severity === 'critical').length}
            </p>
            <p className="text-sm text-gray-500">Critical</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <AlertTriangle className="h-8 w-8 mx-auto mb-2 text-orange-500" />
            <p className="text-2xl font-bold text-orange-600">
              {violations.filter(v => v.severity === 'high').length}
            </p>
            <p className="text-sm text-gray-500">High</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Clock className="h-8 w-8 mx-auto mb-2 text-yellow-500" />
            <p className="text-2xl font-bold text-yellow-600">
              {violations.filter(v => v.remediation_status === 'pending').length}
            </p>
            <p className="text-sm text-gray-500">Pending</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <p className="text-2xl font-bold text-green-600">
              {violations.filter(v => v.remediation_status === 'completed').length}
            </p>
            <p className="text-sm text-gray-500">Resolved</p>
          </CardContent>
        </Card>
      </div>

      {/* Violations List */}
      <div className="space-y-4">
        {violations.map((violation) => (
          <Card key={violation.violation_id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <AlertTriangle className={`h-5 w-5 ${getSeverityColor(violation.severity)}`} />
                  <div>
                    <h4 className="font-semibold">{violation.violation_type}</h4>
                    <p className="text-sm text-gray-500">{violation.regulation} • {violation.article}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Badge variant={getStatusBadgeVariant(violation.remediation_status)}>
                    {violation.remediation_status}
                  </Badge>
                  <Badge variant="outline" className={getSeverityColor(violation.severity)}>
                    {violation.severity}
                  </Badge>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedViolation(violation)}>
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              <p className="text-sm mb-4">{violation.description}</p>
              
              <div className="grid md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Risk Impact:</span>
                  <span className="ml-2 font-medium">{violation.risk_impact}</span>
                </div>
                <div>
                  <span className="text-gray-500">Detection Method:</span>
                  <span className="ml-2 font-medium">{violation.detection_method}</span>
                </div>
                <div>
                  <span className="text-gray-500">Assigned To:</span>
                  <span className="ml-2 font-medium">{violation.compliance_officer_assigned || 'Unassigned'}</span>
                </div>
              </div>

              {violation.remediation_actions && violation.remediation_actions.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm font-medium mb-2">Remediation Actions:</p>
                  <ul className="text-sm space-y-1">
                    {violation.remediation_actions.map((action, idx) => (
                      <li key={idx} className="flex items-center space-x-2">
                        <CheckCircle className="h-3 w-3 text-blue-500" />
                        <span>{action}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  /**
   * Render Investigations Tab
   */
  const renderInvestigations = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Compliance Investigations</h3>
        <Button>
          <Microscope className="h-4 w-4 mr-2" />
          Create Investigation
        </Button>
      </div>

      <div className="space-y-4">
        {investigations.map((investigation) => (
          <Card key={investigation.investigation_id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <Microscope className="h-5 w-5 text-purple-500" />
                  <div>
                    <h4 className="font-semibold">{investigation.title}</h4>
                    <p className="text-sm text-gray-500">
                      {investigation.investigation_type} • Priority: {investigation.priority}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Badge variant={getStatusBadgeVariant(investigation.status)}>
                    {investigation.status}
                  </Badge>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedInvestigation(investigation)}>
                    <Eye className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              <p className="text-sm mb-4">{investigation.description}</p>
              
              <div className="grid md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Assigned To:</span>
                  <span className="ml-2 font-medium">{investigation.assigned_investigator}</span>
                </div>
                <div>
                  <span className="text-gray-500">Created:</span>
                  <span className="ml-2 text-xs">{formatTimestamp(investigation.created_timestamp)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Deadline:</span>
                  <span className="ml-2 text-xs">
                    {investigation.deadline ? formatTimestamp(investigation.deadline) : 'No deadline'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500">Related Events:</span>
                  <span className="ml-2 font-medium">{investigation.related_events_count || 0}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  /**
   * Render Reports Tab
   */
  const renderReports = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Compliance Reports</h3>
        <Button onClick={() => {
          // Demo report generation
          const startDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
          const endDate = new Date().toISOString();
          generateReport('regulatory_compliance', 'GDPR', startDate, endDate);
        }}>
          <BarChart3 className="h-4 w-4 mr-2" />
          Generate Report
        </Button>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-6 text-center">
            <FileText className="h-8 w-8 mx-auto mb-2 text-blue-500" />
            <p className="text-lg font-bold">12</p>
            <p className="text-sm text-gray-500">Reports Generated</p>
            <Button variant="outline" size="sm" className="mt-2">
              <Download className="h-4 w-4 mr-2" />
              Download Latest
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Calendar className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <p className="text-lg font-bold">Weekly</p>
            <p className="text-sm text-gray-500">Automated Schedule</p>
            <Button variant="outline" size="sm" className="mt-2">
              <Settings className="h-4 w-4 mr-2" />
              Configure
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 text-center">
            <Users className="h-8 w-8 mx-auto mb-2 text-purple-500" />
            <p className="text-lg font-bold">8</p>
            <p className="text-sm text-gray-500">Report Recipients</p>
            <Button variant="outline" size="sm" className="mt-2">
              <Bell className="h-4 w-4 mr-2" />
              Manage
            </Button>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Available Report Types</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            {[
              { type: 'regulatory_compliance', name: 'Regulatory Compliance Report', description: 'Comprehensive compliance status by regulation' },
              { type: 'audit_summary', name: 'Audit Trail Summary', description: 'Summary of audit activities and integrity' },
              { type: 'incident_report', name: 'Incident Report', description: 'Detailed analysis of compliance incidents' },
              { type: 'trend_analysis', name: 'Trend Analysis', description: 'Compliance trends and predictive insights' }
            ].map((reportType) => (
              <div key={reportType.type} className="p-4 border rounded-lg">
                <h4 className="font-medium mb-2">{reportType.name}</h4>
                <p className="text-sm text-gray-500 mb-3">{reportType.description}</p>
                <Button variant="outline" size="sm">
                  <FileText className="h-4 w-4 mr-2" />
                  Generate
                </Button>
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
      <h3 className="text-lg font-semibold">Compliance Analytics</h3>
      
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Compliance Trends</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>This Week</span>
                  <span className="text-green-600">+5.2%</span>
                </div>
                <Progress value={95.2} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>This Month</span>
                  <span className="text-green-600">+2.8%</span>
                </div>
                <Progress value={93.8} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>This Quarter</span>
                  <span className="text-yellow-600">-1.2%</span>
                </div>
                <Progress value={94.5} className="h-2" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="h-5 w-5" />
              <span>Pattern Detection</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm">Active Patterns</span>
                <span className="font-medium">{patterns.filter(p => p.is_active).length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">High Confidence</span>
                <span className="font-medium">{patterns.filter(p => p.confidence_score > 0.8).length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm">Anomalies Detected</span>
                <span className="font-medium text-red-600">3</span>
              </div>
              <Button variant="outline" size="sm" className="w-full">
                <Target className="h-4 w-4 mr-2" />
                Analyze Patterns
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-4 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-blue-600">{metrics.system_metrics?.events_logged || 0}</p>
              <p className="text-sm text-gray-500">Events Logged</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-red-600">{metrics.system_metrics?.violations_detected || 0}</p>
              <p className="text-sm text-gray-500">Violations Detected</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-purple-600">{metrics.system_metrics?.investigations_created || 0}</p>
              <p className="text-sm text-gray-500">Investigations Created</p>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <p className="text-2xl font-bold text-green-600">{metrics.system_metrics?.reports_generated || 0}</p>
              <p className="text-sm text-gray-500">Reports Generated</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  if (loading && auditEvents.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-2" />
          <p>Loading Compliance Audit Trail...</p>
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
          <span>Comprehensive Compliance Audit Trail</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Advanced compliance logging, investigation, and regulatory reporting with immutable audit trails
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="dashboard" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Dashboard</span>
          </TabsTrigger>
          <TabsTrigger value="events" className="flex items-center space-x-2">
            <FileText className="h-4 w-4" />
            <span>Audit Events</span>
          </TabsTrigger>
          <TabsTrigger value="violations" className="flex items-center space-x-2">
            <AlertTriangle className="h-4 w-4" />
            <span>Violations</span>
          </TabsTrigger>
          <TabsTrigger value="investigations" className="flex items-center space-x-2">
            <Microscope className="h-4 w-4" />
            <span>Investigations</span>
          </TabsTrigger>
          <TabsTrigger value="reports" className="flex items-center space-x-2">
            <BookOpen className="h-4 w-4" />
            <span>Reports</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Analytics</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard">
          {renderDashboard()}
        </TabsContent>

        <TabsContent value="events">
          {renderAuditEvents()}
        </TabsContent>

        <TabsContent value="violations">
          {renderViolations()}
        </TabsContent>

        <TabsContent value="investigations">
          {renderInvestigations()}
        </TabsContent>

        <TabsContent value="reports">
          {renderReports()}
        </TabsContent>

        <TabsContent value="analytics">
          {renderAnalytics()}
        </TabsContent>
      </Tabs>

      {/* Event Details Modal */}
      {selectedEvent && (
        <Dialog open={!!selectedEvent} onOpenChange={() => setSelectedEvent(null)}>
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle>Audit Event Details</DialogTitle>
              <DialogDescription>
                Event ID: {selectedEvent.event_id}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <Label>Event Type</Label>
                  <p className="font-medium">{selectedEvent.event_type}</p>
                </div>
                <div>
                  <Label>Action</Label>
                  <p className="font-medium">{selectedEvent.action}</p>
                </div>
                <div>
                  <Label>Entity</Label>
                  <p className="font-medium">{selectedEvent.entity_type}: {selectedEvent.entity_id}</p>
                </div>
                <div>
                  <Label>Regulation</Label>
                  <p className="font-medium">{selectedEvent.regulation || 'None'}</p>
                </div>
                <div>
                  <Label>Severity</Label>
                  <Badge variant="outline" className={getSeverityColor(selectedEvent.severity)}>
                    {selectedEvent.severity}
                  </Badge>
                </div>
                <div>
                  <Label>Risk Score</Label>
                  <p className="font-medium">{(selectedEvent.risk_score * 100).toFixed(1)}%</p>
                </div>
              </div>
              
              <div>
                <Label>Event Details</Label>
                <pre className="text-xs bg-gray-100 p-3 rounded-lg overflow-auto">
                  {JSON.stringify(selectedEvent.details || {}, null, 2)}
                </pre>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setSelectedEvent(null)}>
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default ComplianceAuditTrail;