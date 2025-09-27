/**
 * Compliance Network Graph Visualization Component
 * 
 * This component provides an interactive interface for visualizing and exploring
 * compliance networks, including risk propagation analysis, relationship mapping,
 * and regulatory change impact assessment.
 * 
 * Features:
 * - Interactive network visualization with D3.js
 * - Real-time risk propagation analysis
 * - Dynamic filtering and exploration
 * - Regulatory change simulation
 * - Anomaly detection and alerts
 * - Comprehensive analytics dashboard
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
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
  Slider,
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
  TablePagination
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Fullscreen as FullscreenIcon,
  Settings as SettingsIcon,
  Analytics as AnalyticsIcon,
  AccountTree as NetworkIcon,
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon,
  Psychology as PsychologyIcon,
  Timeline as TimelineIcon,
  FilterList as FilterIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon
} from '@mui/icons-material';
import * as d3 from 'd3';

interface NetworkNode {
  id: string;
  label: string;
  type: string;
  risk_score: number;
  compliance_status: string;
  x: number;
  y: number;
  centrality_scores: { [key: string]: number };
  community_id: string | null;
  properties: { [key: string]: any };
}

interface NetworkEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight: number;
  confidence: number;
  risk_contribution: number;
  properties: { [key: string]: any };
}

interface NetworkData {
  clients: any[];
  advisors: any[];
  regulations: any[];
  relationships: NetworkEdge[];
}

interface VisualizationData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  layout: string;
  statistics: any;
  legends: any;
}

interface RiskPropagationResult {
  propagation_id: string;
  source_nodes: string[];
  affected_nodes: { [key: string]: number };
  propagation_paths: any[];
  total_risk_increase: number;
  propagation_time_steps: number;
  convergence_achieved: boolean;
  critical_paths: any[];
  mitigation_recommendations: string[];
}

interface AnalysisResult {
  analysis_id: string;
  network_stats: any;
  centrality_analysis: any;
  community_structure: any;
  risk_assessment: any;
  anomaly_detection: any;
  recommendations: string[];
  visualization_data: VisualizationData;
  analysis_timestamp: string;
}

const ComplianceNetworkVisualization: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);
  const [riskPropagationResult, setRiskPropagationResult] = useState<RiskPropagationResult | null>(null);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [hoveredNode, setHoveredNode] = useState<NetworkNode | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  
  // Filter states
  const [riskThreshold, setRiskThreshold] = useState(0.0);
  const [selectedNodeTypes, setSelectedNodeTypes] = useState<string[]>([]);
  const [selectedEdgeTypes, setSelectedEdgeTypes] = useState<string[]>([]);
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('spring');
  
  // Analysis options
  const [propagationType, setPropagationType] = useState('cascade');
  const [maxPropagationSteps, setMaxPropagationSteps] = useState(10);
  const [enableRealTimeAnalysis, setEnableRealTimeAnalysis] = useState(false);
  
  // D3 visualization refs
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<NetworkNode, NetworkEdge> | null>(null);
  
  // Constants
  const NODE_TYPE_COLORS = {
    client: '#3f51b5',
    advisor: '#2196f3',
    regulation: '#f44336',
    mortgage_product: '#4caf50',
    lender: '#ff9800',
    compliance_rule: '#9c27b0',
    risk_factor: '#e91e63',
    document: '#607d8b',
    process: '#795548',
    audit_event: '#ff5722'
  };

  const RISK_COLORS = {
    very_low: '#4caf50',
    low: '#8bc34a',
    medium: '#ffc107',
    high: '#ff9800',
    very_high: '#f44336',
    critical: '#d32f2f'
  };

  // Sample data for demonstration
  const sampleNetworkData: NetworkData = useMemo(() => ({
    clients: [
      { id: 'client_001', name: 'Jan de Jong', risk_score: 0.3, compliance_status: 'compliant' },
      { id: 'client_002', name: 'Marie van der Berg', risk_score: 0.7, compliance_status: 'review_required' },
      { id: 'client_003', name: 'Pieter Janssen', risk_score: 0.9, compliance_status: 'non_compliant' }
    ],
    advisors: [
      { id: 'advisor_001', name: 'Hans Mulder', risk_score: 0.2, compliance_status: 'active' },
      { id: 'advisor_002', name: 'Sophie de Vries', risk_score: 0.4, compliance_status: 'active' }
    ],
    regulations: [
      { id: 'regulation_001', title: 'AFM Gedragscode', severity: 0.8, compliance_status: 'active' },
      { id: 'regulation_002', title: 'WFTE Compliance', severity: 0.6, compliance_status: 'active' },
      { id: 'regulation_003', title: 'BKR Verification', severity: 0.9, compliance_status: 'active' }
    ],
    relationships: [
      { 
        id: 'rel_001', 
        source: 'advisor_001', 
        target: 'client_001', 
        type: 'advises', 
        weight: 1.0, 
        confidence: 0.9, 
        risk_contribution: 0.2, 
        properties: {} 
      },
      { 
        id: 'rel_002', 
        source: 'advisor_002', 
        target: 'client_002', 
        type: 'advises', 
        weight: 1.2, 
        confidence: 0.8, 
        risk_contribution: 0.4, 
        properties: {} 
      },
      { 
        id: 'rel_003', 
        source: 'regulation_001', 
        target: 'advisor_001', 
        type: 'applies_to', 
        weight: 0.8, 
        confidence: 1.0, 
        risk_contribution: 0.3, 
        properties: {} 
      }
    ]
  }), []);

  // Initialize with sample data
  useEffect(() => {
    setNetworkData(sampleNetworkData);
  }, [sampleNetworkData]);

  // API functions
  const analyzeNetwork = useCallback(async (data: NetworkData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/compliance-network/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Analysis failed');
      }

      setAnalysisResult(result);
      setVisualizationData(result.visualization_data);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const analyzeRiskPropagation = useCallback(async (sourceNodes: string[]) => {
    if (!networkData) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/compliance-network/risk-propagation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          network_data: networkData,
          propagation_request: {
            source_nodes: sourceNodes,
            propagation_type: propagationType,
            max_steps: maxPropagationSteps,
            convergence_tolerance: 0.001
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Risk propagation analysis failed');
      }

      setRiskPropagationResult(result);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [networkData, propagationType, maxPropagationSteps]);

  const generateVisualization = useCallback(async (options: any = {}) => {
    if (!networkData) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/compliance-network/visualization', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          network_data: networkData,
          layout_algorithm: layoutAlgorithm,
          include_centrality: true,
          include_communities: true,
          filter_options: {
            min_risk_score: riskThreshold,
            node_types: selectedNodeTypes.length > 0 ? selectedNodeTypes : undefined,
            edge_types: selectedEdgeTypes.length > 0 ? selectedEdgeTypes : undefined
          },
          ...options
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (!result.success) {
        throw new Error(result.error || 'Visualization generation failed');
      }

      setVisualizationData(result.visualization_data);
      
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [networkData, layoutAlgorithm, riskThreshold, selectedNodeTypes, selectedEdgeTypes]);

  // D3 visualization setup
  const setupVisualization = useCallback(() => {
    if (!visualizationData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous visualization

    const width = 800;
    const height = 600;
    
    svg.attr('width', width).attr('height', height);

    // Create zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    const g = svg.append('g');

    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(visualizationData.edges)
      .enter().append('line')
      .attr('stroke-width', d => Math.sqrt(d.weight))
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6);

    // Create nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(visualizationData.nodes)
      .enter().append('circle')
      .attr('r', d => 5 + Math.sqrt(d.risk_score) * 10)
      .attr('fill', d => NODE_TYPE_COLORS[d.type as keyof typeof NODE_TYPE_COLORS] || '#999')
      .attr('stroke', d => d.risk_score > 0.7 ? '#f44336' : '#fff')
      .attr('stroke-width', d => d.risk_score > 0.7 ? 3 : 1)
      .style('cursor', 'pointer')
      .call(d3.drag<SVGCircleElement, NetworkNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended)
      );

    // Add labels
    const labels = g.append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(visualizationData.nodes)
      .enter().append('text')
      .text(d => d.label)
      .attr('font-size', 12)
      .attr('dx', 12)
      .attr('dy', 4)
      .style('pointer-events', 'none');

    // Create simulation
    const simulation = d3.forceSimulation(visualizationData.nodes)
      .force('link', d3.forceLink(visualizationData.edges).id((d: any) => d.id))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .on('tick', ticked);

    simulationRef.current = simulation;

    function ticked() {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      labels
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    }

    function dragstarted(event: d3.D3DragEvent<SVGCircleElement, NetworkNode, NetworkNode>, d: NetworkNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: d3.D3DragEvent<SVGCircleElement, NetworkNode, NetworkNode>, d: NetworkNode) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: d3.D3DragEvent<SVGCircleElement, NetworkNode, NetworkNode>, d: NetworkNode) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Add event handlers
    node
      .on('click', (event, d) => {
        setSelectedNodes(prev => 
          prev.includes(d.id) 
            ? prev.filter(id => id !== d.id)
            : [...prev, d.id]
        );
      })
      .on('mouseover', (event, d) => {
        setHoveredNode(d);
      })
      .on('mouseout', () => {
        setHoveredNode(null);
      });

  }, [visualizationData]);

  // Setup visualization when data changes
  useEffect(() => {
    setupVisualization();
  }, [setupVisualization]);

  // Auto-analyze network when data is loaded
  useEffect(() => {
    if (networkData && !analysisResult) {
      analyzeNetwork(networkData);
    }
  }, [networkData, analysisResult, analyzeNetwork]);

  // Event handlers
  const handleRunAnalysis = () => {
    if (networkData) {
      analyzeNetwork(networkData);
    }
  };

  const handleRunRiskPropagation = () => {
    if (selectedNodes.length > 0) {
      analyzeRiskPropagation(selectedNodes);
    }
  };

  const handleRegenerateVisualization = () => {
    generateVisualization();
  };

  const getRiskLevel = (score: number): string => {
    if (score < 0.2) return 'very_low';
    if (score < 0.4) return 'low';
    if (score < 0.6) return 'medium';
    if (score < 0.8) return 'high';
    return score < 0.9 ? 'very_high' : 'critical';
  };

  const getRiskColor = (score: number): string => {
    const level = getRiskLevel(score);
    return RISK_COLORS[level as keyof typeof RISK_COLORS];
  };

  // Tab content components
  const NetworkOverviewTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Network Visualization</Typography>
              <Box>
                <Tooltip title="Zoom In">
                  <IconButton onClick={() => {/* Implement zoom in */}}>
                    <ZoomInIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Zoom Out">
                  <IconButton onClick={() => {/* Implement zoom out */}}>
                    <ZoomOutIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Center View">
                  <IconButton onClick={() => {/* Implement center view */}}>
                    <CenterIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Regenerate Layout">
                  <IconButton onClick={handleRegenerateVisualization} disabled={loading}>
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
            
            <Box sx={{ position: 'relative', height: '600px', border: '1px solid #ddd', borderRadius: 1 }}>
              {loading && (
                <Box sx={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 1 }}>
                  <CircularProgress />
                </Box>
              )}
              <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />
            </Box>

            {selectedNodes.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Selected Nodes: {selectedNodes.length}
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {selectedNodes.map(nodeId => {
                    const node = visualizationData?.nodes.find(n => n.id === nodeId);
                    return (
                      <Chip 
                        key={nodeId}
                        label={node?.label || nodeId}
                        onDelete={() => setSelectedNodes(prev => prev.filter(id => id !== nodeId))}
                        color={node && node.risk_score > 0.7 ? 'error' : 'default'}
                      />
                    );
                  })}
                </Box>
                <Button 
                  variant="outlined" 
                  size="small" 
                  sx={{ mt: 1 }}
                  onClick={handleRunRiskPropagation}
                  disabled={loading}
                >
                  Analyze Risk Propagation
                </Button>
              </Box>
            )}

            {hoveredNode && (
              <Card sx={{ mt: 2, p: 2 }}>
                <Typography variant="h6">{hoveredNode.label}</Typography>
                <Typography variant="body2" color="textSecondary">
                  Type: {hoveredNode.type}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Risk Score: {hoveredNode.risk_score.toFixed(2)}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Status: {hoveredNode.compliance_status}
                </Typography>
                {hoveredNode.centrality_scores && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="subtitle2">Centrality Scores:</Typography>
                    {Object.entries(hoveredNode.centrality_scores).map(([key, value]) => (
                      <Typography key={key} variant="caption" display="block">
                        {key}: {(value as number).toFixed(3)}
                      </Typography>
                    ))}
                  </Box>
                )}
              </Card>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={4}>
        <Grid container spacing={2}>
          {analysisResult && (
            <>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Network Statistics
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      <Typography variant="body2">
                        Nodes: {analysisResult.network_stats.node_count}
                      </Typography>
                      <Typography variant="body2">
                        Edges: {analysisResult.network_stats.edge_count}
                      </Typography>
                      <Typography variant="body2">
                        Density: {(analysisResult.network_stats.density * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2">
                        Connected: {analysisResult.network_stats.is_connected ? 'Yes' : 'No'}
                      </Typography>
                      {analysisResult.network_stats.clustering_coefficient && (
                        <Typography variant="body2">
                          Clustering: {(analysisResult.network_stats.clustering_coefficient * 100).toFixed(1)}%
                        </Typography>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Risk Assessment
                    </Typography>
                    {analysisResult.risk_assessment && analysisResult.risk_assessment.high_risk_nodes && (
                      <Box>
                        <Typography variant="body2" gutterBottom>
                          High Risk Nodes: {analysisResult.risk_assessment.high_risk_nodes.length}
                        </Typography>
                        {analysisResult.risk_assessment.high_risk_nodes.slice(0, 3).map((nodeId: string) => {
                          const node = visualizationData?.nodes.find(n => n.id === nodeId);
                          return (
                            <Chip 
                              key={nodeId}
                              label={node?.label || nodeId}
                              size="small"
                              color="error"
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          );
                        })}
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Anomalies
                    </Typography>
                    {analysisResult.anomaly_detection && (
                      <Box>
                        <Typography variant="body2" color="error">
                          Anomalous Nodes: {analysisResult.anomaly_detection.anomalous_nodes?.length || 0}
                        </Typography>
                        <Typography variant="body2" color="warning.main">
                          Anomalous Edges: {analysisResult.anomaly_detection.anomalous_edges?.length || 0}
                        </Typography>
                        <Typography variant="body2" color="info.main">
                          Structural Issues: {analysisResult.anomaly_detection.structural_anomalies?.length || 0}
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </>
          )}
        </Grid>
      </Grid>
    </Grid>
  );

  const RiskAnalysisTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Risk Propagation Analysis
            </Typography>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Propagation Type</InputLabel>
              <Select
                value={propagationType}
                onChange={(e) => setPropagationType(e.target.value)}
                label="Propagation Type"
              >
                <MenuItem value="linear">Linear</MenuItem>
                <MenuItem value="exponential">Exponential</MenuItem>
                <MenuItem value="logarithmic">Logarithmic</MenuItem>
                <MenuItem value="threshold">Threshold</MenuItem>
                <MenuItem value="cascade">Cascade</MenuItem>
              </Select>
            </FormControl>

            <Typography gutterBottom>Max Propagation Steps: {maxPropagationSteps}</Typography>
            <Slider
              value={maxPropagationSteps}
              onChange={(_, value) => setMaxPropagationSteps(value as number)}
              min={1}
              max={20}
              step={1}
              marks
              sx={{ mb: 2 }}
            />

            <Button 
              variant="contained" 
              fullWidth
              onClick={handleRunRiskPropagation}
              disabled={loading || selectedNodes.length === 0}
            >
              Run Risk Propagation Analysis
            </Button>

            {selectedNodes.length === 0 && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Select nodes in the network visualization to analyze risk propagation.
              </Alert>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        {riskPropagationResult && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Propagation Results
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Total Risk Increase: <strong>{riskPropagationResult.total_risk_increase.toFixed(3)}</strong>
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Affected Nodes: <strong>{Object.keys(riskPropagationResult.affected_nodes).length}</strong>
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Propagation Steps: <strong>{riskPropagationResult.propagation_time_steps}</strong>
                </Typography>
                <Typography variant="body2" gutterBottom>
                  Converged: <strong>{riskPropagationResult.convergence_achieved ? 'Yes' : 'No'}</strong>
                </Typography>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                Mitigation Recommendations:
              </Typography>
              <List dense>
                {riskPropagationResult.mitigation_recommendations.map((recommendation, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckCircleIcon fontSize="small" color="primary" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={recommendation}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        )}
      </Grid>

      {riskPropagationResult && (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Critical Risk Paths
              </Typography>
              
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Source</TableCell>
                      <TableCell>Target</TableCell>
                      <TableCell>Path Length</TableCell>
                      <TableCell>Risk Score</TableCell>
                      <TableCell>Impact</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {riskPropagationResult.critical_paths.slice(0, 10).map((path, index) => (
                      <TableRow key={index}>
                        <TableCell>{path.source}</TableCell>
                        <TableCell>{path.target}</TableCell>
                        <TableCell>{path.length}</TableCell>
                        <TableCell>
                          <Box sx={{ color: getRiskColor(path.risk_score) }}>
                            {path.risk_score.toFixed(3)}
                          </Box>
                        </TableCell>
                        <TableCell>{path.impact.toFixed(3)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      )}
    </Grid>
  );

  const RecommendationsTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        {analysisResult && (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Compliance Recommendations
              </Typography>
              
              <List>
                {analysisResult.recommendations.map((recommendation, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <CheckCircleIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={recommendation}
                      primaryTypographyProps={{ variant: 'body1' }}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        )}
      </Grid>
    </Grid>
  );

  const AnalyticsTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Node Type Distribution
            </Typography>
            {analysisResult && analysisResult.network_stats.node_type_distribution && (
              <Box>
                {Object.entries(analysisResult.network_stats.node_type_distribution).map(([type, count]) => (
                  <Box key={type} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">{type}:</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{count as number}</Typography>
                  </Box>
                ))}
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Edge Type Distribution
            </Typography>
            {analysisResult && analysisResult.network_stats.edge_type_distribution && (
              <Box>
                {Object.entries(analysisResult.network_stats.edge_type_distribution).map(([type, count]) => (
                  <Box key={type} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">{type}:</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>{count as number}</Typography>
                  </Box>
                ))}
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>

      {analysisResult && analysisResult.network_stats.risk_statistics && (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Distribution
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2">Mean Risk:</Typography>
                  <Typography variant="h6" color={getRiskColor(analysisResult.network_stats.risk_statistics.mean)}>
                    {analysisResult.network_stats.risk_statistics.mean.toFixed(3)}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2">Max Risk:</Typography>
                  <Typography variant="h6" color={getRiskColor(analysisResult.network_stats.risk_statistics.max)}>
                    {analysisResult.network_stats.risk_statistics.max.toFixed(3)}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2">95th Percentile:</Typography>
                  <Typography variant="h6" color={getRiskColor(analysisResult.network_stats.risk_statistics.percentiles['95'])}>
                    {analysisResult.network_stats.risk_statistics.percentiles['95'].toFixed(3)}
                  </Typography>
                </Grid>
                <Grid item xs={6} md={3}>
                  <Typography variant="body2">Standard Deviation:</Typography>
                  <Typography variant="h6">
                    {analysisResult.network_stats.risk_statistics.std.toFixed(3)}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      )}
    </Grid>
  );

  // Settings Dialog
  const SettingsDialog = () => (
    <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="md" fullWidth>
      <DialogTitle>Visualization Settings</DialogTitle>
      <DialogContent>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Layout Algorithm</InputLabel>
              <Select
                value={layoutAlgorithm}
                onChange={(e) => setLayoutAlgorithm(e.target.value)}
                label="Layout Algorithm"
              >
                <MenuItem value="spring">Spring Layout</MenuItem>
                <MenuItem value="circular">Circular Layout</MenuItem>
                <MenuItem value="hierarchical">Hierarchical Layout</MenuItem>
                <MenuItem value="force_atlas">Force Atlas</MenuItem>
              </Select>
            </FormControl>

            <Typography gutterBottom>Risk Threshold: {riskThreshold.toFixed(2)}</Typography>
            <Slider
              value={riskThreshold}
              onChange={(_, value) => setRiskThreshold(value as number)}
              min={0}
              max={1}
              step={0.05}
              marks
              sx={{ mb: 2 }}
            />

            <FormControlLabel
              control={
                <Switch
                  checked={enableRealTimeAnalysis}
                  onChange={(e) => setEnableRealTimeAnalysis(e.target.checked)}
                />
              }
              label="Enable Real-time Analysis"
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>
              Node Types Filter
            </Typography>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <Select
                multiple
                value={selectedNodeTypes}
                onChange={(e) => setSelectedNodeTypes(e.target.value as string[])}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {Object.keys(NODE_TYPE_COLORS).map((type) => (
                  <MenuItem key={type} value={type}>
                    {type}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Typography variant="subtitle1" gutterBottom>
              Edge Types Filter
            </Typography>
            <FormControl fullWidth>
              <Select
                multiple
                value={selectedEdgeTypes}
                onChange={(e) => setSelectedEdgeTypes(e.target.value as string[])}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {(selected as string[]).map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                <MenuItem value="advises">Advises</MenuItem>
                <MenuItem value="applies_to">Applies To</MenuItem>
                <MenuItem value="complies_with">Complies With</MenuItem>
                <MenuItem value="violates">Violates</MenuItem>
                <MenuItem value="depends_on">Depends On</MenuItem>
                <MenuItem value="influences">Influences</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowSettings(false)}>Cancel</Button>
        <Button onClick={() => { setShowSettings(false); handleRegenerateVisualization(); }}>
          Apply & Regenerate
        </Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Compliance Network Visualization
          </Typography>
          <Typography variant="body1" color="textSecondary">
            Interactive analysis of compliance relationships, risk propagation, and regulatory impact
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRunAnalysis}
            disabled={loading}
          >
            Refresh Analysis
          </Button>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            disabled={!visualizationData}
          >
            Export Data
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <AlertTitle>Analysis Error</AlertTitle>
          {error}
        </Alert>
      )}

      {loading && (
        <Box sx={{ mb: 3 }}>
          <LinearProgress />
          <Typography variant="body2" sx={{ mt: 1 }}>
            Processing network analysis...
          </Typography>
        </Box>
      )}

      <Paper sx={{ mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, value) => setActiveTab(value)}>
          <Tab icon={<NetworkIcon />} label="Network Overview" />
          <Tab icon={<SecurityIcon />} label="Risk Analysis" />
          <Tab icon={<PsychologyIcon />} label="Recommendations" />
          <Tab icon={<AnalyticsIcon />} label="Analytics" />
        </Tabs>
      </Paper>

      {activeTab === 0 && <NetworkOverviewTab />}
      {activeTab === 1 && <RiskAnalysisTab />}
      {activeTab === 2 && <RecommendationsTab />}
      {activeTab === 3 && <AnalyticsTab />}

      <SettingsDialog />
    </Box>
  );
};

export default ComplianceNetworkVisualization;
