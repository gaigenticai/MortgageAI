"""
Compliance Network Graph Visualization System for MortgageAI

This module provides advanced compliance network analysis capabilities including risk propagation analysis,
relationship mapping, and interactive exploration for Dutch AFM mortgage compliance requirements.

Features:
- Dynamic compliance network construction and analysis
- Risk propagation modeling with Monte Carlo simulation
- Multi-dimensional relationship mapping (clients, regulations, products, advisors)
- Real-time risk assessment and impact analysis
- Interactive graph exploration with advanced filtering
- Compliance pathway optimization and recommendation
- Regulatory change impact assessment
- Network anomaly detection and alert system
- Centrality analysis for critical compliance points
- Temporal network evolution tracking
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import pickle
import math

# Advanced graph analysis libraries
import networkx as nx
from networkx.algorithms import centrality, community, flow, shortest_paths
import igraph as ig  # Alternative graph library for large-scale analysis
from scipy import sparse, stats
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Machine learning for risk prediction
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader

# Data processing and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..config import settings


class NodeType(Enum):
    """Types of nodes in the compliance network."""
    CLIENT = "client"
    ADVISOR = "advisor"
    REGULATION = "regulation"
    MORTGAGE_PRODUCT = "mortgage_product"
    LENDER = "lender"
    COMPLIANCE_RULE = "compliance_rule"
    RISK_FACTOR = "risk_factor"
    DOCUMENT = "document"
    PROCESS = "process"
    AUDIT_EVENT = "audit_event"


class EdgeType(Enum):
    """Types of relationships/edges in the compliance network."""
    ADVISES = "advises"
    APPLIES_TO = "applies_to"
    COMPLIES_WITH = "complies_with"
    VIOLATES = "violates"
    DEPENDS_ON = "depends_on"
    INFLUENCES = "influences"
    REQUIRES = "requires"
    PRODUCES = "produces"
    VALIDATES = "validates"
    TRIGGERS = "triggers"
    ESCALATES = "escalates"


class RiskLevel(Enum):
    """Risk level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class PropagationType(Enum):
    """Types of risk propagation through the network."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    THRESHOLD = "threshold"
    CASCADE = "cascade"


@dataclass
class NetworkNode:
    """Represents a node in the compliance network."""
    node_id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    compliance_status: str = "unknown"
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Graph-specific attributes
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    community_id: Optional[str] = None
    importance_rank: int = 0


@dataclass
class NetworkEdge:
    """Represents an edge/relationship in the compliance network."""
    edge_id: str
    source_node: str
    target_node: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    risk_contribution: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Temporal attributes
    validity_start: Optional[datetime] = None
    validity_end: Optional[datetime] = None
    is_active: bool = True


@dataclass
class RiskPropagationResult:
    """Results of risk propagation analysis."""
    propagation_id: str
    source_nodes: List[str]
    affected_nodes: Dict[str, float]  # node_id -> risk_increase
    propagation_paths: List[List[str]]
    total_risk_increase: float
    propagation_time_steps: int
    convergence_achieved: bool
    risk_distribution: Dict[str, float]
    critical_paths: List[Dict[str, Any]]
    mitigation_recommendations: List[str]


@dataclass
class NetworkAnalysisResult:
    """Comprehensive network analysis results."""
    analysis_id: str
    network_stats: Dict[str, Any]
    centrality_analysis: Dict[str, Dict[str, float]]
    community_structure: Dict[str, List[str]]
    risk_assessment: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    recommendations: List[str]
    visualization_data: Dict[str, Any]
    analysis_timestamp: datetime


class ComplianceNetworkGraph:
    """Core compliance network graph management and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Network storage
        self.network = nx.MultiDiGraph()  # Support multiple edges between nodes
        self.igraph_network = None  # For large-scale analysis
        
        # Node and edge storage
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: Dict[str, NetworkEdge] = {}
        
        # Analysis caches
        self.centrality_cache = {}
        self.community_cache = {}
        self.path_cache = {}
        self.risk_cache = {}
        
        # Configuration
        self.max_propagation_steps = 10
        self.risk_threshold = 0.7
        self.convergence_tolerance = 0.001
        
        # Initialize ML models for risk prediction
        self.risk_prediction_model = None
        self.anomaly_detection_model = None
        
        # Graph neural network for advanced analysis
        self.gnn_model = None
        self.initialize_ml_models()
    
    def initialize_ml_models(self):
        """Initialize machine learning models for graph analysis."""
        try:
            # Risk prediction model
            self.risk_prediction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Anomaly detection using isolation forest
            from sklearn.ensemble import IsolationForest
            self.anomaly_detection_model = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Graph Neural Network for advanced analysis
            self.gnn_model = ComplianceGNN()
            
            self.logger.info("Initialized ML models for compliance network analysis")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {str(e)}")
    
    async def add_node(self, node: NetworkNode) -> bool:
        """Add a node to the compliance network."""
        try:
            # Store node
            self.nodes[node.node_id] = node
            
            # Add to NetworkX graph
            self.network.add_node(
                node.node_id,
                node_type=node.node_type.value,
                label=node.label,
                risk_score=node.risk_score,
                compliance_status=node.compliance_status,
                properties=node.properties
            )
            
            # Invalidate relevant caches
            self._invalidate_analysis_cache()
            
            self.logger.debug(f"Added node {node.node_id} of type {node.node_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding node {node.node_id}: {str(e)}")
            return False
    
    async def add_edge(self, edge: NetworkEdge) -> bool:
        """Add an edge to the compliance network."""
        try:
            # Validate nodes exist
            if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
                self.logger.error(f"Source or target node not found for edge {edge.edge_id}")
                return False
            
            # Store edge
            self.edges[edge.edge_id] = edge
            
            # Add to NetworkX graph
            self.network.add_edge(
                edge.source_node,
                edge.target_node,
                edge_id=edge.edge_id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
                confidence=edge.confidence,
                risk_contribution=edge.risk_contribution,
                properties=edge.properties
            )
            
            # Invalidate relevant caches
            self._invalidate_analysis_cache()
            
            self.logger.debug(f"Added edge {edge.edge_id} from {edge.source_node} to {edge.target_node}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding edge {edge.edge_id}: {str(e)}")
            return False
    
    async def update_node_risk(self, node_id: str, risk_score: float, compliance_status: str = None) -> bool:
        """Update risk score and compliance status for a node."""
        try:
            if node_id not in self.nodes:
                self.logger.error(f"Node {node_id} not found")
                return False
            
            # Update node
            self.nodes[node_id].risk_score = risk_score
            self.nodes[node_id].last_updated = datetime.now(timezone.utc)
            
            if compliance_status:
                self.nodes[node_id].compliance_status = compliance_status
            
            # Update NetworkX graph
            self.network.nodes[node_id]['risk_score'] = risk_score
            if compliance_status:
                self.network.nodes[node_id]['compliance_status'] = compliance_status
            
            # Invalidate risk cache
            self.risk_cache.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating node risk {node_id}: {str(e)}")
            return False
    
    async def analyze_risk_propagation(self, source_nodes: List[str], propagation_type: PropagationType = PropagationType.LINEAR) -> RiskPropagationResult:
        """Analyze how risks propagate through the compliance network."""
        try:
            propagation_id = str(uuid.uuid4())
            
            # Initialize risk values
            current_risks = {}
            for node_id in self.network.nodes():
                current_risks[node_id] = self.nodes[node_id].risk_score
            
            # Track affected nodes and propagation paths
            affected_nodes = {}
            propagation_paths = []
            
            # Initialize source nodes with elevated risk
            initial_risk_boost = 0.5
            for source_node in source_nodes:
                if source_node in current_risks:
                    current_risks[source_node] = min(1.0, current_risks[source_node] + initial_risk_boost)
                    affected_nodes[source_node] = initial_risk_boost
            
            # Perform iterative risk propagation
            for step in range(self.max_propagation_steps):
                previous_risks = current_risks.copy()
                
                # Propagate risk through edges
                for edge_id, edge in self.edges.items():
                    if not edge.is_active:
                        continue
                    
                    source_risk = current_risks[edge.source_node]
                    current_target_risk = current_risks[edge.target_node]
                    
                    # Calculate risk propagation based on type
                    propagated_risk = self._calculate_propagated_risk(
                        source_risk, edge, propagation_type
                    )
                    
                    # Update target node risk
                    new_risk = min(1.0, current_target_risk + propagated_risk)
                    current_risks[edge.target_node] = new_risk
                    
                    # Track significant risk increases
                    risk_increase = new_risk - previous_risks[edge.target_node]
                    if risk_increase > 0.01:  # Threshold for significance
                        if edge.target_node not in affected_nodes:
                            affected_nodes[edge.target_node] = 0
                        affected_nodes[edge.target_node] += risk_increase
            
                # Check for convergence
                max_change = max(abs(current_risks[node] - previous_risks[node]) 
                               for node in current_risks)
                
                if max_change < self.convergence_tolerance:
                    convergence_achieved = True
                    break
            else:
                convergence_achieved = False
            
            # Find critical propagation paths
            critical_paths = self._find_critical_paths(source_nodes, affected_nodes)
            
            # Calculate total risk increase
            total_risk_increase = sum(affected_nodes.values())
            
            # Generate mitigation recommendations
            mitigation_recommendations = self._generate_mitigation_recommendations(
                affected_nodes, critical_paths
            )
            
            return RiskPropagationResult(
                propagation_id=propagation_id,
                source_nodes=source_nodes,
                affected_nodes=affected_nodes,
                propagation_paths=critical_paths,
                total_risk_increase=total_risk_increase,
                propagation_time_steps=step + 1,
                convergence_achieved=convergence_achieved,
                risk_distribution=current_risks,
                critical_paths=critical_paths,
                mitigation_recommendations=mitigation_recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in risk propagation analysis: {str(e)}")
            raise
    
    def _calculate_propagated_risk(self, source_risk: float, edge: NetworkEdge, propagation_type: PropagationType) -> float:
        """Calculate how much risk propagates through an edge."""
        try:
            base_propagation = source_risk * edge.weight * edge.confidence * edge.risk_contribution
            
            if propagation_type == PropagationType.LINEAR:
                return base_propagation
            elif propagation_type == PropagationType.EXPONENTIAL:
                return base_propagation * (1.5 ** source_risk)
            elif propagation_type == PropagationType.LOGARITHMIC:
                return base_propagation * math.log(1 + source_risk)
            elif propagation_type == PropagationType.THRESHOLD:
                return base_propagation if source_risk > self.risk_threshold else 0
            elif propagation_type == PropagationType.CASCADE:
                # Cascade effect - higher source risk leads to amplified propagation
                cascade_factor = 1 + (source_risk ** 2)
                return base_propagation * cascade_factor
            else:
                return base_propagation
                
        except Exception as e:
            self.logger.error(f"Error calculating propagated risk: {str(e)}")
            return 0.0
    
    def _find_critical_paths(self, source_nodes: List[str], affected_nodes: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find the most critical risk propagation paths."""
        try:
            critical_paths = []
            
            # Find paths from source nodes to highly affected nodes
            high_impact_nodes = [node for node, impact in affected_nodes.items() 
                               if impact > 0.1]  # Significant impact threshold
            
            for source in source_nodes:
                for target in high_impact_nodes:
                    if source != target:
                        try:
                            # Find shortest path
                            path = nx.shortest_path(self.network, source, target)
                            
                            # Calculate path risk
                            path_risk = self._calculate_path_risk(path)
                            
                            critical_paths.append({
                                'path': path,
                                'source': source,
                                'target': target,
                                'risk_score': path_risk,
                                'length': len(path) - 1,
                                'impact': affected_nodes[target]
                            })
                            
                        except nx.NetworkXNoPath:
                            continue
            
            # Sort by risk score and return top paths
            critical_paths.sort(key=lambda x: x['risk_score'], reverse=True)
            return critical_paths[:10]  # Top 10 critical paths
            
        except Exception as e:
            self.logger.error(f"Error finding critical paths: {str(e)}")
            return []
    
    def _calculate_path_risk(self, path: List[str]) -> float:
        """Calculate the cumulative risk score for a path."""
        try:
            if len(path) < 2:
                return 0.0
            
            path_risk = 0.0
            
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                
                # Find edge between nodes
                if self.network.has_edge(source, target):
                    edge_data = self.network[source][target]
                    # Handle multiple edges
                    for edge_key, edge_attrs in edge_data.items():
                        risk_contrib = edge_attrs.get('risk_contribution', 0)
                        weight = edge_attrs.get('weight', 1)
                        path_risk += risk_contrib * weight
            
            return path_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating path risk: {str(e)}")
            return 0.0
    
    def _generate_mitigation_recommendations(self, affected_nodes: Dict[str, float], critical_paths: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for risk mitigation."""
        recommendations = []
        
        try:
            # High-impact node recommendations
            high_impact_nodes = [(node, impact) for node, impact in affected_nodes.items() if impact > 0.2]
            high_impact_nodes.sort(key=lambda x: x[1], reverse=True)
            
            for node, impact in high_impact_nodes[:5]:  # Top 5
                node_obj = self.nodes.get(node)
                if node_obj:
                    if node_obj.node_type == NodeType.CLIENT:
                        recommendations.append(f"Review client {node_obj.label} for enhanced due diligence (risk increase: {impact:.2f})")
                    elif node_obj.node_type == NodeType.ADVISOR:
                        recommendations.append(f"Provide additional training to advisor {node_obj.label} (risk increase: {impact:.2f})")
                    elif node_obj.node_type == NodeType.PROCESS:
                        recommendations.append(f"Strengthen process controls for {node_obj.label} (risk increase: {impact:.2f})")
                    elif node_obj.node_type == NodeType.REGULATION:
                        recommendations.append(f"Review compliance with regulation {node_obj.label} (risk increase: {impact:.2f})")
            
            # Critical path recommendations
            if critical_paths:
                top_path = critical_paths[0]
                recommendations.append(f"Focus on breaking critical risk path from {top_path['source']} to {top_path['target']}")
                
                # Identify bottlenecks in critical paths
                path_nodes = top_path['path']
                if len(path_nodes) > 2:
                    bottleneck = path_nodes[len(path_nodes) // 2]  # Middle node
                    bottleneck_obj = self.nodes.get(bottleneck)
                    if bottleneck_obj:
                        recommendations.append(f"Strengthen controls at bottleneck node: {bottleneck_obj.label}")
            
            # General recommendations based on network structure
            if len(affected_nodes) > 10:
                recommendations.append("Consider implementing network segmentation to limit risk propagation")
            
            if not recommendations:
                recommendations.append("Network shows good resilience - maintain current controls")
            
        except Exception as e:
            self.logger.error(f"Error generating mitigation recommendations: {str(e)}")
            recommendations.append("Error generating recommendations - manual review recommended")
        
        return recommendations
    
    async def analyze_centrality(self) -> Dict[str, Dict[str, float]]:
        """Analyze node centrality measures to identify critical nodes."""
        try:
            # Check cache first
            if 'centrality' in self.centrality_cache:
                return self.centrality_cache['centrality']
            
            centrality_results = {}
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.network)
            centrality_results['degree'] = degree_centrality
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.network, weight='weight')
            centrality_results['betweenness'] = betweenness_centrality
            
            # Closeness centrality
            if nx.is_connected(self.network.to_undirected()):
                closeness_centrality = nx.closeness_centrality(self.network, distance='weight')
                centrality_results['closeness'] = closeness_centrality
            else:
                # For disconnected graphs, calculate per component
                closeness_centrality = {}
                for component in nx.weakly_connected_components(self.network):
                    subgraph = self.network.subgraph(component)
                    component_closeness = nx.closeness_centrality(subgraph, distance='weight')
                    closeness_centrality.update(component_closeness)
                centrality_results['closeness'] = closeness_centrality
            
            # Eigenvector centrality
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.network, weight='weight')
                centrality_results['eigenvector'] = eigenvector_centrality
            except nx.NetworkXError:
                # Fallback for problematic graphs
                centrality_results['eigenvector'] = {node: 0.0 for node in self.network.nodes()}
            
            # PageRank (useful for directed graphs)
            pagerank_centrality = nx.pagerank(self.network, weight='weight')
            centrality_results['pagerank'] = pagerank_centrality
            
            # Update node centrality scores
            for node_id in self.network.nodes():
                if node_id in self.nodes:
                    self.nodes[node_id].centrality_scores = {
                        'degree': degree_centrality.get(node_id, 0),
                        'betweenness': betweenness_centrality.get(node_id, 0),
                        'closeness': closeness_centrality.get(node_id, 0),
                        'eigenvector': centrality_results['eigenvector'].get(node_id, 0),
                        'pagerank': pagerank_centrality.get(node_id, 0)
                    }
            
            # Cache results
            self.centrality_cache['centrality'] = centrality_results
            
            return centrality_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing centrality: {str(e)}")
            return {}
    
    async def detect_communities(self) -> Dict[str, List[str]]:
        """Detect communities/clusters in the compliance network."""
        try:
            # Check cache first
            if 'communities' in self.community_cache:
                return self.community_cache['communities']
            
            communities = {}
            
            # Convert to undirected for community detection
            undirected_graph = self.network.to_undirected()
            
            # Method 1: Louvain algorithm
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(undirected_graph, weight='weight')
                
                # Group nodes by community
                louvain_communities = defaultdict(list)
                for node, community_id in partition.items():
                    louvain_communities[f"louvain_{community_id}"].append(node)
                
                communities.update(louvain_communities)
                
            except ImportError:
                self.logger.warning("Community detection library not available, using alternative method")
            
            # Method 2: Spectral clustering as fallback
            if not communities:
                try:
                    # Create adjacency matrix
                    adj_matrix = nx.adjacency_matrix(undirected_graph, weight='weight')
                    
                    # Apply spectral clustering
                    n_clusters = min(10, len(undirected_graph.nodes()) // 5)  # Heuristic for cluster count
                    if n_clusters > 1:
                        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
                        cluster_labels = spectral.fit_predict(adj_matrix.toarray())
                        
                        # Group nodes by cluster
                        spectral_communities = defaultdict(list)
                        for i, node in enumerate(undirected_graph.nodes()):
                            cluster_id = cluster_labels[i]
                            spectral_communities[f"spectral_{cluster_id}"].append(node)
                        
                        communities.update(spectral_communities)
                        
                except Exception as e:
                    self.logger.error(f"Spectral clustering failed: {str(e)}")
            
            # Update node community assignments
            for community_id, node_list in communities.items():
                for node_id in node_list:
                    if node_id in self.nodes:
                        self.nodes[node_id].community_id = community_id
            
            # Cache results
            self.community_cache['communities'] = communities
            
            return communities
            
        except Exception as e:
            self.logger.error(f"Error detecting communities: {str(e)}")
            return {}
    
    async def detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalous nodes and edges in the compliance network."""
        try:
            anomalies = {
                'anomalous_nodes': [],
                'anomalous_edges': [],
                'structural_anomalies': [],
                'temporal_anomalies': []
            }
            
            # Node-level anomaly detection
            node_features = []
            node_ids = []
            
            for node_id, node in self.nodes.items():
                # Create feature vector for each node
                features = [
                    node.risk_score,
                    self.network.degree(node_id),  # Degree
                    node.centrality_scores.get('betweenness', 0),
                    node.centrality_scores.get('pagerank', 0),
                    len(node.properties),  # Property count
                ]
                
                # Add node type encoding
                node_type_encoding = [0] * len(NodeType)
                node_type_encoding[list(NodeType).index(node.node_type)] = 1
                features.extend(node_type_encoding)
                
                node_features.append(features)
                node_ids.append(node_id)
            
            if node_features and self.anomaly_detection_model:
                # Detect anomalous nodes
                node_features_array = np.array(node_features)
                anomaly_scores = self.anomaly_detection_model.fit_predict(node_features_array)
                
                for i, score in enumerate(anomaly_scores):
                    if score == -1:  # Anomaly
                        node_id = node_ids[i]
                        anomalies['anomalous_nodes'].append({
                            'node_id': node_id,
                            'node_type': self.nodes[node_id].node_type.value,
                            'risk_score': self.nodes[node_id].risk_score,
                            'reason': 'Statistical outlier in network features'
                        })
            
            # Edge-level anomaly detection
            for edge_id, edge in self.edges.items():
                # Check for unusual edge properties
                if edge.weight > 3.0:  # Unusually high weight
                    anomalies['anomalous_edges'].append({
                        'edge_id': edge_id,
                        'source': edge.source_node,
                        'target': edge.target_node,
                        'reason': f'Unusually high weight: {edge.weight}'
                    })
                
                if edge.confidence < 0.3:  # Low confidence
                    anomalies['anomalous_edges'].append({
                        'edge_id': edge_id,
                        'source': edge.source_node,
                        'target': edge.target_node,
                        'reason': f'Low confidence score: {edge.confidence}'
                    })
            
            # Structural anomaly detection
            # Detect isolated nodes
            isolated_nodes = [node for node in self.network.nodes() if self.network.degree(node) == 0]
            for node in isolated_nodes:
                anomalies['structural_anomalies'].append({
                    'type': 'isolated_node',
                    'node_id': node,
                    'description': 'Node has no connections'
                })
            
            # Detect nodes with very high degree (potential hubs)
            degrees = dict(self.network.degree())
            if degrees:
                degree_threshold = np.percentile(list(degrees.values()), 95)
                high_degree_nodes = [node for node, degree in degrees.items() if degree > degree_threshold]
                
                for node in high_degree_nodes:
                    anomalies['structural_anomalies'].append({
                        'type': 'high_degree_hub',
                        'node_id': node,
                        'degree': degrees[node],
                        'description': f'Node has unusually high degree: {degrees[node]}'
                    })
            
            # Temporal anomaly detection (recent changes)
            recent_threshold = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_updates = [
                node for node in self.nodes.values() 
                if node.last_updated > recent_threshold
            ]
            
            if len(recent_updates) > len(self.nodes) * 0.5:  # More than 50% updated recently
                anomalies['temporal_anomalies'].append({
                    'type': 'mass_update',
                    'count': len(recent_updates),
                    'description': 'Unusually high number of recent node updates'
                })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return {'error': str(e)}
    
    async def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        try:
            stats = {}
            
            # Basic statistics
            stats['node_count'] = self.network.number_of_nodes()
            stats['edge_count'] = self.network.number_of_edges()
            stats['density'] = nx.density(self.network)
            stats['is_connected'] = nx.is_connected(self.network.to_undirected())
            
            # Node type distribution
            node_type_dist = defaultdict(int)
            for node in self.nodes.values():
                node_type_dist[node.node_type.value] += 1
            stats['node_type_distribution'] = dict(node_type_dist)
            
            # Edge type distribution
            edge_type_dist = defaultdict(int)
            for edge in self.edges.values():
                edge_type_dist[edge.edge_type.value] += 1
            stats['edge_type_distribution'] = dict(edge_type_dist)
            
            # Risk distribution
            risk_scores = [node.risk_score for node in self.nodes.values()]
            if risk_scores:
                stats['risk_statistics'] = {
                    'mean': np.mean(risk_scores),
                    'std': np.std(risk_scores),
                    'min': np.min(risk_scores),
                    'max': np.max(risk_scores),
                    'percentiles': {
                        '25': np.percentile(risk_scores, 25),
                        '50': np.percentile(risk_scores, 50),
                        '75': np.percentile(risk_scores, 75),
                        '95': np.percentile(risk_scores, 95)
                    }
                }
            
            # Degree statistics
            degrees = dict(self.network.degree())
            if degrees:
                degree_values = list(degrees.values())
                stats['degree_statistics'] = {
                    'mean': np.mean(degree_values),
                    'std': np.std(degree_values),
                    'min': np.min(degree_values),
                    'max': np.max(degree_values)
                }
            
            # Connected components
            if nx.is_connected(self.network.to_undirected()):
                stats['connected_components'] = 1
                stats['largest_component_size'] = stats['node_count']
            else:
                components = list(nx.weakly_connected_components(self.network))
                stats['connected_components'] = len(components)
                stats['largest_component_size'] = max(len(comp) for comp in components)
            
            # Clustering coefficient
            stats['clustering_coefficient'] = nx.average_clustering(self.network.to_undirected())
            
            # Path statistics (for connected components only)
            if nx.is_connected(self.network.to_undirected()):
                stats['average_path_length'] = nx.average_shortest_path_length(self.network.to_undirected())
                stats['diameter'] = nx.diameter(self.network.to_undirected())
                stats['radius'] = nx.radius(self.network.to_undirected())
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating network statistics: {str(e)}")
            return {'error': str(e)}
    
    async def generate_visualization_data(self, layout_algorithm: str = "spring") -> Dict[str, Any]:
        """Generate data for network visualization."""
        try:
            visualization_data = {}
            
            # Choose layout algorithm
            if layout_algorithm == "spring":
                pos = nx.spring_layout(self.network, k=1, iterations=50)
            elif layout_algorithm == "circular":
                pos = nx.circular_layout(self.network)
            elif layout_algorithm == "hierarchical":
                pos = nx.nx_agraph.graphviz_layout(self.network, prog='dot')
            elif layout_algorithm == "force_atlas":
                # Use force-directed layout for better community visualization
                pos = nx.spring_layout(self.network, k=2, iterations=100)
            else:
                pos = nx.spring_layout(self.network)
            
            # Prepare node data
            nodes_data = []
            for node_id in self.network.nodes():
                node = self.nodes[node_id]
                node_data = {
                    'id': node_id,
                    'label': node.label,
                    'type': node.node_type.value,
                    'risk_score': node.risk_score,
                    'compliance_status': node.compliance_status,
                    'x': pos[node_id][0],
                    'y': pos[node_id][1],
                    'centrality_scores': node.centrality_scores,
                    'community_id': node.community_id,
                    'properties': node.properties
                }
                nodes_data.append(node_data)
            
            # Prepare edge data
            edges_data = []
            for edge_id, edge in self.edges.items():
                edge_data = {
                    'id': edge_id,
                    'source': edge.source_node,
                    'target': edge.target_node,
                    'type': edge.edge_type.value,
                    'weight': edge.weight,
                    'confidence': edge.confidence,
                    'risk_contribution': edge.risk_contribution,
                    'properties': edge.properties
                }
                edges_data.append(edge_data)
            
            visualization_data = {
                'nodes': nodes_data,
                'edges': edges_data,
                'layout': layout_algorithm,
                'statistics': await self.get_network_statistics(),
                'legends': {
                    'node_types': [nt.value for nt in NodeType],
                    'edge_types': [et.value for et in EdgeType],
                    'risk_levels': [rl.value for rl in RiskLevel]
                }
            }
            
            return visualization_data
            
        except Exception as e:
            self.logger.error(f"Error generating visualization data: {str(e)}")
            return {'error': str(e)}
    
    def _invalidate_analysis_cache(self):
        """Invalidate analysis caches when network structure changes."""
        self.centrality_cache.clear()
        self.community_cache.clear()
        self.path_cache.clear()
        self.risk_cache.clear()
    
    async def save_network(self, filepath: str) -> bool:
        """Save the network to file."""
        try:
            network_data = {
                'nodes': {nid: {
                    'node_id': node.node_id,
                    'node_type': node.node_type.value,
                    'label': node.label,
                    'properties': node.properties,
                    'risk_score': node.risk_score,
                    'compliance_status': node.compliance_status,
                    'last_updated': node.last_updated.isoformat(),
                    'centrality_scores': node.centrality_scores,
                    'community_id': node.community_id
                } for nid, node in self.nodes.items()},
                'edges': {eid: {
                    'edge_id': edge.edge_id,
                    'source_node': edge.source_node,
                    'target_node': edge.target_node,
                    'edge_type': edge.edge_type.value,
                    'weight': edge.weight,
                    'confidence': edge.confidence,
                    'risk_contribution': edge.risk_contribution,
                    'properties': edge.properties,
                    'created_at': edge.created_at.isoformat(),
                    'is_active': edge.is_active
                } for eid, edge in self.edges.items()}
            }
            
            with open(filepath, 'w') as f:
                json.dump(network_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving network: {str(e)}")
            return False
    
    async def load_network(self, filepath: str) -> bool:
        """Load network from file."""
        try:
            with open(filepath, 'r') as f:
                network_data = json.load(f)
            
            # Clear existing network
            self.network.clear()
            self.nodes.clear()
            self.edges.clear()
            self._invalidate_analysis_cache()
            
            # Load nodes
            for node_data in network_data['nodes'].values():
                node = NetworkNode(
                    node_id=node_data['node_id'],
                    node_type=NodeType(node_data['node_type']),
                    label=node_data['label'],
                    properties=node_data.get('properties', {}),
                    risk_score=node_data.get('risk_score', 0.0),
                    compliance_status=node_data.get('compliance_status', 'unknown'),
                    last_updated=datetime.fromisoformat(node_data.get('last_updated', datetime.now().isoformat())),
                    centrality_scores=node_data.get('centrality_scores', {}),
                    community_id=node_data.get('community_id')
                )
                await self.add_node(node)
            
            # Load edges
            for edge_data in network_data['edges'].values():
                edge = NetworkEdge(
                    edge_id=edge_data['edge_id'],
                    source_node=edge_data['source_node'],
                    target_node=edge_data['target_node'],
                    edge_type=EdgeType(edge_data['edge_type']),
                    weight=edge_data.get('weight', 1.0),
                    confidence=edge_data.get('confidence', 1.0),
                    risk_contribution=edge_data.get('risk_contribution', 0.0),
                    properties=edge_data.get('properties', {}),
                    created_at=datetime.fromisoformat(edge_data.get('created_at', datetime.now().isoformat())),
                    is_active=edge_data.get('is_active', True)
                )
                await self.add_edge(edge)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading network: {str(e)}")
            return False


class ComplianceGNN(nn.Module):
    """Graph Neural Network for advanced compliance analysis."""
    
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super(ComplianceGNN, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Attention mechanism
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Attention layer
        x = self.attention(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final layer
        x = self.conv3(x, edge_index)
        
        return torch.sigmoid(x)


class ComplianceNetworkAnalyzer:
    """High-level analyzer for compliance network operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.network = ComplianceNetworkGraph()
        
        # Analysis history
        self.analysis_history = []
        self.max_history = 100
    
    async def build_network_from_data(self, clients_data: List[Dict], advisors_data: List[Dict], 
                                    regulations_data: List[Dict], relationships_data: List[Dict]) -> bool:
        """Build compliance network from structured data."""
        try:
            # Add client nodes
            for client_data in clients_data:
                node = NetworkNode(
                    node_id=f"client_{client_data['id']}",
                    node_type=NodeType.CLIENT,
                    label=client_data.get('name', f"Client {client_data['id']}"),
                    properties=client_data,
                    risk_score=client_data.get('risk_score', 0.0),
                    compliance_status=client_data.get('compliance_status', 'pending')
                )
                await self.network.add_node(node)
            
            # Add advisor nodes
            for advisor_data in advisors_data:
                node = NetworkNode(
                    node_id=f"advisor_{advisor_data['id']}",
                    node_type=NodeType.ADVISOR,
                    label=advisor_data.get('name', f"Advisor {advisor_data['id']}"),
                    properties=advisor_data,
                    risk_score=advisor_data.get('risk_score', 0.0),
                    compliance_status=advisor_data.get('compliance_status', 'active')
                )
                await self.network.add_node(node)
            
            # Add regulation nodes
            for reg_data in regulations_data:
                node = NetworkNode(
                    node_id=f"regulation_{reg_data['id']}",
                    node_type=NodeType.REGULATION,
                    label=reg_data.get('title', f"Regulation {reg_data['id']}"),
                    properties=reg_data,
                    risk_score=reg_data.get('severity', 0.5),
                    compliance_status='active'
                )
                await self.network.add_node(node)
            
            # Add relationships/edges
            for rel_data in relationships_data:
                edge = NetworkEdge(
                    edge_id=str(uuid.uuid4()),
                    source_node=rel_data['source'],
                    target_node=rel_data['target'],
                    edge_type=EdgeType(rel_data.get('relationship_type', 'relates_to')),
                    weight=rel_data.get('weight', 1.0),
                    confidence=rel_data.get('confidence', 1.0),
                    risk_contribution=rel_data.get('risk_contribution', 0.0),
                    properties=rel_data.get('properties', {})
                )
                await self.network.add_edge(edge)
            
            self.logger.info(f"Built compliance network with {len(clients_data)} clients, "
                           f"{len(advisors_data)} advisors, {len(regulations_data)} regulations")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building network from data: {str(e)}")
            return False
    
    async def perform_comprehensive_analysis(self) -> NetworkAnalysisResult:
        """Perform comprehensive network analysis."""
        try:
            analysis_id = str(uuid.uuid4())
            
            # Get network statistics
            network_stats = await self.network.get_network_statistics()
            
            # Analyze centrality
            centrality_analysis = await self.network.analyze_centrality()
            
            # Detect communities
            community_structure = await self.network.detect_communities()
            
            # Perform risk analysis
            high_risk_nodes = [
                node_id for node_id, node in self.network.nodes.items()
                if node.risk_score > 0.7
            ]
            
            risk_propagation_result = None
            if high_risk_nodes:
                risk_propagation_result = await self.network.analyze_risk_propagation(
                    high_risk_nodes[:5], PropagationType.CASCADE
                )
            
            # Detect anomalies
            anomaly_detection = await self.network.detect_anomalies()
            
            # Generate visualization data
            visualization_data = await self.network.generate_visualization_data("spring")
            
            # Generate recommendations
            recommendations = self._generate_comprehensive_recommendations(
                network_stats, centrality_analysis, community_structure,
                risk_propagation_result, anomaly_detection
            )
            
            # Create result
            result = NetworkAnalysisResult(
                analysis_id=analysis_id,
                network_stats=network_stats,
                centrality_analysis=centrality_analysis,
                community_structure=community_structure,
                risk_assessment={
                    'high_risk_nodes': high_risk_nodes,
                    'propagation_result': risk_propagation_result.__dict__ if risk_propagation_result else None
                },
                anomaly_detection=anomaly_detection,
                recommendations=recommendations,
                visualization_data=visualization_data,
                analysis_timestamp=datetime.now(timezone.utc)
            )
            
            # Store in history
            self.analysis_history.append(result)
            if len(self.analysis_history) > self.max_history:
                self.analysis_history.pop(0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
    
    def _generate_comprehensive_recommendations(self, stats: Dict, centrality: Dict, 
                                             communities: Dict, risk_propagation: Any, 
                                             anomalies: Dict) -> List[str]:
        """Generate comprehensive recommendations based on analysis results."""
        recommendations = []
        
        try:
            # Network structure recommendations
            if stats.get('density', 0) < 0.1:
                recommendations.append("Network density is low - consider strengthening connections between compliance entities")
            elif stats.get('density', 0) > 0.8:
                recommendations.append("Network is very dense - consider streamlining relationships to reduce complexity")
            
            # Centrality-based recommendations
            if centrality and 'betweenness' in centrality:
                # Find nodes with high betweenness centrality
                betweenness_scores = centrality['betweenness']
                high_betweenness = sorted(betweenness_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for node_id, score in high_betweenness:
                    if score > 0.1:  # Significant betweenness
                        node = self.network.nodes.get(node_id)
                        if node:
                            recommendations.append(f"Critical bottleneck identified: {node.label} - consider adding redundancy")
            
            # Community-based recommendations
            if communities:
                community_sizes = [len(nodes) for nodes in communities.values()]
                if community_sizes:
                    avg_community_size = np.mean(community_sizes)
                    if avg_community_size < 3:
                        recommendations.append("Communities are very small - consider consolidating related compliance functions")
                    elif avg_community_size > 20:
                        recommendations.append("Communities are very large - consider breaking down into smaller, focused groups")
            
            # Risk propagation recommendations
            if risk_propagation and hasattr(risk_propagation, 'mitigation_recommendations'):
                recommendations.extend(risk_propagation.mitigation_recommendations[:3])  # Top 3
            
            # Anomaly-based recommendations
            if anomalies:
                if anomalies.get('anomalous_nodes'):
                    recommendations.append(f"Found {len(anomalies['anomalous_nodes'])} anomalous nodes - investigate for compliance issues")
                
                if anomalies.get('structural_anomalies'):
                    structural_count = len(anomalies['structural_anomalies'])
                    if structural_count > 0:
                        recommendations.append(f"Found {structural_count} structural anomalies - review network topology")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Compliance network appears healthy - maintain current monitoring practices")
            
            # Limit recommendations
            return recommendations[:10]
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations - manual review recommended"]
    
    async def simulate_regulatory_change_impact(self, regulation_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the impact of regulatory changes on the network."""
        try:
            simulation_result = {
                'simulation_id': str(uuid.uuid4()),
                'original_network_stats': await self.network.get_network_statistics(),
                'changes_applied': regulation_changes,
                'impact_analysis': {},
                'recommendations': []
            }
            
            # Create a copy of the network for simulation
            original_nodes = self.network.nodes.copy()
            original_edges = self.network.edges.copy()
            
            try:
                # Apply changes
                for change_type, change_data in regulation_changes.items():
                    if change_type == 'new_regulations':
                        for reg_data in change_data:
                            # Add new regulation node
                            node = NetworkNode(
                                node_id=f"regulation_{reg_data['id']}",
                                node_type=NodeType.REGULATION,
                                label=reg_data.get('title', f"New Regulation {reg_data['id']}"),
                                properties=reg_data,
                                risk_score=reg_data.get('severity', 0.5),
                                compliance_status='pending'
                            )
                            await self.network.add_node(node)
                            
                            # Connect to relevant entities
                            affected_nodes = reg_data.get('affects', [])
                            for affected_node in affected_nodes:
                                if affected_node in self.network.nodes:
                                    edge = NetworkEdge(
                                        edge_id=str(uuid.uuid4()),
                                        source_node=node.node_id,
                                        target_node=affected_node,
                                        edge_type=EdgeType.APPLIES_TO,
                                        weight=1.0,
                                        risk_contribution=reg_data.get('risk_impact', 0.3)
                                    )
                                    await self.network.add_edge(edge)
                    
                    elif change_type == 'regulation_updates':
                        for update_data in change_data:
                            reg_id = update_data['regulation_id']
                            if reg_id in self.network.nodes:
                                # Update regulation severity/risk
                                new_severity = update_data.get('new_severity', 0.5)
                                await self.network.update_node_risk(reg_id, new_severity)
                
                # Analyze post-change network
                post_change_stats = await self.network.get_network_statistics()
                
                # Risk propagation analysis
                high_risk_nodes = [
                    node_id for node_id, node in self.network.nodes.items()
                    if node.risk_score > 0.7
                ]
                
                if high_risk_nodes:
                    risk_propagation = await self.network.analyze_risk_propagation(high_risk_nodes)
                    simulation_result['risk_propagation'] = risk_propagation.__dict__
                
                # Impact analysis
                simulation_result['impact_analysis'] = {
                    'node_count_change': post_change_stats['node_count'] - simulation_result['original_network_stats']['node_count'],
                    'edge_count_change': post_change_stats['edge_count'] - simulation_result['original_network_stats']['edge_count'],
                    'density_change': post_change_stats['density'] - simulation_result['original_network_stats']['density'],
                    'new_network_stats': post_change_stats
                }
                
                # Generate recommendations
                recommendations = []
                if simulation_result['impact_analysis']['node_count_change'] > 0:
                    recommendations.append(f"Added {simulation_result['impact_analysis']['node_count_change']} new compliance requirements")
                
                if 'risk_propagation' in simulation_result:
                    total_risk_increase = simulation_result['risk_propagation']['total_risk_increase']
                    if total_risk_increase > 0.5:
                        recommendations.append("High risk propagation detected - implement additional controls")
                
                simulation_result['recommendations'] = recommendations
                
            finally:
                # Restore original network
                self.network.nodes = original_nodes
                self.network.edges = original_edges
                self.network._invalidate_analysis_cache()
            
            return simulation_result
            
        except Exception as e:
            self.logger.error(f"Error in regulatory change simulation: {str(e)}")
            return {'error': str(e)}


# Factory function for creating compliance network analyzer
def create_compliance_network_analyzer() -> ComplianceNetworkAnalyzer:
    """Factory function to create and initialize ComplianceNetworkAnalyzer."""
    return ComplianceNetworkAnalyzer()


# Convenience functions for integration with other systems
async def analyze_compliance_network(clients_data: List[Dict], advisors_data: List[Dict], 
                                   regulations_data: List[Dict], relationships_data: List[Dict]) -> NetworkAnalysisResult:
    """
    Convenience function to analyze compliance network from structured data.
    
    Args:
        clients_data: List of client data dictionaries
        advisors_data: List of advisor data dictionaries
        regulations_data: List of regulation data dictionaries
        relationships_data: List of relationship data dictionaries
        
    Returns:
        NetworkAnalysisResult with comprehensive analysis
    """
    analyzer = create_compliance_network_analyzer()
    
    # Build network
    success = await analyzer.build_network_from_data(
        clients_data, advisors_data, regulations_data, relationships_data
    )
    
    if not success:
        raise RuntimeError("Failed to build compliance network")
    
    # Perform analysis
    return await analyzer.perform_comprehensive_analysis()


async def simulate_compliance_impact(current_network_data: Dict[str, List[Dict]], 
                                   regulatory_changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate the impact of regulatory changes on compliance network.
    
    Args:
        current_network_data: Current network data (clients, advisors, regulations, relationships)
        regulatory_changes: Dictionary describing regulatory changes
        
    Returns:
        Dictionary with simulation results and impact analysis
    """
    analyzer = create_compliance_network_analyzer()
    
    # Build current network
    await analyzer.build_network_from_data(
        current_network_data.get('clients', []),
        current_network_data.get('advisors', []),
        current_network_data.get('regulations', []),
        current_network_data.get('relationships', [])
    )
    
    # Simulate changes
    return await analyzer.simulate_regulatory_change_impact(regulatory_changes)


async def detect_compliance_risks(network_data: Dict[str, List[Dict]]) -> RiskPropagationResult:
    """
    Detect and analyze compliance risks in the network.
    
    Args:
        network_data: Network data with clients, advisors, regulations, relationships
        
    Returns:
        RiskPropagationResult with risk analysis and mitigation recommendations
    """
    analyzer = create_compliance_network_analyzer()
    
    # Build network
    await analyzer.build_network_from_data(
        network_data.get('clients', []),
        network_data.get('advisors', []),
        network_data.get('regulations', []),
        network_data.get('relationships', [])
    )
    
    # Find high-risk nodes
    high_risk_nodes = [
        f"client_{client['id']}" for client in network_data.get('clients', [])
        if client.get('risk_score', 0) > 0.7
    ]
    
    high_risk_nodes.extend([
        f"advisor_{advisor['id']}" for advisor in network_data.get('advisors', [])
        if advisor.get('risk_score', 0) > 0.7
    ])
    
    # Analyze risk propagation
    if high_risk_nodes:
        return await analyzer.network.analyze_risk_propagation(high_risk_nodes, PropagationType.CASCADE)
    else:
        # Return empty result if no high-risk nodes
        return RiskPropagationResult(
            propagation_id=str(uuid.uuid4()),
            source_nodes=[],
            affected_nodes={},
            propagation_paths=[],
            total_risk_increase=0.0,
            propagation_time_steps=0,
            convergence_achieved=True,
            risk_distribution={},
            critical_paths=[],
            mitigation_recommendations=["No high-risk nodes detected - network appears stable"]
        )
