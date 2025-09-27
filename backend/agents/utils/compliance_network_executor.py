#!/usr/bin/env python3
"""
Compliance Network Graph Executor

This script serves as the execution interface for the compliance network graph analysis system.
It handles requests from the Node.js API and coordinates with the ComplianceNetworkGraph classes
to perform various network analysis operations.

Usage:
    python3 compliance_network_executor.py <operation> [options]

Operations:
    analyze_network - Build and analyze compliance network
    analyze_risk_propagation - Perform risk propagation analysis
    simulate_regulatory_impact - Simulate regulatory change impact
    network_statistics - Calculate network statistics
    detect_anomalies - Detect network anomalies
    generate_visualization - Generate visualization data
    find_shortest_paths - Find shortest paths between nodes
    health_check - Service health check

Input/Output:
    - Input data is provided via stdin as JSON
    - Output results are returned via stdout as JSON
    - Errors are logged to stderr
"""

import sys
import json
import logging
import asyncio
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import uuid
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import compliance network components
from compliance_network_graph import (
    ComplianceNetworkAnalyzer,
    NetworkNode, NetworkEdge, NodeType, EdgeType, PropagationType,
    analyze_compliance_network, simulate_compliance_impact, detect_compliance_risks
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('/app/logs/compliance_network.log', mode='a') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


class ComplianceNetworkExecutor:
    """Main executor class for compliance network operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzer = None
    
    async def initialize_analyzer(self):
        """Initialize the compliance network analyzer."""
        if not self.analyzer:
            self.analyzer = ComplianceNetworkAnalyzer()
        return self.analyzer
    
    def validate_input_data(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate input data contains required fields."""
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        return True
    
    def convert_node_type(self, node_type_str: str) -> NodeType:
        """Convert string to NodeType enum."""
        try:
            return NodeType(node_type_str.lower())
        except ValueError:
            self.logger.warning(f"Unknown node type: {node_type_str}, using CLIENT as default")
            return NodeType.CLIENT
    
    def convert_edge_type(self, edge_type_str: str) -> EdgeType:
        """Convert string to EdgeType enum."""
        try:
            return EdgeType(edge_type_str.lower())
        except ValueError:
            self.logger.warning(f"Unknown edge type: {edge_type_str}, using RELATES_TO as default")
            return EdgeType.ADVISES  # Default edge type
    
    def convert_propagation_type(self, prop_type_str: str) -> PropagationType:
        """Convert string to PropagationType enum."""
        try:
            return PropagationType(prop_type_str.lower())
        except ValueError:
            self.logger.warning(f"Unknown propagation type: {prop_type_str}, using LINEAR as default")
            return PropagationType.LINEAR
    
    async def analyze_network(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build compliance network from data and perform comprehensive analysis."""
        try:
            self.validate_input_data(input_data, ['clients', 'advisors', 'regulations', 'relationships'])
            
            clients_data = input_data['clients']
            advisors_data = input_data['advisors']
            regulations_data = input_data['regulations']
            relationships_data = input_data['relationships']
            analysis_options = input_data.get('analysis_options', {})
            
            # Initialize analyzer
            analyzer = await self.initialize_analyzer()
            
            # Build network from data
            success = await analyzer.build_network_from_data(
                clients_data, advisors_data, regulations_data, relationships_data
            )
            
            if not success:
                raise RuntimeError("Failed to build compliance network from data")
            
            # Perform comprehensive analysis
            analysis_result = await analyzer.perform_comprehensive_analysis()
            
            return {
                'success': True,
                'analysis_id': analysis_result.analysis_id,
                'network_stats': analysis_result.network_stats,
                'centrality_analysis': analysis_result.centrality_analysis,
                'community_structure': analysis_result.community_structure,
                'risk_assessment': analysis_result.risk_assessment,
                'anomaly_detection': analysis_result.anomaly_detection,
                'recommendations': analysis_result.recommendations,
                'visualization_data': analysis_result.visualization_data,
                'analysis_timestamp': analysis_result.analysis_timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_network: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def analyze_risk_propagation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk propagation analysis."""
        try:
            self.validate_input_data(input_data, ['network_data', 'propagation_request'])
            
            network_data = input_data['network_data']
            propagation_request = input_data['propagation_request']
            
            # Initialize analyzer and build network
            analyzer = await self.initialize_analyzer()
            
            success = await analyzer.build_network_from_data(
                network_data.get('clients', []),
                network_data.get('advisors', []),
                network_data.get('regulations', []),
                network_data.get('relationships', [])
            )
            
            if not success:
                raise RuntimeError("Failed to build network for risk propagation analysis")
            
            # Extract propagation parameters
            source_nodes = propagation_request['source_nodes']
            propagation_type = self.convert_propagation_type(
                propagation_request.get('propagation_type', 'linear')
            )
            
            # Set configuration
            if 'max_steps' in propagation_request:
                analyzer.network.max_propagation_steps = propagation_request['max_steps']
            if 'convergence_tolerance' in propagation_request:
                analyzer.network.convergence_tolerance = propagation_request['convergence_tolerance']
            
            # Perform risk propagation analysis
            propagation_result = await analyzer.network.analyze_risk_propagation(
                source_nodes, propagation_type
            )
            
            return {
                'success': True,
                'propagation_id': propagation_result.propagation_id,
                'source_nodes': propagation_result.source_nodes,
                'affected_nodes': propagation_result.affected_nodes,
                'propagation_paths': propagation_result.propagation_paths,
                'total_risk_increase': propagation_result.total_risk_increase,
                'propagation_time_steps': propagation_result.propagation_time_steps,
                'convergence_achieved': propagation_result.convergence_achieved,
                'critical_paths': propagation_result.critical_paths,
                'mitigation_recommendations': propagation_result.mitigation_recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_risk_propagation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def simulate_regulatory_impact(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate regulatory change impact."""
        try:
            self.validate_input_data(input_data, ['current_network_data', 'regulatory_changes'])
            
            current_network_data = input_data['current_network_data']
            regulatory_changes = input_data['regulatory_changes']
            
            # Initialize analyzer and build current network
            analyzer = await self.initialize_analyzer()
            
            success = await analyzer.build_network_from_data(
                current_network_data.get('clients', []),
                current_network_data.get('advisors', []),
                current_network_data.get('regulations', []),
                current_network_data.get('relationships', [])
            )
            
            if not success:
                raise RuntimeError("Failed to build current network for simulation")
            
            # Perform impact simulation
            simulation_result = await analyzer.simulate_regulatory_change_impact(regulatory_changes)
            
            return {
                'success': True,
                'simulation_id': simulation_result['simulation_id'],
                'original_network_stats': simulation_result['original_network_stats'],
                'changes_applied': simulation_result['changes_applied'],
                'impact_analysis': simulation_result['impact_analysis'],
                'recommendations': simulation_result['recommendations']
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulate_regulatory_impact: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def network_statistics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive network statistics."""
        try:
            self.validate_input_data(input_data, ['network_data'])
            
            network_data = input_data['network_data']
            
            # Initialize analyzer and build network
            analyzer = await self.initialize_analyzer()
            
            success = await analyzer.build_network_from_data(
                network_data.get('clients', []),
                network_data.get('advisors', []),
                network_data.get('regulations', []),
                network_data.get('relationships', [])
            )
            
            if not success:
                raise RuntimeError("Failed to build network for statistics calculation")
            
            # Calculate statistics
            statistics = await analyzer.network.get_network_statistics()
            
            return {
                'success': True,
                'statistics': statistics
            }
            
        except Exception as e:
            self.logger.error(f"Error in network_statistics: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def detect_anomalies(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in the network."""
        try:
            self.validate_input_data(input_data, ['network_data'])
            
            network_data = input_data['network_data']
            
            # Initialize analyzer and build network
            analyzer = await self.initialize_analyzer()
            
            success = await analyzer.build_network_from_data(
                network_data.get('clients', []),
                network_data.get('advisors', []),
                network_data.get('regulations', []),
                network_data.get('relationships', [])
            )
            
            if not success:
                raise RuntimeError("Failed to build network for anomaly detection")
            
            # Detect anomalies
            anomalies = await analyzer.network.detect_anomalies()
            
            return {
                'success': True,
                'anomalous_nodes': anomalies.get('anomalous_nodes', []),
                'anomalous_edges': anomalies.get('anomalous_edges', []),
                'structural_anomalies': anomalies.get('structural_anomalies', []),
                'temporal_anomalies': anomalies.get('temporal_anomalies', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error in detect_anomalies: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def generate_visualization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization data for the network."""
        try:
            self.validate_input_data(input_data, ['network_data'])
            
            network_data = input_data['network_data']
            layout_algorithm = input_data.get('layout_algorithm', 'spring')
            include_centrality = input_data.get('include_centrality', True)
            include_communities = input_data.get('include_communities', True)
            filter_options = input_data.get('filter_options', {})
            
            # Initialize analyzer and build network
            analyzer = await self.initialize_analyzer()
            
            success = await analyzer.build_network_from_data(
                network_data.get('clients', []),
                network_data.get('advisors', []),
                network_data.get('regulations', []),
                network_data.get('relationships', [])
            )
            
            if not success:
                raise RuntimeError("Failed to build network for visualization")
            
            # Optional: perform centrality and community analysis first
            if include_centrality:
                await analyzer.network.analyze_centrality()
            
            if include_communities:
                await analyzer.network.detect_communities()
            
            # Generate visualization data
            visualization_data = await analyzer.network.generate_visualization_data(layout_algorithm)
            
            # Apply filters if provided
            if filter_options:
                visualization_data = self._apply_visualization_filters(visualization_data, filter_options)
            
            return {
                'success': True,
                'visualization_data': visualization_data
            }
            
        except Exception as e:
            self.logger.error(f"Error in generate_visualization: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _apply_visualization_filters(self, viz_data: Dict[str, Any], filter_options: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to visualization data."""
        try:
            filtered_nodes = []
            filtered_edges = []
            
            # Filter nodes
            min_risk = filter_options.get('min_risk_score', 0.0)
            allowed_node_types = filter_options.get('node_types', [])
            
            for node in viz_data.get('nodes', []):
                # Risk score filter
                if node.get('risk_score', 0) < min_risk:
                    continue
                
                # Node type filter
                if allowed_node_types and node.get('type') not in allowed_node_types:
                    continue
                
                filtered_nodes.append(node)
            
            # Get node IDs for edge filtering
            node_ids = {node['id'] for node in filtered_nodes}
            
            # Filter edges
            allowed_edge_types = filter_options.get('edge_types', [])
            
            for edge in viz_data.get('edges', []):
                # Check if both nodes exist after filtering
                if edge['source'] not in node_ids or edge['target'] not in node_ids:
                    continue
                
                # Edge type filter
                if allowed_edge_types and edge.get('type') not in allowed_edge_types:
                    continue
                
                filtered_edges.append(edge)
            
            # Update visualization data
            viz_data['nodes'] = filtered_nodes
            viz_data['edges'] = filtered_edges
            
            # Update statistics
            if 'statistics' in viz_data:
                viz_data['statistics']['filtered_node_count'] = len(filtered_nodes)
                viz_data['statistics']['filtered_edge_count'] = len(filtered_edges)
            
            return viz_data
            
        except Exception as e:
            self.logger.error(f"Error applying visualization filters: {str(e)}")
            return viz_data  # Return unfiltered data on error
    
    async def find_shortest_paths(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find shortest paths between nodes."""
        try:
            self.validate_input_data(input_data, ['network_data', 'source_nodes', 'target_nodes'])
            
            network_data = input_data['network_data']
            source_nodes = input_data['source_nodes']
            target_nodes = input_data['target_nodes']
            max_path_length = input_data.get('max_path_length', 10)
            weight_attribute = input_data.get('weight_attribute', 'weight')
            
            # Initialize analyzer and build network
            analyzer = await self.initialize_analyzer()
            
            success = await analyzer.build_network_from_data(
                network_data.get('clients', []),
                network_data.get('advisors', []),
                network_data.get('regulations', []),
                network_data.get('relationships', [])
            )
            
            if not success:
                raise RuntimeError("Failed to build network for path finding")
            
            # Find paths
            import networkx as nx
            paths = []
            
            for source in source_nodes:
                for target in target_nodes:
                    if source != target:
                        try:
                            # Find shortest path
                            if weight_attribute in ['weight']:
                                path = nx.shortest_path(
                                    analyzer.network.network, source, target, weight=weight_attribute
                                )
                                path_length = nx.shortest_path_length(
                                    analyzer.network.network, source, target, weight=weight_attribute
                                )
                            else:
                                path = nx.shortest_path(analyzer.network.network, source, target)
                                path_length = len(path) - 1
                            
                            if len(path) <= max_path_length + 1:  # +1 because path includes both endpoints
                                paths.append({
                                    'source': source,
                                    'target': target,
                                    'path': path,
                                    'length': path_length,
                                    'node_count': len(path)
                                })
                        
                        except nx.NetworkXNoPath:
                            # No path exists
                            paths.append({
                                'source': source,
                                'target': target,
                                'path': None,
                                'length': float('inf'),
                                'node_count': 0,
                                'no_path': True
                            })
            
            # Calculate path statistics
            valid_paths = [p for p in paths if p.get('path') is not None]
            path_statistics = {
                'total_path_requests': len(paths),
                'valid_paths_found': len(valid_paths),
                'no_path_count': len(paths) - len(valid_paths),
                'average_path_length': sum(p['length'] for p in valid_paths) / len(valid_paths) if valid_paths else 0,
                'shortest_path_length': min(p['length'] for p in valid_paths) if valid_paths else None,
                'longest_path_length': max(p['length'] for p in valid_paths) if valid_paths else None
            }
            
            return {
                'success': True,
                'paths': paths,
                'path_statistics': path_statistics
            }
            
        except Exception as e:
            self.logger.error(f"Error in find_shortest_paths: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def health_check(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic functionality
            test_analyzer = ComplianceNetworkAnalyzer()
            
            # Test network creation
            from compliance_network_graph import NetworkNode, NodeType
            test_node = NetworkNode(
                node_id="test_node",
                node_type=NodeType.CLIENT,
                label="Test Node"
            )
            await test_analyzer.network.add_node(test_node)
            
            # Test statistics calculation
            stats = await test_analyzer.network.get_network_statistics()
            
            return {
                'success': True,
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'test_results': {
                    'network_creation': True,
                    'statistics_calculation': True,
                    'node_count': stats.get('node_count', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'success': False,
                'status': 'unhealthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }


async def main():
    """Main execution function."""
    try:
        if len(sys.argv) < 2:
            print(json.dumps({
                'success': False,
                'error': 'No operation specified',
                'usage': 'python3 compliance_network_executor.py <operation>'
            }))
            sys.exit(1)
        
        operation = sys.argv[1].lower()
        
        # Read input data from stdin if available
        input_data = {}
        if not sys.stdin.isatty():
            try:
                input_data = json.load(sys.stdin)
            except json.JSONDecodeError as e:
                print(json.dumps({
                    'success': False,
                    'error': f'Invalid JSON input: {str(e)}'
                }))
                sys.exit(1)
        
        # Create executor
        executor = ComplianceNetworkExecutor()
        
        # Route to appropriate operation
        operations = {
            'analyze_network': executor.analyze_network,
            'analyze_risk_propagation': executor.analyze_risk_propagation,
            'simulate_regulatory_impact': executor.simulate_regulatory_impact,
            'network_statistics': executor.network_statistics,
            'detect_anomalies': executor.detect_anomalies,
            'generate_visualization': executor.generate_visualization,
            'find_shortest_paths': executor.find_shortest_paths,
            'health_check': executor.health_check
        }
        
        if operation not in operations:
            print(json.dumps({
                'success': False,
                'error': f'Unknown operation: {operation}',
                'available_operations': list(operations.keys())
            }))
            sys.exit(1)
        
        # Execute operation
        result = await operations[operation](input_data)
        
        # Output result
        print(json.dumps(result, indent=2))
        
        # Exit with appropriate code
        sys.exit(0 if result.get('success', False) else 1)
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == '__main__':
    # Run async main
    asyncio.run(main())
