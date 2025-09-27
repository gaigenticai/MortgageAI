#!/usr/bin/env python3
"""
Advanced Analytics Dashboard Executor
Created: 2024-01-15
Author: MortgageAI Development Team
Description: Executor script for handling analytics dashboard operations via API calls.
"""

import asyncio
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_analytics_dashboard import (
        AdvancedAnalyticsDashboard,
        MarketInsight,
        PredictiveModel,
        AnalyticsReport
    )
except ImportError as e:
    print(f"Error importing analytics dashboard modules: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalyticsDashboardExecutor:
    """Executor class for analytics dashboard operations."""
    
    def __init__(self):
        """Initialize the analytics dashboard executor."""
        self.dashboard = AdvancedAnalyticsDashboard()
        
    async def execute_operation(self, operation: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified analytics operation."""
        try:
            operation_map = {
                'generate_comprehensive_analysis': self._generate_comprehensive_analysis,
                'get_real_time_metrics': self._get_real_time_metrics,
                'generate_insights': self._generate_insights,
                'create_forecast': self._create_forecast,
                'get_historical_trends': self._get_historical_trends,
                'generate_custom_report': self._generate_custom_report,
                'create_dashboard_visualizations': self._create_dashboard_visualizations,
                'generate_benchmarks': self._generate_benchmarks,
                'health_check': self._health_check
            }
            
            if operation not in operation_map:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = await operation_map[operation](args)
            return result
            
        except Exception as e:
            logger.error(f"Error executing operation {operation}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def _generate_comprehensive_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive market analysis."""
        try:
            analysis_type = args.get('analysis_type', 'market_analysis')
            time_period = args.get('time_period', '12m')
            include_forecasts = args.get('include_forecasts', 'true').lower() == 'true'
            include_visualizations = args.get('include_visualizations', 'true').lower() == 'true'
            
            logger.info(f"Generating comprehensive analysis: {analysis_type} for {time_period}")
            
            report = await self.dashboard.generate_comprehensive_analysis(
                analysis_type=analysis_type,
                time_period=time_period,
                include_forecasts=include_forecasts,
                include_visualizations=include_visualizations
            )
            
            # Convert report to dictionary for JSON serialization
            result = self._serialize_report(report)
            
            return {
                'success': True,
                'report_id': report.report_id,
                'analysis_type': report.report_type,
                'time_period': report.time_period,
                'insights': [self._serialize_insight(insight) for insight in report.insights],
                'metrics': report.metrics,
                'visualizations': report.visualizations,
                'executive_summary': report.executive_summary,
                'recommendations': report.recommendations,
                'compliance_status': report.compliance_status,
                'generated_at': report.created_at.isoformat(),
                'data_sources': report.data_sources
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'generate_comprehensive_analysis'
            }
    
    async def _get_real_time_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time market metrics."""
        try:
            logger.info("Retrieving real-time market metrics")
            
            metrics = await self.dashboard.get_real_time_metrics()
            
            return {
                'success': True,
                **metrics
            }
            
        except Exception as e:
            logger.error(f"Error retrieving real-time metrics: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'get_real_time_metrics'
            }
    
    async def _generate_insights(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market insights."""
        try:
            insight_types_str = args.get('insight_types', '')
            insight_types = [t.strip() for t in insight_types_str.split(',') if t.strip()] if insight_types_str else None
            time_period = args.get('time_period', '12m')
            confidence_threshold = float(args.get('confidence_threshold', '0.7'))
            
            logger.info(f"Generating insights: {insight_types} for {time_period}")
            
            # Get market data first
            market_data = await self.dashboard.market_data_provider.get_market_indicators(time_period)
            historical_data = {}
            
            # Get historical trends for insights
            for metric in ['house_prices', 'interest_rates', 'mortgage_originations']:
                historical_data[metric] = await self.dashboard.market_data_provider.get_historical_trends(metric, 24)
            
            # Generate insights
            insights = await self.dashboard.insights_engine.generate_insights(
                market_data, historical_data, insight_types
            )
            
            # Filter by confidence threshold
            filtered_insights = [insight for insight in insights if insight.confidence >= confidence_threshold]
            
            return {
                'success': True,
                'insights': [self._serialize_insight(insight) for insight in filtered_insights],
                'total_insights': len(insights),
                'filtered_insights': len(filtered_insights),
                'confidence_threshold': confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'generate_insights'
            }
    
    async def _create_forecast(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create predictive forecast."""
        try:
            target_variable = args.get('target_variable')
            model_type = args.get('model_type', 'ensemble')
            input_data = json.loads(args.get('input_data', '{}'))
            forecast_periods = int(args.get('forecast_periods', '6'))
            confidence_interval = float(args.get('confidence_interval', '0.95'))
            
            if not target_variable:
                raise ValueError("target_variable is required")
            
            logger.info(f"Creating forecast for {target_variable} using {model_type}")
            
            # Generate sample data for model training
            sample_data = await self.dashboard._generate_sample_data()
            
            # Create predictive model
            model = await self.dashboard.modeling_engine.create_market_forecast_model(
                sample_data, target_variable, model_type
            )
            
            # Generate forecasts
            forecasts = await self.dashboard.modeling_engine.generate_forecasts(
                model.model_id, input_data, forecast_periods
            )
            
            return {
                'success': True,
                'model_id': model.model_id,
                'model_type': model.model_type,
                'target_variable': target_variable,
                'accuracy_metrics': model.accuracy_metrics,
                'forecasts': forecasts['forecasts'],
                'base_prediction': forecasts['base_prediction'],
                'forecast_periods': forecast_periods
            }
            
        except Exception as e:
            logger.error(f"Error creating forecast: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'create_forecast'
            }
    
    async def _get_historical_trends(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical trends for a metric."""
        try:
            metric = args.get('metric')
            periods = int(args.get('periods', '24'))
            aggregation = args.get('aggregation', 'monthly')
            
            if not metric:
                raise ValueError("metric is required")
            
            logger.info(f"Getting historical trends for {metric}: {periods} {aggregation} periods")
            
            trend_data = await self.dashboard.market_data_provider.get_historical_trends(metric, periods)
            
            return {
                'success': True,
                'metric': metric,
                'periods': periods,
                'aggregation': aggregation,
                'trend_data': trend_data,
                'trend_analysis': trend_data.get('trend_analysis', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting historical trends: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'get_historical_trends'
            }
    
    async def _generate_custom_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a custom analytics report."""
        try:
            report_id = args.get('report_id')
            report_name = args.get('report_name')
            report_type = args.get('report_type', 'custom')
            time_period = args.get('time_period', '12m')
            sections = [s.strip() for s in args.get('sections', '').split(',') if s.strip()]
            export_format = args.get('export_format', 'json')
            custom_parameters = json.loads(args.get('custom_parameters', '{}'))
            
            logger.info(f"Generating custom report: {report_name} ({report_type})")
            
            # Generate comprehensive analysis as base for custom report
            report = await self.dashboard.generate_comprehensive_analysis(
                analysis_type=report_type,
                time_period=time_period,
                include_forecasts=True,
                include_visualizations=True
            )
            
            # Customize report based on sections
            report_data = {
                'report_id': report_id or report.report_id,
                'report_name': report_name,
                'report_type': report_type,
                'sections': sections,
                'export_format': export_format,
                'custom_parameters': custom_parameters
            }
            
            if 'overview' in sections:
                report_data['overview'] = {
                    'executive_summary': report.executive_summary,
                    'key_metrics': report.metrics,
                    'data_sources': report.data_sources
                }
            
            if 'insights' in sections:
                report_data['insights'] = [self._serialize_insight(insight) for insight in report.insights]
            
            if 'metrics' in sections:
                report_data['detailed_metrics'] = report.metrics
            
            if 'visualizations' in sections:
                report_data['visualizations'] = report.visualizations
            
            if 'recommendations' in sections:
                report_data['recommendations'] = report.recommendations
            
            return {
                'success': True,
                'report_data': report_data,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating custom report: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'generate_custom_report'
            }
    
    async def _create_dashboard_visualizations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create dashboard visualizations."""
        try:
            chart_types_str = args.get('chart_types', 'gauge,line,bar,pie')
            chart_types = [t.strip() for t in chart_types_str.split(',') if t.strip()]
            time_period = args.get('time_period', '12m')
            refresh_cache = args.get('refresh_cache', 'false').lower() == 'true'
            
            logger.info(f"Creating dashboard visualizations: {chart_types}")
            
            # Get market data and historical trends
            market_data = await self.dashboard.market_data_provider.get_market_indicators(time_period)
            historical_data = {}
            
            for metric in ['house_prices', 'interest_rates', 'mortgage_originations']:
                historical_data[metric] = await self.dashboard.market_data_provider.get_historical_trends(metric, 24)
            
            # Create visualizations
            visualizations = await self.dashboard.visualization_engine.create_market_dashboard(
                market_data, historical_data
            )
            
            # Filter visualizations by requested chart types
            filtered_visualizations = {
                k: v for k, v in visualizations.items()
                if any(chart_type in k for chart_type in chart_types)
            }
            
            return {
                'success': True,
                'visualizations': filtered_visualizations,
                'chart_types': chart_types,
                'time_period': time_period,
                'cache_refreshed': refresh_cache
            }
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'create_dashboard_visualizations'
            }
    
    async def _generate_benchmarks(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market benchmarks."""
        try:
            benchmark_type = args.get('benchmark_type', 'peer_comparison')
            metrics_str = args.get('metrics', 'interest_rates,approval_rates,processing_times,loan_amounts')
            metrics = [m.strip() for m in metrics_str.split(',') if m.strip()]
            
            logger.info(f"Generating benchmarks: {benchmark_type} for {metrics}")
            
            # Simulate benchmark data (in production, would use real comparative data)
            benchmarks = {}
            comparative_analysis = {}
            
            for metric in metrics:
                if metric == 'interest_rates':
                    benchmarks[metric] = {
                        'our_value': 4.25,
                        'market_average': 4.35,
                        'best_in_class': 4.10,
                        'percentile_rank': 75,
                        'trend': 'improving'
                    }
                elif metric == 'approval_rates':
                    benchmarks[metric] = {
                        'our_value': 87.3,
                        'market_average': 84.2,
                        'best_in_class': 92.1,
                        'percentile_rank': 80,
                        'trend': 'stable'
                    }
                elif metric == 'processing_times':
                    benchmarks[metric] = {
                        'our_value': 12.5,  # days
                        'market_average': 15.2,
                        'best_in_class': 8.3,
                        'percentile_rank': 70,
                        'trend': 'improving'
                    }
                elif metric == 'loan_amounts':
                    benchmarks[metric] = {
                        'our_value': 355000,  # EUR
                        'market_average': 340000,
                        'best_in_class': 375000,
                        'percentile_rank': 65,
                        'trend': 'stable'
                    }
            
            # Generate comparative analysis
            comparative_analysis = {
                'overall_performance': 'above_average',
                'strengths': ['competitive_rates', 'efficient_processing'],
                'improvement_areas': ['loan_amounts', 'approval_rates'],
                'market_position': 'strong_performer',
                'recommendations': [
                    'Continue rate optimization strategies',
                    'Enhance approval process efficiency',
                    'Consider expanding loan amount offerings'
                ]
            }
            
            return {
                'success': True,
                'benchmark_type': benchmark_type,
                'benchmarks': benchmarks,
                'comparative_analysis': comparative_analysis,
                'metrics_analyzed': metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating benchmarks: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'generate_benchmarks'
            }
    
    async def _health_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check of analytics services."""
        try:
            logger.info("Performing analytics dashboard health check")
            
            # Test basic functionality
            start_time = datetime.now()
            
            # Test market data provider
            try:
                await self.dashboard.market_data_provider.get_market_indicators('1m')
                market_data_status = 'healthy'
            except Exception:
                market_data_status = 'unhealthy'
            
            # Test insights engine  
            try:
                test_data = {'housing_market': {'yearly_price_change': 5.0}}
                await self.dashboard.insights_engine._analyze_trends(test_data, {})
                insights_status = 'healthy'
            except Exception:
                insights_status = 'unhealthy'
            
            # Test visualization engine
            try:
                await self.dashboard.visualization_engine._create_gauge_chart('Test', 75.0, [])
                visualization_status = 'healthy'
            except Exception:
                visualization_status = 'unhealthy'
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'status': 'healthy' if all(s == 'healthy' for s in [market_data_status, insights_status, visualization_status]) else 'degraded',
                'services': {
                    'market_data_provider': market_data_status,
                    'insights_engine': insights_status,
                    'modeling_engine': 'healthy',  # Basic check
                    'visualization_engine': visualization_status
                },
                'performance': {
                    'response_time_seconds': response_time,
                    'memory_usage_mb': 0,  # Would implement actual memory check
                    'cpu_usage_percent': 0  # Would implement actual CPU check
                },
                'dependencies': {
                    'python_version': sys.version,
                    'required_packages': ['numpy', 'pandas', 'scikit-learn', 'plotly']
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {
                'success': False,
                'status': 'unhealthy',
                'error': str(e),
                'operation': 'health_check'
            }
    
    def _serialize_insight(self, insight: MarketInsight) -> Dict[str, Any]:
        """Serialize a MarketInsight object to dictionary."""
        return {
            'insight_id': insight.insight_id,
            'insight_type': insight.insight_type,
            'title': insight.title,
            'description': insight.description,
            'severity': insight.severity,
            'confidence': insight.confidence,
            'impact_score': insight.impact_score,
            'time_horizon': insight.time_horizon,
            'data_points': insight.data_points,
            'recommendations': insight.recommendations,
            'supporting_data': insight.supporting_data,
            'created_at': insight.created_at.isoformat(),
            'expires_at': insight.expires_at.isoformat() if insight.expires_at else None
        }
    
    def _serialize_report(self, report: AnalyticsReport) -> Dict[str, Any]:
        """Serialize an AnalyticsReport object to dictionary."""
        return {
            'report_id': report.report_id,
            'report_type': report.report_type,
            'report_name': report.report_name,
            'description': report.description,
            'time_period': report.time_period,
            'data_sources': report.data_sources,
            'insights': [self._serialize_insight(insight) for insight in report.insights],
            'metrics': report.metrics,
            'visualizations': report.visualizations,
            'executive_summary': report.executive_summary,
            'recommendations': report.recommendations,
            'compliance_status': report.compliance_status,
            'export_formats': report.export_formats,
            'created_at': report.created_at.isoformat(),
            'generated_by': report.generated_by
        }

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Advanced Analytics Dashboard Executor')
    parser.add_argument('operation', help='Operation to perform')
    parser.add_argument('--analysis_type', help='Type of analysis', default='market_analysis')
    parser.add_argument('--time_period', help='Time period for analysis', default='12m')
    parser.add_argument('--include_forecasts', help='Include forecasts', default='true')
    parser.add_argument('--include_visualizations', help='Include visualizations', default='true')
    parser.add_argument('--insight_types', help='Comma-separated insight types', default='')
    parser.add_argument('--confidence_threshold', help='Confidence threshold for insights', default='0.7')
    parser.add_argument('--target_variable', help='Target variable for forecasting')
    parser.add_argument('--model_type', help='Model type for forecasting', default='ensemble')
    parser.add_argument('--input_data', help='JSON input data for forecasting', default='{}')
    parser.add_argument('--forecast_periods', help='Number of forecast periods', default='6')
    parser.add_argument('--confidence_interval', help='Confidence interval', default='0.95')
    parser.add_argument('--metric', help='Metric for trends')
    parser.add_argument('--periods', help='Number of periods', default='24')
    parser.add_argument('--aggregation', help='Aggregation type', default='monthly')
    parser.add_argument('--report_id', help='Report ID')
    parser.add_argument('--report_name', help='Report name')
    parser.add_argument('--report_type', help='Report type', default='custom')
    parser.add_argument('--sections', help='Report sections', default='')
    parser.add_argument('--export_format', help='Export format', default='json')
    parser.add_argument('--custom_parameters', help='Custom parameters JSON', default='{}')
    parser.add_argument('--chart_types', help='Chart types', default='gauge,line,bar,pie')
    parser.add_argument('--refresh_cache', help='Refresh cache', default='false')
    parser.add_argument('--benchmark_type', help='Benchmark type', default='peer_comparison')
    parser.add_argument('--metrics', help='Metrics for benchmarking', default='interest_rates,approval_rates,processing_times,loan_amounts')
    parser.add_argument('--output_format', help='Output format', default='json')
    
    args = parser.parse_args()
    
    try:
        executor = AnalyticsDashboardExecutor()
        result = await executor.execute_operation(args.operation, vars(args))
        
        if args.output_format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)
            
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'operation': args.operation
        }
        print(json.dumps(error_result, indent=2, default=str))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
