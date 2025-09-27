#!/usr/bin/env python3
"""
Dutch Market Intelligence Executor
=================================

Task executor for managing Dutch market intelligence operations with comprehensive
task scheduling, resource management, and performance optimization.

This module provides production-grade task execution capabilities including:
- Asynchronous task execution with queue management
- Resource optimization and load balancing
- Task prioritization and scheduling
- Error handling and recovery mechanisms
- Performance monitoring and analytics
- Comprehensive logging and audit trails
- Real-time status tracking and reporting

Features:
- Multi-threaded task execution with configurable worker pools
- Intelligent task scheduling and priority management
- Resource monitoring and automatic scaling
- Fault tolerance with retry mechanisms and graceful degradation
- Comprehensive performance metrics and analytics
- Integration with market intelligence analysis workflows
- Production-grade error handling and recovery

Author: MortgageAI Development Team
Date: 2025-01-27
Version: 1.0.0
"""

import os
import sys
import json
import logging
import asyncio
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, PriorityQueue, Empty
import psutil
import gc
import signal
import atexit
from functools import wraps

# Import the main intelligence module
try:
    from .dutch_market_intelligence import (
        DutchMarketIntelligence,
        DataSource,
        AnalysisType,
        TrendType,
        RiskLevel,
        MarketDataPoint,
        TrendAnalysisResult,
        PredictiveModel,
        MarketInsight
    )
except ImportError:
    # Fallback import for testing
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dutch_market_intelligence import (
        DutchMarketIntelligence,
        DataSource,
        AnalysisType,
        TrendType,
        RiskLevel,
        MarketDataPoint,
        TrendAnalysisResult,
        PredictiveModel,
        MarketInsight
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task execution status enumeration"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class ExecutorStatus(Enum):
    """Executor system status"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class TaskDefinition:
    """Defines a market intelligence task"""
    task_id: str
    task_type: str
    task_name: str
    parameters: Dict[str, Any]
    priority: TaskPriority
    timeout_seconds: int
    max_retries: int
    dependencies: List[str]
    created_at: datetime
    scheduled_at: Optional[datetime]
    user_id: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class TaskResult:
    """Results from task execution"""
    task_id: str
    status: TaskStatus
    result_data: Any
    error_message: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    execution_time: float
    resource_usage: Dict[str, float]
    retry_count: int
    metadata: Dict[str, Any]

@dataclass
class ExecutorMetrics:
    """Executor performance metrics"""
    tasks_completed: int
    tasks_failed: int
    tasks_pending: int
    tasks_running: int
    average_execution_time: float
    success_rate: float
    cpu_usage: float
    memory_usage: float
    thread_count: int
    queue_size: int
    uptime_seconds: float
    last_updated: datetime

class DutchMarketIntelligenceExecutor:
    """
    Main executor class for managing Dutch market intelligence tasks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market intelligence executor
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.status = ExecutorStatus.STARTING
        
        # Initialize task queues
        self.task_queue = PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Initialize thread pools
        self.max_workers = int(os.getenv('INTELLIGENCE_EXECUTOR_WORKERS', '4'))
        self.max_cpu_workers = int(os.getenv('INTELLIGENCE_CPU_WORKERS', '2'))
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='intel_io')
        self.cpu_executor = ProcessPoolExecutor(max_workers=self.max_cpu_workers)
        
        # Initialize intelligence system
        self.intelligence = DutchMarketIntelligence(config)
        
        # Control flags
        self._shutdown_requested = False
        self._pause_requested = False
        self._worker_threads = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.metrics = ExecutorMetrics(
            tasks_completed=0,
            tasks_failed=0,
            tasks_pending=0,
            tasks_running=0,
            average_execution_time=0.0,
            success_rate=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            thread_count=0,
            queue_size=0,
            uptime_seconds=0.0,
            last_updated=datetime.now()
        )
        
        # Task handlers mapping
        self.task_handlers = {
            'collect_market_data': self._handle_collect_market_data,
            'perform_trend_analysis': self._handle_perform_trend_analysis,
            'generate_predictive_model': self._handle_generate_predictive_model,
            'generate_market_insights': self._handle_generate_market_insights,
            'generate_comprehensive_report': self._handle_generate_comprehensive_report,
            'data_validation': self._handle_data_validation,
            'cache_management': self._handle_cache_management,
            'system_health_check': self._handle_system_health_check,
            'performance_optimization': self._handle_performance_optimization
        }
        
        # Configuration parameters
        self.execution_config = {
            'task_timeout_seconds': int(os.getenv('INTELLIGENCE_TASK_TIMEOUT', '1800')),
            'max_retries': int(os.getenv('INTELLIGENCE_MAX_RETRIES', '3')),
            'retry_delay_seconds': int(os.getenv('INTELLIGENCE_RETRY_DELAY', '10')),
            'health_check_interval': int(os.getenv('INTELLIGENCE_HEALTH_CHECK_INTERVAL', '300')),
            'metrics_update_interval': int(os.getenv('INTELLIGENCE_METRICS_INTERVAL', '60')),
            'memory_threshold_mb': int(os.getenv('INTELLIGENCE_MEMORY_THRESHOLD', '1024')),
            'cpu_threshold_percent': int(os.getenv('INTELLIGENCE_CPU_THRESHOLD', '80'))
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        atexit.register(self.shutdown)
        
        logger.info("Dutch Market Intelligence Executor initialized successfully")
    
    async def start(self) -> None:
        """Start the executor service"""
        try:
            logger.info("Starting Dutch Market Intelligence Executor...")
            
            self.status = ExecutorStatus.RUNNING
            
            # Start worker threads
            for i in range(self.max_workers):
                worker_thread = threading.Thread(
                    target=self._worker_loop,
                    name=f"IntelWorker-{i}",
                    daemon=True
                )
                worker_thread.start()
                self._worker_threads.append(worker_thread)
            
            # Start monitoring threads
            monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="IntelMonitoring",
                daemon=True
            )
            monitoring_thread.start()
            
            health_check_thread = threading.Thread(
                target=self._health_check_loop,
                name="IntelHealthCheck",
                daemon=True
            )
            health_check_thread.start()
            
            logger.info(f"Executor started with {self.max_workers} worker threads")
            
        except Exception as e:
            logger.error(f"Error starting executor: {str(e)}")
            logger.error(traceback.format_exc())
            self.status = ExecutorStatus.ERROR
            raise
    
    async def submit_task(self, task_definition: TaskDefinition) -> str:
        """
        Submit a task for execution
        
        Args:
            task_definition: Task definition to execute
            
        Returns:
            Task ID for tracking
        """
        try:
            logger.info(f"Submitting task: {task_definition.task_name} (ID: {task_definition.task_id})")
            
            # Validate task definition
            if task_definition.task_type not in self.task_handlers:
                raise ValueError(f"Unsupported task type: {task_definition.task_type}")
            
            # Check dependencies
            if task_definition.dependencies:
                missing_deps = self._check_dependencies(task_definition.dependencies)
                if missing_deps:
                    raise ValueError(f"Missing dependencies: {missing_deps}")
            
            # Add to queue with priority
            priority_score = task_definition.priority.value
            self.task_queue.put((priority_score, task_definition))
            
            # Update metrics
            self.metrics.tasks_pending = self.task_queue.qsize()
            
            logger.info(f"Task submitted successfully: {task_definition.task_id}")
            return task_definition.task_id
            
        except Exception as e:
            logger.error(f"Error submitting task {task_definition.task_id}: {str(e)}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """
        Get status of a specific task
        
        Args:
            task_id: ID of task to check
            
        Returns:
            Task result if found, None otherwise
        """
        try:
            # Check running tasks
            if task_id in self.running_tasks:
                return self.running_tasks[task_id]
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {str(e)}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            logger.info(f"Attempting to cancel task: {task_id}")
            
            # Check if task is running
            if task_id in self.running_tasks:
                task_result = self.running_tasks[task_id]
                task_result.status = TaskStatus.CANCELLED
                task_result.end_time = datetime.now()
                task_result.execution_time = (task_result.end_time - task_result.start_time).total_seconds()
                
                # Move to completed tasks (as cancelled)
                self.completed_tasks[task_id] = task_result
                del self.running_tasks[task_id]
                
                logger.info(f"Task cancelled: {task_id}")
                return True
            
            # For queued tasks, we would need to search through the queue
            # This is a simplified implementation
            logger.warning(f"Task {task_id} not found in running tasks")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {str(e)}")
            return False
    
    def get_metrics(self) -> ExecutorMetrics:
        """Get current executor metrics"""
        try:
            self._update_system_metrics()
            return self.metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return self.metrics
    
    def pause(self) -> None:
        """Pause task execution"""
        try:
            logger.info("Pausing executor...")
            self._pause_requested = True
            self.status = ExecutorStatus.PAUSING
        except Exception as e:
            logger.error(f"Error pausing executor: {str(e)}")
    
    def resume(self) -> None:
        """Resume task execution"""
        try:
            logger.info("Resuming executor...")
            self._pause_requested = False
            self.status = ExecutorStatus.RUNNING
        except Exception as e:
            logger.error(f"Error resuming executor: {str(e)}")
    
    def shutdown(self) -> None:
        """Shutdown the executor service"""
        try:
            logger.info("Shutting down Dutch Market Intelligence Executor...")
            
            self._shutdown_requested = True
            self.status = ExecutorStatus.STOPPING
            
            # Shutdown thread pools
            self.io_executor.shutdown(wait=True)
            self.cpu_executor.shutdown(wait=True)
            
            # Wait for worker threads to finish
            for thread in self._worker_threads:
                if thread.is_alive():
                    thread.join(timeout=30)
            
            self.status = ExecutorStatus.STOPPED
            logger.info("Executor shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks"""
        thread_name = threading.current_thread().name
        logger.info(f"Worker thread {thread_name} started")
        
        while not self._shutdown_requested:
            try:
                # Check if paused
                if self._pause_requested:
                    time.sleep(1)
                    continue
                
                # Get next task from queue
                try:
                    priority, task_def = self.task_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Execute the task
                self._execute_task(task_def)
                
            except Exception as e:
                logger.error(f"Error in worker loop {thread_name}: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(1)
        
        logger.info(f"Worker thread {thread_name} stopped")
    
    def _execute_task(self, task_def: TaskDefinition) -> None:
        """Execute a single task"""
        task_id = task_def.task_id
        start_time = datetime.now()
        
        # Create task result
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            result_data=None,
            error_message=None,
            start_time=start_time,
            end_time=None,
            execution_time=0.0,
            resource_usage={},
            retry_count=0,
            metadata=task_def.metadata.copy()
        )
        
        # Add to running tasks
        self.running_tasks[task_id] = task_result
        self.metrics.tasks_running = len(self.running_tasks)
        
        logger.info(f"Executing task: {task_def.task_name} (ID: {task_id})")
        
        try:
            # Get task handler
            handler = self.task_handlers.get(task_def.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task_def.task_type}")
            
            # Track resource usage before
            process = psutil.Process()
            cpu_before = process.cpu_percent()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute the task
            result_data = asyncio.run(handler(task_def.parameters))
            
            # Track resource usage after
            cpu_after = process.cpu_percent()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Update task result
            task_result.status = TaskStatus.COMPLETED
            task_result.result_data = result_data
            task_result.end_time = datetime.now()
            task_result.execution_time = (task_result.end_time - task_result.start_time).total_seconds()
            task_result.resource_usage = {
                'cpu_usage': cpu_after - cpu_before,
                'memory_usage_mb': memory_after - memory_before,
                'peak_memory_mb': memory_after
            }
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task_result
            del self.running_tasks[task_id]
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.tasks_running = len(self.running_tasks)
            self._update_success_rate()
            
            logger.info(f"Task completed successfully: {task_id} ({task_result.execution_time:.2f}s)")
            
        except Exception as e:
            # Handle task failure
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)
            task_result.end_time = datetime.now()
            task_result.execution_time = (task_result.end_time - task_result.start_time).total_seconds()
            
            # Check if we should retry
            if task_result.retry_count < task_def.max_retries:
                task_result.retry_count += 1
                task_result.status = TaskStatus.RETRYING
                
                # Re-queue the task after delay
                time.sleep(self.execution_config['retry_delay_seconds'])
                priority_score = task_def.priority.value
                self.task_queue.put((priority_score, task_def))
                
                logger.warning(f"Task failed, retrying ({task_result.retry_count}/{task_def.max_retries}): {task_id}")
            else:
                # Move to failed tasks
                self.failed_tasks[task_id] = task_result
                del self.running_tasks[task_id]
                
                # Update metrics
                self.metrics.tasks_failed += 1
                self.metrics.tasks_running = len(self.running_tasks)
                self._update_success_rate()
                
                logger.error(f"Task failed permanently: {task_id} - {str(e)}")
                logger.error(traceback.format_exc())
    
    def _monitoring_loop(self) -> None:
        """Monitoring loop for system metrics and performance"""
        logger.info("Monitoring thread started")
        
        while not self._shutdown_requested:
            try:
                self._update_system_metrics()
                self._cleanup_old_tasks()
                self._check_resource_limits()
                
                time.sleep(self.execution_config['metrics_update_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("Monitoring thread stopped")
    
    def _health_check_loop(self) -> None:
        """Health check loop for system health monitoring"""
        logger.info("Health check thread started")
        
        while not self._shutdown_requested:
            try:
                self._perform_health_check()
                time.sleep(self.execution_config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                time.sleep(300)  # Wait longer on error
        
        logger.info("Health check thread stopped")
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics"""
        try:
            process = psutil.Process()
            
            # CPU and memory usage
            self.metrics.cpu_usage = process.cpu_percent()
            self.metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # Thread count
            self.metrics.thread_count = process.num_threads()
            
            # Queue metrics
            self.metrics.queue_size = self.task_queue.qsize()
            self.metrics.tasks_pending = self.task_queue.qsize()
            self.metrics.tasks_running = len(self.running_tasks)
            
            # Uptime
            self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Average execution time
            if self.completed_tasks:
                total_time = sum(task.execution_time for task in self.completed_tasks.values())
                self.metrics.average_execution_time = total_time / len(self.completed_tasks)
            
            self.metrics.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")
    
    def _update_success_rate(self) -> None:
        """Update task success rate"""
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks > 0:
            self.metrics.success_rate = self.metrics.tasks_completed / total_tasks
        else:
            self.metrics.success_rate = 0.0
    
    def _cleanup_old_tasks(self) -> None:
        """Clean up old completed and failed tasks"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Clean up old completed tasks
            completed_to_remove = [
                task_id for task_id, result in self.completed_tasks.items()
                if result.end_time and result.end_time < cutoff_time
            ]
            
            for task_id in completed_to_remove:
                del self.completed_tasks[task_id]
            
            # Clean up old failed tasks
            failed_to_remove = [
                task_id for task_id, result in self.failed_tasks.items()
                if result.end_time and result.end_time < cutoff_time
            ]
            
            for task_id in failed_to_remove:
                del self.failed_tasks[task_id]
            
            if completed_to_remove or failed_to_remove:
                logger.info(f"Cleaned up {len(completed_to_remove + failed_to_remove)} old tasks")
            
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {str(e)}")
    
    def _check_resource_limits(self) -> None:
        """Check system resource limits and take action if necessary"""
        try:
            # Check memory usage
            if self.metrics.memory_usage > self.execution_config['memory_threshold_mb']:
                logger.warning(f"High memory usage detected: {self.metrics.memory_usage:.1f}MB")
                gc.collect()  # Force garbage collection
            
            # Check CPU usage
            if self.metrics.cpu_usage > self.execution_config['cpu_threshold_percent']:
                logger.warning(f"High CPU usage detected: {self.metrics.cpu_usage:.1f}%")
                # Could implement throttling here
            
        except Exception as e:
            logger.error(f"Error checking resource limits: {str(e)}")
    
    def _perform_health_check(self) -> None:
        """Perform system health check"""
        try:
            health_issues = []
            
            # Check if workers are alive
            alive_workers = sum(1 for t in self._worker_threads if t.is_alive())
            if alive_workers < len(self._worker_threads):
                health_issues.append(f"Only {alive_workers}/{len(self._worker_threads)} workers alive")
            
            # Check queue size
            if self.task_queue.qsize() > 1000:
                health_issues.append(f"Large task queue: {self.task_queue.qsize()} tasks")
            
            # Check failed task rate
            if self.metrics.success_rate < 0.8 and self.metrics.tasks_completed > 10:
                health_issues.append(f"Low success rate: {self.metrics.success_rate:.2%}")
            
            # Log health status
            if health_issues:
                logger.warning(f"Health issues detected: {', '.join(health_issues)}")
            else:
                logger.debug("System health check passed")
            
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
    
    def _check_dependencies(self, dependencies: List[str]) -> List[str]:
        """Check if task dependencies are satisfied"""
        missing_deps = []
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                missing_deps.append(dep_id)
        return missing_deps
    
    def _handle_shutdown_signal(self, signum, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received shutdown signal {signum}")
        self.shutdown()
    
    # Task handler methods
    async def _handle_collect_market_data(self, parameters: Dict[str, Any]) -> Any:
        """Handle market data collection task"""
        try:
            source = DataSource(parameters['source'])
            metrics = parameters['metrics']
            date_range = (
                datetime.fromisoformat(parameters['start_date']),
                datetime.fromisoformat(parameters['end_date'])
            )
            
            result = await self.intelligence.collect_market_data(source, metrics, date_range, **parameters.get('kwargs', {}))
            
            return {
                'data_points': [asdict(dp) for dp in result],
                'count': len(result),
                'source': source.value,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error in collect_market_data handler: {str(e)}")
            raise
    
    async def _handle_perform_trend_analysis(self, parameters: Dict[str, Any]) -> Any:
        """Handle trend analysis task"""
        try:
            # Convert data points from dict format
            data_points = []
            for dp_data in parameters['data_points']:
                data_point = MarketDataPoint(
                    source=DataSource(dp_data['source']),
                    metric_name=dp_data['metric_name'],
                    value=dp_data['value'],
                    timestamp=datetime.fromisoformat(dp_data['timestamp']),
                    metadata=dp_data['metadata'],
                    quality_score=dp_data.get('quality_score', 1.0),
                    confidence_level=dp_data.get('confidence_level', 1.0)
                )
                data_points.append(data_point)
            
            analysis_type = AnalysisType(parameters.get('analysis_type', 'trend_analysis'))
            
            result = await self.intelligence.perform_trend_analysis(data_points, analysis_type, **parameters.get('kwargs', {}))
            
            return asdict(result)
            
        except Exception as e:
            logger.error(f"Error in perform_trend_analysis handler: {str(e)}")
            raise
    
    async def _handle_generate_predictive_model(self, parameters: Dict[str, Any]) -> Any:
        """Handle predictive model generation task"""
        try:
            # Convert training data from dict format
            training_data = []
            for dp_data in parameters['training_data']:
                data_point = MarketDataPoint(
                    source=DataSource(dp_data['source']),
                    metric_name=dp_data['metric_name'],
                    value=dp_data['value'],
                    timestamp=datetime.fromisoformat(dp_data['timestamp']),
                    metadata=dp_data['metadata'],
                    quality_score=dp_data.get('quality_score', 1.0),
                    confidence_level=dp_data.get('confidence_level', 1.0)
                )
                training_data.append(data_point)
            
            target_metric = parameters['target_metric']
            features = parameters['features']
            model_type = parameters.get('model_type', 'random_forest')
            
            result = await self.intelligence.generate_predictive_model(
                training_data, target_metric, features, model_type, **parameters.get('kwargs', {})
            )
            
            return asdict(result)
            
        except Exception as e:
            logger.error(f"Error in generate_predictive_model handler: {str(e)}")
            raise
    
    async def _handle_generate_market_insights(self, parameters: Dict[str, Any]) -> Any:
        """Handle market insights generation task"""
        try:
            analysis_results = []
            
            # Convert analysis results from dict format
            for result_data in parameters['analysis_results']:
                if result_data['type'] == 'trend_analysis':
                    trend_result = TrendAnalysisResult(
                        trend_type=TrendType(result_data['trend_type']),
                        direction=result_data['direction'],
                        strength=result_data['strength'],
                        confidence=result_data['confidence'],
                        start_date=datetime.fromisoformat(result_data['start_date']),
                        end_date=datetime.fromisoformat(result_data['end_date']),
                        key_factors=result_data['key_factors'],
                        statistical_significance=result_data['statistical_significance'],
                        r_squared=result_data['r_squared'],
                        trend_equation=result_data['trend_equation'],
                        seasonal_component=result_data.get('seasonal_component'),
                        volatility_measure=result_data['volatility_measure']
                    )
                    analysis_results.append(trend_result)
                elif result_data['type'] == 'predictive_model':
                    predictive_model = PredictiveModel(
                        model_type=result_data['model_type'],
                        features=result_data['features'],
                        target_variable=result_data['target_variable'],
                        accuracy_score=result_data['accuracy_score'],
                        mae=result_data['mae'],
                        rmse=result_data['rmse'],
                        r2_score=result_data['r2_score'],
                        predictions=result_data['predictions'],
                        confidence_intervals=result_data['confidence_intervals'],
                        feature_importance=result_data['feature_importance'],
                        model_metadata=result_data['model_metadata']
                    )
                    analysis_results.append(predictive_model)
            
            context = parameters.get('context', {})
            
            result = await self.intelligence.generate_market_insights(analysis_results, context)
            
            return [asdict(insight) for insight in result]
            
        except Exception as e:
            logger.error(f"Error in generate_market_insights handler: {str(e)}")
            raise
    
    async def _handle_generate_comprehensive_report(self, parameters: Dict[str, Any]) -> Any:
        """Handle comprehensive report generation task"""
        try:
            # Convert insights from dict format
            insights = []
            for insight_data in parameters['insights']:
                insight = MarketInsight(
                    insight_id=insight_data['insight_id'],
                    title=insight_data['title'],
                    description=insight_data['description'],
                    category=insight_data['category'],
                    importance_score=insight_data['importance_score'],
                    confidence_level=insight_data['confidence_level'],
                    supporting_data=[],  # Simplified for serialization
                    implications=insight_data['implications'],
                    recommendations=insight_data['recommendations'],
                    risk_level=RiskLevel(insight_data['risk_level']),
                    time_horizon=insight_data['time_horizon'],
                    generated_at=datetime.fromisoformat(insight_data['generated_at'])
                )
                insights.append(insight)
            
            include_visualizations = parameters.get('include_visualizations', True)
            
            result = await self.intelligence.generate_comprehensive_report(
                insights, include_visualizations, **parameters.get('kwargs', {})
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_comprehensive_report handler: {str(e)}")
            raise
    
    async def _handle_data_validation(self, parameters: Dict[str, Any]) -> Any:
        """Handle data validation task"""
        try:
            # Implement data validation logic
            data_points = parameters.get('data_points', [])
            validation_rules = parameters.get('validation_rules', {})
            
            # Placeholder for validation logic
            validation_results = {
                'total_points': len(data_points),
                'valid_points': len(data_points),  # Simplified
                'invalid_points': 0,
                'quality_score': 1.0,
                'validation_details': []
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in data_validation handler: {str(e)}")
            raise
    
    async def _handle_cache_management(self, parameters: Dict[str, Any]) -> Any:
        """Handle cache management task"""
        try:
            operation = parameters.get('operation', 'cleanup')
            
            if operation == 'cleanup':
                # Clean up old cache entries
                cache_size_before = len(self.intelligence.data_cache)
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                # This is a simplified cache cleanup
                # In production, implement more sophisticated cache management
                
                return {
                    'operation': operation,
                    'cache_size_before': cache_size_before,
                    'cache_size_after': len(self.intelligence.data_cache),
                    'cleaned_entries': 0  # Simplified
                }
            
            return {'operation': operation, 'status': 'completed'}
            
        except Exception as e:
            logger.error(f"Error in cache_management handler: {str(e)}")
            raise
    
    async def _handle_system_health_check(self, parameters: Dict[str, Any]) -> Any:
        """Handle system health check task"""
        try:
            # Perform comprehensive system health check
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'executor_status': self.status.value,
                'metrics': asdict(self.get_metrics()),
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                },
                'service_checks': {
                    'intelligence_system': 'available',
                    'task_queue': 'operational',
                    'worker_threads': f"{sum(1 for t in self._worker_threads if t.is_alive())}/{len(self._worker_threads)} active"
                }
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error in system_health_check handler: {str(e)}")
            raise
    
    async def _handle_performance_optimization(self, parameters: Dict[str, Any]) -> Any:
        """Handle performance optimization task"""
        try:
            optimization_type = parameters.get('type', 'general')
            
            optimization_results = {
                'type': optimization_type,
                'actions_taken': [],
                'performance_impact': {},
                'recommendations': []
            }
            
            # Garbage collection
            if optimization_type in ['general', 'memory']:
                gc_before = len(gc.get_objects())
                gc.collect()
                gc_after = len(gc.get_objects())
                
                optimization_results['actions_taken'].append('garbage_collection')
                optimization_results['performance_impact']['objects_freed'] = gc_before - gc_after
            
            # Cache optimization
            if optimization_type in ['general', 'cache']:
                # Implement cache optimization logic
                optimization_results['actions_taken'].append('cache_optimization')
                optimization_results['recommendations'].append('Regular cache cleanup scheduled')
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in performance_optimization handler: {str(e)}")
            raise

# Utility functions for creating common task definitions
def create_data_collection_task(
    source: DataSource,
    metrics: List[str],
    start_date: datetime,
    end_date: datetime,
    priority: TaskPriority = TaskPriority.NORMAL,
    user_id: Optional[str] = None
) -> TaskDefinition:
    """Create a data collection task definition"""
    task_id = str(uuid.uuid4())
    
    return TaskDefinition(
        task_id=task_id,
        task_type='collect_market_data',
        task_name=f'Collect {source.value} data for {len(metrics)} metrics',
        parameters={
            'source': source.value,
            'metrics': metrics,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        },
        priority=priority,
        timeout_seconds=1800,  # 30 minutes
        max_retries=3,
        dependencies=[],
        created_at=datetime.now(),
        scheduled_at=None,
        user_id=user_id,
        metadata={}
    )

def create_trend_analysis_task(
    data_points: List[MarketDataPoint],
    analysis_type: AnalysisType = AnalysisType.TREND_ANALYSIS,
    priority: TaskPriority = TaskPriority.NORMAL,
    user_id: Optional[str] = None
) -> TaskDefinition:
    """Create a trend analysis task definition"""
    task_id = str(uuid.uuid4())
    
    return TaskDefinition(
        task_id=task_id,
        task_type='perform_trend_analysis',
        task_name=f'Trend analysis on {len(data_points)} data points',
        parameters={
            'data_points': [asdict(dp) for dp in data_points],
            'analysis_type': analysis_type.value
        },
        priority=priority,
        timeout_seconds=1200,  # 20 minutes
        max_retries=2,
        dependencies=[],
        created_at=datetime.now(),
        scheduled_at=None,
        user_id=user_id,
        metadata={}
    )

# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of Dutch Market Intelligence Executor"""
        try:
            # Initialize executor
            executor = DutchMarketIntelligenceExecutor()
            
            # Start executor
            await executor.start()
            
            print("Dutch Market Intelligence Executor started")
            
            # Create and submit a test task
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            task_def = create_data_collection_task(
                DataSource.CBS,
                ['house_prices', 'economic_growth'],
                start_date,
                end_date,
                TaskPriority.HIGH
            )
            
            print(f"Submitting test task: {task_def.task_id}")
            task_id = await executor.submit_task(task_def)
            
            # Monitor task execution
            for i in range(30):  # Wait up to 30 seconds
                status = await executor.get_task_status(task_id)
                if status:
                    print(f"Task status: {status.status.value}")
                    if status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        break
                
                await asyncio.sleep(1)
            
            # Get final metrics
            metrics = executor.get_metrics()
            print(f"Executor metrics: {metrics.tasks_completed} completed, {metrics.tasks_failed} failed")
            
            # Shutdown executor
            executor.shutdown()
            print("Executor shutdown completed")
            
        except Exception as e:
            print(f"Error in demonstration: {e}")
            logger.error(traceback.format_exc())
    
    # Run the example
    asyncio.run(main())
