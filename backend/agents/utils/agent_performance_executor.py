"""
Agent Performance Metrics Executor - Task Orchestration Module
Created: 2024-01-15
Author: MortgageAI Development Team

Executor module for managing Agent Performance Metrics tasks including data collection,
analysis scheduling, dashboard generation, and optimization recommendations.

This module provides:
- Asynchronous task execution and management
- Performance metrics collection orchestration
- Analysis pipeline coordination
- Real-time monitoring and alerting
- Batch processing capabilities
- Error handling and retry mechanisms
- Resource management and optimization
- Task prioritization and scheduling
"""

import os
import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
import queue
import threading
from contextlib import asynccontextmanager
import uuid
import signal
import sys
from collections import defaultdict, deque

from .agent_performance_metrics import AgentPerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskType(Enum):
    """Types of performance analysis tasks"""
    METRICS_COLLECTION = "metrics_collection"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    DASHBOARD_GENERATION = "dashboard_generation"
    OPTIMIZATION_RECOMMENDATIONS = "optimization_recommendations"
    HEALTH_CHECK = "health_check"
    BATCH_ANALYSIS = "batch_analysis"
    ALERTING = "alerting"
    CLEANUP = "cleanup"

@dataclass
class PerformanceTask:
    """Performance analysis task definition"""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    parameters: Dict[str, Any]
    scheduled_at: datetime
    created_at: datetime
    status: TaskStatus
    attempts: int = 0
    max_attempts: int = 3
    retry_delay: float = 60.0
    timeout: Optional[float] = None
    dependencies: List[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

class AgentPerformanceExecutor:
    """
    Executor for managing agent performance metrics tasks and analysis workflows.
    
    Provides comprehensive task management, scheduling, and execution capabilities
    for the Agent Performance Metrics system.
    """

    def __init__(self, metrics_system: Optional[AgentPerformanceMetrics] = None):
        """Initialize the executor."""
        self.metrics_system = metrics_system or AgentPerformanceMetrics()
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        self.failed_tasks = deque(maxlen=500)
        
        # Execution control
        self.is_running = False
        self.worker_threads = []
        self.max_workers = int(os.getenv('PERFORMANCE_EXECUTOR_WORKERS', 4))
        self.shutdown_event = threading.Event()
        
        # Performance monitoring
        self.execution_stats = {
            'tasks_executed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0,
            'average_execution_time': 0,
            'peak_queue_size': 0,
            'worker_utilization': 0
        }
        
        # Scheduling and timing
        self.scheduler_thread = None
        self.collection_interval = int(os.getenv('METRICS_COLLECTION_INTERVAL', 300))  # 5 minutes
        self.analysis_interval = int(os.getenv('ANALYSIS_INTERVAL', 3600))  # 1 hour
        self.dashboard_refresh_interval = int(os.getenv('DASHBOARD_REFRESH_INTERVAL', 900))  # 15 minutes
        
        # Task handlers mapping
        self.task_handlers = {
            TaskType.METRICS_COLLECTION: self._handle_metrics_collection,
            TaskType.PERFORMANCE_ANALYSIS: self._handle_performance_analysis,
            TaskType.DASHBOARD_GENERATION: self._handle_dashboard_generation,
            TaskType.OPTIMIZATION_RECOMMENDATIONS: self._handle_optimization_recommendations,
            TaskType.HEALTH_CHECK: self._handle_health_check,
            TaskType.BATCH_ANALYSIS: self._handle_batch_analysis,
            TaskType.ALERTING: self._handle_alerting,
            TaskType.CLEANUP: self._handle_cleanup
        }
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_patterns = defaultdict(list)
        
        logger.info(f"Agent Performance Executor initialized with {self.max_workers} workers")

    async def initialize(self):
        """Initialize the executor and underlying systems."""
        try:
            await self.metrics_system.initialize()
            self._setup_signal_handlers()
            logger.info("Agent Performance Executor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize executor: {e}")
            raise

    def start(self):
        """Start the executor and worker threads."""
        if self.is_running:
            logger.warning("Executor is already running")
            return

        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"PerformanceWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="PerformanceScheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        
        # Schedule initial tasks
        self._schedule_periodic_tasks()
        
        logger.info(f"Agent Performance Executor started with {len(self.worker_threads)} workers")

    def stop(self, timeout: float = 30.0):
        """Stop the executor and all worker threads."""
        if not self.is_running:
            return

        logger.info("Stopping Agent Performance Executor...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for worker threads to complete
        for worker in self.worker_threads:
            worker.join(timeout=timeout / len(self.worker_threads))
        
        # Wait for scheduler thread
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("Agent Performance Executor stopped")

    def submit_task(
        self,
        task_type: TaskType,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        timeout: Optional[float] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_type: Type of task to execute
            parameters: Task parameters
            priority: Task priority level
            scheduled_at: When to execute the task (None for immediate)
            timeout: Task execution timeout in seconds
            dependencies: List of task IDs that must complete first
            metadata: Additional task metadata
            
        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())
        
        task = PerformanceTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            parameters=parameters,
            scheduled_at=scheduled_at or datetime.now(),
            created_at=datetime.now(),
            status=TaskStatus.PENDING,
            timeout=timeout,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        # Add to queue with priority (lower priority value = higher priority)
        queue_priority = (5 - priority.value, task.scheduled_at.timestamp())
        self.task_queue.put((queue_priority, task))
        
        # Update statistics
        current_size = self.task_queue.qsize()
        if current_size > self.execution_stats['peak_queue_size']:
            self.execution_stats['peak_queue_size'] = current_size
        
        logger.info(f"Task {task_id} ({task_type.value}) submitted with priority {priority.value}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'task_id': task_id,
                'status': task.status.value,
                'task_type': task.task_type.value,
                'priority': task.priority.value,
                'created_at': task.created_at.isoformat(),
                'scheduled_at': task.scheduled_at.isoformat(),
                'attempts': task.attempts,
                'execution_time': task.execution_time,
                'error': task.error
            }
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return {
                    'task_id': task_id,
                    'status': task.status.value,
                    'task_type': task.task_type.value,
                    'priority': task.priority.value,
                    'created_at': task.created_at.isoformat(),
                    'scheduled_at': task.scheduled_at.isoformat(),
                    'attempts': task.attempts,
                    'execution_time': task.execution_time,
                    'result': task.result,
                    'error': task.error
                }
        
        # Check failed tasks
        for task in self.failed_tasks:
            if task.task_id == task_id:
                return {
                    'task_id': task_id,
                    'status': task.status.value,
                    'task_type': task.task_type.value,
                    'priority': task.priority.value,
                    'created_at': task.created_at.isoformat(),
                    'scheduled_at': task.scheduled_at.isoformat(),
                    'attempts': task.attempts,
                    'execution_time': task.execution_time,
                    'error': task.error
                }
        
        return None

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get executor performance statistics."""
        active_count = len(self.active_tasks)
        queue_size = self.task_queue.qsize()
        
        stats = self.execution_stats.copy()
        stats.update({
            'active_tasks': active_count,
            'queue_size': queue_size,
            'completed_tasks_total': len(self.completed_tasks),
            'failed_tasks_total': len(self.failed_tasks),
            'worker_threads': len(self.worker_threads),
            'is_running': self.is_running,
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
            'error_patterns': dict(self.error_counts)
        })
        
        return stats

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED
                logger.info(f"Task {task_id} cancelled")
                return True
        
        return False

    def _worker_loop(self):
        """Main worker loop for executing tasks."""
        thread_name = threading.current_thread().name
        logger.info(f"Worker {thread_name} started")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get next task with timeout
                try:
                    priority, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if task should be executed now
                if task.scheduled_at > datetime.now():
                    # Put back in queue for later
                    self.task_queue.put((priority, task))
                    time.sleep(1)
                    continue
                
                # Check dependencies
                if not self._check_dependencies(task):
                    # Put back in queue for later
                    self.task_queue.put((priority, task))
                    time.sleep(5)
                    continue
                
                # Execute task
                self.active_tasks[task.task_id] = task
                self._execute_task(task)
                
                # Clean up
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {thread_name} error: {e}")
                time.sleep(1)
        
        logger.info(f"Worker {thread_name} stopped")

    def _scheduler_loop(self):
        """Scheduler loop for periodic task scheduling."""
        logger.info("Performance scheduler started")
        
        last_collection = datetime.now()
        last_analysis = datetime.now()
        last_dashboard = datetime.now()
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                now = datetime.now()
                
                # Schedule metrics collection
                if (now - last_collection).total_seconds() >= self.collection_interval:
                    self._schedule_metrics_collection()
                    last_collection = now
                
                # Schedule performance analysis
                if (now - last_analysis).total_seconds() >= self.analysis_interval:
                    self._schedule_performance_analysis()
                    last_analysis = now
                
                # Schedule dashboard refresh
                if (now - last_dashboard).total_seconds() >= self.dashboard_refresh_interval:
                    self._schedule_dashboard_refresh()
                    last_dashboard = now
                
                # Health check every 5 minutes
                if now.minute % 5 == 0 and now.second < 5:
                    self._schedule_health_check()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(10)
        
        logger.info("Performance scheduler stopped")

    def _execute_task(self, task: PerformanceTask):
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        start_time = time.time()
        
        try:
            logger.info(f"Executing task {task.task_id} ({task.task_type.value})")
            
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type {task.task_type.value}")
            
            # Set timeout if specified
            if task.timeout:
                # Note: In a production system, you might want to use asyncio.timeout
                # or implement a more sophisticated timeout mechanism
                pass
            
            # Execute task handler
            result = asyncio.run(handler(task))
            
            # Record success
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.execution_time = time.time() - start_time
            
            self.completed_tasks.append(task)
            self.execution_stats['tasks_completed'] += 1
            self.execution_stats['total_execution_time'] += task.execution_time
            self.execution_stats['average_execution_time'] = (
                self.execution_stats['total_execution_time'] / 
                max(self.execution_stats['tasks_completed'], 1)
            )
            
            logger.info(f"Task {task.task_id} completed in {task.execution_time:.2f}s")
            
        except Exception as e:
            # Record failure
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.execution_time = time.time() - start_time
            task.attempts += 1
            
            error_type = type(e).__name__
            self.error_counts[error_type] += 1
            self.error_patterns[error_type].append({
                'task_id': task.task_id,
                'task_type': task.task_type.value,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            # Retry logic
            if task.attempts < task.max_attempts and self._should_retry_task(task, e):
                task.status = TaskStatus.RETRYING
                retry_time = datetime.now() + timedelta(seconds=task.retry_delay * task.attempts)
                task.scheduled_at = retry_time
                
                # Re-queue for retry
                priority = (5 - task.priority.value, retry_time.timestamp())
                self.task_queue.put((priority, task))
                
                logger.warning(f"Task {task.task_id} failed, retrying in {task.retry_delay * task.attempts}s")
            else:
                self.failed_tasks.append(task)
                self.execution_stats['tasks_failed'] += 1
                
                logger.error(f"Task {task.task_id} failed permanently: {e}")
        
        finally:
            self.execution_stats['tasks_executed'] += 1

    def _check_dependencies(self, task: PerformanceTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            # Check if dependency is completed
            found = False
            for completed_task in self.completed_tasks:
                if completed_task.task_id == dep_id and completed_task.status == TaskStatus.COMPLETED:
                    found = True
                    break
            
            if not found:
                return False
        
        return True

    def _should_retry_task(self, task: PerformanceTask, error: Exception) -> bool:
        """Determine if a task should be retried based on the error."""
        # Don't retry certain types of errors
        non_retryable_errors = [
            'ValueError',
            'TypeError',
            'KeyError'
        ]
        
        error_type = type(error).__name__
        if error_type in non_retryable_errors:
            return False
        
        # Don't retry if error rate is too high
        if self.error_counts[error_type] > 10:
            return False
        
        return True

    def _schedule_periodic_tasks(self):
        """Schedule initial periodic tasks."""
        # Schedule immediate health check
        self.submit_task(
            TaskType.HEALTH_CHECK,
            parameters={},
            priority=TaskPriority.HIGH
        )
        
        # Schedule initial metrics collection
        self.submit_task(
            TaskType.METRICS_COLLECTION,
            parameters={'collection_type': 'initial'},
            priority=TaskPriority.NORMAL
        )

    def _schedule_metrics_collection(self):
        """Schedule metrics collection for all active agents."""
        task_id = self.submit_task(
            TaskType.METRICS_COLLECTION,
            parameters={
                'collection_type': 'scheduled',
                'include_all_agents': True,
                'time_period': 'recent'
            },
            priority=TaskPriority.NORMAL
        )
        logger.debug(f"Scheduled metrics collection task {task_id}")

    def _schedule_performance_analysis(self):
        """Schedule performance analysis tasks."""
        task_id = self.submit_task(
            TaskType.PERFORMANCE_ANALYSIS,
            parameters={
                'analysis_type': 'comprehensive',
                'include_forecasting': True,
                'include_recommendations': True
            },
            priority=TaskPriority.NORMAL
        )
        logger.debug(f"Scheduled performance analysis task {task_id}")

    def _schedule_dashboard_refresh(self):
        """Schedule dashboard data refresh."""
        task_id = self.submit_task(
            TaskType.DASHBOARD_GENERATION,
            parameters={
                'dashboard_type': 'comprehensive',
                'time_range': 'last_30_days',
                'include_forecasts': True
            },
            priority=TaskPriority.LOW
        )
        logger.debug(f"Scheduled dashboard refresh task {task_id}")

    def _schedule_health_check(self):
        """Schedule system health check."""
        task_id = self.submit_task(
            TaskType.HEALTH_CHECK,
            parameters={},
            priority=TaskPriority.HIGH
        )
        logger.debug(f"Scheduled health check task {task_id}")

    # Task Handlers
    async def _handle_metrics_collection(self, task: PerformanceTask) -> Dict[str, Any]:
        """Handle metrics collection task."""
        parameters = task.parameters
        
        if parameters.get('include_all_agents', False):
            # Collect metrics for all agents
            agent_ids = await self._get_all_active_agents()
            results = {}
            
            for agent_id in agent_ids:
                try:
                    result = await self.metrics_system.collect_agent_metrics(
                        agent_id=agent_id,
                        time_period=parameters.get('time_period', 'daily'),
                        include_context=parameters.get('include_context', True)
                    )
                    results[agent_id] = result
                except Exception as e:
                    logger.error(f"Failed to collect metrics for agent {agent_id}: {e}")
                    results[agent_id] = {'success': False, 'error': str(e)}
            
            return {
                'collection_type': parameters.get('collection_type', 'scheduled'),
                'agent_count': len(agent_ids),
                'results': results,
                'successful_collections': sum(1 for r in results.values() if r.get('success', False)),
                'failed_collections': sum(1 for r in results.values() if not r.get('success', False))
            }
        else:
            # Collect metrics for specific agent
            agent_id = parameters.get('agent_id')
            if not agent_id:
                raise ValueError("Agent ID required for metrics collection")
            
            result = await self.metrics_system.collect_agent_metrics(
                agent_id=agent_id,
                time_period=parameters.get('time_period', 'daily'),
                include_context=parameters.get('include_context', True)
            )
            
            return result

    async def _handle_performance_analysis(self, task: PerformanceTask) -> Dict[str, Any]:
        """Handle performance analysis task."""
        parameters = task.parameters
        
        analysis_type = parameters.get('analysis_type', 'standard')
        agent_ids = parameters.get('agent_ids')
        
        if not agent_ids:
            agent_ids = await self._get_all_active_agents()
        
        results = {}
        
        for agent_id in agent_ids:
            try:
                result = await self.metrics_system.analyze_agent_performance(
                    agent_id=agent_id,
                    analysis_period=parameters.get('analysis_period', 'monthly'),
                    include_forecasting=parameters.get('include_forecasting', True),
                    include_recommendations=parameters.get('include_recommendations', True)
                )
                results[agent_id] = result
            except Exception as e:
                logger.error(f"Failed to analyze performance for agent {agent_id}: {e}")
                results[agent_id] = {'success': False, 'error': str(e)}
        
        return {
            'analysis_type': analysis_type,
            'agent_count': len(agent_ids),
            'results': results,
            'successful_analyses': sum(1 for r in results.values() if r.get('success', False))
        }

    async def _handle_dashboard_generation(self, task: PerformanceTask) -> Dict[str, Any]:
        """Handle dashboard generation task."""
        parameters = task.parameters
        
        result = await self.metrics_system.generate_performance_dashboard(
            agent_ids=parameters.get('agent_ids'),
            time_range=parameters.get('time_range', 'last_30_days'),
            dashboard_type=parameters.get('dashboard_type', 'comprehensive')
        )
        
        return result

    async def _handle_optimization_recommendations(self, task: PerformanceTask) -> Dict[str, Any]:
        """Handle optimization recommendations task."""
        parameters = task.parameters
        agent_id = parameters.get('agent_id')
        
        if not agent_id:
            raise ValueError("Agent ID required for optimization recommendations")
        
        result = await self.metrics_system.get_optimization_recommendations(
            agent_id=agent_id,
            focus_areas=parameters.get('focus_areas'),
            priority_filter=parameters.get('priority_filter')
        )
        
        return result

    async def _handle_health_check(self, task: PerformanceTask) -> Dict[str, Any]:
        """Handle health check task."""
        result = await self.metrics_system.get_health_status()
        
        # Add executor-specific health information
        executor_health = {
            'executor_status': 'healthy' if self.is_running else 'stopped',
            'active_workers': len([w for w in self.worker_threads if w.is_alive()]),
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'execution_stats': self.execution_stats
        }
        
        if result.get('success'):
            result['health']['executor'] = executor_health
        
        return result

    async def _handle_batch_analysis(self, task: PerformanceTask) -> Dict[str, Any]:
        """Handle batch analysis task."""
        parameters = task.parameters
        agent_ids = parameters.get('agent_ids', [])
        
        if not agent_ids:
            agent_ids = await self._get_all_active_agents()
        
        batch_results = []
        
        for agent_id in agent_ids:
            # Submit individual analysis tasks
            analysis_task_id = self.submit_task(
                TaskType.PERFORMANCE_ANALYSIS,
                parameters={
                    'agent_id': agent_id,
                    'analysis_period': parameters.get('analysis_period', 'weekly')
                },
                priority=TaskPriority.LOW
            )
            batch_results.append({
                'agent_id': agent_id,
                'analysis_task_id': analysis_task_id
            })
        
        return {
            'batch_type': 'performance_analysis',
            'submitted_tasks': len(batch_results),
            'task_details': batch_results
        }

    async def _handle_alerting(self, task: PerformanceTask) -> Dict[str, Any]:
        """Handle alerting task."""
        parameters = task.parameters
        
        # Check for performance alerts
        alerts = await self._check_performance_alerts(parameters.get('agent_ids'))
        
        # Process alerts and send notifications if needed
        processed_alerts = []
        for alert in alerts:
            if alert.get('severity') in ['high', 'critical']:
                # Send notification (would integrate with notification system)
                logger.warning(f"Performance alert: {alert}")
            processed_alerts.append(alert)
        
        return {
            'alerts_checked': len(alerts),
            'high_priority_alerts': len([a for a in alerts if a.get('severity') in ['high', 'critical']]),
            'alerts': processed_alerts
        }

    async def _handle_cleanup(self, task: PerformanceTask) -> Dict[str, Any]:
        """Handle cleanup task."""
        parameters = task.parameters
        
        # Cleanup old metrics data
        cleanup_results = {
            'metrics_cleaned': 0,
            'analysis_cleaned': 0,
            'tasks_cleaned': 0
        }
        
        # Clean up completed/failed task history
        if len(self.completed_tasks) > 500:
            removed = len(self.completed_tasks) - 500
            self.completed_tasks = deque(list(self.completed_tasks)[-500:], maxlen=1000)
            cleanup_results['tasks_cleaned'] += removed
        
        if len(self.failed_tasks) > 250:
            removed = len(self.failed_tasks) - 250
            self.failed_tasks = deque(list(self.failed_tasks)[-250:], maxlen=500)
            cleanup_results['tasks_cleaned'] += removed
        
        return cleanup_results

    async def _get_all_active_agents(self) -> List[str]:
        """Get list of all active agent IDs."""
        # This would query the database for active agents
        # For now, return a mock list
        return ['agent_001', 'agent_002', 'agent_003']

    async def _check_performance_alerts(self, agent_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        
        if not agent_ids:
            agent_ids = await self._get_all_active_agents()
        
        for agent_id in agent_ids:
            # Check recent performance metrics for alerts
            # This would implement actual alerting logic
            pass
        
        return alerts

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def cleanup(self):
        """Clean up resources."""
        try:
            self.stop()
            await self.metrics_system.cleanup()
            logger.info("Agent Performance Executor cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup executor: {e}")

# Context manager for executor lifecycle
@asynccontextmanager
async def performance_executor_context():
    """Context manager for managing executor lifecycle."""
    executor = AgentPerformanceExecutor()
    try:
        await executor.initialize()
        executor.start()
        yield executor
    finally:
        await executor.cleanup()

# Example usage and testing
async def main():
    """Main function for testing the executor."""
    async with performance_executor_context() as executor:
        # Submit test tasks
        print("Submitting test tasks...")
        
        # Metrics collection task
        metrics_task_id = executor.submit_task(
            TaskType.METRICS_COLLECTION,
            parameters={'agent_id': 'agent_001', 'time_period': 'daily'},
            priority=TaskPriority.HIGH
        )
        
        # Performance analysis task
        analysis_task_id = executor.submit_task(
            TaskType.PERFORMANCE_ANALYSIS,
            parameters={'agent_ids': ['agent_001'], 'analysis_period': 'weekly'},
            priority=TaskPriority.NORMAL
        )
        
        # Dashboard generation task
        dashboard_task_id = executor.submit_task(
            TaskType.DASHBOARD_GENERATION,
            parameters={'time_range': 'last_7_days'},
            priority=TaskPriority.LOW
        )
        
        # Wait for tasks to complete
        await asyncio.sleep(10)
        
        # Check task statuses
        print(f"Metrics task status: {executor.get_task_status(metrics_task_id)}")
        print(f"Analysis task status: {executor.get_task_status(analysis_task_id)}")
        print(f"Dashboard task status: {executor.get_task_status(dashboard_task_id)}")
        
        # Get execution statistics
        stats = executor.get_execution_statistics()
        print(f"Execution statistics: {stats}")
        
        # Wait a bit more for processing
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
