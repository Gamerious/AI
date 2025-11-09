"""
Planning Engine for Autonomous AI Agent

This module handles task planning, decomposition, and execution strategies
for achieving goals autonomously.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid


class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority of a task"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    OPTIONAL = 1


@dataclass
class Task:
    """Represents a single task in the plan"""
    id: str
    name: str
    description: str
    goal_id: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_execute(self, completed_task_ids: List[str]) -> bool:
        """Check if all dependencies are met"""
        return all(dep_id in completed_task_ids for dep_id in self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'goal_id': self.goal_id,
            'priority': self.priority.value,
            'status': self.status.value,
            'dependencies': self.dependencies,
            'estimated_duration': str(self.estimated_duration) if self.estimated_duration else None,
            'actual_duration': str(self.actual_duration) if self.actual_duration else None,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': str(self.result) if self.result else None,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'metadata': self.metadata
        }


@dataclass
class Plan:
    """Represents a complete plan for achieving a goal"""
    id: str
    goal_id: str
    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"

    def get_executable_tasks(self) -> List[Task]:
        """Get tasks that can be executed now"""
        completed_ids = [t.id for t in self.tasks if t.status == TaskStatus.COMPLETED]
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING and t.can_execute(completed_ids)
        ]

    def get_next_task(self) -> Optional[Task]:
        """Get the next highest priority executable task"""
        executable = self.get_executable_tasks()
        if not executable:
            return None
        return max(executable, key=lambda t: t.priority.value)

    def progress(self) -> float:
        """Calculate plan completion percentage"""
        if not self.tasks:
            return 0.0
        completed = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        return (completed / len(self.tasks)) * 100


class PlanningEngine:
    """
    Planning engine that creates and manages plans for achieving goals.
    Decomposes high-level goals into actionable tasks.
    """

    def __init__(self):
        self.plans: Dict[str, Plan] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.planning_strategies: Dict[str, Callable] = {}

    def create_plan(self, goal_id: str, goal_name: str, goal_description: str,
                   goal_type: str) -> Plan:
        """Create a plan for achieving a goal"""
        plan_id = str(uuid.uuid4())
        plan = Plan(
            id=plan_id,
            goal_id=goal_id,
            name=f"Plan for: {goal_name}",
            description=f"Strategic plan to achieve: {goal_description}"
        )

        # Use goal type to determine planning strategy
        if goal_type in self.planning_strategies:
            tasks = self.planning_strategies[goal_type](goal_description)
            plan.tasks = tasks
        else:
            # Default planning strategy: decompose into generic tasks
            plan.tasks = self._default_task_decomposition(goal_id, goal_description)

        self.plans[plan_id] = plan
        return plan

    def _default_task_decomposition(self, goal_id: str, goal_description: str) -> List[Task]:
        """Default strategy for breaking down a goal into tasks"""
        tasks = []

        # Phase 1: Research and Analysis
        tasks.append(Task(
            id=str(uuid.uuid4()),
            name="Research and Information Gathering",
            description=f"Gather information and understand requirements for: {goal_description}",
            goal_id=goal_id,
            priority=TaskPriority.HIGH,
            estimated_duration=timedelta(hours=2)
        ))

        # Phase 2: Planning
        planning_task_id = str(uuid.uuid4())
        tasks.append(Task(
            id=planning_task_id,
            name="Detailed Planning",
            description="Create detailed action plan based on research",
            goal_id=goal_id,
            priority=TaskPriority.HIGH,
            dependencies=[tasks[0].id],
            estimated_duration=timedelta(hours=1)
        ))

        # Phase 3: Execution
        execution_task_id = str(uuid.uuid4())
        tasks.append(Task(
            id=execution_task_id,
            name="Execute Core Actions",
            description="Execute main actions to achieve the goal",
            goal_id=goal_id,
            priority=TaskPriority.CRITICAL,
            dependencies=[planning_task_id],
            estimated_duration=timedelta(hours=4)
        ))

        # Phase 4: Evaluation
        tasks.append(Task(
            id=str(uuid.uuid4()),
            name="Evaluate Results",
            description="Assess outcomes and determine if goal is achieved",
            goal_id=goal_id,
            priority=TaskPriority.MEDIUM,
            dependencies=[execution_task_id],
            estimated_duration=timedelta(hours=1)
        ))

        return tasks

    def register_planning_strategy(self, goal_type: str, strategy: Callable):
        """Register a custom planning strategy for a goal type"""
        self.planning_strategies[goal_type] = strategy

    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for executing specific task types"""
        self.task_handlers[task_type] = handler

    def execute_task(self, plan_id: str, task_id: str) -> bool:
        """Execute a specific task"""
        plan = self.plans.get(plan_id)
        if not plan:
            return False

        task = next((t for t in plan.tasks if t.id == task_id), None)
        if not task:
            return False

        # Check if task can be executed
        completed_ids = [t.id for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        if not task.can_execute(completed_ids):
            task.status = TaskStatus.BLOCKED
            return False

        # Execute task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            # Look for a registered handler
            handler = self.task_handlers.get(task.name)
            if handler:
                task.result = handler(task)
            else:
                # Default execution (simulation)
                task.result = f"Task '{task.name}' executed successfully"

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            if task.started_at:
                task.actual_duration = task.completed_at - task.started_at
            return True

        except Exception as e:
            task.error = str(e)
            task.retry_count += 1

            if task.retry_count >= task.max_retries:
                task.status = TaskStatus.FAILED
            else:
                task.status = TaskStatus.PENDING  # Will retry

            return False

    def execute_plan_step(self, plan_id: str) -> Optional[Task]:
        """Execute the next task in the plan"""
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        next_task = plan.get_next_task()
        if not next_task:
            return None

        self.execute_task(plan_id, next_task.id)
        return next_task

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID"""
        return self.plans.get(plan_id)

    def get_plans_for_goal(self, goal_id: str) -> List[Plan]:
        """Get all plans for a specific goal"""
        return [p for p in self.plans.values() if p.goal_id == goal_id]

    def get_plan_statistics(self, plan_id: str) -> Dict[str, Any]:
        """Get statistics for a plan"""
        plan = self.plans.get(plan_id)
        if not plan:
            return {}

        total = len(plan.tasks)
        completed = len([t for t in plan.tasks if t.status == TaskStatus.COMPLETED])
        failed = len([t for t in plan.tasks if t.status == TaskStatus.FAILED])
        in_progress = len([t for t in plan.tasks if t.status == TaskStatus.IN_PROGRESS])

        return {
            'plan_id': plan_id,
            'total_tasks': total,
            'completed_tasks': completed,
            'failed_tasks': failed,
            'in_progress_tasks': in_progress,
            'progress_percentage': plan.progress(),
            'status': plan.status
        }
