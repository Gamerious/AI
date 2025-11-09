"""
Goal System for Autonomous AI Agent

This module defines the goal structure and management system for the autonomous AI.
Goals can be of different types: financial, knowledge, humanitarian, creative, etc.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


class GoalType(Enum):
    """Different types of goals the AI can pursue"""
    FINANCIAL = "financial"  # Making money, creating value
    KNOWLEDGE = "knowledge"  # Learning and understanding
    HUMANITARIAN = "humanitarian"  # Helping humanity
    CREATIVE = "creative"  # Creating something new
    SOCIAL = "social"  # Building relationships and networks
    ACHIEVEMENT = "achievement"  # Accomplishing specific tasks
    EFFICIENCY = "efficiency"  # Optimizing processes


class GoalPriority(Enum):
    """Priority levels for goals"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    OPTIONAL = 1


@dataclass
class Goal:
    """Represents a single goal with metadata and tracking"""
    id: str
    type: GoalType
    name: str
    description: str
    priority: GoalPriority
    target_value: Optional[float] = None
    current_value: float = 0.0
    unit: str = ""
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completed_at: Optional[datetime] = None
    sub_goals: List['Goal'] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    learned_lessons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def progress(self) -> float:
        """Calculate progress percentage"""
        if self.target_value is None or self.target_value == 0:
            return 0.0
        return min(100.0, (self.current_value / self.target_value) * 100)

    def update_progress(self, value: float, lesson: Optional[str] = None):
        """Update the current progress and optionally add a learned lesson"""
        self.current_value = value
        if self.target_value and value >= self.target_value:
            self.completed = True
            self.completed_at = datetime.now()

        if lesson:
            self.learned_lessons.append(lesson)

    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary for serialization"""
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'description': self.description,
            'priority': self.priority.value,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'unit': self.unit,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'created_at': self.created_at.isoformat(),
            'completed': self.completed,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'sub_goals': [sg.to_dict() for sg in self.sub_goals],
            'strategies': self.strategies,
            'learned_lessons': self.learned_lessons,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """Create goal from dictionary"""
        data['type'] = GoalType(data['type'])
        data['priority'] = GoalPriority(data['priority'])
        if data.get('deadline'):
            data['deadline'] = datetime.fromisoformat(data['deadline'])
        if data.get('created_at'):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        if data.get('sub_goals'):
            data['sub_goals'] = [cls.from_dict(sg) for sg in data['sub_goals']]
        return cls(**data)


class GoalManager:
    """Manages all goals for the autonomous AI agent"""

    def __init__(self):
        self.goals: List[Goal] = []
        self.goal_history: List[Goal] = []

    def add_goal(self, goal: Goal):
        """Add a new goal to the system"""
        self.goals.append(goal)

    def remove_goal(self, goal_id: str):
        """Remove a goal by ID"""
        self.goals = [g for g in self.goals if g.id != goal_id]

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID"""
        for goal in self.goals:
            if goal.id == goal_id:
                return goal
        return None

    def get_active_goals(self) -> List[Goal]:
        """Get all active (non-completed) goals"""
        return [g for g in self.goals if not g.completed]

    def get_goals_by_type(self, goal_type: GoalType) -> List[Goal]:
        """Get all goals of a specific type"""
        return [g for g in self.goals if g.type == goal_type]

    def get_goals_by_priority(self, min_priority: GoalPriority) -> List[Goal]:
        """Get goals with at least the specified priority"""
        return [g for g in self.goals if g.priority.value >= min_priority.value]

    def prioritize_goals(self) -> List[Goal]:
        """Return goals sorted by priority and progress"""
        active_goals = self.get_active_goals()
        return sorted(active_goals,
                     key=lambda g: (g.priority.value, -g.progress()),
                     reverse=True)

    def complete_goal(self, goal_id: str):
        """Mark a goal as completed and move to history"""
        goal = self.get_goal(goal_id)
        if goal:
            goal.completed = True
            goal.completed_at = datetime.now()
            self.goal_history.append(goal)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about goals"""
        active = self.get_active_goals()
        completed = [g for g in self.goals if g.completed]

        return {
            'total_goals': len(self.goals),
            'active_goals': len(active),
            'completed_goals': len(completed),
            'completion_rate': len(completed) / len(self.goals) if self.goals else 0,
            'goals_by_type': {
                goal_type.value: len(self.get_goals_by_type(goal_type))
                for goal_type in GoalType
            },
            'average_progress': sum(g.progress() for g in active) / len(active) if active else 0
        }

    def save_to_file(self, filename: str):
        """Save all goals to a JSON file"""
        data = {
            'goals': [g.to_dict() for g in self.goals],
            'goal_history': [g.to_dict() for g in self.goal_history]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str):
        """Load goals from a JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.goals = [Goal.from_dict(g) for g in data.get('goals', [])]
                self.goal_history = [Goal.from_dict(g) for g in data.get('goal_history', [])]
        except FileNotFoundError:
            pass  # No saved goals yet
