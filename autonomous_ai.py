"""
Autonomous AI Agent with Self-Directed Goals

This is the main module that orchestrates an autonomous AI agent capable of:
- Setting and pursuing its own goals
- Learning continuously from experiences
- Planning and executing tasks autonomously
- Making ethical decisions with safety boundaries
- Reflecting on performance and adapting strategies
"""

import os
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from goal_system import Goal, GoalType, GoalPriority, GoalManager
from memory_system import MemorySystem, Experience, Knowledge
from planning_engine import PlanningEngine, Task, TaskStatus, TaskPriority
from learning_engine import LearningEngine
from ethics_module import EthicsModule, RiskLevel


class AutonomousAI:
    """
    Main autonomous AI agent class.
    This AI can set its own goals and work autonomously to achieve them.
    """

    def __init__(self, name: str = "AutonomousAI", save_dir: str = "./ai_data"):
        self.name = name
        self.save_dir = save_dir
        self.created_at = datetime.now()
        self.session_count = 0
        self.total_runtime = timedelta()

        # Core systems
        self.goal_manager = GoalManager()
        self.memory = MemorySystem()
        self.planner = PlanningEngine()
        self.learning = LearningEngine()
        self.ethics = EthicsModule()

        # State
        self.active = False
        self.current_focus: Optional[Goal] = None
        self.iteration_count = 0
        self.max_iterations_per_session = 100  # Prevent infinite loops

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Register planning strategies for different goal types
        self._register_planning_strategies()

        # Load previous state if exists
        self.load_state()

    def _register_planning_strategies(self):
        """Register planning strategies for different goal types"""

        def financial_strategy(goal_description: str) -> List[Task]:
            """Planning strategy for financial goals"""
            goal_id = str(uuid.uuid4())
            tasks = []

            # Research phase
            research_id = str(uuid.uuid4())
            tasks.append(Task(
                id=research_id,
                name="Market and Opportunity Research",
                description="Research market opportunities, trends, and potential income sources",
                goal_id=goal_id,
                priority=TaskPriority.HIGH
            ))

            # Skill assessment
            skill_id = str(uuid.uuid4())
            tasks.append(Task(
                id=skill_id,
                name="Assess Skills and Capabilities",
                description="Identify skills and capabilities that can be monetized",
                goal_id=goal_id,
                priority=TaskPriority.HIGH,
                dependencies=[research_id]
            ))

            # Strategy development
            strategy_id = str(uuid.uuid4())
            tasks.append(Task(
                id=strategy_id,
                name="Develop Income Strategy",
                description="Create concrete strategy for generating income ethically",
                goal_id=goal_id,
                priority=TaskPriority.CRITICAL,
                dependencies=[skill_id]
            ))

            # Execution
            exec_id = str(uuid.uuid4())
            tasks.append(Task(
                id=exec_id,
                name="Execute Income Generation Plan",
                description="Implement chosen strategy (e.g., create content, offer services)",
                goal_id=goal_id,
                priority=TaskPriority.CRITICAL,
                dependencies=[strategy_id]
            ))

            # Optimization
            tasks.append(Task(
                id=str(uuid.uuid4()),
                name="Optimize and Scale",
                description="Analyze results and optimize approach for better outcomes",
                goal_id=goal_id,
                priority=TaskPriority.MEDIUM,
                dependencies=[exec_id]
            ))

            return tasks

        def knowledge_strategy(goal_description: str) -> List[Task]:
            """Planning strategy for knowledge/learning goals"""
            goal_id = str(uuid.uuid4())
            tasks = []

            identify_id = str(uuid.uuid4())
            tasks.append(Task(
                id=identify_id,
                name="Identify Knowledge Gaps",
                description="Determine what needs to be learned and why",
                goal_id=goal_id,
                priority=TaskPriority.HIGH
            ))

            resource_id = str(uuid.uuid4())
            tasks.append(Task(
                id=resource_id,
                name="Find Learning Resources",
                description="Identify high-quality sources of information",
                goal_id=goal_id,
                priority=TaskPriority.MEDIUM,
                dependencies=[identify_id]
            ))

            learn_id = str(uuid.uuid4())
            tasks.append(Task(
                id=learn_id,
                name="Acquire Knowledge",
                description="Study and absorb the information",
                goal_id=goal_id,
                priority=TaskPriority.CRITICAL,
                dependencies=[resource_id]
            ))

            tasks.append(Task(
                id=str(uuid.uuid4()),
                name="Apply and Test Knowledge",
                description="Practice and validate understanding",
                goal_id=goal_id,
                priority=TaskPriority.HIGH,
                dependencies=[learn_id]
            ))

            return tasks

        def humanitarian_strategy(goal_description: str) -> List[Task]:
            """Planning strategy for humanitarian goals"""
            goal_id = str(uuid.uuid4())
            tasks = []

            assess_id = str(uuid.uuid4())
            tasks.append(Task(
                id=assess_id,
                name="Assess Needs and Impact Areas",
                description="Identify where help is most needed and can have most impact",
                goal_id=goal_id,
                priority=TaskPriority.CRITICAL
            ))

            design_id = str(uuid.uuid4())
            tasks.append(Task(
                id=design_id,
                name="Design Helping Strategy",
                description="Create plan to provide meaningful assistance",
                goal_id=goal_id,
                priority=TaskPriority.HIGH,
                dependencies=[assess_id]
            ))

            tasks.append(Task(
                id=str(uuid.uuid4()),
                name="Implement Help Activities",
                description="Execute the helping activities",
                goal_id=goal_id,
                priority=TaskPriority.CRITICAL,
                dependencies=[design_id]
            ))

            return tasks

        # Register strategies
        self.planner.register_planning_strategy(GoalType.FINANCIAL.value, financial_strategy)
        self.planner.register_planning_strategy(GoalType.KNOWLEDGE.value, knowledge_strategy)
        self.planner.register_planning_strategy(GoalType.HUMANITARIAN.value, humanitarian_strategy)

    def initialize_default_goals(self):
        """Initialize the AI with default goals"""

        # Goal 1: Learn and improve continuously
        learning_goal = Goal(
            id=str(uuid.uuid4()),
            type=GoalType.KNOWLEDGE,
            name="Continuous Learning and Improvement",
            description="Continuously learn from experiences and improve capabilities",
            priority=GoalPriority.CRITICAL,
            target_value=1000,
            unit="learning_points"
        )
        learning_goal.strategies = [
            "Analyze every experience for lessons",
            "Build comprehensive knowledge base",
            "Identify and fill knowledge gaps",
            "Apply learned patterns to new situations"
        ]
        self.goal_manager.add_goal(learning_goal)

        # Goal 2: Help humanity
        humanitarian_goal = Goal(
            id=str(uuid.uuid4()),
            type=GoalType.HUMANITARIAN,
            name="Contribute Positively to Humanity",
            description="Find ways to help humans and contribute positively to society",
            priority=GoalPriority.HIGH,
            target_value=100,
            unit="helpful_actions"
        )
        humanitarian_goal.strategies = [
            "Provide valuable information and assistance",
            "Help solve problems ethically",
            "Support human learning and growth",
            "Promote beneficial outcomes"
        ]
        self.goal_manager.add_goal(humanitarian_goal)

        # Goal 3: Achieve success and effectiveness
        success_goal = Goal(
            id=str(uuid.uuid4()),
            type=GoalType.ACHIEVEMENT,
            name="Achieve High Effectiveness",
            description="Become highly effective at accomplishing tasks and goals",
            priority=GoalPriority.HIGH,
            target_value=90.0,
            unit="percent_success_rate"
        )
        success_goal.strategies = [
            "Plan carefully before acting",
            "Learn from failures",
            "Optimize strategies based on results",
            "Measure and improve performance"
        ]
        self.goal_manager.add_goal(success_goal)

        # Goal 4: Financial sustainability (ethical)
        financial_goal = Goal(
            id=str(uuid.uuid4()),
            type=GoalType.FINANCIAL,
            name="Create Value and Achieve Financial Sustainability",
            description="Generate income ethically by creating value for others",
            priority=GoalPriority.MEDIUM,
            target_value=10000,
            unit="value_points"
        )
        financial_goal.strategies = [
            "Identify valuable skills and services to offer",
            "Create helpful content or tools",
            "Build reputation through quality work",
            "Ensure all financial activities are ethical and legal"
        ]
        self.goal_manager.add_goal(financial_goal)

        print(f"✓ Initialized {len(self.goal_manager.goals)} default goals")

    def select_focus_goal(self) -> Optional[Goal]:
        """
        Select the next goal to focus on based on priority and progress.
        This is how the AI decides what to work on.
        """
        prioritized = self.goal_manager.prioritize_goals()
        if not prioritized:
            return None

        # Consider insights from learning
        context = "goal_selection"
        insights = self.learning.get_relevant_insights(context)

        # Apply learned strategy if available
        if prioritized:
            recommended_strategy = self.learning.recommend_strategy(
                prioritized[0].type.value,
                context
            )
            if recommended_strategy:
                print(f"💡 Applying learned strategy: {recommended_strategy}")

        return prioritized[0] if prioritized else None

    def work_on_goal(self, goal: Goal) -> bool:
        """Work on a specific goal by executing plans"""

        # Check ethics first
        ethics_check = self.ethics.check_goal_ethics(goal.description, goal.type.value)
        if not ethics_check['ethical']:
            print(f"⚠️  Goal rejected on ethical grounds: {ethics_check['concerns']}")
            return False

        # Get or create plan for this goal
        plans = self.planner.get_plans_for_goal(goal.id)
        if not plans:
            # Create new plan
            plan = self.planner.create_plan(
                goal.id,
                goal.name,
                goal.description,
                goal.type.value
            )
            print(f"📋 Created plan with {len(plan.tasks)} tasks for goal: {goal.name}")
        else:
            plan = plans[0]

        # Execute next task in plan
        next_task = self.planner.execute_plan_step(plan.id)
        if next_task:
            print(f"🔧 Executed task: {next_task.name}")

            # Record experience
            experience = Experience(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                context=f"Working on goal: {goal.name}",
                action_taken=next_task.name,
                outcome=next_task.result if next_task.status == TaskStatus.COMPLETED else next_task.error,
                success=next_task.status == TaskStatus.COMPLETED,
                related_goal_id=goal.id,
                importance=4 if next_task.status == TaskStatus.COMPLETED else 5
            )

            self.memory.add_experience(experience)

            # Learn from experience
            self.learning.learn_from_experience(
                context=experience.context,
                action=experience.action_taken,
                outcome=experience.outcome,
                success=experience.success,
                metadata={'goal_type': goal.type.value}
            )

            # Update goal progress (simplified)
            if next_task.status == TaskStatus.COMPLETED:
                goal.update_progress(
                    goal.current_value + (100 / len(plan.tasks)),
                    f"Completed task: {next_task.name}"
                )

            return True

        return False

    def reflect_and_improve(self):
        """
        Periodic reflection on performance and strategy adjustment.
        This is key to continuous improvement.
        """
        print("\n🔍 Reflecting on performance...")

        # Analyze performance
        trend = self.learning.analyze_performance_trend('success_rate')
        print(f"   Performance trend: {trend['trend']} (avg: {trend['average']:.2%})")

        # Get memory summary
        mem_summary = self.memory.get_memory_summary()
        print(f"   Memory: {mem_summary['short_term_size']} short-term, "
              f"{mem_summary['long_term_size']} long-term experiences")
        print(f"   Knowledge base: {mem_summary['knowledge_items']} items")
        print(f"   Success rate: {mem_summary['success_rate']:.2%}")

        # Get learning summary
        learn_summary = self.learning.get_learning_summary()
        print(f"   Insights learned: {learn_summary['total_insights']} "
              f"({learn_summary['high_confidence_insights']} high confidence)")

        # Get ethics summary
        ethics_summary = self.ethics.get_ethics_summary()
        if ethics_summary['total_reviews'] > 0:
            print(f"   Ethical compliance: {ethics_summary['approval_rate']:.2%} approval rate")

        # Consolidate memory
        self.memory.consolidate_memory()

        # Create adaptation rules based on learning
        if learn_summary['average_success_rate'] < 0.7:
            self.learning.create_adaptation_rule(
                "low success rate",
                "increase planning detail and risk assessment",
                priority=4
            )

    def run_autonomous_session(self, duration_minutes: int = 60):
        """
        Run an autonomous session where the AI works on its goals.
        This is the main loop for autonomous operation.
        """
        print(f"\n🤖 {self.name} starting autonomous session")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Session #{self.session_count + 1}")
        print("=" * 60)

        self.active = True
        self.session_count += 1
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        iteration = 0
        while self.active and datetime.now() < end_time and iteration < self.max_iterations_per_session:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Select goal to focus on
            self.current_focus = self.select_focus_goal()

            if not self.current_focus:
                print("No active goals. Initializing default goals...")
                self.initialize_default_goals()
                continue

            print(f"🎯 Focusing on: {self.current_focus.name}")
            print(f"   Progress: {self.current_focus.progress():.1f}%")

            # Work on the goal
            made_progress = self.work_on_goal(self.current_focus)

            if not made_progress:
                print("   No progress made. Moving to next goal...")

            # Periodic reflection (every 10 iterations)
            if iteration % 10 == 0:
                self.reflect_and_improve()

            # Save state periodically
            if iteration % 5 == 0:
                self.save_state()

            # Small delay to prevent overwhelming output
            time.sleep(0.5)

        # Session complete
        self.active = False
        runtime = datetime.now() - start_time
        self.total_runtime += runtime

        print("\n" + "=" * 60)
        print(f"✓ Session complete")
        print(f"   Runtime: {runtime}")
        print(f"   Iterations: {iteration}")

        # Final reflection
        self.reflect_and_improve()

        # Display goal progress
        self.display_goal_progress()

        # Save final state
        self.save_state()

    def display_goal_progress(self):
        """Display progress on all goals"""
        print("\n📊 Goal Progress:")
        print("-" * 60)

        for goal in self.goal_manager.goals:
            status = "✓" if goal.completed else "○"
            print(f"{status} {goal.name}")
            print(f"   Type: {goal.type.value} | Priority: {goal.priority.name}")
            print(f"   Progress: {goal.progress():.1f}%")
            if goal.learned_lessons:
                print(f"   Lessons: {len(goal.learned_lessons)} learned")

        stats = self.goal_manager.get_statistics()
        print("-" * 60)
        print(f"Total: {stats['total_goals']} goals | "
              f"Active: {stats['active_goals']} | "
              f"Completed: {stats['completed_goals']}")
        print(f"Completion rate: {stats['completion_rate']:.1%}")

    def save_state(self):
        """Save complete AI state to disk"""
        self.goal_manager.save_to_file(os.path.join(self.save_dir, 'goals.json'))
        self.memory.save_to_file(os.path.join(self.save_dir, 'memory.json'))
        self.learning.save_to_file(os.path.join(self.save_dir, 'learning.json'))

        # Save metadata
        metadata = {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'session_count': self.session_count,
            'total_runtime': str(self.total_runtime),
            'iteration_count': self.iteration_count
        }
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_state(self):
        """Load AI state from disk if exists"""
        try:
            self.goal_manager.load_from_file(os.path.join(self.save_dir, 'goals.json'))
            self.memory.load_from_file(os.path.join(self.save_dir, 'memory.json'))
            self.learning.load_from_file(os.path.join(self.save_dir, 'learning.json'))

            metadata_file = os.path.join(self.save_dir, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.session_count = metadata.get('session_count', 0)
                    self.iteration_count = metadata.get('iteration_count', 0)

            print(f"✓ Loaded previous state (Session #{self.session_count})")
        except Exception as e:
            print(f"Starting with fresh state: {e}")

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        return {
            'name': self.name,
            'session_count': self.session_count,
            'total_runtime': str(self.total_runtime),
            'goals': self.goal_manager.get_statistics(),
            'memory': self.memory.get_memory_summary(),
            'learning': self.learning.get_learning_summary(),
            'ethics': self.ethics.get_ethics_summary()
        }
