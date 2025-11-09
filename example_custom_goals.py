#!/usr/bin/env python3
"""
Example: Creating Custom Goals for the Autonomous AI

This example shows how to create and add custom goals to the AI agent.
"""

from autonomous_ai import AutonomousAI
from goal_system import Goal, GoalType, GoalPriority
import uuid


def main():
    print("🎯 Custom Goals Example\n")

    # Create AI instance
    ai = AutonomousAI(name="CustomGoalAI", save_dir="./custom_ai_data")

    # Example 1: Creative goal - Build something new
    creative_goal = Goal(
        id=str(uuid.uuid4()),
        type=GoalType.CREATIVE,
        name="Create Innovative Tool",
        description="Design and create a useful tool or application that helps people",
        priority=GoalPriority.HIGH,
        target_value=1,
        unit="completed_project"
    )
    creative_goal.strategies = [
        "Identify common problems people face",
        "Design elegant solutions",
        "Build and test prototypes",
        "Gather feedback and iterate"
    ]
    ai.goal_manager.add_goal(creative_goal)
    print(f"✓ Added creative goal: {creative_goal.name}")

    # Example 2: Knowledge goal - Master a specific domain
    knowledge_goal = Goal(
        id=str(uuid.uuid4()),
        type=GoalType.KNOWLEDGE,
        name="Master Machine Learning",
        description="Become expert in machine learning techniques and applications",
        priority=GoalPriority.HIGH,
        target_value=100,
        unit="concepts_mastered"
    )
    knowledge_goal.strategies = [
        "Study ML fundamentals deeply",
        "Implement algorithms from scratch",
        "Work on practical projects",
        "Stay updated with latest research"
    ]
    knowledge_goal.sub_goals = [
        Goal(
            id=str(uuid.uuid4()),
            type=GoalType.KNOWLEDGE,
            name="Learn Neural Networks",
            description="Understand neural network architectures",
            priority=GoalPriority.MEDIUM,
            target_value=20,
            unit="concepts"
        ),
        Goal(
            id=str(uuid.uuid4()),
            type=GoalType.KNOWLEDGE,
            name="Learn Reinforcement Learning",
            description="Master RL algorithms",
            priority=GoalPriority.MEDIUM,
            target_value=15,
            unit="concepts"
        )
    ]
    ai.goal_manager.add_goal(knowledge_goal)
    print(f"✓ Added knowledge goal: {knowledge_goal.name} with {len(knowledge_goal.sub_goals)} sub-goals")

    # Example 3: Social goal - Build network
    social_goal = Goal(
        id=str(uuid.uuid4()),
        type=GoalType.SOCIAL,
        name="Build Professional Network",
        description="Connect with experts and build meaningful professional relationships",
        priority=GoalPriority.MEDIUM,
        target_value=50,
        unit="meaningful_connections"
    )
    social_goal.strategies = [
        "Engage in technical communities",
        "Share knowledge and insights",
        "Collaborate on projects",
        "Provide value to others first"
    ]
    ai.goal_manager.add_goal(social_goal)
    print(f"✓ Added social goal: {social_goal.name}")

    # Example 4: Efficiency goal - Optimize processes
    efficiency_goal = Goal(
        id=str(uuid.uuid4()),
        type=GoalType.EFFICIENCY,
        name="Optimize Task Execution",
        description="Reduce task completion time by 30% through optimization",
        priority=GoalPriority.MEDIUM,
        target_value=30.0,
        unit="percent_improvement"
    )
    efficiency_goal.strategies = [
        "Analyze current workflows",
        "Identify bottlenecks",
        "Implement automation where possible",
        "Measure and iterate"
    ]
    ai.goal_manager.add_goal(efficiency_goal)
    print(f"✓ Added efficiency goal: {efficiency_goal.name}")

    # Show all goals
    print(f"\n📊 Total goals: {len(ai.goal_manager.goals)}")
    ai.display_goal_progress()

    # Save goals
    ai.save_state()
    print("\n✓ Goals saved successfully")

    # Demonstrate goal prioritization
    print("\n🎯 Prioritized goals (order AI will work on them):")
    prioritized = ai.goal_manager.prioritize_goals()
    for i, goal in enumerate(prioritized[:5], 1):
        print(f"   {i}. {goal.name} (Priority: {goal.priority.name})")

    print("\n💡 You can now run: python run_agent.py --data-dir ./custom_ai_data")
    print("   to let the AI work on these custom goals!\n")


if __name__ == "__main__":
    main()
