#!/usr/bin/env python3
"""
Example: Demonstrating the Learning System

This example shows how the AI learns from experiences and improves over time.
"""

from learning_engine import LearningEngine
from memory_system import MemorySystem, Experience, Knowledge
from datetime import datetime
import uuid


def main():
    print("🧠 Learning System Example\n")

    # Create learning engine and memory
    learning = LearningEngine()
    memory = MemorySystem()

    # Simulate experiences
    experiences = [
        {
            'context': 'solving programming problem',
            'action': 'break problem into smaller pieces',
            'outcome': 'successfully solved complex problem',
            'success': True,
            'metadata': {'strategy': 'decomposition', 'goal_type': 'knowledge'}
        },
        {
            'context': 'solving programming problem',
            'action': 'try to solve everything at once',
            'outcome': 'got overwhelmed and made mistakes',
            'success': False,
            'metadata': {'strategy': 'brute_force', 'goal_type': 'knowledge'}
        },
        {
            'context': 'learning new topic',
            'action': 'practice with examples',
            'outcome': 'understood the concept well',
            'success': True,
            'metadata': {'strategy': 'hands_on_practice', 'goal_type': 'knowledge'}
        },
        {
            'context': 'learning new topic',
            'action': 'just read theory without practice',
            'outcome': 'forgot most of it quickly',
            'success': False,
            'metadata': {'strategy': 'passive_reading', 'goal_type': 'knowledge'}
        },
        {
            'context': 'helping someone',
            'action': 'listen first, then provide tailored advice',
            'outcome': 'person was very grateful and found it helpful',
            'success': True,
            'metadata': {'strategy': 'empathetic_approach', 'goal_type': 'humanitarian'}
        }
    ]

    print("📚 Learning from experiences...\n")

    # Learn from each experience
    for i, exp in enumerate(experiences, 1):
        print(f"Experience {i}:")
        print(f"  Context: {exp['context']}")
        print(f"  Action: {exp['action']}")
        print(f"  Outcome: {exp['outcome']}")
        print(f"  Success: {'✓' if exp['success'] else '✗'}")

        # AI learns from this experience
        learning.learn_from_experience(
            context=exp['context'],
            action=exp['action'],
            outcome=exp['outcome'],
            success=exp['success'],
            metadata=exp['metadata']
        )

        # Also add to memory
        experience = Experience(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context=exp['context'],
            action_taken=exp['action'],
            outcome=exp['outcome'],
            success=exp['success'],
            importance=4
        )
        memory.add_experience(experience)
        print()

    # Show what was learned
    print("=" * 70)
    print("🎓 LEARNING SUMMARY\n")

    summary = learning.get_learning_summary()
    print(f"Total insights learned: {summary['total_insights']}")
    print(f"High confidence insights: {summary['high_confidence_insights']}")
    print(f"Strategies learned: {summary['strategies_learned']}")
    print(f"Average success rate: {summary['average_success_rate']:.1%}\n")

    # Show specific insights
    print("💡 Insights gained:")
    for insight in learning.insights[:5]:
        print(f"  • {insight.insight}")
        print(f"    Confidence: {insight.confidence:.1%} | Evidence: {insight.evidence_count}")

    # Show strategy effectiveness
    print("\n📊 Strategy Effectiveness:")
    for strategy, stats in learning.strategy_effectiveness.items():
        if stats['attempts'] > 0:
            print(f"  • {strategy}")
            print(f"    Attempts: {stats['attempts']} | "
                  f"Success rate: {stats['effectiveness']:.1%}")

    # Demonstrate applying learned knowledge
    print("\n" + "=" * 70)
    print("🔮 APPLYING LEARNED KNOWLEDGE\n")

    # Get recommendations for specific contexts
    contexts = [
        'solving programming problem',
        'learning new topic',
        'helping someone'
    ]

    for context in contexts:
        print(f"Context: {context}")

        # Get relevant insights
        insights = learning.get_relevant_insights(context, min_confidence=0.5)
        if insights:
            print(f"  Relevant insights:")
            for insight in insights[:2]:
                print(f"    • {insight}")

        # Get recommended strategy
        strategy = learning.recommend_strategy('knowledge', context)
        if strategy:
            print(f"  Recommended strategy: {strategy}")

        print()

    # Performance trend analysis
    print("=" * 70)
    print("📈 PERFORMANCE TREND ANALYSIS\n")

    trend = learning.analyze_performance_trend('success_rate')
    print(f"Trend: {trend['trend']}")
    print(f"Average performance: {trend['average']:.1%}")
    print(f"Recent performance: {trend['recent_average']:.1%}")

    # Adaptation rules
    print("\n🔧 Creating adaptation rules based on learning...")
    learning.create_adaptation_rule(
        condition="when success rate is low",
        action="increase planning detail and break tasks into smaller steps",
        priority=5
    )
    learning.create_adaptation_rule(
        condition="when learning new topics",
        action="always include hands-on practice, not just theory",
        priority=4
    )

    print(f"✓ Created {len(learning.adaptation_rules)} adaptation rules")
    for rule in learning.adaptation_rules:
        print(f"  • If {rule['condition']}: {rule['action']}")

    # Save learning data
    learning.save_to_file('learning_example_data.json')
    memory.save_to_file('memory_example_data.json')
    print("\n✓ Learning data saved\n")


if __name__ == "__main__":
    main()
