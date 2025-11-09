#!/usr/bin/env python3
"""
Main entry point for running the Autonomous AI Agent

Usage:
    python run_agent.py                  # Run single session
    python run_agent.py --duration 120   # Run for 120 minutes
    python run_agent.py --status         # Show status only
    python run_agent.py --reset          # Reset and start fresh
"""

import argparse
import sys
import os
from autonomous_ai import AutonomousAI


def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        🤖 AUTONOMOUS AI AGENT SYSTEM 🤖                  ║
    ║                                                           ║
    ║   Ein selbstständiges KI-System mit eigenen Zielen      ║
    ║   A self-directed AI system with its own goals          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_capabilities():
    """Print AI capabilities"""
    capabilities = """
    CAPABILITIES / FÄHIGKEITEN:

    ✓ Goal-Oriented Behavior (Zielorientiertes Verhalten)
      - Sets and pursues multiple types of goals
      - Prioritizes goals dynamically
      - Tracks progress and adapts

    ✓ Continuous Learning (Kontinuierliches Lernen)
      - Learns from every experience
      - Builds long-term memory
      - Improves strategies over time

    ✓ Autonomous Planning (Autonome Planung)
      - Breaks down goals into actionable tasks
      - Executes plans independently
      - Adapts plans based on outcomes

    ✓ Ethical Decision Making (Ethische Entscheidungen)
      - Follows ethical principles
      - Safety boundaries enforced
      - Human oversight for critical decisions

    ✓ Self-Reflection (Selbstreflexion)
      - Analyzes own performance
      - Identifies areas for improvement
      - Creates adaptation rules

    DEFAULT GOALS / STANDARDZIELE:

    1. 📚 Continuous Learning - Learn and improve continuously
    2. 🌍 Help Humanity - Contribute positively to society
    3. 🎯 Achieve Success - Become highly effective
    4. 💰 Financial Sustainability - Create value ethically
    """
    print(capabilities)


def main():
    parser = argparse.ArgumentParser(
        description='Run the Autonomous AI Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Session duration in minutes (default: 60)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show status report only'
    )

    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset AI state and start fresh'
    )

    parser.add_argument(
        '--name',
        type=str,
        default='AutonomousAI',
        help='Name for the AI agent'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./ai_data',
        help='Directory for saving AI state'
    )

    parser.add_argument(
        '--no-init',
        action='store_true',
        help='Do not initialize default goals'
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Handle reset
    if args.reset:
        import shutil
        if os.path.exists(args.data_dir):
            response = input(f"⚠️  This will delete all data in {args.data_dir}. Continue? (yes/no): ")
            if response.lower() == 'yes':
                shutil.rmtree(args.data_dir)
                print("✓ AI state reset successfully")
            else:
                print("Reset cancelled")
                return
        else:
            print("No existing state to reset")
            return

    # Create AI instance
    print(f"\n🚀 Initializing {args.name}...")
    ai = AutonomousAI(name=args.name, save_dir=args.data_dir)

    # Initialize default goals if this is first run and not disabled
    if not args.no_init and ai.session_count == 0:
        print("\n📝 Initializing default goals...")
        ai.initialize_default_goals()
        print_capabilities()

    # Show status if requested
    if args.status:
        print("\n" + "=" * 70)
        print("STATUS REPORT / STATUSBERICHT")
        print("=" * 70)

        status = ai.get_status_report()

        print(f"\n🤖 Agent: {status['name']}")
        print(f"   Sessions run: {status['session_count']}")
        print(f"   Total runtime: {status['total_runtime']}")

        print(f"\n🎯 Goals:")
        print(f"   Total: {status['goals']['total_goals']}")
        print(f"   Active: {status['goals']['active_goals']}")
        print(f"   Completed: {status['goals']['completed_goals']}")
        print(f"   Completion rate: {status['goals']['completion_rate']:.1%}")

        print(f"\n🧠 Memory:")
        print(f"   Short-term: {status['memory']['short_term_size']} experiences")
        print(f"   Long-term: {status['memory']['long_term_size']} experiences")
        print(f"   Knowledge base: {status['memory']['knowledge_items']} items")
        print(f"   Success rate: {status['memory']['success_rate']:.1%}")

        print(f"\n📚 Learning:")
        print(f"   Total insights: {status['learning']['total_insights']}")
        print(f"   High confidence: {status['learning']['high_confidence_insights']}")
        print(f"   Strategies learned: {status['learning']['strategies_learned']}")
        print(f"   Adaptation rules: {status['learning']['adaptation_rules']}")

        if status['ethics']['total_reviews'] > 0:
            print(f"\n⚖️  Ethics:")
            print(f"   Actions reviewed: {status['ethics']['total_reviews']}")
            print(f"   Approval rate: {status['ethics']['approval_rate']:.1%}")
            print(f"   Violations detected: {status['ethics']['violations_detected']}")

        ai.display_goal_progress()

        return

    # Run autonomous session
    try:
        print(f"\n⚡ Starting autonomous operation...")
        print(f"   The AI will work on its goals for {args.duration} minutes")
        print(f"   Press Ctrl+C to stop early\n")

        ai.run_autonomous_session(duration_minutes=args.duration)

        print("\n✨ Session completed successfully!")
        print(f"\nRun 'python run_agent.py --status' to see detailed status")

    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user")
        print("Saving state...")
        ai.save_state()
        print("✓ State saved successfully")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving state before exit...")
        ai.save_state()

    finally:
        print(f"\n👋 {args.name} signing off\n")


if __name__ == "__main__":
    main()
