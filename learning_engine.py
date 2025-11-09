"""
Learning Engine for Autonomous AI Agent

This module enables the AI to learn from experiences, adapt strategies,
and improve its performance over time.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class LearningInsight:
    """Represents a learned insight"""
    id: str
    category: str
    insight: str
    confidence: float  # 0.0 to 1.0
    evidence_count: int
    created_at: datetime
    last_reinforced: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'category': self.category,
            'insight': self.insight,
            'confidence': self.confidence,
            'evidence_count': self.evidence_count,
            'created_at': self.created_at.isoformat(),
            'last_reinforced': self.last_reinforced.isoformat()
        }


@dataclass
class PerformanceMetric:
    """Tracks performance over time"""
    metric_name: str
    timestamp: datetime
    value: float
    context: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_name': self.metric_name,
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'context': self.context
        }


class LearningEngine:
    """
    Learning engine that enables the AI to improve through experience.
    Implements various learning strategies and adaptation mechanisms.
    """

    def __init__(self):
        self.insights: List[LearningInsight] = []
        self.performance_history: List[PerformanceMetric] = []
        self.strategy_effectiveness: Dict[str, Dict[str, Any]] = {}
        self.adaptation_rules: List[Dict[str, Any]] = []

    def learn_from_experience(self, context: str, action: str, outcome: str,
                             success: bool, metadata: Dict[str, Any] = None):
        """
        Learn from an experience by extracting insights and updating knowledge.
        This is a key method for continuous learning.
        """
        metadata = metadata or {}

        # Extract insights based on success/failure
        if success:
            insight = self._extract_success_insight(context, action, outcome, metadata)
        else:
            insight = self._extract_failure_insight(context, action, outcome, metadata)

        if insight:
            self._add_or_reinforce_insight(insight)

        # Update strategy effectiveness
        if 'strategy' in metadata:
            self._update_strategy_effectiveness(
                metadata['strategy'],
                success,
                metadata.get('goal_type')
            )

        # Record performance metric
        self.performance_history.append(PerformanceMetric(
            metric_name='success_rate',
            timestamp=datetime.now(),
            value=1.0 if success else 0.0,
            context=context
        ))

    def _extract_success_insight(self, context: str, action: str,
                                 outcome: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract insight from successful experience"""
        # This is a simplified version - in a real implementation,
        # this would use more sophisticated pattern recognition
        return f"When {context}, action '{action}' leads to positive outcome"

    def _extract_failure_insight(self, context: str, action: str,
                                 outcome: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract insight from failed experience"""
        return f"Avoid action '{action}' in context {context} as it leads to: {outcome}"

    def _add_or_reinforce_insight(self, insight_text: str):
        """Add a new insight or reinforce existing one"""
        # Check if similar insight exists
        for insight in self.insights:
            if insight.insight == insight_text:
                # Reinforce existing insight
                insight.evidence_count += 1
                insight.confidence = min(1.0, insight.confidence + 0.1)
                insight.last_reinforced = datetime.now()
                return

        # Add new insight
        from uuid import uuid4
        new_insight = LearningInsight(
            id=str(uuid4()),
            category='general',
            insight=insight_text,
            confidence=0.5,
            evidence_count=1,
            created_at=datetime.now(),
            last_reinforced=datetime.now()
        )
        self.insights.append(new_insight)

    def _update_strategy_effectiveness(self, strategy: str, success: bool,
                                      goal_type: Optional[str] = None):
        """Update effectiveness metrics for a strategy"""
        key = f"{strategy}_{goal_type}" if goal_type else strategy

        if key not in self.strategy_effectiveness:
            self.strategy_effectiveness[key] = {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'effectiveness': 0.0
            }

        stats = self.strategy_effectiveness[key]
        stats['attempts'] += 1

        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1

        stats['effectiveness'] = stats['successes'] / stats['attempts']

    def get_relevant_insights(self, context: str, min_confidence: float = 0.6) -> List[str]:
        """Get insights relevant to a given context"""
        relevant = []
        for insight in self.insights:
            if (insight.confidence >= min_confidence and
                context.lower() in insight.insight.lower()):
                relevant.append(insight.insight)
        return relevant

    def recommend_strategy(self, goal_type: str, context: str) -> Optional[str]:
        """
        Recommend the most effective strategy based on learned experience.
        This enables the AI to continuously improve its approach.
        """
        # Find strategies for this goal type
        relevant_strategies = {
            k: v for k, v in self.strategy_effectiveness.items()
            if goal_type in k and v['attempts'] >= 3  # Require minimum data
        }

        if not relevant_strategies:
            return None

        # Return strategy with highest effectiveness
        best_strategy = max(relevant_strategies.items(),
                          key=lambda x: x[1]['effectiveness'])
        return best_strategy[0].split('_')[0]  # Extract strategy name

    def analyze_performance_trend(self, metric_name: str,
                                 window_size: int = 10) -> Dict[str, Any]:
        """Analyze performance trend for self-improvement"""
        relevant_metrics = [
            m for m in self.performance_history[-window_size:]
            if m.metric_name == metric_name
        ]

        if not relevant_metrics:
            return {'trend': 'unknown', 'average': 0.0}

        values = [m.value for m in relevant_metrics]
        average = sum(values) / len(values)

        # Simple trend detection
        if len(values) >= 3:
            recent_avg = sum(values[-3:]) / 3
            older_avg = sum(values[:-3]) / len(values[:-3]) if len(values) > 3 else average

            if recent_avg > older_avg * 1.1:
                trend = 'improving'
            elif recent_avg < older_avg * 0.9:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'trend': trend,
            'average': average,
            'recent_average': sum(values[-3:]) / min(3, len(values)),
            'data_points': len(values)
        }

    def create_adaptation_rule(self, condition: str, action: str, priority: int = 3):
        """
        Create an adaptation rule that guides future behavior.
        This allows the AI to encode learned strategies.
        """
        rule = {
            'id': len(self.adaptation_rules) + 1,
            'condition': condition,
            'action': action,
            'priority': priority,
            'created_at': datetime.now().isoformat(),
            'times_applied': 0
        }
        self.adaptation_rules.append(rule)

    def get_applicable_rules(self, context: str) -> List[Dict[str, Any]]:
        """Get adaptation rules applicable to current context"""
        applicable = []
        for rule in self.adaptation_rules:
            if context.lower() in rule['condition'].lower():
                applicable.append(rule)

        # Sort by priority
        applicable.sort(key=lambda r: r['priority'], reverse=True)
        return applicable

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of learning progress"""
        return {
            'total_insights': len(self.insights),
            'high_confidence_insights': len([i for i in self.insights if i.confidence >= 0.8]),
            'strategies_learned': len(self.strategy_effectiveness),
            'adaptation_rules': len(self.adaptation_rules),
            'performance_records': len(self.performance_history),
            'average_success_rate': sum(m.value for m in self.performance_history[-50:]) / min(50, len(self.performance_history)) if self.performance_history else 0.0
        }

    def save_to_file(self, filename: str):
        """Save learning data to file"""
        data = {
            'insights': [i.to_dict() for i in self.insights],
            'performance_history': [m.to_dict() for m in self.performance_history],
            'strategy_effectiveness': self.strategy_effectiveness,
            'adaptation_rules': self.adaptation_rules
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str):
        """Load learning data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                # Load insights
                self.insights = [
                    LearningInsight(
                        id=i['id'],
                        category=i['category'],
                        insight=i['insight'],
                        confidence=i['confidence'],
                        evidence_count=i['evidence_count'],
                        created_at=datetime.fromisoformat(i['created_at']),
                        last_reinforced=datetime.fromisoformat(i['last_reinforced'])
                    )
                    for i in data.get('insights', [])
                ]
                # Load performance history
                self.performance_history = [
                    PerformanceMetric(
                        metric_name=m['metric_name'],
                        timestamp=datetime.fromisoformat(m['timestamp']),
                        value=m['value'],
                        context=m['context']
                    )
                    for m in data.get('performance_history', [])
                ]
                self.strategy_effectiveness = data.get('strategy_effectiveness', {})
                self.adaptation_rules = data.get('adaptation_rules', [])
        except FileNotFoundError:
            pass
