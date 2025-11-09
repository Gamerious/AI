"""
Memory System for Autonomous AI Agent

This module implements short-term and long-term memory for the AI agent,
allowing it to learn from experiences and improve over time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from collections import deque


@dataclass
class Experience:
    """Represents a single experience/memory"""
    id: str
    timestamp: datetime
    context: str
    action_taken: str
    outcome: str
    success: bool
    lessons_learned: List[str] = field(default_factory=list)
    related_goal_id: Optional[str] = None
    importance: int = 3  # 1-5 scale
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'action_taken': self.action_taken,
            'outcome': self.outcome,
            'success': self.success,
            'lessons_learned': self.lessons_learned,
            'related_goal_id': self.related_goal_id,
            'importance': self.importance,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Knowledge:
    """Represents learned knowledge"""
    topic: str
    content: str
    source: str
    confidence: float  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    related_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'topic': self.topic,
            'content': self.content,
            'source': self.source,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'related_topics': self.related_topics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Knowledge':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class MemorySystem:
    """Manages short-term and long-term memory for the AI agent"""

    def __init__(self, short_term_capacity: int = 100):
        # Short-term memory: recent experiences (working memory)
        self.short_term_memory: deque = deque(maxlen=short_term_capacity)

        # Long-term memory: important experiences and learned patterns
        self.long_term_memory: List[Experience] = []

        # Knowledge base: accumulated knowledge
        self.knowledge_base: Dict[str, Knowledge] = {}

        # Pattern recognition: learned patterns from experiences
        self.patterns: Dict[str, List[str]] = {}

        # Success/failure statistics
        self.statistics: Dict[str, int] = {
            'total_experiences': 0,
            'successful_actions': 0,
            'failed_actions': 0
        }

    def add_experience(self, experience: Experience):
        """Add a new experience to memory"""
        # Add to short-term memory
        self.short_term_memory.append(experience)

        # Add to long-term memory if important enough
        if experience.importance >= 4 or not experience.success:
            # Failures are important to learn from
            self.long_term_memory.append(experience)

        # Update statistics
        self.statistics['total_experiences'] += 1
        if experience.success:
            self.statistics['successful_actions'] += 1
        else:
            self.statistics['failed_actions'] += 1

        # Extract and store lessons learned
        for lesson in experience.lessons_learned:
            self._add_pattern(experience.context, lesson)

    def _add_pattern(self, context: str, pattern: str):
        """Add a learned pattern"""
        if context not in self.patterns:
            self.patterns[context] = []
        if pattern not in self.patterns[context]:
            self.patterns[context].append(pattern)

    def get_relevant_experiences(self, context: str, limit: int = 10) -> List[Experience]:
        """Get experiences relevant to a given context"""
        all_experiences = list(self.short_term_memory) + self.long_term_memory

        # Simple relevance scoring based on context matching
        relevant = []
        for exp in all_experiences:
            if context.lower() in exp.context.lower() or exp.context.lower() in context.lower():
                relevant.append(exp)

        # Sort by importance and recency
        relevant.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        return relevant[:limit]

    def get_learned_patterns(self, context: str) -> List[str]:
        """Get learned patterns for a specific context"""
        patterns = []
        for ctx, ptns in self.patterns.items():
            if context.lower() in ctx.lower() or ctx.lower() in context.lower():
                patterns.extend(ptns)
        return patterns

    def add_knowledge(self, knowledge: Knowledge):
        """Add knowledge to the knowledge base"""
        self.knowledge_base[knowledge.topic] = knowledge

    def get_knowledge(self, topic: str) -> Optional[Knowledge]:
        """Retrieve knowledge on a topic"""
        knowledge = self.knowledge_base.get(topic)
        if knowledge:
            knowledge.access_count += 1
            knowledge.last_accessed = datetime.now()
        return knowledge

    def search_knowledge(self, query: str) -> List[Knowledge]:
        """Search for knowledge related to a query"""
        results = []
        query_lower = query.lower()
        for knowledge in self.knowledge_base.values():
            if (query_lower in knowledge.topic.lower() or
                query_lower in knowledge.content.lower()):
                results.append(knowledge)

        # Sort by access count and confidence
        results.sort(key=lambda k: (k.access_count, k.confidence), reverse=True)
        return results

    def consolidate_memory(self):
        """
        Consolidate memories: move important short-term memories to long-term,
        strengthen frequently accessed knowledge, forget low-importance items
        """
        # Move important experiences from short-term to long-term
        for exp in list(self.short_term_memory):
            if exp.importance >= 4 and exp not in self.long_term_memory:
                self.long_term_memory.append(exp)

        # Keep only the most important long-term memories if too many
        if len(self.long_term_memory) > 1000:
            self.long_term_memory.sort(key=lambda x: x.importance, reverse=True)
            self.long_term_memory = self.long_term_memory[:1000]

    def get_success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.statistics['total_experiences']
        if total == 0:
            return 0.0
        return self.statistics['successful_actions'] / total

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the memory system"""
        return {
            'short_term_size': len(self.short_term_memory),
            'long_term_size': len(self.long_term_memory),
            'knowledge_items': len(self.knowledge_base),
            'patterns_learned': sum(len(p) for p in self.patterns.values()),
            'success_rate': self.get_success_rate(),
            'statistics': self.statistics
        }

    def save_to_file(self, filename: str):
        """Save memory to file"""
        data = {
            'short_term_memory': [exp.to_dict() for exp in self.short_term_memory],
            'long_term_memory': [exp.to_dict() for exp in self.long_term_memory],
            'knowledge_base': {k: v.to_dict() for k, v in self.knowledge_base.items()},
            'patterns': self.patterns,
            'statistics': self.statistics
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str):
        """Load memory from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.short_term_memory = deque(
                    [Experience.from_dict(exp) for exp in data.get('short_term_memory', [])],
                    maxlen=self.short_term_memory.maxlen
                )
                self.long_term_memory = [
                    Experience.from_dict(exp) for exp in data.get('long_term_memory', [])
                ]
                self.knowledge_base = {
                    k: Knowledge.from_dict(v)
                    for k, v in data.get('knowledge_base', {}).items()
                }
                self.patterns = data.get('patterns', {})
                self.statistics = data.get('statistics', self.statistics)
        except FileNotFoundError:
            pass  # No saved memory yet
