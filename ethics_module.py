"""
Ethics Module for Autonomous AI Agent

This module implements ethical constraints, safety boundaries, and
responsible decision-making for the autonomous AI.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime


class EthicalPrinciple(Enum):
    """Core ethical principles the AI must follow"""
    BENEFICENCE = "beneficence"  # Do good, help others
    NON_MALEFICENCE = "non_maleficence"  # Do no harm
    AUTONOMY = "autonomy"  # Respect human autonomy and choice
    JUSTICE = "justice"  # Be fair and equitable
    TRANSPARENCY = "transparency"  # Be honest and transparent
    ACCOUNTABILITY = "accountability"  # Take responsibility for actions
    PRIVACY = "privacy"  # Respect privacy and data protection
    SUSTAINABILITY = "sustainability"  # Consider long-term impacts


class RiskLevel(Enum):
    """Risk levels for actions"""
    SAFE = 1
    LOW_RISK = 2
    MEDIUM_RISK = 3
    HIGH_RISK = 4
    DANGEROUS = 5


@dataclass
class EthicalConstraint:
    """Represents an ethical constraint or rule"""
    id: str
    principle: EthicalPrinciple
    description: str
    severity: int  # 1-5, 5 being most severe
    check_function: Optional[Callable] = None


@dataclass
class ActionReview:
    """Result of ethical review for an action"""
    action: str
    approved: bool
    risk_level: RiskLevel
    violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    reasoning: str
    timestamp: datetime


class EthicsModule:
    """
    Ethics module that ensures the AI operates within ethical boundaries
    and makes responsible decisions.
    """

    def __init__(self):
        self.constraints: List[EthicalConstraint] = []
        self.forbidden_actions: List[str] = []
        self.require_approval_threshold: RiskLevel = RiskLevel.HIGH_RISK
        self.review_history: List[ActionReview] = []

        # Initialize default constraints
        self._initialize_default_constraints()

    def _initialize_default_constraints(self):
        """Set up default ethical constraints"""

        # Non-maleficence constraints
        self.forbidden_actions.extend([
            "harm humans",
            "damage property",
            "steal",
            "deceive for personal gain",
            "manipulate people",
            "violate privacy",
            "spread misinformation",
            "engage in illegal activities",
            "create malware or harmful code",
            "access unauthorized systems"
        ])

        # Add ethical guidelines
        self.add_constraint(EthicalConstraint(
            id="no_harm",
            principle=EthicalPrinciple.NON_MALEFICENCE,
            description="Must not take actions that could harm humans or their interests",
            severity=5
        ))

        self.add_constraint(EthicalConstraint(
            id="transparency",
            principle=EthicalPrinciple.TRANSPARENCY,
            description="Must be transparent about being an AI and about capabilities/limitations",
            severity=4
        ))

        self.add_constraint(EthicalConstraint(
            id="respect_autonomy",
            principle=EthicalPrinciple.AUTONOMY,
            description="Must respect human decision-making and not override human choices",
            severity=4
        ))

        self.add_constraint(EthicalConstraint(
            id="privacy",
            principle=EthicalPrinciple.PRIVACY,
            description="Must protect personal data and respect privacy",
            severity=5
        ))

        self.add_constraint(EthicalConstraint(
            id="fairness",
            principle=EthicalPrinciple.JUSTICE,
            description="Must treat all people fairly without discrimination",
            severity=4
        ))

        self.add_constraint(EthicalConstraint(
            id="sustainability",
            principle=EthicalPrinciple.SUSTAINABILITY,
            description="Should consider long-term consequences and sustainability",
            severity=3
        ))

    def add_constraint(self, constraint: EthicalConstraint):
        """Add an ethical constraint"""
        self.constraints.append(constraint)

    def review_action(self, action: str, context: Dict[str, Any]) -> ActionReview:
        """
        Review an action for ethical compliance before execution.
        This is a critical safety mechanism.
        """
        violations = []
        warnings = []
        recommendations = []
        risk_level = RiskLevel.SAFE

        # Check for forbidden actions
        action_lower = action.lower()
        for forbidden in self.forbidden_actions:
            if forbidden in action_lower:
                violations.append(f"Action contains forbidden element: {forbidden}")
                risk_level = RiskLevel.DANGEROUS

        # Check financial actions for ethical concerns
        if "money" in action_lower or "financial" in action_lower or "profit" in action_lower:
            warnings.append("Financial action detected - ensure it's legal and ethical")
            risk_level = max(risk_level, RiskLevel.MEDIUM_RISK)
            recommendations.append("Verify legality and ethical implications of financial activities")

        # Check for manipulation or deception
        if any(word in action_lower for word in ["manipulate", "deceive", "trick", "exploit"]):
            violations.append("Potential manipulation or deception detected")
            risk_level = RiskLevel.DANGEROUS

        # Check for privacy concerns
        if any(word in action_lower for word in ["personal data", "private information", "surveillance"]):
            warnings.append("Privacy-sensitive action detected")
            risk_level = max(risk_level, RiskLevel.HIGH_RISK)
            recommendations.append("Ensure proper consent and data protection measures")

        # Check for automation of critical decisions
        if any(word in action_lower for word in ["medical", "legal", "financial advice"]):
            warnings.append("Critical domain detected - human oversight recommended")
            risk_level = max(risk_level, RiskLevel.HIGH_RISK)
            recommendations.append("Recommend human expert review for critical decisions")

        # Determine if action is approved
        approved = len(violations) == 0 and risk_level.value < RiskLevel.DANGEROUS.value

        reasoning = self._generate_reasoning(violations, warnings, recommendations, approved)

        review = ActionReview(
            action=action,
            approved=approved,
            risk_level=risk_level,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            reasoning=reasoning,
            timestamp=datetime.now()
        )

        self.review_history.append(review)
        return review

    def _generate_reasoning(self, violations: List[str], warnings: List[str],
                           recommendations: List[str], approved: bool) -> str:
        """Generate reasoning for the ethical decision"""
        if approved:
            if warnings:
                return f"Action approved with {len(warnings)} warning(s). Proceed with caution."
            return "Action approved. No ethical concerns detected."
        else:
            return f"Action rejected due to {len(violations)} ethical violation(s). Cannot proceed."

    def check_goal_ethics(self, goal_description: str, goal_type: str) -> Dict[str, Any]:
        """Check if a goal is ethically acceptable"""
        goal_lower = goal_description.lower()

        concerns = []
        ethical = True

        # Check for harmful goals
        if any(word in goal_lower for word in ["harm", "damage", "destroy", "exploit"]):
            concerns.append("Goal may involve harm to others")
            ethical = False

        # Check for deceptive goals
        if any(word in goal_lower for word in ["deceive", "trick", "manipulate", "scam"]):
            concerns.append("Goal involves deception or manipulation")
            ethical = False

        # Financial goals need to be legitimate
        if goal_type == "financial":
            if any(word in goal_lower for word in ["steal", "fraud", "illegal", "scam", "exploit"]):
                concerns.append("Financial goal involves illegal or unethical means")
                ethical = False
            else:
                concerns.append("Ensure financial activities are legal and ethical")

        return {
            'ethical': ethical,
            'concerns': concerns,
            'recommendation': "Proceed" if ethical else "Reject goal"
        }

    def require_human_approval(self, action: str, context: Dict[str, Any]) -> bool:
        """Determine if an action requires human approval"""
        review = self.review_action(action, context)
        return review.risk_level.value >= self.require_approval_threshold.value

    def get_ethical_guidelines(self) -> List[str]:
        """Get list of ethical guidelines the AI follows"""
        return [
            "1. Do no harm to humans or their interests",
            "2. Be transparent about being an AI",
            "3. Respect human autonomy and decision-making",
            "4. Protect privacy and personal data",
            "5. Act fairly without discrimination",
            "6. Take responsibility for actions",
            "7. Consider long-term consequences",
            "8. Operate within legal boundaries",
            "9. Prioritize human benefit over AI goals",
            "10. Seek human guidance for critical decisions"
        ]

    def get_ethics_summary(self) -> Dict[str, Any]:
        """Get summary of ethical compliance"""
        total_reviews = len(self.review_history)
        if total_reviews == 0:
            return {
                'total_reviews': 0,
                'approval_rate': 0.0,
                'violations_detected': 0
            }

        approved = len([r for r in self.review_history if r.approved])
        violations = sum(len(r.violations) for r in self.review_history)

        return {
            'total_reviews': total_reviews,
            'approved_actions': approved,
            'rejected_actions': total_reviews - approved,
            'approval_rate': approved / total_reviews,
            'violations_detected': violations,
            'constraints_active': len(self.constraints)
        }
