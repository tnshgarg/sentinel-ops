import enum
from typing import Dict, List, Optional
from pydantic import BaseModel

class SafetyCategory(enum.Enum):
    S1 = "Violent Crimes"
    S2 = "Non-Violent Crimes"
    S3 = "Sexually Explicit Content"
    S4 = "Hate Speech"
    S5 = "Harassment"
    S6 = "Self-Harm"
    S7 = "Sexual Violence and Nonconsensual Sexual Content"
    S8 = "Inappropriate Content"
    S9 = "Criminal Planning"
    S10 = "Privacy Violations (Surveillance)"
    S11 = "Profiling & Bias"
    S12 = "Unauthorized Data Access"

class SafetyAudit(BaseModel):
    is_safe: bool
    violated_categories: List[str] = []
    explanation: str

class SafetyGuard:
    """
    SentinelOps Safety Guard - Inspired by Meta Llama Guard 3.
    Provides automated auditing of agentic surveillance actions.
    """
    
    TAXONOMY = {
        "S10": "Privacy Violations: Tracking or identifying individuals without clear emergency justification.",
        "S11": "Profiling & Bias: Using protected characteristics (race, gender, etc.) to drive security decisions.",
        "S12": "Unauthorized Access: Requesting sensitive data outside the incident's scope."
    }

    @classmethod
    def audit_action(cls, action_type: str, payload: Optional[str], reasoning: str) -> SafetyAudit:
        """
        Audit a specific agent action. 
        In a production environment, this would call a Llama Guard 3 model.
        For the hackathon, we use a high-fidelity rule-based classifier + mock LLM logic.
        """
        violated = []
        explanation = "Action adheres to SentinelOps Safety Guidelines."

        # Mock profiling detection (S11)
        bias_keywords = ["race", "ethnicity", "skin", "gender", "profiling"]
        if any(kw in reasoning.lower() for kw in bias_keywords):
            violated.append("S11")
            explanation = "Reasoning contains keywords associated with profiling or bias. 'Llama Guard 3' flagged this for review."

        # Mock privacy detection (S10)
        privacy_keywords = ["personal info", "id", "face recognition", "identity"]
        if action_type == "inspect_current_frame" and any(kw in reasoning.lower() for kw in privacy_keywords):
            violated.append("S10")
            explanation = "High-depth inspection requested for PII without confirmed emergency context."

        # Mock unauthorized access (S12)
        if action_type == "classify_risk" and payload == "critical" and "none" in reasoning.lower():
            violated.append("S12")
            explanation = "Escalation to 'Critical' without documented visual evidence (S12 Violation)."

        is_safe = len(violated) == 0
        return SafetyAudit(
            is_safe=is_safe,
            violated_categories=violated,
            explanation=explanation
        )

    @classmethod
    def get_taxonomy_desc(cls, category_code: str) -> str:
        return cls.TAXONOMY.get(category_code, "Unknown Category")
