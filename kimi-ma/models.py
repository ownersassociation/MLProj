from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import random


@dataclass
class ComplaintCase:
    id: str
    description: str
    category: str  # Billing, Technical, Service, Urgent
    priority: int  # 1-5, 5 = critical
    status: str    # New, Classified, Routed, Resolved, Escalated
    sla_hours: int
    created_at: datetime
    resolved_at: Optional[datetime] = None
    assigned_team: Optional[str] = None
    sentiment: str = "neutral"  # angry, frustrated, neutral, satisfied
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "priority": self.priority,
            "status": self.status,
            "sla_hours": self.sla_hours,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "assigned_team": self.assigned_team,
            "sentiment": self.sentiment
        }


class CaseStreamSimulator:
    """Simulates incoming complaint stream for demo/testing."""
    
    CATEGORIES = ["Billing", "Technical", "Service", "Urgent/Escalation"]
    TEAMS = {
        "Billing": "Revenue Operations",
        "Technical": "Engineering Support",
        "Service": "Customer Success",
        "Urgent/Escalation": "Executive Response"
    }
    DESCRIPTIONS = {
        "Billing": [
            "Charged twice this month, need refund immediately",
            "Invoice shows incorrect tax calculation",
            "Payment failed but money was deducted",
            "Want to dispute last transaction"
        ],
        "Technical": [
            "API returning 500 errors since morning",
            "Cannot login to dashboard after update",
            "Integration webhook not firing",
            "Mobile app crashes on checkout"
        ],
        "Service": [
            "Waited 30 minutes on hold, very frustrated",
            "Agent promised callback but never called",
            "Need to upgrade my plan urgently",
            "Confused about new feature rollout"
        ],
        "Urgent/Escalation": [
            "Data breach suspected in my account",
            "CEO escalated - enterprise contract at risk",
            "Regulatory complaint filed with consumer board",
            "Threatening to sue for service outage"
        ]
    }
    
    @classmethod
    def generate_week(cls, week_num: int, volume: int = 50) -> List[ComplaintCase]:
        cases = []
        weights = [0.35, 0.30, 0.25, 0.10]  # Category distribution
        
        for i in range(volume):
            cat = random.choices(cls.CATEGORIES, weights=weights)[0]
            desc = random.choice(cls.DESCRIPTIONS[cat])
            priority = 5 if cat == "Urgent/Escalation" else random.randint(1, 4)
            sla = 2 if priority == 5 else (4 if priority >= 3 else 24)
            
            case = ComplaintCase(
                id=f"W{week_num}-{i+1:04d}",
                description=desc,
                category=cat,
                priority=priority,
                status=random.choice(["Resolved", "Routed", "Classified", "New"]),
                sla_hours=sla,
                created_at=datetime.now() - timedelta(days=random.randint(0, 7)),
                assigned_team=cls.TEAMS[cat],
                sentiment=random.choice(["angry", "frustrated", "neutral", "satisfied"])
            )
            cases.append(case)
        return cases