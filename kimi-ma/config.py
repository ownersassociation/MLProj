import os

class Config:
    # Kimi K2.6 API Configuration
    KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
    KIMI_BASE_URL = "https://api.moonshot.ai/v1"
    KIMI_MODEL = "kimi-k2.6"
    
    # Agent Swarm Parameters
    MAX_SUB_AGENTS = 300
    MAX_COORDINATED_STEPS = 4000
    TEMPERATURE = 0.3
    TOP_P = 0.95
    
    # SLA Thresholds
    SLA_CRITICAL_HOURS = 2
    SLA_HIGH_HOURS = 4
    SLA_NORMAL_HOURS = 24
    
    # Routing Rules
    TEAM_MAP = {
        "Billing": "Revenue Operations",
        "Technical": "Engineering Support", 
        "Service": "Customer Success",
        "Urgent/Escalation": "Executive Response"
    }
    
    # Dashboard
    REFRESH_INTERVAL_SECONDS = 30