import os

class Config:
    # LOCAL vLLM Configuration (No cloud API)
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy-vllm-no-auth")
    LOCAL_MODEL_NAME = "kimi-k2.6"  # Must match --served-model-name
    
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