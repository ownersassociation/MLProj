"""
Kimi K2.6 Multi-Agent Framework for Complaint Classification & Routing
Compatible with Moonshot AI's OpenAI-compatible API endpoint.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KimiSwarm")


@dataclass
class AgentTask:
    agent_name: str
    system_prompt: str
    user_prompt: str
    model: str = "kimi-k2.6"
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class AgentOutput:
    agent_name: str
    content: str
    timestamp: str
    tokens_used: Optional[int] = None


class KimiAgent:
    """Individual agent powered by Kimi K2.6."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.moonshot.ai/v1"):
        self.client = OpenAI(
            api_key=api_key or os.getenv("KIMI_API_KEY"),
            base_url=base_url
        )
        self.model = "kimi-k2.6"
    
    def execute(self, task: AgentTask) -> AgentOutput:
        """Execute a single agent task via Kimi API."""
        try:
            response = self.client.chat.completions.create(
                model=task.model,
                messages=[
                    {"role": "system", "content": task.system_prompt},
                    {"role": "user", "content": task.user_prompt}
                ],
                temperature=task.temperature,
                max_tokens=task.max_tokens,
                top_p=0.95
            )
            
            # Kimi K2.6 exposes reasoning content when in thinking mode
            reasoning = getattr(response.choices[0].message, 'reasoning', None)
            content = response.choices[0].message.content
            
            # If reasoning exists, prepend it for debugging
            if reasoning:
                logger.debug(f"[{task.agent_name}] Reasoning: {reasoning[:200]}...")
            
            return AgentOutput(
                agent_name=task.agent_name,
                content=content,
                timestamp=datetime.utcnow().isoformat(),
                tokens_used=response.usage.total_tokens if response.usage else None
            )
        except Exception as e:
            logger.error(f"Agent {task.agent_name} failed: {e}")
            return AgentOutput(
                agent_name=task.agent_name,
                content=f"Error: {str(e)}",
                timestamp=datetime.utcnow().isoformat()
            )


class KimiSwarmCoordinator:
    """
    Coordinator that mimics Kimi K2.6 Agent Swarm behavior.
    Decomposes dashboard generation into parallel sub-agent tasks.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.agent = KimiAgent(api_key=api_key)
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def _build_context(self, case_data: List[Dict], week_range: str) -> str:
        """Serialize recent case data for agent consumption."""
        context_lines = [
            f"Reporting Period: {week_range}",
            f"Total Cases: {len(case_data)}",
            f"Unique Categories: {len(set(c.get('category') for c in case_data))}",
            "---",
        ]
        for case in case_data[-20:]:  # Last 20 cases as sample
            context_lines.append(
                f"Case #{case.get('id')}: [{case.get('category')}] {case.get('description', '')[:100]}... "
                f"(Priority: {case.get('priority')}, Status: {case.get('status')}, "
                f"SLA: {case.get('sla_hours')}h)"
            )
        return "\n".join(context_lines)
    
    def generate_dashboard(self, case_data: List[Dict], week_range: str) -> Dict[str, Any]:
        """
        Parallel swarm execution:
        1. Analyst → Weekly Status Report
        2. Scorer → Application Score Card
        3. Strategist → Improvement Plan
        4. Advisor → Recommendations
        """
        context = self._build_context(case_data, week_range)
        
        tasks = [
            AgentTask(
                agent_name="Analyst",
                system_prompt=(
                    "You are a Senior Support Operations Analyst. "
                    "Generate a concise weekly status report with: Executive Summary, "
                    "Volume Trends, Category Breakdown, SLA Performance, and Top 3 Issues. "
                    "Use structured markdown. Be data-driven."
                ),
                user_prompt=f"Analyze this week's case data and generate the Weekly Status Report:\n\n{context}"
            ),
            AgentTask(
                agent_name="Scorer",
                system_prompt=(
                    "You are a QA Director evaluating an AI complaint routing system. "
                    "Score the application on: Classification Accuracy, Routing Precision, "
                    "SLA Compliance, Response Speed, Agent Utilization, Customer Satisfaction. "
                    "Output a JSON scorecard with current_score, target_score, gap, and trend."
                ),
                user_prompt=f"Evaluate system performance from this data:\n\n{context}\n\n"
                           f"Return ONLY valid JSON with keys: metrics (list), overall_score, grade."
            ),
            AgentTask(
                agent_name="Strategist",
                system_prompt=(
                    "You are a Digital Transformation Strategist. "
                    "Given performance gaps, create a prioritized improvement plan. "
                    "For each initiative provide: name, impact (High/Med/Low), effort (High/Med/Low), "
                    "timeline, and expected outcome."
                ),
                user_prompt=f"Based on this week's performance data, create an Improvement Plan:\n\n{context}"
            ),
            AgentTask(
                agent_name="Advisor",
                system_prompt=(
                    "You are an AI Solutions Architect specializing in customer service automation. "
                    "Provide 5 actionable recommendations leveraging Kimi K2.6 Agent Swarm capabilities. "
                    "Format as a numbered list with rationale and implementation steps."
                ),
                user_prompt=f"Recommend optimizations for this complaint routing system:\n\n{context}"
            )
        ]
        
        # Parallel execution (simulating Agent Swarm)
        results = {}
        future_to_task = {
            self.executor.submit(self.agent.execute, task): task 
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            output = future.result()
            results[task.agent_name] = output
            logger.info(f"Agent {task.agent_name} completed. Tokens: {output.tokens_used}")
        
        # Synthesis
        return {
            "week_range": week_range,
            "generated_at": datetime.utcnow().isoformat(),
            "weekly_status_report": results["Analyst"].content,
            "application_score_card": self._safe_parse_json(results["Scorer"].content),
            "improvement_plan": results["Strategist"].content,
            "recommendations": results["Advisor"].content,
            "agent_metadata": {name: {"tokens": out.tokens_used, "time": out.timestamp} 
                           for name, out in results.items()}
        }
    
    @staticmethod
    def _safe_parse_json(text: str) -> Any:
        """Extract JSON from agent response (handles markdown fences)."""
        try:
            # Find JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception:
            return {"raw": text, "parse_error": True}


# Singleton coordinator
_coordinator: Optional[KimiSwarmCoordinator] = None

def get_coordinator(api_key: Optional[str] = None) -> KimiSwarmCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = KimiSwarmCoordinator(api_key=api_key)
    return _coordinator