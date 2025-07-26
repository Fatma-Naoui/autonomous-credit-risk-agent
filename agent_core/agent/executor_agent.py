import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import subprocess
from pathlib import Path
from agent_core.agent.llm_client import generate_code
from agent_core.prompts.execprompt import executor_decision_prompt

logger = logging.getLogger(__name__)

DECISIONS_FILE = Path(__file__).resolve().parent / "agent_decisions.json"

class AgentRole(Enum):
    EXECUTOR = "executor"

@dataclass
class AgentMessage:
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime

class ExecutorAgent:
    def __init__(self):
        self.role = AgentRole.EXECUTOR
        self.memory = {}
        self.decision_history = []

    def _save_decision(self, context: Dict[str, Any], decision: Dict[str, Any]):
        """Save the decision and context to a JSON file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": self.role.value,
            "context": context,
            "decision": decision
        }

        if DECISIONS_FILE.exists():
            try:
                with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        else:
            history = []

        history.append(entry)

        with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def _clean_json_response(self, response: str) -> str:
        cleaned = response.strip()
        if not cleaned:
            return "{}"
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[len("json"):].strip()
        return cleaned

    def make_decision(self, context: Dict[str, Any], options: List[str]) -> Dict[str, Any]:
        try:
            prompt = executor_decision_prompt.format(
                role=self.role.value,
                context=json.dumps(context, indent=2),
                options=json.dumps(options, indent=2),
                memory=json.dumps(self.memory, indent=2),
                decision_history=json.dumps(self.decision_history[-3:], indent=2)
            )
            response = generate_code(prompt)
            logger.info(f"Raw Executor Response: {response!r}")
            cleaned_response = self._clean_json_response(response)
            decision = json.loads(cleaned_response)

            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "decision": decision,
                "options": options
            })

            # Save to JSON file
            self._save_decision(context, decision)
            return decision
        except Exception as e:
            logger.error(f"Executor decision failed: {e}")
            fallback = {
                "decision": options[0],
                "reasoning": f"Fallback due to error: {e}",
                "confidence": 0.3,
                "risk_assessment": "high",
                "alternative_plan": "manual intervention"
            }
            self._save_decision(context, fallback)
            return fallback
