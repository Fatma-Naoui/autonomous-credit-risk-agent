import os
import psutil
import logging
import json
from datetime import datetime
from typing import Any, Dict, List
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from agent_core.prompts.orchprompt import orchestrator_decision_prompt
from agent_core.agent.llm_client import generate_code

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "models"
DECISIONS_FILE = Path(__file__).resolve().parent / "agent_decisions.json"

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    CODE_GENERATOR = "code_generator"
    DEBUGGER = "debugger"
    EXECUTOR = "executor"

@dataclass
class AgentMessage:
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime

class AgenticAgent:
    def __init__(self, role: AgentRole):
        self.role = role
        self.memory: Dict[str, Any] = {}
        self.decision_history: List[Dict[str, Any]] = []

    # ---------- Persistence ----------
    def _save_decision(self, context: Dict[str, Any], decision: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": self.role.value,
            "context": context,
            "decision": decision,
        }
        try:
            if DECISIONS_FILE.exists():
                with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
        except Exception as e:
            logger.error(f"Failed reading {DECISIONS_FILE}: {e}")
            history = []

        history.append(entry)
        try:
            with open(DECISIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed writing {DECISIONS_FILE}: {e}")

    # ---------- Utils ----------
    def _clean_json_response(self, response: str) -> str:
        cleaned = response.strip()
        if not cleaned:
            return "{}"
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[len("json"):].strip()
        return cleaned

    # ---------- Core ----------
    def make_decision(self, context: Dict[str, Any], options: List[str]) -> Dict[str, Any]:
        try:
            prompt_str = orchestrator_decision_prompt.format(
                role=self.role.value,
                context=json.dumps(context, indent=2),
                options=json.dumps(options, indent=2),
                memory=json.dumps(self.memory, indent=2),
                decision_history=json.dumps(self.decision_history[-3:], indent=2)
            )

            raw = generate_code(prompt_str)
            logger.info(f"[{self.role.value}] raw decision: {raw!r}")

            cleaned = self._clean_json_response(raw)
            decision = json.loads(cleaned)

            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "decision": decision,
                "options": options
            })

            self._save_decision(context, decision)
            return decision

        except Exception as e:
            logger.error(f"{self.role.value} decision failed: {e}")
            fallback = {
                "decision": options[0],
                "reasoning": f"Fallback due to: {e}",
                "confidence": 0.3,
                "risk_assessment": "high",
                "alternative_plan": "manual intervention"
            }
            self._save_decision(context, fallback)
            return fallback

    def update_memory(self, key: str, value: Any):
        self.memory[key] = value

class OrchestratorAgent(AgenticAgent):
    def __init__(self):
        super().__init__(AgentRole.ORCHESTRATOR)

    def plan_execution_strategy(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        context = {
            "dataset_size": self._estimate_dataset_size(arguments),
            "complexity": self._estimate_complexity(arguments),
            "system_resources": self._assess_system_resources(),
            "previous_runs": self.memory.get("similar_runs", [])
        }
        options = [
            "standard_sequential",
            "parallel_with_checkpoints",
            "conservative_with_validation",
            "aggressive_fast_track",
            "incremental_with_feedback"
        ]
        return self.make_decision(context, options)

    def _estimate_dataset_size(self, arguments: Dict[str, Any]) -> str:
        try:
            size = os.path.getsize(arguments["train_path"])
            if size > 100_000_000:
                return "large"
            elif size > 10_000_000:
                return "medium"
            else:
                return "small"
        except Exception:
            return "unknown"

    def _estimate_complexity(self, arguments: Dict[str, Any]) -> str:
        return "medium"

    def _assess_system_resources(self) -> Dict[str, Any]:
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
