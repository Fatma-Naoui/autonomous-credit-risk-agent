import os
import sys
import json
import hashlib
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, TypedDict

from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from agent_core.agent.llm_code_generator import CodeGeneratorAgent
from agent_core.agent.debugger_agent import DebuggerAgent
from agent_core.agent.executor_agent import ExecutorAgent
from agent_core.agent.orchestrator_agent import OrchestratorAgent, AgentMessage, AgentRole
from agent_core.prompts import prompts

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "pipeline_cache.json"
PIPELINE_FILE = BASE_DIR / "generated_pipeline.py"
ARTIFACTS_DIR = BASE_DIR / "models"
PROMPT_HASH_FILE = BASE_DIR / "pipeline_prompt_hash.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    arguments: Dict[str, Any]
    generation_status: str
    debug_status: str
    execution_status: str
    error_message: str
    artifacts: list
    artifacts_path: str
    pipeline_ready: bool
    final_result: Dict[str, Any]
    execution_strategy: Dict[str, Any]
    execution_plan: Dict[str, Any]
    agent_messages: list
    orchestrator_decision: Dict[str, Any]


class AutonomousPipelineWorkflow:
    def __init__(self):
        self.code_gen_agent = CodeGeneratorAgent()
        self.debugger_agent = DebuggerAgent()
        self.orchestrator_agent = OrchestratorAgent()
        self.executor_agent = ExecutorAgent()
        self.prompt_hash_file = PROMPT_HASH_FILE
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(PipelineState)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("debug", self._debug_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "generate")
        workflow.add_conditional_edges("generate", self._should_continue_after_generate, {
            "debug": "debug",
            "error": "finalize"
        })
        workflow.add_conditional_edges("debug", self._should_continue_after_debug, {
            "execute": "execute",
            "error": "finalize"
        })
        workflow.add_edge("execute", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    async def run_pipeline(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if "artifacts_path" not in arguments:
            arguments["artifacts_path"] = str(ARTIFACTS_DIR)

        initial_state: PipelineState = {
            "arguments": arguments,
            "generation_status": "pending",
            "debug_status": "pending",
            "execution_status": "pending",
            "error_message": "",
            "artifacts": [],
            "artifacts_path": str(arguments["artifacts_path"]),
            "pipeline_ready": False,
            "final_result": {},
            "execution_strategy": {},
            "execution_plan": {},
            "agent_messages": [],
            "orchestrator_decision": {}
        }
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state.get("final_result", {})

    async def _plan_node(self, state: PipelineState) -> PipelineState:
        logger.info("Orchestrator planning execution strategy...")
        try:
            execution_strategy = self.orchestrator_agent.plan_execution_strategy(state["arguments"])
            self.orchestrator_agent.update_memory("current_run", {
                "arguments": state["arguments"],
                "strategy": execution_strategy,
                "timestamp": datetime.now().isoformat()
            })
            state["execution_strategy"] = execution_strategy
            state["orchestrator_decision"] = execution_strategy
            state["agent_messages"].append({
                "from_agent": AgentRole.ORCHESTRATOR.value,
                "to_agent": AgentRole.CODE_GENERATOR.value,
                "message_type": "execution_plan",
                "content": execution_strategy,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"Orchestrator decided: {execution_strategy.get('decision', 'unknown')}")
        except Exception as e:
            logger.error(f"Orchestrator planning failed: {e}")
            state["execution_strategy"] = {
                "decision": "standard_sequential",
                "reasoning": f"Fallback due to planning error: {e}",
                "confidence": 0.3
            }
        return state

    async def _generate_node(self, state: PipelineState) -> PipelineState:
        try:
            strategy = state["execution_strategy"]
            force_regenerate = strategy.get("decision") == "aggressive_fast_track"
            if self._should_regenerate_code() or force_regenerate:
                enhanced_arguments = {
                    **state["arguments"],
                    "execution_strategy": strategy,
                    "orchestrator_guidance": strategy.get("reasoning", "")
                }
                await self.code_gen_agent.generate(enhanced_arguments)
                self._mark_code_generation()
                self._update_agent_memory("code_generation", "success")
            state["generation_status"] = "success"
            state["agent_messages"].append({
                "from_agent": AgentRole.CODE_GENERATOR.value,
                "to_agent": AgentRole.DEBUGGER.value,
                "message_type": "generation_complete",
                "content": {"status": "success", "strategy": state["execution_strategy"]},
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            self._update_agent_memory("code_generation", f"error: {e}")
            state["generation_status"] = "error"
            state["error_message"] = str(e)
        return state

    async def _debug_node(self, state: PipelineState) -> PipelineState:
        try:
            strategy = state["execution_strategy"]
            if strategy.get("decision") == "aggressive_fast_track":
                logger.info("Skipping debug due to aggressive strategy")
                state["debug_status"] = "skipped"
                return state
            if self._should_run_debugger():
                enhanced_arguments = {
                    **state["arguments"],
                    "execution_strategy": strategy,
                    "orchestrator_messages": state["agent_messages"]
                }
                result = await self.debugger_agent.debug_pipeline(enhanced_arguments)
                if not result or result.get("status") == "error":
                    state["debug_status"] = "error"
                    state["error_message"] = result.get("message", "Debug failed")
                    self._update_agent_memory("debugging", state["error_message"])
                else:
                    state["debug_status"] = "success"
                    self._mark_debugger_run()
                    self._update_agent_memory("debugging", "success")
            else:
                state["debug_status"] = "success"
            state["agent_messages"].append({
                "from_agent": AgentRole.DEBUGGER.value,
                "to_agent": AgentRole.EXECUTOR.value,
                "message_type": "debug_complete",
                "content": {"status": state["debug_status"], "error": state.get("error_message", "")},
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            state["debug_status"] = "error"
            state["error_message"] = str(e)
            self._update_agent_memory("debugging", f"error: {e}")
        return state

    async def _execute_node(self, state: PipelineState) -> PipelineState:
        logger.info("Executor agent taking control...")
        try:
            args = state["arguments"]
            required = ["train_path", "test_path", "label_column"]
            if not all(k in args for k in required):
                state["execution_status"] = "error"
                state["error_message"] = f"Missing required arguments: {required}"
                return state

            dataset_hash = self._get_dataset_hash(args["train_path"], args["test_path"], args["label_column"])
            parent_artifacts_dir = Path(args.get("artifacts_path", ARTIFACTS_DIR))
            dataset_artifacts = parent_artifacts_dir / dataset_hash
            dataset_artifacts.mkdir(parents=True, exist_ok=True)

            execution_context = {
                "debugger_status": state["debug_status"],
                "orchestrator_strategy": state["execution_strategy"],
                "agent_messages": state["agent_messages"],
                "run_mode": state["execution_strategy"].get("decision", "standard_sequential"),
                "risk_flags": self._assess_risk_flags(state)
            }

            execution_plan = self.executor_agent.assess_execution_path(execution_context)
            state["execution_plan"] = execution_plan
            logger.info(f"Executor decided: {execution_plan.get('decision', 'unknown')}")

            cache = self._load_json(CACHE_FILE)
            if dataset_hash in cache and execution_plan.get("decision") != "recovery_mode_execution":
                logger.info("Executor using cached artifacts.")
                cached_result = cache[dataset_hash]
                state["artifacts"] = cached_result["artifacts"]
                state["artifacts_path"] = cached_result["artifacts_path"]
                state["execution_status"] = "success"
                return state

            cmd = [
                sys.executable,
                str(PIPELINE_FILE.resolve()),
                str(Path(args["train_path"]).resolve()),
                str(Path(args["test_path"]).resolve()),
                args["label_column"],
                str(dataset_artifacts.resolve())
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            (dataset_artifacts / "stdout.log").write_text(stdout.decode())
            (dataset_artifacts / "stderr.log").write_text(stderr.decode())

            artifacts = [f.name for f in dataset_artifacts.glob("*.*")]
            state["artifacts"] = artifacts
            state["artifacts_path"] = str(dataset_artifacts)
            state["execution_status"] = "success" if proc.returncode == 0 else "error"
            if proc.returncode != 0:
                state["error_message"] = stderr.decode()
            else:
                state["final_result"] = {"stdout": stdout.decode()}

            cache[dataset_hash] = {
                "artifacts": artifacts,
                "artifacts_path": str(dataset_artifacts),
                "execution_plan": execution_plan,
                "timestamp": datetime.now().isoformat()
            }
            self._save_json(CACHE_FILE, cache)
        except Exception as e:
            logger.error(f"Executor agent failed: {e}")
            state["execution_status"] = "error"
            state["error_message"] = str(e)
        return state

    async def _finalize_node(self, state: PipelineState) -> PipelineState:
        logger.info("Finalizing pipeline with agent summary...")
        state["pipeline_ready"] = all(
            state.get(s) in ["success", "skipped"] for s in ["generation_status", "debug_status", "execution_status"]
        )
        final_summary = {
            "orchestrator_strategy": state.get("orchestrator_decision", {}),
            "execution_plan": state.get("execution_plan", {}),
            "agent_messages_count": len(state.get("agent_messages", [])),
            "pipeline_ready": state["pipeline_ready"],
            "total_agents_involved": 4
        }
        if "final_result" not in state:
            state["final_result"] = {}
        state["final_result"]["agent_summary"] = final_summary
        logger.info(f"Pipeline completed with {len(state.get('agent_messages', []))} agent interactions")
        return state

    def _assess_risk_flags(self, state: PipelineState) -> list:
        risk_flags = []
        if state.get("debug_status") == "error":
            risk_flags.append("debug_failed")
        if state.get("generation_status") == "error":
            risk_flags.append("generation_failed")
        if state["execution_strategy"].get("confidence", 1.0) < 0.5:
            risk_flags.append("low_confidence_strategy")
        return risk_flags

    def _get_prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _should_regenerate_code(self) -> bool:
        prompt = prompts.autogluon_pipeline_generator_prompt.template
        stored = self._load_json(self.prompt_hash_file)
        return stored.get("code_prompt_hash") != self._get_prompt_hash(prompt) or not PIPELINE_FILE.exists()

    def _mark_code_generation(self):
        prompt = prompts.autogluon_pipeline_generator_prompt.template
        data = self._load_json(self.prompt_hash_file)
        data["code_prompt_hash"] = self._get_prompt_hash(prompt)
        data["timestamp"] = datetime.now().isoformat()
        self._save_json(self.prompt_hash_file, data)

    def _should_run_debugger(self) -> bool:
        prompt = prompts.autogluon_pipeline_debugger_prompt.template
        stored = self._load_json(self.prompt_hash_file)
        return stored.get("debug_prompt_hash") != self._get_prompt_hash(prompt)

    def _mark_debugger_run(self):
        prompt = prompts.autogluon_pipeline_debugger_prompt.template
        data = self._load_json(self.prompt_hash_file)
        data["debug_prompt_hash"] = self._get_prompt_hash(prompt)
        data["debug_timestamp"] = datetime.now().isoformat()
        self._save_json(self.prompt_hash_file, data)

    def _get_dataset_hash(self, train_path: str, test_path: str, label_column: str) -> str:
        def hash_file(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()
        combined = f"{hash_file(train_path)}_{hash_file(test_path)}_{label_column}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _should_continue_after_generate(self, state: PipelineState) -> str:
        return "debug" if state["generation_status"] == "success" else "error"

    def _should_continue_after_debug(self, state: PipelineState) -> str:
        return "execute" if state["debug_status"] in ["success", "skipped"] else "error"

    def _load_json(self, file: Path) -> Dict[str, Any]:
        if file.exists():
            try:
                with open(file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
        return {}

    def _save_json(self, file: Path, data: Dict[str, Any]):
        try:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {file}: {e}")

    def _update_agent_memory(self, key: str, value: str):
        cache = self._load_json(CACHE_FILE)
        if "agent_memory" not in cache:
            cache["agent_memory"] = {}
        cache["agent_memory"][key] = {
            "status": value,
            "timestamp": datetime.now().isoformat()
        }
        self._save_json(CACHE_FILE, cache)
