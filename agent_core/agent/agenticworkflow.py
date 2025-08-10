# agent_core/agent/agenticworkflow.py
import sys
import json
import hashlib
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, TypedDict
import os
import traceback
import subprocess

from langgraph.graph import StateGraph, END

# project path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from agent_core.agent.llm_code_generator import CodeGeneratorAgent
from agent_core.agent.debugger_agent import DebuggerAgent
from agent_core.agent.executor_agent import ExecutorAgent
from agent_core.agent.orchestrator_agent import OrchestratorAgent, AgentMessage, AgentRole
from agent_core.agent.report_gen_agent import ReportGeneratorAgent
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
    report_status: str
    report_result: Dict[str, Any]
    report_path: str


class AutonomousPipelineWorkflow:
    def __init__(self):
        self.code_gen_agent = CodeGeneratorAgent()
        self.debugger_agent = DebuggerAgent()
        self.orchestrator_agent = OrchestratorAgent()
        self.executor_agent = ExecutorAgent()
        self.report_agent = ReportGeneratorAgent()
        self.prompt_hash_file = PROMPT_HASH_FILE
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(PipelineState)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("debug", self._debug_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("report", self._report_node)
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
        workflow.add_conditional_edges("execute", self._should_continue_after_execute, {
            "report": "report",
            "error": "finalize"
        })
        workflow.add_edge("report", "finalize")
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
            "orchestrator_decision": {},
            "report_status": "pending",
            "report_result": {},
            "report_path": "",
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
            logger.exception("Orchestrator planning failed")
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
                    state["error_message"] = (result or {}).get("message", "Debug failed")
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

    # ---------- helpers for spawning ----------
    def _preflight_spawn(self, py_exec: str, script: Path, train: Path, test: Path):
        problems = []
        try:
            if not Path(py_exec).exists():
                problems.append(f"Python executable not found: {py_exec}")
        except Exception:
            pass
        if not script.exists():
            problems.append(f"generated_pipeline.py not found: {script}")
        if not train.exists():
            problems.append(f"Train CSV not found: {train}")
        if not test.exists():
            problems.append(f"Test CSV not found: {test}")
        return problems

    async def _run_pipeline_subprocess(self, cmd, cwd: Path, env: Dict[str, str]):
        """
        Windows-safe: run synchronously in a background thread and return (rc, stdout, stderr, mode).
        """
        import subprocess

        def _runner():
            return subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=False,
                env=env,
                shell=False  # IMPORTANT on Windows when passing list args
            )

        result = await asyncio.to_thread(_runner)
        return (
            result.returncode,
            result.stdout or b"",
            result.stderr or b"",
            "to_thread/subprocess.run",
        )

    async def _execute_node(self, state: PipelineState) -> PipelineState:
        logger.info("Executor agent taking control...")
        try:
            args = state["arguments"]
            required = ["train_path", "test_path", "label_column"]
            if not all(k in args for k in required):
                state["execution_status"] = "error"
                state["error_message"] = f"Missing required arguments: {required}"
                return state

            # Verify inputs exist
            train_p = Path(args["train_path"])
            test_p  = Path(args["test_path"])
            if not train_p.exists() or not test_p.exists():
                msg = f"Input file missing. train_exists={train_p.exists()} test_exists={test_p.exists()}"
                logger.error(msg)
                state["execution_status"] = "error"
                state["error_message"] = msg
                return state

            # Dataset hash & artifacts dir
            dataset_hash = self._get_dataset_hash(str(train_p), str(test_p), args["label_column"])
            parent_artifacts_dir = Path(args.get("artifacts_path", ARTIFACTS_DIR))
            dataset_artifacts = parent_artifacts_dir / dataset_hash
            dataset_artifacts.mkdir(parents=True, exist_ok=True)

            logger.info(f"Dataset hash: {dataset_hash}")
            logger.info(f"Upload paths: train={args['train_path']}, test={args['test_path']}")
            logger.info(f"Artifacts will be stored in: {dataset_artifacts}")

            # Decide execution plan
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

            # Cache & filesystem check
            cache = self._load_json(CACHE_FILE)
            force = bool(state["arguments"].get("force_rerun"))
            artifacts_exist = self._check_artifacts_exist(dataset_artifacts)
            logger.info(f"Artifacts exist check for {dataset_artifacts}: {artifacts_exist}")

            if (dataset_hash in cache) and artifacts_exist and \
               (execution_plan.get("decision") != "recovery_mode_execution") and not force:
                logger.info("✅ Using existing artifacts (cache verified with filesystem)")
                cached_result = cache[dataset_hash]
                state["artifacts"] = cached_result.get("artifacts", [])
                state["artifacts_path"] = cached_result.get("artifacts_path", str(dataset_artifacts))
                state["execution_status"] = "success"
                return state
            else:
                if not (dataset_hash in cache):
                    logger.info("🔄 No cache entry for this dataset - will run modeling")
                elif not artifacts_exist:
                    logger.info("🔄 Cache exists but artifacts missing from filesystem - will regenerate")
                elif force:
                    logger.info("🔄 Force rerun requested - will regenerate")
                elif execution_plan.get("decision") == "recovery_mode_execution":
                    logger.info("🔄 Recovery mode execution - will regenerate")

            # Ensure training script exists
            if not PIPELINE_FILE.exists():
                msg = f"generated_pipeline.py not found at {PIPELINE_FILE}. Code gen step may have failed."
                logger.error(msg)
                state["execution_status"] = "error"
                state["error_message"] = msg
                (dataset_artifacts / "agent_error.txt").write_text(msg, encoding="utf-8")
                return state

            # Build command & environment (Windows/headless safe)
            cmd = [
                sys.executable,
                str(PIPELINE_FILE.resolve()),
                str(train_p.resolve()),
                str(test_p.resolve()),
                args["label_column"],
                str(dataset_artifacts.resolve())
            ]
            env = os.environ.copy()
            env.setdefault("MPLBACKEND", "Agg")                 # no GUI
            env.setdefault("MPLCONFIGDIR", str(dataset_artifacts))  # writable dir for Matplotlib cache
            cwd = str(PIPELINE_FILE.parent.resolve())

            logger.info(f"🚀 Executing modeling pipeline (threaded subprocess): {' '.join(cmd)}")

            # Run synchronously in a background thread (Windows-safe, keeps this async)
            def _run():
                return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)

            completed = await asyncio.to_thread(_run)

            stdout_b = completed.stdout.encode("utf-8", errors="replace")
            stderr_b = completed.stderr.encode("utf-8", errors="replace")
            return_code = completed.returncode

            # Persist logs
            try:
                (dataset_artifacts / "stdout.log").write_text(stdout_b.decode("utf-8", errors="replace"))
                (dataset_artifacts / "stderr.log").write_text(stderr_b.decode("utf-8", errors="replace"))
            except Exception:
                logger.exception("Failed writing stdout/stderr logs")

            # Artifacts list
            artifacts = [f.name for f in dataset_artifacts.glob("*.*")]
            state["artifacts"] = artifacts
            state["artifacts_path"] = str(dataset_artifacts)

            if return_code != 0:
                err_txt = stderr_b.decode("utf-8", errors="replace").strip()
                state["execution_status"] = "error"
                state["error_message"] = err_txt or f"Pipeline exited with code {return_code}"
                logger.error(f"❌ Pipeline execution failed (rc={return_code})")
                try:
                    (dataset_artifacts / "agent_error.txt").write_text(state["error_message"], encoding="utf-8")
                except Exception:
                    pass
            else:
                out_txt = stdout_b.decode("utf-8", errors="replace")
                state["final_result"] = {"stdout": out_txt}
                state["execution_status"] = "success"
                logger.info(f"✅ Pipeline executed successfully, generated {len(artifacts)} artifacts")

            # Update cache
            cache[dataset_hash] = {
                "artifacts": artifacts,
                "artifacts_path": str(dataset_artifacts),
                "execution_plan": execution_plan,
                "timestamp": datetime.now().isoformat()
            }
            self._save_json(CACHE_FILE, cache)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Executor agent failed: {e!r}")
            try:
                if 'dataset_artifacts' in locals():
                    (dataset_artifacts / "agent_error.txt").write_text(tb, encoding="utf-8")
            except Exception:
                pass
            state["execution_status"] = "error"
            state["error_message"] = str(e) if str(e) else tb
        return state

    def _check_artifacts_exist(self, artifacts_dir: Path) -> bool:
        if not artifacts_dir.exists():
            logger.info(f"❌ Artifacts directory does not exist: {artifacts_dir}")
            return False
        key_artifacts = ["model_leaderboard.csv", "learner.pkl", "predictor.pkl"]
        existing_files = [f.name for f in artifacts_dir.glob("*.*")]
        logger.info(f"📁 Checking artifacts in: {artifacts_dir}")
        logger.info(f"📄 Found files: {existing_files}")

        has_key_artifact = any(artifact in existing_files for artifact in key_artifacts)
        found_key_artifacts = [artifact for artifact in key_artifacts if artifact in existing_files]
        non_log_artifacts = [f for f in existing_files if not f.endswith(('.log', '.txt'))]
        has_model_artifacts = len(non_log_artifacts) > 0

        logger.info("🔍 Artifacts analysis:")
        logger.info(f"   - Key artifacts found: {found_key_artifacts}")
        logger.info(f"   - Has key artifact: {has_key_artifact}")
        logger.info(f"   - Non-log artifacts: {len(non_log_artifacts)}")
        logger.info(f"   - Has model artifacts: {has_model_artifacts}")

        result = has_key_artifact and has_model_artifacts
        logger.info(f"✅ Final artifacts check result: {result}")
        return result

    async def _report_node(self, state: PipelineState) -> PipelineState:
        logger.info("Generating HTML report from artifacts...")
        try:
            art_dir = state.get("artifacts_path") or ""
            if not art_dir or not Path(art_dir).exists():
                state["report_status"] = "error"
                state["error_message"] = (state.get("error_message", "") + "\nMissing artifacts_path for report.").strip()
                return state

            result = await self.report_agent.generate_report(
                model_type="AutoGluon",
                artifact_dir=art_dir
            )

            status = (result or {}).get("status", "error")
            state["report_status"] = "success" if status in ("success", "cached") else "error"
            state["report_result"] = result or {}
            state["report_path"] = (result or {}).get("html_path", "")

            state["agent_messages"].append({
                "from_agent": AgentRole.EXECUTOR.value,
                "to_agent": "ReportGenerator",
                "message_type": "report_generated",
                "content": {"status": state["report_status"], "report_path": state["report_path"]},
                "timestamp": datetime.now().isoformat()
            })

            if "final_result" not in state or not isinstance(state["final_result"], dict):
                state["final_result"] = {}
            state["final_result"]["report"] = {
                "status": status,
                "path": state["report_path"],
                "details": state["report_result"],
            }
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            state["report_status"] = "error"
            state["error_message"] = (state.get("error_message", "") + f"\nReport error: {e}").strip()
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
            "total_agents_involved": 4,
            "report_status": state.get("report_status", "unknown"),
            "report_path": state.get("report_path", ""),
            "artifacts_path": state.get("artifacts_path", ""),
        }
        if "final_result" not in state:
            state["final_result"] = {}
        state["final_result"]["agent_summary"] = final_summary
        state["final_result"]["artifacts_path"] = state.get("artifacts_path", "")
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
        def hash_file(path: str) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()
        combined = f"{hash_file(train_path)}_{hash_file(test_path)}_{label_column}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _should_continue_after_generate(self, state: PipelineState) -> str:
        return "debug" if state["generation_status"] == "success" else "error"

    def _should_continue_after_debug(self, state: PipelineState) -> str:
        return "execute" if state["debug_status"] in ["success", "skipped"] else "error"

    def _should_continue_after_execute(self, state: PipelineState) -> str:
        return "report" if state["execution_status"] == "success" else "error"

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
