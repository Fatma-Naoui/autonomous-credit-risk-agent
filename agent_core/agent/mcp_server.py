import os
import sys
import json
import hashlib
import logging
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from mcp.server.fastmcp import FastMCP
from agent_core.agent.llm_code_generator import CodeGeneratorAgent
from agent_core.agent.debugger_agent import DebuggerAgent
from agent_core.prompts import prompts

BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "pipeline_cache.json"
PIPELINE_FILE = BASE_DIR / "generated_pipeline.py"
METRICS_FILE = BASE_DIR / "metrics.json"
ARTIFACTS_DIR = BASE_DIR / "models"
PROMPT_HASH_FILE = BASE_DIR / "pipeline_prompt_hash.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("autonomous-credit-risk-mcp-server")
code_gen_agent = CodeGeneratorAgent()
debugger_agent = DebuggerAgent()

def get_dataset_hash(arguments: Dict[str, Any]) -> str:
    def file_hash(file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    train_hash = file_hash(arguments['train_path'])
    test_hash = file_hash(arguments['test_path'])
    combined_key = f"{train_hash}_{test_hash}_{arguments['label_column']}"
    return hashlib.sha256(combined_key.encode()).hexdigest()

def load_json(file: Path) -> Any:
    if file.exists():
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json(file: Path, data: Any):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

@mcp.tool()
async def generate_pipeline(arguments: Dict[str, Any]) -> Dict[str, Any]:
    if "label_column" not in arguments:
        return {"status": "error", "message": "Missing argument: label_column"}

    code_prompt = prompts.autogluon_pipeline_generator_prompt.template
    prompt_hash = hashlib.sha256(code_prompt.encode()).hexdigest()
    stored = load_json(PROMPT_HASH_FILE)

    if stored.get("code_prompt_hash") != prompt_hash or not PIPELINE_FILE.exists():
        logger.info("Prompt changed or pipeline missing; regenerating pipeline.")
        await code_gen_agent.generate({"label_column": arguments["label_column"]})
        logger.info("Code generation completed, saving prompt hash.")
        
        stored["code_prompt_hash"] = prompt_hash
        stored["timestamp"] = datetime.now().isoformat()
        save_json(PROMPT_HASH_FILE, stored)
        
        return {"status": "success", "message": "Pipeline generated successfully."}
    else:
        logger.info("Pipeline is up-to-date with current generator prompt, skipping regeneration.")
        return {"status": "success", "message": "Pipeline already up-to-date."}

@mcp.tool()
async def debug_pipeline(arguments: Dict[str, Any]) -> Dict[str, Any]:
    debug_prompt = prompts.autogluon_pipeline_debugger_prompt.template
    prompt_hash = hashlib.sha256(debug_prompt.encode()).hexdigest()
    stored = load_json(PROMPT_HASH_FILE)

    # Ensure the hash file contains both code and debug hashes
    if not stored:
        stored = {}

    if stored.get("debug_prompt_hash") != prompt_hash:
        logger.info("Debugger prompt changed or missing; running debugger.")

        debug_result = await debugger_agent.debug_pipeline()  # returns a flat dict

        logger.info(f"Debugging completed, result: {debug_result}")

        # Ensure even if debugging fails, the hash updates for consistency
        stored["debug_prompt_hash"] = prompt_hash
        stored["debug_timestamp"] = datetime.now().isoformat()
        if "code_prompt_hash" not in stored:
            stored["code_prompt_hash"] = "pending"
        save_json(PROMPT_HASH_FILE, stored)

        if not debug_result:
            return {"status": "error", "message": "Debugger returned no result. Please check debugger implementation."}

        return debug_result

    else:
        logger.info("Debugger prompt unchanged, skipping debugging.")
        return {"status": "success", "message": "Pipeline already debugged with current prompt."}

@mcp.tool()
async def run_generated_pipeline(arguments: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"Running run_generated_pipeline with arguments: {arguments}")

    # Validate files
    for k in ["train_path", "test_path"]:
        if k not in arguments:
            return {"status": "error", "message": f"Missing argument: {k}"}
        if not os.path.isfile(arguments[k]):
            return {"status": "error", "message": f"File not found: {arguments[k]}"}

    if "label_column" not in arguments:
        return {"status": "error", "message": "Missing argument: label_column"}

    dataset_hash = get_dataset_hash(arguments)
    dataset_artifacts = ARTIFACTS_DIR / dataset_hash
    dataset_artifacts.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache = load_json(CACHE_FILE)
    if dataset_hash in cache:
        logger.info("Returning cached results for dataset")
        return cache[dataset_hash]

    logger.info(f"Starting pipeline execution at {datetime.now().isoformat()}")
    try:
        # Use the full path to agent_core/models as artifacts directory
        artifacts_dir = Path("agent_core/models").resolve()
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        process = await asyncio.create_subprocess_exec(
            "python", str(PIPELINE_FILE),
            arguments["train_path"], arguments["test_path"], arguments["label_column"], str(artifacts_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        logger.info(f"Pipeline process started with PID: {process.pid}")
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)  # 5 minutes
        logger.info(f"Pipeline execution completed at {datetime.now().isoformat()}")

        if process.returncode != 0:
            logger.error(f"Pipeline execution failed: {stderr.decode()}")
            return {"status": "error", "error": stderr.decode()}

        # Collect artifacts from agent_core/models, excluding .pkl
        artifacts = [
            str(file) for file in artifacts_dir.glob("*")
            if file.is_file() and file.suffix not in [".pkl"]
        ]

        result = {
            "status": "success",
            "message": "Pipeline executed successfully",
            "artifacts": artifacts,
            "artifacts_path": str(artifacts_dir),
            "pipeline_file": str(PIPELINE_FILE),
            "stdout": stdout.decode()
        }
        
        cache[dataset_hash] = result
        save_json(CACHE_FILE, cache)
        return result
        
    except asyncio.TimeoutError:
        logger.error("Pipeline execution timed out after 5 minutes")
        return {"status": "error", "message": "Pipeline execution timed out"}
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    logger.info("Starting FastMCP server.")
    mcp.run()