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
        try:
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading JSON file {file}: {e}")
            return {}
    return {}

def save_json(file: Path, data: Any):
    try:
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except (IOError, TypeError) as e:
        logger.error(f"Error saving JSON file {file}: {e}")

@mcp.tool()
async def generate_pipeline(arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if "label_column" not in arguments:
            return {"status": "error", "message": "Missing argument: label_column"}

        code_prompt = prompts.autogluon_pipeline_generator_prompt.template
        prompt_hash = hashlib.sha256(code_prompt.encode()).hexdigest()
        stored = load_json(PROMPT_HASH_FILE)

        pipeline_needs_update = stored.get("code_prompt_hash") != prompt_hash or not PIPELINE_FILE.exists()

        if pipeline_needs_update:
            logger.info("Prompt changed or pipeline missing; generating pipeline.")

            generation_result = await code_gen_agent.generate({"label_column": arguments["label_column"]})

            # Ensure pipeline file was created
            if not PIPELINE_FILE.exists():
                return {"status": "error", "message": "Pipeline generation failed: file not created"}

            # Save updated prompt hash
            stored["code_prompt_hash"] = prompt_hash
            stored["timestamp"] = datetime.now().isoformat()
            save_json(PROMPT_HASH_FILE, stored)

            result = {"status": "success", "message": "Pipeline generated successfully."}
            logger.info(f"Returning from generate_pipeline with: {result}")
            return result

        else:
            result = {"status": "success", "message": "Pipeline already up-to-date."}
            logger.info(f"Returning from generate_pipeline with: {result}")
            return result

    except Exception as e:
        logger.error(f"Error in generate_pipeline: {e}")
        return {"status": "error", "message": f"Pipeline generation failed: {str(e)}"}


@mcp.tool()
async def debug_pipeline(arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
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
    
    except Exception as e:
        logger.error(f"Error in debug_pipeline: {e}")
        return {"status": "error", "message": f"Pipeline debugging failed: {str(e)}"}

async def run_pipeline_async(cmd: list, artifacts_dir: Path, timeout: int = 400) -> Dict[str, Any]:
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(BASE_DIR)
        )

        try:
            # Run process and wait in parallel
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            returncode = process.returncode

            # Decode early to ensure text is flushed
            stdout_text = stdout.decode('utf-8', errors="replace")
            stderr_text = stderr.decode('utf-8', errors="replace")

            return {
                "returncode": returncode,
                "stdout": stdout_text,
                "stderr": stderr_text
            }

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Process timed out",
                "timeout": True
            }

    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "error": True
        }

@mcp.tool()
async def run_generated_pipeline(arguments: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"Running run_generated_pipeline with arguments: {arguments}")

    try:
        # Validate arguments
        for k in ["train_path", "test_path"]:
            if k not in arguments:
                return {"status": "error", "message": f"Missing argument: {k}"}
            if not os.path.isfile(arguments[k]):
                return {"status": "error", "message": f"File not found: {arguments[k]}"}

        if "label_column" not in arguments:
            return {"status": "error", "message": "Missing argument: label_column"}

        if not PIPELINE_FILE.exists():
            return {"status": "error", "message": "Pipeline file not found. Please generate pipeline first."}

        dataset_hash = get_dataset_hash(arguments)
        dataset_artifacts = ARTIFACTS_DIR / dataset_hash
        dataset_artifacts.mkdir(parents=True, exist_ok=True)

        cache = load_json(CACHE_FILE)
        if dataset_hash in cache:
            logger.info("Returning cached results for dataset")
            return cache[dataset_hash]

        # Execute the pipeline
        cmd = [
            sys.executable, "-u", str(PIPELINE_FILE.resolve()),
            str(Path(arguments["train_path"]).resolve()),
            str(Path(arguments["test_path"]).resolve()),
            arguments["label_column"],
            str(dataset_artifacts.resolve())
        ]

        result = await run_pipeline_async(cmd, dataset_artifacts)

        # Write logs regardless of connection status
        with open(dataset_artifacts / "stdout.log", "w") as f:
            f.write(result.get("stdout", ""))
        with open(dataset_artifacts / "stderr.log", "w") as f:
            f.write(result.get("stderr", ""))

        # Save partial results to disk
        artifacts = [
            str(file.relative_to(dataset_artifacts)) for file in dataset_artifacts.glob("**/*")
            if file.is_file() and file.suffix not in [".pkl"]
        ]

        response = {
            "status": "success" if result["returncode"] == 0 else "error",
            "artifacts": artifacts,
            "artifacts_path": str(dataset_artifacts),
        }

        logger.info(f"Returning final result: {response}")

        cache[dataset_hash] = response
        save_json(CACHE_FILE, cache)

        return response

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"status": "error", "message": str(e)}



if __name__ == "__main__":
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)