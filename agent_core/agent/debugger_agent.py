import asyncio
import subprocess
import sys
import time
import json
from pathlib import Path
from llm_client import generate_code
from agent_core.prompts.prompts import autogluon_pipeline_debugger_prompt

class DebuggerAgent:
    def __init__(self):
        self.MAX_RETRIES = 3  # Reduced to limit runtime
        self.CODE_FILE = Path(__file__).resolve().parent / "generated_pipeline.py"
        self.PER_RUN_TIMEOUT = 100

    async def run_generated_code(self):
        process = await asyncio.create_subprocess_exec(
            "python", str(self.CODE_FILE),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.PER_RUN_TIMEOUT)
            return stdout.decode(), stderr.decode(), process.returncode
        except asyncio.TimeoutError:
            process.kill()
            return "", "Timeout during pipeline execution", 1

    def extract_relevant_code_snippet(self, code_text, lines=60):
        code_lines = code_text.splitlines()
        if len(code_lines) <= lines:
            return code_text
        return "\n".join(code_lines[:lines//2] + ["# ... (code truncated) ..."] + code_lines[-lines//2:])

    def extract_relevant_error_snippet(self, stderr_text, lines=30):
        error_lines = stderr_text.strip().splitlines()
        return "\n".join(error_lines[-lines:])

    def clean_generated_code(self, fixed_code):
        fixed_code = fixed_code.strip()
        if fixed_code.startswith("```"):
            fixed_code = fixed_code.split("```", 1)[-1].strip()
            if fixed_code.lower().startswith("python"):
                fixed_code = fixed_code[len("python"):].lstrip("\n").lstrip()
        if fixed_code.endswith("```"):
            fixed_code = fixed_code.rsplit("```", 1)[0].strip()
        return fixed_code

    async def auto_debug(self):
        try:
            retries = 0
            total_start_time = time.time()

            while retries < self.MAX_RETRIES:
                print(f"[AutoDebug] Attempt {retries+1}/{self.MAX_RETRIES}")
                stdout, stderr, returncode = await self.run_generated_code()
                print(f"[AutoDebug] Return code: {returncode}")
                print(f"[AutoDebug] Stdout:\n{stdout[:500]}")
                print(f"[AutoDebug] Stderr:\n{stderr[:500]}")

                if returncode == 0 and stdout.strip():
                    print("[AutoDebug] Pipeline executed successfully with output.")
                    return {"status": "success", "message": "Pipeline executed successfully.", "stdout": stdout}
                else:
                    print("[AutoDebug] Pipeline failed or produced no output, calling LLM to regenerate.")

                    code_content = self.CODE_FILE.read_text(encoding="utf-8")
                    code_snippet = self.extract_relevant_code_snippet(code_content)
                    error_snippet = self.extract_relevant_error_snippet(stderr)

                    prompt = autogluon_pipeline_debugger_prompt.format(
                        code_snippet=code_snippet,
                        error_snippet=error_snippet
                    )
                    print(f"[AutoDebug] Prompt to LLM:\n{prompt[:500]}")

                    fixed_code = await generate_code(prompt)  # Assume generate_code is async
                    print(f"[AutoDebug] Raw LLM response:\n{fixed_code[:500]}")

                    fixed_code = self.clean_generated_code(fixed_code)
                    print(f"[AutoDebug] Cleaned code:\n{fixed_code[:500]}")

                    if not fixed_code.strip():
                        print("[AutoDebug] LLM returned empty code. Aborting debug.")
                        return {"status": "error", "message": "LLM returned empty code. Aborting debug."}

                    self.CODE_FILE.write_text(fixed_code, encoding="utf-8")
                    print("[AutoDebug] Fixed code written to generated_pipeline.py")

                    retries += 1
                    if time.time() - total_start_time > 600:  # 10-minute total timeout
                        print("[AutoDebug] Total timeout reached.")
                        return {"status": "error", "message": "Total debug timeout exceeded"}

                    await asyncio.sleep(1)  # Non-blocking delay

            print("[AutoDebug] Maximum retries reached. Debugging did not succeed.")
            return {"status": "error", "message": "Maximum retries reached. Please review the code manually.", "stderr": stderr}

        except Exception as e:
            print(f"[AutoDebug] Exception occurred: {e}")
            return {"status": "error", "message": str(e)}

    async def debug_pipeline(self):
        return await self.auto_debug()