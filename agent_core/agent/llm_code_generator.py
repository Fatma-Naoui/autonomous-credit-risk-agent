import os
import sys
import json
import subprocess
import time
from pathlib import Path
from llm_client import generate_code
from agent_core.prompts.prompts import autogluon_pipeline_generator_prompt

class CodeGeneratorAgent:
    def clean_generated_code(self, code):
        code = code.strip()
        if code.startswith("```"):
            code = code.split("```", 1)[-1].strip()
            if code.lower().startswith("python"):
                code = code[len("python"):].lstrip('\n').lstrip()
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0].strip()
        return code

    async def generate(self, arguments):
        try:
            prompt = autogluon_pipeline_generator_prompt.format(
                label_column=arguments["label_column"]
            )

            code = generate_code(prompt)
            code = self.clean_generated_code(code)

            current_dir = Path(__file__).resolve().parent
            out_path = current_dir / "generated_pipeline.py"

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)

            return {
             "status": "success",
             "message": f"Pipeline generated and saved to {out_path}",
             "generated_code": code 
             }

        except Exception as e:
           return {
           "status": "error",
           "message": str(e) 
           }

