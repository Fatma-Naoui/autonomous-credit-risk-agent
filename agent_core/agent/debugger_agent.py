import subprocess
import time
from llm_client import generate_code

MAX_RETRIES = 5
CODE_FILE = "generated_pipeline.py"

def run_generated_code():
    """Run the generated pipeline and capture stdout and stderr."""
    result = subprocess.run(
        ["python", CODE_FILE],
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr, result.returncode

def auto_debug():
    retries = 0

    while retries < MAX_RETRIES:
        print(f"\n⚡ Attempt {retries + 1} of {MAX_RETRIES}...\n")
        stdout, stderr, returncode = run_generated_code()

        if returncode == 0:
            print("Code executed successfully.\n")
            print(stdout)
            break
        else:
            print("Code execution failed. Capturing error and sending to LLM for fixing.\n")
            print(stderr)

            prompt = f"""
You are a senior Python software engineer with deep expertise in AutoGluon for tabular data modeling, 
especially credit risk classification tasks.

The following Python code implements an AutoGluon pipeline for binary classification on credit risk data. 
It should load `train_data.csv` and `test_data.csv`, train a model using `loan_status` as the label, save 
the model, generate a leaderboard CSV, output evaluation metrics in JSON, and save a feature importance plot.

Your task is to fix the code so it runs without errors. Ensure the code:

- Remains modular, readable, and well-structured.
- Uses correct AutoGluon API usage and best practices.
- Handles any file paths or imports correctly.
- Outputs the same artifacts: model saved, leaderboard.csv, evaluation_metrics.json, feature_importance.png.
- Does NOT include any explanation or commentary, just the fixed runnable Python code.
- Remove any markdown code fences such as ```python or ``` from the output. Return only raw Python code.

Below is the code that caused the error, followed by the exact error message.

--- CODE START ---
{open(CODE_FILE, 'r').read()}
--- CODE END ---

--- ERROR START ---
{stderr}
--- ERROR END ---

Please ONLY return the corrected Python code, preserving its modular structure and functionality.
"""

            fixed_code = generate_code(prompt)
            with open(CODE_FILE, "w") as f:
                f.write(fixed_code)

            retries += 1
            time.sleep(1)

    else:
        print("Maximum retries reached. Please review the code manually.")

if __name__ == "__main__":
    auto_debug()
