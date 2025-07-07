import subprocess
import time
import traceback
from llm_client import generate_code

MAX_RETRIES = 2
CODE_FILE = os.path.join(os.path.dirname(__file__), "generated_pipeline.py")
FAST_DEBUG = True  # Toggle debug mode for faster iterations

def run_generated_code():
    """Run the generated pipeline and capture stdout and stderr."""
    result = subprocess.run(
        ["python", CODE_FILE],
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr, result.returncode

def extract_relevant_code_snippet(code_text, lines=60):
    """Extract the first and last parts of the code to reduce LLM load."""
    code_lines = code_text.splitlines()
    if len(code_lines) <= lines:
        return code_text
    return "\n".join(code_lines[:lines//2] + ["# ... (code truncated) ..."] + code_lines[-lines//2:])

def extract_relevant_error_snippet(stderr_text, lines=30):
    """Extract the last lines of stderr for relevant traceback."""
    error_lines = stderr_text.strip().splitlines()
    return "\n".join(error_lines[-lines:])

def auto_debug():
    retries = 0

    while retries < MAX_RETRIES:
        print(f"\n⚡ Attempt {retries + 1} of {MAX_RETRIES}...\n")
        stdout, stderr, returncode = run_generated_code()

        if returncode == 0:
            print("✅ Code executed successfully.\n")
            print(stdout)
            break
        else:
            print("❌ Code execution failed. Capturing error and sending to LLM for fixing.\n")
            print(stderr)

            code_content = open(CODE_FILE, 'r').read()
            code_snippet = extract_relevant_code_snippet(code_content)
            error_snippet = extract_relevant_error_snippet(stderr)

            prompt = f"""
You are a senior Python software engineer with deep expertise in AutoGluon for tabular data modeling, especially credit risk classification tasks.

The following Python code implements an AutoGluon pipeline for binary classification on credit risk data. It should load `agent_core/data/train_data.csv` and `agent_core/data/test_data.csv`, train a model using `loan_status` as the label, save the model, generate a leaderboard CSV, output evaluation metrics in JSON, and save a feature importance plot.

Your task is to fix the code so it runs without errors and aligns with the following corrections:

- Replace `metrics.to_dict()` with just `metrics` since `predictor.evaluate()` returns a dict.
- Use robust absolute path handling:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data')
    train_path = os.path.join(data_path, 'train_data.csv')
    test_path = os.path.join(data_path, 'test_data.csv')
    model_path = os.path.join(project_root, 'models')
- Use AutoGluon `predictor.predict` and `predictor.predict_proba` directly on `test_data` for the top 3 models:
    y_pred = predictor.predict(test_data, model=model_name)
    y_pred_proba_df = predictor.predict_proba(test_data, model=model_name)
    positive_class = predictor.class_labels[-1]
    y_pred_proba = y_pred_proba_df[positive_class] if positive_class in y_pred_proba_df else y_pred_proba_df.iloc[:, -1]
- Add `plt.close()` after each plot.
- Save models in the correct `models` folder.
- Use `time_limit=60` in `TabularPredictor.fit(...)` to keep debug cycles fast.
- Keep the modular structure: load_data, train_model, generate_leaderboard, generate_evaluation_metrics, generate_feature_importance, plot_feature_importance, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix, plot_shap_summary, main.

STRICT INSTRUCTIONS:
- Return ONLY the corrected, clean, runnable Python code.
- Do NOT include any markdown code fences or explanations.
- Do NOT add comments in the code.
- Ensure the code is ready to drop directly into `generated_pipeline.py` in the MCP pipeline workflow.

Below is the code that needs fixing:

--- CODE START ---
{code_snippet}
--- CODE END ---

--- ERROR START ---
{error_snippet}
--- ERROR END ---

Please ONLY return the corrected Python code, preserving its modular structure and functionality.
"""

            fixed_code = generate_code(prompt)
            clean_code = fixed_code.replace("```python", "").replace("```", "").strip()
            with open(CODE_FILE, "w") as f:
             f.write(clean_code)


            retries += 1
            time.sleep(1)

    else:
        print("⚠️ Maximum retries reached. Please review the code manually.")

if __name__ == "__main__":
    auto_debug()
