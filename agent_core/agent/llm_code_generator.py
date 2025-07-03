import re
import os
from llm_client import generate_code

prompt = """
You are a senior data scientist. Generate clean, runnable Python code for a binary classification credit risk project using AutoGluon.

Requirements:
- Use `from autogluon.tabular import TabularPredictor`.
- Load training data from `agent_core/data/train_data.csv` and test data from `agent_core/data/test_data.csv`, relative to the project root.
- Use `loan_status` as the label column.
- Train the model using **only default AutoGluon hyperparameters**
— **do NOT specify or include any 'hyperparameters' argument in the code**.
- Save the trained model in `agent_core/models`.
- Generate a leaderboard CSV saved as `agent_core/models/model_leaderboard.csv`.
- Generate evaluation metrics JSON saved as `agent_core/models/model_metrics.json`.
- Create and save a feature importance plot as `agent_core/models/shap_summary.png`.
- Also save a copy of the leaderboard CSV as `agent_core/leaderboard.csv` and evaluation metrics JSON as `agent_core/evaluation_metrics.json` (for easy access).
- Use pandas for CSV operations and matplotlib for plotting.
- Do not import or use the shap library.
- Do not include code fencing or markdown formatting.
- Do not include any explanations or comments, only runnable Python code.

"""

code = generate_code(prompt)

# Strip markdown code fences if present (``` or ```python)
code = re.sub(r"^```(?:python)?\n", "", code, flags=re.MULTILINE)  # Remove opening fence
code = re.sub(r"\n```$", "", code, flags=re.MULTILINE)             # Remove closing fence

# Get current script folder (llm_code_generator.py folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(current_dir, "generated_pipeline.py")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(code)

print(f"Generated pipeline saved to: {out_path}")
