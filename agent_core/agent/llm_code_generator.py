import re
import os
from llm_client import generate_code

prompt = """
You are a senior data scientist. Generate clean, runnable Python code for a binary classification credit risk project using AutoGluon.

Requirements:

- Use `from autogluon.tabular import TabularPredictor`.
- Load training data from `agent_core/data/train_data.csv` and test data from `agent_core/data/test_data.csv` using these exact relative paths with `pd.read_csv()`.
- Train the model using only default AutoGluon hyperparameters. Do not include any `hyperparameters` argument in the code.
- Save the trained model in `agent_core/models`.
- Generate:
  - A leaderboard CSV as `agent_core/models/model_leaderboard.csv` and a copy as `agent_core/leaderboard.csv`.
  - An evaluation metrics JSON as `agent_core/models/model_metrics.json` and a copy as `agent_core/evaluation_metrics.json`.
  - A feature importance plot as `agent_core/models/shap_summary.png`.
- Use pandas for CSV operations and matplotlib for all plots.
- Do not use or import the `shap` library.
- Do not include any explanations or comments, only runnable Python code.
- Ensure `agent_core/models` exists before saving outputs.
- Import scikit-learn as `import sklearn` (not `scikit-learn`).
- After training, extract the top 3 models from the leaderboard.
- For each of the top 3 models:
  - Generate and save:
    - ROC curve as `agent_core/models/roc_{model_name}.png`
    - Precision-Recall curve as `agent_core/models/pr_{model_name}.png`
    - Confusion Matrix as `agent_core/models/confusion_matrix_{model_name}.png`
  - Use `sklearn.metrics` for ROC, AUC, Precision-Recall, and Confusion Matrix calculations.
  - Use consistent colors for each model across all plots so that if model A uses red in the ROC curve, it also uses red in the PR and Confusion Matrix plots.
  - Save all plots with `dpi=300` and `bbox_inches="tight"`.
- Remove any incorrect calls to `predictor.evaluate(test_data, 'roc_auc')`.
- Compute ROC AUC using `sklearn.metrics.roc_auc_score`.
- Use structured, reusable functions:
  - `load_data`
  - `train_model`
  - `generate_leaderboard`
  - `generate_evaluation_metrics`
  - `generate_feature_importance` (pass `test_data` to `predictor.feature_importance`)
  - `plot_feature_importance`
  - `plot_roc_curve`
  - `plot_precision_recall_curve`
  - `plot_confusion_matrix`
  - `main`
- Use the correct Pandas indexing (`iloc` or column name) to extract positive class probabilities for `roc_auc_score`.
- Ensure it works even if labels are categorical by dynamically selecting the positive class column.
- Return only the raw, clean, runnable Python code without markdown fences under any circumstance.

Your output will be used directly in a production MCP autonomous pipeline for credit risk modeling, so it must be robust, modular, and aligned strictly with these requirements.
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
