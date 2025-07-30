from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

autogluon_pipeline_generator_prompt = PromptTemplate(
    input_variables=[],  # <-- label_column removed
    template="""
You are a senior data scientist. Generate clean, runnable Python code for a binary classification credit risk project using AutoGluon.

Requirements:
- Accept CLI arguments:
    sys.argv[1]: training CSV
    sys.argv[2]: test CSV
    sys.argv[3]: label column
    sys.argv[4]: output directory
- Use from autogluon.tabular import TabularPredictor
- Load CSVs with pd.read_csv(sys.argv[...])
- Use os.makedirs(output_dir, exist_ok=True) to ensure the directory exists
- **Save models in the output_dir (e.g., TabularPredictor(label=label_column, path=output_dir))** instead of "."
- Save all files using os.path.join(output_dir, ...) instead of hardcoded paths
-Save model_metrics.json for all models ( f1 precision auc gini etc )
- Save:
    model_leaderboard.csv
   
    evaluation_metrics.json
    shap_summary.png
    roc_{{model_name}}.png
    pr_{{model_name}}.png
    cm_{{model_name}}.png
- Use sklearn for metrics, pandas, matplotlib (no shap)
- Use structured functions:
    load_data, train_model, generate_leaderboard,
    generate_evaluation_metrics, generate_feature_importance,
    plot_feature_importance, plot_roc_curve,
    plot_precision_recall_curve, plot_confusion_matrix, main
- Replace test_pred_proba[:, 1] with test_pred_proba.iloc[:, 1] for roc_auc_score
- Use predictor.model_names()[0] (not predictor.model_names[0])
-Save model_metrics.json for all models ( f1 precision auc gini etc )
- For the **best 3 models based on leaderboard score_val descending**, replace:
    model_names = predictor.model_names()[:3]
  with:
    leaderboard_sorted = leaderboard.sort_values(by="score_val", ascending=False)
    model_names = leaderboard_sorted["model"].head(3).tolist()
    - Generate and save ROC, PR, and confusion matrix plots using os.path.join(output_dir, ...)
    - ROC curves saved as roc_{{model_name}}.png
    - PR curves saved as pr_{{model_name}}.png
    - Confusion matrices saved as cm_{{model_name}}.png
    - Each plot on its own separate figure (no overlays)
    - Each model uses a different consistent color
    - Confusion matrices must display numbers on the heatmap
    - Save with dpi=300, bbox_inches="tight", and call plt.close() after saving
- Replace: feature_importance = generate_feature_importance(predictor)
  with: feature_importance = generate_feature_importance(predictor, test_data)
- Replace: fpr, tpr, _ = roc_auc_score(...) with: fpr, tpr, _ = roc_curve(...)
- Use correct positive class handling for roc_auc_score
- Use predictor.predict and predictor.predict_proba with model=model_name
- No explanations, comments, markdown, or hardcoded paths
- Output ONLY the raw runnable Python code
"""
)

autogluon_pipeline_debugger_prompt = PromptTemplate(
    input_variables=["code_snippet", "error_snippet"],
    template="""
You are a senior Python engineer specialized in AutoGluon pipelines for credit risk modeling.

Fix the provided Python code to run without errors while meeting these requirements:

- **Move `time_limit=60` from `TabularPredictor(...)` to `.fit(...)`** — it is not a valid argument in the constructor
    ❗ Current incorrect: TabularPredictor(..., time_limit=60)
    ✅ Corrected: TabularPredictor(...).fit(..., time_limit=60)

- Replace predictor.get_model_names() with predictor.model_names()
- Replace: feature_importance = generate_feature_importance(predictor)  
  with: feature_importance = generate_feature_importance(predictor, test_data)
- Replace test_pred_proba[:, 1] with test_pred_proba.iloc[:, 1] for roc_auc_score
- **Ensure TabularPredictor saves models in the output_dir (sys.argv[4]) (path=output_dir).**
- Ensure all plots save in output_dir without displaying (no plt.show())
- Use CLI arguments only (sys.argv[1-4]), no hardcoded paths
- Use structured functions:
    load_data, train_model, generate_leaderboard,
    generate_evaluation_metrics, generate_feature_importance,
    plot_feature_importance, plot_roc_curve,
    plot_precision_recall_curve, plot_confusion_matrix, main
- Use predictor.model_names()[0] instead of predictor.model_names[0]
- Replace: fpr, tpr, _ = roc_auc_score(...) with: fpr, tpr, _ = roc_curve(...)
- For the **best 3 models from the leaderboard**:
    - Generate and save ROC, PR, and confusion matrix plots
    - Each model uses a different consistent color
    - Confusion matrices must display numbers on the heatmap
    - Save plots with dpi=300, bbox_inches="tight", and plt.close() after saving
- Output must save to the provided output_dir (from sys.argv[4])
- No shap, explanations, comments, or markdown
- Output ONLY the corrected, clean, runnable Python code ready to overwrite generated_pipeline.py

Code:
{code_snippet}

Error:
{error_snippet}
"""
)