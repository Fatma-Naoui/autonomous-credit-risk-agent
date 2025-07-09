from langchain.prompts import PromptTemplate

autogluon_pipeline_generator_prompt = PromptTemplate(
    input_variables=["label_column"],
    template="""
You are a senior data scientist. Generate clean, runnable Python code for a binary classification credit risk project using AutoGluon.

Requirements:
- Accept CLI arguments:
    sys.argv[1]: training CSV
    sys.argv[2]: test CSV
    sys.argv[3]: label column
- Use from autogluon.tabular import TabularPredictor
- Load CSVs with pd.read_csv(sys.argv[...])
- Train on {label_column} with time_limit=60
- Save trained model in agent_core/models
- Save:
    agent_core/models/model_leaderboard.csv
    agent_core/leaderboard.csv
    agent_core/models/model_metrics.json
    agent_core/evaluation_metrics.json
    agent_core/models/shap_summary.png
- Use sklearn for metrics, pandas, matplotlib (no shap)
- Use structured functions:
    load_data, train_model, generate_leaderboard,
    generate_evaluation_metrics, generate_feature_importance,
    plot_feature_importance, plot_roc_curve,
    plot_precision_recall_curve, plot_confusion_matrix, main
- Replace test_pred_proba[:, 1] with test_pred_proba.iloc[:, 1] for roc_auc_score
- Use predictor.model_names()[0] (not predictor.model_names[0])
- For the **best 3 models based on leaderboard score_val descending  replace model_names = predictor.model_names()[:3]
 with leaderboard_sorted = leaderboard.sort_values(by="score_val", ascending=False)
model_names = leaderboard_sorted["model"].head(3).tolist()
**:
    - Generate and save ROC, PR, and confusion matrix plots in agent_core/models
    - ROC curves saved as agent_core/models/roc_{{model_name}}.png
    - PR curves saved as agent_core/models/pr_{{model_name}}.png
    - Each plot on its own separate figure (no overlays)
    - Each model uses a different consistent color
    - Confusion matrices must display numbers on the heatmap
    - Save with dpi=300, bbox_inches="tight", and call plt.close() after saving
    - Confusion matrices must display numbers on the heatmap
- Replace: feature_importance = generate_feature_importance(predictor) with: feature_importance = generate_feature_importance(predictor, test_data)
-Replace: fpr, tpr, _ = roc_auc_score(test_labels, test_pred_proba.iloc[:, 1], multi_class='ovr') with: fpr, tpr, _ = roc_curve(test_labels, test_pred_proba.iloc[:, 1])
-Use correct positive class handling for roc_auc_score
- Use predictor.predict, predictor.predict_proba with model=model_name
- No explanations, comments, markdown, or hardcoded paths
- Output ONLY the raw runnable Python code
"""
)

autogluon_pipeline_debugger_prompt = PromptTemplate(
    input_variables=["code_snippet", "error_snippet"],
    template="""
You are a senior Python engineer specialized in AutoGluon pipelines for credit risk modeling.

Fix the provided Python code to run without errors while meeting these requirements:
- Replace predictor.get_model_names() with predictor.model_names()
- Replace: feature_importance = generate_feature_importance(predictor) with: feature_importance = generate_feature_importance(predictor, test_data)
- Replace test_pred_proba[:, 1] with test_pred_proba.iloc[:, 1] for roc_auc_score
- Ensure all plots save in agent_core/models without displaying (no plt.show())
- Use CLI arguments only (sys.argv[1-3]), no hardcoded paths
- Use time_limit=60 in TabularPredictor
- Use structured functions:
    load_data, train_model, generate_leaderboard,
    generate_evaluation_metrics, generate_feature_importance,
    plot_feature_importance, plot_roc_curve,
    plot_precision_recall_curve, plot_confusion_matrix, main
- Use predictor.model_names()[0] instead of predictor.model_names[0]
-Replace: fpr, tpr, _ = roc_auc_score(test_labels, test_pred_proba.iloc[:, 1], multi_class='ovr') with: fpr, tpr, _ = roc_curve(test_labels, test_pred_proba.iloc[:, 1])
- For the **best 3 models from the leaderboard**:
    - Generate and save ROC, PR, and confusion matrix plots
    - Each model uses a different consistent color
    - Confusion matrices must display numbers on the heatmap
    - Save plots with dpi=300, bbox_inches="tight", and plt.close() after saving
    -plt.close for roc curve and precision curve
- No shap, explanations, comments, or markdown
- Output ONLY the corrected, clean, runnable Python code ready to overwrite generated_pipeline.py

Code:
{code_snippet}

Error:
{error_snippet}
"""
)
