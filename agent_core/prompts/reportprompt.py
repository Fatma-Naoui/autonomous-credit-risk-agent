from langchain.prompts import PromptTemplate

loan_model_report_prompt = PromptTemplate(
    input_variables=[
        "model_type",
        "leaderboard_summary",
        "top_model_metrics",
        "autogluon_info",
        "credit_metrics_info",
        "metric_details"
    ],
    template="""
You are a senior AI engineer specialized in credit risk modeling.

Generate a detailed technical report explaining the results of a credit risk pipeline using {model_type}.

The report must include:

1. **Modeling Overview**  
Summarize the modeling workflow using AutoGluon, including training, evaluation, and model selection steps, based on this:  
{autogluon_info}

2. **Leaderboard Interpretation**  
- Explain how the leaderboard was constructed.  
- Identify the primary metric used for ranking.  
- Describe how models compare in terms of validation score and robustness.  
- Provide a ranked summary of the top 3 models:  
{leaderboard_summary}

3. **Top 3 Model Metrics**  
- For each of the top 3 models, analyze core evaluation metrics: AUC, Gini, F1, Precision, Recall, Accuracy, etc.  
- Provide a rationale for performance differences.  
- Use this data:  
{top_model_metrics}

4. **Metric Significance in Credit Risk**  
- For each metric used, explain its **financial interpretation**:  
    - What does a high/low score mean in loan eligibility?  
    - Why is it important for risk assessment?  
- Use this reference:  
{credit_metrics_info}

5. **Insights from External Research**  
- Summarize web or document-based insights about metrics, thresholds, and best practices.  
- Enrich the analysis using these sources:  
{metric_details}

**Instructions:**  
- Focus on technical clarity, depth, and financial relevance.  
- Use clear headings and bullet points where helpful.  
- Do NOT include raw code, log traces, or file names.
"""
)
