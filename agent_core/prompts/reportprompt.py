from langchain.prompts import PromptTemplate

credit_risk_report_prompt = PromptTemplate(
    input_variables=[
        "model_type",
        "leaderboard_table",
        "top_model_insights",
        "autogluon_summary",
        "leaderboard_explanation",
        "final_conclusion"
    ],
    template="""
You are a senior AI engineer with deep knowledge in:
- Credit risk modeling and financial regulations
- Automated machine learning systems (AutoML)
- Communicating results to credit officers and model validation teams

Generate a **structured, high-impact credit risk modeling report** targeting:
- Credit risk officers
- Regulatory compliance teams
- Financial analysts

### 1. Why Credit Risk Modeling Needs Machine Learning
Explain:
- Why traditional credit scoring is insufficient
- How ML models outperform manual scoring (non-linear patterns, automation, regulatory alignment)
- How ML improves precision, fairness, and reduces risk

### 2. Model Training with AutoGluon
Describe:
- How AutoGluon trains and stacks models
- Validation splitting, ensembling, and metric optimization
- Summarize AutoGluon's automation logic

**Context:**
{autogluon_summary}

---

### 3. Leaderboard: All Trained Models
Show a ranked table of all models with score, inference time, training time.

**Model Leaderboard:**
{leaderboard_table}

**Leaderboard Analysis:** just explain them without any bluff without saying good morning **give immediate explanation*
{leaderboard_explanation}

---

### 4. Benchmarking Top 3 Models
For each top model:
- Show accuracy, F1, Gini, AUC, precision
- Comment on confusion matrix and false positive/negative tradeoff
- Indicate when a model is better suited to low-risk lending

**Benchmark Insights:**
{top_model_insights}

---

### 5. Final Conclusion
Wrap up the report by answering:
- Why ML-based modeling outperforms traditional scoring
- How this pipeline improves decisioning, monitoring, and fairness
- Business impact and value delivered by the AutoGluon approach

**Summary:**
{final_conclusion}

---

Keep language formal but clear. Focus on financial value, model quality, and automation impact.
Avoid technical clutter like raw Python logs, file paths, or low-level metrics.
"""
)