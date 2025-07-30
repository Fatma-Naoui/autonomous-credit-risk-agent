# === Fixed mcp_server.py ===

from fastmcp import FastMCP
import httpx
import json
from typing import List, Optional
import asyncio
from bs4 import BeautifulSoup
import re

mcp = FastMCP("MCP server", stateless_http=True)

AUTOGLUON_API_URL = "https://auto.gluon.ai/0.4.1/api/index.html"

@mcp.tool()
async def get_autogluon_tabular_workflow(specific_step: Optional[str] = None) -> str:
    """
    Explain AutoGluon's tabular data modeling workflow.
    """
    autogluon_workflow = {
        "overview": """
        AutoGluon TabularPredictor automates machine learning for structured/tabular data:
        1. Data preprocessing
        2. Multi-model training
        3. Ensembling
        4. Validation & selection
        5. Prediction interface
        """,
        "training": """
        Training includes LightGBM, CatBoost, XGBoost, NN, RF with Bayesian optimization.
        """,
        "ensemble": """
        Uses weighted and stacked ensembles to boost performance automatically.
        """,
        "prediction": """
        Supports .predict() and .predict_proba() with batch and real-time support.
        """,
        "evaluation": """
        Produces metrics, cross-validation results, leaderboard, and diagnostics.
        """,
        "data_prep": """
        Handles missing values, encodes categoricals, validates types, and splits data.
        """
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(AUTOGLUON_API_URL)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            tabular_info = []
            for section in soup.find_all(['div', 'section']):
                text = section.get_text()
                if 'TabularPredictor' in text or 'tabular' in text.lower():
                    for p in section.find_all('p')[:3]:
                        if len(p.get_text(strip=True)) > 50:
                            tabular_info.append(p.get_text(strip=True))
            autogluon_workflow["api_docs_extract"] = tabular_info[:3]
    except Exception as e:
        autogluon_workflow["api_docs_extract"] = [f"Could not fetch API docs: {str(e)}"]

    if specific_step and specific_step in autogluon_workflow:
        result = {
            "step": specific_step,
            "explanation": autogluon_workflow[specific_step],
            "source": "AutoGluon + Docs"
        }
    else:
        result = {
            "full_workflow": autogluon_workflow,
            "available_steps": list(autogluon_workflow.keys()),
            "source": "AutoGluon + Docs"
        }
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_loan_eligibility_metrics() -> str:
    """
    Get information about loan eligibility metrics used in credit risk assessment.
    """
    metrics_info = {
        "AUC": {
            "description": "Area Under the ROC Curve - measures model's ability to distinguish between classes",
            "interpretation": "Higher values (0.5-1.0) indicate better discriminatory power",
            "credit_context": "Critical for separating good vs bad borrowers"
        },
        "Gini": {
            "description": "Gini coefficient - related to AUC, measures inequality in predictions",
            "interpretation": "Gini = 2*AUC - 1, ranges from 0 to 1",
            "credit_context": "Standard metric in credit scoring, >0.3 is typically acceptable"
        },
        "F1": {
            "description": "Harmonic mean of precision and recall",
            "interpretation": "Balances false positives and false negatives",
            "credit_context": "Important when both missed defaults and false rejections are costly"
        },
        "Precision": {
            "description": "True positives / (True positives + False positives)",
            "interpretation": "Minimizes false alarms",
            "credit_context": "High precision reduces unnecessary loan rejections"
        },
        "Recall": {
            "description": "True positives / (True positives + False negatives)",
            "interpretation": "Minimizes missed cases",
            "credit_context": "High recall catches more potential defaults"
        },
        "Accuracy": {
            "description": "Overall correct predictions / Total predictions",
            "interpretation": "General measure of correctness",
            "credit_context": "Less important than AUC/Gini in imbalanced credit datasets"
        }
    }
    
    return json.dumps(metrics_info, indent=2)

@mcp.tool()
async def web_search_credit_metrics(query: str) -> str:
    """
    Simulate web search for credit metrics information.
    In a real implementation, this would call a search API.
    """
    # Simulated responses for different metrics
    responses = {
        "AUC": "AUC (Area Under Curve) is widely used in credit risk modeling. Industry benchmarks suggest AUC > 0.7 is acceptable, > 0.8 is good, and > 0.9 is excellent for credit scoring models.",
        "Gini": "Gini coefficient in credit risk typically ranges from 0.3-0.6 for production models. Values above 0.4 are considered strong discriminatory power in retail credit.",
        "F1": "F1 score balances precision and recall. In credit risk, F1 scores of 0.6-0.8 are typical, but the optimal threshold depends on business objectives and cost of errors.",
        "Accuracy": "While accuracy is intuitive, it can be misleading in credit risk due to class imbalance. Most loans don't default, so high accuracy doesn't guarantee good model performance.",
        "Precision": "High precision in credit scoring means fewer false positives (good customers rejected). Financial institutions balance precision with business growth targets.",
        "Recall": "High recall captures more defaults but may increase false positives. Credit teams optimize recall based on risk appetite and regulatory requirements."
    }
    
    # Simple keyword matching for simulation
    for metric, response in responses.items():
        if metric.lower() in query.lower():
            return response
    
    return f"General credit metrics research for query: {query}. Credit risk models should be evaluated using multiple metrics including AUC, Gini, precision, recall, and business-specific KPIs."

if __name__ == "__main__":
    mcp.run(transport="http")