import json
from datetime import datetime
from agent_core.prompts.reportprompt import loan_model_report_prompt
from agent_core.agent.mcp_client import run_tool
from agent_core.agent.llm_client import generate_code
from pathlib import Path
import asyncio


class ReportGeneratorAgent:
    def __init__(self):
        self.name = "ReportGeneratorAgent"

    async def generate_report(self, model_type="AutoGluon", focus_metrics=None, artifact_dir=None):
        if focus_metrics is None:
            focus_metrics = ["AUC", "Gini", "F1", "Accuracy"]

        # === Fetch MCP knowledge ===
        autogluon_info = await run_tool("get_autogluon_tabular_workflow", {"specific_step": None})
        credit_metrics_info = await run_tool("get_loan_eligibility_metrics", {})

        # Fallback if tools failed
        autogluon_info = autogluon_info if isinstance(autogluon_info, str) else "Autogluon info unavailable."
        credit_metrics_info = credit_metrics_info if isinstance(credit_metrics_info, str) else "Credit metrics info unavailable."

        # === Web search for each metric, safely ignore None responses ===
        metric_search_raw = await asyncio.gather(*[
            run_tool("web_search_credit_metrics", {"query": f"{metric} loan eligibility"})
            for metric in focus_metrics
        ])

        metric_searches = []
        for i, response in enumerate(metric_search_raw):
            if isinstance(response, str):
                metric_searches.append(response)
            else:
                metric_searches.append(f"No result found for {focus_metrics[i]}.")

        # === Load leaderboard and top 3 model metrics ===
        leaderboard_summary, top_model_metrics = "", ""
        if artifact_dir:
            leaderboard_path = Path(artifact_dir) / "model_leaderboard.csv"
            metrics_dir = Path(artifact_dir)

            try:
                import pandas as pd
                df = pd.read_csv(leaderboard_path)
                df_sorted = df.sort_values("score_val", ascending=False).head(3)
                leaderboard_summary = df_sorted[["model", "score_val"]].to_string(index=False)

                metrics = []
                for model_name in df_sorted["model"]:
                    metrics_path = metrics_dir / f"model_metrics_{model_name}.json"
                    if metrics_path.exists():
                        with open(metrics_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        metrics.append(f"{model_name}:\n" + "\n".join([f"  {k}: {v}" for k, v in data.items()]))
                    else:
                        metrics.append(f"{model_name}: No metrics file found.")
                top_model_metrics = "\n\n".join(metrics)

            except Exception as e:
                leaderboard_summary = f"Failed to load leaderboard: {e}"
                top_model_metrics = f"Failed to load model metrics: {e}"

        # === Build the final report prompt ===
        prompt = loan_model_report_prompt.format(
            model_type=model_type,
            leaderboard_summary=leaderboard_summary,
            top_model_metrics=top_model_metrics,
            autogluon_info=autogluon_info,
            credit_metrics_info=credit_metrics_info,
            metric_details="\n\n".join(metric_searches)
        )

        # === Generate report ===
        report = generate_code(prompt)
        return {
            "generated_at": datetime.now().isoformat(),
            "report_text": report,
            "metrics": focus_metrics,
            "used_artifacts": str(artifact_dir) if artifact_dir else "N/A"
        }
