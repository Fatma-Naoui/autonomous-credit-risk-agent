import asyncio
import sys
import os

# Add project root to sys.path using os.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from agent_core.agent.report_gen_agent import ReportGeneratorAgent

async def main():
    agent = ReportGeneratorAgent()
    artifact_path = os.path.join(project_root, "agent_core", "agent", "models",
                                 "0d9b1776c56f7d5718acd6b8b58e6b2f5cf1dd21289b2180babd924860f5b6fd")

    report = await agent.generate_report(
        model_type="AutoGluon",
        focus_metrics=["AUC", "Gini", "F1", "Accuracy"],
        artifact_dir=artifact_path
    )

    print("\n=== Generated Report ===\n")
    # The report now returns html_path instead of report_text
    print(f"HTML report saved at: {report['html_path']}")

if __name__ == "__main__":
    asyncio.run(main())