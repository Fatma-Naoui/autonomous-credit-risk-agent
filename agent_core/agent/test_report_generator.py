import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from agent_core.agent.report_gen_agent import ReportGeneratorAgent

async def main():
    agent = ReportGeneratorAgent()
    
    # Replace with your actual artifact output path if you have one
    artifact_path = "agent_core/models/0d9b1776c56f7d5718acd6b8b58e6b2f5cf1dd21289b2180babd924860f5b6fd"  # or wherever model_leaderboard.csv & metrics live

    report = await agent.generate_report(
        model_type="AutoGluon",
        focus_metrics=["AUC", "Gini", "F1", "Accuracy"],
        artifact_dir=artifact_path
    )

    print("\n=== Generated Report ===\n")
    print(report["report_text"])

if __name__ == "__main__":
    asyncio.run(main())
