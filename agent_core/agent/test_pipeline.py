import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import asyncio
from agent_core.agent.agenticworkflow import AutonomousPipelineWorkflow

async def main():
    orchestrator = AutonomousPipelineWorkflow()
    arguments = {
        "train_path": "agent_core/data/train_data.csv",
        "test_path": "agent_core/data/test_data.csv",
        "label_column": "loan_status"
    }
    result = await orchestrator.run_pipeline(arguments)
    print("\n🔍 Final Result:\n", result)

if __name__ == "__main__":
    asyncio.run(main())
