import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from agent_core.agent import executor_agent

def main():
    print("Launching Autonomous Risk Agent MCP Pipeline...")
    executor_agent.run_mcp_pipeline()
    print("Pipeline completed.")

if __name__ == "__main__":
    main()