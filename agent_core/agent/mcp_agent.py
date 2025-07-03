import yaml
import subprocess
import time
import os
import sys

def load_mcp_schema(schema_path=None):
    if schema_path is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_path = os.path.join(root_dir, "mcp_schema.yaml")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"❌ mcp_schema.yaml not found at: {schema_path}")
    with open(schema_path, 'r') as f:
        return yaml.safe_load(f)

def is_task_active(schema, task_name):
    for task in schema.get('tasks', []):
        if task.get('name') == task_name:
            return task.get('active', False)
    return False

def run_task(script_name, description):
    print(f"\n🚀 {description} [{script_name}]\n")

    script_path = os.path.join(os.path.dirname(__file__), script_name)

    if not os.path.isfile(script_path):
        print(f"❌ {script_name} not found at {script_path}. Stopping MCP pipeline.\n")
        sys.exit(1)

    # Use sys.executable to ensure it runs inside the current venv
    result = subprocess.run([sys.executable, script_path])

    if result.returncode != 0:
        print(f"❌ {script_name} failed. Stopping MCP pipeline.\n")
        sys.exit(1)

    print(f"✅ {script_name} completed.\n")

def run_mcp_pipeline():
    schema = load_mcp_schema()
    start_time = time.time()

    if is_task_active(schema, "code_generation"):
        run_task("llm_code_generator.py", "Generating pipeline code with LLM")

    if is_task_active(schema, "debugging"):
        run_task("debugger_agent.py", "Debugging generated pipeline code")

    if is_task_active(schema, "pipeline_run"):
        run_task("generated_pipeline.py", "Running generated AutoGluon pipeline")

    if is_task_active(schema, "dashboard"):
        run_task("app.py", "Launching Streamlit dashboard for visualization")

    elapsed = time.time() - start_time
    print(f"\n✅ MCP pipeline completed in {elapsed:.2f} seconds.\n")

if __name__ == "__main__":
    run_mcp_pipeline()
