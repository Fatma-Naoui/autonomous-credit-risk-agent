import subprocess
import time

def run_script(script_name):
    print(f"\nRunning {script_name}...\n")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        print(f"❌ {script_name} failed. Check logs above.\n")
        return False
    print(f"✅ {script_name} completed successfully.\n")
    return True

def main():
    start_time = time.time()

    steps = [
        ("code_generator_agent.py", "Generating pipeline code with LLM..."),
        ("debugger_agent.py", "Debugging generated code until it runs successfully..."),
        ("generated_pipeline.py", "Running the final generated AutoGluon pipeline..."),
        ("app.py", "Launching Streamlit dashboard for visualization...")
    ]

    for script, description in steps:
        print(f"\n {description}")
        success = run_script(script)
        if not success:
            print(f"⚠️ Halting pipeline due to failure in {script}.")
            return

    elapsed = time.time() - start_time
    print(f"\n All steps completed in {elapsed:.2f} seconds. Your autonomous agent pipeline is ready.\n")

if __name__ == "__main__":
    main()
