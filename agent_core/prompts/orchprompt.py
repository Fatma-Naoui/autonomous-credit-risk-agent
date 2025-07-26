from langchain.prompts import PromptTemplate

orchestrator_decision_prompt = PromptTemplate(
    input_variables=["role", "context", "options", "memory", "decision_history"],
    template="""
You are an AI agent with role: {role}

Context:
{context}

Options:
{options}

Memory:
{memory}

Decision history:
{decision_history}

Respond with a JSON:
{{
  "decision": "chosen_option",
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "risk_assessment": "low/medium/high",
  "alternative_plan": "..."
}}
""".strip()
)
