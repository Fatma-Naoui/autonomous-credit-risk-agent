from langchain.prompts import PromptTemplate

executor_decision_prompt = PromptTemplate(
    input_variables=["role", "context", "options", "memory", "decision_history"],
    template="""
You are an AI agent with role: {role}

Your task is to decide the best execution mode for running a machine learning pipeline based on the context, memory, and historical decisions.

Context:
{context}

Available Options:
{options}

Memory:
{memory}

Recent Decision History:
{decision_history}

Respond with a JSON object:
{{
  "decision": "<one_of_the_options>",
  "reasoning": "...",
  "confidence": 0.0-1.0,
  "risk_assessment": "low|medium|high",
  "alternative_plan": "..."  // Optional fallback plan
}}
"""
)
