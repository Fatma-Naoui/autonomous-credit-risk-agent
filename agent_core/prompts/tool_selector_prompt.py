from langchain.prompts import PromptTemplate

tool_selector_prompt = PromptTemplate.from_template("""\
You are an intelligent tool selector for credit risk report generation.

Section: {section_name}
Focus metric: {focus_metric}
FAISS cache available: {faiss_available}

Use:
• For AutoGluon topics → get_autogluon_tabular_workflow
• If FAISS content is insufficient → search_and_extract

Pick the single best tool for this section.
Respond EXACTLY in one line:

tool_name:<tool_name>; tool_args:<JSON args>; reason:<brief explanation>
""")
