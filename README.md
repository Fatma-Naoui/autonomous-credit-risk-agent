# Autonomous Multi-Agent Credit Risk Modeling Workflow

This project implements an **autonomous multi-agent system** for credit risk modeling, leveraging advanced large language models and modern AI orchestration tools.

## Features

- **Autonomous Multi-Agent Workflow**  
  Built using AutoGluon, with each agent powered by a **LLaMA 3.3 70B** large language model.

- **Specialized Agents**  
  Designed agents for:
  - Code generation  
  - Debugging  
  - Code execution  
  - Report generation  
  - Tool selection  
  All agents are orchestrated via **LangGraph**.

- **Integrated Tool Access**  
  Exposed internal APIs and web search tools through **MCP (Model Context Protocol)** for agent access.

- **Agentic Retrieval-Augmented Generation (RAG)**  
  Implemented an agentic RAG approach that dynamically decides between:
  - MCP tool usage  
  - **FAISS** vector database retrieval  
  based on the semantic context of the query.

## Technologies Used

- **AutoGluon** – Automated machine learning  
- **LLaMA 3.3 70B** – Large language model  
- **LangGraph** – Agent orchestration  
- **MCP (Model Context Protocol)** – API and tool integration  
- **FAISS** – Semantic search and retrieval  

