import json
import re
from typing import Dict, Any, Optional, Tuple

from agent_core.agent.llm_client import generate_code
from agent_core.agent.mcp_client import run_tool
from agent_core.rag.vectorstore import VectorDBHandler


class ToolSelectorAgent:
    def __init__(self):
        self.vector_db = VectorDBHandler()

    # Canonical names and simple aliases
    TOOL_NAME_ALIASES: Dict[str, str] = {
        "autogluon": "get_autogluon_tabular_workflow",
        "autogluon_tabular": "get_autogluon_tabular_workflow",
        "get_autogluon": "get_autogluon_tabular_workflow",
    }

    # Arg schema per tool — MUST match MCP tool signatures exactly
    TOOL_SCHEMAS: Dict[str, Dict[str, set]] = {
        "get_autogluon_tabular_workflow": {
            "required": set(),
            "optional": set(),  # strict: no args
        },
        "search_and_extract": {
            "required": {"query"},
            "optional": {"num_results", "content_type", "lr"},
        },
    }

    DEFAULT_TOOL = "search_and_extract"

    def _normalize_tool_name(self, name: str) -> str:
        name = (name or "").strip()
        return self.TOOL_NAME_ALIASES.get(name, name)

    def _extract_decision(self, resp: str) -> Dict[str, Any]:
        """
        Expect a line like: tool_name:<name>; tool_args:<JSON>; reason:<text>
        Falls back to DEFAULT_TOOL on parse failure.
        """
        m = re.search(
            r"tool_name\s*:\s*([\w_]+)\s*;\s*tool_args\s*:\s*(\{.*?\})\s*;\s*reason\s*:\s*(.*)",
            resp,
            flags=re.DOTALL,
        )
        if not m:
            return {"tool_name": self.DEFAULT_TOOL, "tool_args": {}, "reason": ""}
        name = self._normalize_tool_name(m.group(1))
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        reason = (m.group(3) or "").strip()
        return {"tool_name": name, "tool_args": args, "reason": reason}

    def _get_llm_decision(self, section_name: str, focus_metric: Optional[str]) -> Dict[str, Any]:
        from agent_core.prompts.tool_selector_prompt import tool_selector_prompt
        prompt = tool_selector_prompt.format(
            section_name=section_name,
            focus_metric=focus_metric or "None",
            faiss_available="yes" if self.vector_db.vstore else "no",
        )
        resp = generate_code(prompt)
        return self._extract_decision(resp)

    def _build_default_args(self, tool_name: str, section_name: str, focus_metric: Optional[str]) -> Dict[str, Any]:
        if tool_name == "search_and_extract":
            bits = [section_name]
            if focus_metric:
                bits.append(str(focus_metric))
            bits.append("credit risk metrics significance")
            return {
                "query": " ".join(bits),
                "num_results": 5,
                "content_type": "metrics",
                "lr": "en-US",
            }
        return {}

    def _shape_args(
        self,
        tool_name: str,
        raw_args: Dict[str, Any],
        section_name: str,
        focus_metric: Optional[str],
    ) -> Tuple[str, Dict[str, Any], list]:
        """
        Enforce per-tool schema and fill sensible defaults.
        Output dict includes ONLY keys accepted by the MCP tool.
        """
        schema = self.TOOL_SCHEMAS.get(tool_name)
        if not schema:
            tool_name = self.DEFAULT_TOOL
            schema = self.TOOL_SCHEMAS[tool_name]
            raw_args = {}

        required = schema["required"]
        optional = schema["optional"]

        # Start from clean defaults for this tool
        args = self._build_default_args(tool_name, section_name, focus_metric)

        # Overlay LLM-provided args but drop anything not in required|optional
        for k, v in (raw_args or {}).items():
            if k in required or k in optional:
                args[k] = v

        # Required check
        missing = [k for k in required if k not in args]
        # Final pruning to the exact accepted keys
        accepted_keys = required | optional
        args = {k: args[k] for k in args.keys() if k in accepted_keys}
        return tool_name, args, missing

    async def decide_and_execute(
        self,
        section_name: str,
        focus_metric: Optional[str] = None,
        max_distance: float = 0.35,
        force_fetch: bool = False,
    ) -> Dict[str, Any]:
        """
        1) Cache-first via FAISS top1 on "<section_name> <focus_metric>".
        2) Else, ask LLM which tool to run, enforce schema, run via MCP.
        3) Cache the result back to FAISS.
        """
        query = f"{section_name} {focus_metric or ''}".strip()
        is_general = section_name.strip().lower() in {
            "credit_risk_fundamentals",
            "why credit risk modeling needs machine learning",
        }

        # FAISS cache
        if not force_fetch and self.vector_db.vstore:
            hit = self.vector_db.top1_with_score(query)
            if hit:
                doc, dist = hit
                threshold = 0.45 if is_general else max_distance
                if dist <= threshold:
                    return {"content": doc.page_content, "tool": "faiss", "cached": True, "score": float(dist)}

        # LLM tool selection
        decision = self._get_llm_decision(section_name, focus_metric)
        tool_name = self._normalize_tool_name(decision.get("tool_name") or "")
        raw_args = decision.get("tool_args") or {}

        # Enforce schema
        tool_name, tool_args, missing = self._shape_args(tool_name, raw_args, section_name, focus_metric)

        # If missing required args, fall back to default search
        if missing:
            tool_name = self.DEFAULT_TOOL
            _, tool_args, _ = self._shape_args(tool_name, {}, section_name, focus_metric)

        # Ensure {} for strict/no-arg tool
        schema = self.TOOL_SCHEMAS[tool_name]
        if not schema["required"] and not schema["optional"]:
            tool_args = {}

        # Execute via MCP with safe retry
        try:
            resp = await run_tool(tool_name, tool_args)
        except Exception:
            if tool_name == "search_and_extract":
                resp = await run_tool(tool_name, self._build_default_args(tool_name, section_name, focus_metric))
            else:
                resp = await run_tool(tool_name, {})

        # Unpack tool response
        if isinstance(resp, dict) and "structuredContent" in resp:
            content = resp["structuredContent"].get("result", "")
        elif isinstance(resp, dict) and "result" in resp:
            content = str(resp["result"])
        else:
            content = json.dumps(resp)

        # Cache for next time
        try:
            self.vector_db.add_section(section_name, content)
        except Exception:
            pass

        return {"content": content, "tool": tool_name, "cached": False, "score": None}
