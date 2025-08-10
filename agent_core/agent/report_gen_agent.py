import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent_core.agent.llm_client import generate_code
from agent_core.rag.vectorstore import VectorDBHandler
from agent_core.agent.tool_selector_agent import ToolSelectorAgent


# ------------- Helpers: dataset + prompt hashing -------------
def compute_dataset_hash(artifact_dir: str) -> str:
    """
    Hash all files in artifact_dir (excluding report_output/) to fingerprint the dataset.
    Safe for caching decisions at the workflow level.
    """
    if not artifact_dir or not os.path.isdir(artifact_dir):
        return "no_artifacts"
    sha = hashlib.sha256()
    for root, _, files in os.walk(artifact_dir):
        for fname in sorted(files):
            path = os.path.join(root, fname)
            if "report_output" in Path(path).parts:
                continue
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    sha.update(chunk)
    return sha.hexdigest()


def _compute_prompt_hash(parts: List[str]) -> str:
    sha = hashlib.sha256()
    for p in parts:
        if p:
            sha.update(p.encode("utf-8"))
            sha.update(b"|")
    # short hash is enough for caching
    return sha.hexdigest()[:16]


def _safe_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        import tiktoken as _t
        enc = _t.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def create_focused_summary(content: str, max_tokens: int = 300, focus_area: Optional[str] = None) -> str:
    if not content or not content.strip():
        return "Content not available."
    if _safe_token_count(content) <= max_tokens:
        return content

    sentences = [s.strip() + "." for s in content.split(".") if s.strip()]

    keywords_map = {
        "autogluon": ["autogluon", "ensemble", "stacking", "validation", "automation", "training"],
        "metrics": ["auc", "gini", "f1", "accuracy", "precision", "recall", "performance"],
        "regulatory": ["regulatory", "compliance", "risk", "financial", "standard"],
        "benchmarking": ["model", "benchmark", "performance", "comparison", "evaluation"],
        "machine_learning_advantages": ["nonlinear", "features", "generalize", "automation", "bias", "drift"],
    }
    kws = keywords_map.get(focus_area, []) if focus_area else []
    if focus_area and not kws:
        kws = [w.lower() for w in focus_area.split()]

    seen = set()
    scored = []
    for s in sentences:
        lw = s.lower()
        norm = lw.replace(" ", "")
        if norm in seen:
            continue
        seen.add(norm)
        sc = sum(2 for kw in kws if kw in lw)
        if any(w in lw for w in ["important", "significant", "key", "critical"]):
            sc += 1
        if "%" in s and any(ch.isdigit() for ch in s):
            sc += 1
        if len(s.split()) < 25:
            sc += 0.5
        scored.append((sc, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    out, used = [], 0
    for _, sent in scored:
        tok = _safe_token_count(sent)
        if used + tok <= max_tokens:
            out.append(sent)
            used += tok
        if len(out) >= 8:
            break
    return " ".join(out) if out else sentences[0]


# Single source of truth for the explainer instruction (used + hashed)
LEADERBOARD_EXPLANATION_INSTR = (
    "You are a senior ML engineer. Explain the AutoGluon leaderboard results in business terms for credit risk. "
    "Without greeting nor talking too much, just explain the results thoroughly. "
    "Use only the information shown below."
)

CONCLUSION_INSTR_BASE = (
    "Write a concise business-impact conclusion focusing on {metrics} for an {model_type} credit risk modeling report. "
    "No greetings, no filler."
)


class ReportGeneratorAgent:
    def __init__(self, max_tokens_per_section: int = 1000):
        self.vector_db = VectorDBHandler()
        self.selector = ToolSelectorAgent()
        self.max_tokens_per_section = max_tokens_per_section

    async def clean_tool_content(self, raw: str, focus: str) -> str:
        if not raw or not raw.strip():
            return "Content not available."

        # unwrap {"result": "..."} if present
        if "{" in raw and "}" in raw:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "result" in obj:
                    raw = obj["result"]
            except Exception:
                pass

        compression_prompt = f"""
        Clean and summarize for a financial risk modeling report.
        Focus: {focus}
        - Remove duplicates, logs, and noise.
        - Keep it formal, concise, accurate.
        - Do not invent facts.

        Text:
        {raw}
        """
        try:
            return generate_code(compression_prompt, max_tokens=800)
        except Exception:
            return raw[:1000]

    async def process_artifacts(self, artifact_dir: str) -> Dict[str, object]:
        """
        - Render FULL AutoGluon leaderboard CSV as HTML
        - Pull top-3 models
        - Load per-model TEST metrics JSONs
        - Generate a concise, no-greeting leaderboard explanation
        """
        res: Dict[str, object] = {
            "leaderboard_table": "",
            "leaderboard_models": [],
            "model_metrics": [],
            "top_model_insights": "",
            "leaderboard_explanation": "",
            # also return prompt parts used here so caller can hash for caching
            "prompt_parts": [],
        }
        if not artifact_dir or not os.path.isdir(artifact_dir):
            return res

        lb_path = os.path.join(artifact_dir, "model_leaderboard.csv")
        if not os.path.isfile(lb_path):
            return res

        # Load leaderboard (entire CSV)
        df = pd.read_csv(lb_path)
        if "score_val" in df.columns:
            df = df.sort_values("score_val", ascending=False)

        # Format numerics
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].apply(lambda s: s.round(4))

        # HTML table (no Markdown) — FULL table (no head truncation)
        res["leaderboard_table"] = df.to_html(
            index=False,
            escape=False,
            classes="autogluon-lb",
            border=0,
        )

        # Top-3 model names
        if "model" in df.columns:
            res["leaderboard_models"] = df["model"].head(3).tolist()

        # Load TEST metrics for those models
        metrics_list: List[Dict[str, object]] = []
        for model_name in res["leaderboard_models"]:
            metrics_file = None
            for fn in os.listdir(artifact_dir):
                if fn.startswith(f"model_metrics_{model_name}") and fn.endswith(".json"):
                    metrics_file = os.path.join(artifact_dir, fn)
                    break
            if metrics_file and os.path.isfile(metrics_file):
                with open(metrics_file, "r") as f:
                    test_metrics = json.load(f)
                metrics_display = []
                for metric_name in ["accuracy", "f1", "auc", "gini", "precision"]:
                    if metric_name in test_metrics:
                        try:
                            metrics_display.append(f"{metric_name.upper()}: {float(test_metrics[metric_name]):.4f}")
                        except Exception:
                            metrics_display.append(f"{metric_name.upper()}: {test_metrics[metric_name]}")
                metrics_list.append({
                    "name": model_name,
                    "metrics": " | ".join(metrics_display),
                    "full_metrics": test_metrics,
                })
        res["model_metrics"] = metrics_list

        # Compact insights (used in prompts or optional display)
        insights = []
        for i, m in enumerate(metrics_list):
            insights.append(f"**Model {i+1}: {m['name']} (Test Set Performance)**\n{m['metrics']}")
        res["top_model_insights"] = "\n\n".join(insights)

        # Leaderboard explanation — keep it direct, no greeting
        leaderboard_explanation_prompt = f"""
        {LEADERBOARD_EXPLANATION_INSTR}

        LEADERBOARD DATA:
        {res["leaderboard_table"]}
        """
        # record prompt instruction for hashing
        res["prompt_parts"].append(LEADERBOARD_EXPLANATION_INSTR)

        try:
            lb_text = generate_code(leaderboard_explanation_prompt, max_tokens=600)
            lb_text = re.sub(
                r"^\s*(?:good\s+(?:morning|afternoon|evening)|hello|hi|dear\s+(?:team|all|colleagues)|greetings)[^\n\.]*[\.!\n]+\s*",
                "",
                lb_text,
                flags=re.IGNORECASE,
            )
            res["leaderboard_explanation"] = lb_text
        except Exception:
            res["leaderboard_explanation"] = (
                "AutoGluon ranks models by validation performance as a proxy for generalization; "
                "ensembles often lead by combining complementary errors."
            )

        return res

    async def gather_content_sections(self) -> Dict[str, str]:
        async def fetch(sec: str, focus: str):
            r = await self.selector.decide_and_execute(sec, focus)
            cleaned = await self.clean_tool_content(r.get("content", ""), focus)
            return create_focused_summary(cleaned, self.max_tokens_per_section, focus)

        out = {
            "ml_fundamentals": await fetch("credit_risk_fundamentals", "machine_learning_advantages"),
            "autogluon_summary": await fetch("autogluon_workflow", "autogluon"),
        }
        return out

    async def generate_report(
        self,
        model_type: str = "AutoGluon",
        focus_metrics: Optional[List[str]] = None,
        artifact_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        if focus_metrics is None:
            focus_metrics = ["AUC", "Gini", "F1", "Accuracy"]

        # Paths / template pre-read (needed for prompt hash)
        base_path = Path(__file__).resolve().parent.parent
        tmpl_path = base_path / "templates"
        template_name = "credit_report_template.html"
        if not tmpl_path.is_dir():
            raise FileNotFoundError(f"Template folder not found: {tmpl_path}")
        template_path = tmpl_path / template_name
        template_text = template_path.read_text(encoding="utf-8")

        # Build the static prompt parts to hash (no manual version strings)
        conclusion_instr = CONCLUSION_INSTR_BASE.format(
            metrics=", ".join(focus_metrics[:4]),
            model_type=model_type,
        )

        # dataset hash
        dh = compute_dataset_hash(artifact_dir or "")

        # preliminary artifact processing to get the leaderboard instruction into hash parts
        art = await self.process_artifacts(artifact_dir or "")

        # prompt hash (template + instruction strings only; no data-dependent content)
        prompt_parts = [
            LEADERBOARD_EXPLANATION_INSTR,
            CONCLUSION_INSTR_BASE,      # base pattern (stable shape)
            conclusion_instr,           # concrete instruction this run
            template_text,              # template content — if you edit it, cache invalidates
        ] + (art.get("prompt_parts", []) or [])
        ph = _compute_prompt_hash(prompt_parts)

        # Fingerprint & cache check
        outd = os.path.join(artifact_dir or ".", "report_output")
        os.makedirs(outd, exist_ok=True)
        html_path = os.path.join(outd, "report.html")
        fp_path = os.path.join(outd, "fingerprint.txt")
        fingerprint = hashlib.sha256(f"{dh}|{ph}".encode("utf-8")).hexdigest()

        if os.path.isfile(html_path) and os.path.isfile(fp_path):
            try:
                if Path(fp_path).read_text(encoding="utf-8").strip() == fingerprint:
                    # ✅ Up-to-date: skip regeneration entirely
                    return {
                        "status": "cached",
                        "html_path": html_path,
                        "dataset_hash": dh,
                        "prompt_hash": ph,
                        "generated_at": datetime.now().isoformat(),
                    }
            except Exception:
                # if fingerprint read fails, fall through to regenerate
                pass

        # (Only now) gather content sections (can be more expensive)
        sect = await self.gather_content_sections()

        # Conclusion (LLM) — keep no-greeting guard
        try:
            concl = generate_code(conclusion_instr, max_tokens=250)
            concl = re.sub(
                r"^\s*(?:good\s+(?:morning|afternoon|evening)|hello|hi|dear\s+(?:team|all|colleagues)|greetings)[^\n\.]*[\.!\n]+\s*",
                "",
                concl,
                flags=re.IGNORECASE,
            )
        except Exception:
            concl = ""

        # Render HTML
        env = Environment(
            loader=FileSystemLoader(str(tmpl_path)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        tpl = env.get_template(template_name)
        html = tpl.render(
            model_type=model_type,
            ml_fundamentals=sect["ml_fundamentals"],
            autogluon_summary=sect["autogluon_summary"],
            leaderboard_table=art["leaderboard_table"],
            leaderboard_models=art["leaderboard_models"],
            model_metrics=art["model_metrics"],
            leaderboard_explanation=art["leaderboard_explanation"],
            final_conclusion=concl,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Persist fingerprint so future runs can skip
        try:
            Path(fp_path).write_text(fingerprint, encoding="utf-8")
        except Exception:
            pass

        return {
            "status": "success",
            "html_path": html_path,
            "dataset_hash": dh,
            "prompt_hash": ph,
            "generated_at": datetime.now().isoformat(),
        }
