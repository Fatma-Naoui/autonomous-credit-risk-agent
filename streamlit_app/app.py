import streamlit as st
import pandas as pd
import json
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import plotly.express as px
import plotly.graph_objects as go
import pdfkit
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent  # Go up one level from streamlit_app
sys.path.insert(0, str(project_root))

# Import your workflow
from agent_core.agent.agenticworkflow import AutonomousPipelineWorkflow


def run_async(coro):
    """Run an async coroutine safely from Streamlit (handles nested event loop)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        import nest_asyncio  # type: ignore
        nest_asyncio.apply()
        return loop.run_until_complete(coro)


class StreamlitMLDashboard:
    def __init__(self):
        self.workflow = AutonomousPipelineWorkflow()

    def apply_custom_css(self):
        """Apply clean, professional CSS styling"""
        st.markdown("""
        <style>
        /* Clean professional theme */
        .stApp {
            background-color: #f8fafc;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1e3a8a;
        }
        
        .css-1d391kg .stMarkdown h1,
        .css-1d391kg .stMarkdown h2,
        .css-1d391kg .stMarkdown h3 {
            color: white;
        }
        
        .css-1d391kg .stMarkdown p {
            color: #cbd5e1;
        }
        
        /* Main content area */
        .main .block-container {
            padding: 2rem;
            max-width: 1200px;
        }
        
        /* Header */
        .main-header {
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            color: white;
            padding: 3rem 2rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            margin: 0;
        }
        
        .main-header p {
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        /* Cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #2563eb;
        }
        
        .stDownloadButton > button {
            background-color: #059669;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        
        .stDownloadButton > button:hover {
            background-color: #047857;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f1f5f9;
            border-radius: 8px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #475569;
            font-weight: 500;
            border-radius: 4px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3b82f6;
            color: white;
        }
        
        /* File uploader */
        .stFileUploader {
            border: 2px dashed #cbd5e1;
            border-radius: 8px;
            padding: 1rem;
        }
        
        /* Success/Error messages */
        .stSuccess {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
        }
        
        .stError {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            color: #dc2626;
        }
        
        .stWarning {
            background-color: #fffbeb;
            border: 1px solid #fed7aa;
            color: #d97706;
        }
        
        .stInfo {
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #2563eb;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    # ---------- Small helpers ----------

    def _get_artifacts_dir(self, artifacts_path: str) -> Path:
        base = Path(artifacts_path)
        actual = self._find_dataset_artifacts_dir(base)
        return actual or base

    def _top_models(self, artifacts_dir: Path, n: int = 3) -> List[str]:
        """
        Choose top-n models from model_leaderboard.csv by score_val.
        Fallback to models detected from existing image files.
        """
        lb_path = artifacts_dir / "model_leaderboard.csv"
        if lb_path.exists():
            try:
                df = pd.read_csv(lb_path)
                if {"model", "score_val"} <= set(df.columns):
                    return (
                        df.sort_values("score_val", ascending=False)["model"]
                          .head(n).tolist()
                    )
                # fallback if score_val missing but there is a model column
                if "model" in df.columns:
                    return df["model"].head(n).tolist()
            except Exception:
                pass

        # Fallback: infer from files
        names = []
        for p in sorted(artifacts_dir.glob("confusion_matrix_*.png")):
            m = p.stem.replace("confusion_matrix_", "")
            if m not in names:
                names.append(m)
        return names[:n]

    # ---------- Page & Uploads ----------

    def setup_page(self):
        st.set_page_config(
            page_title="ML AutoPipeline Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Apply custom CSS
        self.apply_custom_css()

        # Clean header
        st.markdown("""
        <div class="main-header">
            <h1>🤖 ML AutoPipeline Dashboard</h1>
            <p>Autonomous Machine Learning Pipeline</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar settings
        st.sidebar.markdown("## ⚙️ Settings")

        default_upload_dir = project_root / "uploads"
        default_artifacts_dir = project_root / "artifacts"

        upload_dir = st.sidebar.text_input(
            "Upload Directory",
            value=str(default_upload_dir),
            help="Directory where uploaded files will be stored"
        )

        artifacts_dir = st.sidebar.text_input(
            "Artifacts Directory",
            value=str(default_artifacts_dir),
            help="Directory where pipeline artifacts will be stored"
        )

        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

        st.session_state.upload_dir = upload_dir
        st.session_state.artifacts_dir = artifacts_dir

        with st.sidebar.expander("📍 Storage Locations"):
            st.text(f"Uploads: {Path(upload_dir).name}")
            st.text(f"Artifacts: {Path(artifacts_dir).name}")

    def file_upload_section(self):
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📁 Data Upload")

        train_file = st.sidebar.file_uploader(
            "Training Data", type=["csv"], key="train_upload"
        )
        test_file = st.sidebar.file_uploader(
            "Test Data", type=["csv"], key="test_upload"
        )
        label_column = st.sidebar.text_input(
            "Label Column", placeholder="e.g., target, label, class"
        )

        st.sidebar.markdown("## 🚀 Run Options")
        force_rerun = st.sidebar.checkbox("Force re-run (ignore cache)")

        st.session_state.force_rerun = force_rerun
        return train_file, test_file, label_column

    def save_uploaded_file(self, uploaded_file, suffix: str) -> Optional[str]:
        if uploaded_file is None:
            return None

        upload_dir = Path(getattr(st.session_state, "upload_dir", project_root / "uploads"))
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_hash = str(abs(hash(uploaded_file.getvalue())))[:8]
        original_name = uploaded_file.name.split(".")[0]
        file_path = upload_dir / f"{original_name}_{suffix}_{file_hash}.csv"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success(f"✅ {suffix.title()} file saved")
        return str(file_path)

    async def run_pipeline_async(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self.workflow.run_pipeline(arguments)

    # ---------- Views ----------

    def display_leaderboard(self, artifacts_path: str):
        artifacts_dir = self._get_artifacts_dir(artifacts_path)

        leaderboard_files = []
        for pattern in ["model_leaderboard.csv", "leaderboard*.csv", "*leaderboard*.csv"]:
            leaderboard_files.extend(list(artifacts_dir.glob(pattern)))

        if not leaderboard_files:
            st.warning("Model leaderboard not found")
            return

        try:
            leaderboard_path = leaderboard_files[0]
            df = pd.read_csv(leaderboard_path)

            st.subheader("📊 Model Leaderboard")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Enhanced bar chart with gradient
            score_columns = [c for c in df.columns if "score" in c.lower() or "accuracy" in c.lower()]
            if len(df) > 0 and score_columns:
                score_col = score_columns[0]
                
                # Create bar chart with gradient colors based on performance
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df["model"] if "model" in df.columns else df.iloc[:, 0],
                    y=df[score_col],
                    marker=dict(
                        color=df[score_col],
                        colorscale='Blues',
                        colorbar=dict(title=score_col.replace('_', ' ').title()),
                        line=dict(color='rgba(59, 130, 246, 0.8)', width=1)
                    ),
                    text=[f'{val:.3f}' for val in df[score_col]],
                    textposition='auto',
                    textfont=dict(color='white', size=12)
                ))
                
                fig.update_layout(
                    title=f"Model Performance ({score_col.replace('_', ' ').title()})",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    height=400,
                    xaxis=dict(tickangle=-45),
                    title_font=dict(size=16, color='#1e3a8a')
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading leaderboard: {e}")

    def display_metrics(self, artifacts_path: str):
        artifacts_dir = self._get_artifacts_dir(artifacts_path)
        st.subheader("🎯 Model Metrics")

        top = self._top_models(artifacts_dir, n=3)
        selected_files = []
        for m in top:
            f = artifacts_dir / f"model_metrics_{m}.json"
            if f.exists():
                selected_files.append(f)

        if not selected_files:
            selected_files = sorted(artifacts_dir.glob("model_metrics_*.json"))[:3]

        if not selected_files:
            st.info("No model metrics found")
            return

        cols = st.columns(len(selected_files))
        for idx, metric_file in enumerate(selected_files):
            try:
                with open(metric_file, "r") as f:
                    metrics = json.load(f)
                model_name = metric_file.stem.replace("model_metrics_", "")
                
                with cols[idx]:
                    st.markdown(f"#### {model_name}")
                    if isinstance(metrics, dict):
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                st.metric(k.replace("_", " ").title(), f"{v:.4f}")
                            else:
                                st.text(f"{k}: {v}")
                                
            except Exception as e:
                st.error(f"Error loading metrics: {e}")

    def display_confusion_matrices(self, artifacts_path: str):
        artifacts_dir = self._get_artifacts_dir(artifacts_path)
        st.subheader("🔄 Confusion Matrices")

        top = self._top_models(artifacts_dir, n=3)
        if not top:
            st.info("No confusion matrices found")
            return

        cols = st.columns(len(top))
        for i, m in enumerate(top):
            img = artifacts_dir / f"confusion_matrix_{m}.png"
            with cols[i]:
                st.markdown(f"**{m}**")
                if img.exists():
                    st.image(str(img), use_container_width=True)
                else:
                    st.warning(f"Missing: {img.name}")

    def display_roc_pr_curves(self, artifacts_path: str):
        artifacts_dir = self._get_artifacts_dir(artifacts_path)
        top = self._top_models(artifacts_dir, n=3)

        st.subheader("📈 ROC Curves")
        if top:
            cols = st.columns(len(top))
            for i, m in enumerate(top):
                img = artifacts_dir / f"roc_curve_{m}.png"
                with cols[i]:
                    st.markdown(f"**{m}**")
                    if img.exists():
                        st.image(str(img), use_container_width=True)
                    else:
                        st.warning(f"Missing: {img.name}")
        else:
            st.info("No ROC curves found")

        st.subheader("🎯 Precision-Recall Curves")
        if top:
            cols = st.columns(len(top))
            for i, m in enumerate(top):
                img = artifacts_dir / f"precision_recall_curve_{m}.png"
                with cols[i]:
                    st.markdown(f"**{m}**")
                    if img.exists():
                        st.image(str(img), use_container_width=True)
                    else:
                        st.warning(f"Missing: {img.name}")
        else:
            st.info("No Precision-Recall curves found")

    def display_feature_importance(self, artifacts_path: str):
        artifacts_dir = self._get_artifacts_dir(artifacts_path)
        st.subheader("⭐ Feature Importance")
        
        img = artifacts_dir / "feature_importance.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            alt = next(iter(sorted(artifacts_dir.glob("*importance*.png"))), None)
            if alt:
                st.image(str(alt), use_container_width=True)
            else:
                st.info("Feature importance plot not found")

    def _find_dataset_artifacts_dir(self, artifacts_path: Path) -> Optional[Path]:
        if not artifacts_path.exists():
            return None

        direct_files = list(artifacts_path.glob("model_leaderboard.csv"))
        if direct_files:
            return artifacts_path

        subdirs = [d for d in artifacts_path.iterdir() if d.is_dir()]
        for subdir in subdirs:
            if any(subdir.glob(pattern) for pattern in [
                "model_leaderboard.csv", "*.json", "*.png", "stdout.log", "stderr.log"
            ]):
                return subdir
        return None

    # ---------- Export ----------

    def html_to_pdf(self, html_path: str) -> Optional[str]:
        try:
            pdf_path = html_path.replace(".html", ".pdf")
            options = {
                "page-size": "A4",
                "margin-top": "0.75in",
                "margin-right": "0.75in",
                "margin-bottom": "0.75in",
                "margin-left": "0.75in",
                "encoding": "UTF-8",
                "no-outline": None,
                "enable-local-file-access": None
            }
            pdfkit.from_file(html_path, pdf_path, options=options)
            return pdf_path
        except Exception as e:
            st.error(f"Error converting HTML to PDF: {e}")
            return None

    def create_download_button(self, file_path: str, filename: str, label: str):
        try:
            with open(file_path, "rb") as file:
                file_data = file.read()
            st.download_button(
                label=label,
                data=file_data,
                file_name=filename,
                mime="application/pdf" if filename.endswith(".pdf") else "text/html"
            )
        except Exception as e:
            st.error(f"Error creating download button: {e}")

    # ---------- Main ----------

    def main(self):
        self.setup_page()

        train_file, test_file, label_column = self.file_upload_section()

        if not all([train_file, test_file, label_column]):
            st.info("👆 Please upload training data, test data, and specify the label column to begin")
            return

        if st.sidebar.button("🚀 Run Pipeline", type="primary"):
            with st.spinner("Saving files..."):
                train_path = self.save_uploaded_file(train_file, "train")
                test_path = self.save_uploaded_file(test_file, "test")

            if not train_path or not test_path:
                st.error("Error saving files")
                return

            st.session_state.train_path = train_path
            st.session_state.test_path = test_path
            st.session_state.label_column = label_column

            # Show configuration
            with st.container():
                st.markdown("### 🔍 Configuration")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Training:** {Path(train_path).name}")
                with col2:
                    st.info(f"**Test:** {Path(test_path).name}")
                with col3:
                    st.info(f"**Target:** {label_column}")

            with st.spinner("🤖 Running ML pipeline..."):
                try:
                    arguments = {
                        "train_path": train_path,
                        "test_path": test_path,
                        "label_column": label_column,
                        "artifacts_path": getattr(st.session_state, "artifacts_dir", str(project_root / "artifacts")),
                        "force_rerun": bool(getattr(st.session_state, "force_rerun", False)),
                    }

                    result = run_async(self.run_pipeline_async(arguments))
                    st.session_state.pipeline_result = result

                    art_from_result = (
                        result.get("artifacts_path")
                        or result.get("agent_summary", {}).get("artifacts_path")
                        or arguments["artifacts_path"]
                    )
                    st.session_state.artifacts_path = art_from_result

                    st.success("🎉 Pipeline completed successfully!")

                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    return

        # Display results
        if hasattr(st.session_state, "pipeline_result") and st.session_state.pipeline_result:
            result = st.session_state.pipeline_result
            artifacts_path = st.session_state.artifacts_path

            # Pipeline summary
            with st.expander("📋 Pipeline Summary", expanded=True):
                agent_summary = result.get("agent_summary", {})
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    status = "✅ Success" if agent_summary.get("pipeline_ready") else "❌ Failed"
                    st.metric("Status", status)
                with col2:
                    st.metric("Agents", agent_summary.get("total_agents_involved", "N/A"))
                with col3:
                    st.metric("Messages", agent_summary.get("agent_messages_count", "N/A"))
                with col4:
                    st.metric("Report", agent_summary.get("report_status", "N/A"))

            # Results tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Leaderboard", "🎯 Metrics", "🔄 Confusion", "📈 Curves", "⭐ Features"
            ])

            with tab1:
                self.display_leaderboard(artifacts_path)
            with tab2:
                self.display_metrics(artifacts_path)
            with tab3:
                self.display_confusion_matrices(artifacts_path)
            with tab4:
                self.display_roc_pr_curves(artifacts_path)
            with tab5:
                self.display_feature_importance(artifacts_path)

            # Report download
            st.markdown("---")
            st.subheader("📄 Reports")
            
            report_info = result.get("report", {})
            html_path = report_info.get("path", "")

            if html_path and Path(html_path).exists():
                col1, col2 = st.columns(2)
                with col1:
                    self.create_download_button(
                        html_path,
                        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        "📄 Download HTML"
                    )
                with col2:
                    if st.button("📑 Convert to PDF"):
                        with st.spinner("Converting..."):
                            pdf_path = self.html_to_pdf(html_path)
                            if pdf_path and Path(pdf_path).exists():
                                self.create_download_button(
                                    pdf_path,
                                    f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    "📑 Download PDF"
                                )
                            else:
                                st.error("PDF conversion failed")
            else:
                st.warning("No report available")


# Run the application
if __name__ == "__main__":
    dashboard = StreamlitMLDashboard()
    dashboard.main()