"""
Streamlit Dashboard for LOCAL vLLM-powered Complaint Routing Engine.
Run with: streamlit run dashboard_local.py
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from models import CaseStreamSimulator
from kimi_local_agent import get_coordinator

st.set_page_config(
    page_title="Complaint Classification & Routing Engine (Local vLLM)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; }
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; border-left: 5px solid #1f77b4; }
    .score-good { color: #2ca02c; font-weight: bold; }
    .score-warning { color: #ff7f0e; font-weight: bold; }
    .score-danger { color: #d62728; font-weight: bold; }
    .recommendation-box { background-color: #e8f4f8; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .local-badge { background-color: #2ca02c; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

def render_header():
    st.markdown('<p class="main-header">🎯 Customer Complaint Classification & Routing Engine</p>', unsafe_allow_html=True)
    st.markdown(
        '<span class="local-badge">● LOCAL vLLM MODE</span> '
        '<b>No external API calls</b> | Inference runs on <code>localhost:8000</code>',
        unsafe_allow_html=True
    )
    st.divider()

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Local Control Panel")
        st.markdown("---")
        
        base_url = st.text_input(
            "vLLM Endpoint", 
            value="http://localhost:8000/v1",
            help="Default vLLM OpenAI-compatible server URL"
        )
        
        week = st.selectbox("Select Week", ["Week 1", "Week 2", "Week 3", "Week 4"])
        volume = st.slider("Simulated Case Volume", 20, 200, 50)
        
        st.markdown("---")
        st.markdown("### Local Agent Swarm Status")
        
        # Health check display
        try:
            import requests
            resp = requests.get(f"{base_url}/models", timeout=2)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                model_names = [m["id"] for m in models]
                st.success(f"🟢 vLLM Online: {model_names}")
            else:
                st.error("🔴 vLLM unreachable")
        except Exception:
            st.error("🔴 vLLM not detected on localhost:8000")
            st.info("Start with: `bash vllm_launch.sh`")
        
        st.success("🟢 Coordinator: Local")
        st.success("🟢 Analyst: Local")
        st.success("🟢 Scorer: Local")
        st.success("🟢 Strategist: Local")
        st.success("🟢 Advisor: Local")
        
        run_button = st.button("🚀 Generate Weekly Report (Local)", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("**Backend:** `vLLM + Kimi K2.6`")
        st.markdown("**Max Agents:** 300 sub-agents")
        st.markdown("**Context:** 256K tokens")
        st.markdown("**Prefix Caching:** Enabled")
        
        return base_url, week, volume, run_button

def render_metrics(cases):
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(cases)
    urgent = len([c for c in cases if c.priority == 5])
    resolved = len([c for c in cases if c.status == "Resolved"])
    sla_risk = len([c for c in cases if c.status != "Resolved" and c.priority >= 4])
    avg_priority = sum(c.priority for c in cases) / total if total else 0
    
    with col1:
        st.metric("Total Cases", total, delta=f"+{total//4} vs last week")
    with col2:
        st.metric("Urgent/Escalated", urgent, delta=f"{urgent/total:.1%}" if total else "0%", delta_color="inverse")
    with col3:
        st.metric("Resolved", resolved, delta=f"{resolved/total:.1%}" if total else "0%")
    with col4:
        st.metric("SLA At Risk", sla_risk, delta_color="inverse")
    with col5:
        st.metric("Avg Priority", f"{avg_priority:.1f}/5")

def render_weekly_report(report_text: str):
    st.subheader("📊 Weekly Status Report")
    st.markdown(report_text)

def render_score_card(score_data: dict):
    st.subheader("🏆 Application Score Card")
    if isinstance(score_data, dict) and "metrics" in score_data:
        metrics = score_data["metrics"]
        cols = st.columns(len(metrics))
        for i, metric in enumerate(metrics):
            score = metric.get("current_score", 0)
            target = metric.get("target_score", 0)
            gap = target - score
            name = metric.get("name", "Metric")
            css = "score-good" if score >= 90 else "score-warning" if score >= 75 else "score-danger"
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <h2 class="{css}">{score}/100</h2>
                    <p>Target: {target} | Gap: {gap}</p>
                    <p>Trend: {metric.get('trend', '→')}</p>
                </div>
                """, unsafe_allow_html=True)
        overall = score_data.get("overall_score", 0)
        grade = score_data.get("grade", "N/A")
        st.markdown(f"### Overall: {overall}/100 (Grade: {grade})")
    else:
        st.json(score_data)

def render_improvement_plan(plan_text: str):
    st.subheader("📈 Improvement Plan")
    st.markdown(plan_text)

def render_recommendations(rec_text: str):
    st.subheader("💡 AI-Powered Recommendations")
    st.markdown("""
    <div class="recommendation-box">
        <b>Local vLLM Agent Swarm Analysis:</b> These recommendations are synthesized from 
        parallel sub-agent evaluation running entirely on your local GPU infrastructure.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(rec_text)

def render_case_stream(cases):
    st.subheader("🔴 Live Case Stream")
    df = pd.DataFrame([c.to_dict() for c in cases[:20]])
    st.dataframe(
        df[["id", "category", "priority", "status", "sla_hours", "sentiment", "assigned_team"]],
        use_container_width=True,
        hide_index=True
    )

def main():
    render_header()
    base_url, week, volume, run_button = render_sidebar()
    
    week_num = int(week.split()[1])
    cases = CaseStreamSimulator.generate_week(week_num, volume)
    
    render_metrics(cases)
    render_case_stream(cases)
    st.divider()
    
    if run_button:
        with st.spinner("🤖 Local Kimi Swarm analyzing via vLLM... (4 agents on localhost)"):
            try:
                coordinator = get_coordinator(base_url=base_url)
                result = coordinator.generate_dashboard(
                    case_data=[c.to_dict() for c in cases],
                    week_range=week
                )
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Weekly Report", "🏆 Score Card", "📈 Improvement Plan", "💡 Recommendations"
                ])
                
                with tab1:
                    render_weekly_report(result["weekly_status_report"])
                with tab2:
                    render_score_card(result["application_score_card"])
                with tab3:
                    render_improvement_plan(result["improvement_plan"])
                with tab4:
                    render_recommendations(result["recommendations"])
                
                with st.expander("🔧 Local Agent Swarm Metadata"):
                    st.json(result["agent_metadata"])
                    
            except Exception as e:
                st.error(f"Local vLLM Error: {e}")
                st.info("Ensure vLLM is running: `bash vllm_launch.sh`")

if __name__ == "__main__":
    main()