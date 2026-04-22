from __future__ import annotations

import os
import json
import subprocess
import sys
from datetime import datetime
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS
from webui.reports import (
    build_report_csv_bytes,
    build_reports_summary_csv_bytes,
    build_pdf_bytes,
    build_structured_report,
    list_saved_reports,
    load_saved_report,
    save_structured_report,
)


load_dotenv()
load_dotenv(".env.enterprise", override=False)


PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "google": None,
    "anthropic": "https://api.anthropic.com/",
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4/",
    "openrouter": "https://openrouter.ai/api/v1",
    "azure": None,
    "ollama": "http://localhost:11434/v1",
}

ANALYST_OPTIONS = [
    ("Market Analyst", "market"),
    ("Social Media Analyst", "social"),
    ("News Analyst", "news"),
    ("Fundamentals Analyst", "fundamentals"),
]
ANALYST_LABELS = {value: label for label, value in ANALYST_OPTIONS}
REPORT_LABELS = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Team Plan",
    "final_trade_decision": "Portfolio Management Decision",
}
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}
FIXED_AGENTS = [
    "Bull Researcher",
    "Bear Researcher",
    "Research Manager",
    "Trader",
    "Aggressive Analyst",
    "Conservative Analyst",
    "Neutral Analyst",
    "Portfolio Manager",
]
AGENT_TEAMS = {
    "Analyst Team": [
        "Market Analyst",
        "Social Analyst",
        "News Analyst",
        "Fundamentals Analyst",
    ],
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": [
        "Aggressive Analyst",
        "Conservative Analyst",
        "Neutral Analyst",
    ],
    "Portfolio Management": ["Portfolio Manager"],
}


def _model_values(provider: str, mode: str) -> list[str]:
    if provider not in MODEL_OPTIONS:
        return []
    return [value for _, value in MODEL_OPTIONS[provider][mode]]


def _build_config(
    provider: str,
    deep_model: str,
    quick_model: str,
    backend_url: str | None,
    depth: int,
    output_language: str,
) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider
    config["deep_think_llm"] = deep_model
    config["quick_think_llm"] = quick_model
    config["max_debate_rounds"] = depth
    config["max_risk_discuss_rounds"] = depth
    config["output_language"] = output_language
    if backend_url:
        config["backend_url"] = backend_url
    return config


def _render_sidebar() -> dict[str, Any]:
    st.sidebar.header("Run Settings")
    provider = st.sidebar.selectbox(
        "LLM Provider",
        options=list(PROVIDER_BASE_URLS.keys()),
        index=list(PROVIDER_BASE_URLS.keys()).index("ollama"),
    )

    quick_choices = _model_values(provider, "quick")
    deep_choices = _model_values(provider, "deep")
    default_quick = quick_choices[0] if quick_choices else DEFAULT_CONFIG["quick_think_llm"]
    default_deep = deep_choices[0] if deep_choices else DEFAULT_CONFIG["deep_think_llm"]

    if quick_choices:
        quick_model = st.sidebar.selectbox("Quick Model", options=quick_choices, index=0)
    else:
        quick_model = st.sidebar.text_input("Quick Model", value=default_quick).strip()

    if deep_choices:
        deep_model = st.sidebar.selectbox("Deep Model", options=deep_choices, index=0)
    else:
        deep_model = st.sidebar.text_input("Deep Model", value=default_deep).strip()

    backend_url_default = PROVIDER_BASE_URLS.get(provider) or ""
    backend_url = st.sidebar.text_input("Backend URL", value=backend_url_default)

    research_depth = st.sidebar.slider("Research Depth", min_value=1, max_value=5, value=1, step=1)
    output_language = st.sidebar.text_input("Output Language", value="English")

    selected_analysts = st.sidebar.multiselect(
        "Analysts",
        options=[value for _, value in ANALYST_OPTIONS],
        default=[value for _, value in ANALYST_OPTIONS],
        format_func=lambda x: ANALYST_LABELS[x],
    )

    return {
        "provider": provider,
        "quick_model": quick_model,
        "deep_model": deep_model,
        "backend_url": backend_url.strip() or None,
        "research_depth": research_depth,
        "output_language": output_language.strip() or "English",
        "selected_analysts": selected_analysts,
    }


def _run_analysis(
    ticker: str,
    trade_date: str,
    settings: dict[str, Any],
    progress_bar,
    status_box,
    logs_box,
    agent_box,
) -> tuple[dict[str, Any], str, list[str]]:
    graph = TradingAgentsGraph(
        selected_analysts=settings["selected_analysts"],
        debug=False,
        config=_build_config(
            provider=settings["provider"],
            deep_model=settings["deep_model"],
            quick_model=settings["quick_model"],
            backend_url=settings["backend_url"],
            depth=settings["research_depth"],
            output_language=settings["output_language"],
        ),
    )

    expected_sections = _expected_sections(settings["selected_analysts"])
    done_sections: set[str] = set()
    logs: list[str] = []
    processed_message_ids: set[str] = set()
    report_sections = {section: None for section in expected_sections}
    agent_status = _init_agent_status(settings["selected_analysts"])

    init_agent_state = graph.propagator.create_initial_state(ticker, trade_date)
    args = graph.propagator.get_graph_args()

    final_state: dict[str, Any] | None = None
    progress_bar.progress(0, text="Started")
    status_box.info("Running agents...")
    _render_agent_status(agent_box, agent_status)

    for chunk in graph.graph.stream(init_agent_state, **args):
        final_state = chunk
        _collect_chunk_logs(chunk, logs, processed_message_ids)
        if logs:
            logs_box.code("\n".join(logs[-25:]), language="text")

        for section in expected_sections:
            if _section_has_content(chunk, section) and report_sections.get(section) is None:
                report_sections[section] = chunk.get(section)
                done_sections.add(section)

        _update_agent_statuses(
            agent_status=agent_status,
            selected_analysts=settings["selected_analysts"],
            report_sections=report_sections,
            chunk=chunk,
        )
        _render_agent_status(agent_box, agent_status)

        progress = int(len(done_sections) * 100 / len(expected_sections))
        progress_bar.progress(
            progress,
            text=f"Progress {progress}% ({len(done_sections)}/{len(expected_sections)})",
        )
        completed_text = ", ".join(REPORT_LABELS[s] for s in sorted(done_sections))
        if not completed_text:
            completed_text = "None yet"
        status_box.info(f"Stage completed: {completed_text}")

    if final_state is None:
        raise RuntimeError("Analysis did not produce a final state.")

    for agent in agent_status:
        agent_status[agent] = "completed"
    _render_agent_status(agent_box, agent_status)

    decision = graph.process_signal(final_state["final_trade_decision"])
    return final_state, decision, logs


def _expected_sections(selected_analysts: list[str]) -> list[str]:
    sections = [
        "investment_plan",
        "trader_investment_plan",
        "final_trade_decision",
    ]
    analyst_map = {
        "market": "market_report",
        "social": "sentiment_report",
        "news": "news_report",
        "fundamentals": "fundamentals_report",
    }
    for analyst in selected_analysts:
        section = analyst_map.get(analyst)
        if section:
            sections.append(section)
    return sections


def _section_has_content(chunk: dict[str, Any], section: str) -> bool:
    value = chunk.get(section)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _extract_content_string(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        text = content.strip()
        return text if text else None
    if isinstance(content, dict):
        text = str(content.get("text", "")).strip()
        return text if text else None
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                item_text = str(item.get("text", "")).strip()
                if item_text:
                    text_parts.append(item_text)
            elif isinstance(item, str):
                item_text = item.strip()
                if item_text:
                    text_parts.append(item_text)
        text = " ".join(text_parts).strip()
        return text if text else None
    text = str(content).strip()
    return text if text else None


def _collect_chunk_logs(
    chunk: dict[str, Any],
    logs: list[str],
    processed_message_ids: set[str],
) -> None:
    for message in chunk.get("messages", []):
        msg_id = getattr(message, "id", None)
        if msg_id and msg_id in processed_message_ids:
            continue
        if msg_id:
            processed_message_ids.add(msg_id)

        content = _extract_content_string(getattr(message, "content", None))
        role = message.__class__.__name__.replace("Message", "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        if content:
            logs.append(f"[{timestamp}] [{role}] {content}")

        for tool_call in getattr(message, "tool_calls", []) or []:
            if isinstance(tool_call, dict):
                name = tool_call.get("name", "tool")
                args = tool_call.get("args", {})
            else:
                name = getattr(tool_call, "name", "tool")
                args = getattr(tool_call, "args", {})
            logs.append(f"[{timestamp}] [Tool] {name}({args})")


def _init_agent_status(selected_analysts: list[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    selected_set = set(selected_analysts)
    for analyst in ANALYST_ORDER:
        if analyst in selected_set:
            status[ANALYST_AGENT_NAMES[analyst]] = "pending"
    for agent in FIXED_AGENTS:
        status[agent] = "pending"
    return status


def _has_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _set_status(status_map: dict[str, str], agent: str, value: str) -> None:
    if agent not in status_map:
        return
    if status_map[agent] == "completed":
        return
    status_map[agent] = value


def _update_agent_statuses(
    agent_status: dict[str, str],
    selected_analysts: list[str],
    report_sections: dict[str, Any],
    chunk: dict[str, Any],
) -> None:
    selected_set = set(selected_analysts)
    found_active = False
    all_analysts_completed = True

    # Analyst team progression
    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected_set:
            continue
        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]
        if _has_text(report_sections.get(report_key)):
            _set_status(agent_status, agent_name, "completed")
        elif not found_active:
            _set_status(agent_status, agent_name, "in_progress")
            found_active = True
            all_analysts_completed = False
        else:
            _set_status(agent_status, agent_name, "pending")
            all_analysts_completed = False

    # Research team progression
    invest_state = chunk.get("investment_debate_state") or {}
    bull_done = _has_text(invest_state.get("bull_history"))
    bear_done = _has_text(invest_state.get("bear_history"))
    judge_done = _has_text(invest_state.get("judge_decision"))

    if all_analysts_completed:
        if bull_done:
            _set_status(agent_status, "Bull Researcher", "completed")
        else:
            _set_status(agent_status, "Bull Researcher", "in_progress")

        if bear_done:
            _set_status(agent_status, "Bear Researcher", "completed")
        elif bull_done:
            _set_status(agent_status, "Bear Researcher", "in_progress")

        if judge_done:
            _set_status(agent_status, "Research Manager", "completed")
        elif bull_done or bear_done:
            _set_status(agent_status, "Research Manager", "in_progress")

    # Trader progression
    trader_done = _has_text(report_sections.get("trader_investment_plan"))
    if trader_done:
        _set_status(agent_status, "Trader", "completed")
    elif judge_done:
        _set_status(agent_status, "Trader", "in_progress")

    # Risk + portfolio progression
    risk_state = chunk.get("risk_debate_state") or {}
    agg_done = _has_text(risk_state.get("aggressive_history"))
    con_done = _has_text(risk_state.get("conservative_history"))
    neu_done = _has_text(risk_state.get("neutral_history"))
    pm_done = _has_text(risk_state.get("judge_decision"))

    if agg_done:
        _set_status(agent_status, "Aggressive Analyst", "completed")
    elif trader_done:
        _set_status(agent_status, "Aggressive Analyst", "in_progress")

    if con_done:
        _set_status(agent_status, "Conservative Analyst", "completed")
    elif agg_done:
        _set_status(agent_status, "Conservative Analyst", "in_progress")

    if neu_done:
        _set_status(agent_status, "Neutral Analyst", "completed")
    elif agg_done or con_done:
        _set_status(agent_status, "Neutral Analyst", "in_progress")

    if pm_done:
        _set_status(agent_status, "Aggressive Analyst", "completed")
        _set_status(agent_status, "Conservative Analyst", "completed")
        _set_status(agent_status, "Neutral Analyst", "completed")
        _set_status(agent_status, "Portfolio Manager", "completed")
    elif agg_done or con_done or neu_done:
        _set_status(agent_status, "Portfolio Manager", "in_progress")


def _render_agent_status(agent_box, agent_status: dict[str, str]) -> None:
    icon = {"pending": "⚪", "in_progress": "🟡", "completed": "🟢"}
    lines = ["### Agent Progress"]
    for team, members in AGENT_TEAMS.items():
        filtered = [m for m in members if m in agent_status]
        if not filtered:
            continue
        lines.append(f"**{team}**")
        for member in filtered:
            state = agent_status.get(member, "pending")
            lines.append(f"- {icon[state]} {member}: `{state}`")
    agent_box.markdown("\n".join(lines))


def run_streamlit_app() -> None:
    st.set_page_config(page_title="TradingAgents WebUI", layout="wide")
    st.title("TradingAgents WebUI")
    st.caption("Run multi-agent trading analysis from your browser.")

    settings = _render_sidebar()

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker", value="NVDA").strip().upper()
    with col2:
        run_date = st.date_input("Analysis Date", value=date.today())

    if not settings["selected_analysts"]:
        st.warning("Select at least one analyst.")
        return

    if st.button("Run Analysis", type="primary", use_container_width=True):
        if not ticker:
            st.error("Ticker is required.")
            return

        progress_bar = st.progress(0, text="Waiting")
        status_box = st.empty()
        agent_box = st.empty()
        logs_box = st.empty()

        with st.spinner("Running analysis..."):
            try:
                final_state, decision, logs = _run_analysis(
                    ticker=ticker,
                    trade_date=run_date.isoformat(),
                    settings=settings,
                    progress_bar=progress_bar,
                    status_box=status_box,
                    logs_box=logs_box,
                    agent_box=agent_box,
                )
            except Exception as exc:
                st.exception(exc)
                return

        structured_report = build_structured_report(
            ticker=ticker,
            trade_date=run_date.isoformat(),
            settings=settings,
            final_state=final_state,
            decision=decision,
            logs=logs,
        )
        report_path = save_structured_report(structured_report)
        report_json = json.dumps(structured_report, indent=2, ensure_ascii=False, default=str)
        report_pdf = build_pdf_bytes(structured_report)

        st.success("Analysis complete.")
        progress_bar.progress(100, text="Completed 100%")
        st.caption(f"Report saved: `{report_path}`")
        st.subheader("Final Decision")
        st.code(decision)

        st.subheader("Key Reports")
        report_keys = [
            ("Market", "market_report"),
            ("Social Sentiment", "sentiment_report"),
            ("News", "news_report"),
            ("Fundamentals", "fundamentals_report"),
            ("Investment Plan", "investment_plan"),
            ("Trader Plan", "trader_investment_plan"),
            ("Final Trade Decision", "final_trade_decision"),
        ]
        for label, key in report_keys:
            content = final_state.get(key)
            if content:
                with st.expander(label, expanded=(key == "final_trade_decision")):
                    st.markdown(str(content))

        st.download_button(
            "Download structured JSON report",
            data=report_json,
            file_name=f"{structured_report['report_id']}.json",
            mime="application/json",
        )
        st.download_button(
            "Download PDF report",
            data=report_pdf,
            file_name=f"{structured_report['report_id']}.pdf",
            mime="application/pdf",
        )

        with st.expander("Structured Data Preview"):
            st.json(structured_report["outputs"])

    st.divider()
    st.subheader("History Reports")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        ticker_filter = st.text_input("Filter ticker", value="").strip().upper()
    with col_f2:
        decision_filter = st.text_input("Filter decision contains", value="").strip().lower()
    with col_f3:
        history_limit = st.slider("History limit", min_value=10, max_value=200, value=50, step=10)

    history = list_saved_reports(limit=history_limit)
    if ticker_filter:
        history = [h for h in history if str(h.get("ticker", "")).upper() == ticker_filter]
    if decision_filter:
        history = [h for h in history if decision_filter in str(h.get("decision", "")).lower()]

    if not history:
        st.caption("No matching reports.")
        return

    st.download_button(
        "Download filtered summary CSV",
        data=build_reports_summary_csv_bytes(history),
        file_name="reports_summary.csv",
        mime="text/csv",
        key="hist_summary_csv",
    )

    selected = st.selectbox(
        "Select a report",
        options=list(range(len(history))),
        format_func=lambda i: (
            f"{history[i]['created_at']} | {history[i]['ticker']} | "
            f"{history[i]['trade_date']} | {history[i]['report_id']}"
        ),
    )
    selected_meta = history[selected]
    selected_report = load_saved_report(selected_meta["path"])
    st.caption(f"Stored file: `{selected_meta['path']}`")
    st.code((selected_report.get("summary", {}) or {}).get("decision", ""))
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download selected JSON",
            data=json.dumps(selected_report, indent=2, ensure_ascii=False, default=str),
            file_name=f"{selected_report.get('report_id', 'report')}.json",
            mime="application/json",
            key=f"hist_json_{selected_report.get('report_id', selected)}",
        )
    with col_b:
        st.download_button(
            "Download selected PDF",
            data=build_pdf_bytes(selected_report),
            file_name=f"{selected_report.get('report_id', 'report')}.pdf",
            mime="application/pdf",
            key=f"hist_pdf_{selected_report.get('report_id', selected)}",
        )
    st.download_button(
        "Download selected CSV",
        data=build_report_csv_bytes(selected_report),
        file_name=f"{selected_report.get('report_id', 'report')}.csv",
        mime="text/csv",
        key=f"hist_csv_{selected_report.get('report_id', selected)}",
    )


def main() -> None:
    script_path = Path(__file__).resolve()
    command = [sys.executable, "-m", "streamlit", "run", str(script_path)]
    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    run_streamlit_app()
