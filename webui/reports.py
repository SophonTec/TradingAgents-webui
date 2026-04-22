from __future__ import annotations

import io
import json
import os
import uuid
import csv
from datetime import datetime
from pathlib import Path
from typing import Any
from html import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


def _reports_dir() -> Path:
    base = os.getenv(
        "TRADINGAGENTS_REPORTS_DIR",
        os.path.join(os.path.expanduser("~"), ".tradingagents", "webui_reports"),
    )
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    return str(value)


def _report_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"rep_{stamp}_{uuid.uuid4().hex[:6]}"


def build_structured_report(
    *,
    ticker: str,
    trade_date: str,
    settings: dict[str, Any],
    final_state: dict[str, Any],
    decision: str,
    logs: list[str],
) -> dict[str, Any]:
    report_id = _report_id()
    created_at = datetime.now().isoformat(timespec="seconds")

    sections = {
        "market_report": final_state.get("market_report"),
        "sentiment_report": final_state.get("sentiment_report"),
        "news_report": final_state.get("news_report"),
        "fundamentals_report": final_state.get("fundamentals_report"),
        "investment_plan": final_state.get("investment_plan"),
        "trader_investment_plan": final_state.get("trader_investment_plan"),
        "final_trade_decision": final_state.get("final_trade_decision"),
    }
    investment = final_state.get("investment_debate_state", {}) or {}
    risk = final_state.get("risk_debate_state", {}) or {}

    report = {
        "report_id": report_id,
        "created_at": created_at,
        "ticker": ticker,
        "trade_date": trade_date,
        "settings": _sanitize(settings),
        "summary": {
            "decision": decision,
            "log_count": len(logs),
        },
        "outputs": {
            "sections": _sanitize(sections),
            "investment_debate": _sanitize(investment),
            "risk_debate": _sanitize(risk),
        },
        "telemetry": {
            "logs": _sanitize(logs),
        },
        "raw_state": _sanitize(final_state),
    }
    return report


def save_structured_report(report: dict[str, Any]) -> Path:
    path = _reports_dir() / f"{report['report_id']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_saved_reports(limit: int = 30) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _reports_dir().glob("rep_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append(
            {
                "report_id": data.get("report_id", path.stem),
                "created_at": data.get("created_at", ""),
                "ticker": data.get("ticker", "N/A"),
                "trade_date": data.get("trade_date", "N/A"),
                "decision": (data.get("summary", {}) or {}).get("decision", ""),
                "path": str(path),
            }
        )
    records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return records[:limit]


def load_saved_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pdf_bytes(report: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"TradingAgents Report {report.get('report_id', '')}",
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = styles["Heading2"]
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
    )
    mono = ParagraphStyle(
        "Mono",
        parent=styles["Code"],
        fontSize=8.8,
        leading=11,
    )

    story = []
    story.append(Paragraph("TradingAgents Analysis Report", title_style))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            escape(
                f"Report ID: {report.get('report_id', '')} | "
                f"Created: {report.get('created_at', '')} | "
                f"Ticker: {report.get('ticker', '')} | "
                f"Trade Date: {report.get('trade_date', '')}"
            ),
            meta_style,
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph("Table of Contents", h2))
    toc_items = [
        "1. Summary",
        "2. Detailed Sections",
        "3. Runtime Logs",
    ]
    for item in toc_items:
        story.append(Paragraph(escape(item), styles["Normal"]))
    story.append(PageBreak())

    summary = report.get("summary", {}) or {}
    story.append(Paragraph("1. Summary", h2))
    story.append(Paragraph(escape(str(summary.get("decision", ""))), styles["Normal"]))
    story.append(Spacer(1, 8))

    sections = ((report.get("outputs", {}) or {}).get("sections", {}) or {})
    label_map = {
        "market_report": "Market Analysis",
        "sentiment_report": "Social Sentiment",
        "news_report": "News Analysis",
        "fundamentals_report": "Fundamentals Analysis",
        "investment_plan": "Research Team Decision",
        "trader_investment_plan": "Trading Team Plan",
        "final_trade_decision": "Portfolio Management Decision",
    }
    story.append(Paragraph("2. Detailed Sections", h2))
    for key, label in label_map.items():
        content = sections.get(key)
        if not content:
            continue
        story.append(Paragraph(escape(label), styles["Heading3"]))
        story.append(Preformatted(str(content), mono))
        story.append(Spacer(1, 5))

    logs = ((report.get("telemetry", {}) or {}).get("logs", []) or [])
    if logs:
        story.append(PageBreak())
        story.append(Paragraph("3. Runtime Logs (last 120 lines)", h2))
        for line in logs[-120:]:
            story.append(Preformatted(str(line), mono))

    doc.build(story)
    return buf.getvalue()


def build_report_csv_bytes(report: dict[str, Any]) -> bytes:
    rows: list[tuple[str, str]] = []

    def add_row(key: str, value: Any) -> None:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        rows.append((key, "" if value is None else str(value)))

    add_row("report_id", report.get("report_id"))
    add_row("created_at", report.get("created_at"))
    add_row("ticker", report.get("ticker"))
    add_row("trade_date", report.get("trade_date"))
    add_row("decision", (report.get("summary", {}) or {}).get("decision"))
    add_row("log_count", (report.get("summary", {}) or {}).get("log_count"))

    sections = ((report.get("outputs", {}) or {}).get("sections", {}) or {})
    for key, value in sections.items():
        add_row(f"section.{key}", value)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["field", "value"])
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def build_reports_summary_csv_bytes(records: list[dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["report_id", "created_at", "ticker", "trade_date", "decision", "path"])
    for row in records:
        writer.writerow(
            [
                row.get("report_id", ""),
                row.get("created_at", ""),
                row.get("ticker", ""),
                row.get("trade_date", ""),
                row.get("decision", ""),
                row.get("path", ""),
            ]
        )
    return buf.getvalue().encode("utf-8")
