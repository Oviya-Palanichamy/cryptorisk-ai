"""Overview tab: the whole project at a glance."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import data as D
from .. import theme as T
from .market import gauge


def _component_card(icon: str, name: str, status: str, status_color: str, headline: str, caption: str, lines: list[str]) -> str:
    items = "".join(f"<li>{x}</li>" for x in lines)
    return (f'<div class="card glow"><div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div style="font-size:15px;font-weight:700;color:#fff">{icon} {name}</div>{T.pill(status, status_color)}</div>'
            f'<div class="kpi-value" style="font-size:34px;margin-top:14px">{headline}</div>'
            f'<div class="kpi-sub">{caption}</div>'
            f'<ul style="margin:12px 0 0 0;padding-left:18px;color:#9AA6BC;font-size:12.8px;line-height:1.7">{items}</ul></div>')


def render() -> None:
    mk, g, ss = D.load_market_latest(), D.load_graph(), D.load_sentiment_summary()
    gs, daily = g["summary"], D.load_daily_sentiment()
    age = D.market_age_days(mk)
    r = mk["risk"]
    last = daily.iloc[-1]

    T.section("Project at a glance", "Three independent AI components, each explainable, plus an analyst that answers questions about all of them.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        T.html(_component_card(
            "🕸️", "Graph · wallets", "Static benchmark", T.CYAN,
            f"PR-AUC {gs['test_pr_auc']:.2f}", "GraphSAGE on the Bitcoin wallet graph",
            [f"{gs['num_nodes']:,} wallets · {gs['num_edges']:,} links",
             f"Precision {gs['test_precision']:.0%} · recall {gs['test_recall']:.0%}",
             "SHAP explains every flagged wallet"]))
    with c2:
        T.html(_component_card(
            "📰", "Sentiment · news", f"Data ends {ss['date_max']}", T.AMBER,
            f"{ss['label_pct']['positive']}% / {ss['label_pct']['negative']}%", "positive vs negative share of articles",
            [f"{ss['total_articles']:,} articles · {ss['n_sources']} sources",
             f"Mean score {ss['score_mean']:+.3f} (−1…+1)",
             f"Last day: {last['avg_sentiment']:+.2f} on {last['date']:%d %b %Y}"]))
    with c3:
        stale = age > 2
        T.html(_component_card(
            "📉", "Market · BTC", f"Stale · {age:.0f}d old" if stale else "Fresh", T.RED if stale else T.GREEN,
            f"{r['score_0_100']:.1f} · {r['level']}", f"risk score as of {mk['data_as_of_utc'][:10]}",
            [f"Last close ${mk['price']['last_close']:,.0f}",
             f"Next-day vol {mk['forecast']['vol_next_day_daily']*100:.2f}% · VaR95 {mk['tail_risk_1d']['var95']*100:.2f}%",
             "Forecasts risk, not price direction"]))

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        T.section("How it fits together")
        T.html("""
        <div class="flow">
          <div class="step"><b>1 · Data</b><span>Bitcoin wallet graph, CoinDesk news, Binance BTC/USDT candles</span></div>
          <div class="arrow">›</div>
          <div class="step"><b>2 · Models</b><span>GraphSAGE · FinBERT · HAR-RV volatility &amp; tail-risk</span></div>
          <div class="arrow">›</div>
          <div class="step"><b>3 · Explain</b><span>SHAP, score decomposition, calibration checks</span></div>
          <div class="arrow">›</div>
          <div class="step"><b>4 · Ask</b><span>Free LLM analyst grounded in these results</span></div>
        </div>""")
        T.section("Key findings")
        rel = D.sentiment_market_relationship()
        same = float(rel[(rel.signal == "Average sentiment") & (rel.versus == "Same-day BTC return")].spearman.iloc[0])
        nxt = float(rel[(rel.signal == "Average sentiment") & (rel.versus == "Next-day BTC return")].spearman.iloc[0])
        base = gs["test_support"]["licit"] / sum(gs["test_support"].values())
        tm = mk["model_card"]["test_metrics"]
        st.markdown(
            f"""
- **Graph:** the model finds **{gs['test_recall']:.0%}** of illicit wallets with **{gs['test_precision']:.0%}** precision. Headline accuracy ({gs['test_accuracy']:.1%}) is inflated by class imbalance ({base:.1%} licit).
- **Market:** volatility tails are well calibrated (VaR95 breached on **{tm['var95_violation_pct']}%** of test days vs 5% target), but **no directional edge** exists, so the component forecasts *risk*, not price.
- **Sentiment:** coverage is only mildly positive overall (mean {ss['score_mean']:+.2f}). Sentiment tracks the *same-day* return ({same:+.2f}) but not the *next* day's ({nxt:+.2f}): news reacts, it does not predict.
- **Data freshness:** the market snapshot is **{age:.0f} days old** and the news series stops at {ss['date_max']}; refresh them for current conditions.""")
    with right:
        st.markdown("##### Market risk now")
        st.plotly_chart(gauge(r["score_0_100"], r["level"], 235), width="stretch", key="ov_gauge")
        T.callout("💬 <b>Try the AI Analyst tab.</b> Ask things like “Why is the market risk Moderate?”, “What does a 95% VaR of 3.16% mean for "
                  "$10,000?” or “Explain why wallet 3KGiq1… was classed licit.”")

    T.section("Data coverage")
    cov = pd.DataFrame([
        ("Graph", "Elliptic++-style wallet graph", f"{gs['num_nodes']:,} wallets", "static benchmark", "✅ complete"),
        ("Sentiment", "CoinDesk news + FinBERT", f"{ss['total_articles']:,} articles", f"{ss['date_min']} → {ss['date_max']}", "ℹ️ historical"),
        ("Market", "Binance BTC/USDT (1h → daily)", f"{len(D.load_market_daily()):,} days", f"2021-01-01 → {mk['data_as_of_utc'][:10]}",
         f"⚠️ stale ({age:.0f}d)" if age > 2 else "✅ fresh"),
    ], columns=["Component", "Source", "Size", "Period", "Status"])
    st.dataframe(cov, width="stretch", hide_index=True)
    st.caption(mk["disclaimer"] + " Graph outputs are model estimates, not proof of wrongdoing.")
