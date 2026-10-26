"""Market tab: BTCUSDT risk score, volatility forecasts, tail risk, event probabilities."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import data as D
from .. import theme as T


def gauge(score: float, level: str, height: int = 250) -> go.Figure:
    color = T.LEVEL_COLORS.get(level, T.BTC)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"font": {"size": 46, "color": "#fff", "family": "Arial, sans-serif"}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 30, 50, 70, 100], "tickcolor": T.MUTED,
                     "tickfont": {"color": T.MUTED, "size": 11}},
            "bar": {"color": color, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(52,211,153,.20)"},
                {"range": [30, 50], "color": "rgba(251,191,36,.20)"},
                {"range": [50, 70], "color": "rgba(247,147,26,.22)"},
                {"range": [70, 100], "color": "rgba(248,113,113,.24)"},
            ],
        },
    ))
    fig.update_layout(height=height, margin=dict(l=24, r=24, t=18, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=T.TEXT))
    return fig


def _freshness_banner(mk: dict, age: float) -> None:
    as_of = mk["data_as_of_utc"][:16]
    if age > 2:
        T.callout(
            f"⏱️ <b>Stale snapshot.</b> The market model was run on candles up to <b>{as_of} UTC</b>, "
            f"which is <b>{age:.1f} days</b> ago. Risk scores, forecasts and probabilities below describe "
            f"that moment, not today. Re-run the market notebook with fresh candles and re-export to update them.",
            "warn")
    else:
        T.callout(f"✅ Data is fresh (as of {as_of} UTC, {age:.1f} days old).", "good")


def render() -> None:
    mk, md = D.load_market_latest(), D.load_market_daily()
    age = D.market_age_days(mk)
    r, f, t, ev = mk["risk"], mk["forecast"], mk["tail_risk_1d"], mk["event_probabilities"]
    level = r["level"]
    lc = T.LEVEL_COLORS[level]

    T.section("Market risk", "Bitcoin (BTC/USDT, Binance spot) risk from volatility, drawdown, downside moves and turnover. "
                             "It measures <b>risk</b>, not price direction.")
    _freshness_banner(mk, age)

    T.kpi_row([
        ("Market risk score", f"{r['score_0_100']:.1f}", f"Level: {level}", lc),
        ("Last close", f"${mk['price']['last_close']:,.0f}", f"{mk['price']['last_day_return']*100:+.2f}% on the day", T.CYAN),
        ("Next-day volatility", f"{f['vol_next_day_daily']*100:.2f}%", f"7-day avg {f['vol_next_7d_daily_avg']*100:.2f}% / day", T.VIOLET),
        ("1-day VaR (95%)", f"{t['var95']*100:.2f}%", f"Expected shortfall {t['es95']*100:.2f}%", T.BTC),
        ("P(|move| ≥ 5%)", f"{ev['abs_move_ge_5pct_next_day']*100:.1f}%", "next day", T.GREEN),
    ])

    # ---- score gauge + decomposition ---------------------------------------------
    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        st.markdown(f"##### Current reading {T.pill(level, lc)}", unsafe_allow_html=True)
        st.plotly_chart(gauge(r["score_0_100"], level), width="stretch", key="mk_gauge")
        st.caption("Bands: Low < 30 · Moderate 30–50 · High 50–70 · Extreme ≥ 70")
    with c2:
        comp = pd.DataFrame({
            "component": [D.COMPONENT_LABELS[k] for k in r["component_percentiles"]],
            "percentile": list(r["component_percentiles"].values()),
            "weight": [r["weights"][k] for k in r["component_percentiles"]],
        })
        comp["contribution"] = comp.percentile * comp.weight
        comp = comp.sort_values("contribution")
        fig = go.Figure(go.Bar(
            x=comp.contribution, y=comp.component, orientation="h",
            marker=dict(color=comp.contribution, colorscale=[[0, "#3B2A12"], [1, T.BTC]]),
            text=[f"{p:.0f}th pct × {w:.0%}" for p, w in zip(comp.percentile, comp.weight)],
            textposition="auto", insidetextfont=dict(color="#fff"),
            hovertemplate="%{y}<br>contributes %{x:.2f} points<extra></extra>"))
        fig.update_layout(title="What makes up the score (points contributed)", xaxis_title="points of the 0–100 score")
        T.style_fig(fig, 262, legend=False)
        st.plotly_chart(fig, width="stretch", key="mk_components")
    st.caption("Score = Σ weight × percentile rank (vs the previous 2 years) of five indicators. "
               f"Top drivers right now: {', '.join(D.COMPONENT_LABELS[d] for d in r['top_drivers'])}.")

    # ---- history ---------------------------------------------------------------------
    T.section("Risk history", "How the score and the BTC price have moved together since 2021.")
    span = st.radio("Window", ["3M", "6M", "1Y", "2Y", "All"], index=3, horizontal=True, key="mk_span",
                    label_visibility="collapsed")
    days = {"3M": 92, "6M": 183, "1Y": 365, "2Y": 730, "All": None}[span]
    h = md if days is None else md.tail(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04)
    for (name, lo, hi), col in zip(D.RISK_LEVELS, [T.GREEN, T.AMBER, T.BTC, T.RED]):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.07, line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=h.date, y=h.market_risk_score, name="Risk score", mode="lines",
                             line=dict(color=T.BTC, width=2), fill="tozeroy", fillcolor="rgba(247,147,26,.10)",
                             hovertemplate="%{x|%d %b %Y}<br>score %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=h.date, y=h.close, name="BTC close (USD)", mode="lines",
                             line=dict(color=T.CYAN, width=1.8),
                             hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(range=[0, 100], title_text="Risk score", row=1, col=1)
    fig.update_yaxes(title_text="BTC (USD)", row=2, col=1)
    T.style_fig(fig, 500)
    st.plotly_chart(fig, width="stretch", key="mk_history")

    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        v = h.copy()
        v["forecast_aligned"] = v["forecast_vol_next_day"].shift(1)  # forecast made at t targets day t+1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v.date, y=v.realised_vol_1d * 100, name="Realised volatility", mode="lines",
                                 line=dict(color="rgba(167,139,250,.55)", width=1.2)))
        fig.add_trace(go.Scatter(x=v.date, y=v.forecast_aligned * 100, name="Forecast made the day before",
                                 mode="lines", line=dict(color=T.BTC, width=2)))
        fig.update_layout(title="Daily volatility: forecast vs realised (%)", yaxis_title="% per day")
        T.style_fig(fig, 330)
        st.plotly_chart(fig, width="stretch", key="mk_vol")
    with c2:
        counts = md["market_risk_level"].value_counts().reindex([n for n, _, _ in D.RISK_LEVELS]).fillna(0)
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62, sort=False,
                               marker=dict(colors=[T.LEVEL_COLORS[x] for x in counts.index]),
                               textinfo="percent", hovertemplate="%{label}: %{value} days<extra></extra>"))
        fig.update_layout(title="Days spent at each level (since 2021)")
        T.style_fig(fig, 330)
        st.plotly_chart(fig, width="stretch", key="mk_levels")

    # ---- tail risk calculator -------------------------------------------------------
    T.section("Tail risk: what could a bad day cost?",
              f"1-day Value-at-Risk (VaR) and Expected Shortfall (ES), issued for {f['forecast_for_day']}.")
    pos = st.number_input("Position size (USD)", min_value=100, max_value=10_000_000, value=10_000, step=500,
                          key="mk_pos")
    tail = pd.DataFrame({
        "Measure": ["VaR 95%", "ES 95%", "VaR 99%", "ES 99%"],
        "Loss (% of position)": [t["var95"], t["es95"], t["var99"], t["es99"]],
        "Loss (USD)": [t["var95"] * pos, t["es95"] * pos, t["var99"] * pos, t["es99"] * pos],
        "Meaning": ["Exceeded on ~1 day in 20", "Average loss on those 1-in-20 days",
                    "Exceeded on ~1 day in 100", "Average loss on those 1-in-100 days"],
    })
    tail["Loss (% of position)"] = tail["Loss (% of position)"].map(lambda x: f"{x*100:.2f}%")
    tail["Loss (USD)"] = tail["Loss (USD)"].map(lambda x: f"${x:,.0f}")
    st.dataframe(tail, width="stretch", hide_index=True)

    # ---- events + price bands -----------------------------------------------------------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        labels = ["|move| ≥ 5% next day", "Drop ≥ 5% next day", "Drawdown ≥ 5% in 7d", "Drawdown ≥ 10% in 7d",
                  "Drawdown ≥ 15% in 7d", "Drawdown ≥ 20% in 7d"]
        vals = [ev["abs_move_ge_5pct_next_day"], ev["drop_ge_5pct_next_day"], ev["drawdown_ge_5pct_next_7d"],
                ev["drawdown_ge_10pct_next_7d"], ev["drawdown_ge_15pct_next_7d"], ev["drawdown_ge_20pct_next_7d"]]
        fig = go.Figure(go.Bar(x=[v_ * 100 for v_ in vals][::-1], y=labels[::-1], orientation="h",
                               marker_color=T.VIOLET, text=[f"{v_*100:.1f}%" for v_ in vals][::-1],
                               textposition="outside", cliponaxis=False))
        fig.update_layout(title="Event probabilities", xaxis_title="probability (%)")
        fig.update_xaxes(range=[0, max(vals) * 100 * 1.25])
        T.style_fig(fig, 340, legend=False)
        st.plotly_chart(fig, width="stretch", key="mk_events")
    with c2:
        b, last = mk["price_bands_7d"], mk["price"]["last_close"]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=["95% band"], x=[b["p97.5"] - b["p2.5"]], base=[b["p2.5"]], orientation="h",
                             marker_color="rgba(34,211,238,.28)", name="95% band",
                             hovertemplate="$%{base:,.0f} → $%{customdata:,.0f}<extra>95% band</extra>",
                             customdata=[b["p97.5"]]))
        fig.add_trace(go.Bar(y=["95% band"], x=[b["p90"] - b["p10"]], base=[b["p10"]], orientation="h",
                             marker_color="rgba(34,211,238,.65)", name="80% band",
                             hovertemplate="$%{base:,.0f} → $%{customdata:,.0f}<extra>80% band</extra>",
                             customdata=[b["p90"]]))
        fig.add_vline(x=last, line_dash="dot", line_color=T.BTC, annotation_text=f"last close ${last:,.0f}",
                      annotation_position="top")
        fig.add_vline(x=b["p50"], line_color="#fff", line_width=1.5)
        fig.update_layout(barmode="overlay", title="7-day price range implied by the volatility model",
                          xaxis_title="BTC price (USD)", yaxis=dict(showticklabels=False))
        fig.update_xaxes(range=[b["p2.5"] * 0.97, b["p97.5"] * 1.03], tickprefix="$")
        T.style_fig(fig, 340)
        st.plotly_chart(fig, width="stretch", key="mk_bands")
        st.caption("A volatility-based range (white line = median), not a directional prediction. "
                   "In very calm regimes these bands tend to be slightly too narrow.")

    # ---- model card ---------------------------------------------------------------------
    with st.expander("🧪 Model card: how the market component was built and tested"):
        mc = mk["model_card"]
        tm = mc["test_metrics"]
        st.markdown(
            f"**Selected models** · next-day vol: `{mc['selected_models']['next_day_vol']}` · "
            f"7-day vol: `{mc['selected_models']['next_7d_vol']}` · tail risk: `{mc['selected_models']['tail_risk']}`  \n"
            f"**Validation:** {mc['validation']}  \n**Does not forecast:** {mc['not_forecast']}")
        met = pd.DataFrame([
            ("1-day vol QLIKE", f"{tm['vol_1d_QLIKE']}", "lower is better"),
            ("1-day vol R² (log)", f"{tm['vol_1d_R2_log']}", "typical for noisy daily vol"),
            ("7-day vol QLIKE", f"{tm['vol_7d_QLIKE']}", "lower is better"),
            ("VaR 95% violations", f"{tm['var95_violation_pct']}%", "target 5%"),
            ("VaR 99% violations", f"{tm['var99_violation_pct']}%", "target 1%"),
            ("Big-move AUC", f"{tm['big_move_AUC']}", "0.5 = no skill"),
            ("Big-move lift", f"{tm['big_move_lift']}×", "vs an average day"),
        ], columns=["Test metric (2025–26)", "Value", "Reading"])
        st.dataframe(met, width="stretch", hide_index=True)
        st.caption(mk["disclaimer"])
