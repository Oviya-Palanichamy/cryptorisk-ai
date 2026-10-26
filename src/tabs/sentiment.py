"""Sentiment tab: FinBERT news sentiment, trends, article explorer, relation to the market."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import data as D
from .. import theme as T

LABEL_ICON = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}


def render() -> None:
    ss = D.load_sentiment_summary()
    daily = D.load_daily_sentiment()
    tabs_ = D.load_sentiment_tables()
    lab = ss["label_pct"]

    T.section("Sentiment intelligence", "<b>FinBERT</b> read every Bitcoin-related news article and scored it positive, neutral or negative.")
    T.callout(f"📅 <b>Historical dataset, not a live feed.</b> The scored news covers {ss['date_min']} → {ss['date_max']}. "
              "Use it to study how coverage mood evolved, not as today's sentiment.", "warn")

    T.kpi_row([
        ("Articles scored", f"{ss['total_articles']:,}", f"{ss['n_sources']} news sources", T.CYAN),
        ("Positive", f"{lab['positive']}%", f"{ss['label_counts']['positive']:,} articles", T.GREEN),
        ("Neutral", f"{lab['neutral']}%", f"{ss['label_counts']['neutral']:,} articles", "#94A3B8"),
        ("Negative", f"{lab['negative']}%", f"{ss['label_counts']['negative']:,} articles", T.RED),
        ("Mean score", f"{ss['score_mean']:+.3f}", "−1 very negative … +1 very positive", T.BTC),
    ])

    # ---- daily trend ----------------------------------------------------------------------
    T.section("Sentiment over time")
    c0, c1 = st.columns([2, 1])
    with c0:
        lo, hi = daily.date.min().date(), daily.date.max().date()
        rng = st.slider("Date range", min_value=lo, max_value=hi, value=(lo, hi), key="s_range")
    with c1:
        smooth = st.radio("Smoothing", ["Raw", "7-day", "30-day"], index=2, horizontal=True, key="s_smooth")
    d = daily[(daily.date.dt.date >= rng[0]) & (daily.date.dt.date <= rng[1])].copy()
    win = {"Raw": 1, "7-day": 7, "30-day": 30}[smooth]
    d["score_s"] = d.avg_sentiment.rolling(win, min_periods=1).mean()
    d["pos_s"] = d.positive_ratio.rolling(win, min_periods=1).mean()
    d["neg_s"] = d.negative_ratio.rolling(win, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.date, y=d.news_count, name="Articles per day", marker_color="rgba(34,211,238,.18)", yaxis="y2",
                         hovertemplate="%{x|%d %b %Y}<br>%{y} articles<extra></extra>"))
    fig.add_trace(go.Scatter(x=d.date, y=d.score_s, name=f"Average sentiment ({smooth})", mode="lines",
                             line=dict(color=T.BTC, width=2.2),
                             hovertemplate="%{x|%d %b %Y}<br>score %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,.35)")
    fig.update_layout(title="Daily average sentiment and news volume",
                      yaxis=dict(title="sentiment score", range=[-0.8, 0.8]),
                      yaxis2=dict(title="articles", overlaying="y", side="right", showgrid=False))
    T.style_fig(fig, 400)
    st.plotly_chart(fig, width="stretch", key="s_trend")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.date, y=d.pos_s * 100, name="Positive", stackgroup="a", mode="lines",
                                 line=dict(width=0.5, color=T.GREEN)))
        neutral = (1 - d.pos_s - d.neg_s).clip(lower=0) * 100
        fig.add_trace(go.Scatter(x=d.date, y=neutral, name="Neutral", stackgroup="a", mode="lines",
                                 line=dict(width=0.5, color="#64748B")))
        fig.add_trace(go.Scatter(x=d.date, y=d.neg_s * 100, name="Negative", stackgroup="a", mode="lines",
                                 line=dict(width=0.5, color=T.RED)))
        fig.update_layout(title="Share of articles by sentiment (%)", yaxis=dict(range=[0, 100]))
        T.style_fig(fig, 340)
        st.plotly_chart(fig, width="stretch", key="s_mix")
    with c2:
        h = ss["score_hist"]
        centers = [(a + b) / 2 for a, b in zip(h["edges"][:-1], h["edges"][1:])]
        colors = [T.RED if c < -0.2 else (T.GREEN if c > 0.2 else "#64748B") for c in centers]
        fig = go.Figure(go.Bar(x=centers, y=h["counts"], marker_color=colors,
                               hovertemplate="score %{x:.2f}<br>%{y:,} articles<extra></extra>"))
        fig.update_layout(title="Distribution of article sentiment scores", xaxis_title="score (positive − negative probability)",
                          yaxis_title="articles")
        T.style_fig(fig, 340, legend=False)
        st.plotly_chart(fig, width="stretch", key="s_hist")

    # ---- heatmap + sources + yearly ---------------------------------------------------------
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        m = daily.copy()
        m["year"], m["month"] = m.date.dt.year, m.date.dt.month
        pv = m.pivot_table(index="year", columns="month", values="avg_sentiment", aggfunc="mean")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = go.Figure(go.Heatmap(z=pv.values, x=[months[i - 1] for i in pv.columns], y=pv.index.astype(str),
                                   colorscale=[[0, T.RED], [0.5, "#1B2233"], [1, T.GREEN]], zmid=0,
                                   colorbar=dict(title="avg score", thickness=10), xgap=2, ygap=2,
                                   hovertemplate="%{y} %{x}<br>%{z:+.3f}<extra></extra>"))
        fig.update_layout(title="Monthly average sentiment heatmap")
        fig.update_yaxes(autorange="reversed", type="category")
        T.style_fig(fig, 330, legend=False)
        st.plotly_chart(fig, width="stretch", key="s_heat")
    with c2:
        bs = tabs_["by_source"].sort_values("articles")
        fig = go.Figure(go.Bar(x=bs.articles, y=bs.source, orientation="h",
                               marker=dict(color=bs.mean_score, colorscale=[[0, T.RED], [0.5, "#334155"], [1, T.GREEN]],
                                           cmid=0, colorbar=dict(title="avg score", thickness=10)),
                               hovertemplate="%{y}<br>%{x:,} articles<extra></extra>"))
        fig.update_layout(title="Top 15 sources (colour = average score)")
        T.style_fig(fig, 330, legend=False)
        st.plotly_chart(fig, width="stretch", key="s_src")

    with st.expander("📊 Yearly summary"):
        y = tabs_["yearly"].copy()
        y["mean_score"] = y.mean_score.map(lambda v: f"{v:+.3f}")
        for c in ["positive", "neutral", "negative"]:
            y[c] = y[c].map(lambda v: f"{v*100:.1f}%")
        y["articles"] = y.articles.map(lambda v: f"{v:,}")
        y.columns = ["Year", "Articles", "Mean score", "Positive", "Negative", "Neutral"]
        st.dataframe(y, width="stretch", hide_index=True)
        st.caption("Article volume grew about 12× between 2019 and 2024, so recent years dominate the overall averages.")

    # ---- sentiment vs market -----------------------------------------------------------------
    T.section("Does news sentiment predict the market?",
              "Spearman rank correlation between daily sentiment and BTC market variables (2021 → Feb 2025, where both datasets overlap).")
    rel = D.sentiment_market_relationship()
    piv = rel.pivot(index="signal", columns="versus", values="spearman")
    order = ["Same-day BTC return", "Next-day BTC return", "Next-day |return| (turbulence)",
             "Same-day market risk score", "Next-day market risk score"]
    piv = piv[order]
    fig = go.Figure(go.Heatmap(z=piv.values, x=[o.replace(" (turbulence)", "") for o in order], y=piv.index,
                               colorscale=[[0, T.RED], [0.5, "#1B2233"], [1, T.GREEN]], zmid=0, zmin=-0.3, zmax=0.3,
                               text=[[f"{v:+.2f}" for v in r] for r in piv.values], texttemplate="%{text}",
                               textfont=dict(size=14, color="#fff"), showscale=False, xgap=3, ygap=3, hoverinfo="skip"))
    fig.update_layout(title="Correlation of sentiment with market variables")
    T.style_fig(fig, 250, legend=False)
    st.plotly_chart(fig, width="stretch", key="s_rel")
    same = float(rel[(rel.signal == "Average sentiment") & (rel.versus == "Same-day BTC return")].spearman.iloc[0])
    nxt = float(rel[(rel.signal == "Average sentiment") & (rel.versus == "Next-day BTC return")].spearman.iloc[0])
    T.callout(f"<b>Reading it.</b> Sentiment moves with the <i>same-day</i> return ({same:+.2f}) but is almost uncorrelated with the "
              f"<i>next-day</i> return ({nxt:+.2f}). News mood mostly <b>reacts</b> to price moves; it did not lead them. "
              "This is an exploratory statistic, not a trading signal.")

    # ---- article explorer ---------------------------------------------------------------------
    T.section("Article explorer", "Browse FinBERT's predictions on individual articles (reproducible random sample of 12,000 of the "
                                  f"{ss['total_articles']:,} scored articles).")
    arts = D.load_articles_sample()
    f1, f2, f3 = st.columns([1.1, 1.1, 1.4])
    with f1:
        labels = st.multiselect("Sentiment", ["positive", "neutral", "negative"], default=["positive", "neutral", "negative"], key="s_lab")
    with f2:
        top_src = arts.source.value_counts().head(25).index.tolist()
        srcs = st.multiselect("Source", top_src, key="s_src_f", placeholder="All sources")
    with f3:
        kw = st.text_input("Search title or text", placeholder="e.g. ETF, halving, hack", key="s_kw")
    yr = st.slider("Published between", min_value=arts.published_on.min().date(), max_value=arts.published_on.max().date(),
                   value=(arts.published_on.min().date(), arts.published_on.max().date()), key="s_yr")
    q = arts[arts.finbert_label.isin(labels) & (arts.published_on.dt.date >= yr[0]) & (arts.published_on.dt.date <= yr[1])]
    if srcs:
        q = q[q.source.isin(srcs)]
    if kw.strip():
        k = kw.strip()
        q = q[q.title.str.contains(k, case=False, regex=False) | q.body.str.contains(k, case=False, regex=False)]
    st.caption(f"{len(q):,} matching articles in the sample")
    if len(q):
        shown = q.sort_values("published_on", ascending=False).head(200).reset_index(drop=True)
        table = pd.DataFrame({
            "Date": shown.published_on.dt.strftime("%Y-%m-%d"),
            "Sentiment": shown.finbert_label.map(lambda s: f"{LABEL_ICON.get(s, '')} {s}"),
            "Score": shown.finbert_sentiment_score.round(3),
            "Source": shown.source,
            "Headline": shown.title,
        })
        st.dataframe(table, width="stretch", hide_index=True, height=300)
        idx = st.number_input("Inspect row #", min_value=1, max_value=len(shown), value=1, step=1, key="s_row")
        a = shown.iloc[int(idx) - 1]
        c1, c2 = st.columns([1.6, 1], gap="large")
        with c1:
            T.html(f"""<div class="card"><div class="kpi-label">{a.source} · {a.published_on:%d %b %Y}</div>
                <div style="font-size:17px;font-weight:700;color:#fff;margin:6px 0 8px 0">{a.title}</div>
                <div class="kpi-sub" style="font-size:13.5px">{a.body}</div></div>""")
        with c2:
            fig = go.Figure(go.Bar(x=["Positive", "Neutral", "Negative"],
                                   y=[a.finbert_positive_prob, a.finbert_neutral_prob, a.finbert_negative_prob],
                                   marker_color=[T.GREEN, "#64748B", T.RED],
                                   text=[f"{v:.0%}" for v in (a.finbert_positive_prob, a.finbert_neutral_prob, a.finbert_negative_prob)],
                                   textposition="outside"))
            fig.update_layout(title=f"FinBERT: {a.finbert_label} (score {a.finbert_sentiment_score:+.2f})", yaxis=dict(range=[0, 1.1], tickformat=".0%"))
            T.style_fig(fig, 270, legend=False)
            st.plotly_chart(fig, width="stretch", key="s_prob")

    with st.expander("🎯 Most confident predictions"):
        mc = D.load_most_confident()
        for lbl, col in [("positive", "finbert_positive_prob"), ("negative", "finbert_negative_prob")]:
            sub = mc[mc.finbert_label == lbl].sort_values(col, ascending=False).head(8)
            st.markdown(f"**{LABEL_ICON[lbl]} Most confidently {lbl}**")
            st.dataframe(pd.DataFrame({"Date": sub.published_on.dt.strftime("%Y-%m-%d"), "Confidence": sub[col].map(lambda v: f"{v:.1%}"),
                                       "Source": sub.source, "Headline": sub.title}),
                         width="stretch", hide_index=True)

    with st.expander("ℹ️ Pipeline & validation status"):
        st.markdown(
            f"""
**Pipeline.** CoinDesk crypto-news dataset (229,172 articles) → Bitcoin-relevance filtering and de-duplication (165,560 articles) → **ProsusAI/FinBERT** probabilities for positive/negative/neutral → score = P(positive) − P(negative) → daily aggregates.

**Model confidence.** Average top-class probability is {ss['mean_confidence']:.0%} (negative {ss['mean_confidence_by_label']['negative']:.0%}, positive {ss['mean_confidence_by_label']['positive']:.0%}, neutral {ss['mean_confidence_by_label']['neutral']:.0%}).
High confidence is not the same as being correct.

**Validation status.** The `finbert_manual_evaluation`, `finbert_independent_evaluation_sample` and `finbert_annotation_50` files in your upload have an empty `human_label` column,
so no FinBERT accuracy has been computed and none is claimed here. Once labelled, accuracy and a confusion matrix can be added.""")
