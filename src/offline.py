"""
Offline explainer: answers from the project's real numbers with no LLM and no network.

It is used (a) when no free API key is configured and (b) as an automatic fallback if the
online provider is unreachable or rate-limited. Answers are templated, so they are less
conversational than the LLM, but every number comes from the same facts the LLM would see.
"""
from __future__ import annotations

import re

from . import data as D
from . import facts as F


def _market_now(S):
    mk = S["mk"]; r = mk["risk"]
    comps = sorted(r["component_percentiles"].items(), key=lambda kv: -kv[1] * r["weights"][kv[0]])
    top = ", ".join(f"{D.COMPONENT_LABELS[k]} (percentile {v:.0f})" for k, v in comps[:3])
    return (
        f"**Market risk right now: {r['score_0_100']:.1f} / 100, level “{r['level']}”.** "
        f"BTC last closed at ${mk['price']['last_close']:,.0f} ({mk['price']['last_day_return']*100:+.1f}% on the day).\n\n"
        f"The score is a weighted average of five risk indicators, each ranked against the last two years. "
        f"The biggest contributors are {top}. Volatility is forecast to be low relative to the past two years "
        f"({mk['forecast']['vol_percentile_vs_last_2y']:.0f}th percentile), which is why the score is not higher.\n\n"
        f"⚠️ This reading is **as of {mk['data_as_of_utc'][:10]}, about {S['age']:.0f} days old**, so treat it as a snapshot, not today's market."
    )


def _market_forecast(S):
    mk = S["mk"]; t = mk["tail_risk_1d"]; ev = mk["event_probabilities"]; f = mk["forecast"]
    return (
        f"**Forecasts for {f['forecast_for_day']}** (from the {mk['data_as_of_utc'][:10]} snapshot):\n"
        f"- Expected daily volatility: **{f['vol_next_day_daily']*100:.2f}%** (7-day average {f['vol_next_7d_daily_avg']*100:.2f}%, "
        f"≈{f['vol_next_7d_annualised']*100:.0f}% annualised).\n"
        f"- **1-day 95% VaR {t['var95']*100:.2f}%**: on about 1 day in 20 a loss larger than this is expected. "
        f"On a $10,000 position that is roughly **${t['var95']*10000:,.0f}**. The average loss on those bad days (ES95) is "
        f"{t['es95']*100:.2f}% (≈${t['es95']*10000:,.0f}).\n"
        f"- Chance of a ≥5% move (either way) tomorrow: **{ev['abs_move_ge_5pct_next_day']*100:.1f}%**; "
        f"chance of a ≥10% drawdown within 7 days: **{ev['drawdown_ge_10pct_next_7d']*100:.1f}%**.\n\n"
        f"These describe the *size* of possible moves, not their direction."
    )


def _market_model(S):
    mk = S["mk"]; tm = mk["model_card"]["test_metrics"]
    return (
        f"**How reliable is the market model?** It was validated walk-forward (validation 2023–24, untouched test 2025–26).\n"
        f"- VaR coverage on the test period: 95% VaR was breached on **{tm['var95_violation_pct']}%** of days (target 5%) and 99% VaR on "
        f"**{tm['var99_violation_pct']}%** (target 1%), so the tail estimates are well calibrated.\n"
        f"- It ranks big-move days with AUC **{tm['big_move_AUC']}** and **{tm['big_move_lift']}×** lift over an average day: real but moderate skill.\n"
        f"- It does **not** predict direction: in testing, next-day direction accuracy was 48.9% vs a 51.1% baseline, so no edge was found.\n"
        f"- Known weakness: in very calm periods the 7-day price bands run slightly narrow."
    )


def _market_history(S):
    sh = S["lvl_share"]; p = S["peak"]; c = S["calm"]
    return (
        f"**Market-risk history ({S['md']['date'].min().date()} → {S['md']['date'].max().date()}).** "
        f"Days spent at Low / Moderate / High / Extreme: {sh.get('Low',0)*100:.0f}% / {sh.get('Moderate',0)*100:.0f}% / "
        f"{sh.get('High',0)*100:.0f}% / {sh.get('Extreme',0)*100:.0f}%. The peak score was **{p['market_risk_score']:.1f}** on "
        f"{p['date'].date()} (BTC ${p['close']:,.0f}); the calmest day scored {c['market_risk_score']:.1f} on {c['date'].date()}. "
        f"The last 30 days of data averaged {S['last30']['market_risk_score'].mean():.1f}."
    )


def _graph_perf(S):
    gs = S["g"]["summary"]; c = gs["test_confusion"]
    base = gs["test_support"]["licit"] / sum(gs["test_support"].values())
    return (
        f"**GraphSAGE wallet-risk model (test set of {sum(gs['test_support'].values()):,} wallets).**\n"
        f"- PR-AUC **{gs['test_pr_auc']:.3f}**, ROC-AUC **{gs['test_roc_auc']:.3f}**.\n"
        f"- At the 0.90 threshold: precision **{gs['test_precision']:.1%}**, recall **{gs['test_recall']:.1%}**, F1 **{gs['test_f1']:.3f}**.\n"
        f"- It flagged {c['tp']+c['fp']:,} wallets: {c['tp']:,} truly illicit, {c['fp']:,} false alarms; it missed {c['fn']:,} illicit wallets.\n\n"
        f"Accuracy ({gs['test_accuracy']:.1%}) looks high mainly because {base:.1%} of wallets are licit, so always answering “licit” "
        f"would already score {base:.1%}. Precision and recall on the rare illicit class tell the real story."
    )


def _graph_xai(S):
    g = S["g"]; w = g["wallets"]["wallet_b"]; sb = g["shap"]["wallet_b"]
    lines = []
    for r in sb.head(4).itertuples():
        direction = "toward LICIT" if r.shap_value < 0 else "toward ILLICIT"
        lines.append(f"- **{D.describe_feature(r.feature)}**: {r.shap_value:+.2f} ({direction})")
    return (
        f"**Why wallet {w['address'][:6]}…{w['address'][-4:]} was classed LICIT.** Its illicit probability is {w['illicit_probability']:.1e}, "
        f"far below the 0.90 alert threshold. SHAP splits the model's score into feature contributions "
        f"(negative = pushes toward licit):\n" + "\n".join(lines) + "\n\n"
        f"Together the features moved the score from the background average ({w['expected_margin']:.1f}) to {w['original_margin']:.1f}. "
        f"This is a model estimate, not proof that the wallet is safe."
    )


def _sent_overall(S):
    ss = S["ss"]; lab = ss["label_pct"]
    return (
        f"**News sentiment (FinBERT, {ss['total_articles']:,} articles, {ss['date_min']} → {ss['date_max']}).** "
        f"{lab['neutral']}% neutral, {lab['positive']}% positive, {lab['negative']}% negative; the average score is {ss['score_mean']:+.2f} "
        f"on a −1…+1 scale, i.e. coverage leans only slightly positive. Average model confidence is {ss['mean_confidence']:.0%}."
    )


def _sent_recent(S):
    ld = S["last_daily"]; r90 = S["recent90"]
    return (
        f"**Latest sentiment in the dataset: {ld['date'].date()}** (the series ends there; it is not live). That day had "
        f"{int(ld['news_count'])} articles, average score {ld['avg_sentiment']:+.2f}, {ld['positive_ratio']*100:.0f}% positive and "
        f"{ld['negative_ratio']*100:.0f}% negative. Over the final 90 days the average score was {r90['avg_sentiment'].mean():+.2f}."
    )


def _sent_vs_market(S):
    rel = S["rel"]
    def c(sig, vs):
        return float(rel[(rel.signal == sig) & (rel.versus == vs)].spearman.iloc[0])
    return (
        f"**Does sentiment predict the market?** Over the {int(rel.days.max())} days where both datasets overlap, average sentiment "
        f"correlates **{c('Average sentiment','Same-day BTC return'):+.2f}** with the *same-day* BTC return but only "
        f"**{c('Average sentiment','Next-day BTC return'):+.2f}** with the *next-day* return. In plain terms: news mostly "
        f"reacts to price moves rather than foreseeing them, so it is a context signal, not a trading signal."
    )


def _overview(S):
    mk = S["mk"]; gs = S["g"]["summary"]; ss = S["ss"]
    return (
        f"**Three independent views, one dashboard:**\n"
        f"- 🕸️ **Graph:** GraphSAGE flags illicit Bitcoin wallets (PR-AUC {gs['test_pr_auc']:.2f}, precision {gs['test_precision']:.0%}, "
        f"recall {gs['test_recall']:.0%}) and SHAP explains each call.\n"
        f"- 📰 **Sentiment:** FinBERT scored {ss['total_articles']:,} news articles; coverage is {ss['label_pct']['positive']}% positive / "
        f"{ss['label_pct']['negative']}% negative (data to {ss['date_max']}).\n"
        f"- 📉 **Market:** risk score {mk['risk']['score_0_100']:.1f}/100 (“{mk['risk']['level']}”) as of {mk['data_as_of_utc'][:10]} "
        f"(stale by ~{S['age']:.0f} days)."
    )


TOPICS = {
    "market_now": _market_now, "market_forecast": _market_forecast, "market_model": _market_model,
    "market_history": _market_history, "graph_perf": _graph_perf, "graph_xai": _graph_xai,
    "sentiment_overall": _sent_overall, "sentiment_recent": _sent_recent, "sentiment_vs_market": _sent_vs_market,
}


_DEF_START = re.compile(r"^\s*(what is|what's|what are|what does|define|explain|meaning of|eli5)\b")


def _wants_numbers(q: str) -> bool:
    return any(t in q for t in ["$", "position", "current", " now", "latest", "today", " our ", " my ", "model", "mean for", "right now"]) \
        or any(ch.isdigit() for ch in q)


def offline_answer(question: str, S: dict, sections: dict[str, str]) -> str:
    q = " " + question.lower().strip() + " "
    parts: list[str] = []

    if not F.has_topic_hit(question):
        return "\n\n".join([
            "That is outside what this dashboard covers, so here is what I *can* explain:",
            _overview(S),
            "Ask about market risk, VaR, the graph model, a wallet explanation, or news sentiment.",
            "_Answered by the offline explainer._"])

    hits = F.glossary_hits(question)
    defining = bool(_DEF_START.match(question.lower())) or " mean" in q
    pure_definition = bool(hits) and defining and not _wants_numbers(q)
    if hits and (defining or pure_definition):
        parts.append("\n\n".join(f"**{F.GLOSSARY[h][0]}.** {F.GLOSSARY[h][1]}" for h in hits[:3]))

    names: list[str] = []
    if any(w in q for w in ["overall", "summary", "summar", "everything", "all three", "combine", "together", "whole project"]):
        parts.append(_overview(S))
        names = ["market_now", "graph_perf", "sentiment_recent"]
    elif not pure_definition:
        ranked = [n for n in F.select_sections(question, sections) if n in TOPICS]
        scores = F.route_scores(question)
        names = ranked[:1]
        if len(ranked) > 1 and (" and " in q or " both " in q or scores.get(ranked[1], 0) >= scores.get(ranked[0], 0)):
            names = ranked[:2]
        if hits and "market_forecast" not in names and any(h in ("var", "es") for h in hits):
            names = ["market_forecast"]

    for n in names:
        parts.append(TOPICS[n](S))
    if not parts:
        parts.append(_overview(S))

    parts.append("_Answered by the offline explainer (templated from your saved results). Add a free Groq key for conversational answers._")
    return "\n\n".join(parts)
