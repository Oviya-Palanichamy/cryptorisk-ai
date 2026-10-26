"""
Turns the saved project results into short, factual text sections.

The AI analyst NEVER answers from memory about your results: it receives only the sections
relevant to the question, built here from the real files. This keeps the prompt small enough
for free-tier token limits and keeps answers grounded.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd

from . import data as D

GLOSSARY = {
    "var": ("Value at Risk (VaR)",
            "A loss threshold that should only be exceeded on a small share of days. "
            "A 95% 1-day VaR of 3.16% means: on about 1 day in 20, BTC may lose more than 3.16%."),
    "es": ("Expected Shortfall (ES)",
           "The average loss on the bad days that go beyond VaR. It describes how bad the tail is, "
           "not just where it starts."),
    "qlike": ("QLIKE",
              "A loss function for volatility forecasts. Lower is better; it penalises under-predicting "
              "volatility more than over-predicting it."),
    "har": ("HAR-RV",
            "Heterogeneous AutoRegressive Realised Volatility: predicts tomorrow's volatility from "
            "yesterday's, last week's and last month's realised volatility."),
    "volatility": ("Volatility",
                   "How much the price typically moves per day (standard deviation of returns). "
                   "It measures the size of moves, not their direction."),
    "drawdown": ("Drawdown",
                 "The drop from a recent peak to a later low, e.g. a 10% drawdown means price fell 10% from its high."),
    "pr-auc": ("PR-AUC",
               "Area under the precision-recall curve. It is the most informative ranking metric when "
               "the positive class (illicit wallets, 5.4%) is rare. 1.0 is perfect."),
    "roc-auc": ("ROC-AUC",
                "Probability that the model scores a random illicit wallet higher than a random licit one. "
                "0.5 = coin flip, 1.0 = perfect."),
    "precision": ("Precision",
                  "Of the wallets the model flagged as illicit, the share that really were illicit."),
    "recall": ("Recall",
               "Of all truly illicit wallets, the share the model caught."),
    "f1": ("F1-score", "The harmonic mean of precision and recall."),
    "shap": ("SHAP",
             "A method that splits one prediction into additive contributions per input feature. "
             "Here it explains the illicit-minus-licit logit margin: negative values push toward LICIT, "
             "positive toward ILLICIT."),
    "graphsage": ("GraphSAGE",
                  "A graph neural network. For each wallet it averages the features of a sample of "
                  "incoming-neighbour wallets (10 first-hop, 5 second-hop) and combines them with the wallet's own features."),
    "finbert": ("FinBERT",
                "A BERT language model fine-tuned on financial text. It outputs probabilities for "
                "positive, negative and neutral sentiment for each article."),
    "sentiment score": ("Sentiment score",
                        "positive probability minus negative probability, from -1 (very negative) to +1 (very positive)."),
    "threshold": ("Decision threshold",
                  "The illicit probability above which a wallet is flagged. Raising it gives fewer false alarms "
                  "but misses more illicit wallets. Yours is 0.90, chosen on the validation set for best F1."),
    "walk-forward": ("Walk-forward validation",
                     "Models are trained only on the past and tested on later data, repeatedly, "
                     "which avoids look-ahead bias."),
    "lift": ("Lift",
             "How many times more often the event happens on days the model flags as high-probability "
             "compared with an average day."),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_state() -> dict:
    """Structured numbers used by both the LLM context and the offline explainer."""
    mk = D.load_market_latest()
    md = D.load_market_daily()
    g = D.load_graph()
    ss = D.load_sentiment_summary()
    daily = D.load_daily_sentiment()
    rel = D.sentiment_market_relationship()

    age = D.market_age_days(mk)
    last30 = md.tail(30)
    lvl_share = md["market_risk_level"].value_counts(normalize=True).to_dict()
    peak = md.loc[md["market_risk_score"].idxmax()]
    calm = md.loc[md["market_risk_score"].idxmin()]
    last_daily = daily.iloc[-1]
    recent90 = daily.tail(90)
    return dict(mk=mk, md=md, g=g, ss=ss, daily=daily, rel=rel, age=age, last30=last30,
                lvl_share=lvl_share, peak=peak, calm=calm, last_daily=last_daily, recent90=recent90)


def build_sections(S: dict | None = None) -> dict[str, str]:
    S = S or build_state()
    mk, g, ss, md = S["mk"], S["g"], S["ss"], S["md"]
    r, f, t, ev, bands = mk["risk"], mk["forecast"], mk["tail_risk_1d"], mk["event_probabilities"], mk["price_bands_7d"]
    gs = g["summary"]
    conf = gs["test_confusion"]
    today = _now().strftime("%Y-%m-%d")

    sec: dict[str, str] = {}

    sec["overview"] = (
        f"Today is {today} (UTC). CryptoRisk AI has three independent components: (1) Graph: GraphSAGE wallet "
        f"illicitness classifier with SHAP explanations; (2) Sentiment: FinBERT sentiment of {ss['total_articles']:,} "
        f"Bitcoin news articles ({ss['date_min']} to {ss['date_max']}); (3) Market: BTCUSDT daily risk score, "
        f"volatility/tail-risk forecasts. They are NOT merged into one score because they cover different "
        f"entities (wallets, news, price) and different time periods."
    )

    drivers = ", ".join(D.COMPONENT_LABELS[d] for d in r["top_drivers"])
    comp = "; ".join(
        f"{D.COMPONENT_LABELS[k]} percentile {v:.1f} (weight {r['weights'][k]:.0%})"
        for k, v in r["component_percentiles"].items())
    sec["market_now"] = (
        f"MARKET RISK, latest reading. Data as of {mk['data_as_of_utc'][:16]} UTC; that is {S['age']:.1f} days before "
        f"today, so the reading is STALE (it was not refreshed after the export). Last close ${mk['price']['last_close']:,.0f} "
        f"(last-day return {mk['price']['last_day_return']*100:+.2f}%). Market Risk Score {r['score_0_100']:.1f}/100 = "
        f"'{r['level']}' (levels: Low <30, Moderate 30-50, High 50-70, Extreme >=70). Score components: {comp}. "
        f"Top drivers: {drivers}. The score is a weighted average of percentile ranks vs the last 2 years."
    )
    sec["market_forecast"] = (
        f"MARKET FORECASTS issued for {f['forecast_for_day']}: next-day daily volatility {f['vol_next_day_daily']*100:.2f}%; "
        f"next-7-day average daily volatility {f['vol_next_7d_daily_avg']*100:.2f}% "
        f"(annualised {f['vol_next_7d_annualised']*100:.1f}%); this is the {f['vol_percentile_vs_last_2y']:.0f}th percentile "
        f"of the last 2 years (calm). 1-day tail risk: VaR95 {t['var95']*100:.2f}%, ES95 {t['es95']*100:.2f}%, "
        f"VaR99 {t['var99']*100:.2f}%, ES99 {t['es99']*100:.2f}%. Event probabilities: |move|>=5% next day "
        f"{ev['abs_move_ge_5pct_next_day']*100:.1f}%; drop>=5% next day {ev['drop_ge_5pct_next_day']*100:.2f}%; "
        f"7-day drawdown >=5% {ev['drawdown_ge_5pct_next_7d']*100:.1f}%, >=10% {ev['drawdown_ge_10pct_next_7d']*100:.1f}%, "
        f">=15% {ev['drawdown_ge_15pct_next_7d']*100:.1f}%, >=20% {ev['drawdown_ge_20pct_next_7d']*100:.2f}%. "
        f"7-day price bands (from the model, not a prediction of direction): 2.5%=${bands['p2.5']:,.0f}, "
        f"10%=${bands['p10']:,.0f}, median=${bands['p50']:,.0f}, 90%=${bands['p90']:,.0f}, 97.5%=${bands['p97.5']:,.0f}."
    )
    tm = mk["model_card"]["test_metrics"]
    sel = mk["model_card"]["selected_models"]
    sec["market_model"] = (
        f"MARKET MODEL CARD. Models: next-day vol = {sel['next_day_vol']}; 7-day vol = {sel['next_7d_vol']}; "
        f"tail risk = {sel['tail_risk']}. Validation: {mk['model_card']['validation']}. Untouched test metrics: "
        f"1-day vol QLIKE {tm['vol_1d_QLIKE']}, log-R2 {tm['vol_1d_R2_log']}; 7-day QLIKE {tm['vol_7d_QLIKE']}; "
        f"VaR95 violation rate {tm['var95_violation_pct']}% (target 5%), VaR99 {tm['var99_violation_pct']}% (target 1%); "
        f"big-move (|move|>=5%) AUC {tm['big_move_AUC']}, lift {tm['big_move_lift']}x. "
        f"The component deliberately does NOT forecast price direction: {mk['model_card']['not_forecast']}. "
        f"In the notebook, next-day direction accuracy was 48.9% vs 51.1% majority baseline (AUC 0.506): no edge. "
        f"Calm-regime 7-day price bands tend to be slightly too narrow (lowest-volatility tercile under-covers)."
    )
    sh = S["lvl_share"]
    sec["market_history"] = (
        f"MARKET HISTORY {md['date'].min().date()} to {md['date'].max().date()} ({len(md)} days): share of days at "
        f"Low {sh.get('Low',0)*100:.0f}%, Moderate {sh.get('Moderate',0)*100:.0f}%, High {sh.get('High',0)*100:.0f}%, "
        f"Extreme {sh.get('Extreme',0)*100:.0f}%. Highest score {S['peak']['market_risk_score']:.1f} on "
        f"{S['peak']['date'].date()} (BTC ${S['peak']['close']:,.0f}); lowest {S['calm']['market_risk_score']:.1f} on "
        f"{S['calm']['date'].date()}. Average score over the last 30 days of data: "
        f"{S['last30']['market_risk_score'].mean():.1f}."
    )

    tp = conf["tp"]; fp = conf["fp"]; fn = conf["fn"]; tn = conf["tn"]
    base = gs["test_support"]["licit"] / sum(gs["test_support"].values())
    sec["graph_perf"] = (
        f"GRAPH MODEL ({gs['model']}), task: classify Bitcoin wallets as illicit vs licit using 55 wallet features "
        f"and incoming-neighbour aggregation (fan-out {gs['fanout_1']}/{gs['fanout_2']}). Graph: {gs['num_nodes']:,} wallets, "
        f"{gs['num_edges']:,} directed edges. Labelled wallets: {gs['dataset']['labeled_licit']+gs['dataset']['labeled_illicit']:,} "
        f"({gs['dataset']['labeled_illicit']:,} illicit = 5.4%); {gs['dataset']['unlabeled']:,} unlabelled. Test set "
        f"{sum(gs['test_support'].values()):,} wallets: PR-AUC {gs['test_pr_auc']:.4f}, ROC-AUC {gs['test_roc_auc']:.4f}, "
        f"accuracy {gs['test_accuracy']:.4f}, precision {gs['test_precision']:.4f}, recall {gs['test_recall']:.4f}, "
        f"F1 {gs['test_f1']:.4f} at decision threshold {gs['decision_threshold']} (chosen on validation for best F1). "
        f"Confusion matrix: TN {tn:,}, FP {fp:,}, FN {fn:,}, TP {tp:,}. Always predicting 'licit' would already score "
        f"{base*100:.1f}% accuracy, so precision/recall/PR-AUC matter more than accuracy. Best epoch {gs['best_epoch']}; "
        f"validation PR-AUC {gs['validation_pr_auc']:.4f}. At threshold 0.50 validation recall was 93.4% but precision only 33.7%."
    )
    wb, wa = g["wallets"]["wallet_b"], g["wallets"]["wallet_a"]
    sb = g["shap"]["wallet_b"]
    top_b = "; ".join(f"{D.describe_feature(r_.feature)} ({r_.feature}) SHAP {r_.shap_value:+.2f}" for r_ in sb.head(5).itertuples())
    sec["graph_xai"] = (
        f"GRAPH EXPLAINABILITY (Kernel SHAP on the illicit-minus-licit logit margin; negative = toward LICIT). Saved examples: "
        f"wallet {wb['address']} (node {wb['node_id']}): predicted {wb['predicted_class']}, illicit probability "
        f"{wb['illicit_probability']:.2e}, threshold {gs['decision_threshold']}, margin {wb['original_margin']:.2f} "
        f"vs background average {wb['expected_margin']:.2f}; top drivers: {top_b}. "
        f"Second example: node {wa['node_id']} predicted {wa['predicted_class']} (margin {wa['original_margin']:.2f}). "
        f"For both, the largest push toward LICIT came from first_sent_block (SHAP about -2.5). SHAP additivity error was 0. "
        f"Only these saved wallets can be explained in the web app; live inference on arbitrary addresses needs the "
        f"170 MB graph feature file, which is not hosted. A prediction is a model estimate, not proof a wallet is safe or criminal."
    )

    lab = ss["label_pct"]
    sec["sentiment_overall"] = (
        f"SENTIMENT (FinBERT on {ss['total_articles']:,} Bitcoin-relevant articles from {ss['n_sources']} sources, "
        f"{ss['date_min']} to {ss['date_max']}): neutral {lab['neutral']}%, positive {lab['positive']}%, "
        f"negative {lab['negative']}%. Mean sentiment score {ss['score_mean']:+.3f} (std {ss['score_std']:.3f}); mean model "
        f"confidence {ss['mean_confidence']:.2f}. Daily series has {len(S['daily']):,} days; the busiest days had 550-600+ articles."
    )
    ld, r90 = S["last_daily"], S["recent90"]
    sec["sentiment_recent"] = (
        f"SENTIMENT, most recent data: the daily series ENDS on {ld['date'].date()} (it is not live). On that day: "
        f"{int(ld['news_count'])} articles, average score {ld['avg_sentiment']:+.3f}, positive {ld['positive_ratio']*100:.0f}%, "
        f"negative {ld['negative_ratio']*100:.0f}%. Over the final 90 days of data the average daily score was "
        f"{r90['avg_sentiment'].mean():+.3f} and the average negative share {r90['negative_ratio'].mean()*100:.0f}%."
    )
    rel = S["rel"]
    def _c(sig, vs):
        return float(rel[(rel.signal == sig) & (rel.versus == vs)].spearman.iloc[0])
    sec["sentiment_vs_market"] = (
        f"SENTIMENT vs MARKET (Spearman, {int(rel.days.max())} overlapping days 2021-2025): average sentiment vs same-day BTC "
        f"return {_c('Average sentiment','Same-day BTC return'):+.2f}; vs NEXT-day return "
        f"{_c('Average sentiment','Next-day BTC return'):+.2f}; vs next-day absolute return "
        f"{_c('Average sentiment','Next-day |return| (turbulence)'):+.2f}; negative-news share vs same-day market risk score "
        f"{_c('Negative-news share','Same-day market risk score'):+.2f}. Interpretation: news sentiment mostly REACTS to "
        f"price moves the same day and has essentially no demonstrated predictive power for the next day."
    )
    sec["limitations"] = (
        "LIMITATIONS: (1) Market data is a snapshot, last candle "
        f"{mk['data_as_of_utc'][:10]}; the sentiment series ends {ss['date_max']}; the graph benchmark is static. "
        "(2) Nothing here predicts price direction. (3) Graph predictions are statistical estimates on a research "
        "benchmark, not proof of wrongdoing. (4) FinBERT predictions have not been checked against completed human labels "
        "in the provided files, so no FinBERT accuracy figure is claimed. (5) Not investment advice."
    )
    sec["glossary"] = " ".join(f"{k}: {v[1]}" for k, v in GLOSSARY.items())
    return sec


# ---- keyword routing -----------------------------------------------------------
ROUTES = {
    "market_now": ["risk score", "market risk", "current", "now", "today", "latest", "level", "moderate", "price",
                   "stale", "fresh", "driver", "component", "btc", "bitcoin", "score", "safe", "dangerous", "risky"],
    "market_forecast": ["forecast", "volatility", "vol", "var", "value at risk", "shortfall", "es95", "tail", "probab",
                        "chance", "drawdown", "crash", "drop", "band", "range", "next", "week", "tomorrow", "move",
                        "loss", "lose", "leverage"],
    "market_model": ["model", "accura", "reliab", "trust", "validat", "backtest", "qlike", "har", "metric", "direction",
                     "predict", "edge", "how good", "test"],
    "market_history": ["history", "histor", "past", "peak", "highest", "lowest", "trend", "average", "since", "2021",
                       "2022", "2023", "2024", "2025", "distribution"],
    "graph_perf": ["graph", "wallet", "graphsage", "illicit", "licit", "precision", "recall", "f1", "pr-auc", "roc",
                   "confusion", "threshold", "false", "fraud", "launder", "criminal", "aml", "accuracy", "elliptic"],
    "graph_xai": ["shap", "explain", "why", "feature", "xai", "wallet", "contribut", "3kgiq", "793784", "783573",
                  "reason", "interpret"],
    "sentiment_overall": ["sentiment", "news", "finbert", "article", "positive", "negative", "neutral", "headline",
                          "source", "media"],
    "sentiment_recent": ["sentiment", "news", "recent", "latest", "now", "today", "current", "2025", "mood"],
    "sentiment_vs_market": ["correlat", "relationship", "predict", "lead", "affect", "impact", "drive", "sentiment",
                            "versus", "vs", "combine", "together", "all three", "overall"],
    "glossary": ["what is", "what's", "mean", "define", "explain", "jargon", "simple", "eli5", "term"],
}


def _kw_hit(kw: str, q: str) -> bool:
    """Short keywords must match whole words ('now' must not fire on 'know'); longer ones may be stems."""
    if len(kw) <= 4:
        return re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", q) is not None
    return kw in q


DOMAINS = {
    "graph": (["graph", "wallet", "illicit", "licit", "graphsage", "shap", "aml", "launder"], ["graph_perf", "graph_xai"]),
    "market": (["market", "btc", "volatil", "drawdown", "risk score", "var", "shortfall", "price"],
               ["market_now", "market_forecast", "market_model", "market_history"]),
    "sentiment": (["sentiment", "news", "finbert", "article", "headline"],
                  ["sentiment_overall", "sentiment_recent", "sentiment_vs_market"]),
}


def route_scores(question: str) -> dict[str, int]:
    """Keyword score per section, plus a domain boost so 'accuracy of the graph model' is not routed to the market model."""
    q = question.lower()
    scores = {name: sum(1 for kw in kws if _kw_hit(kw, q)) for name, kws in ROUTES.items()}
    for words, secs in DOMAINS.values():
        if any(_kw_hit(w, q) for w in words):
            for sname in secs:
                scores[sname] += 3
    sent = any(_kw_hit(w, q) for w in DOMAINS["sentiment"][0])
    other = any(_kw_hit(w, q) for w in DOMAINS["market"][0] + ["predict", "lead", "correlat", "affect", "drive", "impact", "relationship"])
    if sent and other:  # cross-domain question: sentiment <-> market
        scores["sentiment_vs_market"] += 6
    return scores


def has_topic_hit(question: str) -> bool:
    """False when nothing in the question relates to the dashboard (e.g. 'what's the weather')."""
    scores = route_scores(question)
    strong = {k: v for k, v in scores.items() if k != "glossary"}
    generic = ["summar", "overall", "project", "dashboard", "everything", "whole", "all three", "combine", "together",
               "what can you", "help", "overview"]
    return (any(v > 0 for v in strong.values()) or bool(glossary_hits(question))
            or any(g in question.lower() for g in generic))


def select_sections(question: str, sections: dict[str, str], k: int = 4) -> list[str]:
    q = question.lower()
    scores = route_scores(question)
    ranked = [n for n, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0][:k]
    if not ranked:  # vague question -> give a broad snapshot
        ranked = ["market_now", "graph_perf", "sentiment_overall"]
    if any(w in q for w in ["overall", "summary", "summar", "all three", "everything", "combine", "together"]):
        ranked = list(dict.fromkeys(ranked + ["market_now", "graph_perf", "sentiment_overall", "sentiment_recent"]))[:5]
    out = ["overview"] + ranked + ["limitations"]
    return list(dict.fromkeys(out))


def build_context(question: str, sections: dict[str, str], max_chars: int = 9000) -> tuple[str, list[str]]:
    names = select_sections(question, sections)
    parts, used, total = [], [], 0
    for n in names:
        txt = sections[n]
        if n == "glossary":  # only include the glossary entries the user actually mentions
            txt = glossary_snippet(question) or ""
            if not txt:
                continue
        if total + len(txt) > max_chars:
            continue
        parts.append(f"[{n}] {txt}")
        used.append(n)
        total += len(txt)
    return "\n\n".join(parts), used


def glossary_hits(question: str) -> list[str]:
    q = question.lower()
    hits = []
    for key in GLOSSARY:
        pat = r"(?<![a-z])" + re.escape(key) + r"(?![a-z])"
        if re.search(pat, q):
            hits.append(key)
    if "value at risk" in q and "var" not in hits:
        hits.append("var")
    if "expected shortfall" in q and "es" not in hits:
        hits.append("es")
    return hits


def glossary_snippet(question: str) -> str:
    hits = glossary_hits(question)
    return " ".join(f"{GLOSSARY[h][0]}: {GLOSSARY[h][1]}" for h in hits[:4])
