"""Cached loaders for every saved result used by the dashboard."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RISK_LEVELS = [("Low", 0, 30), ("Moderate", 30, 50), ("High", 50, 70), ("Extreme", 70, 100)]

COMPONENT_LABELS = {
    "vol_forecast": "Volatility forecast",
    "drawdown_90d": "90-day drawdown",
    "vol_shock": "Volatility shock",
    "downside_semivol": "Downside semi-volatility",
    "turnover_spike": "Turnover spike",
}

# Plain-English names for the 55 wallet features (taken from your Streamlit notebook).
FEATURE_DESCRIPTIONS = {
    "num_txs_as_sender": "Number of transactions sent",
    "num_txs_as receiver": "Number of transactions received",
    "first_block_appeared_in": "First blockchain block the wallet appeared in",
    "last_block_appeared_in": "Last blockchain block the wallet appeared in",
    "lifetime_in_blocks": "Wallet activity span (in blocks)",
    "total_txs": "Total recorded transactions",
    "first_sent_block": "First block in which the wallet sent Bitcoin",
    "first_received_block": "First block in which the wallet received Bitcoin",
    "num_timesteps_appeared_in": "Number of time periods with activity",
    "btc_transacted_total": "Total Bitcoin transacted",
    "btc_transacted_min": "Smallest Bitcoin amount transacted",
    "btc_transacted_max": "Largest Bitcoin amount transacted",
    "btc_transacted_mean": "Average Bitcoin amount transacted",
    "btc_transacted_median": "Median Bitcoin amount transacted",
    "btc_sent_total": "Total Bitcoin sent",
    "btc_sent_min": "Smallest amount sent",
    "btc_sent_max": "Largest amount sent",
    "btc_sent_mean": "Average amount sent",
    "btc_sent_median": "Median amount sent",
    "btc_received_total": "Total Bitcoin received",
    "btc_received_min": "Smallest amount received",
    "btc_received_max": "Largest amount received",
    "btc_received_mean": "Average amount received",
    "btc_received_median": "Median amount received",
    "fees_total": "Total transaction fees paid",
    "fees_min": "Smallest transaction fee",
    "fees_max": "Largest transaction fee",
    "fees_mean": "Average transaction fee",
    "fees_median": "Median transaction fee",
    "fees_as_share_total": "Total fees as a share of transaction value",
    "fees_as_share_min": "Smallest fee share",
    "fees_as_share_max": "Largest fee share",
    "fees_as_share_mean": "Average fee share",
    "fees_as_share_median": "Median fee share",
    "blocks_btwn_txs_total": "Total block gaps between transactions",
    "blocks_btwn_txs_min": "Shortest gap between transactions (blocks)",
    "blocks_btwn_txs_max": "Longest gap between transactions (blocks)",
    "blocks_btwn_txs_mean": "Average gap between transactions (blocks)",
    "blocks_btwn_txs_median": "Median gap between transactions (blocks)",
    "blocks_btwn_input_txs_total": "Total block gaps between incoming transactions",
    "blocks_btwn_input_txs_min": "Shortest gap between incoming transactions",
    "blocks_btwn_input_txs_max": "Longest gap between incoming transactions",
    "blocks_btwn_input_txs_mean": "Average gap between incoming transactions",
    "blocks_btwn_input_txs_median": "Median gap between incoming transactions",
    "blocks_btwn_output_txs_total": "Total block gaps between outgoing transactions",
    "blocks_btwn_output_txs_min": "Shortest gap between outgoing transactions",
    "blocks_btwn_output_txs_max": "Longest gap between outgoing transactions",
    "blocks_btwn_output_txs_mean": "Average gap between outgoing transactions",
    "blocks_btwn_output_txs_median": "Median gap between outgoing transactions",
    "num_addr_transacted_multiple": "Addresses it transacted with more than once",
    "transacted_w_address_total": "Total interactions with other addresses",
    "transacted_w_address_min": "Fewest interactions with one address",
    "transacted_w_address_max": "Most interactions with one address",
    "transacted_w_address_mean": "Average interactions per address",
    "transacted_w_address_median": "Median interactions per address",
}


def describe_feature(name: str) -> str:
    return FEATURE_DESCRIPTIONS.get(name, name.replace("_", " "))


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- market -----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_market_daily() -> pd.DataFrame:
    df = pd.read_csv(DATA / "market" / "market_risk_daily_v2.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_market_latest() -> dict:
    return _read_json(DATA / "market" / "market_risk_latest_v2.json")


def market_age_days(latest: dict) -> float:
    """Age of the market data relative to *now*, computed live (the JSON value is frozen at export time)."""
    as_of = pd.Timestamp(latest["data_as_of_utc"])
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    return (datetime.now(timezone.utc) - as_of.to_pydatetime()).total_seconds() / 86400


def level_for(score: float) -> str:
    for name, lo, hi in RISK_LEVELS:
        if lo <= score < hi:
            return name
    return "Extreme"


# ---- sentiment --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_daily_sentiment() -> pd.DataFrame:
    df = pd.read_csv(DATA / "sentiment" / "daily_bitcoin_sentiment.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sentiment_summary() -> dict:
    return _read_json(DATA / "sentiment" / "summary.json")


@st.cache_data(show_spinner=False)
def load_sentiment_tables() -> dict[str, pd.DataFrame]:
    d = DATA / "sentiment"
    return {
        "yearly": pd.read_csv(d / "yearly.csv"),
        "by_source": pd.read_csv(d / "by_source.csv"),
        "monthly": pd.read_csv(d / "monthly.csv", parse_dates=["date"]),
    }


@st.cache_data(show_spinner=False)
def load_articles_sample() -> pd.DataFrame:
    df = pd.read_csv(DATA / "sentiment" / "articles_sample.csv.gz", parse_dates=["published_on"])
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_most_confident() -> pd.DataFrame:
    return pd.read_csv(DATA / "sentiment" / "most_confident.csv", parse_dates=["published_on"])


@st.cache_data(show_spinner=False)
def sentiment_market_relationship() -> pd.DataFrame:
    """Spearman correlation of daily sentiment with market variables over the overlapping period."""
    s = load_daily_sentiment().set_index("date")
    m = load_market_daily().set_index("date")
    j = s.join(m, how="inner")
    j["ret"] = np.log(j["close"]).diff()
    j["ret_next"] = j["ret"].shift(-1)
    j["abs_ret_next"] = j["ret"].abs().shift(-1)
    j["risk_next"] = j["market_risk_score"].shift(-1)
    rows = []
    for col, label in [("avg_sentiment", "Average sentiment"), ("negative_ratio", "Negative-news share")]:
        for tgt, tl in [("ret", "Same-day BTC return"), ("ret_next", "Next-day BTC return"),
                        ("abs_ret_next", "Next-day |return| (turbulence)"),
                        ("market_risk_score", "Same-day market risk score"),
                        ("risk_next", "Next-day market risk score")]:
            rows.append({"signal": label, "versus": tl,
                         "spearman": float(j[col].corr(j[tgt], method="spearman")),
                         "days": int(j[[col, tgt]].dropna().shape[0])})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def sentiment_market_join() -> pd.DataFrame:
    s = load_daily_sentiment().set_index("date")
    m = load_market_daily().set_index("date")
    j = s.join(m, how="inner")
    j["ret"] = np.log(j["close"]).diff()
    return j.reset_index()


# ---- graph ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_graph() -> dict:
    d = DATA / "graph"
    return {
        "summary": _read_json(d / "graph_summary.json"),
        "wallets": _read_json(d / "wallets.json"),
        "history": pd.read_csv(d / "training_history.csv"),
        "thresholds": pd.read_csv(d / "threshold_analysis.csv"),
        "shap": {"wallet_a": pd.read_csv(d / "wallet_a_shap_top10.csv"),
                 "wallet_b": pd.read_csv(d / "wallet_b_shap_top10.csv")},
    }


# ---- live spot price (optional, never used by the risk model) -----------------
@st.cache_data(ttl=90, show_spinner=False)
def fetch_live_btc() -> dict | None:
    """Free CoinGecko endpoint, no key needed. Returns None on any failure (offline, rate limit...)."""
    try:
        import requests
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd", "include_24hr_change": "true"},
            timeout=3,
        )
        r.raise_for_status()
        b = r.json()["bitcoin"]
        return {"price": float(b["usd"]), "change_24h": float(b.get("usd_24h_change", 0.0))}
    except Exception:
        return None
