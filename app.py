"""CryptoRisk AI: Streamlit dashboard (Overview · Graph · Sentiment · Market · AI Analyst)."""
import streamlit as st

st.set_page_config(page_title="CryptoRisk AI", page_icon="₿", layout="wide", initial_sidebar_state="collapsed")

from src import data as D  # noqa: E402
from src import theme as T  # noqa: E402
from src.tabs import analyst, graph, market, overview, sentiment  # noqa: E402

T.inject_css()


def hero() -> None:
    live = D.fetch_live_btc()
    mk = D.load_market_latest()
    if live:
        up = live["change_24h"] >= 0
        col = T.GREEN if up else T.RED
        ticker = (f'<span class="dot"></span> BTC <b>${live["price"]:,.0f}</b> '
                  f'<span style="color:{col}">{"▲" if up else "▼"} {live["change_24h"]:+.2f}% 24h</span>'
                  f' <span style="color:{T.MUTED}">· live spot, CoinGecko (not used by the risk model)</span>')
    else:
        ticker = (f'<span class="dot off"></span> BTC <b>${mk["price"]["last_close"]:,.0f}</b> '
                  f'<span style="color:{T.MUTED}">· model snapshot {mk["data_as_of_utc"][:10]} (live price unavailable)</span>')
    T.html(f"""
    <div class="hero">
      <div class="brand"><div class="logo">₿</div>
        <h1 style="color:#fff">Crypto<span>Risk</span> AI</h1></div>
      <p>Explainable risk intelligence for Bitcoin: graph-based wallet screening, FinBERT news sentiment and a
      volatility &amp; tail-risk engine, with an AI analyst that explains every result.</p>
      <div class="ticker">{ticker}</div>
    </div>""")


hero()
t_over, t_graph, t_sent, t_market, t_ai = st.tabs(
    ["🏠 Overview", "🕸️ Graph", "📰 Sentiment", "📉 Market", "🤖 AI Analyst"])
with t_over:
    overview.render()
with t_graph:
    graph.render()
with t_sent:
    sentiment.render()
with t_market:
    market.render()
with t_ai:
    analyst.render()
