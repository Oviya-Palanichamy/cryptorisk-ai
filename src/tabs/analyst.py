"""AI Analyst tab: a free LLM (Groq / OpenRouter) grounded in the project's results, with offline fallback."""
from __future__ import annotations

import streamlit as st

from .. import facts as F
from .. import llm as L
from .. import offline as O
from .. import theme as T

MAX_LIVE_QUESTIONS = 40  # per browser session; protects a shared free key from runaway use

SUGGESTIONS = [
    "Summarise the whole project in plain English",
    "Why is the market risk score “Moderate” right now?",
    "What does a 95% VaR of 3.16% mean for a $10,000 position?",
    "How reliable is the market risk model?",
    "Explain why wallet 3KGiq1… was classed licit",
    "Is 97% accuracy good for the graph model?",
    "Does news sentiment predict Bitcoin's price?",
    "What is SHAP and why does it matter?",
]

SECTION_NAMES = {
    "overview": "Project overview", "market_now": "Market · now", "market_forecast": "Market · forecasts",
    "market_model": "Market · model card", "market_history": "Market · history", "graph_perf": "Graph · performance",
    "graph_xai": "Graph · SHAP", "sentiment_overall": "Sentiment · overall", "sentiment_recent": "Sentiment · recent",
    "sentiment_vs_market": "Sentiment vs market", "limitations": "Limitations", "glossary": "Glossary",
}


@st.cache_resource(ttl=600, show_spinner=False)
def _knowledge():
    S = F.build_state()
    return S, F.build_sections(S)


def _settings() -> L.LLMConfig:
    with st.expander("⚙️ AI settings: free LLM provider", expanded=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            choice = st.selectbox(
                "Provider",
                ["auto", "groq", "openrouter", "custom", "offline"],
                format_func=lambda x: {"auto": "Auto (use configured key)", "groq": "Groq · free",
                                       "openrouter": "OpenRouter · free models", "custom": "Custom OpenAI-compatible",
                                       "offline": "Offline explainer only"}[x],
                key="llm_choice")
        with c2:
            key = st.text_input("Your free API key (kept in this session only)", type="password",
                                placeholder="gsk_… (Groq) or sk-or-… (OpenRouter)", key="llm_key")
        model, url = "", ""
        if choice in ("groq", "openrouter", "custom"):
            model = st.text_input("Model (optional override)", placeholder=L.PROVIDERS[choice]["model"], key="llm_model")
        if choice == "custom":
            url = st.text_input("Chat-completions URL", value=L.PROVIDERS["custom"]["url"], key="llm_url")
        st.caption("Get a free key in a minute, no credit card: **Groq** → console.groq.com/keys · **OpenRouter** → openrouter.ai/keys. "
                   "Without a key the app still answers using its built-in offline explainer.")
    return L.resolve_config(choice, key, model, url)


def _answer(question: str, cfg: L.LLMConfig, S: dict, sections: dict[str, str]):
    """Returns (text, used_sections, mode) and renders the answer (streaming when live)."""
    context, used = F.build_context(question, sections)
    live_ok = cfg.is_live and st.session_state.get("live_count", 0) < MAX_LIVE_QUESTIONS
    if live_ok:
        msgs = L.build_messages(question, context, st.session_state.chat)
        raw: list[str] = []

        def _gen():
            for chunk in L.stream_answer(cfg, msgs):
                raw.append(chunk)
                yield T.esc(chunk)

        try:
            st.write_stream(_gen())
            text = "".join(raw)
            st.session_state.live_count = st.session_state.get("live_count", 0) + 1
            return text, used, "live"
        except L.LLMError as e:
            note = f"⚠️ {e} Showing the offline explainer instead."
        except Exception as e:  # never let the chat crash the page
            note = f"⚠️ The LLM call failed ({type(e).__name__}). Showing the offline explainer instead."
        st.warning(note)
    elif cfg.is_live:
        st.info("Session limit for live AI answers reached; using the offline explainer.")
    text = O.offline_answer(question, S, sections)
    st.markdown(T.esc(text))
    return text, used, "offline"


def render() -> None:
    S, sections = _knowledge()
    T.section("AI Risk Analyst", "Ask questions in plain English. Answers are generated <b>only</b> from this dashboard's results: "
                                 "market model, graph model, SHAP explanations and news sentiment.")
    cfg = _settings()
    dot = "" if cfg.is_live else " off"
    T.html(f'<div class="ticker"><span class="dot{dot}"></span> Engine: {cfg.label}</div>')
    if not cfg.is_live:
        T.callout("💡 <b>Running in offline mode.</b> Answers are templated from your saved results. For fluent, conversational answers, paste a "
                  "free Groq key in <i>AI settings</i> (or add <code>GROQ_API_KEY</code> to the app's secrets).")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    st.markdown("**Try one:**")
    cols = st.columns(4)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 4].button(s, key=f"sugg_{i}", width="stretch"):
            st.session_state.pending_q = s

    for m in st.session_state.chat:
        with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
            st.markdown(T.esc(m["content"]))
            if m.get("used"):
                st.caption("Grounded in: " + " · ".join(SECTION_NAMES.get(u, u) for u in m["used"] if u not in ("overview",))
                           + f"  |  {'LLM' if m.get('mode') == 'live' else 'offline explainer'}")

    typed = st.chat_input("Ask about the market risk, a wallet, sentiment, or any term…", key="chat_input")
    question = typed or st.session_state.pop("pending_q", None)
    if question:
        question = question.strip()[:1000]
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(T.esc(question))
        with st.chat_message("assistant", avatar="🤖"):
            text, used, mode = _answer(question, cfg, S, sections)
            st.caption("Grounded in: " + " · ".join(SECTION_NAMES.get(u, u) for u in used if u != "overview")
                       + f"  |  {'LLM' if mode == 'live' else 'offline explainer'}")
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({"role": "assistant", "content": text, "used": used, "mode": mode})

    if st.session_state.chat and st.button("🧹 Clear conversation", key="clear_chat"):
        st.session_state.chat = []
        st.rerun()

    with st.expander("How this analyst stays honest"):
        st.markdown(
            """
- **Grounded.** For every question the app picks the relevant result sections (market, graph, SHAP, sentiment, glossary), and the LLM may only use those numbers.
- **Free by design.** Uses the free tiers of Groq or OpenRouter (no card). If no key is set or the provider is rate-limited, the built-in offline explainer answers instead.
- **No advice, no price calls.** The system prompt forbids buy/sell advice and price-direction predictions; the market model itself found no directional edge.
- **Dated.** Answers state when the market snapshot and news data end, because neither is live.""")
