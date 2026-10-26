# ₿ CryptoRisk AI

Explainable Bitcoin risk dashboard built with Streamlit. Five tabs:

| Tab | What it shows |
|---|---|
| 🏠 Overview | All three components at a glance, key findings, data freshness |
| 🕸️ Graph | GraphSAGE wallet-illicitness results, confusion matrix, threshold explorer, SHAP explanations |
| 📰 Sentiment | FinBERT sentiment of 165,560 news articles, trends, article explorer, sentiment-vs-market study |
| 📉 Market | BTC risk score, volatility forecasts, VaR/ES calculator, event probabilities, model card |
| 🤖 AI Analyst | Free-LLM chat grounded in the results above (Groq / OpenRouter), with an offline fallback |

Everything is free: Streamlit Community Cloud hosting, GitHub, and a free Groq API key (no credit card).

---

## 1 · Run it locally (2 minutes)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The AI tab works immediately in **offline mode**. For conversational answers add a free key (step 3).

## 2 · Host it for free on Streamlit Community Cloud

1. Create a free GitHub account and a **new public repository** (e.g. `cryptorisk-ai`).
2. Upload everything in this folder (drag-and-drop in the GitHub web UI works; the whole project is ~4 MB).
   Do **not** upload `.streamlit/secrets.toml` if you create one.
3. Go to https://share.streamlit.io → sign in with GitHub → **Create app** → pick the repo, branch `main`, main file `app.py` → **Deploy**.
4. In a minute you get a public URL like `https://your-name-cryptorisk-ai.streamlit.app`.

## 3 · Turn on the free LLM

1. Get a free key at https://console.groq.com/keys (sign-up only, no card). It starts with `gsk_`.
2. In Streamlit Cloud: your app → **Settings → Secrets** → paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. Save. The AI tab now shows *Engine: Groq (free) · llama-3.3-70b-versatile*.

Alternatives: `OPENROUTER_API_KEY` (free `:free` models), or a custom OpenAI-compatible endpoint. Visitors can also paste their own key in **AI settings**
(kept in their session only).

Free-tier limits (Groq, at time of writing): ~30 requests/min and ~1,000/day for the 70B model. The app retries on the smaller `llama-3.1-8b-instant`
model if rate-limited, falls back to the offline explainer if the provider fails, and caps live answers at 40 per browser session.

## 4 · How the AI Analyst stays accurate

`src/facts.py` converts your saved results into short text sections (market now, forecasts, model card, graph metrics, SHAP, sentiment, limitations, glossary).
For each question only the relevant sections (~2–3k tokens) are sent to the LLM, with a system prompt that forbids answering from outside those facts,
giving trade advice or predicting price direction. Every answer lists the sections it was grounded in.

## 5 · Project layout

```
app.py                      entry point (hero + 5 tabs)
src/
  data.py                   cached loaders, feature descriptions, live price (CoinGecko, optional)
  theme.py                  crypto dark theme, CSS, Plotly styling
  facts.py                  results → facts, keyword routing, glossary
  llm.py                    Groq/OpenRouter/custom client, streaming, retries
  offline.py                no-key fallback explainer
  tabs/                     overview.py · graph.py · sentiment.py · market.py · analyst.py
data/                       slim results (~2.3 MB)
models/graphsage_best.pt    your trained checkpoint (kept for reference)
scripts/prepare_data.py     rebuilds data/ from your full files
tests/test_app.py           11 automated tests
```

## 6 · Updating the results

* **Market (currently ~2 weeks stale):** re-run `BTC_Market_Risk_Component.ipynb` with fresh candles, then replace
  `data/market/market_risk_daily_v2.csv` and `market_risk_latest_v2.json`. The app computes the "days old" badge live.
* **Sentiment / Graph:** re-run `python scripts/prepare_data.py --uploads <folder with your new files>`.
* Push to GitHub; Streamlit Cloud redeploys automatically.

## 7 · Optional: live wallet scoring

The graph tab shows saved predictions + SHAP for the two wallets explained in your notebook. Scoring *any* address needs
`graph_node_features.npy` (173 MB), `graph_edges.npy` (21 MB), `address_node_mapping.parquet` (30 MB) and `graphsage_scaler.joblib`,
which exceed GitHub's 100 MB file limit. To enable it later, host those files on Hugging Face Datasets (free), download them at start-up, and port the
`predict_wallet_risk` function from `graphsage_final_xai.ipynb` (needs `torch`, ~200 MB RAM extra).

## 8 · Honest limitations

* Market data is a snapshot (last candle 2026-10-12); the news series ends 2025-02-01; the graph benchmark is static.
* The market model forecasts **risk, not direction** (no out-of-sample directional edge was found).
* FinBERT accuracy is **not** reported: the three `human_label` columns in the evaluation CSVs are empty.
* Graph predictions are model estimates, not proof of wrongdoing. Nothing here is investment advice.
* The three components are shown side by side, not merged into one score: they describe different entities and time periods.

Run the tests with `python -m pytest -q`.
