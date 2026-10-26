"""
Build the slim, deployable data files for the CryptoRisk AI dashboard.

WHY: the raw FinBERT article file is ~300 MB, far above GitHub's 100 MB limit, so it
cannot be hosted on Streamlit Community Cloud. This script reduces your saved results to a
few MB *without changing any model output*:

  * all article-level statistics are computed on the FULL 165,560-article file
  * the article explorer uses a reproducible random sample of articles
  * graph numbers are transcribed from the outputs printed in graphsage_final_xai.ipynb

Usage (from the project root):
    python scripts/prepare_data.py --uploads /path/to/your/files
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def clean_snippet(text, n=320):
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def build_sentiment(up: Path):
    out = DATA / "sentiment"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(up / "daily_bitcoin_sentiment.csv", out / "daily_bitcoin_sentiment.csv")

    cols = ["published_on", "title", "body", "source", "finbert_positive_prob",
            "finbert_negative_prob", "finbert_neutral_prob", "finbert_sentiment_score",
            "finbert_label"]
    src = up / "bitcoin_news_with_finbert_1.csv"
    if not src.exists():
        src = up / "bitcoin_news_with_finbert.csv"
    print("reading", src)
    parts = []
    for chunk in pd.read_csv(src, usecols=cols, chunksize=40000, low_memory=False):
        chunk["body"] = chunk["body"].map(clean_snippet)
        chunk["title"] = chunk["title"].fillna("").astype(str).str.slice(0, 220)
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    df["published_on"] = pd.to_datetime(df["published_on"], errors="coerce")
    df = df.dropna(subset=["published_on", "finbert_label"])
    df["finbert_label"] = df["finbert_label"].astype(str).str.lower().str.strip()
    df["year"] = df["published_on"].dt.year
    df["conf"] = df[["finbert_positive_prob", "finbert_negative_prob",
                     "finbert_neutral_prob"]].max(axis=1)
    n = len(df)
    print("articles:", n)

    labels = df["finbert_label"].value_counts()
    summary = {
        "total_articles": int(n),
        "date_min": str(df.published_on.min().date()),
        "date_max": str(df.published_on.max().date()),
        "label_counts": {k: int(v) for k, v in labels.items()},
        "label_pct": {k: round(float(v) / n * 100, 2) for k, v in labels.items()},
        "score_mean": round(float(df.finbert_sentiment_score.mean()), 4),
        "score_std": round(float(df.finbert_sentiment_score.std()), 4),
        "mean_confidence": round(float(df.conf.mean()), 4),
        "mean_confidence_by_label": {k: round(float(v), 4) for k, v in
                                     df.groupby("finbert_label").conf.mean().items()},
        "n_sources": int(df.source.nunique()),
    }
    hist, edges = np.histogram(df.finbert_sentiment_score, bins=40, range=(-1, 1))
    summary["score_hist"] = {"counts": hist.tolist(), "edges": [round(float(e), 3) for e in edges]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    yearly = (df.groupby("year")
                .agg(articles=("title", "size"),
                     mean_score=("finbert_sentiment_score", "mean"),
                     positive=("finbert_label", lambda s: (s == "positive").mean()),
                     negative=("finbert_label", lambda s: (s == "negative").mean()),
                     neutral=("finbert_label", lambda s: (s == "neutral").mean()))
                .reset_index())
    yearly.round(4).to_csv(out / "yearly.csv", index=False)

    top_sources = df.source.value_counts().head(15).index
    by_source = (df[df.source.isin(top_sources)].groupby("source")
                   .agg(articles=("title", "size"),
                        mean_score=("finbert_sentiment_score", "mean"),
                        positive=("finbert_label", lambda s: (s == "positive").mean()),
                        negative=("finbert_label", lambda s: (s == "negative").mean()))
                   .reset_index().sort_values("articles", ascending=False))
    by_source.round(4).to_csv(out / "by_source.csv", index=False)

    keep = ["published_on", "title", "body", "source", "finbert_positive_prob",
            "finbert_negative_prob", "finbert_neutral_prob", "finbert_sentiment_score",
            "finbert_label"]
    dd = df[df.title.str.len() > 15].drop_duplicates("title")
    extremes = pd.concat([
        dd.nlargest(15, "finbert_positive_prob")[keep],
        dd.nlargest(15, "finbert_negative_prob")[keep],
    ])
    extremes.round(4).to_csv(out / "most_confident.csv", index=False)

    sample = df[df.title.str.len() > 15].sample(n=min(12000, len(df)), random_state=42)
    sample = sample.sort_values("published_on")[keep].round(4)
    sample.to_csv(out / "articles_sample.csv.gz", index=False, compression="gzip")
    print("sample rows:", len(sample))

    daily = pd.read_csv(out / "daily_bitcoin_sentiment.csv", parse_dates=["date"])
    monthly = (daily.set_index("date")
                    .resample("MS")
                    .agg(news_count=("news_count", "sum"),
                         avg_sentiment=("avg_sentiment", "mean"),
                         positive_ratio=("positive_ratio", "mean"),
                         negative_ratio=("negative_ratio", "mean"))
                    .reset_index())
    monthly.round(4).to_csv(out / "monthly.csv", index=False)


def build_market(up: Path):
    out = DATA / "market"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(up / "market_risk_daily_v2.csv", out / "market_risk_daily_v2.csv")
    shutil.copy(up / "market_risk_latest_v2.json", out / "market_risk_latest_v2.json")


def build_graph(up: Path):
    """Numbers below are copied from the printed outputs of graphsage_final_xai.ipynb
    (steps 40, 41, 42, 51, 52, 54F, 55D) and graphsage_metadata.json."""
    out = DATA / "graph"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((up / "graphsage_metadata.json").read_text())
    meta.pop("feature_columns", None)
    struct = json.loads((up / "graph_structure_metadata.json").read_text())
    meta["test_confusion"] = {"tn": 37130, "fp": 534, "fn": 592, "tp": 1548}
    meta["test_support"] = {"licit": 37664, "illicit": 2140}
    meta["dataset"] = {"labeled_licit": 251088, "labeled_illicit": 14266,
                       "unlabeled": 557588,
                       "split": {"train": 185747, "validation": 39803, "test": 39804}}
    meta["feature_names"] = struct["feature_columns"]
    (out / "graph_summary.json").write_text(json.dumps(meta, indent=2))

    pd.DataFrame({
        "epoch": [1, 2, 3, 4, 5],
        "train_loss": [0.2473, 0.2243, 0.2104, 0.2021, 0.1951],
        "val_pr_auc": [0.7457, 0.7652, 0.7749, 0.7923, 0.8101],
        "val_roc_auc": [0.9661, 0.9706, 0.9714, 0.9744, 0.9760],
    }).to_csv(out / "training_history.csv", index=False)

    rows = [(0.10, .1727, .9846, .2938, 2107, 10094, 33, 27569, 12201),
            (0.15, .1940, .9813, .3239, 2100, 8725, 40, 28938, 10825),
            (0.20, .2247, .9790, .3656, 2095, 7227, 45, 30436, 9322),
            (0.25, .2421, .9743, .3878, 2085, 6527, 55, 31136, 8612),
            (0.30, .2597, .9696, .4097, 2075, 5915, 65, 31748, 7990),
            (0.35, .2758, .9640, .4289, 2063, 5417, 77, 32246, 7480),
            (0.40, .2936, .9593, .4496, 2053, 4940, 87, 32723, 6993),
            (0.45, .3133, .9495, .4712, 2032, 4453, 108, 33210, 6485),
            (0.50, .3366, .9336, .4948, 1998, 3938, 142, 33725, 5936),
            (0.55, .3623, .9234, .5204, 1976, 3478, 164, 34185, 5454),
            (0.60, .3888, .9065, .5442, 1940, 3050, 200, 34613, 4990),
            (0.65, .4224, .8911, .5731, 1907, 2608, 233, 35055, 4515),
            (0.70, .4673, .8706, .6081, 1863, 2124, 277, 35539, 3987),
            (0.75, .5198, .8402, .6423, 1798, 1661, 342, 36002, 3459),
            (0.80, .5865, .8028, .6778, 1718, 1211, 422, 36452, 2929),
            (0.85, .6624, .7682, .7114, 1644, 838, 496, 36825, 2482),
            (0.90, .7483, .7252, .7366, 1552, 522, 588, 37141, 2074),
            (0.95, .8419, .6098, .7073, 1305, 245, 835, 37418, 1550)]
    pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "tp", "fp", "fn",
                                "tn", "alerts"]).to_csv(out / "threshold_analysis.csv", index=False)

    shap_a = [("first_sent_block", -0.944673, -2.460503),
              ("transacted_w_address_total", 0.767413, -1.586038),
              ("num_txs_as_sender", -0.896037, -1.176374),
              ("first_received_block", 0.684178, 0.993735),
              ("last_block_appeared_in", -1.452318, -0.850788),
              ("fees_max", -0.406897, -0.668444),
              ("fees_total", -0.408793, -0.419688),
              ("first_block_appeared_in", -1.396960, -0.380681),
              ("total_txs", -0.410959, -0.172396),
              ("fees_as_share_min", -0.135734, -0.121437)]
    shap_b = [("first_sent_block", -0.944673, -2.491021),
              ("num_txs_as_sender", -0.896037, -1.472967),
              ("transacted_w_address_total", 0.535500, -1.197308),
              ("first_received_block", 0.690769, 0.851294),
              ("num_txs_as receiver", 0.472219, 0.714282),
              ("fees_max", -0.417964, -0.613864),
              ("total_txs", -0.410959, -0.314028),
              ("fees_median", -0.393693, -0.250233),
              ("fees_as_share_max", -0.152879, -0.093231),
              ("last_block_appeared_in", -0.905650, -0.086156)]
    wallets = {
        "wallet_b": {"label": "Wallet 3KGiq1…yo4cT (node 793784)", "node_id": 793784,
                     "address": "3KGiq1k2JW7SFvNCTBdKLoqPSAAvgyo4cT",
                     "predicted_class": "LICIT", "benchmark_label": "LICIT",
                     "illicit_probability": 3.462070452542321e-08,
                     "sampled_incoming_neighbors": 10, "sampled_local_nodes": 13,
                     "original_margin": -17.178814, "expected_margin": -12.225582,
                     "sum_shap": -4.953232},
        "wallet_a": {"label": "Test wallet (node 783573)", "node_id": 783573, "address": None,
                     "predicted_class": "LICIT", "benchmark_label": "LICIT",
                     "illicit_probability": None,
                     "sampled_incoming_neighbors": 9, "sampled_local_nodes": None,
                     "original_margin": -21.116768, "expected_margin": -14.274154,
                     "sum_shap": -6.842614},
    }
    for k, rows_ in (("wallet_a", shap_a), ("wallet_b", shap_b)):
        d = pd.DataFrame(rows_, columns=["feature", "feature_value", "shap_value"])
        d["absolute_shap"] = d.shap_value.abs()
        d.to_csv(out / f"{k}_shap_top10.csv", index=False)
    (out / "wallets.json").write_text(json.dumps(wallets, indent=2))
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(up / "graphsage_best.pt", ROOT / "models" / "graphsage_best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", default="/mnt/user-data/uploads")
    a = ap.parse_args()
    up = Path(a.uploads)
    build_market(up)
    build_graph(up)
    build_sentiment(up)
    print("done")
