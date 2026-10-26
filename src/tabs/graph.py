"""Graph tab: GraphSAGE wallet-illicitness model, evaluation and SHAP explanations."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import data as D
from .. import theme as T


def _confusion_fig(c: dict) -> go.Figure:
    z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    rowtot = [sum(z[0]), sum(z[1])]
    text = [[f"{z[i][j]:,}<br><span style='font-size:11px'>{z[i][j]/rowtot[i]*100:.1f}% of row</span>"
             for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=[[z[0][0] / rowtot[0], z[0][1] / rowtot[0]],
           [z[1][0] / rowtot[1], z[1][1] / rowtot[1]]],
        x=["Predicted licit", "Predicted illicit"], y=["Actual licit", "Actual illicit"],
        text=text, texttemplate="%{text}", textfont=dict(size=15, color="#fff"),
        colorscale=[[0, "#141A28"], [1, T.BTC]], showscale=False, hoverinfo="skip", xgap=4, ygap=4))
    fig.update_layout(title="Test-set confusion matrix (threshold 0.90)")
    fig.update_yaxes(autorange="reversed")
    return T.style_fig(fig, 340, legend=False)


def _waterfall(w: dict, shap: pd.DataFrame) -> go.Figure:
    d = shap.sort_values("absolute_shap", ascending=False).reset_index(drop=True)
    rest = w["sum_shap"] - d.shap_value.sum()
    names = ["Baseline<br>(average)"] + [f.replace("_", " ") for f in d.feature]
    vals = [w["expected_margin"]] + d.shap_value.tolist()
    measure = ["absolute"] + ["relative"] * len(d)
    if abs(rest) > 1e-6:
        names.append("other features"); vals.append(rest); measure.append("relative")
    names.append("Final<br>(this wallet)"); vals.append(w["original_margin"]); measure.append("total")
    fig = go.Figure(go.Waterfall(
        x=names, y=vals, measure=measure,
        connector=dict(line=dict(color="rgba(255,255,255,.25)")),
        increasing=dict(marker=dict(color=T.RED)), decreasing=dict(marker=dict(color=T.GREEN)),
        totals=dict(marker=dict(color=T.BTC)),
        text=[f"{v:+.2f}" if m == "relative" else f"{v:.2f}" for v, m in zip(vals, measure)],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    fig.update_layout(title="How each feature moved the score (illicit − licit logit margin; lower = more licit)",
                      yaxis_title="logit margin", xaxis=dict(tickangle=-40), margin=dict(b=130))
    return T.style_fig(fig, 520, legend=False)


def render() -> None:
    G = D.load_graph()
    gs, wal, hist, thr, shap = G["summary"], G["wallets"], G["history"], G["thresholds"], G["shap"]
    conf = gs["test_confusion"]
    n_test = sum(gs["test_support"].values())
    licit_share = gs["test_support"]["licit"] / n_test

    T.section("Graph intelligence", "A two-layer sampled <b>GraphSAGE</b> network classifies Bitcoin wallets as "
                                    "<b>licit</b> or <b>illicit</b> from their own features plus those of the wallets that send to them.")

    T.kpi_row([
        ("PR-AUC", f"{gs['test_pr_auc']:.3f}", "ranking quality on the rare illicit class", T.BTC),
        ("ROC-AUC", f"{gs['test_roc_auc']:.3f}", "licit vs illicit separation", T.CYAN),
        ("Precision", f"{gs['test_precision']:.1%}", "of flagged wallets, truly illicit", T.GREEN),
        ("Recall", f"{gs['test_recall']:.1%}", "of illicit wallets, caught", T.VIOLET),
        ("F1-score", f"{gs['test_f1']:.3f}", f"threshold {gs['decision_threshold']}", T.AMBER),
    ])
    st.write("")
    T.kpi_row([
        ("Wallets (nodes)", f"{gs['num_nodes']:,}", "unique Bitcoin addresses", T.CYAN),
        ("Transactions links", f"{gs['num_edges']:,}", "directed wallet → wallet edges", T.VIOLET),
        ("Features / wallet", f"{len(gs['feature_names'])}", "log1p + standard scaling", T.BTC),
        ("Labelled wallets", f"{gs['dataset']['labeled_licit'] + gs['dataset']['labeled_illicit']:,}",
         f"{gs['dataset']['labeled_illicit']:,} illicit (5.4%)", T.RED),
    ])
    T.callout(
        f"<b>Read accuracy with care.</b> The model's {gs['test_accuracy']:.1%} accuracy looks excellent, but {licit_share:.1%} of "
        f"test wallets are licit, so a model that always says “licit” scores {licit_share:.1%}. On the rare illicit class the honest "
        f"numbers are precision {gs['test_precision']:.1%} and recall {gs['test_recall']:.1%}.")

    # ---- evaluation -------------------------------------------------------------------
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        st.plotly_chart(_confusion_fig(conf), width="stretch", key="g_conf")
        st.caption(f"Of {conf['tp'] + conf['fp']:,} wallets flagged, {conf['tp']:,} were truly illicit and {conf['fp']:,} were false alarms; "
                   f"{conf['fn']:,} illicit wallets were missed.")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.epoch, y=hist.val_pr_auc, name="Validation PR-AUC", mode="lines+markers",
                                 line=dict(color=T.BTC, width=3)))
        fig.add_trace(go.Scatter(x=hist.epoch, y=hist.val_roc_auc, name="Validation ROC-AUC", mode="lines+markers",
                                 line=dict(color=T.CYAN, width=3)))
        fig.add_trace(go.Scatter(x=hist.epoch, y=hist.train_loss, name="Training loss", mode="lines+markers",
                                 line=dict(color=T.VIOLET, width=2, dash="dot"), yaxis="y2"))
        fig.update_layout(title="Training progress (best epoch = 5)", xaxis=dict(title="epoch", dtick=1),
                          yaxis=dict(title="validation score", range=[0.7, 1.0]),
                          yaxis2=dict(title="train loss", overlaying="y", side="right", showgrid=False))
        T.style_fig(fig, 340)
        st.plotly_chart(fig, width="stretch", key="g_hist")

    # ---- threshold explorer -------------------------------------------------------------
    T.section("Decision-threshold explorer", "Move the alert threshold to see the trade-off between catching illicit wallets and raising false alarms "
                                              "(validation set, 39,803 wallets).")
    thresholds = thr.threshold.round(2).tolist()
    sel = st.select_slider("Flag a wallet as illicit when its probability is at least", options=thresholds,
                           value=0.90, key="g_thr")
    row = thr[thr.threshold.round(2) == sel].iloc[0]
    T.kpi_row([
        ("Precision", f"{row.precision:.1%}", "flagged wallets that are illicit", T.GREEN),
        ("Recall", f"{row.recall:.1%}", "illicit wallets caught", T.VIOLET),
        ("F1", f"{row.f1:.3f}", "balance of the two", T.AMBER),
        ("Alerts raised", f"{int(row.alerts):,}", f"{int(row.fp):,} false alarms · {int(row.fn):,} missed", T.RED),
    ])
    fig = go.Figure()
    for col, name, colr in [("precision", "Precision", T.GREEN), ("recall", "Recall", T.VIOLET), ("f1", "F1", T.AMBER)]:
        fig.add_trace(go.Scatter(x=thr.threshold, y=thr[col], name=name, mode="lines+markers", line=dict(color=colr, width=2.5)))
    fig.add_vline(x=sel, line_dash="dot", line_color="#fff")
    fig.add_vline(x=0.90, line_color=T.BTC, line_width=1, annotation_text="chosen: 0.90", annotation_position="top left")
    fig.update_layout(title="Precision / recall trade-off by threshold", xaxis_title="decision threshold", yaxis_title="score")
    T.style_fig(fig, 340)
    st.plotly_chart(fig, width="stretch", key="g_thr_fig")
    if sel < 0.6:
        st.caption("Low thresholds catch nearly every illicit wallet but bury analysts in false alarms; the validation set picked 0.90 for the best F1.")

    # ---- explainability -------------------------------------------------------------------
    T.section("Explainable AI: why did the model decide this?",
              "Kernel SHAP breaks one prediction into contributions from individual features.")
    options = {"wallet_b": wal["wallet_b"]["label"], "wallet_a": wal["wallet_a"]["label"]}
    pick = st.radio("Saved wallet explanation", list(options), format_func=options.get, horizontal=True, key="g_wallet")
    w = wal[pick]
    d = shap[pick].copy()
    prob = w.get("illicit_probability")
    prob_txt = f"{prob:.2e}" if prob is not None else "≈ 0"

    c1, c2 = st.columns([1, 2.4], gap="large")
    with c1:
        T.html(f"""
        <div class="card glow">
          <div class="kpi-label">Model verdict</div>
          <div class="verdict" style="color:{T.GREEN if w['predicted_class']=='LICIT' else T.RED}">{w['predicted_class']}</div>
          <div class="kpi-sub">Illicit probability <b style="color:#fff">{prob_txt}</b><br>
          Alert threshold <b style="color:#fff">{gs['decision_threshold']}</b><br>
          Benchmark label <b style="color:#fff">{w['benchmark_label']}</b><br>
          Sampled incoming neighbours <b style="color:#fff">{w['sampled_incoming_neighbors']}</b><br>
          Node ID <b style="color:#fff">{w['node_id']}</b></div>
          <div class="kpi-sub" style="margin-top:10px">⚠️ A model estimate, not proof that a wallet is safe or criminal.</div>
        </div>""")
    with c2:
        st.plotly_chart(_waterfall(w, d), width="stretch", key=f"g_wf_{pick}")

    tbl = d.copy()
    tbl["Feature"] = tbl.feature.map(D.describe_feature)
    tbl["Technical name"] = tbl.feature
    tbl["Standardised value"] = tbl.feature_value.map(lambda v: f"{v:+.2f}")
    tbl["SHAP"] = tbl.shap_value.map(lambda v: f"{v:+.3f}")
    tbl["Pushes toward"] = tbl.shap_value.map(lambda v: "🟢 LICIT" if v < 0 else "🔴 ILLICIT")
    st.dataframe(tbl[["Feature", "Technical name", "Standardised value", "SHAP", "Pushes toward"]],
                 width="stretch", hide_index=True)
    top = d.iloc[0]
    T.callout(
        f"<b>In plain English.</b> The biggest push toward <b>licit</b> is “{D.describe_feature(top.feature).lower()}”, "
        f"which contributes {top.shap_value:+.2f}. Several features push the other way (red bars), but the sum of all contributions moves the score from the "
        f"average-wallet baseline of {w['expected_margin']:.1f} down to {w['original_margin']:.1f}. Negative margins mean the model "
        f"considers the wallet far more likely licit than illicit. “Standardised value” 0 = an average wallet; negative = below average.")

    # ---- lookup ---------------------------------------------------------------------------
    with st.expander("🔎 Look up a wallet"):
        q = st.text_input("Bitcoin address or node ID", placeholder="3KGiq1k2JW7SFvNCTBdKLoqPSAAvgyo4cT", key="g_lookup").strip()
        if q:
            hit = next((k for k, v in wal.items() if q == str(v["node_id"]) or (v["address"] and q == v["address"])), None)
            if hit:
                v = wal[hit]
                st.success(f"Saved result found · {v['predicted_class']} · node {v['node_id']}. Select it above to see the SHAP breakdown.")
            else:
                st.info("This wallet has no saved result. The web app shows precomputed predictions only: scoring any address live "
                        "needs the 170 MB graph feature file and neighbour index, which are too large for free hosting. "
                        "See the README section “Optional: live wallet scoring”.")

    with st.expander("How the GraphSAGE model works"):
        st.markdown(
            f"""
1. **Each wallet is a node** with {len(gs['feature_names'])} behavioural features (transaction counts, amounts, fees, timing gaps, counterparties).
2. **Neighbour sampling.** For a wallet, the model samples up to **{gs['fanout_1']}** wallets that send to it, and up to **{gs['fanout_2']}** that send to *those*.
3. **Mean aggregation.** Two layers average the neighbours' features and combine them with the wallet's own features, so a wallet is judged partly by *who it deals with*.
4. **Classifier head** outputs an illicit probability. Class weights offset the 5.4% illicit rate; the 0.90 threshold was tuned on the validation set.

Trainable parameters: ~15.6k · Split 70/15/15 stratified ({gs['dataset']['split']['train']:,} / {gs['dataset']['split']['validation']:,} / {gs['dataset']['split']['test']:,}) · Best epoch {gs['best_epoch']}.""")
