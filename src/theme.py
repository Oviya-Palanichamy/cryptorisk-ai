"""Visual identity: dark crypto theme, HTML helpers and Plotly styling."""
from __future__ import annotations

import streamlit as st

# ---- palette ---------------------------------------------------------------
BTC = "#F7931A"      # bitcoin orange (primary)
CYAN = "#22D3EE"
VIOLET = "#A78BFA"
GREEN = "#34D399"
AMBER = "#FBBF24"
RED = "#F87171"
TEXT = "#E6EAF2"
MUTED = "#8B95A9"
BG = "#0B0E14"
CARD = "rgba(255,255,255,0.035)"
GRID = "rgba(255,255,255,0.07)"

LEVEL_COLORS = {"Low": GREEN, "Moderate": AMBER, "High": BTC, "Extreme": RED}
SENTIMENT_COLORS = {"positive": GREEN, "neutral": "#64748B", "negative": RED}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, .stApp, .stMarkdown, .stMarkdown p, button, input, textarea, label, [data-testid="stWidgetLabel"] {{ font-family: 'Inter', sans-serif; }}
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {{ font-family: 'Material Symbols Rounded', 'Material Icons' !important; }}
.stApp {{
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(247,147,26,.16), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(139,92,246,.15), transparent 55%),
    radial-gradient(700px 500px at 50% 110%, rgba(34,211,238,.08), transparent 60%),
    {BG};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1280px; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ---------- hero ---------- */
.hero {{
  position: relative; overflow: hidden;
  padding: 30px 34px; border-radius: 22px; margin-bottom: 18px;
  border: 1px solid rgba(247,147,26,.28);
  background: linear-gradient(120deg, rgba(247,147,26,.14), rgba(139,92,246,.10) 55%, rgba(11,14,20,.6));
}}
.hero:before {{
  content:""; position:absolute; inset:0; opacity:.5; pointer-events:none;
  background-image: linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px);
  background-size: 36px 36px;
  -webkit-mask-image: radial-gradient(circle at 80% 20%, #000, transparent 70%);
          mask-image: radial-gradient(circle at 80% 20%, #000, transparent 70%);
}}
.hero > * {{ position: relative; }}
.brand {{ display:flex; align-items:center; gap:14px; }}
.logo {{
  width:46px; height:46px; border-radius:14px; display:grid; place-items:center;
  background: linear-gradient(135deg, {BTC}, #FFB84D); color:#1a1000; font-weight:800; font-size:26px;
  box-shadow: 0 0 28px rgba(247,147,26,.45);
}}
.hero h1 {{ margin:0 !important; padding:0 !important; font-size:34px; font-weight:800; letter-spacing:-1px; color:#fff; line-height:1.1; }}
.hero h1 span {{ color:{BTC} !important; }}
.hero p {{ margin:10px 0 0 0; color:#B6C0D4; font-size:15px; max-width:760px; }}
.ticker {{
  display:inline-flex; align-items:center; gap:10px; margin-top:16px; padding:7px 14px; border-radius:999px;
  background: rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.10); font-family:'JetBrains Mono',monospace; font-size:13px;
}}
.dot {{ width:8px; height:8px; border-radius:50%; background:{GREEN}; box-shadow:0 0 10px {GREEN}; display:inline-block; }}
.dot.off {{ background:{MUTED}; box-shadow:none; }}

/* ---------- cards ---------- */
.card {{
  border-radius:18px; padding:18px 20px; height:100%;
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  backdrop-filter: blur(6px);
}}
.card.glow {{ box-shadow: 0 0 0 1px rgba(247,147,26,.12), 0 18px 40px -22px rgba(247,147,26,.45); }}
.kpi-label {{ font-size:11.5px; letter-spacing:.9px; text-transform:uppercase; color:{MUTED}; font-weight:600; }}
.kpi-value {{ font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:700; color:#fff; margin-top:6px; line-height:1.15; }}
.kpi-sub {{ font-size:12.5px; color:#9AA6BC; margin-top:6px; }}
.kpi-bar {{ height:3px; border-radius:3px; margin-bottom:12px; }}

.sec-title {{ font-size:21px; font-weight:700; color:#fff; margin:26px 0 2px 0; letter-spacing:-.3px; }}
.sec-sub {{ font-size:13.5px; color:{MUTED}; margin-bottom:14px; }}

.pill {{
  display:inline-block; padding:3px 11px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.3px;
  border:1px solid; white-space:nowrap;
}}
.callout {{
  border-radius:14px; padding:13px 16px; font-size:13.8px; line-height:1.55; color:#CBD5E6;
  background: rgba(34,211,238,.07); border:1px solid rgba(34,211,238,.25); margin:8px 0 14px 0;
}}
.callout.warn {{ background: rgba(251,191,36,.08); border-color: rgba(251,191,36,.35); }}
.callout.bad  {{ background: rgba(248,113,113,.08); border-color: rgba(248,113,113,.35); }}
.callout.good {{ background: rgba(52,211,153,.08); border-color: rgba(52,211,153,.30); }}
.callout b {{ color:#fff; }}

.flow {{ display:flex; gap:10px; flex-wrap:wrap; align-items:stretch; }}
.flow .step {{ flex:1 1 120px; border-radius:14px; padding:14px; background:{CARD}; border:1px solid rgba(255,255,255,.08); }}
.flow .step b {{ display:block; color:#fff; font-size:13.5px; margin-bottom:4px; }}
.flow .step span {{ color:#9AA6BC; font-size:12.5px; }}
.flow .arrow {{ align-self:center; color:{BTC}; font-size:20px; }}

.verdict {{ font-size:26px; font-weight:800; letter-spacing:-.5px; }}

/* ---------- streamlit widgets ---------- */
[role="tablist"] {{
  gap:6px; padding:5px; border-radius:16px; background: rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.08); margin-bottom:10px; overflow-x:auto;
}}
[role="tab"] {{
  height:42px; border-radius:11px; padding:0 18px; font-weight:600; color:#9AA6BC; background:transparent; border:0;
}}
[role="tab"] p {{ font-size:15px; font-weight:600; }}
[role="tab"][aria-selected="true"] {{
  background: linear-gradient(135deg, rgba(247,147,26,.28), rgba(247,147,26,.10)); color:#fff !important;
  box-shadow: inset 0 0 0 1px rgba(247,147,26,.45);
}}
[role="tab"][aria-selected="true"] p {{ color:#fff; }}
[data-testid="stTabs"] [role="tablist"] + div, [role="tablist"] [data-baseweb] {{ border-bottom:0; }}
div[data-testid="stPlotlyChart"] {{ border-radius:16px; }}
.stButton > button {{ white-space:normal; height:auto; min-height:58px; line-height:1.35; border-radius:11px; border:1px solid rgba(247,147,26,.35); background: rgba(247,147,26,.08); }}
.stButton > button p {{ white-space:normal; overflow:visible; text-overflow:clip; }}
.stButton > button:hover {{ border-color:{BTC}; color:#fff; background: rgba(247,147,26,.18); }}
[data-testid="stChatMessage"] {{ background: {CARD}; border:1px solid rgba(255,255,255,.07); border-radius:16px; }}
[data-testid="stExpander"] {{ border-radius:14px; border:1px solid rgba(255,255,255,.08); background:{CARD}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render raw HTML safely inside markdown (removes indentation/newlines so it is never read as a code block)."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines()).replace("$", "&#36;")
    st.markdown(flat, unsafe_allow_html=True)


def esc(text: str) -> str:
    """Escape $ so Streamlit does not read '$10,000 … $316' as a LaTeX formula."""
    return text.replace("$", "\\$")


def section(title: str, subtitle: str = "") -> None:
    html(f'<div class="sec-title">{title}</div><div class="sec-sub">{subtitle}</div>')


def kpi(label: str, value: str, sub: str = "", color: str = BTC) -> str:
    return (f'<div class="card"><div class="kpi-bar" style="background:{color}"></div>'
            f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            html(kpi(*it))


def pill(text: str, color: str) -> str:
    return (f'<span class="pill" style="color:{color};border-color:{color}66;'
            f'background:{color}1f">{text}</span>')


def callout(text: str, kind: str = "") -> None:
    html(f'<div class="callout {kind}">{text}</div>')


def style_fig(fig, height: int = 380, legend: bool = True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12.5),
        margin=dict(l=8, r=8, t=82, b=8),
        hoverlabel=dict(bgcolor="#151A26", bordercolor="rgba(255,255,255,.2)", font_color="#fff"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)"),
        title=dict(font=dict(size=15, color="#fff"), x=0.0, xanchor="left", y=0.97, yanchor="top"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"
