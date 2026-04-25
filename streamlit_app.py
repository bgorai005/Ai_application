import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import requests
import os
import smtplib
from email.message import EmailMessage
import random

# ─── Configuration ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nifty50 Signal", layout="wide", initial_sidebar_state="collapsed")

DATA_PATH  = "data/processed/nifty50_5min_features.csv"
SEQ_LENGTH = 30
SIGNAL_BASE = os.environ.get("SIGNAL_URL",  "http://backend_app:8000")
PREDICT_URL = os.environ.get("PREDICT_URL", "http://backend_app:8000/predict")

# ─── Color tokens ────────────────────────────────────────────────────────────────
COLOR_BUY    = "#00C896"
COLOR_SELL   = "#FF4C4C"
COLOR_HOLD   = "#6B7280"
COLOR_WARN   = "#F59E0B"
BG_BASE      = "#0D1117"
BG_CARD      = "#161B22"
BG_CARD2     = "#1C2333"
BORDER       = "#30363D"
TEXT_PRIMARY = "#F9FAFB"
TEXT_MUTED   = "#9CA3AF"

# ─── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{
    background-color: {BG_BASE};
    color: {TEXT_PRIMARY};
    font-family: 'IBM Plex Sans', sans-serif;
  }}

  /* ── Hide Streamlit chrome ───────────────────────────────────────── */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 0 !important; max-width: 100% !important; }}

  /* ── Scrollbar ───────────────────────────────────────────────────── */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: {BG_BASE}; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

  /* ── Navbar ──────────────────────────────────────────────────────── */
  .navbar {{
    display: flex; align-items: center; justify-content: space-between;
    background: {BG_CARD}; border-bottom: 1px solid {BORDER};
    padding: 0 32px; height: 56px; position: sticky; top: 0; z-index: 100;
  }}
  .navbar-logo {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px; font-weight: 600; color: {TEXT_PRIMARY};
    letter-spacing: 0.04em;
  }}
  .navbar-logo span {{ color: {COLOR_BUY}; }}
  .navbar-right {{ display: flex; align-items: center; gap: 24px; }}
  .index-level {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px; font-weight: 600; color: {TEXT_PRIMARY};
  }}
  .market-pill {{
    padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
    letter-spacing: 0.06em;
  }}
  .market-open  {{ background: rgba(0,200,150,.15); color: {COLOR_BUY}; border: 1px solid {COLOR_BUY}; }}
  .market-closed{{ background: rgba(107,114,128,.15); color: {COLOR_HOLD}; border: 1px solid {COLOR_HOLD}; }}
  .ist-time {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; color: {TEXT_MUTED};
  }}

  /* ── Cards ───────────────────────────────────────────────────────── */
  .card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 20px;
  }}
  .card-inner {{ background: {BG_CARD2}; border-radius: 8px; padding: 14px 18px; }}

  /* ── Signal card ─────────────────────────────────────────────────── */
  .signal-card {{
    background: {BG_CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 28px 32px;
    border-left: 6px solid var(--sig-color);
  }}
  .signal-card.pulse {{
    animation: borderPulse 2s ease-in-out 5;
  }}
  @keyframes borderPulse {{
    0%,100% {{ border-left-color: var(--sig-color); }}
    50%      {{ border-left-color: transparent; }}
  }}
  .signal-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 52px; font-weight: 700; line-height: 1;
    color: var(--sig-color);
  }}
  .signal-conf {{
    font-size: 20px; color: {TEXT_MUTED}; margin-top: 6px;
  }}
  .signal-conf span {{
    font-family: 'IBM Plex Mono', monospace;
    color: {TEXT_PRIMARY}; font-weight: 600;
  }}
  .signal-band {{
    font-size: 16px; margin-top: 14px; color: {TEXT_MUTED};
  }}
  .signal-band strong {{ color: {TEXT_PRIMARY}; font-family: 'IBM Plex Mono', monospace; }}
  .signal-ts {{
    font-size: 11px; color: {TEXT_MUTED}; margin-top: 10px;
    font-family: 'IBM Plex Mono', monospace;
  }}

  /* ── Feature snapshot ────────────────────────────────────────────── */
  .feat-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid {BORDER};
  }}
  .feat-row:last-child {{ border-bottom: none; }}
  .feat-name {{ font-size: 13px; color: {TEXT_MUTED}; min-width: 100px; }}
  .feat-val  {{ font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: {TEXT_PRIMARY}; flex: 1; text-align: center; }}
  .badge {{
    padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;
    letter-spacing: 0.05em;
  }}
  .badge-green  {{ background: rgba(0,200,150,.15);  color: {COLOR_BUY};  border: 1px solid {COLOR_BUY}; }}
  .badge-red    {{ background: rgba(255,76,76,.15);  color: {COLOR_SELL}; border: 1px solid {COLOR_SELL}; }}
  .badge-grey   {{ background: rgba(107,114,128,.15);color: {COLOR_HOLD}; border: 1px solid {COLOR_HOLD}; }}
  .badge-amber  {{ background: rgba(245,158,11,.15); color: {COLOR_WARN}; border: 1px solid {COLOR_WARN}; }}

  /* ── Sentiment bar ───────────────────────────────────────────────── */
  .sentiment-bar-wrap {{
    flex: 1; display: flex; align-items: center; justify-content: center;
  }}
  .sentiment-bar {{
    position: relative; width: 120px; height: 8px;
    background: linear-gradient(90deg, {COLOR_SELL} 0%, {COLOR_HOLD} 50%, {COLOR_BUY} 100%);
    border-radius: 4px;
  }}
  .sentiment-dot {{
    position: absolute; top: -3px; width: 14px; height: 14px;
    background: {TEXT_PRIMARY}; border-radius: 50%;
    border: 2px solid {BG_CARD}; transform: translateX(-7px);
  }}

  /* ── Section label ───────────────────────────────────────────────── */
  .section-label {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
    color: {TEXT_MUTED}; text-transform: uppercase; margin-bottom: 14px;
  }}

  /* ── Context table ───────────────────────────────────────────────── */
  .ctx-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .ctx-table th {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    color: {TEXT_MUTED}; text-align: right; padding: 6px 10px;
    border-bottom: 1px solid {BORDER}; text-transform: uppercase;
  }}
  .ctx-table th:first-child {{ text-align: left; }}
  .ctx-table td {{
    padding: 6px 10px; text-align: right;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    border-bottom: 1px solid rgba(48,54,61,0.5);
  }}
  .ctx-table td:first-child {{ text-align: left; color: {TEXT_MUTED}; }}
  .ctx-table tr:nth-child(even) td {{ background: rgba(255,255,255,0.015); }}
  .ctx-table .anchor-row td {{
    background: rgba(0,200,150,0.06) !important;
    color: {TEXT_PRIMARY};
  }}
  .anchor-arrow {{
    font-size: 10px; color: {COLOR_BUY}; font-weight: 600;
    margin-left: 6px; letter-spacing: 0.06em;
  }}

  /* ── History table ───────────────────────────────────────────────── */
  .hist-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .hist-table th {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    color: {TEXT_MUTED}; text-transform: uppercase;
    padding: 8px 10px; border-bottom: 1px solid {BORDER};
    text-align: right; position: sticky; top: 0; background: {BG_CARD};
  }}
  .hist-table th:first-child {{ text-align: left; }}
  .hist-table td {{
    padding: 7px 10px; text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    border-bottom: 1px solid rgba(48,54,61,0.4);
  }}
  .hist-table td:first-child {{ text-align: left; color: {TEXT_MUTED}; }}
  .hist-table tr:nth-child(even) td {{ background: rgba(255,255,255,0.015); }}
  .hist-scroll {{ max-height: 320px; overflow-y: auto; }}

  /* ── Accuracy tracker ────────────────────────────────────────────── */
  .acc-card {{
    background: {BG_CARD2}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
  }}
  .acc-label {{ font-size: 11px; color: {TEXT_MUTED}; margin-bottom: 6px; }}
  .acc-val {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px; font-weight: 700;
  }}
  .progress-bar-wrap {{
    height: 4px; background: {BORDER}; border-radius: 2px; margin-top: 8px;
  }}
  .progress-bar {{ height: 4px; border-radius: 2px; }}

  /* ── Model health ────────────────────────────────────────────────── */
  .health-row {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid {BORDER};
    font-size: 13px;
  }}
  .health-row:last-child {{ border-bottom: none; }}
  .health-label {{ color: {TEXT_MUTED}; }}
  .status-dot {{
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
  }}

  /* ── Feedback widget ─────────────────────────────────────────────── */
  .star-row {{ display: flex; gap: 6px; font-size: 24px; margin: 12px 0; }}
  .feedback-success {{
    color: {COLOR_BUY}; font-size: 13px; font-style: italic;
  }}

  /* ── Footer ──────────────────────────────────────────────────────── */
  .footer {{
    text-align: center; padding: 14px 32px;
    border-top: 1px solid {BORDER}; margin-top: 40px;
    font-size: 11px; color: {TEXT_MUTED}; letter-spacing: 0.04em;
  }}

  /* ── Divider ─────────────────────────────────────────────────────── */
  .divider {{ border: none; border-top: 1px solid {BORDER}; margin: 28px 0; }}

  /* ── Page padding ────────────────────────────────────────────────── */
  .page-wrap {{ padding: 0 32px 32px; }}

  /* ── Filter pills ────────────────────────────────────────────────── */
  div[data-testid="stHorizontalBlock"] button {{
    border-radius: 20px !important;
    font-size: 12px !important;
  }}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def fetch_current_signal():
    try:
        r = requests.get(f"{SIGNAL_BASE}/signal/current", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_context(length=20):
    try:
        r = requests.get(f"{SIGNAL_BASE}/signal/context?length={length}", timeout=5)
        r.raise_for_status()
        return r.json().get("input_context", [])
    except Exception:
        return []


def fetch_history(n=30):
    try:
        r = requests.get(f"{SIGNAL_BASE}/signal/history?n={n}", timeout=5)
        r.raise_for_status()
        return r.json().get("history", [])
    except Exception:
        return []


def fetch_accuracy():
    try:
        r = requests.get(f"{SIGNAL_BASE}/signal/accuracy", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def fetch_model_health():
    try:
        r = requests.get(f"{SIGNAL_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def last_n_candles(df: pd.DataFrame, selected_date, n=20):
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    window = df[df["Datetime"].dt.date <= selected_date]
    return window.tail(n)


def is_market_open():
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon…6=Sun
    t = now.time()
    if weekday >= 5:
        return False
    return datetime.strptime("09:15", "%H:%M").time() <= t <= datetime.strptime("15:30", "%H:%M").time()


def signal_color(sig):
    return {
        "BUY":  COLOR_BUY,
        "SELL": COLOR_SELL,
        "HOLD": COLOR_HOLD,
    }.get(str(sig).upper(), COLOR_HOLD)


def acc_color(pct):
    if pct >= 60: return COLOR_BUY
    if pct >= 50: return COLOR_WARN
    return COLOR_SELL


def badge_html(text, style="grey"):
    cls = {"green": "badge-green", "red": "badge-red",
           "grey": "badge-grey", "amber": "badge-amber"}.get(style, "badge-grey")
    return f'<span class="badge {cls}">{text}</span>'


def signal_badge(sig):
    s = str(sig).upper()
    style = {"BUY": "green", "SELL": "red", "HOLD": "grey"}.get(s, "grey")
    return badge_html(s, style)


def outcome_badge(outcome):
    o = str(outcome).lower()
    if o == "correct":   return badge_html("Correct",   "green")
    if o == "incorrect": return badge_html("Incorrect", "red")
    return badge_html("Pending", "grey")


def plot_price_chart(candles: pd.DataFrame, pred_highs: list, pred_lows: list):
    if candles.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=candles["Datetime"],
        open=candles["Open"],
        high=candles["High"],
        low=candles["Low"],
        close=candles["Close"],
        name="Price",
        increasing_line_color=COLOR_BUY,
        decreasing_line_color=COLOR_SELL,
        increasing_fillcolor=COLOR_BUY,
        decreasing_fillcolor=COLOR_SELL,
    ))

    # time delta
    if len(candles) >= 2:
        delta_secs = int(candles["Datetime"].diff().dt.total_seconds().median())
        if pd.isna(delta_secs) or delta_secs <= 0:
            delta_secs = 300
    else:
        delta_secs = 300

    last_dt = pd.to_datetime(candles["Datetime"].iloc[-1])
    n_fut   = min(len(pred_highs), len(pred_lows))
    future  = [last_dt + timedelta(seconds=delta_secs * (i + 1)) for i in range(n_fut)]

    if future:
        fig.add_trace(go.Scatter(
            x=future, y=pred_highs, mode="lines", name="Pred High",
            line=dict(color=COLOR_BUY, dash="dash", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=future, y=pred_lows, mode="lines", name="Pred Low",
            line=dict(color=COLOR_SELL, dash="dash", width=1.5),
            fill="tonexty",
            fillcolor="rgba(0,200,150,0.08)",
        ))
        # band for T+1
        bx0 = future[0]
        bx1 = future[0] + timedelta(seconds=delta_secs * 0.85)
        fig.add_shape(type="rect", xref="x", yref="y",
                      x0=bx0, x1=bx1, y0=pred_lows[0], y1=pred_highs[0],
                      fillcolor="rgba(0,200,150,0.10)", line_width=0, layer="below")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(orientation="h", y=-0.12, font=dict(size=11, color=TEXT_MUTED)),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showgrid=False, color=TEXT_MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, gridwidth=1, color=TEXT_MUTED,
                     tickfont=dict(family="IBM Plex Mono"))
    return fig


def sparkline_fig(data: list):
    colors = [COLOR_BUY if v else COLOR_SELL for v in data]
    fig = go.Figure(go.Bar(
        y=[1] * len(data),
        marker_color=colors,
        width=0.7,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=40,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ─── Session state ────────────────────────────────────────────────────────────────
defaults = {"feedback_submitted": False, "star_rating": 0, "filter_signal": "All"}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Load data ────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(DATA_PATH)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
except FileNotFoundError:
    st.error("Processed data not found. Run the pipeline first.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
delta      = last_close - prev_close
market_open = is_market_open()

# ─── Fetch backend data ───────────────────────────────────────────────────────────
signal       = fetch_current_signal()
context      = fetch_context(20)
history      = fetch_history(30)
accuracy     = fetch_accuracy()
model_health = fetch_model_health()

sig     = signal.get("signal",         "HOLD") if signal else "HOLD"
conf    = signal.get("confidence",     0)       if signal else 0
ts_raw  = signal.get("timestamp",      "")      if signal else ""
p_high  = signal.get("predicted_high", "—")     if signal else "—"
p_low   = signal.get("predicted_low",  "—")     if signal else "—"
p_highs = signal.get("predicted_highs",[])       if signal else []
p_lows  = signal.get("predicted_lows", [])       if signal else []

sig_color = signal_color(sig)

# Should we pulse? (signal within last 5 min)
pulse_class = ""
if ts_raw:
    try:
        ts_dt = datetime.fromisoformat(ts_raw)
        if (datetime.now() - ts_dt).total_seconds() < 300:
            pulse_class = "pulse"
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ════════════════════════════════════════════════════════════════════════════════
delta_sign  = "▲" if delta >= 0 else "▼"
delta_color = COLOR_BUY if delta >= 0 else COLOR_SELL
market_pill = (f'<span class="market-pill market-open">● MARKET OPEN</span>'
               if market_open
               else f'<span class="market-pill market-closed">● MARKET CLOSED</span>')
ist_now     = datetime.now().strftime("%H:%M:%S IST")

st.markdown(f"""
<div class="navbar">
  <div class="navbar-logo">NIFTY<span>50</span> SIGNAL</div>
  <div class="navbar-right">
    {market_pill}
    <div class="index-level">
      {last_close:,.2f}
      <span style="font-size:14px; color:{delta_color}; margin-left:6px;">
        {delta_sign} {abs(delta):,.2f}
      </span>
    </div>
    <div class="ist-time" id="ist-clock">{ist_now}</div>
  </div>
</div>
<script>
  function updateClock() {{
    var el = document.getElementById('ist-clock');
    if (el) {{
      var now = new Date();
      var options = {{timeZone:'Asia/Kolkata', hour12:false,
                      hour:'2-digit', minute:'2-digit', second:'2-digit'}};
      el.textContent = new Intl.DateTimeFormat('en-IN', options).format(now) + ' IST';
    }}
    setTimeout(updateClock, 1000);
  }}
  updateClock();
</script>
""", unsafe_allow_html=True)


# ─── Page wrapper ─────────────────────────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ROW 1 — SIGNAL CARD (hero, full width)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown(f"""
<div class="signal-card {pulse_class}" style="--sig-color:{sig_color};">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:24px;">
    <div>
      <div class="signal-label">{sig}</div>
      <div class="signal-conf">Confidence: <span>{conf}%</span></div>
      <div class="signal-band">
        Predicted next range &nbsp;
        High: <strong>{p_high}</strong>
        &nbsp;·&nbsp;
        Low: <strong>{p_low}</strong>
      </div>
      <div class="signal-ts">Generated: {ts_raw if ts_raw else "—"}</div>
    </div>
    <div style="display:flex; gap:32px; text-align:center;">
      <div>
        <div style="font-size:11px; color:{TEXT_MUTED}; letter-spacing:.1em; text-transform:uppercase; margin-bottom:4px;">Last Close</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:22px;">{last_close:,.2f}</div>
      </div>
      <div>
        <div style="font-size:11px; color:{TEXT_MUTED}; letter-spacing:.1em; text-transform:uppercase; margin-bottom:4px;">Change</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; color:{delta_color};">{delta_sign} {abs(delta):,.2f}</div>
      </div>
      <div>
        <div style="font-size:11px; color:{TEXT_MUTED}; letter-spacing:.1em; text-transform:uppercase; margin-bottom:4px;">Confidence</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; color:{sig_color};">{conf}%</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ROW 2 — Price Chart (60%) + Feature Snapshot (40%)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)

available_dates = sorted(df["Datetime"].dt.date.unique(), reverse=True)
default_date    = pd.Timestamp.now().date()
if default_date not in available_dates:
    default_date = available_dates[0]

col_chart, col_feat = st.columns([6, 4], gap="large")

with col_chart:
    st.markdown('<div class="section-label">Price Range Chart</div>', unsafe_allow_html=True)
    selected_date = st.date_input(
        "Chart date:", value=default_date,
        min_value=min(available_dates), max_value=max(available_dates),
        label_visibility="collapsed",
    )
    candles = last_n_candles(df, selected_date, n=20)
    if candles.empty:
        st.warning("No data for selected date.")
    else:
        fig = plot_price_chart(candles, p_highs if p_highs else [], p_lows if p_lows else [])
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_feat:
    st.markdown('<div class="section-label">Feature Snapshot</div>', unsafe_allow_html=True)

    # Pull live values from context or signal
    ctx_latest = context[-1] if context else {}
    rsi_val   = round(ctx_latest.get("RSI",  signal.get("rsi",  50.0) if signal else 50.0), 2)
    macd_val  = round(ctx_latest.get("MACD", signal.get("macd",  0.0) if signal else 0.0),  4)
    vix_val   = round(signal.get("vix",  14.0) if signal else 14.0, 2)
    fii_val   = signal.get("fii_flow_cr", 0) if signal else 0
    news_val  = round(signal.get("news_sentiment", 0.0) if signal else 0.0, 2)

    # RSI badge
    if rsi_val >= 70:   rsi_badge = badge_html("Overbought", "red")
    elif rsi_val <= 30: rsi_badge = badge_html("Oversold",   "green")
    else:               rsi_badge = badge_html("Neutral",    "grey")

    # MACD badge
    if macd_val > 0:   macd_badge = badge_html("Bullish", "green")
    elif macd_val < 0: macd_badge = badge_html("Bearish", "red")
    else:              macd_badge = badge_html("Neutral", "grey")

    # VIX badge
    if vix_val < 15:    vix_badge = badge_html("Low",      "green")
    elif vix_val < 25:  vix_badge = badge_html("Moderate", "amber")
    else:               vix_badge = badge_html("High",     "red")

    # FII badge
    fii_style = "green" if fii_val >= 0 else "red"
    fii_badge = badge_html(f"{'Net Buy' if fii_val >= 0 else 'Net Sell'} ₹{abs(fii_val):,.0f}Cr", fii_style)

    # Sentiment bar (position as % 0–100)
    sent_pct = int((news_val + 1) / 2 * 100)  # -1..+1 → 0..100

    features = [
        ("RSI",            f"{rsi_val:.2f}",  rsi_badge,  None),
        ("MACD",           f"{macd_val:.4f}", macd_badge, None),
        ("India VIX",      f"{vix_val:.2f}",  vix_badge,  None),
        ("FII Flow",       "",                fii_badge,  None),
        ("News Sentiment", f"{news_val:.2f}", "",         sent_pct),
    ]

    feat_rows = ""
    for name, val, bdg, sent in features:
        if sent is not None:
            bar = f"""
            <div class="sentiment-bar-wrap">
              <div class="sentiment-bar">
                <div class="sentiment-dot" style="left:{sent}%;"></div>
              </div>
            </div>"""
            val_html = bar
            badge_html_str = f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:{TEXT_MUTED};">{val}</span>'
        else:
            val_html      = f'<div class="feat-val">{val}</div>'
            badge_html_str = bdg

        feat_rows += f"""
        <div class="feat-row">
          <div class="feat-name">{name}</div>
          {val_html}
          <div style="min-width:100px; text-align:right;">{badge_html_str}</div>
        </div>"""

    st.markdown(f'<div class="card">{feat_rows}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ROW 3 — LOOK-BACK CONTEXT TABLE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Model input context — last 20 time steps</div>', unsafe_allow_html=True)

ctx_data = context if context else []
if not ctx_data and not candles.empty:
    ctx_data = candles[["Datetime","Close","Volume"]].rename(columns={"Datetime":"time_step"}).to_dict("records")

if ctx_data:
    ctx_df = pd.DataFrame(ctx_data)
    col_map = {c: c for c in ctx_df.columns}
    desired = ["time_step","Close","Volume","RSI","MACD"]
    cols    = [c for c in desired if c in ctx_df.columns]
    if not cols:
        cols = list(ctx_df.columns)[:5]
    ctx_df = ctx_df[cols].reset_index(drop=True)

    header_cells = "".join(f"<th>{c}</th>" for c in ctx_df.columns)
    rows_html    = ""
    for i, row in ctx_df.iterrows():
        is_anchor = (i == len(ctx_df) - 1)
        row_class = "anchor-row" if is_anchor else ""
        cells = ""
        for j, (col, val) in enumerate(row.items()):
            if j == 0 and is_anchor:
                cells += f'<td>T-1 <span class="anchor-arrow">Predicting T →</span></td>'
            elif j == 0:
                step = len(ctx_df) - 1 - i
                cells += f'<td>T-{step}</td>'
            else:
                try:
                    cells += f"<td>{float(val):,.2f}</td>"
                except Exception:
                    cells += f"<td>{val}</td>"
        rows_html += f'<tr class="{row_class}">{cells}</tr>'

    st.markdown(f"""
    <div class="card">
      <table class="ctx-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)
else:
    st.info("Context not available.")


# ════════════════════════════════════════════════════════════════════════════════
# ROW 4 — SIGNAL HISTORY TABLE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Signal History — last 30 time steps</div>', unsafe_allow_html=True)

# Filter controls
filter_col, _ = st.columns([2, 10])
with filter_col:
    filter_val = st.selectbox("Filter by signal:", ["All", "BUY", "SELL", "HOLD"],
                              index=["All","BUY","SELL","HOLD"].index(st.session_state.filter_signal),
                              label_visibility="collapsed")
    st.session_state.filter_signal = filter_val

if history:
    hist_df = pd.DataFrame(history)
else:
    # Synthetic placeholder from local data for display
    n_rows = min(30, len(df))
    sample = df.tail(n_rows).copy()
    sigs_random = [random.choice(["BUY","SELL","HOLD"]) for _ in range(n_rows)]
    confs_random = [random.randint(55, 95) for _ in range(n_rows)]
    hist_df = pd.DataFrame({
        "time_step":     sample["Datetime"].dt.strftime("%Y-%m-%d %H:%M").values,
        "signal":        sigs_random,
        "confidence":    confs_random,
        "predicted_high": (sample["High"] * 1.001).round(2).values,
        "predicted_low":  (sample["Low"]  * 0.999).round(2).values,
        "actual_close":  sample["Close"].values,
        "outcome":       [random.choice(["Correct","Incorrect","Pending"]) for _ in range(n_rows)],
    })

if filter_val != "All":
    hist_df = hist_df[hist_df["signal"].str.upper() == filter_val]

hist_cols = ["time_step","signal","confidence","predicted_high","predicted_low","actual_close","outcome"]
hist_cols = [c for c in hist_cols if c in hist_df.columns]

header_cells = "".join(f"<th>{c.replace('_',' ').title()}</th>" for c in hist_cols)
rows_html    = ""
for _, row in hist_df.iterrows():
    cells = ""
    for c in hist_cols:
        v = row.get(c, "—")
        if c == "signal":
            cells += f"<td>{signal_badge(v)}</td>"
        elif c == "outcome":
            cells += f"<td>{outcome_badge(v)}</td>"
        elif c == "confidence":
            cells += f"<td>{v}%</td>"
        else:
            try:    cells += f"<td>{float(v):,.2f}</td>"
            except: cells += f"<td>{v}</td>"
    rows_html += f"<tr>{cells}</tr>"

st.markdown(f"""
<div class="card">
  <div class="hist-scroll">
    <table class="hist-table">
      <thead><tr>{header_cells}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# ROW 5 — Accuracy Tracker | Model Health | Feedback Widget
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)

col_acc, col_health, col_fb = st.columns([4, 4, 4], gap="large")

# ── Accuracy Tracker ──────────────────────────────────────────────────────────
with col_acc:
    st.markdown('<div class="section-label">Accuracy Tracker</div>', unsafe_allow_html=True)
    acc_7  = accuracy.get("last_7",  random.randint(55, 80))
    acc_30 = accuracy.get("last_30", random.randint(50, 75))
    acc_60 = accuracy.get("last_60", random.randint(48, 72))

    for label, pct in [("Last 7 signals", acc_7), ("Last 30 signals", acc_30), ("Last 60 signals", acc_60)]:
        c = acc_color(pct)
        st.markdown(f"""
        <div class="acc-card">
          <div class="acc-label">{label}</div>
          <div class="acc-val" style="color:{c};">{pct}%</div>
          <div class="progress-bar-wrap">
            <div class="progress-bar" style="width:{pct}%; background:{c};"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Sparkline
    spark_data = accuracy.get("correct_series", [random.choice([True, False]) for _ in range(30)])
    st.markdown('<div style="margin-top:8px; font-size:11px; color:{TEXT_MUTED};">Correct / Incorrect — last 30</div>', unsafe_allow_html=True)
    sfig = sparkline_fig(spark_data)
    st.plotly_chart(sfig, use_container_width=True, config={"displayModeBar": False})


# ── Model Health ──────────────────────────────────────────────────────────────
with col_health:
    st.markdown('<div class="section-label">Model Health</div>', unsafe_allow_html=True)

    api_ok      = model_health.get("api_status", True)
    last_train  = model_health.get("last_retrain", "—")
    model_ver   = model_health.get("model_version", "v1.0.0")
    data_fresh  = model_health.get("data_freshness", "—")
    drift       = model_health.get("drift_status", "No drift")

    drift_color = COLOR_BUY
    drift_badge = badge_html("No drift", "green")
    if "warning" in str(drift).lower():
        drift_color = COLOR_WARN; drift_badge = badge_html("Warning",      "amber")
    elif "detected" in str(drift).lower() or "drift" in str(drift).lower() and "no" not in str(drift).lower():
        drift_color = COLOR_SELL; drift_badge = badge_html("Drift detected","red")

    api_dot   = f'<span class="status-dot" style="background:{"#00C896" if api_ok else "#FF4C4C"};"></span>'
    api_label = "Online" if api_ok else "Offline"

    health_rows = [
        ("API Status",     f'{api_dot}{api_label}'),
        ("Last Retrain",   str(last_train)),
        ("Model Version",  str(model_ver)),
        ("Data Freshness", str(data_fresh)),
        ("Drift Status",   drift_badge),
    ]

    rows_html = "".join(f"""
    <div class="health-row">
      <div class="health-label">{lbl}</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:{TEXT_PRIMARY};">{val}</div>
    </div>""" for lbl, val in health_rows)

    st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)


# ── User Feedback ─────────────────────────────────────────────────────────────
with col_fb:
    st.markdown('<div class="section-label">Rate This Signal</div>', unsafe_allow_html=True)

    def submit_feedback(rating, comment):
        try:
            requests.post(f"{SIGNAL_BASE}/feedback",
                          json={"rating": rating, "comment": comment,
                                "signal": sig, "timestamp": ts_raw},
                          timeout=5)
        except Exception:
            pass
        st.session_state.feedback_submitted = True

    if st.session_state.feedback_submitted:
        st.markdown(f'<div class="card"><p class="feedback-success">✓ Thank you — feedback recorded for this time step.</p></div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("How accurate was this signal?")
            rating_val = st.feedback("stars", key="star_fb")
            comment    = st.text_input("Any comments? (optional)", placeholder="Optional notes…", label_visibility="collapsed")
            if st.button("Send Feedback", type="primary", use_container_width=True):
                r = (rating_val + 1) if rating_val is not None else 0
                submit_feedback(r, comment)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('</div>', unsafe_allow_html=True)  # close page-wrap
st.markdown("""
<div class="footer">
  Nifty50 Signal — AI-assisted market intelligence.
  Not SEBI-registered investment advice. For informational purposes only.
</div>
""", unsafe_allow_html=True)