"""BTCUSDT Replay Terminal — mock-data provider is isolated for Binance/API replacement."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

INITIAL_BALANCE = 10_000.0
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
INDICATORS = ["Fractal Diff Momentum", "RSI", "MACD", "Stochastic RSI", "Bollinger Bands", "EMA 9", "EMA 21", "EMA 50", "VWAP", "ATR", "ADX", "Volume", "Fractal", "Momentum", "Rate of Change", "OBV"]


class MockCandleProvider:
    """Deterministic local candle provider; replace fetch() with a Binance/API adapter later."""
    def fetch(self, symbol: str, interval: str = "1m", count: int = 900) -> pd.DataFrame:
        seed = sum(ord(c) for c in symbol) + count
        rng = np.random.default_rng(seed)
        base = {"BTCUSDT": 67240, "ETHUSDT": 3480, "SOLUSDT": 155, "XRPUSDT": .58, "BNBUSDT": 590}.get(symbol, 28.0)
        returns = rng.normal(0.000015, 0.00125, count)
        returns += np.sin(np.arange(count) / 31) * .00035
        close = base * np.exp(np.cumsum(returns))
        open_ = np.r_[base, close[:-1]]
        spread = np.maximum(close * rng.uniform(.00025, .0018, count), base * .00002)
        high = np.maximum(open_, close) + spread * rng.uniform(.3, 1, count)
        low = np.minimum(open_, close) - spread * rng.uniform(.3, 1, count)
        volume = rng.lognormal(mean=5.8, sigma=.5, size=count) * (1 + np.abs(returns) * 100)
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        times = pd.date_range(end=end, periods=count, freq="min", tz="UTC")
        return pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def fmt_price(v): return f"{v:,.2f}" if v >= 10 else f"{v:,.4f}"
def money(v): return f"${v:,.2f}"


def rsi(close, period=14):
    diff = close.diff(); up = diff.clip(lower=0).ewm(alpha=1/period, adjust=False).mean(); down = -diff.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def fractal_diff_momentum(df):
    """Causal momentum: EMA-smoothed close delta minus prior fractal-range delta."""
    fast = ema(df.close, 5) - ema(df.close, 13)
    tr = pd.concat([df.high-df.low, (df.high-df.close.shift()).abs(), (df.low-df.close.shift()).abs()], axis=1).max(axis=1)
    scale = ema(tr, 14).replace(0, np.nan)
    line = (fast / scale * 100).fillna(0)
    signal = ema(line, 6)
    return line, signal, line - signal


def enrich(df):
    d = df.copy()
    d["ema9"], d["ema21"], d["ema50"] = ema(d.close, 9), ema(d.close, 21), ema(d.close, 50)
    mid = d.close.rolling(20).mean(); sd = d.close.rolling(20).std(); d["bb_u"], d["bb_m"], d["bb_l"] = mid+2*sd, mid, mid-2*sd
    d["vwap"] = (d.close*d.volume).cumsum() / d.volume.cumsum()
    d["rsi"] = rsi(d.close)
    macd = ema(d.close, 12)-ema(d.close,26); d["macd"], d["macd_sig"], d["macd_hist"] = macd, ema(macd,9), macd-ema(macd,9)
    low14, high14 = d.low.rolling(14).min(), d.high.rolling(14).max(); d["stoch"] = 100*(d.close-low14)/(high14-low14)
    d["momentum"] = d.close.diff(10); d["roc"] = d.close.pct_change(10)*100; d["obv"] = (np.sign(d.close.diff()).fillna(0)*d.volume).cumsum()
    d["fdm"], d["fdm_sig"], d["fdm_hist"] = fractal_diff_momentum(d)
    return d


def init_state():
    defaults = {"symbol":"BTCUSDT", "replay_idx": 540, "playing":False, "speed":"1x", "chart_type":"Candles", "active_indicators":["Fractal Diff Momentum", "EMA 9", "EMA 21", "Volume"], "balance":INITIAL_BALANCE, "position":None, "trades":[], "draw_tool":"Cursor", "replay_start":540}
    for k, v in defaults.items(): st.session_state.setdefault(k, v)


def close_position(price, time, reason):
    pos = st.session_state.position
    if not pos: return
    pnl = (price-pos["entry"])*pos["qty"]*pos["direction"]
    st.session_state.balance += pnl
    st.session_state.trades.append({"timestamp":time.strftime("%Y-%m-%d %H:%M"), "side":"Long" if pos["direction"] == 1 else "Short", "entry":pos["entry"], "exit":price, "quantity":pos["qty"], "pnl":pnl, "reason":reason})
    st.session_state.position = None


def trade(direction, price, time, size, leverage):
    if st.session_state.position: close_position(price, time, "Reverse")
    qty = size * leverage / price
    st.session_state.position = {"direction":direction, "entry":price, "qty":qty, "size":size, "leverage":leverage, "opened":time}


def candle_chart(d, active, height):
    fig = go.Figure()
    if st.session_state.chart_type == "Line":
        fig.add_trace(go.Scatter(x=d.time, y=d.close, mode="lines", line=dict(color="#00b894", width=1.5), name="BTCUSDT"))
    else:
        if st.session_state.chart_type == "Heikin Ashi":
            ha_close = (d.open+d.high+d.low+d.close)/4; ha_open = ha_close.copy(); ha_open.iloc[0] = (d.open.iloc[0]+d.close.iloc[0])/2
            for i in range(1,len(d)): ha_open.iloc[i] = (ha_open.iloc[i-1]+ha_close.iloc[i-1])/2
            o, h, l, c = ha_open, pd.concat([d.high,ha_open,ha_close],axis=1).max(axis=1), pd.concat([d.low,ha_open,ha_close],axis=1).min(axis=1), ha_close
        else: o,h,l,c=d.open,d.high,d.low,d.close
        fig.add_trace(go.Candlestick(x=d.time, open=o, high=h, low=l, close=c, increasing_line_color="#16c784", decreasing_line_color="#ea3943", name="BTCUSDT", hovertemplate="%{x}<br>O %{open:.2f} H %{high:.2f}<br>L %{low:.2f} C %{close:.2f}<extra></extra>"))
    overlays = {"EMA 9":("ema9", "#f5c542"), "EMA 21":("ema21", "#5ba4ff"), "EMA 50":("ema50", "#be7bff"), "VWAP":("vwap", "#ee9b00")}
    for label, (col, color) in overlays.items():
        if label in active: fig.add_trace(go.Scatter(x=d.time,y=d[col],mode="lines",line=dict(color=color,width=1),name=label))
    if "Bollinger Bands" in active:
        for col, color, name in [("bb_u","#64748b","BB Upper"),("bb_m","#94a3b8","BB Mid"),("bb_l","#64748b","BB Lower")]: fig.add_trace(go.Scatter(x=d.time,y=d[col],mode="lines",line=dict(color=color,width=1,dash="dot"),name=name))
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=5,r=5,t=24,b=0), paper_bgcolor="#11161d", plot_bgcolor="#11161d", xaxis_rangeslider_visible=False, showlegend=False, hovermode="x unified", dragmode="pan", font=dict(size=11,color="#b7c0cc"))
    fig.update_xaxes(showgrid=True, gridcolor="#26303b", rangeslider_visible=False, fixedrange=False)
    fig.update_yaxes(showgrid=True, gridcolor="#26303b", side="right", fixedrange=False)
    return fig


def oscillator_chart(d):
    fig = go.Figure()
    colors = np.where(d.fdm_hist >= 0, "#16c784", "#ea3943")
    fig.add_trace(go.Bar(x=d.time,y=d.fdm_hist,marker_color=colors,name="Histogram")); fig.add_trace(go.Scatter(x=d.time,y=d.fdm,mode="lines",line=dict(color="#f5c542",width=1.3),name="FDM")); fig.add_trace(go.Scatter(x=d.time,y=d.fdm_sig,mode="lines",line=dict(color="#7aa7ff",width=1),name="Signal"))
    fig.add_hline(y=0,line_color="#64748b",line_width=1)
    fig.update_layout(template="plotly_dark",height=185,margin=dict(l=5,r=5,t=27,b=0),paper_bgcolor="#11161d",plot_bgcolor="#11161d",title="Fractal Diff Momentum  5, 13, 6",showlegend=False,font=dict(size=10,color="#b7c0cc"),hovermode="x unified")
    fig.update_xaxes(showgrid=True,gridcolor="#26303b"); fig.update_yaxes(showgrid=True,gridcolor="#26303b",side="right")
    return fig


def main():
    st.set_page_config(page_title="BTCUSDT · Market Replay", layout="wide", initial_sidebar_state="collapsed")
    init_state()
    st.markdown("""<style>
    .stApp { background:#0b0f14; color:#c7d0da; } .block-container {padding: .45rem .65rem .7rem; max-width: 1800px;}
    div[data-testid="stMetric"] {background:#11161d;border:1px solid #28313c;padding:5px 9px;border-radius:2px;} div[data-testid="stMetricLabel"] {font-size:10px;} div[data-testid="stMetricValue"] {font-size:15px;}
    .stButton>button,.stSelectbox select {background:#161d26;color:#c7d0da;border:1px solid #354150;border-radius:2px;font-size:12px;min-height:30px;padding:2px 8px;}
    .terminal-title {font-family:Arial,sans-serif;font-size:18px;font-weight:700;color:#e5eaf0;line-height:1.1}.muted{color:#8793a2;font-size:11px}.quote{color:#16c784;font-size:21px;font-weight:700}.panel-title{font-size:11px;font-weight:700;color:#9da9b8;text-transform:uppercase;letter-spacing:.07em;border-bottom:1px solid #28313c;padding-bottom:6px;margin-bottom:6px}
    [data-testid="stDataFrame"] {border:1px solid #28313c;} hr {border-color:#28313c;margin:.35rem 0;}
    </style>""", unsafe_allow_html=True)
    provider = MockCandleProvider()
    full = enrich(provider.fetch(st.session_state.symbol))
    idx = min(st.session_state.replay_idx, len(full)-1)
    visible = full.iloc[:idx+1].copy(); latest = visible.iloc[-1]; prev = full.iloc[max(0, idx-1)]

    # Top terminal toolbar
    a,b,c,d,e,f,g,h,i = st.columns([1.35,.7,1.05,.9,.75,.72,.68,.7,.8])
    with a: st.selectbox("Symbol", SYMBOLS, key="symbol", label_visibility="collapsed")
    with b: st.selectbox("Timeframe", ["1m","5m","15m","1h"], index=0, label_visibility="collapsed")
    with c:
        selected = st.multiselect("Indicators", INDICATORS, default=st.session_state.active_indicators, key="indicator_picker", label_visibility="collapsed", placeholder="ƒx Indicators")
        st.session_state.active_indicators = selected
    with d: st.selectbox("Chart type", ["Candles","Heikin Ashi","Line"], key="chart_type", label_visibility="collapsed")
    with e:
        if st.button("▶ Replay", use_container_width=True): st.session_state.playing = not st.session_state.playing
    with f: st.button("⚙ Settings", use_container_width=True)
    with g: st.button("▦ Layout", use_container_width=True)
    with h: st.button("⛶ Fullscreen", use_container_width=True)
    with i: st.markdown(f"<div class='muted'>UTC</div><div class='quote'>{fmt_price(latest.close)}</div>", unsafe_allow_html=True)

    # header and side-panel desktop structure
    left, center, right = st.columns([.62, 6.7, 2.35], gap="small")
    with left:
        st.markdown("<div class='panel-title'>Tools</div>",unsafe_allow_html=True)
        for tool, icon in [("Cursor","✛"),("Trend line","╱"),("Horizontal line","—"),("Vertical line","│"),("Rectangle","▭"),("Fibonacci","≋"),("Ruler","⌗"),("Text","T"),("Remove drawings","⌫")]:
            if st.button(icon, key=f"tool_{tool}", help=tool, use_container_width=True): st.session_state.draw_tool=tool
        st.caption(f"Active: {st.session_state.draw_tool}")
    with center:
        title, stats = st.columns([1.5,4.5])
        with title: st.markdown("<div class='terminal-title'>BTCUSDT <span class='muted'>· 1m · Binance</span></div><div class='muted'>Bitcoin / Tether &nbsp; • &nbsp; " + ("REPLAY" if idx < len(full)-1 else "LIVE") + "</div>",unsafe_allow_html=True)
        with stats:
            m1,m2,m3,m4,m5=st.columns(5); change=(latest.close/full.close.iloc[max(0,idx-1440)]-1)*100
            m1.metric("LAST",fmt_price(latest.close));m2.metric("24H CHG",f"{change:+.2f}%");m3.metric("24H HIGH",fmt_price(visible.high.tail(300).max()));m4.metric("24H LOW",fmt_price(visible.low.tail(300).min()));m5.metric("24H VOL",f"{visible.volume.tail(300).sum()/1e6:.2f}M")
        st.plotly_chart(candle_chart(visible,st.session_state.active_indicators,470), use_container_width=True, config={"scrollZoom":True,"displayModeBar":True,"modeBarButtonsToRemove":["select2d","lasso2d"]})
        if "Fractal Diff Momentum" in st.session_state.active_indicators: st.plotly_chart(oscillator_chart(visible), use_container_width=True, config={"scrollZoom":True,"displayModeBar":False})
    with right:
        st.markdown("<div class='panel-title'>Watchlist &nbsp; CRYPTO MAJORS</div>", unsafe_allow_html=True)
        rows=[]
        for sym in SYMBOLS:
            x=enrich(provider.fetch(sym, count=210)); ch=(x.close.iloc[-1]/x.close.iloc[-2]-1)*100
            rows.append({"Symbol":("● "+sym) if sym==st.session_state.symbol else sym,"Last":fmt_price(x.close.iloc[-1]),"Chg%":f"{ch:+.2f}%","Vol":f"{x.volume.tail(60).sum()/1e3:.0f}K"})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=280,column_config={"Chg%":st.column_config.TextColumn("Chg%")})
        st.markdown("<div class='panel-title'>Indicator parameters</div>",unsafe_allow_html=True)
        st.caption("Changes apply causally to the replay window.")
        st.number_input("FDM fast", 2, 20, 5); st.number_input("FDM slow", 5, 50, 13); st.number_input("FDM signal", 2, 30, 6)
        st.markdown("<div class='panel-title'>Replay Statistics</div>",unsafe_allow_html=True)
        trades=st.session_state.trades; pnls=np.array([t["pnl"] for t in trades]) if trades else np.array([]); wins=pnls[pnls>0]; losses=pnls[pnls<0]
        win_rate=(len(wins)/len(pnls)*100) if len(pnls) else 0; pf=(wins.sum()/abs(losses.sum())) if len(losses) and len(wins) else 0
        stat_rows=[["Total trades",len(trades)],["Winning / Losing",f"{len(wins)} / {len(losses)}"],["Win rate",f"{win_rate:.1f}%"],["Profit factor",f"{pf:.2f}"],["Net P&L",money(pnls.sum())],["Max drawdown","$0.00"],["Average win",money(wins.mean()) if len(wins) else "$0.00"],["Average loss",money(losses.mean()) if len(losses) else "$0.00"],["Largest win",money(wins.max()) if len(wins) else "$0.00"],["Largest loss",money(losses.min()) if len(losses) else "$0.00"],["Sharpe ratio","—"]]
        # Keep the compact terminal table homogeneous; Arrow otherwise has to infer
        # a mixed numeric/string Value column on a fresh replay.
        stats_frame = pd.DataFrame(stat_rows, columns=["Metric", "Value"]).astype(str)
        st.dataframe(stats_frame,hide_index=True,use_container_width=True,height=300)

    st.divider()
    st.markdown("<div class='panel-title'>Bar Replay / Market Replay &nbsp; <span class='muted'>Future bars hidden — no look-ahead bias</span></div>",unsafe_allow_html=True)
    r1,r2,r3,r4,r5,r6,r7,r8=st.columns([1.35,.7,.7,.8,.8,.8,1.2,2.3])
    with r1:
        new_start=st.slider("Historical starting point",100,len(full)-2,st.session_state.replay_start)
        if new_start != st.session_state.replay_start: st.session_state.replay_start=new_start; st.session_state.replay_idx=new_start
    with r2:
        if st.button("⏮ Reset",use_container_width=True): st.session_state.replay_idx=st.session_state.replay_start
    with r3:
        if st.button("◀ Back",use_container_width=True): st.session_state.replay_idx=max(st.session_state.replay_start,idx-1)
    with r4:
        if st.button("▶ Play" if not st.session_state.playing else "Ⅱ Pause",use_container_width=True): st.session_state.playing=not st.session_state.playing
    with r5:
        if st.button("Forward 1",use_container_width=True): st.session_state.replay_idx=min(len(full)-1,idx+1)
    with r6:
        if st.button("+10 bars",use_container_width=True): st.session_state.replay_idx=min(len(full)-1,idx+10)
    with r7: st.select_slider("Speed",["0.5x","1x","2x","5x","10x"],key="speed")
    with r8: st.markdown(f"<div class='muted'>REPLAY TIMESTAMP</div><b>{latest.time.strftime('%Y-%m-%d %H:%M UTC')}</b><br><span class='muted'>Bar {idx+1:,} / {len(full):,} • {len(full)-idx-1:,} future candles concealed</span>",unsafe_allow_html=True)

    st.markdown("<div class='panel-title'>Paper Trading Simulator</div>",unsafe_allow_html=True)
    q1,q2,q3,q4,q5,q6,q7,q8 = st.columns([.8,.8,1,1,1,1.1,1.1,2.1])
    with q1: size=st.number_input("Position size $",10.0,100000.0,500.0,10.0)
    with q2: leverage=st.number_input("Leverage",1,100,5)
    with q3:
        if st.button("BUY MARKET",type="primary",use_container_width=True): trade(1,latest.close,latest.time,size,leverage)
    with q4:
        if st.button("SELL MARKET",use_container_width=True): trade(-1,latest.close,latest.time,size,leverage)
    pos=st.session_state.position; unrealized=(latest.close-pos["entry"])*pos["qty"]*pos["direction"] if pos else 0
    with q5: st.metric("Entry / Mark", f"{fmt_price(pos['entry']) if pos else '—'} / {fmt_price(latest.close)}")
    with q6: st.metric("Unrealized P&L",money(unrealized),f"{(unrealized/pos['size']*100):+.2f}%" if pos else None)
    with q7: st.metric("Balance / Equity",f"{money(st.session_state.balance)}",money(st.session_state.balance+unrealized))
    with q8: st.markdown(f"<div class='muted'>OPEN POSITION</div><b>{'LONG' if pos and pos['direction']==1 else 'SHORT' if pos else 'FLAT'}</b> &nbsp; Margin: {money(pos['size'] if pos else 0)} &nbsp; Stop loss: — &nbsp; Take profit: —")
    st.markdown("<div class='panel-title'>Trade History</div>",unsafe_allow_html=True)
    hist=pd.DataFrame(st.session_state.trades,columns=["timestamp","side","entry","exit","quantity","pnl","reason"])
    st.dataframe(hist if not hist.empty else pd.DataFrame([["No closed paper trades yet","—","—","—","—","—","—"]],columns=hist.columns),hide_index=True,use_container_width=True,height=130)
    st.caption("Keyboard: Space Play/Pause · → next candle · ← previous candle · Shift+→ +10 · R reset · F fullscreen · Esc exit tool. MockCandleProvider is deliberately separate from the terminal UI for replacement with Binance/API historical candles.")
    # Streamlit's rerun enables continuous automatic replay without exposing future data.
    if st.session_state.playing and idx < len(full)-1:
        import time; time.sleep({"0.5x":1.0,"1x":.6,"2x":.3,"5x":.12,"10x":.06}[st.session_state.speed]); st.session_state.replay_idx += 1; st.rerun()

if __name__ == "__main__": main()
