import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

# Configuration
INITIAL_CAPITAL = 1000.0
RISK_PERCENT = 0.01
LEVERAGE = 200
FEE_RATE = 0.0005
AGENT_COUNT = 100


def get_agent_metadata(count):
    """Create a richer set of agent strategies with parameters.

    Archetypes:
      0: Oscillator
      1: Trend-following (MA crossover)
      2: Volatility-targeting (lower vol -> larger long)
      3: Mean-reversion (z-score)
      4: Momentum (rate-of-change)
      5: Vol breakout
    """
    archetypes = [0, 1, 2, 3, 4, 5]
    metas = []
    for i in range(count):
        arch = archetypes[i % len(archetypes)]
        # base period scaled by id for heterogeneity
        base_period = 10 + (i % 50)
        metas.append({
            "id": i,
            "name": f"Agent #{i+1}",
            "color": f"hsl({(i * 137.5) % 360},65%,45%)",
            "archetype": arch,
            "period": base_period,
            "short": max(3, int(base_period / 3)),
            "long": base_period * 2,
            "vol_target": 0.02 + ((i % 10) / 100.0),
            "z_thresh": 1.0 + ((i % 5) * 0.2),
        })
    return metas


def detect_delimiter(sample: str):
    if "\t" in sample:
        return "\t"
    # fall back to comma
    return ","


def load_prices_from_file(uploaded_file):
    raw = uploaded_file.getvalue().decode("utf-8")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    delim = detect_delimiter(lines[0])
    df = pd.read_csv(pd.io.common.StringIO("\n".join(lines[1:])), header=None, sep=delim)
    # Try to locate a close column; common positions: 1,4,5
    close_col = None
    for c in [1, 4, 5]:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        # pick last column
        close_col = df.columns[-1]
    df = df.rename(columns={0: "date", close_col: "close"})
    df = df[["date", "close"]]
    # ensure numeric
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def compute_signals_cpu(prices, agent_meta):
    n = len(prices)
    signals_for_agents = []
    for meta in agent_meta:
        signals = np.zeros(n, dtype=np.int8)
        period = int(meta["period"])
        arch = int(meta["archetype"])
        short = int(meta.get("short", max(3, period // 3)))
        long = int(meta.get("long", period * 2))
        vol_target = float(meta.get("vol_target", 0.02))
        z_thresh = float(meta.get("z_thresh", 1.0))

        # precompute rolling stats where needed
        for t in range(60, n):
            if arch == 0:  # Oscillator (momentum based)
                up = 0.0
                down = 0.0
                for j in range(1, period):
                    diff = prices[t - period + j] - prices[t - period + j - 1]
                    if diff > 0:
                        up += diff
                    else:
                        down -= diff
                factor = (up / (down + 1e-4)) * ((meta["id"] % 10) + 1)
                if factor > 5:
                    signals[t] = 1
                elif factor < 2:
                    signals[t] = 2
            elif arch == 1:  # Trend-following MA crossover
                if t - long <= 0:
                    continue
                short_ma = np.mean(prices[t - short + 1:t + 1])
                long_ma = np.mean(prices[t - long + 1:t + 1])
                if short_ma > long_ma * 1.001:
                    signals[t] = 1
                elif short_ma < long_ma * 0.999:
                    signals[t] = 2
            elif arch == 2:  # Volatility targeting
                win = max(5, period)
                if t - win <= 0:
                    continue
                ret = np.diff(prices[t - win + 1:t + 1]) / prices[t - win + 1:t]
                realized_vol = np.std(ret)
                if realized_vol < vol_target:
                    signals[t] = 1
                else:
                    signals[t] = 2
            elif arch == 3:  # Mean reversion (z-score to rolling mean)
                win = max(10, period)
                if t - win <= 0:
                    continue
                window = prices[t - win + 1:t + 1]
                mu = np.mean(window)
                sigma = np.std(window) + 1e-9
                z = (prices[t] - mu) / sigma
                if z < -z_thresh:
                    signals[t] = 1
                elif z > z_thresh:
                    signals[t] = 2
            elif arch == 4:  # Momentum (rate of change)
                look = max(3, period // 2)
                if t - look <= 0:
                    continue
                roc = (prices[t] - prices[t - look]) / prices[t - look]
                if roc > 0.01:
                    signals[t] = 1
                elif roc < -0.01:
                    signals[t] = 2
            elif arch == 5:  # Vol breakout
                win = max(10, period)
                if t - win <= 0:
                    continue
                highs = prices[t - win + 1:t + 1]
                prev_max = np.max(highs[:-1])
                if prices[t] > prev_max * 1.002:
                    signals[t] = 1
                else:
                    # small chance to go short when below moving median
                    med = np.median(highs)
                    if prices[t] < med * 0.998:
                        signals[t] = 2
        signals_for_agents.append(signals)
    return signals_for_agents


def run_backtest(prices, agent_meta, use_gpu=False):
    # prices: numpy array
    n = len(prices)
    signals_matrix = None
    if use_gpu and _HAS_CUPY:
        try:
            xp = cp
            prices_gpu = xp.array(prices)
            # For simplicity implement CPU signals even if GPU is available for this version
            # (GPU implementation could be added later). We'll mark that GPU is available.
            signals_matrix = compute_signals_cpu(prices, agent_meta)
        except Exception:
            signals_matrix = compute_signals_cpu(prices, agent_meta)
    else:
        signals_matrix = compute_signals_cpu(prices, agent_meta)

    results = []
    for idx, meta in enumerate(agent_meta):
        signals = signals_matrix[idx]
        balance = INITIAL_CAPITAL
        active_trade = None
        total_fees = 0.0
        history = []
        for i in range(n):
            p = float(prices[i])
            sig = "BUY" if signals[i] == 1 else ("SELL" if signals[i] == 2 else None)

            if active_trade:
                pnl = (p - active_trade["ep"]) * active_trade["u"] if active_trade["type"] == "BUY" else (active_trade["ep"] - p) * active_trade["u"]
                equity = active_trade["ie"] + pnl
                if equity <= 0:
                    balance = 0.0
                    active_trade = None
                elif (active_trade["type"] == "BUY" and sig == "SELL") or (active_trade["type"] == "SELL" and sig == "BUY"):
                    fee = (active_trade["u"] * p) * FEE_RATE
                    total_fees += fee
                    balance = equity - fee
                    active_trade = None

            if not active_trade and balance > 0 and sig:
                size = balance * RISK_PERCENT * LEVERAGE
                fee = size * FEE_RATE
                total_fees += fee
                balance -= fee
                if balance > 0:
                    active_trade = {"ep": p, "type": sig, "u": size / p, "ie": balance}

            cur_equity = 0.0
            if active_trade:
                pnl = (p - active_trade["ep"]) * active_trade["u"] if active_trade["type"] == "BUY" else (active_trade["ep"] - p) * active_trade["u"]
                cur_equity = max(0.0, active_trade["ie"] + pnl)
            else:
                cur_equity = balance

            history.append(cur_equity)
            if cur_equity <= 0:
                break

        final_val = history[-1] if history else 0.0
        roi = ((final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0
        results.append({
            **meta,
            "roi": roi,
            "equity_history": history,
            "total_fees_paid": total_fees,
            "final_value": final_val,
        })

    results = sorted(results, key=lambda r: r["roi"], reverse=True)
    return results


def main():
    st.set_page_config(page_title="Swarm GPU-X (Streamlit)", layout="wide")
    st.title("Swarm GPU-X — Streamlit Edition")

    st.sidebar.header("Data & Execution")
    uploaded = st.sidebar.file_uploader("Import CSV", type=["csv", "txt"])
    agent_count = st.sidebar.number_input("Agent Count", min_value=10, max_value=500, value=AGENT_COUNT, step=10)
    use_gpu = st.sidebar.checkbox("Prefer GPU (CuPy)", value=False)

    gpu_status = "Available" if _HAS_CUPY else "Unavailable"
    st.sidebar.markdown(f"**GPU (CuPy):** {gpu_status}")

    df = None
    if uploaded:
        df = load_prices_from_file(uploaded)

    if df is None:
        st.info("Upload a CSV with at least two columns: date and close. Common positions for close are 1,4,5.")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Execute Backtest"):
            if df is None:
                st.error("No dataset loaded")
            else:
                # Run backtest immediately in the same interaction to avoid
                # races where session_state persists but the uploaded file does not.
                with st.spinner("Running backtest — this may take a few seconds..."):
                    prices = df["close"].to_numpy(dtype=float)
                    agents = get_agent_metadata(agent_count)
                    results = run_backtest(prices, agents, use_gpu=use_gpu and _HAS_CUPY)
                    st.session_state["results"] = results
                    st.session_state["prices_df"] = df

    results = st.session_state.get("results")
    if results:
        df_prices = st.session_state.get("prices_df")
        n = len(df_prices)
        top = results[:10]

        # Chart: top equity curves
        import plotly.graph_objs as go
        fig = go.Figure()
        for r in top:
            # pad history to match price length
            hist = r["equity_history"]
            padded = hist + [hist[-1]] * (n - len(hist)) if len(hist) > 0 else [0] * n
            fig.add_trace(go.Scatter(x=df_prices["date"], y=padded, mode="lines", name=r["name"]))
        fig.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, width="stretch")

        # Rankings and metrics
        colA, colB = st.columns([1, 2])
        with colA:
            st.subheader("Swarm Ranking")
            rows = []
            for i, r in enumerate(results[:30]):
                rows.append({"#": i + 1, "name": r["name"], "roi": f"{r['roi']:.1f}%", "final": f"${r['final_value']:.2f}"})
            st.table(pd.DataFrame(rows))

        with colB:
            st.subheader("Processing Metrics")
            liq = len([r for r in results if r["final_value"] == 0])
            profitable = len([r for r in results if r["roi"] > 0])
            c1, c2 = st.columns(2)
            c1.metric("Liquidation Count", f"{liq}")
            c2.metric("Profitable Clusters", f"{profitable}")

        st.markdown("---")
        st.subheader("Top Clusters Details")
        for r in results[:5]:
            st.write(f"**{r['name']}** — ROI: {r['roi']:.2f}% — Final: ${r['final_value']:.2f} — Fees: ${r['total_fees_paid']:.2f}")


if __name__ == "__main__":
    main()
