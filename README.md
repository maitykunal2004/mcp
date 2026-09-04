# BTCUSDT Market Replay Terminal

A dense Streamlit trading workstation for manually replaying deterministic BTCUSDT 1-minute OHLCV data without look-ahead bias. It uses a dark, classic-terminal layout with chart controls, watchlist, causal indicators, paper trading, replay controls, and trading statistics.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Included

- BTCUSDT 1-minute candlestick, Heikin Ashi, and line charts with Plotly pan, zoom, crosshair hover, and OHLC tooltips.
- Bar replay with a historical start point, reset, back, next candle, advance 10 bars, five speed choices, and automatic progressive reveal.
- Causal Fractal Diff Momentum oscillator plus configurable indicator selection for EMAs, VWAP, Bollinger Bands, volume, RSI, MACD, Stochastic RSI, ATR, ADX, fractals, momentum, ROC, and OBV.
- Watchlist and a paper-trading ticket that supports long/short entries during a replay, open-position mark-to-market, closed-trade history, and replay statistics.
- A clearly isolated `MockCandleProvider`. Replace its `fetch()` implementation with a Binance/API historical-candle adapter when production data is available.

The app intentionally calculates the displayed price data and oscillator only from candles visible at the current replay position; future candles are not rendered or used by the paper-trading interaction.
