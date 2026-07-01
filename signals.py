"""
S&P 500 market signal analyzer using SPY ETF data from Yahoo Finance.

Calculates RSI, MACD, Bollinger Bands, MA trend, and volume to produce
a composite BUY / HOLD / SELL recommendation with confidence score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("signals")

TICKER = "SPY"   # S&P 500 ETF — liquid, accurate


@dataclass
class Signal:
    direction: str          # "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
    confidence: float       # 0.0 – 1.0
    score: float            # raw composite score 0–100 (higher = more bullish)
    price: float
    change_1d: float        # % change today
    change_5d: float        # % change last 5 days
    rsi: float
    macd_signal: str        # "bullish cross" | "bearish cross" | "neutral"
    above_50ma: bool
    above_200ma: bool
    volume_ratio: float     # current vol / 20-day avg
    near_bb_lower: bool     # price near lower Bollinger Band
    is_alert_worthy: bool   # True if a strong buy or sell opportunity is detected
    alert_type: str         # "buy" | "sell" | "none"
    alert_reasons: list[str]
    fetched_at: datetime


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, lower


def analyze() -> Signal:
    """Download SPY data and compute the composite signal. Takes ~2-3 seconds."""
    logger.info("Fetching SPY data...")
    df = yf.download(TICKER, period="6mo", interval="1d", auto_adjust=True, progress=False)

    if df is None or len(df) < 30:
        raise RuntimeError("Not enough market data returned from Yahoo Finance")

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    # ── Indicators ────────────────────────────────────────────────────────────
    rsi_series = _rsi(close)
    rsi = float(rsi_series.iloc[-1])

    macd_line, signal_line, histogram = _macd(close)
    macd_prev_diff = float(macd_line.iloc[-2] - signal_line.iloc[-2])
    macd_curr_diff = float(macd_line.iloc[-1] - signal_line.iloc[-1])
    if macd_prev_diff < 0 and macd_curr_diff >= 0:
        macd_signal = "bullish cross"
    elif macd_prev_diff > 0 and macd_curr_diff <= 0:
        macd_signal = "bearish cross"
    else:
        macd_signal = "above" if macd_curr_diff > 0 else "below"

    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    price = float(close.iloc[-1])
    above_50ma  = price > ma50
    above_200ma = price > ma200

    vol_avg20 = float(volume.rolling(20).mean().iloc[-1])
    vol_today = float(volume.iloc[-1])
    volume_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 1.0

    bb_upper, bb_lower = _bollinger(close)
    bb_lower_val = float(bb_lower.iloc[-1])
    bb_upper_val = float(bb_upper.iloc[-1])
    near_bb_lower = price <= bb_lower_val * 1.01   # within 1% of lower band

    change_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    change_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0

    # ── Composite score (0=bearish, 100=bullish) ──────────────────────────────
    # RSI component (30 pts): <30=30, 30-45=22, 45-55=15, 55-70=8, >70=0
    if rsi < 30:
        rsi_score = 30.0
    elif rsi < 45:
        rsi_score = 22.0
    elif rsi < 55:
        rsi_score = 15.0
    elif rsi < 70:
        rsi_score = 8.0
    else:
        rsi_score = 0.0

    # MACD component (20 pts)
    if macd_signal == "bullish cross":
        macd_score = 20.0
    elif macd_signal == "above":
        macd_score = 13.0
    elif macd_signal == "bearish cross":
        macd_score = 0.0
    else:
        macd_score = 5.0

    # MA trend (25 pts)
    if above_50ma and above_200ma:
        ma_score = 25.0
    elif above_50ma:
        ma_score = 15.0
    elif above_200ma:
        ma_score = 10.0
    else:
        ma_score = 0.0

    # Bollinger (15 pts): near lower band is bullish (oversold)
    bb_pct = (price - bb_lower_val) / (bb_upper_val - bb_lower_val) if bb_upper_val != bb_lower_val else 0.5
    bb_score = max(0.0, 15.0 * (1 - bb_pct))   # 15 when at lower band, 0 when at upper

    # Volume (10 pts): high volume on dips = accumulation
    if volume_ratio > 1.5 and change_1d < 0:
        vol_score = 10.0   # heavy selling volume — contrarian buy signal
    elif volume_ratio > 1.5 and change_1d > 0:
        vol_score = 8.0    # strong breakout volume
    elif volume_ratio > 1.0:
        vol_score = 5.0
    else:
        vol_score = 2.0

    # Momentum bonus (up to 8 pts): rising price above 50MA shouldn't land in SELL
    if above_50ma and change_1d > 0.5:
        momentum_score = 8.0
    elif above_50ma and change_1d > 0.0:
        momentum_score = 4.0
    else:
        momentum_score = 0.0

    score = rsi_score + macd_score + ma_score + bb_score + vol_score + momentum_score

    # ── Direction + confidence ────────────────────────────────────────────────
    if score >= 75:
        direction = "STRONG BUY"
        confidence = min(0.95, 0.70 + (score - 75) / 100)
    elif score >= 58:
        direction = "BUY"
        confidence = 0.55 + (score - 58) / 120
    elif score >= 42:
        direction = "HOLD"
        confidence = 0.50
    elif score >= 25:
        direction = "SELL"
        confidence = 0.55 + (42 - score) / 120
    else:
        direction = "STRONG SELL"
        confidence = min(0.95, 0.70 + (25 - score) / 100)

    # ── Big-opportunity alert detection ──────────────────────────────────────
    alert_reasons: list[str] = []
    alert_type: str = "none"   # "buy" | "sell" | "none"

    # --- Buy signals ---
    if rsi < 30:
        alert_reasons.append(f"RSI deeply oversold ({rsi:.1f})")
    elif rsi < 35:
        alert_reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi < 45:
        alert_reasons.append(f"RSI cooling off ({rsi:.1f}) — potential entry")

    if near_bb_lower:
        alert_reasons.append("Price at lower Bollinger Band support")

    if macd_signal == "bullish cross":
        alert_reasons.append("MACD bullish crossover confirmed")
    elif macd_signal == "above":
        alert_reasons.append("MACD holding bullish")

    if change_1d < -1.0:
        alert_reasons.append(f"S&P 500 down {change_1d:.1f}% today (potential dip buy)")

    if change_5d < -3.0:
        alert_reasons.append(f"S&P 500 down {change_5d:.1f}% this week")

    if not above_50ma and close.iloc[-1] > close.iloc[-2]:
        alert_reasons.append("Bouncing from below 50-day MA")

    if above_50ma and above_200ma and rsi < 55:
        alert_reasons.append("Strong uptrend with RSI room to run")

    # Fire on score >= 58 with at least 1 reason (was 70 + 2 reasons)
    if score >= 58 and len(alert_reasons) >= 1:
        alert_type = "buy"

    # --- Sell signals (checked separately so reasons are independent) ---
    sell_reasons: list[str] = []

    if rsi > 75:
        sell_reasons.append(f"RSI deeply overbought ({rsi:.1f})")
    elif rsi > 70:
        sell_reasons.append(f"RSI overbought ({rsi:.1f})")
    elif rsi > 65:
        sell_reasons.append(f"RSI elevated ({rsi:.1f}) — watch for reversal")

    near_bb_upper = price >= bb_upper_val * 0.99
    if near_bb_upper:
        sell_reasons.append("Price at upper Bollinger Band resistance")

    # Only flag MACD as bearish if price isn't clearly rising today
    if macd_signal == "bearish cross" and change_1d <= 0.5:
        sell_reasons.append("MACD bearish crossover confirmed")
    elif macd_signal == "below" and change_1d <= 0.5:
        sell_reasons.append("MACD turned bearish")

    if change_1d > 1.5:
        sell_reasons.append(f"S&P 500 up {change_1d:.1f}% today (extended move)")

    if change_5d > 4.0:
        sell_reasons.append(f"S&P 500 up {change_5d:.1f}% this week (extended)")

    # Require 2+ sell reasons to avoid false signals from a single indicator
    if score <= 42 and len(sell_reasons) >= 2:
        alert_type = "sell"
        alert_reasons = sell_reasons   # replace buy reasons with sell reasons

    is_alert_worthy = alert_type in ("buy", "sell") and confidence >= 0.70

    return Signal(
        direction=direction,
        confidence=confidence,
        score=score,
        price=price,
        change_1d=change_1d,
        change_5d=change_5d,
        rsi=rsi,
        macd_signal=macd_signal,
        above_50ma=above_50ma,
        above_200ma=above_200ma,
        volume_ratio=volume_ratio,
        near_bb_lower=near_bb_lower,
        is_alert_worthy=is_alert_worthy,
        alert_type=alert_type,
        alert_reasons=alert_reasons,
        fetched_at=datetime.utcnow(),
    )


def format_signal_message(sig: Signal, header: str = "📊 S&P 500 Analysis") -> str:
    # Weak BUY/SELL signals below alert threshold display as HOLD to avoid acting on noise
    ALERT_THRESHOLD = 0.70
    display_direction = sig.direction
    if sig.direction in ("SELL", "BUY") and sig.confidence < ALERT_THRESHOLD:
        display_direction = f"HOLD (leaning {sig.direction})"

    emoji = {
        "STRONG BUY":  "🟢",
        "BUY":         "📈",
        "HOLD":        "⚖️",
        "SELL":        "📉",
        "STRONG SELL": "🔴",
    }.get(sig.direction, "❓")
    if sig.direction in ("SELL", "BUY") and sig.confidence < ALERT_THRESHOLD:
        emoji = "⚖️"

    change_1d_str = f"+{sig.change_1d:.2f}%" if sig.change_1d >= 0 else f"{sig.change_1d:.2f}%"
    change_5d_str = f"+{sig.change_5d:.2f}%" if sig.change_5d >= 0 else f"{sig.change_5d:.2f}%"

    lines = [
        f"*{header}*",
        f"SPY — ${sig.price:.2f}  {change_1d_str} today · {change_5d_str} 5d",
        "",
        f"{emoji} *{display_direction}*  ({sig.confidence*100:.0f}% confidence)",
        "",
        "*Technical indicators:*",
        f"• RSI(14): {sig.rsi:.1f}{'  ⚠️ oversold' if sig.rsi < 35 else '  ⚠️ overbought' if sig.rsi > 70 else ''}",
        f"• MACD: {sig.macd_signal}",
        f"• Trend: {'✅ above' if sig.above_50ma else '❌ below'} 50MA · {'✅ above' if sig.above_200ma else '❌ below'} 200MA",
        f"• Volume: {sig.volume_ratio:.1f}x avg{'  🔥 spike' if sig.volume_ratio > 1.5 else ''}",
        f"• Bollinger: {'📍 at lower band' if sig.near_bb_lower else 'normal range'}",
        "",
        f"_Updated {sig.fetched_at.strftime('%H:%M UTC')}_",
    ]

    if sig.alert_reasons:
        lines.insert(-1, "")
        lines.insert(-1, "⚡ *Why this signal:*")
        for r in sig.alert_reasons:
            lines.insert(-1, f"  — {r}")

    return "\n".join(lines)


def format_alert_message(sig: Signal) -> str:
    if sig.alert_type == "sell":
        header = "🔴 Strong Sell Signal" if sig.direction == "STRONG SELL" else "📉 Sell Signal Detected"
    else:
        header = "🟢 Strong Buy Opportunity" if sig.direction == "STRONG BUY" else "📈 Buy Signal Detected"
    return format_signal_message(sig, header=header)
