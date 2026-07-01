"""
S&P 500 professional multi-timeframe signal analyzer.

Timeframes analyzed:
  - 15-min  : intraday entries, candlestick patterns, VWAP
  - 1-hour  : intermediate trend, hourly RSI/MACD
  - Daily   : macro trend, MA50/200, trend structure (HH/HL vs LH/LL)

Event awareness:
  - FOMC meeting dates (2025-2026 schedule hardcoded)
  - News headline scanning for CPI/NFP/Fed keywords
  - High-impact events flag as "Event Risk" and cap confidence at 55%
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("signals")

TICKER = "SPY"

# ── FOMC rate-decision dates 2025-2026 ────────────────────────────────────────
FOMC_DATES: list[date] = [
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 10, 29), date(2025, 12, 10),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
]

# Keywords that indicate a market-moving event in headlines
EVENT_KEYWORDS = [
    "fomc", "fed rate", "federal reserve", "rate decision",
    "jerome powell", "cpi", "inflation data", "jobs report",
    "nonfarm payroll", "non-farm", "gdp report", "ppi",
    "tariff", "trade war", "recession",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandlePattern:
    name: str
    bias: str       # "bullish" | "bearish" | "neutral"
    strength: str   # "strong" | "moderate"
    timeframe: str  # "15min" | "1hr"


@dataclass
class EventRisk:
    name: str
    days_away: int
    impact: str         # "high" | "medium"
    description: str = ""


@dataclass
class Signal:
    direction: str      # "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
    confidence: float
    score: float
    price: float
    vwap: float
    above_vwap: bool
    change_1d: float
    change_5d: float

    # Macro (daily)
    rsi_daily: float
    macd_signal_daily: str
    above_50ma: bool
    above_200ma: bool
    trend_structure: str    # "uptrend" | "downtrend" | "ranging"
    near_bb_lower: bool
    near_bb_upper: bool

    # Intermediate (1hr)
    rsi_hourly: float
    macd_signal_hourly: str

    # Intraday (15min)
    rsi_15min: float
    volume_ratio: float

    # Key levels
    nearest_support: float
    nearest_resistance: float

    # Patterns detected across timeframes
    patterns: list[CandlePattern]

    # Economic events
    event_risks: list[EventRisk]
    event_override: bool    # True when high-impact event is within 1 day

    # Alert
    is_alert_worthy: bool
    alert_type: str         # "buy" | "sell" | "event_warning" | "none"
    alert_reasons: list[str]

    fetched_at: datetime


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def _macd_label(line: pd.Series, signal: pd.Series) -> str:
    prev = float(line.iloc[-2] - signal.iloc[-2])
    curr = float(line.iloc[-1] - signal.iloc[-1])
    if prev < 0 and curr >= 0:
        return "bullish cross"
    if prev > 0 and curr <= 0:
        return "bearish cross"
    return "above" if curr > 0 else "below"


def _bollinger(series: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma + 2 * std, ma - 2 * std


def _vwap(df: pd.DataFrame) -> float:
    """Volume-weighted average price across all bars in df."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]
    total_vol = float(vol.sum())
    return float((typical * vol).sum() / total_vol) if total_vol > 0 else float(df["Close"].iloc[-1])


def _trend_structure(close: pd.Series, lookback: int = 20) -> str:
    """Identify macro trend by comparing four quarter-averages of recent closes."""
    if len(close) < lookback:
        return "ranging"
    window = close.iloc[-lookback:]
    q = max(1, len(window) // 4)
    q1 = float(window.iloc[:q].mean())
    q2 = float(window.iloc[q:2*q].mean())
    q3 = float(window.iloc[2*q:3*q].mean())
    q4 = float(window.iloc[3*q:].mean())
    if q1 < q2 < q3 < q4:
        return "uptrend"
    if q1 > q2 > q3 > q4:
        return "downtrend"
    return "ranging"


def _swing_levels(
    high: pd.Series, low: pd.Series, close: float, lookback: int = 30
) -> tuple[float, float]:
    """Find nearest swing support (below price) and resistance (above price)."""
    local_highs: list[float] = []
    local_lows: list[float] = []
    h = high.iloc[-lookback:]
    l = low.iloc[-lookback:]
    for i in range(2, len(h) - 2):
        hv = float(h.iloc[i])
        lv = float(l.iloc[i])
        if hv >= float(h.iloc[i-1]) and hv >= float(h.iloc[i-2]) and hv >= float(h.iloc[i+1]) and hv >= float(h.iloc[i+2]):
            local_highs.append(hv)
        if lv <= float(l.iloc[i-1]) and lv <= float(l.iloc[i-2]) and lv <= float(l.iloc[i+1]) and lv <= float(l.iloc[i+2]):
            local_lows.append(lv)

    resistances = [x for x in local_highs if x > close * 1.001]
    supports    = [x for x in local_lows  if x < close * 0.999]

    nearest_resistance = min(resistances) if resistances else close * 1.02
    nearest_support    = max(supports)    if supports    else close * 0.98
    return nearest_support, nearest_resistance


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick pattern detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_patterns(df: pd.DataFrame, timeframe: str) -> list[CandlePattern]:
    """Detect common reversal and continuation patterns in the last 3 candles."""
    if len(df) < 3:
        return []

    patterns: list[CandlePattern] = []

    o0 = float(df["Open"].iloc[-1]);  h0 = float(df["High"].iloc[-1])
    l0 = float(df["Low"].iloc[-1]);   c0 = float(df["Close"].iloc[-1])
    o1 = float(df["Open"].iloc[-2]);  h1 = float(df["High"].iloc[-2])
    l1 = float(df["Low"].iloc[-2]);   c1 = float(df["Close"].iloc[-2])

    body0   = abs(c0 - o0)
    range0  = max(h0 - l0, 1e-6)
    upper0  = h0 - max(o0, c0)
    lower0  = min(o0, c0) - l0
    body1   = abs(c1 - o1)

    bull0 = c0 > o0
    bull1 = c1 > o1

    # ── Bullish Engulfing ──────────────────────────────────────────────────────
    if not bull1 and bull0 and c0 > o1 and o0 < c1 and body0 > body1 * 1.1:
        patterns.append(CandlePattern("Bullish Engulfing", "bullish", "strong", timeframe))

    # ── Bearish Engulfing ──────────────────────────────────────────────────────
    elif bull1 and not bull0 and o0 > c1 and c0 < o1 and body0 > body1 * 1.1:
        patterns.append(CandlePattern("Bearish Engulfing", "bearish", "strong", timeframe))

    # ── Bullish Pin Bar (hammer / long lower wick — rejection of lows) ─────────
    if lower0 >= range0 * 0.55 and body0 <= range0 * 0.35 and upper0 <= range0 * 0.20:
        patterns.append(CandlePattern("Bullish Pin Bar", "bullish", "strong", timeframe))

    # ── Bearish Pin Bar (shooting star — rejection of highs) ──────────────────
    elif upper0 >= range0 * 0.55 and body0 <= range0 * 0.35 and lower0 <= range0 * 0.20:
        patterns.append(CandlePattern("Bearish Pin Bar", "bearish", "strong", timeframe))

    # ── Hammer (long lower wick, small body, can be bull or bear body) ─────────
    elif lower0 >= body0 * 2.0 and upper0 <= body0 * 0.5 and body0 > 0:
        patterns.append(CandlePattern("Hammer", "bullish", "moderate", timeframe))

    # ── Shooting Star (long upper wick, small body) ───────────────────────────
    elif upper0 >= body0 * 2.0 and lower0 <= body0 * 0.5 and body0 > 0:
        patterns.append(CandlePattern("Shooting Star", "bearish", "moderate", timeframe))

    # ── Doji (open ≈ close — indecision) ──────────────────────────────────────
    if body0 <= range0 * 0.08:
        patterns.append(CandlePattern("Doji", "neutral", "moderate", timeframe))

    # ── Inside Bar (consolidation; narrow range within prior candle) ───────────
    elif h0 < h1 and l0 > l1:
        patterns.append(CandlePattern("Inside Bar", "neutral", "moderate", timeframe))

    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# Economic event detection
# ─────────────────────────────────────────────────────────────────────────────

def _check_events() -> list[EventRisk]:
    """Check for upcoming FOMC dates and scan recent news for market-moving events."""
    events: list[EventRisk] = []
    today = date.today()

    for fomc_date in FOMC_DATES:
        delta = (fomc_date - today).days
        if -1 <= delta <= 4:
            if delta < 0:
                desc = f"FOMC decision was {abs(delta)} day(s) ago — market still digesting"
                impact = "medium"
            elif delta == 0:
                desc = "FOMC rate decision TODAY — extreme volatility possible, technicals unreliable"
                impact = "high"
            elif delta == 1:
                desc = "FOMC decision TOMORROW — pre-event uncertainty, avoid new positions"
                impact = "high"
            else:
                desc = f"FOMC decision in {delta} days — positioning phase, watch for whipsaws"
                impact = "medium"
            events.append(EventRisk("FOMC Meeting", delta, impact, desc))
            break  # only show the nearest FOMC

    # Scan recent news headlines for event keywords
    try:
        ticker = yf.Ticker(TICKER)
        news_items = ticker.news or []
        for item in news_items[:12]:
            title = (item.get("title") or "").lower()
            if any(kw in title for kw in EVENT_KEYWORDS):
                original = item.get("title", "")[:90]
                events.append(EventRisk(
                    "Breaking News",
                    0,
                    "medium",
                    f"Market-moving headline: {original}",
                ))
                break
    except Exception as e:
        logger.debug(f"News scan failed (non-critical): {e}")

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def _squeeze(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse any MultiIndex columns from yfinance into plain columns."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in df.columns:
        if hasattr(df[col], "squeeze"):
            try:
                df[col] = df[col].squeeze()
            except Exception:
                pass
    return df


def analyze() -> Signal:
    """
    Multi-timeframe SPY analysis.
    Downloads 3 timeframes and runs full technical + event analysis.
    Takes ~5-10 seconds.
    """
    logger.info("Fetching SPY data (daily / 1hr / 15min)...")

    # ── Data fetch ─────────────────────────────────────────────────────────────
    df_daily  = yf.download(TICKER, period="1y",  interval="1d",  auto_adjust=True, progress=False)
    df_hourly = yf.download(TICKER, period="30d", interval="1h",  auto_adjust=True, progress=False)
    df_15min  = yf.download(TICKER, period="5d",  interval="15m", auto_adjust=True, progress=False)

    if df_daily is None or len(df_daily) < 30:
        raise RuntimeError("Not enough daily market data from Yahoo Finance")

    df_daily  = _squeeze(df_daily)
    df_hourly = _squeeze(df_hourly) if df_hourly is not None and len(df_hourly) >= 10 else None
    df_15min  = _squeeze(df_15min)  if df_15min  is not None and len(df_15min)  >= 10 else None

    # ── Daily (macro) ──────────────────────────────────────────────────────────
    close_d = df_daily["Close"].squeeze()
    vol_d   = df_daily["Volume"].squeeze()
    price   = float(close_d.iloc[-1])

    rsi_daily       = float(_rsi(close_d).iloc[-1])
    ml_d, ms_d, _   = _macd(close_d)
    macd_signal_daily = _macd_label(ml_d, ms_d)

    ma50  = float(close_d.rolling(50).mean().iloc[-1])
    ma200 = float(close_d.rolling(200).mean().iloc[-1])
    above_50ma  = price > ma50
    above_200ma = price > ma200

    change_1d = float((close_d.iloc[-1] / close_d.iloc[-2] - 1) * 100)
    change_5d = float((close_d.iloc[-1] / close_d.iloc[-6] - 1) * 100) if len(close_d) >= 6 else 0.0

    bb_upper_s, bb_lower_s = _bollinger(close_d)
    bb_lower_val = float(bb_lower_s.iloc[-1])
    bb_upper_val = float(bb_upper_s.iloc[-1])
    near_bb_lower = price <= bb_lower_val * 1.01
    near_bb_upper = price >= bb_upper_val * 0.99
    bb_pct = (price - bb_lower_val) / (bb_upper_val - bb_lower_val) if bb_upper_val != bb_lower_val else 0.5

    vol_avg20    = float(vol_d.rolling(20).mean().iloc[-1])
    vol_today    = float(vol_d.iloc[-1])
    volume_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 1.0

    trend_structure = _trend_structure(close_d)

    nearest_support, nearest_resistance = _swing_levels(
        df_daily["High"].squeeze(), df_daily["Low"].squeeze(), price
    )

    # ── Hourly (intermediate) ──────────────────────────────────────────────────
    rsi_hourly        = 50.0
    macd_signal_hourly = "neutral"
    if df_hourly is not None:
        close_h = df_hourly["Close"].squeeze()
        if len(close_h) >= 26:
            rsi_hourly = float(_rsi(close_h).iloc[-1])
            ml_h, ms_h, _ = _macd(close_h)
            macd_signal_hourly = _macd_label(ml_h, ms_h)

    # ── 15-min (intraday) ──────────────────────────────────────────────────────
    rsi_15min  = 50.0
    vwap       = price
    above_vwap = True
    if df_15min is not None:
        close_15 = df_15min["Close"].squeeze()
        if len(close_15) >= 14:
            rsi_15min = float(_rsi(close_15).iloc[-1])
        vwap       = _vwap(df_15min)
        above_vwap = price > vwap

    # ── Candlestick patterns ───────────────────────────────────────────────────
    patterns: list[CandlePattern] = []
    if df_15min  is not None and len(df_15min)  >= 3:
        patterns.extend(_detect_patterns(df_15min,  "15min"))
    if df_hourly is not None and len(df_hourly) >= 3:
        patterns.extend(_detect_patterns(df_hourly, "1hr"))

    # ── Economic events ────────────────────────────────────────────────────────
    event_risks   = _check_events()
    event_override = any(e.impact == "high" and e.days_away <= 1 for e in event_risks)

    # ─────────────────────────────────────────────────────────────────────────
    # Composite score  (0 = max bearish · 100 = max bullish)
    # ─────────────────────────────────────────────────────────────────────────

    # Macro trend — 30 pts
    ma_score = 0.0
    if above_50ma and above_200ma:
        ma_score = 25.0
    elif above_50ma:
        ma_score = 15.0
    elif above_200ma:
        ma_score = 8.0
    if trend_structure == "uptrend":
        ma_score += 5.0
    elif trend_structure == "downtrend":
        ma_score -= 3.0
    ma_score = max(0.0, min(30.0, ma_score))

    # Daily RSI — 15 pts
    if rsi_daily < 30:       rsi_d_score = 15.0
    elif rsi_daily < 40:     rsi_d_score = 12.0
    elif rsi_daily < 50:     rsi_d_score = 8.0
    elif rsi_daily < 60:     rsi_d_score = 5.0
    elif rsi_daily < 70:     rsi_d_score = 2.0
    else:                    rsi_d_score = 0.0

    # Hourly RSI — 10 pts
    if rsi_hourly < 35:      rsi_h_score = 10.0
    elif rsi_hourly < 45:    rsi_h_score = 8.0
    elif rsi_hourly < 55:    rsi_h_score = 5.0
    elif rsi_hourly < 65:    rsi_h_score = 2.0
    else:                    rsi_h_score = 0.0

    # 15min RSI — 5 pts (intraday entry timing)
    if rsi_15min < 35:       rsi_15_score = 5.0
    elif rsi_15min < 45:     rsi_15_score = 4.0
    elif rsi_15min < 55:     rsi_15_score = 3.0
    elif rsi_15min < 65:     rsi_15_score = 1.0
    else:                    rsi_15_score = 0.0

    # Daily MACD — 10 pts
    if macd_signal_daily == "bullish cross":    macd_d_score = 10.0
    elif macd_signal_daily == "above":          macd_d_score = 7.0
    elif macd_signal_daily == "bearish cross":  macd_d_score = 0.0
    else:                                        macd_d_score = 2.0

    # Hourly MACD — 8 pts
    if macd_signal_hourly == "bullish cross":   macd_h_score = 8.0
    elif macd_signal_hourly == "above":         macd_h_score = 5.0
    elif macd_signal_hourly == "bearish cross": macd_h_score = 0.0
    else:                                        macd_h_score = 2.0

    # VWAP — 7 pts (price above VWAP = institutional buy bias)
    vwap_score = 7.0 if above_vwap else 0.0

    # Bollinger — 8 pts (near lower band = oversold / buy zone)
    bb_score = max(0.0, 8.0 * (1.0 - bb_pct))

    # Volume — 4 pts
    if volume_ratio > 1.5 and change_1d < 0:    vol_score = 4.0   # heavy selling = potential exhaustion
    elif volume_ratio > 1.5 and change_1d > 0:  vol_score = 3.0   # strong buying volume
    elif volume_ratio > 1.0:                     vol_score = 2.0
    else:                                        vol_score = 1.0

    # Candlestick patterns — ±8 pts
    pat_adj = 0.0
    for p in patterns:
        if p.bias == "bullish":
            pat_adj += 4.0 if p.strength == "strong" else 2.0
        elif p.bias == "bearish":
            pat_adj -= 4.0 if p.strength == "strong" else 2.0
    pat_adj = max(-8.0, min(8.0, pat_adj))

    base = ma_score + rsi_d_score + rsi_h_score + rsi_15_score + macd_d_score + macd_h_score + vwap_score + bb_score + vol_score
    score = max(0.0, min(100.0, base + pat_adj))

    # ── Direction + confidence ─────────────────────────────────────────────────
    if score >= 75:
        direction  = "STRONG BUY"
        confidence = min(0.95, 0.72 + (score - 75) / 100)
    elif score >= 58:
        direction  = "BUY"
        confidence = 0.56 + (score - 58) / 120
    elif score >= 42:
        direction  = "HOLD"
        confidence = 0.50
    elif score >= 25:
        direction  = "SELL"
        confidence = 0.56 + (42 - score) / 120
    else:
        direction  = "STRONG SELL"
        confidence = min(0.95, 0.72 + (25 - score) / 100)

    # Cap confidence when a high-impact event is imminent — uncertainty is real
    if event_override:
        confidence = min(confidence, 0.55)

    # ── Alert detection ────────────────────────────────────────────────────────
    alert_reasons: list[str] = []
    alert_type = "none"

    # Event warning fires first when FOMC is today or tomorrow
    high_events = [e for e in event_risks if e.days_away <= 1 and e.impact == "high"]
    if high_events:
        alert_type    = "event_warning"
        alert_reasons = [high_events[0].description]

    if alert_type != "event_warning":
        # Buy signals
        buy_reasons: list[str] = []
        if rsi_daily < 30:
            buy_reasons.append(f"Daily RSI deeply oversold ({rsi_daily:.1f})")
        elif rsi_daily < 40:
            buy_reasons.append(f"Daily RSI oversold ({rsi_daily:.1f})")
        if near_bb_lower:
            buy_reasons.append("Price at lower Bollinger Band support")
        if macd_signal_daily == "bullish cross":
            buy_reasons.append("Daily MACD bullish crossover")
        if macd_signal_hourly == "bullish cross":
            buy_reasons.append("Hourly MACD bullish crossover")
        if rsi_hourly < 35:
            buy_reasons.append(f"Hourly RSI oversold ({rsi_hourly:.1f})")
        if above_vwap and rsi_15min < 40:
            buy_reasons.append("Above VWAP with 15min RSI oversold — dip in uptrend")
        for p in patterns:
            if p.bias == "bullish" and p.strength == "strong":
                buy_reasons.append(f"{p.name} on {p.timeframe} chart")
        if change_1d < -1.5:
            buy_reasons.append(f"SPY down {change_1d:.1f}% today (potential dip buy)")

        # Sell signals
        sell_reasons: list[str] = []
        if rsi_daily > 75:
            sell_reasons.append(f"Daily RSI deeply overbought ({rsi_daily:.1f})")
        elif rsi_daily > 70:
            sell_reasons.append(f"Daily RSI overbought ({rsi_daily:.1f})")
        if near_bb_upper:
            sell_reasons.append("Price at upper Bollinger Band resistance")
        if macd_signal_daily == "bearish cross":
            sell_reasons.append("Daily MACD bearish crossover")
        if macd_signal_hourly == "bearish cross":
            sell_reasons.append("Hourly MACD bearish crossover")
        if not above_vwap and rsi_15min > 60:
            sell_reasons.append("Below VWAP with 15min RSI elevated — fade the move")
        for p in patterns:
            if p.bias == "bearish" and p.strength == "strong":
                sell_reasons.append(f"{p.name} on {p.timeframe} chart")
        if change_1d > 2.0:
            sell_reasons.append(f"SPY up {change_1d:.1f}% today (extended move)")
        if change_5d > 4.0:
            sell_reasons.append(f"SPY up {change_5d:.1f}% this week (extended)")

        if score >= 58 and len(buy_reasons) >= 1:
            alert_type    = "buy"
            alert_reasons = buy_reasons
        elif score <= 42 and len(sell_reasons) >= 2:
            alert_type    = "sell"
            alert_reasons = sell_reasons

    is_alert_worthy = (
        (alert_type in ("buy", "sell") and confidence >= 0.68) or
        alert_type == "event_warning"
    )

    return Signal(
        direction=direction,
        confidence=confidence,
        score=score,
        price=price,
        vwap=vwap,
        above_vwap=above_vwap,
        change_1d=change_1d,
        change_5d=change_5d,
        rsi_daily=rsi_daily,
        macd_signal_daily=macd_signal_daily,
        above_50ma=above_50ma,
        above_200ma=above_200ma,
        trend_structure=trend_structure,
        near_bb_lower=near_bb_lower,
        near_bb_upper=near_bb_upper,
        rsi_hourly=rsi_hourly,
        macd_signal_hourly=macd_signal_hourly,
        rsi_15min=rsi_15min,
        volume_ratio=volume_ratio,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        patterns=patterns,
        event_risks=event_risks,
        event_override=event_override,
        is_alert_worthy=is_alert_worthy,
        alert_type=alert_type,
        alert_reasons=alert_reasons,
        fetched_at=datetime.utcnow(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Message formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_signal_message(sig: Signal, header: str = "📊 S&P 500 Analysis") -> str:
    ALERT_THRESHOLD = 0.68
    display_direction = sig.direction
    if sig.direction in ("BUY", "SELL") and sig.confidence < ALERT_THRESHOLD:
        display_direction = f"HOLD (leaning {sig.direction})"

    emoji = {
        "STRONG BUY":  "🟢",
        "BUY":         "📈",
        "HOLD":        "⚖️",
        "SELL":        "📉",
        "STRONG SELL": "🔴",
    }.get(sig.direction, "❓")
    if sig.direction in ("BUY", "SELL") and sig.confidence < ALERT_THRESHOLD:
        emoji = "⚖️"

    chg1 = f"+{sig.change_1d:.2f}%" if sig.change_1d >= 0 else f"{sig.change_1d:.2f}%"
    chg5 = f"+{sig.change_5d:.2f}%" if sig.change_5d >= 0 else f"{sig.change_5d:.2f}%"

    dist_sup = ((sig.price - sig.nearest_support)    / sig.price) * 100
    dist_res = ((sig.nearest_resistance - sig.price) / sig.price) * 100

    lines = [
        f"*{header}*",
        f"SPY — ${sig.price:.2f}  {chg1} today · {chg5} 5d",
        "",
        f"{emoji} *{display_direction}*  ({sig.confidence*100:.0f}% confidence)",
        f"Score: {sig.score:.0f}/100 · Trend: *{sig.trend_structure}*",
        "",
        "*Multi-Timeframe Indicators:*",
        f"• Daily RSI(14):  {sig.rsi_daily:.1f}{'  ⚠️ oversold' if sig.rsi_daily < 35 else '  ⚠️ overbought' if sig.rsi_daily > 70 else ''}",
        f"• Hourly RSI(14): {sig.rsi_hourly:.1f}{'  ⚠️ oversold' if sig.rsi_hourly < 35 else '  ⚠️ overbought' if sig.rsi_hourly > 70 else ''}",
        f"• 15min RSI(14):  {sig.rsi_15min:.1f}",
        f"• Daily MACD:  {sig.macd_signal_daily}",
        f"• Hourly MACD: {sig.macd_signal_hourly}",
        f"• Trend:   {'✅' if sig.above_50ma else '❌'} 50MA · {'✅' if sig.above_200ma else '❌'} 200MA",
        f"• VWAP:    {'✅ above' if sig.above_vwap else '❌ below'} (${sig.vwap:.2f})",
        f"• Volume:  {sig.volume_ratio:.1f}x avg{'  🔥 spike' if sig.volume_ratio > 1.5 else ''}",
        f"• Bollinger: {'📍 lower band' if sig.near_bb_lower else '📍 upper band' if sig.near_bb_upper else 'mid-range'}",
        "",
        "*Key Levels:*",
        f"• Support:    ${sig.nearest_support:.2f} ({dist_sup:.1f}% below)",
        f"• Resistance: ${sig.nearest_resistance:.2f} ({dist_res:.1f}% above)",
    ]

    if sig.patterns:
        lines.append("")
        lines.append("*Candle Patterns:*")
        for p in sig.patterns:
            bias_emoji = "🟢" if p.bias == "bullish" else "🔴" if p.bias == "bearish" else "⚖️"
            lines.append(f"• {bias_emoji} {p.name} ({p.timeframe})")

    if sig.event_risks:
        lines.append("")
        lines.append("*⚡ Event Risk:*")
        for e in sig.event_risks:
            lines.append(f"  — {e.description}")

    if sig.alert_reasons and sig.alert_type not in ("none", "event_warning"):
        lines.append("")
        lines.append("*Why this signal:*")
        for r in sig.alert_reasons:
            lines.append(f"  — {r}")

    lines.append("")
    lines.append(f"_Updated {sig.fetched_at.strftime('%H:%M UTC')}_")
    return "\n".join(lines)


def format_alert_message(sig: Signal) -> str:
    if sig.alert_type == "event_warning":
        header = "⚡ Market Event Warning"
    elif sig.alert_type == "sell":
        header = "🔴 Strong Sell Signal" if sig.direction == "STRONG SELL" else "📉 Sell Signal Detected"
    else:
        header = "🟢 Strong Buy Opportunity" if sig.direction == "STRONG BUY" else "📈 Buy Signal Detected"
    return format_signal_message(sig, header=header)
