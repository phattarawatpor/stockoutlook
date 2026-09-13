"""
analysis.py
------------
Core statistics for the stock outlook tool: pulls real historical OHLC data
with yfinance and turns it into a historical-volatility-based estimate of:

  - probability that tomorrow closes up vs. down
  - a projected next-day high/low range
  - ATR / annualized volatility
  - recent headlines and the next earnings date (when available)

None of this is a guaranteed prediction or investment advice -- it is a
descriptive summary of how the stock has actually behaved recently, blended
a few standard ways. See README.md for the exact methodology.
"""
from __future__ import annotations

import math
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

ALPACA_LATEST_TRADE_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
US_EASTERN = ZoneInfo("America/New_York")


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE's actual holiday list -- close to but not identical to US federal
    holidays (NYSE also closes for Good Friday, which isn't a federal
    holiday, and stays open on Columbus Day / Veterans Day, which are)."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-01-01"),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def is_us_market_closed_today() -> bool:
    """True on weekends or NYSE holidays (checked in US Eastern time,
    the exchange's own timezone) -- i.e. days when polling for fresh US
    intraday/real-time data is pointless because nothing will move."""
    today_et = datetime.now(US_EASTERN).date()
    if today_et.weekday() >= 5:  # Saturday=5, Sunday=6
        return True
    holidays = NYSEHolidayCalendar().holidays(start=today_et, end=today_et)
    return len(holidays) > 0


@dataclass
class Stats:
    ticker: str
    company_name: str
    last_close: float
    prev_close: float
    last_date: str
    day_open: float
    day_high: float
    day_low: float
    six_month_high: float
    six_month_low: float
    atr14: float
    atr14_pct: float
    annualized_vol_pct: float
    up_days: int
    down_days: int
    total_days: int
    pct_up_frequency: float
    pct_up_normal_full: float
    pct_up_normal_recent: float
    pct_up_blended: float
    proj_high_typical: float
    proj_high_wide: float
    proj_low_typical: float
    proj_low_wide: float
    history: pd.DataFrame = field(repr=False)


def _norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    if sigma == 0:
        return 1.0 if x > mu else 0.0
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Daily OHLCV history, oldest first, columns: Open High Low Close Volume."""
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if df.empty:
        raise ValueError(f"No price data returned for '{ticker}'. Check the ticker symbol.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def compute_stats(ticker: str, period: str = "6mo", recent_window: int = 40) -> Stats:
    df = fetch_history(ticker, period=period)
    closes = df["Close"].tolist()
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    n = len(df)
    if n < 15:
        raise ValueError("Not enough history to compute stable statistics (need 15+ trading days).")

    # --- daily log returns, close-to-close ---
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, n)]
    up_days = sum(1 for r in rets if r > 0)
    down_days = sum(1 for r in rets if r < 0)

    pct_up_freq = up_days / len(rets) * 100

    mean_full, std_full = statistics.mean(rets), statistics.pstdev(rets)
    pct_up_normal_full = (1 - _norm_cdf(0, mean_full, std_full)) * 100

    recent_rets = rets[-recent_window:] if len(rets) >= recent_window else rets
    mean_r, std_r = statistics.mean(recent_rets), statistics.pstdev(recent_rets)
    pct_up_normal_recent = (1 - _norm_cdf(0, mean_r, std_r)) * 100

    pct_up_blended = (pct_up_freq + pct_up_normal_full + pct_up_normal_recent) / 3

    # --- ATR(14) ---
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    atr14 = statistics.mean(trs[-14:]) if len(trs) >= 14 else statistics.mean(trs)

    # --- 20-day annualized volatility ---
    w20 = rets[-20:] if len(rets) >= 20 else rets
    daily_std_20 = statistics.pstdev(w20)
    annualized_vol_pct = daily_std_20 * math.sqrt(252) * 100

    # --- next-day projected range, from recent high/low % vs prior close ---
    high_pct, low_pct = [], []
    for i in range(1, n):
        prev_close = closes[i - 1]
        high_pct.append((highs[i] - prev_close) / prev_close)
        low_pct.append((lows[i] - prev_close) / prev_close)

    rw = min(recent_window, len(high_pct))
    hp = sorted(high_pct[-rw:])
    lp = sorted(low_pct[-rw:])

    def pctile(sorted_list, p):
        idx = min(int(p * len(sorted_list)), len(sorted_list) - 1)
        return sorted_list[idx]

    last_close = closes[-1]
    proj_high_typical = last_close * (1 + statistics.median(hp))
    proj_high_wide = last_close * (1 + pctile(hp, 0.9))
    proj_low_typical = last_close * (1 + statistics.median(lp))
    proj_low_wide = last_close * (1 + pctile(lp, 0.1))

    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName") or info.get("shortName") or ticker.upper()
    except Exception:
        company_name = ticker.upper()

    last_row = df.iloc[-1]
    prev_close = closes[-2] if n >= 2 else closes[-1]

    return Stats(
        ticker=ticker.upper(),
        company_name=company_name,
        last_close=last_close,
        prev_close=prev_close,
        last_date=str(df.index[-1].date()),
        day_open=float(last_row["Open"]),
        day_high=float(last_row["High"]),
        day_low=float(last_row["Low"]),
        six_month_high=max(highs),
        six_month_low=min(lows),
        atr14=atr14,
        atr14_pct=atr14 / last_close * 100,
        annualized_vol_pct=annualized_vol_pct,
        up_days=up_days,
        down_days=down_days,
        total_days=len(rets),
        pct_up_frequency=pct_up_freq,
        pct_up_normal_full=pct_up_normal_full,
        pct_up_normal_recent=pct_up_normal_recent,
        pct_up_blended=pct_up_blended,
        proj_high_typical=proj_high_typical,
        proj_high_wide=proj_high_wide,
        proj_low_typical=proj_low_typical,
        proj_low_wide=proj_low_wide,
        history=df,
    )


def fetch_news(ticker: str, limit: int = 8) -> list[dict]:
    """Recent headlines from Yahoo Finance (title, publisher, link, published)."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    items = []
    for entry in raw[:limit]:
        content = entry.get("content", entry)  # yfinance has changed this shape across versions
        title = content.get("title") or entry.get("title")
        link = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or entry.get("link")
        )
        publisher = (content.get("provider") or {}).get("displayName") or entry.get("publisher")
        pub_date = content.get("pubDate") or entry.get("providerPublishTime")
        published = ""
        if isinstance(pub_date, (int, float)):
            published = datetime.fromtimestamp(pub_date, tz=timezone.utc).strftime("%Y-%m-%d")
        elif isinstance(pub_date, str):
            published = pub_date[:10]
        summary = content.get("summary") or ""
        if title:
            items.append({
                "title": title,
                "link": link,
                "publisher": publisher,
                "published": published,
                "summary": summary,
            })
    return items


NEWS_POSITIVE_WORDS = {
    "surge", "surges", "surged", "surging", "rally", "rallies", "rallied", "rallying",
    "jump", "jumps", "jumped", "jumping", "soar", "soars", "soared", "soaring",
    "gain", "gains", "gained", "gaining", "rise", "rises", "rising", "rose",
    "beat", "beats", "beating", "outperform", "outperforms", "outperformed",
    "upgrade", "upgraded", "upgrades", "bullish", "growth", "grows", "grew", "growing",
    "profit", "profits", "profitable", "record", "strong", "strength", "strengthens",
    "buy", "boost", "boosts", "boosted", "expand", "expands", "expansion", "expanded",
    "positive", "optimistic", "exceed", "exceeds", "exceeded", "exceeding",
    "top", "tops", "topped", "breakthrough", "win", "wins", "winning",
    "higher", "rebound", "rebounds", "rebounded", "recovery", "recovers", "recovered",
    "milestone", "success", "successful", "improve", "improves", "improved", "improving",
}
NEWS_NEGATIVE_WORDS = {
    "plunge", "plunges", "plunged", "plunging", "drop", "drops", "dropped", "dropping",
    "fall", "falls", "fell", "falling", "decline", "declines", "declined", "declining",
    "slump", "slumps", "slumped", "slumping", "crash", "crashes", "crashed", "crashing",
    "miss", "misses", "missed", "missing", "downgrade", "downgraded", "downgrades",
    "bearish", "loss", "losses", "losing", "weak", "weakness", "weakens",
    "cut", "cuts", "cutting", "sell", "selloff", "sell-off",
    "layoff", "layoffs", "lawsuit", "investigation", "probe", "fraud",
    "warning", "warns", "warned", "recall", "recalls", "recalled",
    "risk", "risks", "risky", "concern", "concerns", "concerning", "concerned",
    "lower", "plummet", "plummets", "plummeted", "plummeting",
    "tumble", "tumbles", "tumbled", "tumbling", "worry", "worries", "worried", "worrying",
    "delay", "delays", "delayed", "shortage", "shortages", "shortfall",
    "slowdown", "slowing", "struggle", "struggles", "struggled", "struggling",
}


def analyze_news_sentiment(title: str, summary: str = "") -> dict:
    """Very simple keyword-count heuristic over a fixed list of
    finance-relevant positive/negative words appearing in the headline +
    summary. This is NOT a trained sentiment model or NLP -- just a
    transparent word-count signal meant to flag likely tone at a glance.
    Ambiguous/sarcastic/negated phrasing ("fails to fall") will fool it.
    """
    text = f" {title} {summary} ".lower()
    words = re.findall(r"[a-z][a-z-]*", text)
    word_set = set(words)
    positive_hits = sorted(word_set & NEWS_POSITIVE_WORDS)
    negative_hits = sorted(word_set & NEWS_NEGATIVE_WORDS)
    score = len(positive_hits) - len(negative_hits)
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": score, "positive_hits": positive_hits, "negative_hits": negative_hits}


def _translate_chunk_en_to_th(translator, text: str, retries: int = 2) -> str:
    """One MyMemory call for a single request-sized chunk, with a couple of
    short retries on the free tier's burst rate limit (raises on final
    failure so callers can distinguish "gave up" from "empty result")."""
    from deep_translator.exceptions import TooManyRequests

    delay = 1.5
    for attempt in range(retries + 1):
        try:
            return translator.translate(text[:490])
        except TooManyRequests:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2


def translate_to_thai(text: str) -> str | None:
    """Best-effort English -> Thai translation via the free MyMemory
    translation API (deep-translator). Returns None on any failure (offline,
    rate-limited, text too long, etc) so callers can fall back to showing
    only the English original."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        from deep_translator import MyMemoryTranslator
        translator = MyMemoryTranslator(source="en-US", target="th-TH")
        return _translate_chunk_en_to_th(translator, text)
    except Exception:
        return None


def fetch_full_article_text(url: str) -> str | None:
    """Best-effort full-article scrape via trafilatura, used when the news
    feed only provides a short truncated teaser summary (Yahoo's own
    summaries routinely end with "[...]"). Works well on ordinary news
    pages; fails gracefully (returns None) on paywalled, JS-rendered, or
    anti-scraping-protected sites -- callers should fall back to the short
    summary in that case."""
    if not url:
        return None
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
        return text.strip() if text else None
    except Exception:
        return None


def translate_long_text_to_thai(text: str, chunk_size: int = 450) -> str | None:
    """Translate longer text (e.g. a full scraped article) to Thai by
    splitting it into MyMemory-request-sized chunks along sentence
    boundaries and translating each in turn. Returns None if any chunk
    fails, so callers fall back to whatever shorter translation they
    already have rather than showing a partially-translated article."""
    text = (text or "").strip()
    if not text:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)

    try:
        from deep_translator import MyMemoryTranslator
        translator = MyMemoryTranslator(source="en-US", target="th-TH")
        translated_chunks = []
        for chunk in chunks:
            translated_chunks.append(_translate_chunk_en_to_th(translator, chunk))
            time.sleep(0.35)  # stay under MyMemory's free-tier burst limit
    except Exception:
        return None
    if any(not c for c in translated_chunks):
        return None
    return " ".join(translated_chunks)


def fetch_live_price(ticker: str) -> float | None:
    """Latest traded price (~15-20 min delayed per Yahoo Finance), or None if unavailable."""
    try:
        price = yf.Ticker(ticker).fast_info["last_price"]
        return float(price) if price is not None else None
    except Exception:
        return None


def fetch_alpaca_price(ticker: str, api_key_id: str, secret_key: str) -> float | None:
    """True real-time latest trade price from Alpaca's free Basic market-data
    plan (US-listed stocks, IEX feed only). Returns None on missing
    credentials, unsupported ticker (e.g. non-US), or any request failure --
    callers should fall back to `fetch_live_price` in that case."""
    if not api_key_id or not secret_key:
        return None
    try:
        resp = requests.get(
            ALPACA_LATEST_TRADE_URL.format(symbol=ticker.upper()),
            headers={"APCA-API-KEY-ID": api_key_id, "APCA-API-SECRET-KEY": secret_key},
            params={"feed": "iex"},
            timeout=5,
        )
        resp.raise_for_status()
        return float(resp.json()["trade"]["p"])
    except Exception:
        return None


def project_range_from_price(stats: Stats, price: float, recent_window: int = 40) -> tuple[float, float, float, float]:
    """Re-anchor the typical/wide projected next-day range onto a given price
    (e.g. the live price) instead of the last daily close, reusing the same
    high/low-vs-prior-close percentile stats already computed in `stats`."""
    df = stats.history
    closes = df["Close"].tolist()
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    n = len(df)

    high_pct, low_pct = [], []
    for i in range(1, n):
        prev_close = closes[i - 1]
        high_pct.append((highs[i] - prev_close) / prev_close)
        low_pct.append((lows[i] - prev_close) / prev_close)

    rw = min(recent_window, len(high_pct))
    hp = sorted(high_pct[-rw:])
    lp = sorted(low_pct[-rw:])

    def pctile(sorted_list, p):
        idx = min(int(p * len(sorted_list)), len(sorted_list) - 1)
        return sorted_list[idx]

    proj_high_typical = price * (1 + statistics.median(hp))
    proj_high_wide = price * (1 + pctile(hp, 0.9))
    proj_low_typical = price * (1 + statistics.median(lp))
    proj_low_wide = price * (1 + pctile(lp, 0.1))
    return proj_low_typical, proj_high_typical, proj_low_wide, proj_high_wide


@dataclass
class Momentum:
    direction: str  # "up" | "down" | "neutral"
    label: str
    rsi: float | None
    sma_fast: float
    sma_slow: float
    last_price: float
    bars_used: int


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = statistics.mean(gains[-period:])
    avg_loss = statistics.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def fetch_intraday(ticker: str) -> pd.DataFrame | None:
    """Today's 1-minute OHLC bars, or None if unavailable (market closed,
    thin ticker, request failure, etc). Shared source for the momentum
    signal and the minute-ahead projection so both reuse one API call."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def compute_momentum(df: pd.DataFrame | None, fast_window: int = 3, slow_window: int = 10) -> Momentum | None:
    """Very short-term (1-minute-bar) momentum signal: SMA(fast) vs SMA(slow)
    crossover confirmed by RSI(14), both on today's 1-minute intraday closes.

    This is NOT a prediction — it only summarizes the last few minutes of
    price action, which is dominated by noise at this timescale.
    """
    if df is None or len(df) < slow_window:
        return None

    closes = df["Close"].dropna().tolist()
    if len(closes) < slow_window:
        return None

    rsi = _rsi(closes, 14)
    sma_fast = statistics.mean(closes[-fast_window:])
    sma_slow = statistics.mean(closes[-slow_window:])
    last_price = closes[-1]

    bullish = sma_fast > sma_slow and (rsi is None or rsi >= 50)
    bearish = sma_fast < sma_slow and (rsi is None or rsi <= 50)
    if bullish and not bearish:
        direction, label = "up", "โมเมนตัมขึ้น"
    elif bearish and not bullish:
        direction, label = "down", "โมเมนตัมลง"
    else:
        direction, label = "neutral", "สัญญาณผสม / ไม่ชัดเจน"

    return Momentum(
        direction=direction,
        label=label,
        rsi=rsi,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        last_price=last_price,
        bars_used=len(closes),
    )


def fetch_momentum(ticker: str, fast_window: int = 3, slow_window: int = 10) -> Momentum | None:
    return compute_momentum(fetch_intraday(ticker), fast_window, slow_window)


@dataclass
class MinuteProjection:
    minutes: list[int]  # 1..horizon_minutes
    central: list[float]
    upper: list[float]
    lower: list[float]
    last_price: float
    per_minute_vol_pct: float
    band_confidence_pct: float


def compute_minute_projection(
    df: pd.DataFrame | None,
    horizon_minutes: int = 10,
    lookback_minutes: int = 60,
    z: float = 1.28,  # ~80% two-sided band under a normal-return assumption
) -> MinuteProjection | None:
    """Forward-looking projection cone for the next `horizon_minutes`, built
    as a random-walk (GBM-style) extrapolation from today's realized 1-minute
    drift and volatility: last_price * exp(mean*t +/- z*std*sqrt(t)).

    This is a statistical "cone of uncertainty" like a weather forecast cone,
    NOT a prediction of the actual price path. Real prices routinely exit
    the band, especially around news or order-flow shocks the model can't
    see. Returns None if there isn't enough intraday data yet.
    """
    if df is None:
        return None
    closes = df["Close"].dropna().tolist()
    if len(closes) < 6:
        return None

    recent = closes[-lookback_minutes:] if len(closes) >= lookback_minutes else closes
    rets = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]
    if len(rets) < 5:
        return None

    mean_r = statistics.mean(rets)
    std_r = statistics.pstdev(rets)
    last_price = closes[-1]

    minutes = list(range(1, horizon_minutes + 1))
    central, upper, lower = [], [], []
    for t in minutes:
        drift = mean_r * t
        spread = std_r * math.sqrt(t) * z
        central.append(last_price * math.exp(drift))
        upper.append(last_price * math.exp(drift + spread))
        lower.append(last_price * math.exp(drift - spread))

    band_confidence_pct = (2 * _norm_cdf(z) - 1) * 100

    return MinuteProjection(
        minutes=minutes,
        central=central,
        upper=upper,
        lower=lower,
        last_price=last_price,
        per_minute_vol_pct=std_r * 100,
        band_confidence_pct=band_confidence_pct,
    )


def fetch_minute_projection(
    ticker: str, horizon_minutes: int = 10, lookback_minutes: int = 60
) -> MinuteProjection | None:
    return compute_minute_projection(fetch_intraday(ticker), horizon_minutes, lookback_minutes)


@dataclass
class HistoricalMoveStats:
    days_used: int
    windows_used: int
    minutes: list[int]  # 1..horizon_minutes
    max_up_typical_pct: list[float]
    max_up_wide_pct: list[float]
    max_down_typical_pct: list[float]
    max_down_wide_pct: list[float]
    max_up_typical_price: list[float]
    max_up_wide_price: list[float]
    max_down_typical_price: list[float]
    max_down_wide_price: list[float]


def compute_historical_minute_moves(
    df: pd.DataFrame | None,
    last_price: float,
    horizon_minutes: int = 10,
    time_window_minutes: int = 45,
) -> HistoricalMoveStats | None:
    """Backward-looking analog: scans the last several trading days of
    1-minute bars for windows that start near the same time of day as the
    most recent bar (+/- `time_window_minutes`). For each such window,
    computes -- separately for +1, +2, ... +horizon_minutes minutes ahead --
    the largest up-move and largest down-move reached *so far within that
    window*, then summarizes each minute's distribution as typical (median)
    and wide (90th/10th percentile) figures.

    This answers "around this time of day, how far has the stock actually
    swung by minute N on recent days?" -- a statistical analog to similar
    past periods, NOT a prediction and NOT informed by news or any event
    that hasn't happened yet in the historical sample.
    """
    if df is None or len(df) < horizon_minutes + 2:
        return None

    closes = df["Close"].dropna()
    if len(closes) < horizon_minutes + 2:
        return None

    idx = closes.index
    values = closes.to_numpy()
    dates = idx.date
    n = len(values)
    ref_minute = idx[-1].hour * 60 + idx[-1].minute

    up_moves_by_t: list[list[float]] = [[] for _ in range(horizon_minutes)]
    down_moves_by_t: list[list[float]] = [[] for _ in range(horizon_minutes)]
    days_used = set()
    for i in range(n - horizon_minutes):
        minute_of_day = idx[i].hour * 60 + idx[i].minute
        if abs(minute_of_day - ref_minute) > time_window_minutes:
            continue
        if dates[i] != dates[i + horizon_minutes]:  # skip windows crossing a session boundary
            continue
        start = values[i]
        window = values[i + 1: i + horizon_minutes + 1]
        running_max, running_min = window[0], window[0]
        for t in range(horizon_minutes):
            running_max = max(running_max, window[t])
            running_min = min(running_min, window[t])
            up_moves_by_t[t].append((running_max - start) / start)
            down_moves_by_t[t].append((running_min - start) / start)
        days_used.add(dates[i])

    if len(up_moves_by_t[0]) < 5:
        return None

    def pctile(sorted_list, p):
        idx_ = min(int(p * len(sorted_list)), len(sorted_list) - 1)
        return sorted_list[idx_]

    minutes = list(range(1, horizon_minutes + 1))
    max_up_typical_pct, max_up_wide_pct = [], []
    max_down_typical_pct, max_down_wide_pct = [], []
    for t in range(horizon_minutes):
        ups = sorted(up_moves_by_t[t])
        downs = sorted(down_moves_by_t[t])
        max_up_typical_pct.append(statistics.median(ups) * 100)
        max_up_wide_pct.append(pctile(ups, 0.9) * 100)
        max_down_typical_pct.append(statistics.median(downs) * 100)
        max_down_wide_pct.append(pctile(downs, 0.1) * 100)

    return HistoricalMoveStats(
        days_used=len(days_used),
        windows_used=len(up_moves_by_t[0]),
        minutes=minutes,
        max_up_typical_pct=max_up_typical_pct,
        max_up_wide_pct=max_up_wide_pct,
        max_down_typical_pct=max_down_typical_pct,
        max_down_wide_pct=max_down_wide_pct,
        max_up_typical_price=[last_price * (1 + p / 100) for p in max_up_typical_pct],
        max_up_wide_price=[last_price * (1 + p / 100) for p in max_up_wide_pct],
        max_down_typical_price=[last_price * (1 + p / 100) for p in max_down_typical_pct],
        max_down_wide_price=[last_price * (1 + p / 100) for p in max_down_wide_pct],
    )


def fetch_historical_minute_moves(ticker: str, horizon_minutes: int = 10) -> HistoricalMoveStats | None:
    try:
        df = yf.Ticker(ticker).history(period="7d", interval="1m", auto_adjust=False)
    except Exception:
        return None
    closes = df["Close"].dropna() if not df.empty else df
    if df.empty or closes.empty:
        return None
    return compute_historical_minute_moves(df, float(closes.iloc[-1]), horizon_minutes)


def fetch_next_earnings_date(ticker: str) -> str | None:
    try:
        dates = yf.Ticker(ticker).get_earnings_dates(limit=8)
        if dates is None or dates.empty:
            return None
        now = pd.Timestamp.now(tz=dates.index.tz)
        upcoming = dates[dates.index >= now]
        if upcoming.empty:
            return None
        return str(upcoming.index[-1].date())
    except Exception:
        return None


NASDAQ_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Thai SET tickers aren't published anywhere with a simple free bulk feed
# (unlike NASDAQ Trader's US listings below), so this is just a small
# hand-picked set of well-known names to keep Thai-market search useful.
THAI_TICKERS = [
    ("PTT.BK", "PTT Public Company Limited"),
    ("AOT.BK", "Airports of Thailand"),
    ("CPALL.BK", "CP All"),
    ("SCB.BK", "SCB X"),
    ("KBANK.BK", "Kasikornbank"),
    ("ADVANC.BK", "Advanced Info Service"),
    ("BBL.BK", "Bangkok Bank"),
    ("PTTEP.BK", "PTT Exploration and Production"),
    ("SCC.BK", "Siam Cement"),
    ("CPF.BK", "Charoen Pokphand Foods"),
    ("TRUE.BK", "True Corporation"),
    ("DELTA.BK", "Delta Electronics (Thailand)"),
    ("GULF.BK", "Gulf Energy Development"),
    ("BDMS.BK", "Bangkok Dusit Medical Services"),
    ("HMPRO.BK", "Home Product Center"),
]


def fetch_us_ticker_directory() -> list[tuple[str, str]]:
    """Full directory of active, ordinary-share US common stocks from
    NASDAQ Trader's official (free, no key needed) symbol listing files --
    covers NASDAQ + NYSE + NYSE American + Arca + Cboe BZX. Excludes ETFs,
    warrants, rights, units, preferred shares, and test issues to keep this
    focused on plain common stocks. Returns [] on any network failure so
    callers can fall back to a smaller hardcoded list."""
    try:
        nasdaq_text = requests.get(NASDAQ_LISTED_URL, timeout=15).text
        other_text = requests.get(OTHER_LISTED_URL, timeout=15).text
    except Exception:
        return []

    def parse(text: str, symbol_idx: int, name_idx: int, test_idx: int, etf_idx: int):
        lines = text.strip().split("\n")[1:-1]  # drop header + "File Creation Time" footer
        rows = []
        for line in lines:
            parts = line.split("|")
            if len(parts) <= max(symbol_idx, name_idx, test_idx, etf_idx):
                continue
            rows.append((parts[symbol_idx], parts[name_idx], parts[test_idx], parts[etf_idx]))
        return rows

    try:
        rows = (
            parse(nasdaq_text, symbol_idx=0, name_idx=1, test_idx=3, etf_idx=6)
            + parse(other_text, symbol_idx=0, name_idx=1, test_idx=6, etf_idx=4)
        )
    except Exception:
        return []

    exclude_keywords = ("warrant", "right", "unit", "preferred", " notes", "depositary shares each representing")
    combined: dict[str, str] = {}
    for symbol, name, test_issue, etf in rows:
        symbol = symbol.strip()
        if not symbol or test_issue == "Y" or etf == "Y" or "$" in symbol:
            continue
        if any(k in name.lower() for k in exclude_keywords):
            continue
        combined[symbol] = name.strip()

    return sorted(combined.items())
