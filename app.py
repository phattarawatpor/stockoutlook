"""
Stock Outlook Dashboard
------------------------
Local Streamlit app: type any ticker, get real historical data (via
yfinance), a candlestick chart, a historical-volatility-based probability
of an up/down day tomorrow, a projected next-day high/low range, and recent
headlines + the next earnings date.

Run:
    streamlit run app.py

This is a statistical / educational tool, NOT investment advice.
"""
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    analyze_news_sentiment,
    compute_minute_projection,
    compute_momentum,
    compute_stats,
    fetch_alpaca_price,
    fetch_full_article_text,
    fetch_historical_minute_moves,
    fetch_intraday,
    fetch_live_price,
    fetch_news,
    fetch_next_earnings_date,
    is_us_market_closed_today,
    project_range_from_price,
    translate_long_text_to_thai,
    translate_to_thai,
)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_translate(text: str) -> str | None:
    return translate_to_thai(text)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_full_article_th(url: str) -> str | None:
    full_text = fetch_full_article_text(url)
    if not full_text:
        return None
    return translate_long_text_to_thai(full_text)

try:
    _alpaca_secrets = st.secrets.get("alpaca", {})
except Exception:
    _alpaca_secrets = {}
ALPACA_KEY_ID = _alpaca_secrets.get("api_key_id", "")
ALPACA_SECRET_KEY = _alpaca_secrets.get("secret_key", "")

MARKET_CLOSED_TODAY = is_us_market_closed_today()

st.set_page_config(page_title="Stock Outlook Dashboard", page_icon="📈", layout="wide")

st.title("📈 Stock Outlook Dashboard")
st.caption("ความน่าจะเป็นขึ้น/ลง และช่วงราคาที่คาดวันถัดไป คำนวณจากความผันผวนของราคาย้อนหลังจริง — ไม่ใช่คำแนะนำการลงทุน")

POPULAR_TICKERS = [
    "GOOGL", "GOOG", "AAPL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "AVGO", "AMD",
    "NFLX", "ADBE", "CRM", "ORCL", "INTC", "MU", "QCOM", "TXN", "IBM", "CSCO",
    "JPM", "BAC", "WFC", "GS", "V", "MA", "PYPL", "DIS", "KO", "PEP",
    "WMT", "COST", "HD", "NKE", "MCD", "SBUX", "PG", "JNJ", "PFE", "UNH",
    "XOM", "CVX", "BA", "CAT", "GE", "F", "GM", "UBER", "ABNB", "SHOP",
    "SPY", "QQQ", "VOO", "DIA", "PTT.BK", "AOT.BK", "CPALL.BK", "SCB.BK", "KBANK.BK", "ADVANC.BK",
]

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    ticker = st.selectbox(
        "Ticker",
        options=POPULAR_TICKERS,
        index=POPULAR_TICKERS.index("GOOGL"),
        accept_new_options=True,
        placeholder="พิมพ์หรือเลือก ticker เช่น GOOGL, AAPL, PTT.BK",
    )
    ticker = (ticker or "").strip().upper()
with col_b:
    period = st.selectbox("ช่วงข้อมูลย้อนหลัง", ["3mo", "6mo", "1y", "2y"], index=1)
with col_c:
    st.write("")
    st.write("")
    run = st.button("วิเคราะห์", type="primary", use_container_width=True)

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

if run or (st.session_state.last_ticker is None and ticker):
    st.session_state.last_ticker = ticker

if not st.session_state.last_ticker:
    st.stop()

ticker = st.session_state.last_ticker

try:
    with st.spinner(f"กำลังดึงข้อมูล {ticker} ..."):
        stats = compute_stats(ticker, period=period)
        news = fetch_news(ticker)
        earnings_date = fetch_next_earnings_date(ticker)
except Exception as e:
    st.error(f"ดึงข้อมูลไม่สำเร็จ: {e}")
    st.stop()

# ---------- Header ----------
chg = stats.last_close - stats.prev_close
chg_pct = chg / stats.prev_close * 100
color = "green" if chg >= 0 else "red"
arrow = "▲" if chg >= 0 else "▼"

h1, h2 = st.columns([3, 2])
with h1:
    st.subheader(f"{stats.ticker} · {stats.company_name}")
    st.markdown(
        f"<span style='font-size:2.2rem; font-weight:700;'>${stats.last_close:,.2f}</span> "
        f"<span style='color:{color}; font-weight:600;'>{arrow} {abs(chg):.2f} ({abs(chg_pct):.2f}%)</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Close of {stats.last_date} · Day range ${stats.day_low:,.2f} – ${stats.day_high:,.2f}"
    )
with h2:
    if earnings_date:
        st.metric("วันประกาศผลประกอบการถัดไป", earnings_date)
    else:
        st.metric("วันประกาศผลประกอบการถัดไป", "ไม่พบข้อมูล")

# ---------- Chart ----------
st.markdown("### ราคาย้อนหลัง (แท่งเทียน + ปริมาณซื้อขาย)")
df = stats.history
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#0ca30c", decreasing_line_color="#d03b3b",
        name=stats.ticker,
    )
)
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], marker_color="#c3c2b7"))
vol_fig.update_layout(height=120, margin=dict(l=10, r=10, t=0, b=10), template="plotly_white")
st.plotly_chart(vol_fig, use_container_width=True)

# ---------- Stat tiles ----------
st.markdown("### สถิติความผันผวน")
t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("สูงสุดในช่วงข้อมูล", f"${stats.six_month_high:,.2f}")
t2.metric("ต่ำสุดในช่วงข้อมูล", f"${stats.six_month_low:,.2f}")
t3.metric("ATR (14 วัน)", f"${stats.atr14:,.2f}", f"{stats.atr14_pct:.1f}% ของราคา")
t4.metric("Volatility รายปี (20 วัน)", f"{stats.annualized_vol_pct:.1f}%")
t5.metric("วันขึ้น / วันลง", f"{stats.up_days} / {stats.down_days}")

# ---------- Outlook ----------
st.markdown("### โอกาสวันถัดไป")
o1, o2 = st.columns(2)
with o1:
    st.progress(int(round(stats.pct_up_blended)), text=f"โอกาสปิดบวก ~{stats.pct_up_blended:.1f}%")
    st.caption(
        f"ความถี่จริง {stats.pct_up_frequency:.1f}% · แบบจำลอง normal (ทั้งชุด) {stats.pct_up_normal_full:.1f}% "
        f"· แบบจำลอง normal (ล่าสุด) {stats.pct_up_normal_recent:.1f}%"
    )
    st.caption(
        "คำนวณจาก 3 มุมมองของผลตอบแทนรายวันในอดีต แล้วเฉลี่ยกัน "
        "ไม่ใช่การพยากรณ์ที่รับประกันผล ยิ่งค่าใกล้ 50% ยิ่งแปลว่าทิศทางคาดเดายากกว่าปกติ"
    )
with o2:
    st.markdown(f"**ช่วงราคาคาดการณ์วันถัดไป** (จากราคาปิดล่าสุด ${stats.last_close:,.2f})")
    st.write(f"- ช่วงปกติ: **${stats.proj_low_typical:,.2f} – ${stats.proj_high_typical:,.2f}**")
    st.write(f"- ช่วงกว้าง (ผันผวนสูง, ~10–90 เปอร์เซ็นไทล์): **${stats.proj_low_wide:,.2f} – ${stats.proj_high_wide:,.2f}**")
    st.caption("คำนวณจากสัดส่วน high/low เทียบราคาปิดก่อนหน้า ของ 40 วันทำการล่าสุด")

# ---------- Live projected price ----------
st.markdown("### ราคาคาดการณ์แบบเรียลไทม์")

HAS_ALPACA = bool(ALPACA_KEY_ID and ALPACA_SECRET_KEY)

if MARKET_CLOSED_TODAY:
    st.info(
        "📅 วันนี้ตลาดหุ้นสหรัฐฯ ปิด (เสาร์-อาทิตย์ หรือวันหยุด NYSE) — แสดงข้อมูลล่าสุดที่มี "
        "โดย**ไม่รีเฟรชอัตโนมัติ** เพื่อไม่ให้ดึงข้อมูลซ้ำโดยเปล่าประโยชน์"
    )


@st.fragment(run_every=None if MARKET_CLOSED_TODAY else (5 if HAS_ALPACA else 30))
def _live_price(stats):
    is_realtime = False
    live_price = None
    if HAS_ALPACA:
        live_price = fetch_alpaca_price(stats.ticker, ALPACA_KEY_ID, ALPACA_SECRET_KEY)
        is_realtime = live_price is not None
    if live_price is None:
        live_price = fetch_live_price(stats.ticker)

    if live_price is None:
        st.warning("ดึงราคาล่าสุดไม่สำเร็จ (ลองใหม่ในรอบถัดไป)")
        return

    live_chg = live_price - stats.last_close
    live_chg_pct = live_chg / stats.last_close * 100
    live_color = "green" if live_chg >= 0 else "red"
    live_arrow = "▲" if live_chg >= 0 else "▼"

    lo_typ, hi_typ, lo_wide, hi_wide = project_range_from_price(stats, live_price)

    if MARKET_CLOSED_TODAY:
        source_badge = ":gray[● ตลาดปิด] · ไม่มีการรีเฟรชอัตโนมัติวันนี้"
    elif is_realtime:
        source_badge = ":green[● เรียลไทม์ (Alpaca / IEX)] · อัปเดตทุก 5 วินาที"
    else:
        source_badge = ":gray[● ดีเลย์ ~15-20 นาที (Yahoo Finance)] · อัปเดตทุก 30 วินาที"

    l1, l2 = st.columns([2, 3])
    with l1:
        st.markdown(
            f"<span style='font-size:1.8rem; font-weight:700;'>${live_price:,.2f}</span> "
            f"<span style='color:{live_color}; font-weight:600;'>{live_arrow} {abs(live_chg):.2f} ({abs(live_chg_pct):.2f}%)</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"เทียบราคาปิดล่าสุด ${stats.last_close:,.2f}")
        st.caption(source_badge)
    with l2:
        st.write(f"- ช่วงปกติ (จากราคาล่าสุด): **${lo_typ:,.2f} – ${hi_typ:,.2f}**")
        st.write(f"- ช่วงกว้าง (~10–90 เปอร์เซ็นไทล์): **${lo_wide:,.2f} – ${hi_wide:,.2f}**")


_live_price(stats)

if not HAS_ALPACA:
    st.caption(
        "💡 ตอนนี้ใช้ราคาดีเลย์จาก Yahoo Finance อยู่ — ใส่ Alpaca API key ใน `.streamlit/secrets.toml` "
        "เพื่อเปิดโหมดราคาเรียลไทม์จริง (เฉพาะหุ้นสหรัฐฯ) ดูวิธีตั้งค่าใน README"
    )


@st.fragment(run_every=None if MARKET_CLOSED_TODAY else 30)
def _momentum_and_forecast(stats):
    st.divider()
    df_intraday = fetch_intraday(stats.ticker)
    momentum = compute_momentum(df_intraday)
    if momentum is None:
        st.caption("ไม่มีข้อมูลรายนาที (ตลาดอาจปิดอยู่ หรือ ticker นี้ไม่มีข้อมูล intraday)")
        return

    icon = {"up": ":material/trending_up:", "down": ":material/trending_down:", "neutral": ":material/trending_flat:"}[momentum.direction]
    m_color = {"up": "green", "down": "red", "neutral": "gray"}[momentum.direction]
    rsi_text = f"RSI(14) {momentum.rsi:.0f}" if momentum.rsi is not None else "RSI ไม่พร้อมใช้งาน"
    st.markdown(f"**{icon} แนวโน้มโมเมนตัมนาทีล่าสุด:** :{m_color}[{momentum.label}]")
    st.caption(
        f"{rsi_text} · SMA เร็ว (3 นาที) ${momentum.sma_fast:,.2f} vs SMA ช้า (10 นาที) ${momentum.sma_slow:,.2f} "
        f"· จากแท่ง 1 นาที {momentum.bars_used} แท่ง"
    )
    st.caption(
        "⚠️ เป็นแค่สรุปทิศทางราคาของไม่กี่นาทีที่ผ่านมา ไม่ใช่การพยากรณ์นาทีถัดไป "
        "การเคลื่อนไหวระยะสั้นขนาดนี้ใกล้เคียงสุ่มมาก แม่นยำต่ำ ใช้ประกอบการอ่านเท่านั้น"
    )

    st.markdown("#### กราฟคาดการณ์ราคา 10 นาทีข้างหน้า")
    projection = compute_minute_projection(df_intraday, horizon_minutes=10)
    if projection is None:
        st.caption("ข้อมูลไม่พอสำหรับกราฟคาดการณ์นาทีข้างหน้า")
        return

    now = datetime.now()
    x_now = [now]
    x_future = [now + timedelta(minutes=m) for m in projection.minutes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_future, y=projection.upper, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_future, y=projection.lower, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(76,120,255,0.15)",
        name=f"ช่วงความเป็นไปได้ (~{projection.band_confidence_pct:.0f}%)",
    ))
    fig.add_trace(go.Scatter(
        x=x_future, y=projection.central, mode="lines+markers",
        line=dict(color="#4c78ff", width=2), marker=dict(size=4),
        name="เส้นกลาง (แนวโน้มปัจจุบัน)",
    ))
    fig.add_trace(go.Scatter(
        x=x_now, y=[projection.last_price], mode="markers",
        marker=dict(color="black", size=9), name="ราคาปัจจุบัน",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
        xaxis_title=None,
        yaxis_title="ราคา ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Volatility ต่อนาที ~{projection.per_minute_vol_pct:.2f}% (จากข้อมูล 1 นาทีล่าสุด) "
        "· คำนวณแบบ random-walk (GBM) จากค่าเฉลี่ย/ส่วนเบี่ยงเบนของผลตอบแทนรายนาทีล่าสุด ยิ่งไกลจากปัจจุบันช่วงยิ่งกว้างขึ้น"
    )
    st.caption(
        "⚠️ เป็นกรวยความไม่แน่นอนเชิงสถิติ (เหมือน cone พยากรณ์อากาศ) ไม่ใช่การพยากรณ์เส้นทางราคาจริง "
        "ราคาจริงอาจหลุดกรอบนี้ได้ง่าย โดยเฉพาะเมื่อมีข่าวหรือคำสั่งซื้อขายผิดปกติเข้ามากะทันหัน"
    )

    st.markdown("#### ตัวเลขคาดการณ์ขึ้นสูงสุด / ลงสูงสุด รายนาที (+1 ถึง +10 นาที)")
    hist_moves = fetch_historical_minute_moves(stats.ticker, horizon_minutes=10)
    if hist_moves is None:
        st.caption("ข้อมูลย้อนหลังไม่พอสำหรับคำนวณสถิติช่วงเวลาเดียวกัน (ต้องการอย่างน้อย ~5 ช่วงเวลาที่ใกล้เคียงกันใน 7 วันทำการล่าสุด)")
    else:
        pct_cols = {
            "ขึ้นสูงสุด (ค่ากลาง)": hist_moves.max_up_typical_pct,
            "ขึ้นสูงสุด (กว้าง ~90th)": hist_moves.max_up_wide_pct,
            "ลงสูงสุด (ค่ากลาง)": hist_moves.max_down_typical_pct,
            "ลงสูงสุด (กว้าง ~10th)": hist_moves.max_down_wide_pct,
        }
        price_cols = {
            "ขึ้นสูงสุด (ค่ากลาง)": hist_moves.max_up_typical_price,
            "ขึ้นสูงสุด (กว้าง ~90th)": hist_moves.max_up_wide_price,
            "ลงสูงสุด (ค่ากลาง)": hist_moves.max_down_typical_price,
            "ลงสูงสุด (กว้าง ~10th)": hist_moves.max_down_wide_price,
        }

        prev_key = f"hist_moves_prev_{stats.ticker}"
        prev_pct_cols = st.session_state.get(prev_key)

        def cell_color(col_name, i, value):
            if not prev_pct_cols or col_name not in prev_pct_cols:
                return ""
            prev_row = prev_pct_cols[col_name]
            if i >= len(prev_row):
                return ""
            prev_value = prev_row[i]
            if value > prev_value:
                return "background-color: rgba(16,185,129,0.28)"  # up since last refresh
            if value < prev_value:
                return "background-color: rgba(239,68,68,0.28)"  # down since last refresh
            return ""

        display_df = pd.DataFrame({"นาทีที่": [f"+{m} นาที" for m in hist_moves.minutes]})
        color_df = pd.DataFrame({"นาทีที่": [""] * len(hist_moves.minutes)})
        sign = {"ขึ้นสูงสุด (ค่ากลาง)": "+", "ขึ้นสูงสุด (กว้าง ~90th)": "+", "ลงสูงสุด (ค่ากลาง)": "", "ลงสูงสุด (กว้าง ~10th)": ""}
        for col_name, pcts in pct_cols.items():
            prices = price_cols[col_name]
            display_df[col_name] = [f"${p:,.2f} ({sign[col_name]}{pct:.2f}%)" for p, pct in zip(prices, pcts)]
            color_df[col_name] = [cell_color(col_name, i, v) for i, v in enumerate(pcts)]

        st.session_state[prev_key] = pct_cols
        styled = display_df.style.apply(lambda _: color_df, axis=None)
        st.dataframe(styled, hide_index=True, width="stretch")
        refresh_note = "ไม่มีการรีเฟรชอัตโนมัติวันนี้ (ตลาดปิด)" if MARKET_CLOSED_TODAY else "เทียบทุก 30 วินาที"
        st.caption(f"🟢 เขียว = ตัวเลขเพิ่มขึ้นจากรอบอัปเดตก่อนหน้า · 🔴 แดง = ลดลงจากรอบก่อนหน้า ({refresh_note})")
        st.caption(
            f"คำนวณจากสถิติย้อนหลัง {hist_moves.days_used} วันทำการล่าสุด ({hist_moves.windows_used} ช่วงเวลาที่นาฬิกาใกล้เคียงเวลาปัจจุบัน ±45 นาที) "
            "โดยแต่ละแถวคือ \"ขึ้น/ลงสูงสุดที่เคยเกิดขึ้นจริงภายใน N นาทีนั้น\" นับจากช่วงเวลาที่ใกล้เคียงกันในอดีต"
        )
        st.caption(
            "⚠️ **ไม่ได้รวมผลจากเนื้อหาข่าว** — ไม่มีวิธีแปลงข่าวเป็นตัวเลขราคาที่แม่นยำได้ ตัวเลขนี้มาจากสถิติย้อนหลังล้วนๆ "
            "เป็นแค่ 'เคยเกิดอะไรได้บ้างในอดีต' ไม่ใช่การรับประกันว่าจะเกิดซ้ำ โปรดอ่านหัวข้อข่าวด้านล่างประกอบการตัดสินใจเอง"
        )


_momentum_and_forecast(stats)

# ---------- News ----------
st.markdown("### ข่าวล่าสุด (วิเคราะห์ผลกระทบ + แปลไทย)")
if not news:
    st.info("ไม่พบข่าวจาก Yahoo Finance สำหรับ ticker นี้ในขณะนี้")
else:
    st.caption(
        "🟢 บวก / 🔴 ลบ / ⚪ เป็นกลาง คือการ**เดาโทนข่าวแบบง่าย**จากการนับคำที่มักปรากฏในข่าวเชิงบวก/ลบ (keyword heuristic) "
        "ไม่ใช่โมเดล AI วิเคราะห์ความรู้สึกจริง พลาดได้ง่ายกับข่าวที่เล่นคำ/แดกดัน ใช้ประกอบการอ่านเท่านั้น ไม่ใช่คำแนะนำซื้อขาย "
        "ส่วนคำแปลไทยใช้บริการแปลฟรี อาจแปลไม่สละสลวยหรือใช้งานไม่ได้ชั่วคราว"
    )
    sentiment_badge = {
        "positive": (":green[🟢 บวกต่อหุ้น (คาดเดา)]"),
        "negative": (":red[🔴 ลบต่อหุ้น (คาดเดา)]"),
        "neutral": (":gray[⚪ เป็นกลาง (คาดเดา)]"),
    }
    summary_preview_len = 80
    for i, item in enumerate(news):
        title = item["title"]
        link = item.get("link")
        summary = item.get("summary", "")
        meta = " · ".join(x for x in [item.get("publisher"), item.get("published")] if x)

        sentiment = analyze_news_sentiment(title, summary)
        st.markdown(f"**{sentiment_badge[sentiment['label']]}**")
        if link:
            st.markdown(f"**[{title}]({link})**")
        else:
            st.markdown(f"**{title}**")
        if meta:
            st.caption(meta)

        title_th = _cached_translate(title)
        summary_th = _cached_translate(summary) if summary else None
        if title_th:
            st.markdown(f"🇹🇭 _{title_th}_")
        else:
            st.caption("แปลไทยไม่สำเร็จตอนนี้ (บริการแปลฟรีอาจติดขัดชั่วคราว)")

        if summary_th:
            preview = summary_th if len(summary_th) <= summary_preview_len else summary_th[:summary_preview_len].rstrip() + " …"
            st.caption(preview)

        if link:
            full_key = f"show_full_article_{i}"
            if st.button("📄 อ่านเนื้อข่าวเต็ม (แปลไทย)", key=f"full_btn_{i}"):
                st.session_state[full_key] = True
            if st.session_state.get(full_key):
                with st.spinner("กำลังดึงและแปลเนื้อข่าวเต็ม..."):
                    full_th = _cached_full_article_th(link)
                if full_th:
                    with st.expander("เนื้อข่าวเต็ม (แปลไทย)", expanded=True):
                        st.write(full_th)
                else:
                    st.warning(
                        "ดึง/แปลเนื้อข่าวเต็มไม่สำเร็จ — อาจเพราะเว็บต้นทางกันการดึงข้อมูล มี paywall "
                        "หรือบริการแปลฟรีติดขัดชั่วคราว ลองกดใหม่อีกครั้ง หรือกดลิงก์หัวข้อข่าวด้านบนเพื่อไปอ่านที่ต้นทาง"
                    )

        st.divider()

# ---------- Disclaimer ----------
st.divider()
st.caption(
    "⚠️ ตัวเลขโอกาสขึ้น/ลงและช่วงราคาคาดการณ์เป็นการประมาณเชิงสถิติจากความผันผวนของราคาในอดีตเท่านั้น "
    "ไม่ใช่การพยากรณ์ที่แม่นยำ ไม่ใช่คำแนะนำการลงทุน และไม่ได้คำนึงถึงข่าวหรือเหตุการณ์ใหม่ที่ยังไม่เกิดขึ้น "
    "ผลตอบแทนในอดีตไม่ได้เป็นเครื่องยืนยันผลในอนาคต โปรดพิจารณาความเสี่ยงด้วยตนเองก่อนตัดสินใจลงทุน"
)
