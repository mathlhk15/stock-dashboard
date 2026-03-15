import datetime
import html
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# =========================================================
# 1. Page Config
# =========================================================
st.set_page_config(
    page_title="실전 투자 리포트",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. Styling
# =========================================================
st.markdown(
    """
<style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }

    .text-gray {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .text-title {
        font-weight: 800;
        font-size: 18px;
        color: #0f172a;
    }

    .evidence-box {
        background-color: #f1f5f9;
        border-left: 3px solid #64748b;
        padding: 10px 14px;
        margin-top: 10px;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
        color: #475569;
        line-height: 1.5;
    }

    .badge-gray {
        background-color: #e2e8f0;
        color: #475569;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
        display: inline-block;
    }

    .reliability {
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 8px;
        font-weight: 700;
    }

    .rel-high {
        background-color: #dcfce7;
        color: #166534;
    }

    .rel-mid {
        background-color: #fef3c7;
        color: #92400e;
    }

    .rel-low {
        background-color: #fee2e2;
        color: #991b1b;
    }

    .zone-box {
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }

    .fallback-warn {
        font-size: 11px;
        color: #b45309;
        margin-top: 8px;
        font-weight: 700;
    }

    .footer-note {
        text-align: right;
        color: #94a3b8;
        font-size: 11px;
        margin-top: 8px;
    }

    .info-note {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
        line-height: 1.5;
    }

    .quality-box {
        background-color: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 8px;
        margin-bottom: 14px;
        color: #475569;
        font-size: 12px;
    }

    .scenario-bull {
        background-color: #ecfdf5;
        border: 1px solid #6ee7b7;
        border-radius: 12px;
        padding: 16px;
        color: #065f46;
        font-size: 13px;
        line-height: 1.6;
        height: 100%;
    }

    .scenario-bear {
        background-color: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 12px;
        padding: 16px;
        color: #991b1b;
        font-size: 13px;
        line-height: 1.6;
        height: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 3. Data Fetch
# =========================================================
@st.cache_data(ttl=900)
def fetch_price_data(ticker_symbol: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker_symbol).history(period="1y", auto_adjust=False)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception as e:
        print(f"[fetch_price_data] {ticker_symbol}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_info_data(ticker_symbol: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(ticker_symbol).info
        return info if isinstance(info, dict) else {}
    except Exception as e:
        print(f"[fetch_info_data] {ticker_symbol}: {e}")
        return {}


# =========================================================
# 4. Input Symbol Resolver
# =========================================================
def normalize_user_input(text: str) -> str:
    return (text or "").strip().upper()


ALIAS_MAP = {
    "TSMC": "TSM",
    "TAIWAN SEMICONDUCTOR": "TSM",
    "MICROSOFT": "MSFT",
    "APPLE": "AAPL",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "BERKSHIRE": "BRK-B",
    "BRK.B": "BRK-B",
    "TSMC ADR": "TSM",
}


@st.cache_data(ttl=3600)
def resolve_symbol(user_input: str) -> Dict[str, Any]:
    raw = normalize_user_input(user_input)

    if not raw:
        return {
            "resolved_symbol": "",
            "display_name": "",
            "method": "failed",
            "candidates": [],
        }

    if raw in ALIAS_MAP:
        resolved = ALIAS_MAP[raw]
        return {
            "resolved_symbol": resolved,
            "display_name": resolved,
            "method": "alias",
            "candidates": [],
        }

    # 티커처럼 보이는 입력 우선 허용
    if any(ch.isdigit() for ch in raw) or "." in raw or "^" in raw or len(raw) <= 5:
        return {
            "resolved_symbol": raw,
            "display_name": raw,
            "method": "direct",
            "candidates": [],
        }

    try:
        search = yf.Search(query=raw, max_results=5, news_count=0)
        quotes = getattr(search, "quotes", []) or []

        candidates = []
        for q in quotes[:5]:
            symbol = q.get("symbol", "")
            shortname = q.get("shortname", "") or q.get("longname", "")
            exch = q.get("exchange", "")
            qtype = q.get("quoteType", "")
            if symbol:
                candidates.append(
                    {
                        "symbol": symbol,
                        "name": shortname,
                        "exchange": exch,
                        "quoteType": qtype,
                    }
                )

        if candidates:
            top = candidates[0]
            return {
                "resolved_symbol": top["symbol"],
                "display_name": top["name"] or top["symbol"],
                "method": "search",
                "candidates": candidates,
            }

    except Exception as e:
        print(f"[resolve_symbol] {raw}: {e}")

    return {
        "resolved_symbol": raw,
        "display_name": raw,
        "method": "direct",
        "candidates": [],
    }


# =========================================================
# 5. Helpers
# =========================================================
def safe_text(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def is_valid_number(value: Any) -> bool:
    return value is not None and not pd.isna(value) and np.isfinite(value)


def fmt_ratio_pct(value: Any) -> str:
    if not is_valid_number(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def fmt_mul(value: Any) -> str:
    if not is_valid_number(value):
        return "N/A"
    return f"{value:.2f}배"


def fmt_price(value: Any) -> str:
    if not is_valid_number(value):
        return "N/A"
    return f"${value:,.2f}"


def fmt_large_dollar(value: Any) -> str:
    if not is_valid_number(value):
        return "N/A"
    v = float(value)
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def fmt_date_from_timestamp(value: Any) -> str:
    if not is_valid_number(value):
        return "N/A"
    try:
        return datetime.datetime.fromtimestamp(int(value)).strftime("%Y.%m.%d")
    except Exception:
        return "N/A"


def reliability_badge(level: str) -> str:
    if level == "high":
        return "<span class='reliability rel-high'>신뢰도: 높음</span>"
    if level == "mid":
        return "<span class='reliability rel-mid'>신뢰도: 보통</span>"
    return "<span class='reliability rel-low'>신뢰도: 낮음</span>"


def get_reliability_by_length(data_len: int, high_cut: int, mid_cut: int) -> str:
    if data_len >= high_cut:
        return "high"
    if data_len >= mid_cut:
        return "mid"
    return "low"


def render_metric_html(
    label: str,
    value: str,
    subvalue: str = "",
    subcolor: str = "#475569",
) -> str:
    sub_html = ""
    if subvalue:
        sub_html = (
            f"<div style='color:{subcolor}; font-size:12px; font-weight:700;'>"
            f"{subvalue}</div>"
        )
    return (
        f"<div class='text-gray'>{label}</div>"
        f"<div style='font-size:22px; font-weight:800;'>{value}</div>"
        f"{sub_html}"
    )


def render_info_card(title: str, value: str, desc: str, badge_html: str = "") -> str:
    return f"""
    <div class='card'>
        <div class='text-gray'>{title} {badge_html}</div>
        <div class='text-title'>{value}</div>
        <div class='evidence-box'>{desc}</div>
    </div>
    """


def validate_ohlc_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = ["Close", "High", "Low", "Volume"]
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing


def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.dropna(how="all")
    return out


def count_info_completeness(info: Dict[str, Any], keys: List[str]) -> Tuple[int, int]:
    total = len(keys)
    filled = 0
    for k in keys:
        v = info.get(k)
        if v is None or v == "":
            continue
        if isinstance(v, (float, np.floating)) and pd.isna(v):
            continue
        filled += 1
    return filled, total


def compute_52w_position(current_price: float, high_52: Any, low_52: Any) -> str:
    if not (is_valid_number(current_price) and is_valid_number(high_52) and is_valid_number(low_52)):
        return "N/A"
    high_52 = float(high_52)
    low_52 = float(low_52)
    if high_52 <= low_52:
        return "N/A"
    pos = (float(current_price) - low_52) / (high_52 - low_52) * 100
    return f"{pos:.1f}%"


def compute_data_quality_summary(df: pd.DataFrame) -> Tuple[int, int]:
    checks = ["MA20", "MA50", "MA200", "RSI14", "ATR14", "MACD", "MACD_SIGNAL"]
    total = len(checks)
    filled = 0
    if len(df) > 0:
        last = df.iloc[-1]
        for col in checks:
            if col in df.columns and is_valid_number(last[col]):
                filled += 1
    return filled, total


# =========================================================
# 6. PBR Module
# =========================================================
def compute_pbr_module(info: Dict[str, Any]) -> Dict[str, Any]:
    current_pbr = info.get("priceToBook")

    result = {
        "current_pbr": current_pbr if is_valid_number(current_pbr) else np.nan,
        "hist_avg_pbr": np.nan,
        "hist_std_pbr": np.nan,
        "pbr_zscore": np.nan,
        "sample_months": np.nan,
        "status": "N/A",
        "note": (
            "과거 평균 PBR 비교는 월별 가격과 분기별 BPS 시계열이 함께 필요합니다. "
            "현재 데이터 소스에서는 역사 BPS 시계열이 안정적으로 확보되지 않아 N/A 처리합니다."
        ),
    }

    if is_valid_number(current_pbr):
        result["status"] = "CURRENT_ONLY"

    return result


# =========================================================
# 7. Asset Classification
# =========================================================
def classify_asset(info: Dict[str, Any]) -> Dict[str, Any]:
    quote_type = str(info.get("quoteType", "") or "").upper()
    sector = str(info.get("sector", "") or "")
    industry = str(info.get("industry", "") or "")

    is_index = quote_type == "INDEX"
    is_etf_like = quote_type in {"ETF", "ETN", "MUTUALFUND"}
    is_financial = sector == "Financial Services"
    is_reit = "REIT" in industry.upper() or "REIT" in sector.upper()

    if is_index:
        badge = "<div class='badge-gray'>시장 지수</div>"
        asset_kind = "index"
    elif is_etf_like:
        badge = "<div class='badge-gray'>ETF/ETN/펀드형 자산</div>"
        asset_kind = "fund"
    elif is_financial and not is_reit:
        badge = "<div class='badge-gray'>금융 섹터</div>"
        asset_kind = "financial"
    elif is_reit:
        badge = "<div class='badge-gray'>REIT</div>"
        asset_kind = "reit"
    else:
        badge = ""
        asset_kind = "equity"

    return {
        "quote_type": quote_type,
        "sector": sector,
        "industry": industry,
        "is_index": is_index,
        "is_etf_like": is_etf_like,
        "is_financial": is_financial and not is_reit,
        "is_reit": is_reit,
        "asset_kind": asset_kind,
        "badge": badge,
    }


# =========================================================
# 8. Indicator Engine
# =========================================================
def calculate_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = clean_price_df(df)

    trend_price_col = "Adj Close" if "Adj Close" in out.columns else "Close"
    display_price_col = "Close" if "Close" in out.columns else trend_price_col

    out["MA20"] = out[trend_price_col].rolling(window=20).mean()
    out["MA50"] = out[trend_price_col].rolling(window=50).mean()
    out["MA200"] = out[trend_price_col].rolling(window=200).mean()

    # Wilder RSI
    delta = out[trend_price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift(1)).abs()
    low_close = (out["Low"] - out["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    # MACD
    ema12 = out[trend_price_col].ewm(span=12, adjust=False).mean()
    ema26 = out[trend_price_col].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    # MDD
    roll_max = out[trend_price_col].cummax()
    drawdown = out[trend_price_col] / roll_max - 1.0
    mdd = drawdown.min() * 100 if len(drawdown.dropna()) > 0 else np.nan

    high_52 = out["High"].max() if "High" in out.columns else np.nan
    low_52 = out["Low"].min() if "Low" in out.columns else np.nan

    # volume
    out["VOL20"] = out["Volume"].rolling(window=20).mean()

    meta = {
        "trend_price_col": trend_price_col,
        "display_price_col": display_price_col,
        "data_len": len(out),
        "mdd": mdd,
        "high_52": high_52,
        "low_52": low_52,
    }
    return out, meta


# =========================================================
# 9. Support / Resistance / Strategy / Scenario
# =========================================================
def build_action_zones(
    current_price: float,
    ma50: Any,
    ma200: Any,
    atr: Any,
) -> Dict[str, Any]:
    fallback_used = False
    fallback_reasons = []

    atr_val = atr
    if not is_valid_number(atr_val) or float(atr_val) <= 0:
        atr_val = current_price * 0.03
        fallback_used = True
        fallback_reasons.append("ATR")

    ma50_val = ma50
    if not is_valid_number(ma50_val):
        ma50_val = current_price * 0.95
        fallback_used = True
        fallback_reasons.append("MA50")

    ma200_val = ma200
    if not is_valid_number(ma200_val):
        ma200_val = current_price * 0.85
        fallback_used = True
        fallback_reasons.append("MA200")

    zone1_low = max(ma50_val, current_price - 1.5 * atr_val)
    zone1_high = current_price
    zone1_low, zone1_high = sorted([zone1_low, zone1_high])

    zone2_low = max(ma200_val, current_price - 3.0 * atr_val)
    zone2_high = min(zone1_low * 0.995, current_price)

    if zone2_high - zone2_low < current_price * 0.005:
        zone2_high = zone2_low + current_price * 0.005

    zone2_low, zone2_high = sorted([zone2_low, zone2_high])

    gap_to_zone1 = ((current_price - zone1_low) / current_price * 100) if current_price != 0 else np.nan

    return {
        "zone1_low": zone1_low,
        "zone1_high": zone1_high,
        "zone2_low": zone2_low,
        "zone2_high": zone2_high,
        "fallback_used": fallback_used,
        "fallback_reasons": fallback_reasons,
        "gap_to_zone1": gap_to_zone1,
    }


def build_support_resistance(
    current_price: float,
    atr: Any,
    ma20: Any,
    ma50: Any,
    high_52: Any,
) -> Dict[str, Any]:
    atr_val = atr if is_valid_number(atr) and float(atr) > 0 else current_price * 0.03
    ma20_val = ma20 if is_valid_number(ma20) else current_price * 0.97
    ma50_val = ma50 if is_valid_number(ma50) else current_price * 0.92

    support_1 = min(ma20_val, current_price - 1.0 * atr_val)
    support_2 = min(ma50_val, current_price - 2.0 * atr_val)
    support_1, support_2 = sorted([support_1, support_2], reverse=True)

    resistance_1 = current_price + 1.0 * atr_val
    resistance_2 = current_price + 2.0 * atr_val

    breakout = high_52 if is_valid_number(high_52) else np.nan

    return {
        "support_1": support_1,
        "support_2": support_2,
        "resistance_1": resistance_1,
        "resistance_2": resistance_2,
        "breakout": breakout,
    }


def build_volume_comment(current_vol: Any, avg20_vol: Any) -> str:
    if not (is_valid_number(current_vol) and is_valid_number(avg20_vol) and avg20_vol > 0):
        return "거래량 데이터가 충분하지 않아 강도 판단은 제한됩니다."
    ratio = float(current_vol) / float(avg20_vol)
    if ratio >= 1.5:
        return "최근 거래량이 20일 평균을 뚜렷하게 상회해 수급 반응이 강한 편으로 볼 수 있습니다."
    if ratio >= 1.0:
        return "거래량이 최근 평균 수준 이상으로 유지되어 가격 움직임의 확인 신호로 해석할 수 있습니다."
    return "거래량이 20일 평균을 밑돌아 가격 움직임의 신뢰도는 다소 약할 수 있습니다."


def build_macd_comment(macd: Any, signal: Any, hist: Any) -> str:
    if not (is_valid_number(macd) and is_valid_number(signal) and is_valid_number(hist)):
        return "MACD 데이터가 충분하지 않아 모멘텀 해석은 제한됩니다."
    if macd > signal and hist > 0:
        return "MACD가 시그널선 위에 있고 히스토그램도 양수라 모멘텀 우위가 유지되는 구간으로 볼 수 있습니다."
    if macd < signal and hist < 0:
        return "MACD가 시그널선 아래에 있고 히스토그램도 음수라 단기 모멘텀은 약한 편으로 해석됩니다."
    return "MACD와 시그널선의 간격이 크지 않아 방향성이 뚜렷하지 않은 중립 구간으로 볼 수 있습니다."


def build_strategy_text(
    current_price: float,
    zone1_low: float,
    zone2_low: float,
    ma200: Any,
    rsi: Any,
) -> Tuple[str, str]:
    if is_valid_number(rsi) and rsi < 35:
        short_term = (
            f"단기적으로는 RSI가 과매도 인근에 있어 반등 시도가 나올 수 있으나, "
            f"{fmt_price(zone1_low)} 부근 지지 확인 여부를 먼저 보는 접근이 적절합니다."
        )
    else:
        short_term = (
            f"단기적으로는 현재가 추격보다 {fmt_price(zone1_low)} 부근 눌림목 형성 여부를 관찰하는 접근이 더 보수적입니다."
        )

    if is_valid_number(ma200) and current_price > ma200:
        mid_term = (
            f"중기적으로는 200일선 위 추세가 유지되는 동안 {fmt_price(zone2_low)}~{fmt_price(zone1_low)} 구간을 분할 관찰 구간으로 볼 수 있습니다."
        )
    else:
        mid_term = (
            f"중기적으로는 200일선 회복 전까지는 공격적 비중 확대보다 {fmt_price(zone2_low)} 부근 방어력 확인이 우선입니다."
        )

    return short_term, mid_term


def build_scenarios(
    current_price: float,
    target_price: Any,
    ma200: Any,
    support_1: float,
    support_2: float,
    resistance_1: float,
    macd: Any,
    signal: Any,
) -> Tuple[str, str]:
    bull_target = fmt_price(target_price) if is_valid_number(target_price) else fmt_price(resistance_1)
    bull = (
        f"추세가 200일선 위에서 유지되고 {fmt_price(support_1)} 지지가 확인되면, "
        f"단기적으로는 {fmt_price(resistance_1)} 테스트, 확장 시 {bull_target} 구간까지 열어둘 수 있습니다."
    )

    if is_valid_number(macd) and is_valid_number(signal) and macd < signal:
        bear = (
            f"모멘텀이 약한 상태에서 {fmt_price(support_1)} 이탈이 나오면 "
            f"{fmt_price(support_2)} 구간까지 조정 폭이 확대될 가능성을 열어둘 필요가 있습니다."
        )
    else:
        bear = (
            f"단기 지지선인 {fmt_price(support_1)} 이탈 시에는 "
            f"{fmt_price(support_2)} 부근 재점검 가능성을 염두에 두는 편이 보수적입니다."
        )

    return bull, bear


# =========================================================
# 10. App Header
# =========================================================
st.caption(
    f"<span style='color:#64748b;'>실전 투자 리포트 · {datetime.datetime.now().strftime('%Y.%m.%d')}</span>",
    unsafe_allow_html=True,
)

user_input_symbol = st.text_input(
    "종목 코드 또는 회사명을 입력하세요 (예: MSFT, TSMC, NVIDIA, QQQ, JPM, ^GSPC)",
    "MSFT",
).strip()


# =========================================================
# 11. Main UI
# =========================================================
if user_input_symbol:
    with st.spinner("데이터를 분석 중입니다..."):
        resolved = resolve_symbol(user_input_symbol)
        ticker = resolved["resolved_symbol"]

        if not ticker:
            st.error("입력값을 해석하지 못했습니다.")
            st.stop()

        price_df = fetch_price_data(ticker)
        info = fetch_info_data(ticker)

        if price_df.empty or len(price_df) < 2:
            st.error("데이터가 부족하거나 유효하지 않은 티커입니다. 최소 2일 이상의 가격 데이터가 필요합니다.")
            st.stop()

        valid_ohlc, missing_cols = validate_ohlc_columns(price_df)
        if not valid_ohlc:
            st.error(f"가격 데이터에 필요한 컬럼이 부족합니다: {', '.join(missing_cols)}")
            st.stop()

        if not isinstance(info, dict):
            info = {}

        df, meta = calculate_indicators(price_df)
        asset = classify_asset(info)
        pbr_module = compute_pbr_module(info)

        if len(df) < 2:
            st.error("정제 후 사용할 수 있는 가격 데이터가 충분하지 않습니다.")
            st.stop()

        trend_price_col = meta["trend_price_col"]
        display_price_col = meta["display_price_col"]
        data_len = meta["data_len"]
        mdd = meta["mdd"]
        high_52 = meta["high_52"]
        low_52 = meta["low_52"]

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        current_price = last_row[display_price_col]
        prev_price = prev_row[display_price_col]
        price_change = current_price - prev_price
        pct_change = (price_change / prev_price) * 100 if prev_price != 0 else 0.0

        short_name = safe_text(info.get("shortName", ticker))
        sector = safe_text(info.get("sector", ""))
        quote_type = safe_text(info.get("quoteType", ""))
        industry = safe_text(info.get("industry", ""))
        fund_family = safe_text(info.get("fundFamily", ""))
        category = safe_text(info.get("category", ""))

        if asset["is_index"]:
            subtitle = "시장 지수"
            info_keys_to_check = ["previousClose", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]
        elif asset["is_etf_like"]:
            if fund_family and category:
                subtitle = f"{fund_family} | {category}"
            elif fund_family:
                subtitle = fund_family
            elif category:
                subtitle = category
            else:
                subtitle = quote_type if quote_type else "펀드형 자산"
            info_keys_to_check = ["yield", "ytdReturn", "totalAssets", "category", "fundFamily"]
        else:
            if sector and industry:
                subtitle = f"{sector} | {industry}"
            elif sector:
                subtitle = sector
            elif quote_type:
                subtitle = quote_type
            else:
                subtitle = "분류 정보 없음"

            if asset["is_financial"]:
                info_keys_to_check = ["targetMeanPrice", "earningsTimestamp", "returnOnEquity", "priceToBook", "forwardPE"]
            elif asset["is_reit"]:
                info_keys_to_check = ["targetMeanPrice", "earningsTimestamp", "yield", "forwardPE", "priceToBook"]
            else:
                info_keys_to_check = ["targetMeanPrice", "earningsTimestamp", "revenueGrowth", "operatingMargins", "forwardPE", "priceToBook"]

        filled_count, total_count = count_info_completeness(info, info_keys_to_check)
        indicator_filled, indicator_total = compute_data_quality_summary(df)

        if resolved["method"] == "alias":
            st.markdown(
                f"<div class='info-note'>입력값 <b>{safe_text(user_input_symbol)}</b> 를 별칭으로 해석하여 "
                f"<b>{safe_text(ticker)}</b> 로 조회했습니다.</div>",
                unsafe_allow_html=True,
            )
        elif resolved["method"] == "search":
            st.markdown(
                f"<div class='info-note'>입력값 <b>{safe_text(user_input_symbol)}</b> 를 검색하여 "
                f"<b>{safe_text(ticker)}</b> 로 조회했습니다.</div>",
                unsafe_allow_html=True,
            )

        st.markdown(asset["badge"], unsafe_allow_html=True)
        st.markdown(
            f"<h1 style='margin-bottom:0; color:#0f172a;'>{short_name} "
            f"<span style='color:#64748b; font-size:20px;'>· {safe_text(ticker)}</span></h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color:#64748b; font-size:13px; margin-top:4px;'>{subtitle} | "
            f"상단 가격 기준: {display_price_col} | 추세/MDD 기준: {trend_price_col}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='quality-box'>메타데이터 완전성: <b>{filled_count}/{total_count}</b> | "
            f"기술지표 계산 가능: <b>{indicator_filled}/{indicator_total}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin: 12px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

        # ---------------- Top Summary ----------------
        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(
            render_metric_html(
                label=f"종가 ({display_price_col})",
                value=fmt_price(current_price),
                subvalue=f"{pct_change:+.2f}%",
                subcolor="#dc2626" if price_change < 0 else "#059669",
            ),
            unsafe_allow_html=True,
        )

        if asset["is_index"]:
            c2.markdown(render_metric_html("52주 고가", fmt_price(high_52)), unsafe_allow_html=True)
            c3.markdown(render_metric_html("52주 저가", fmt_price(low_52)), unsafe_allow_html=True)
            c4.markdown(render_metric_html("52주 위치", compute_52w_position(current_price, high_52, low_52)), unsafe_allow_html=True)

        elif asset["is_etf_like"]:
            c2.markdown(render_metric_html("분배금 수익률", fmt_ratio_pct(info.get("yield"))), unsafe_allow_html=True)
            c3.markdown(render_metric_html("YTD 수익률", fmt_ratio_pct(info.get("ytdReturn"))), unsafe_allow_html=True)
            c4.markdown(render_metric_html("운용자산(AUM)", fmt_large_dollar(info.get("totalAssets"))), unsafe_allow_html=True)

        else:
            target_price = info.get("targetMeanPrice")
            if is_valid_number(target_price) and current_price != 0:
                upside = ((target_price - current_price) / current_price) * 100
                c2.markdown(
                    render_metric_html(
                        "애널리스트 평균 목표가",
                        fmt_price(target_price),
                        subvalue=f"{upside:+.1f}%",
                        subcolor="#059669" if upside >= 0 else "#dc2626",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                c2.markdown(render_metric_html("애널리스트 평균 목표가", "N/A"), unsafe_allow_html=True)

            if asset["is_financial"]:
                c3.markdown(render_metric_html("자기자본이익률(ROE)", fmt_ratio_pct(info.get("returnOnEquity"))), unsafe_allow_html=True)
            elif asset["is_reit"]:
                c3.markdown(render_metric_html("분배금 수익률", fmt_ratio_pct(info.get("yield"))), unsafe_allow_html=True)
            else:
                c3.markdown(render_metric_html("매출 성장률(YoY)", fmt_ratio_pct(info.get("revenueGrowth"))), unsafe_allow_html=True)

            c4.markdown(render_metric_html("예상 실적 일정", fmt_date_from_timestamp(info.get("earningsTimestamp"))), unsafe_allow_html=True)

        st.markdown(
            "<div class='info-note'>기업/펀드 메타데이터 값은 종목별로 누락되거나 지연될 수 있습니다.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------- Fundamentals / Risk ----------------
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            if asset["is_index"]:
                st.markdown(
                    render_info_card(
                        "자산 유형 안내",
                        "지수(Index)",
                        "<b>설명:</b> 지수는 기업 펀더멘털이나 펀드 운용자산 개념이 직접 대응되지 않을 수 있습니다. 가격 추세와 변동성 해석 중심으로 보는 편이 적절합니다.",
                    ),
                    unsafe_allow_html=True,
                )
            elif asset["is_etf_like"]:
                st.markdown(
                    render_info_card(
                        "펀드형 자산 안내",
                        "ETF/ETN/펀드",
                        "<b>설명:</b> 개별 기업용 PER/OPM보다 기초지수, 자금 규모, 분배금, 추세·변동성을 함께 보는 편이 적절합니다.",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                forward_pe = fmt_mul(info.get("forwardPE"))

                if asset["is_financial"]:
                    price_to_book = fmt_mul(info.get("priceToBook"))
                    st.markdown(
                        render_info_card(
                            "가치 평가 지표",
                            f"선행 P/E: {forward_pe} / P/B: {price_to_book}",
                            "<b>해석:</b> 금융주는 일반 제조업처럼 영업이익률보다 자산 건전성과 자본효율 관점의 P/B, ROE를 함께 참고하는 편이 적절합니다.",
                        ),
                        unsafe_allow_html=True,
                    )
                elif asset["is_reit"]:
                    price_to_book = fmt_mul(info.get("priceToBook"))
                    st.markdown(
                        render_info_card(
                            "REIT 해석 안내",
                            f"P/B: {price_to_book}",
                            "<b>해석:</b> REIT는 일반 기업과 동일한 OPM 중심 해석보다 배당, 자산 가치, 금리 민감도까지 함께 보는 편이 적절합니다.",
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    operating_margin = fmt_ratio_pct(info.get("operatingMargins"))
                    st.markdown(
                        render_info_card(
                            "가치 평가 지표",
                            f"선행 P/E: {forward_pe} / OPM: {operating_margin}",
                            "<b>해석:</b> 영업이익률(OPM)은 업종 평균과 비교할 때 해석력이 높아집니다. 단일 수치만으로 경쟁우위를 단정하기보다 비교 기준과 함께 보는 편이 적절합니다.",
                        ),
                        unsafe_allow_html=True,
                    )

        with col_f2:
            mdd_level = get_reliability_by_length(data_len, high_cut=252, mid_cut=120)
            mdd_rel = reliability_badge(mdd_level)

            if data_len >= 252:
                mdd_period = "최근 1년"
            elif data_len >= 120:
                mdd_period = "수집된 데이터 구간"
            else:
                mdd_period = "제한된 데이터 구간"

            mdd_text = f"{mdd:.1f}%" if is_valid_number(mdd) else "N/A"

            st.markdown(
                render_info_card(
                    f"최대낙폭(MDD) {mdd_rel}",
                    mdd_text,
                    f"<b>기준:</b> {mdd_period} 내 고점 대비 최대 하락 비율입니다. 체감 변동성 수준을 보는 참고 지표입니다.",
                ),
                unsafe_allow_html=True,
            )

        # ---------------- PBR Card ----------------
        if not asset["is_index"] and not asset["is_etf_like"]:
            current_pbr_text = fmt_mul(pbr_module["current_pbr"])

            if pbr_module["status"] == "CURRENT_ONLY":
                pbr_value = f"현재 PBR: {current_pbr_text}"
                pbr_desc = (
                    "<b>상태:</b> 현재 PBR은 확인되지만, 과거 평균 PBR 대비 수준은 N/A입니다. "
                    "정확한 역사 평균 비교를 위해서는 월별 가격과 분기별 BPS 시계열이 함께 필요합니다."
                )
            else:
                pbr_value = "N/A"
                pbr_desc = pbr_module["note"]

            st.markdown(
                render_info_card(
                    "PBR 상대 수준",
                    pbr_value,
                    pbr_desc,
                ),
                unsafe_allow_html=True,
            )

        # ---------------- Technicals ----------------
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            ma200 = last_row["MA200"]
            ma200_level = get_reliability_by_length(data_len, high_cut=200, mid_cut=120)
            ma200_rel = reliability_badge(ma200_level)

            if not is_valid_number(ma200):
                trend_title = "판단 불가"
                trend_desc = "상장 이력 또는 데이터 길이가 부족해 200일 이동평균 계산이 제한됩니다."
            else:
                trend_current = last_row[trend_price_col]
                is_bull = trend_current > ma200
                trend_title = "200일선 상회" if is_bull else "200일선 하회"
                trend_desc = f"추세 기준 가격이 200일선({fmt_price(ma200)}) {'위에' if is_bull else '아래에'} 위치해 있습니다."

            st.markdown(
                render_info_card(
                    f"장기 추세(MA200) {ma200_rel}",
                    trend_title,
                    f"<b>설명:</b> {trend_desc} 장기 방향성을 볼 때 자주 참고하는 기준선입니다.",
                ),
                unsafe_allow_html=True,
            )

        with col_t2:
            rsi = last_row["RSI14"]
            rsi_level = get_reliability_by_length(data_len, high_cut=60, mid_cut=14)
            rsi_rel = reliability_badge(rsi_level)

            if not is_valid_number(rsi):
                rsi_title = "판단 불가"
                rsi_desc = "데이터가 부족해 RSI 계산이 제한됩니다."
            else:
                if rsi < 30:
                    rsi_title = f"{rsi:.1f} (과매도 범위)"
                elif rsi > 70:
                    rsi_title = f"{rsi:.1f} (과매수 범위)"
                else:
                    rsi_title = f"{rsi:.1f} (중립 범위)"
                rsi_desc = "Wilder 방식 RSI(14) 기준이며, 단독으로 반등·하락을 단정하기보다 추세와 함께 해석하는 편이 적절합니다."

            st.markdown(
                render_info_card(
                    f"RSI(14일) {rsi_rel}",
                    rsi_title,
                    f"<b>설명:</b> {rsi_desc}",
                ),
                unsafe_allow_html=True,
            )

        # ---------------- Momentum / Volume ----------------
        col_m1, col_m2 = st.columns(2)

        macd_comment = build_macd_comment(last_row["MACD"], last_row["MACD_SIGNAL"], last_row["MACD_HIST"])
        volume_comment = build_volume_comment(last_row["Volume"], last_row["VOL20"])

        with col_m1:
            if is_valid_number(last_row["MACD"]) and is_valid_number(last_row["MACD_SIGNAL"]):
                macd_value = f"MACD: {last_row['MACD']:.2f} / Signal: {last_row['MACD_SIGNAL']:.2f}"
            else:
                macd_value = "N/A"
            st.markdown(
                render_info_card(
                    "모멘텀 지표",
                    macd_value,
                    f"<b>설명:</b> {macd_comment}",
                ),
                unsafe_allow_html=True,
            )

        with col_m2:
            vol_ratio = (last_row["Volume"] / last_row["VOL20"]) if is_valid_number(last_row["VOL20"]) and last_row["VOL20"] > 0 else np.nan
            vol_value = f"현재/20일 평균: {vol_ratio:.2f}배" if is_valid_number(vol_ratio) else "N/A"
            st.markdown(
                render_info_card(
                    "거래량 평가",
                    vol_value,
                    f"<b>설명:</b> {volume_comment}",
                ),
                unsafe_allow_html=True,
            )

        # ---------------- Support / Resistance ----------------
        sr = build_support_resistance(
            current_price=float(last_row[trend_price_col]),
            atr=last_row["ATR14"],
            ma20=last_row["MA20"],
            ma50=last_row["MA50"],
            high_52=high_52,
        )

        st.markdown(
            "<div style='font-size:16px; font-weight:800; color:#0f172a; margin-top:8px; margin-bottom:12px;'>"
            "📌 주요 지지 / 저항 레벨</div>",
            unsafe_allow_html=True,
        )

        s1, s2, r1, r2, br = st.columns(5)
        s1.markdown(render_metric_html("1차 지지", fmt_price(sr["support_1"])), unsafe_allow_html=True)
        s2.markdown(render_metric_html("2차 지지", fmt_price(sr["support_2"])), unsafe_allow_html=True)
        r1.markdown(render_metric_html("1차 저항", fmt_price(sr["resistance_1"])), unsafe_allow_html=True)
        r2.markdown(render_metric_html("2차 저항", fmt_price(sr["resistance_2"])), unsafe_allow_html=True)
        br.markdown(render_metric_html("52주 고가", fmt_price(sr["breakout"])), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------- Action Zones ----------------
        st.markdown(
            "<div style='font-size:16px; font-weight:800; color:#0f172a; margin-top:8px; margin-bottom:12px;'>"
            "🎯 기계적 관심 구간 (참고용)</div>",
            unsafe_allow_html=True,
        )

        zones = build_action_zones(
            current_price=float(last_row[trend_price_col]),
            ma50=last_row["MA50"],
            ma200=last_row["MA200"],
            atr=last_row["ATR14"],
        )

        gap_to_zone1_text = (
            f"현재가 대비 1차 하단까지 {zones['gap_to_zone1']:.2f}%"
            if is_valid_number(zones["gap_to_zone1"])
            else "N/A"
        )

        z1, z2 = st.columns(2)
        with z1:
            st.markdown(
                f"""
                <div class='zone-box' style='background-color: #f0fdf4; border: 1px solid #bbf7d0;'>
                    <div style='color:#166534; font-size:13px; font-weight:800;'>1차 밴드 (MA50 & ATR 기반)</div>
                    <div style='color:#15803d; font-weight:900; font-size:20px; margin-top:6px;'>
                        {fmt_price(zones["zone1_low"])} ~ {fmt_price(zones["zone1_high"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with z2:
            st.markdown(
                f"""
                <div class='zone-box' style='background-color: #eff6ff; border: 1px solid #93c5fd;'>
                    <div style='color:#1e40af; font-size:13px; font-weight:800;'>2차 밴드 (MA200 & ATR 기반)</div>
                    <div style='color:#1d4ed8; font-weight:900; font-size:20px; margin-top:6px;'>
                        {fmt_price(zones["zone2_low"])} ~ {fmt_price(zones["zone2_high"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        fallback_msg = ""
        if zones["fallback_used"]:
            reasons = ", ".join(zones["fallback_reasons"])
            fallback_msg = (
                f"<div class='fallback-warn'>"
                f"⚠️ 일부 지표({reasons})가 부족하여 보정값이 사용되었습니다. "
                f"이 경우 관심 구간 신뢰도는 낮아집니다."
                f"</div>"
            )

        st.markdown(
            f"""
            <div class='info-note'>
                {gap_to_zone1_text} | 이 밴드는 이동평균과 ATR을 조합한 기계적 참고 범위이며, 거래량 집중대·수평 지지선·뉴스 이벤트는 반영하지 않습니다.
            </div>
            {fallback_msg}
            """,
            unsafe_allow_html=True,
        )

        # ---------------- Strategy ----------------
        short_strategy, mid_strategy = build_strategy_text(
            current_price=float(last_row[trend_price_col]),
            zone1_low=zones["zone1_low"],
            zone2_low=zones["zone2_low"],
            ma200=last_row["MA200"],
            rsi=last_row["RSI14"],
        )

        st.markdown(
            "<div style='font-size:16px; font-weight:800; color:#0f172a; margin-top:10px; margin-bottom:12px;'>"
            "🧭 투자 전략</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            render_info_card(
                "단기 전략",
                "1~2주",
                f"<b>전략:</b> {short_strategy}",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            render_info_card(
                "중기 전략",
                "1~3개월",
                f"<b>전략:</b> {mid_strategy}",
            ),
            unsafe_allow_html=True,
        )

        # ---------------- Scenario ----------------
        bull_scenario, bear_scenario = build_scenarios(
            current_price=float(last_row[trend_price_col]),
            target_price=info.get("targetMeanPrice"),
            ma200=last_row["MA200"],
            support_1=sr["support_1"],
            support_2=sr["support_2"],
            resistance_1=sr["resistance_1"],
            macd=last_row["MACD"],
            signal=last_row["MACD_SIGNAL"],
        )

        st.markdown(
            "<div style='font-size:16px; font-weight:800; color:#0f172a; margin-top:10px; margin-bottom:12px;'>"
            "⚖️ 시나리오 분석</div>",
            unsafe_allow_html=True,
        )

        sb, srisk = st.columns(2)
        with sb:
            st.markdown(
                f"""
                <div class='scenario-bull'>
                    <div style='font-weight:900; font-size:15px; margin-bottom:8px;'>강세 시나리오</div>
                    <div>{bull_scenario}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with srisk:
            st.markdown(
                f"""
                <div class='scenario-bear'>
                    <div style='font-weight:900; font-size:15px; margin-bottom:8px;'>약세 시나리오</div>
                    <div>{bear_scenario}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='footer-note'>데이터 출처: Yahoo Finance | 본 화면은 정보 제공용이며, 구간 값과 해석은 참고용입니다.</div>",
            unsafe_allow_html=True,
        )