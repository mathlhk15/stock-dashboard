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
    .stApp { background-color: #f8fafc; color: #0f172a; }
    .card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .text-gray { color: #64748b; font-size: 12px; font-weight: 700; margin-bottom: 4px; }
    .text-title { font-weight: 800; font-size: 18px; color: #0f172a; }
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
        background-color: #e2e8f0; color: #475569;
        padding: 4px 10px; border-radius: 6px;
        font-size: 12px; font-weight: 700;
        margin-bottom: 10px; display: inline-block;
    }
    .reliability {
        font-size: 11px; padding: 2px 6px;
        border-radius: 4px; margin-left: 8px; font-weight: 700;
    }
    .rel-high { background-color: #dcfce7; color: #166534; }
    .rel-mid  { background-color: #fef3c7; color: #92400e; }
    .rel-low  { background-color: #fee2e2; color: #991b1b; }
    .zone-box { border-radius: 10px; padding: 14px; text-align: center; }
    .fallback-warn { font-size: 11px; color: #b45309; margin-top: 8px; font-weight: 700; }
    .footer-note { text-align: right; color: #94a3b8; font-size: 11px; margin-top: 8px; }
    .info-note { font-size: 11px; color: #64748b; margin-top: 6px; line-height: 1.5; }
    .quality-box {
        background-color: #ffffff; border: 1px dashed #cbd5e1;
        border-radius: 10px; padding: 10px 12px;
        margin-top: 8px; margin-bottom: 14px;
        color: #475569; font-size: 12px;
    }
    .scenario-bull {
        background-color: #ecfdf5; border: 1px solid #6ee7b7;
        border-radius: 12px; padding: 16px;
        color: #065f46; font-size: 13px; line-height: 1.6; height: 100%;
    }
    .scenario-bear {
        background-color: #fef2f2; border: 1px solid #fca5a5;
        border-radius: 12px; padding: 16px;
        color: #991b1b; font-size: 13px; line-height: 1.6; height: 100%;
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
        df = yf.Ticker(ticker_symbol).history(period="2y", auto_adjust=False)
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
    "TSMC": "TSM", "TAIWAN SEMICONDUCTOR": "TSM",
    "MICROSOFT": "MSFT", "APPLE": "AAPL", "NVIDIA": "NVDA",
    "TESLA": "TSLA", "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
    "AMAZON": "AMZN", "META": "META", "FACEBOOK": "META",
    "BERKSHIRE": "BRK-B", "BRK.B": "BRK-B", "TSMC ADR": "TSM",
}


@st.cache_data(ttl=3600)
def resolve_symbol(user_input: str) -> Dict[str, Any]:
    raw = normalize_user_input(user_input)
    if not raw:
        return {"resolved_symbol": "", "display_name": "", "method": "failed", "candidates": []}

    if raw in ALIAS_MAP:
        resolved = ALIAS_MAP[raw]
        return {"resolved_symbol": resolved, "display_name": resolved, "method": "alias", "candidates": []}

    if any(ch.isdigit() for ch in raw) or "." in raw or "^" in raw or len(raw) <= 5:
        return {"resolved_symbol": raw, "display_name": raw, "method": "direct", "candidates": []}

    try:
        search = yf.Search(query=raw, max_results=5, news_count=0)
        quotes = getattr(search, "quotes", []) or []
        candidates = []
        for q in quotes[:5]:
            symbol = q.get("symbol", "")
            shortname = q.get("shortname", "") or q.get("longname", "")
            if symbol:
                candidates.append({"symbol": symbol, "name": shortname,
                                    "exchange": q.get("exchange", ""), "quoteType": q.get("quoteType", "")})
        if candidates:
            top = candidates[0]
            return {"resolved_symbol": top["symbol"], "display_name": top["name"] or top["symbol"],
                    "method": "search", "candidates": candidates}
    except Exception as e:
        print(f"[resolve_symbol] {raw}: {e}")

    return {"resolved_symbol": raw, "display_name": raw, "method": "direct", "candidates": []}


# =========================================================
# 5. Helpers
# =========================================================
def safe_text(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""

def is_valid_number(value: Any) -> bool:
    return value is not None and not pd.isna(value) and np.isfinite(value)

def fmt_ratio_pct(value: Any) -> str:
    return f"{value * 100:.1f}%" if is_valid_number(value) else "N/A"

def fmt_mul(value: Any) -> str:
    return f"{value:.2f}배" if is_valid_number(value) else "N/A"

def fmt_price(value: Any) -> str:
    return f"${value:,.2f}" if is_valid_number(value) else "N/A"

def fmt_large_dollar(value: Any) -> str:
    if not is_valid_number(value): return "N/A"
    v = float(value)
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def fmt_date_from_timestamp(value: Any) -> str:
    if not is_valid_number(value): return "N/A"
    try:
        return datetime.datetime.fromtimestamp(int(value)).strftime("%Y.%m.%d")
    except Exception:
        return "N/A"

def fmt_pct_signed(value: Any) -> str:
    if not is_valid_number(value): return "N/A"
    arrow = "▲" if value >= 0 else "▼"
    color = "#16a34a" if value >= 0 else "#dc2626"
    return f'<span style="color:{color};font-weight:700;">{arrow} {abs(value):.2f}%</span>'

def reliability_badge(level: str) -> str:
    if level == "high": return "<span class='reliability rel-high'>신뢰도: 높음</span>"
    if level == "mid":  return "<span class='reliability rel-mid'>신뢰도: 보통</span>"
    return "<span class='reliability rel-low'>신뢰도: 낮음</span>"

def get_reliability_by_length(data_len: int, high_cut: int, mid_cut: int) -> str:
    if data_len >= high_cut: return "high"
    if data_len >= mid_cut:  return "mid"
    return "low"

def render_metric_html(label, value, subvalue="", subcolor="#475569") -> str:
    sub_html = f"<div style='color:{subcolor}; font-size:12px; font-weight:700;'>{subvalue}</div>" if subvalue else ""
    return f"<div class='text-gray'>{label}</div><div style='font-size:22px; font-weight:800;'>{value}</div>{sub_html}"

def render_info_card(title, value, desc, badge_html="") -> str:
    return f"""
    <div class='card'>
        <div class='text-gray'>{title} {badge_html}</div>
        <div class='text-title'>{value}</div>
        <div class='evidence-box'>{desc}</div>
    </div>
    """

def validate_ohlc_columns(df):
    required = ["Close", "High", "Low", "Volume"]
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing

def clean_price_df(df):
    out = df.copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.dropna(how="all")
    return out

def count_info_completeness(info, keys):
    total = len(keys)
    filled = sum(1 for k in keys if info.get(k) not in [None, ""] and
                 not (isinstance(info.get(k), (float, np.floating)) and pd.isna(info.get(k))))
    return filled, total

def compute_52w_position(current_price, high_52, low_52) -> str:
    if not (is_valid_number(current_price) and is_valid_number(high_52) and is_valid_number(low_52)):
        return "N/A"
    h, l = float(high_52), float(low_52)
    if h <= l: return "N/A"
    pos = (float(current_price) - l) / (h - l) * 100
    return f"{pos:.1f}%"

def compute_data_quality_summary(df):
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
def compute_pbr_module(info):
    current_pbr = info.get("priceToBook")
    result = {
        "current_pbr": current_pbr if is_valid_number(current_pbr) else np.nan,
        "hist_avg_pbr": np.nan, "hist_std_pbr": np.nan,
        "pbr_zscore": np.nan, "sample_months": np.nan,
        "status": "N/A",
        "note": "과거 평균 PBR 비교는 월별 가격과 분기별 BPS 시계열이 함께 필요합니다.",
    }
    if is_valid_number(current_pbr):
        result["status"] = "CURRENT_ONLY"
    return result


# =========================================================
# 7. Asset Classification
# =========================================================
def classify_asset(info):
    quote_type = str(info.get("quoteType", "") or "").upper()
    sector = str(info.get("sector", "") or "")
    industry = str(info.get("industry", "") or "")
    is_index = quote_type == "INDEX"
    is_etf_like = quote_type in {"ETF", "ETN", "MUTUALFUND"}
    is_financial = sector == "Financial Services"
    is_reit = "REIT" in industry.upper() or "REIT" in sector.upper()

    if is_index:    badge = "<div class='badge-gray'>시장 지수</div>"; asset_kind = "index"
    elif is_etf_like: badge = "<div class='badge-gray'>ETF/ETN/펀드형 자산</div>"; asset_kind = "fund"
    elif is_financial and not is_reit: badge = "<div class='badge-gray'>금융 섹터</div>"; asset_kind = "financial"
    elif is_reit:   badge = "<div class='badge-gray'>REIT</div>"; asset_kind = "reit"
    else:           badge = ""; asset_kind = "equity"

    return {"quote_type": quote_type, "sector": sector, "industry": industry,
            "is_index": is_index, "is_etf_like": is_etf_like,
            "is_financial": is_financial and not is_reit, "is_reit": is_reit,
            "asset_kind": asset_kind, "badge": badge}


# =========================================================
# 8. Indicator Engine (볼린저밴드 추가)
# =========================================================
def calculate_indicators(df):
    out = clean_price_df(df)
    trend_col = "Adj Close" if "Adj Close" in out.columns else "Close"
    display_col = "Close" if "Close" in out.columns else trend_col

    out["MA20"]  = out[trend_col].rolling(20).mean()
    out["MA50"]  = out[trend_col].rolling(50).mean()
    out["MA200"] = out[trend_col].rolling(200).mean()

    # RSI (Wilder)
    delta = out[trend_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    # ATR
    high_low   = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift(1)).abs()
    low_close  = (out["Low"]  - out["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # MACD
    ema12 = out[trend_col].ewm(span=12, adjust=False).mean()
    ema26 = out[trend_col].ewm(span=26, adjust=False).mean()
    out["MACD"]        = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"]   = out["MACD"] - out["MACD_SIGNAL"]

    # 볼린저밴드 (20일, 2σ)
    bb_mid   = out[trend_col].rolling(20).mean()
    bb_std   = out[trend_col].rolling(20).std(ddof=0)
    out["BB_UPPER"] = bb_mid + 2 * bb_std
    out["BB_MID"]   = bb_mid
    out["BB_LOWER"] = bb_mid - 2 * bb_std
    out["BB_WIDTH"] = (out["BB_UPPER"] - out["BB_LOWER"]) / bb_mid * 100
    out["BB_PCT"]   = (out[trend_col] - out["BB_LOWER"]) / (out["BB_UPPER"] - out["BB_LOWER"]) * 100

    # MDD
    roll_max  = out[trend_col].cummax()
    drawdown  = out[trend_col] / roll_max - 1.0
    mdd = drawdown.min() * 100 if len(drawdown.dropna()) > 0 else np.nan

    # Volume
    out["VOL20"] = out["Volume"].rolling(20).mean()

    high_52 = out["High"].max() if "High" in out.columns else np.nan
    low_52  = out["Low"].min()  if "Low"  in out.columns else np.nan

    meta = {
        "trend_price_col": trend_col,
        "display_price_col": display_col,
        "data_len": len(out),
        "mdd": mdd,
        "high_52": high_52,
        "low_52": low_52,
    }
    return out, meta


# =========================================================
# 9. Momentum (수익률)
# =========================================================
def compute_momentum(df, trend_col):
    close = df[trend_col]
    current = float(close.iloc[-1])

    def ret(days):
        if len(close) > days:
            return (current / float(close.iloc[-days]) - 1) * 100
        return None

    r1m  = ret(21)
    r3m  = ret(63)
    r6m  = ret(126)
    r12m = ret(252)

    ma200 = df["MA200"].iloc[-1]
    ma200_gap = ((current / float(ma200)) - 1) * 100 if is_valid_number(ma200) else None

    high_52 = df["High"].max()
    low_52  = df["Low"].min()
    from_high = ((current / high_52) - 1) * 100 if high_52 > 0 else None
    from_low  = ((current / low_52)  - 1) * 100 if low_52  > 0 else None

    return {
        "r1m": r1m, "r3m": r3m, "r6m": r6m, "r12m": r12m,
        "ma200_gap": ma200_gap, "from_high": from_high, "from_low": from_low,
    }


# =========================================================
# 10. Risk (Beta / Sharpe / Volatility)
# =========================================================
def compute_risk(df, trend_col):
    close = df[trend_col]
    daily_ret = close.pct_change().dropna()

    vol_1y = float(daily_ret.tail(252).std() * np.sqrt(252) * 100) if len(daily_ret) >= 60 else None

    # Beta vs SPY
    beta = None
    try:
        spy = yf.Ticker("SPY").history(period="2y")["Close"].pct_change().dropna()
        merged = pd.concat([daily_ret.rename("stock"), spy.rename("mkt")], axis=1).dropna().tail(252)
        if len(merged) >= 60:
            cov   = np.cov(merged["stock"], merged["mkt"])
            beta  = float(cov[0, 1] / cov[1, 1])
    except Exception:
        pass

    # Sharpe (무위험 3.5%)
    sharpe = None
    if vol_1y and vol_1y > 0:
        ann_ret = float(daily_ret.tail(252).mean() * 252 * 100)
        sharpe  = round((ann_ret - 3.5) / vol_1y, 2)

    return {"vol_1y": vol_1y, "beta": beta, "sharpe": sharpe}


# =========================================================
# 11. Shareholder Return
# =========================================================
def compute_shareholder(info):
    div_yield   = info.get("dividendYield")
    dps         = info.get("lastDividendValue")
    forward_pe  = info.get("forwardPE")
    payout_ratio= info.get("payoutRatio")
    eps_growth  = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")

    # PEG
    peg = None
    if is_valid_number(forward_pe) and is_valid_number(eps_growth) and eps_growth > 0:
        peg = round(forward_pe / (eps_growth * 100), 2)

    # FCF Yield
    fcf_yield = None
    try:
        fcf = info.get("freeCashflow")
        mktcap = info.get("marketCap")
        if is_valid_number(fcf) and is_valid_number(mktcap) and mktcap > 0:
            fcf_yield = float(fcf) / float(mktcap) * 100
    except Exception:
        pass

    return {
        "div_yield": div_yield, "dps": dps,
        "payout_ratio": payout_ratio, "peg": peg,
        "eps_growth": eps_growth, "fcf_yield": fcf_yield,
    }




# =========================================================
# 11b. 애널리스트 의견 분포
# =========================================================
@st.cache_data(ttl=3600)
def fetch_analyst_info(ticker_symbol: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(ticker_symbol)
        info = t.info or {}
        result = {
            "target_high":   info.get("targetHighPrice"),
            "target_low":    info.get("targetLowPrice"),
            "target_mean":   info.get("targetMeanPrice"),
            "target_median": info.get("targetMedianPrice"),
            "rec_mean":      info.get("recommendationMean"),
            "rec_key":       info.get("recommendationKey", ""),
            "num_analysts":  info.get("numberOfAnalystOpinions"),
            "strong_buy":    None,
            "buy":           None,
            "hold":          None,
            "sell":          None,
            "strong_sell":   None,
        }
        # 상세 의견 수
        try:
            rec_df = t.recommendations_summary
            if rec_df is not None and not rec_df.empty:
                latest = rec_df.iloc[0]
                result["strong_buy"]  = int(latest.get("strongBuy", 0)  or 0)
                result["buy"]         = int(latest.get("buy", 0)         or 0)
                result["hold"]        = int(latest.get("hold", 0)        or 0)
                result["sell"]        = int(latest.get("sell", 0)        or 0)
                result["strong_sell"] = int(latest.get("strongSell", 0)  or 0)
        except Exception:
            pass
        return result
    except Exception:
        return {}


# =========================================================
# 11c. 실적 서프라이즈 (최근 4분기)
# =========================================================
@st.cache_data(ttl=3600)
def fetch_earnings_surprise(ticker_symbol: str) -> list:
    try:
        t = yf.Ticker(ticker_symbol)
        hist = t.earnings_history
        if hist is None or (hasattr(hist, "empty") and hist.empty):
            return []
        rows = []
        df = hist if isinstance(hist, pd.DataFrame) else pd.DataFrame(hist)
        df = df.sort_index(ascending=False).head(4)
        for idx, row in df.iterrows():
            eps_est    = row.get("epsEstimate")    or row.get("EPS Estimate")
            eps_actual = row.get("epsActual")      or row.get("Reported EPS")
            surprise   = row.get("epsDifference")  or row.get("Surprise(%)")
            date_str   = str(idx)[:10] if idx else "N/A"
            rows.append({
                "date":       date_str,
                "eps_est":    float(eps_est)    if is_valid_number(eps_est)    else None,
                "eps_actual": float(eps_actual) if is_valid_number(eps_actual) else None,
                "surprise":   float(surprise)   if is_valid_number(surprise)   else None,
            })
        return rows
    except Exception:
        return []


# =========================================================
# 11d. 섹터 ETF 대비 상대 수익률
# =========================================================
SECTOR_ETF_MAP = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Communication Services": "XLC",
}

@st.cache_data(ttl=900)
def fetch_sector_relative(ticker_symbol: str, sector: str) -> Dict[str, Any]:
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf:
        return {"available": False, "reason": f"섹터 ETF 매핑 없음 ({sector})"}
    try:
        stock_df = yf.Ticker(ticker_symbol).history(period="1y")["Close"].dropna()
        etf_df   = yf.Ticker(etf).history(period="1y")["Close"].dropna()
        if len(stock_df) < 20 or len(etf_df) < 20:
            return {"available": False, "reason": "데이터 부족"}
        def ret(series, days):
            if len(series) > days:
                return (float(series.iloc[-1]) / float(series.iloc[-days]) - 1) * 100
            return None
        result = {
            "available": True,
            "etf":       etf,
            "stock_1m":  ret(stock_df, 21),  "etf_1m":  ret(etf_df, 21),
            "stock_3m":  ret(stock_df, 63),  "etf_3m":  ret(etf_df, 63),
            "stock_6m":  ret(stock_df, 126), "etf_6m":  ret(etf_df, 126),
            "stock_12m": ret(stock_df, 252), "etf_12m": ret(etf_df, 252),
        }
        # 상대 수익률
        for p in ["1m", "3m", "6m", "12m"]:
            s, e = result[f"stock_{p}"], result[f"etf_{p}"]
            result[f"rel_{p}"] = (s - e) if s is not None and e is not None else None
        return result
    except Exception as ex:
        return {"available": False, "reason": str(ex)}


# =========================================================
# 11e. 공매도 비율
# =========================================================
@st.cache_data(ttl=3600)
def fetch_short_info(ticker_symbol: str) -> Dict[str, Any]:
    try:
        info = yf.Ticker(ticker_symbol).info or {}
        shares_out  = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        short_int   = info.get("sharesShort")
        short_ratio = info.get("shortRatio")
        short_pct   = info.get("shortPercentOfFloat")
        short_prev  = info.get("sharesShortPriorMonth")
        short_date  = info.get("dateShortInterest")

        pct_calc = None
        if is_valid_number(short_int) and is_valid_number(shares_out) and shares_out > 0:
            pct_calc = float(short_int) / float(shares_out) * 100

        return {
            "available":   short_int is not None,
            "short_int":   short_int,
            "short_pct":   float(short_pct * 100) if is_valid_number(short_pct) else pct_calc,
            "short_ratio": short_ratio,
            "short_prev":  short_prev,
            "short_date":  fmt_date_from_timestamp(short_date),
        }
    except Exception:
        return {"available": False}

# =========================================================
# 12. Support / Resistance / Strategy / Scenario
# =========================================================
def build_action_zones(current_price, ma50, ma200, atr):
    fallback_used = False; fallback_reasons = []
    atr_val = atr
    if not is_valid_number(atr_val) or float(atr_val) <= 0:
        atr_val = current_price * 0.03; fallback_used = True; fallback_reasons.append("ATR")
    ma50_val = ma50
    if not is_valid_number(ma50_val):
        ma50_val = current_price * 0.95; fallback_used = True; fallback_reasons.append("MA50")
    ma200_val = ma200
    if not is_valid_number(ma200_val):
        ma200_val = current_price * 0.85; fallback_used = True; fallback_reasons.append("MA200")

    zone1_low  = max(ma50_val,  current_price - 1.5 * atr_val)
    zone1_high = current_price
    zone1_low, zone1_high = sorted([zone1_low, zone1_high])
    zone2_low  = max(ma200_val, current_price - 3.0 * atr_val)
    zone2_high = min(zone1_low * 0.995, current_price)
    if zone2_high - zone2_low < current_price * 0.005:
        zone2_high = zone2_low + current_price * 0.005
    zone2_low, zone2_high = sorted([zone2_low, zone2_high])
    gap = ((current_price - zone1_low) / current_price * 100) if current_price != 0 else np.nan

    return {"zone1_low": zone1_low, "zone1_high": zone1_high,
            "zone2_low": zone2_low, "zone2_high": zone2_high,
            "fallback_used": fallback_used, "fallback_reasons": fallback_reasons, "gap_to_zone1": gap}


def build_support_resistance(current_price, atr, ma20, ma50, high_52):
    atr_val  = atr  if is_valid_number(atr)  and float(atr)  > 0 else current_price * 0.03
    ma20_val = ma20 if is_valid_number(ma20) else current_price * 0.97
    ma50_val = ma50 if is_valid_number(ma50) else current_price * 0.92
    s1 = min(ma20_val, current_price - 1.0 * atr_val)
    s2 = min(ma50_val, current_price - 2.0 * atr_val)
    s1, s2 = sorted([s1, s2], reverse=True)
    r1 = current_price + 1.0 * atr_val
    r2 = current_price + 2.0 * atr_val
    return {"support_1": s1, "support_2": s2, "resistance_1": r1, "resistance_2": r2,
            "breakout": high_52 if is_valid_number(high_52) else np.nan}


def build_volume_comment(current_vol, avg20_vol):
    if not (is_valid_number(current_vol) and is_valid_number(avg20_vol) and avg20_vol > 0):
        return "거래량 데이터가 충분하지 않아 강도 판단은 제한됩니다."
    ratio = float(current_vol) / float(avg20_vol)
    if ratio >= 1.5: return "최근 거래량이 20일 평균을 뚜렷하게 상회해 수급 반응이 강한 편으로 볼 수 있습니다."
    if ratio >= 1.0: return "거래량이 최근 평균 수준 이상으로 유지되어 가격 움직임의 확인 신호로 해석할 수 있습니다."
    return "거래량이 20일 평균을 밑돌아 가격 움직임의 신뢰도는 다소 약할 수 있습니다."


def build_macd_comment(macd, signal, hist):
    if not (is_valid_number(macd) and is_valid_number(signal) and is_valid_number(hist)):
        return "MACD 데이터가 충분하지 않아 모멘텀 해석은 제한됩니다."
    if macd > signal and hist > 0:
        return "MACD가 시그널선 위에 있고 히스토그램도 양수라 모멘텀 우위가 유지되는 구간으로 볼 수 있습니다."
    if macd < signal and hist < 0:
        return "MACD가 시그널선 아래에 있고 히스토그램도 음수라 단기 모멘텀은 약한 편으로 해석됩니다."
    return "MACD와 시그널선의 간격이 크지 않아 방향성이 뚜렷하지 않은 중립 구간으로 볼 수 있습니다."


def build_strategy_text(current_price, zone1_low, zone2_low, ma200, rsi):
    if is_valid_number(rsi) and rsi < 35:
        short_term = (f"단기적으로는 RSI가 과매도 인근에 있어 반등 시도가 나올 수 있으나, "
                      f"{fmt_price(zone1_low)} 부근 지지 확인 여부를 먼저 보는 접근이 적절합니다.")
    else:
        short_term = (f"단기적으로는 현재가 추격보다 {fmt_price(zone1_low)} 부근 눌림목 형성 여부를 관찰하는 접근이 더 보수적입니다.")

    if is_valid_number(ma200) and current_price > ma200:
        mid_term = (f"중기적으로는 200일선 위 추세가 유지되는 동안 "
                    f"{fmt_price(zone2_low)}~{fmt_price(zone1_low)} 구간을 분할 관찰 구간으로 볼 수 있습니다.")
    else:
        mid_term = (f"중기적으로는 200일선 회복 전까지는 공격적 비중 확대보다 "
                    f"{fmt_price(zone2_low)} 부근 방어력 확인이 우선입니다.")
    return short_term, mid_term


def build_scenarios(current_price, target_price, ma200, support_1, support_2, resistance_1, macd, signal):
    bull_target = fmt_price(target_price) if is_valid_number(target_price) else fmt_price(resistance_1)
    bull = (f"추세가 200일선 위에서 유지되고 {fmt_price(support_1)} 지지가 확인되면, "
            f"단기적으로는 {fmt_price(resistance_1)} 테스트, 확장 시 {bull_target} 구간까지 열어둘 수 있습니다.")
    if is_valid_number(macd) and is_valid_number(signal) and macd < signal:
        bear = (f"모멘텀이 약한 상태에서 {fmt_price(support_1)} 이탈이 나오면 "
                f"{fmt_price(support_2)} 구간까지 조정 폭이 확대될 가능성을 열어둘 필요가 있습니다.")
    else:
        bear = (f"단기 지지선인 {fmt_price(support_1)} 이탈 시에는 "
                f"{fmt_price(support_2)} 부근 재점검 가능성을 염두에 두는 편이 보수적입니다.")
    return bull, bear


# =========================================================
# 13. App Header (즐겨찾기 / 최근검색)
# =========================================================
st.caption(
    f"<span style='color:#64748b;'>실전 투자 리포트 · {datetime.datetime.now().strftime('%Y.%m.%d')}</span>",
    unsafe_allow_html=True,
)

# session_state 초기화
if "us_favorites" not in st.session_state: st.session_state.us_favorites = []
if "us_history"   not in st.session_state: st.session_state.us_history   = []

fav_col, hist_col = st.columns(2)
with fav_col:
    if st.session_state.us_favorites:
        st.markdown("⭐ **즐겨찾기**")
        for fav in st.session_state.us_favorites:
            if st.button(fav, key=f"us_fav_{fav}", use_container_width=True):
                st.session_state["us_input_val"] = fav
                st.rerun()
with hist_col:
    if st.session_state.us_history:
        st.markdown("🕐 **최근 검색**")
        for h in st.session_state.us_history[-5:][::-1]:
            if st.button(h, key=f"us_hist_{h}", use_container_width=True):
                st.session_state["us_input_val"] = h
                st.rerun()

default_val = st.session_state.pop("us_input_val", "MSFT")
user_input_symbol = st.text_input(
    "종목 코드 또는 회사명을 입력하세요 (예: MSFT, TSMC, NVIDIA, QQQ, JPM, ^GSPC)",
    value=default_val,
).strip()


# =========================================================
# 14. Main UI
# =========================================================
if user_input_symbol:
    # ── 단계별 프로그레스바
    progress = st.progress(0)
    status   = st.empty()

    status.text("🔍 종목 검색 중...")
    progress.progress(10)
    resolved = resolve_symbol(user_input_symbol)
    ticker   = resolved["resolved_symbol"]

    if not ticker:
        st.error("입력값을 해석하지 못했습니다.")
        st.stop()

    # 최근 검색 저장
    disp = ticker
    if disp not in st.session_state.us_history:
        st.session_state.us_history.append(disp)
    if len(st.session_state.us_history) > 10:
        st.session_state.us_history = st.session_state.us_history[-10:]

    # 즐겨찾기 버튼
    is_fav = disp in st.session_state.us_favorites
    if st.button("⭐ 즐겨찾기 해제" if is_fav else "☆ 즐겨찾기 추가", key="us_toggle_fav"):
        if is_fav: st.session_state.us_favorites.remove(disp)
        else:      st.session_state.us_favorites.append(disp)
        st.rerun()

    status.text("📈 가격 데이터 로드 중...")
    progress.progress(20)
    price_df = fetch_price_data(ticker)
    info     = fetch_info_data(ticker)

    if price_df.empty or len(price_df) < 2:
        st.error("데이터가 부족하거나 유효하지 않은 티커입니다.")
        st.stop()

    valid_ohlc, missing_cols = validate_ohlc_columns(price_df)
    if not valid_ohlc:
        st.error(f"가격 데이터에 필요한 컬럼이 부족합니다: {', '.join(missing_cols)}")
        st.stop()

    if not isinstance(info, dict): info = {}

    status.text("📐 기술적 지표 / 볼린저밴드 계산 중...")
    progress.progress(40)
    df, meta = calculate_indicators(price_df)
    asset    = classify_asset(info)
    pbr_mod  = compute_pbr_module(info)

    status.text("📊 모멘텀 / 리스크 / 주주환원 분석 중...")
    progress.progress(55)
    trend_col   = meta["trend_price_col"]
    display_col = meta["display_price_col"]
    data_len    = meta["data_len"]
    mdd         = meta["mdd"]
    high_52     = meta["high_52"]
    low_52      = meta["low_52"]

    mo  = compute_momentum(df, trend_col)
    ri  = compute_risk(df, trend_col)
    sh  = compute_shareholder(info)

    status.text("🔍 애널리스트 / 실적 / 섹터 비교 분석 중...")
    progress.progress(70)
    analyst_data  = fetch_analyst_info(ticker)
    earnings_data = fetch_earnings_surprise(ticker)
    sector_rel    = fetch_sector_relative(ticker, asset["sector"])
    short_data    = fetch_short_info(ticker)

    status.text("🧮 분석 완료 중...")
    progress.progress(85)

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    current_price = last_row[display_col]
    prev_price    = prev_row[display_col]
    price_change  = current_price - prev_price
    pct_change    = (price_change / prev_price) * 100 if prev_price != 0 else 0.0

    short_name = safe_text(info.get("shortName", ticker))
    sector     = safe_text(info.get("sector", ""))
    industry   = safe_text(info.get("industry", ""))
    quote_type = safe_text(info.get("quoteType", ""))
    fund_family = safe_text(info.get("fundFamily", ""))
    category   = safe_text(info.get("category", ""))

    if asset["is_index"]:
        subtitle = "시장 지수"
        info_keys = ["previousClose", "fiftyTwoWeekHigh", "fiftyTwoWeekLow"]
    elif asset["is_etf_like"]:
        subtitle = f"{fund_family} | {category}" if fund_family and category else fund_family or category or quote_type or "펀드형 자산"
        info_keys = ["yield", "ytdReturn", "totalAssets", "category", "fundFamily"]
    else:
        subtitle = f"{sector} | {industry}" if sector and industry else sector or quote_type or "분류 정보 없음"
        if asset["is_financial"]:
            info_keys = ["targetMeanPrice", "earningsTimestamp", "returnOnEquity", "priceToBook", "forwardPE"]
        elif asset["is_reit"]:
            info_keys = ["targetMeanPrice", "earningsTimestamp", "yield", "forwardPE", "priceToBook"]
        else:
            info_keys = ["targetMeanPrice", "earningsTimestamp", "revenueGrowth", "operatingMargins", "forwardPE", "priceToBook"]

    filled_count, total_count = count_info_completeness(info, info_keys)
    ind_filled, ind_total     = compute_data_quality_summary(df)

    progress.progress(100)
    status.empty()
    progress.empty()

    # ── resolve 표시
    if resolved["method"] == "alias":
        st.markdown(f"<div class='info-note'>입력값 <b>{safe_text(user_input_symbol)}</b> → 별칭 <b>{safe_text(ticker)}</b></div>", unsafe_allow_html=True)
    elif resolved["method"] == "search":
        st.markdown(f"<div class='info-note'>입력값 <b>{safe_text(user_input_symbol)}</b> → 검색 결과 <b>{safe_text(ticker)}</b></div>", unsafe_allow_html=True)

    st.markdown(asset["badge"], unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-bottom:0; color:#0f172a;'>{short_name} <span style='color:#64748b; font-size:20px;'>· {safe_text(ticker)}</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b; font-size:13px; margin-top:4px;'>{subtitle} | 상단 가격 기준: {display_col} | 추세/MDD 기준: {trend_col}</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='quality-box'>메타데이터 완전성: <b>{filled_count}/{total_count}</b> | 기술지표 계산 가능: <b>{ind_filled}/{ind_total}</b></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 12px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    # ── 핵심 수치 (Top Summary)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(render_metric_html(f"종가 ({display_col})", fmt_price(current_price),
        subvalue=f"{pct_change:+.2f}%", subcolor="#dc2626" if price_change < 0 else "#059669"), unsafe_allow_html=True)

    if asset["is_index"]:
        c2.markdown(render_metric_html("52주 고가", fmt_price(high_52)), unsafe_allow_html=True)
        c3.markdown(render_metric_html("52주 저가", fmt_price(low_52)), unsafe_allow_html=True)
        c4.markdown(render_metric_html("52주 위치", compute_52w_position(current_price, high_52, low_52)), unsafe_allow_html=True)
    elif asset["is_etf_like"]:
        c2.markdown(render_metric_html("분배금 수익률", fmt_ratio_pct(info.get("yield"))), unsafe_allow_html=True)
        c3.markdown(render_metric_html("YTD 수익률",  fmt_ratio_pct(info.get("ytdReturn"))), unsafe_allow_html=True)
        c4.markdown(render_metric_html("운용자산(AUM)", fmt_large_dollar(info.get("totalAssets"))), unsafe_allow_html=True)
    else:
        target_price = info.get("targetMeanPrice")
        if is_valid_number(target_price) and current_price != 0:
            upside = ((target_price - current_price) / current_price) * 100
            c2.markdown(render_metric_html("애널리스트 목표가", fmt_price(target_price),
                subvalue=f"{upside:+.1f}%", subcolor="#059669" if upside >= 0 else "#dc2626"), unsafe_allow_html=True)
        else:
            c2.markdown(render_metric_html("애널리스트 목표가", "N/A"), unsafe_allow_html=True)

        if asset["is_financial"]:
            c3.markdown(render_metric_html("ROE", fmt_ratio_pct(info.get("returnOnEquity"))), unsafe_allow_html=True)
        elif asset["is_reit"]:
            c3.markdown(render_metric_html("분배금 수익률", fmt_ratio_pct(info.get("yield"))), unsafe_allow_html=True)
        else:
            c3.markdown(render_metric_html("매출 성장률(YoY)", fmt_ratio_pct(info.get("revenueGrowth"))), unsafe_allow_html=True)
        c4.markdown(render_metric_html("예상 실적 일정", fmt_date_from_timestamp(info.get("earningsTimestamp"))), unsafe_allow_html=True)

    # 52주 위치 (일반 주식)
    if not asset["is_index"] and not asset["is_etf_like"]:
        pos_52w = compute_52w_position(current_price, high_52, low_52)
        st.markdown(
            f"<div class='info-note'>52주 고가 {fmt_price(high_52)} / 저가 {fmt_price(low_52)} | "
            f"현재가 위치: <b>{pos_52w}</b> (저가=0%, 고가=100%) | "
            f"고가 대비: <b>{((current_price/float(high_52)-1)*100):.1f}%</b></div>"
            if is_valid_number(high_52) and is_valid_number(low_52) else "",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='info-note'>기업/펀드 메타데이터 값은 종목별로 누락되거나 지연될 수 있습니다.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fundamentals / Risk
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if asset["is_index"]:
            st.markdown(render_info_card("자산 유형 안내", "지수(Index)", "<b>설명:</b> 가격 추세와 변동성 해석 중심으로 보는 편이 적절합니다."), unsafe_allow_html=True)
        elif asset["is_etf_like"]:
            st.markdown(render_info_card("펀드형 자산 안내", "ETF/ETN/펀드", "<b>설명:</b> 기초지수, 자금 규모, 분배금, 추세·변동성을 함께 보는 편이 적절합니다."), unsafe_allow_html=True)
        else:
            forward_pe = fmt_mul(info.get("forwardPE"))
            if asset["is_financial"]:
                st.markdown(render_info_card("가치 평가 지표", f"선행 P/E: {forward_pe} / P/B: {fmt_mul(info.get('priceToBook'))}", "<b>해석:</b> 금융주는 P/B, ROE를 함께 참고하는 편이 적절합니다."), unsafe_allow_html=True)
            elif asset["is_reit"]:
                st.markdown(render_info_card("REIT 해석 안내", f"P/B: {fmt_mul(info.get('priceToBook'))}", "<b>해석:</b> 배당, 자산 가치, 금리 민감도까지 함께 보는 편이 적절합니다."), unsafe_allow_html=True)
            else:
                opm = fmt_ratio_pct(info.get("operatingMargins"))
                st.markdown(render_info_card("가치 평가 지표", f"선행 P/E: {forward_pe} / OPM: {opm}", "<b>해석:</b> 업종 평균과 비교할 때 해석력이 높아집니다."), unsafe_allow_html=True)

    with col_f2:
        mdd_rel  = reliability_badge(get_reliability_by_length(data_len, 252, 120))
        mdd_text = f"{mdd:.1f}%" if is_valid_number(mdd) else "N/A"
        period   = "최근 1년" if data_len >= 252 else ("수집된 데이터 구간" if data_len >= 120 else "제한된 데이터 구간")
        st.markdown(render_info_card(f"최대낙폭(MDD) {mdd_rel}", mdd_text, f"<b>기준:</b> {period} 내 고점 대비 최대 하락 비율입니다."), unsafe_allow_html=True)

    # ── PBR
    if not asset["is_index"] and not asset["is_etf_like"]:
        pbr_val  = f"현재 PBR: {fmt_mul(pbr_mod['current_pbr'])}" if pbr_mod["status"] == "CURRENT_ONLY" else "N/A"
        pbr_desc = "<b>상태:</b> 현재 PBR은 확인되지만, 과거 평균 PBR 대비 수준은 N/A입니다." if pbr_mod["status"] == "CURRENT_ONLY" else pbr_mod["note"]
        st.markdown(render_info_card("PBR 상대 수준", pbr_val, pbr_desc), unsafe_allow_html=True)

    # ── 주주환원 카드 (신규)
    if not asset["is_index"] and not asset["is_etf_like"]:
        sh_lines = []
        if is_valid_number(sh["div_yield"]):  sh_lines.append(f"배당수익률: <b>{sh['div_yield']*100:.2f}%</b>")
        if is_valid_number(sh["dps"]):        sh_lines.append(f"주당배당금: <b>${sh['dps']:.2f}</b>")
        if is_valid_number(sh["payout_ratio"]): sh_lines.append(f"배당성향: <b>{sh['payout_ratio']*100:.1f}%</b>")
        if is_valid_number(sh["peg"]):        sh_lines.append(f"PEG: <b>{sh['peg']:.2f}</b> (PER ÷ EPS성장률, &lt;1이면 성장 대비 저평가)")
        if is_valid_number(sh["fcf_yield"]):  sh_lines.append(f"FCF Yield: <b>{sh['fcf_yield']:.2f}%</b>")
        if is_valid_number(sh["eps_growth"]): sh_lines.append(f"EPS 성장률: <b>{sh['eps_growth']*100:.1f}%</b>")
        sh_rel = reliability_badge("high" if sh["peg"] is not None else ("mid" if sh["div_yield"] is not None else "low"))
        sh_val = f"{sh['div_yield']*100:.2f}%" if is_valid_number(sh["div_yield"]) else "N/A"
        sh_desc = "<br>".join(sh_lines) if sh_lines else "주주환원 데이터를 불러오지 못했습니다."
        st.markdown(render_info_card(f"💰 주주환원 {sh_rel}", f"배당수익률 {sh_val}", sh_desc), unsafe_allow_html=True)

    # ── 볼린저밴드 카드 (신규)
    bb_pct   = last_row.get("BB_PCT")
    bb_upper = last_row.get("BB_UPPER")
    bb_lower = last_row.get("BB_LOWER")
    bb_mid   = last_row.get("BB_MID")
    bb_width = last_row.get("BB_WIDTH")
    if is_valid_number(bb_pct):
        if   bb_pct >= 100: bb_pos = "상단 돌파 🔴"
        elif bb_pct >= 80:  bb_pos = "상단 근접 ⚠️"
        elif bb_pct <= 0:   bb_pos = "하단 이탈 🟢"
        elif bb_pct <= 20:  bb_pos = "하단 근접 👀"
        else:               bb_pos = "밴드 내 중립"
        bb_desc = (f"상단: {fmt_price(bb_upper)} / 중간: {fmt_price(bb_mid)} / 하단: {fmt_price(bb_lower)}<br>"
                   f"밴드폭: {bb_width:.1f}% · 밴드 내 위치: {bb_pct:.1f}%")
        st.markdown(render_info_card(f"🎯 볼린저밴드 {reliability_badge('high')}", f"{bb_pos} ({bb_pct:.1f}%)",
            f"<b>설명:</b> {bb_desc}<br>80%↑ 과매수 주의 / 20%↓ 과매도 반등 가능. 밴드폭 축소 후 확장 시 큰 움직임 예고."), unsafe_allow_html=True)

    # ── Momentum 카드 (신규)
    mo_lines = []
    for label, val in [("1개월", mo["r1m"]), ("3개월", mo["r3m"]), ("6개월", mo["r6m"]), ("12개월", mo["r12m"])]:
        if is_valid_number(val):
            color = "#16a34a" if val >= 0 else "#dc2626"
            arrow = "▲" if val >= 0 else "▼"
            mo_lines.append(f"{label}: <b style='color:{color};'>{arrow} {abs(val):.1f}%</b>")
    if is_valid_number(mo["ma200_gap"]):
        color = "#16a34a" if mo["ma200_gap"] >= 0 else "#dc2626"
        mo_lines.append(f"200MA 괴리: <b style='color:{color};'>{mo['ma200_gap']:+.1f}%</b>")
    if is_valid_number(mo["from_high"]):
        mo_lines.append(f"52주 고가 대비: <b style='color:#dc2626;'>{mo['from_high']:.1f}%</b>")
    if is_valid_number(mo["from_low"]):
        mo_lines.append(f"52주 저가 대비: <b style='color:#16a34a;'>{mo['from_low']:+.1f}%</b>")

    r6m_str = f"{mo['r6m']:+.1f}%" if is_valid_number(mo["r6m"]) else "N/A"
    if mo_lines:
        st.markdown(render_info_card(f"📈 Momentum {reliability_badge('high')}", f"6M: {r6m_str}",
            " · ".join(mo_lines)), unsafe_allow_html=True)

    # ── Risk 카드 (신규)
    ri_lines = []
    if is_valid_number(ri["beta"]):    ri_lines.append(f"Beta (vs SPY): <b>{ri['beta']:.2f}</b>")
    if is_valid_number(ri["vol_1y"]):  ri_lines.append(f"연간 변동성: <b>{ri['vol_1y']:.1f}%</b>")
    if is_valid_number(ri["sharpe"]):  ri_lines.append(f"Sharpe Ratio: <b>{ri['sharpe']:.2f}</b>")
    if ri_lines:
        beta_str = fmt_mul(ri["beta"]) if is_valid_number(ri["beta"]) else "N/A"
        ri_rel = reliability_badge("high" if ri["beta"] is not None and ri["sharpe"] is not None else "mid")
        st.markdown(render_info_card(f"⚠️ Risk {ri_rel}", f"Beta {beta_str}",
            "<br>".join(ri_lines) + "<br><b>해석:</b> Beta>1 시장보다 변동 큼 / Sharpe>1 양호한 위험 대비 수익"), unsafe_allow_html=True)


    # ── 애널리스트 의견 분포 (신규)
    if not asset["is_index"] and analyst_data:
        t_mean = analyst_data.get("target_mean")
        t_high = analyst_data.get("target_high")
        t_low  = analyst_data.get("target_low")
        rec    = analyst_data.get("rec_key", "").replace("_", " ").title()
        n_ana  = analyst_data.get("num_analysts")
        sb = analyst_data.get("strong_buy")  or 0
        b  = analyst_data.get("buy")         or 0
        h  = analyst_data.get("hold")        or 0
        s  = analyst_data.get("sell")        or 0
        ss = analyst_data.get("strong_sell") or 0
        total_votes = sb + b + h + s + ss

        upside_str = ""
        if is_valid_number(t_mean) and current_price > 0:
            upside = (float(t_mean) - current_price) / current_price * 100
            color  = "#16a34a" if upside >= 0 else "#dc2626"
            upside_str = f' <span style="color:{color};font-weight:700;">({upside:+.1f}%)</span>'

        vote_html = ""
        if total_votes > 0:
            bar_items = [
                (sb, "#166534", "Strong Buy"),
                (b,  "#2563eb", "Buy"),
                (h,  "#ca8a04", "Hold"),
                (s,  "#ea580c", "Sell"),
                (ss, "#dc2626", "Strong Sell"),
            ]
            bars = "".join([
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<div style="font-size:11px;color:#64748b;width:72px;">{lbl}</div>'
                f'<div style="flex:1;background:#f1f5f9;border-radius:4px;height:10px;">'
                f'<div style="width:{cnt/total_votes*100:.0f}%;background:{col};height:10px;border-radius:4px;"></div></div>'
                f'<div style="font-size:11px;font-weight:700;color:{col};width:20px;text-align:right;">{cnt}</div>'
                f'</div>'
                for cnt, col, lbl in bar_items if cnt > 0
            ])
            vote_html = f'<div style="margin-top:10px;">{bars}</div>'

        ana_desc = (
            f'목표가 범위: {fmt_price(t_low)} ~ {fmt_price(t_high)}<br>'
            f'평균 목표가: <b>{fmt_price(t_mean)}</b>{upside_str}<br>'
            f'컨센서스: <b>{rec}</b> ({n_ana}명 참여)'
            if is_valid_number(t_mean) else "애널리스트 데이터를 불러오지 못했습니다."
        )
        st.markdown(
            render_info_card(
                f"🎯 애널리스트 의견 {reliability_badge('mid')}",
                fmt_price(t_mean) + upside_str if is_valid_number(t_mean) else "N/A",
                ana_desc + vote_html,
            ),
            unsafe_allow_html=True,
        )

    # ── 실적 서프라이즈 (신규)
    if not asset["is_index"] and not asset["is_etf_like"] and earnings_data:
        rows_html = ""
        for e in earnings_data:
            if e["eps_actual"] is None and e["eps_est"] is None:
                continue
            surp = e["surprise"]
            surp_color = "#16a34a" if surp and surp >= 0 else "#dc2626"
            surp_str   = f'<span style="color:{surp_color};font-weight:700;">{surp:+.2f}</span>' if surp is not None else "N/A"
            rows_html += (
                f'<tr style="border-bottom:1px solid #f1f5f9;">'
                f'<td style="padding:7px 8px;font-size:12px;color:#64748b;">{e["date"]}</td>'
                f'<td style="padding:7px 8px;font-size:12px;text-align:right;">'
                f'{"$"+f'{e["eps_est"]:.2f}' if e["eps_est"] is not None else "N/A"}</td>'
                f'<td style="padding:7px 8px;font-size:12px;text-align:right;font-weight:700;">'
                f'{"$"+f'{e["eps_actual"]:.2f}' if e["eps_actual"] is not None else "N/A"}</td>'
                f'<td style="padding:7px 8px;text-align:right;">{surp_str}</td>'
                f'</tr>'
            )
        if rows_html:
            table_html = (
                f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
                f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                f'<th style="padding:6px 8px;text-align:left;color:#64748b;font-size:11px;">분기</th>'
                f'<th style="padding:6px 8px;text-align:right;color:#64748b;font-size:11px;">EPS 예상</th>'
                f'<th style="padding:6px 8px;text-align:right;color:#64748b;font-size:11px;">EPS 실제</th>'
                f'<th style="padding:6px 8px;text-align:right;color:#64748b;font-size:11px;">서프라이즈</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>'
            )
            st.markdown(
                render_info_card(
                    f"📋 실적 서프라이즈 (최근 4분기) {reliability_badge('mid')}",
                    "",
                    table_html,
                ),
                unsafe_allow_html=True,
            )

    # ── 섹터 ETF 대비 성과 (신규)
    if sector_rel.get("available"):
        etf_name = sector_rel["etf"]
        def rel_row(period, label):
            s = sector_rel.get(f"stock_{period}")
            e = sector_rel.get(f"etf_{period}")
            r = sector_rel.get(f"rel_{period}")
            if s is None: return ""
            sc = "#16a34a" if s >= 0 else "#dc2626"
            ec = "#16a34a" if e is not None and e >= 0 else "#dc2626"
            rc = "#16a34a" if r is not None and r >= 0 else "#dc2626"
            e_str = f'<span style="color:{ec};">{e:+.1f}%</span>' if e is not None else "N/A"
            r_str = f'<span style="color:{rc};font-weight:700;">{r:+.1f}%</span>' if r is not None else "N/A"
            return (
                f'<tr style="border-bottom:1px solid #f1f5f9;">'
                f'<td style="padding:7px 8px;font-size:12px;color:#64748b;">{label}</td>'
                f'<td style="padding:7px 8px;font-size:12px;text-align:right;color:{sc};font-weight:700;">{s:+.1f}%</td>'
                f'<td style="padding:7px 8px;font-size:12px;text-align:right;">{e_str}</td>'
                f'<td style="padding:7px 8px;font-size:12px;text-align:right;">{r_str}</td>'
                f'</tr>'
            )
        rows = "".join([rel_row(p, l) for p, l in [("1m","1개월"),("3m","3개월"),("6m","6개월"),("12m","12개월")]])
        if rows:
            table = (
                f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
                f'<thead><tr style="border-bottom:2px solid #e2e8f0;">'
                f'<th style="padding:6px 8px;text-align:left;color:#64748b;font-size:11px;">기간</th>'
                f'<th style="padding:6px 8px;text-align:right;color:#64748b;font-size:11px;">종목</th>'
                f'<th style="padding:6px 8px;text-align:right;color:#64748b;font-size:11px;">{etf_name}</th>'
                f'<th style="padding:6px 8px;text-align:right;color:#64748b;font-size:11px;">상대 성과</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>'
            )
            st.markdown(
                render_info_card(
                    f"📈 섹터 ETF({etf_name}) 대비 성과 {reliability_badge('high')}",
                    "",
                    table,
                ),
                unsafe_allow_html=True,
            )

    # ── 공매도 비율 (신규)
    if short_data.get("available"):
        short_pct   = short_data.get("short_pct")
        short_ratio = short_data.get("short_ratio")
        short_prev  = short_data.get("short_prev")
        short_int   = short_data.get("short_int")
        short_date  = short_data.get("short_date")

        if short_pct is not None:
            if   short_pct >= 20: short_lvl = "🔴 매우 높음 (숏 스퀴즈 주의)"
            elif short_pct >= 10: short_lvl = "🟠 높음"
            elif short_pct >= 5:  short_lvl = "🟡 보통"
            else:                 short_lvl = "🟢 낮음"
        else:
            short_lvl = "N/A"

        change_html = ""
        if short_prev is not None and short_int is not None and is_valid_number(short_prev) and float(short_prev) > 0:
            chg = (float(short_int) - float(short_prev)) / float(short_prev) * 100
            col = "#dc2626" if chg > 0 else "#16a34a"
            change_html = f'<br>전월 대비: <span style="color:{col};font-weight:700;">{chg:+.1f}%</span>'

        short_desc = (
            f'공매도 비율: <b>{short_pct:.1f}%</b><br>'
            f'공매도 잔고: <b>{fmt_large_dollar(short_int)}</b>{change_html}<br>'
            + (f'Days to Cover: <b>{short_ratio:.1f}일</b><br>' if is_valid_number(short_ratio) else "")
            + f'기준일: {short_date}'
        )
        st.markdown(
            render_info_card(
                f"⚡ 공매도 현황 {reliability_badge('mid')}",
                f"{short_pct:.1f}%  {short_lvl}" if short_pct is not None else "N/A",
                short_desc,
            ),
            unsafe_allow_html=True,
        )

    # ── MA200 / RSI
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        ma200     = last_row["MA200"]
        ma200_rel = reliability_badge(get_reliability_by_length(data_len, 200, 120))
        if not is_valid_number(ma200):
            st.markdown(render_info_card(f"장기 추세(MA200) {ma200_rel}", "판단 불가", "<b>설명:</b> 200일 이동평균 계산이 제한됩니다."), unsafe_allow_html=True)
        else:
            is_bull    = last_row[trend_col] > ma200
            trend_title = "200일선 상회" if is_bull else "200일선 하회"
            st.markdown(render_info_card(f"장기 추세(MA200) {ma200_rel}", trend_title,
                f"<b>설명:</b> 추세 기준 가격이 200일선({fmt_price(ma200)}) {'위에' if is_bull else '아래에'} 위치해 있습니다. 장기 방향성을 볼 때 자주 참고하는 기준선입니다."), unsafe_allow_html=True)

    with col_t2:
        rsi     = last_row["RSI14"]
        rsi_rel = reliability_badge(get_reliability_by_length(data_len, 60, 14))
        if not is_valid_number(rsi):
            st.markdown(render_info_card(f"RSI(14일) {rsi_rel}", "판단 불가", "<b>설명:</b> 데이터가 부족해 RSI 계산이 제한됩니다."), unsafe_allow_html=True)
        else:
            if   rsi < 30: rsi_title = f"{rsi:.1f} (과매도 범위)"
            elif rsi > 70: rsi_title = f"{rsi:.1f} (과매수 범위)"
            else:          rsi_title = f"{rsi:.1f} (중립 범위)"
            st.markdown(render_info_card(f"RSI(14일) {rsi_rel}", rsi_title,
                "<b>설명:</b> Wilder 방식 RSI(14) 기준이며, 단독으로 반등·하락을 단정하기보다 추세와 함께 해석하는 편이 적절합니다."), unsafe_allow_html=True)

    # ── MACD / Volume
    col_m1, col_m2 = st.columns(2)
    macd_comment   = build_macd_comment(last_row["MACD"], last_row["MACD_SIGNAL"], last_row["MACD_HIST"])
    volume_comment = build_volume_comment(last_row["Volume"], last_row["VOL20"])

    with col_m1:
        macd_val = f"MACD: {last_row['MACD']:.2f} / Signal: {last_row['MACD_SIGNAL']:.2f}" if is_valid_number(last_row["MACD"]) and is_valid_number(last_row["MACD_SIGNAL"]) else "N/A"
        st.markdown(render_info_card("모멘텀 지표", macd_val, f"<b>설명:</b> {macd_comment}"), unsafe_allow_html=True)

    with col_m2:
        vol_ratio = (last_row["Volume"] / last_row["VOL20"]) if is_valid_number(last_row["VOL20"]) and last_row["VOL20"] > 0 else np.nan
        vol_val   = f"현재/20일 평균: {vol_ratio:.2f}배" if is_valid_number(vol_ratio) else "N/A"
        st.markdown(render_info_card("거래량 평가", vol_val, f"<b>설명:</b> {volume_comment}"), unsafe_allow_html=True)

    # ── Support / Resistance
    sr = build_support_resistance(float(last_row[trend_col]), last_row["ATR14"], last_row["MA20"], last_row["MA50"], high_52)
    st.markdown("<div style='font-size:16px; font-weight:800; color:#0f172a; margin-top:8px; margin-bottom:12px;'>📌 주요 지지 / 저항 레벨</div>", unsafe_allow_html=True)
    s1, s2, r1, r2, br = st.columns(5)
    s1.markdown(render_metric_html("1차 지지",  fmt_price(sr["support_1"])),    unsafe_allow_html=True)
    s2.markdown(render_metric_html("2차 지지",  fmt_price(sr["support_2"])),    unsafe_allow_html=True)
    r1.markdown(render_metric_html("1차 저항",  fmt_price(sr["resistance_1"])), unsafe_allow_html=True)
    r2.markdown(render_metric_html("2차 저항",  fmt_price(sr["resistance_2"])), unsafe_allow_html=True)
    br.markdown(render_metric_html("52주 고가", fmt_price(sr["breakout"])),     unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action Zones
    st.markdown("<div style='font-size:16px; font-weight:800; color:#0f172a; margin-top:8px; margin-bottom:12px;'>🎯 기계적 관심 구간 (참고용)</div>", unsafe_allow_html=True)
    zones = build_action_zones(float(last_row[trend_col]), last_row["MA50"], last_row["MA200"], last_row["ATR14"])
    gap_text = f"현재가 대비 1차 하단까지 {zones['gap_to_zone1']:.2f}%" if is_valid_number(zones["gap_to_zone1"]) else "N/A"

    z1, z2 = st.columns(2)
    with z1:
        st.markdown(f"<div class='zone-box' style='background-color:#f0fdf4;border:1px solid #bbf7d0;'><div style='color:#166534;font-size:13px;font-weight:800;'>1차 밴드 (MA50 & ATR 기반)</div><div style='color:#15803d;font-weight:900;font-size:20px;margin-top:6px;'>{fmt_price(zones['zone1_low'])} ~ {fmt_price(zones['zone1_high'])}</div></div>", unsafe_allow_html=True)
    with z2:
        st.markdown(f"<div class='zone-box' style='background-color:#eff6ff;border:1px solid #93c5fd;'><div style='color:#1e40af;font-size:13px;font-weight:800;'>2차 밴드 (MA200 & ATR 기반)</div><div style='color:#1d4ed8;font-weight:900;font-size:20px;margin-top:6px;'>{fmt_price(zones['zone2_low'])} ~ {fmt_price(zones['zone2_high'])}</div></div>", unsafe_allow_html=True)

    fallback_msg = ""
    if zones["fallback_used"]:
        fallback_msg = f"<div class='fallback-warn'>⚠️ 일부 지표({', '.join(zones['fallback_reasons'])})가 부족하여 보정값이 사용되었습니다.</div>"
    st.markdown(f"<div class='info-note'>{gap_text} | 이 밴드는 이동평균과 ATR을 조합한 기계적 참고 범위입니다.</div>{fallback_msg}", unsafe_allow_html=True)

    # ── Strategy
    short_strategy, mid_strategy = build_strategy_text(float(last_row[trend_col]), zones["zone1_low"], zones["zone2_low"], last_row["MA200"], last_row["RSI14"])
    st.markdown("<div style='font-size:16px; font-weight:800; color:#0f172a; margin-top:10px; margin-bottom:12px;'>🧭 투자 전략</div>", unsafe_allow_html=True)
    st.markdown(render_info_card("단기 전략", "1~2주", f"<b>전략:</b> {short_strategy}"), unsafe_allow_html=True)
    st.markdown(render_info_card("중기 전략", "1~3개월", f"<b>전략:</b> {mid_strategy}"), unsafe_allow_html=True)

    # ── Scenario
    bull_scenario, bear_scenario = build_scenarios(
        float(last_row[trend_col]), info.get("targetMeanPrice"),
        last_row["MA200"], sr["support_1"], sr["support_2"],
        sr["resistance_1"], last_row["MACD"], last_row["MACD_SIGNAL"])

    st.markdown("<div style='font-size:16px; font-weight:800; color:#0f172a; margin-top:10px; margin-bottom:12px;'>⚖️ 시나리오 분석</div>", unsafe_allow_html=True)
    sb, srisk = st.columns(2)
    with sb:
        st.markdown(f"<div class='scenario-bull'><div style='font-weight:900;font-size:15px;margin-bottom:8px;'>강세 시나리오</div><div>{bull_scenario}</div></div>", unsafe_allow_html=True)
    with srisk:
        st.markdown(f"<div class='scenario-bear'><div style='font-weight:900;font-size:15px;margin-bottom:8px;'>약세 시나리오</div><div>{bear_scenario}</div></div>", unsafe_allow_html=True)

    # ── 출처 / 면책 고지
    today = datetime.date.today().strftime("%Y.%m.%d")
    st.markdown(
        f"""
        <div style="margin-top:24px;padding:14px 16px;background:#f8fafc;
                    border-top:1px solid #e2e8f0;border-radius:10px;
                    font-size:12px;color:#94a3b8;line-height:1.8;">
            <div style="font-weight:700;color:#64748b;margin-bottom:4px;">📋 데이터 출처 및 면책 고지</div>
            가격 데이터: Yahoo Finance (yfinance) · 기준일: {today}<br>
            기술적 지표: 자체 계산 (MA·RSI·MACD·ATR·볼린저밴드·Beta·Sharpe)<br>
            펀더멘털: Yahoo Finance info (지연 또는 누락 가능)<br><br>
            <span style="color:#cbd5e1;">
            ⚠️ 본 화면은 정보 제공용이며, 투자 권유가 아닙니다.
            모든 투자 판단과 책임은 본인에게 있습니다.
            데이터는 지연되거나 부정확할 수 있습니다.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
