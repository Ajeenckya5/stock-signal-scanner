"""
US + India equity universes, symbol hygiene, and search.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

_CACHE_US = os.path.join(os.path.dirname(__file__), ".ticker_cache_us.csv")

# --- US mega-caps (fast default scan) ---
US_MEGA = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"), ("GOOGL", "Alphabet"),
    ("AMZN", "Amazon"), ("META", "Meta Platforms"), ("BRK-B", "Berkshire Hathaway"),
    ("LLY", "Eli Lilly"), ("AVGO", "Broadcom"), ("JPM", "JPMorgan Chase"),
    ("TSLA", "Tesla"), ("V", "Visa"), ("UNH", "UnitedHealth"), ("XOM", "Exxon Mobil"),
    ("MA", "Mastercard"), ("JNJ", "Johnson & Johnson"), ("WMT", "Walmart"),
    ("PG", "Procter & Gamble"), ("HD", "Home Depot"), ("COST", "Costco"),
    ("ABBV", "AbbVie"), ("ORCL", "Oracle"), ("BAC", "Bank of America"),
    ("NFLX", "Netflix"), ("KO", "Coca-Cola"), ("CRM", "Salesforce"),
    ("MRK", "Merck"), ("AMD", "AMD"), ("PEP", "PepsiCo"), ("CVX", "Chevron"),
    ("TMO", "Thermo Fisher"), ("CSCO", "Cisco"), ("LIN", "Linde"),
    ("ACN", "Accenture"), ("MCD", "McDonald's"), ("ABT", "Abbott"),
    ("WFC", "Wells Fargo"), ("GE", "GE Aerospace"), ("DIS", "Disney"),
    ("INTU", "Intuit"), ("IBM", "IBM"), ("QCOM", "Qualcomm"), ("CAT", "Caterpillar"),
    ("VZ", "Verizon"), ("NOW", "ServiceNow"), ("TXN", "Texas Instruments"),
    ("AMAT", "Applied Materials"), ("ISRG", "Intuitive Surgical"), ("PM", "Philip Morris"),
    ("MS", "Morgan Stanley"), ("NEE", "NextEra Energy"), ("PFE", "Pfizer"),
    ("GS", "Goldman Sachs"), ("BA", "Boeing"), ("RTX", "RTX"), ("SPGI", "S&P Global"),
    ("T", "AT&T"), ("LOW", "Lowe's"), ("UNP", "Union Pacific"), ("HON", "Honeywell"),
    ("BLK", "BlackRock"), ("BKNG", "Booking"), ("PLTR", "Palantir"), ("UBER", "Uber"),
]

SP100_FALLBACK = US_MEGA + [
    ("ADBE", "Adobe"), ("AXP", "American Express"), ("AMGN", "Amgen"),
    ("BMY", "Bristol-Myers"), ("C", "Citigroup"), ("COP", "ConocoPhillips"),
    ("CMCSA", "Comcast"), ("DE", "Deere"), ("DHR", "Danaher"), ("ELV", "Elevance"),
    ("GILD", "Gilead"), ("GM", "General Motors"), ("INTC", "Intel"),
    ("MDT", "Medtronic"), ("MMM", "3M"), ("MO", "Altria"), ("NKE", "Nike"),
    ("PYPL", "PayPal"), ("SBUX", "Starbucks"), ("SCHW", "Charles Schwab"),
    ("SO", "Southern Co"), ("SYK", "Stryker"), ("TJX", "TJX"), ("TMUS", "T-Mobile"),
    ("UNH", "UnitedHealth"), ("UPS", "UPS"), ("USB", "U.S. Bancorp"),
    ("VRTX", "Vertex"), ("LMT", "Lockheed Martin"), ("CVS", "CVS Health"),
    ("DUK", "Duke Energy"), ("MDLZ", "Mondelez"), ("CB", "Chubb"),
    ("CI", "Cigna"), ("CL", "Colgate-Palmolive"), ("EMR", "Emerson"),
    ("EOG", "EOG Resources"), ("FDX", "FedEx"), ("GD", "General Dynamics"),
    ("ICE", "Intercontinental Exchange"), ("ITW", "Illinois Tool Works"),
    ("MMC", "Marsh McLennan"), ("PGR", "Progressive"), ("SHW", "Sherwin-Williams"),
    ("SLB", "Schlumberger"), ("WM", "Waste Management"), ("ZTS", "Zoetis"),
]

NASDAQ100_FALLBACK = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"), ("AMZN", "Amazon"),
    ("META", "Meta"), ("GOOGL", "Alphabet"), ("GOOG", "Alphabet C"), ("TSLA", "Tesla"),
    ("AVGO", "Broadcom"), ("COST", "Costco"), ("NFLX", "Netflix"), ("AMD", "AMD"),
    ("PEP", "PepsiCo"), ("ADBE", "Adobe"), ("CSCO", "Cisco"), ("TMUS", "T-Mobile"),
    ("INTC", "Intel"), ("INTU", "Intuit"), ("AMGN", "Amgen"), ("ISRG", "Intuitive"),
    ("CMCSA", "Comcast"), ("QCOM", "Qualcomm"), ("TXN", "Texas Instruments"),
    ("AMAT", "Applied Materials"), ("HON", "Honeywell"), ("BKNG", "Booking"),
    ("VRTX", "Vertex"), ("ADP", "ADP"), ("SBUX", "Starbucks"), ("GILD", "Gilead"),
    ("ADI", "Analog Devices"), ("MU", "Micron"), ("LRCX", "Lam Research"),
    ("PANW", "Palo Alto"), ("PYPL", "PayPal"), ("REGN", "Regeneron"),
    ("MELI", "MercadoLibre"), ("KLAC", "KLA"), ("SNPS", "Synopsys"),
    ("CDNS", "Cadence"), ("CRWD", "CrowdStrike"), ("MAR", "Marriott"),
    ("CSX", "CSX"), ("ORLY", "O'Reilly"), ("CTAS", "Cintas"), ("FTNT", "Fortinet"),
    ("DASH", "DoorDash"), ("ABNB", "Airbnb"), ("PCAR", "PACCAR"), ("NXPI", "NXP"),
    ("WDAY", "Workday"), ("ROP", "Roper"), ("ADSK", "Autodesk"), ("MNST", "Monster"),
    ("AEP", "American Electric"), ("PAYX", "Paychex"), ("CPRT", "Copart"),
    ("ROST", "Ross Stores"), ("KDP", "Keurig Dr Pepper"), ("CHTR", "Charter"),
    ("ODFL", "Old Dominion"), ("FAST", "Fastenal"), ("EA", "Electronic Arts"),
    ("VRSK", "Verisk"), ("BKR", "Baker Hughes"), ("GEHC", "GE HealthCare"),
    ("CTSH", "Cognizant"), ("EXC", "Exelon"), ("KHC", "Kraft Heinz"),
    ("XEL", "Xcel Energy"), ("CCEP", "Coca-Cola Europacific"), ("IDXX", "IDEXX"),
    ("FANG", "Diamondback"), ("TTD", "Trade Desk"), ("DDOG", "Datadog"),
    ("ZS", "Zscaler"), ("TEAM", "Atlassian"), ("ON", "ON Semiconductor"),
    ("MRVL", "Marvell"), ("MCHP", "Microchip"), ("ANSS", "Ansys"),
    ("CDW", "CDW"), ("CSGP", "CoStar"), ("DXCM", "Dexcom"), ("BIIB", "Biogen"),
    ("ILMN", "Illumina"), ("GFS", "GlobalFoundries"), ("ARM", "Arm"),
    ("PDD", "PDD Holdings"), ("ASML", "ASML"), ("AZN", "AstraZeneca"),
    ("SHOP", "Shopify"),
]


def yf_us_symbol(sym: str) -> str:
    s = str(sym).strip().upper().replace(".", "-")
    if s.endswith("-NS") or s.endswith("-BO"):
        return s.replace("-NS", ".NS").replace("-BO", ".BO")
    return s


def currency_for(ticker: str) -> str:
    t = str(ticker).upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "INR"
    return "USD"


def currency_symbol(ticker: str) -> str:
    return "₹" if currency_for(ticker) == "INR" else "$"


def market_for(ticker: str) -> str:
    t = str(ticker).upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "IN"
    if t.startswith("^"):
        return "INDEX"
    return "US"


def benchmark_symbol(ticker: str) -> str:
    return "^NSEI" if market_for(ticker) == "IN" else "SPY"


def _fetch_sp500() -> pd.DataFrame:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        dfs = pd.read_html(url)
        for d in dfs:
            cols = [str(c) for c in d.columns]
            if "Symbol" in cols and ("Security" in cols or "Name" in cols):
                name_col = "Security" if "Security" in cols else "Name"
                df = d[["Symbol", name_col]].rename(columns={"Symbol": "symbol", name_col: "name"})
                df["symbol"] = df["symbol"].astype(str).map(yf_us_symbol)
                df["market"] = "US"
                return df.dropna(subset=["symbol", "name"])
    except Exception:
        pass
    return pd.DataFrame(columns=["symbol", "name", "market"])


def _fetch_nasdaq100() -> pd.DataFrame:
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        dfs = pd.read_html(url)
        for d in dfs:
            cols = [str(c).lower() for c in d.columns]
            if any("ticker" in c or "symbol" in c for c in cols) and len(d) >= 80:
                sym_col = next(
                    c for c in d.columns
                    if "ticker" in str(c).lower() or "symbol" in str(c).lower()
                )
                name_col = next(
                    (c for c in d.columns if "company" in str(c).lower() or "name" in str(c).lower()),
                    d.columns[1] if len(d.columns) > 1 else d.columns[0],
                )
                df = d[[sym_col, name_col]].rename(columns={sym_col: "symbol", name_col: "name"})
                df["symbol"] = df["symbol"].astype(str).map(yf_us_symbol)
                df["market"] = "US"
                if len(df) >= 80:
                    return df.dropna(subset=["symbol", "name"])
    except Exception:
        pass
    return pd.DataFrame(columns=["symbol", "name", "market"])


def load_us_tickers() -> pd.DataFrame:
    if os.path.exists(_CACHE_US):
        try:
            df = pd.read_csv(_CACHE_US)
            if len(df) > 40:
                return df
        except Exception:
            pass
    dfs = []
    sp = _fetch_sp500()
    if not sp.empty:
        dfs.append(sp)
    nq = _fetch_nasdaq100()
    if not nq.empty:
        dfs.append(nq)
    fallback = pd.DataFrame(
        [(s, n, "US") for s, n in SP100_FALLBACK + NASDAQ100_FALLBACK],
        columns=["symbol", "name", "market"],
    )
    dfs.append(fallback)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["symbol"], keep="first")
    try:
        df.to_csv(_CACHE_US, index=False)
    except Exception:
        pass
    return df


def load_india_tickers() -> pd.DataFrame:
    from ticker_data import load_all_tickers

    df = load_all_tickers().copy()
    df["market"] = "IN"
    return df


def load_all_markets() -> pd.DataFrame:
    return pd.concat([load_india_tickers(), load_us_tickers()], ignore_index=True).drop_duplicates(
        subset=["symbol"], keep="first"
    )


def search_tickers(query: str, limit: int = 20, market: str = "all") -> List[Dict[str, str]]:
    if not query or len(query.strip()) < 1:
        return []
    q = query.strip().upper()
    if market == "us":
        df = load_us_tickers()
    elif market in ("in", "india"):
        df = load_india_tickers()
    else:
        df = load_all_markets()
    mask_sym = df["symbol"].astype(str).str.upper().str.contains(q, regex=False, na=False)
    mask_name = df["name"].astype(str).str.upper().str.contains(q, regex=False, na=False)
    matches = df[mask_sym | mask_name].head(limit)
    out = []
    for _, r in matches.iterrows():
        out.append({
            "symbol": str(r["symbol"]),
            "name": str(r["name"]),
            "market": str(r.get("market") or market_for(str(r["symbol"]))),
            "currency": currency_for(str(r["symbol"])),
        })
    return out


def universe_symbols(universe: Optional[str]) -> List[str]:
    """
    Resolve a named universe to Yahoo symbols.
    Default (None / '' / mix): Nifty 50 + US mega-caps.
    """
    from scanner import NIFTY50, NIFTY_NEXT50, DEFAULT_TICKERS

    u = (universe or "mix").strip().lower()
    if u in ("", "default"):
        u = "mix"
    if u == "nifty50":
        return list(NIFTY50)
    if u in ("nifty_next50", "next50"):
        return list(NIFTY_NEXT50)
    if u in ("nifty100", "india_large"):
        return list(dict.fromkeys(NIFTY50 + NIFTY_NEXT50))
    if u in ("all", "all_india", "india"):
        df = load_india_tickers()
        return df["symbol"].astype(str).tolist()
    if u in ("us_mega", "mega"):
        return list(dict.fromkeys(s for s, _ in US_MEGA))
    if u in ("sp100", "s&p100"):
        return list(dict.fromkeys(s for s, _ in SP100_FALLBACK))
    if u in ("nasdaq100", "ndx"):
        return list(dict.fromkeys(s for s, _ in NASDAQ100_FALLBACK))
    if u in ("sp500", "us_all", "all_us"):
        return load_us_tickers()["symbol"].astype(str).tolist()
    if u in ("mix", "both"):
        us = [s for s, _ in US_MEGA]
        return list(dict.fromkeys(list(NIFTY50) + us))
    if u in ("trained", "max", "max_liquid"):
        from pred_model import training_universe
        return training_universe()
    if u in ("intraday", "liquid"):
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "AMD", "NFLX",
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
        ]
    return list(DEFAULT_TICKERS)


def ticker_count(market: str = "all") -> Dict[str, int]:
    india = len(load_india_tickers())
    us = len(load_us_tickers())
    if market in ("in", "india"):
        return {"count": india, "india": india, "us": 0}
    if market == "us":
        return {"count": us, "india": 0, "us": us}
    return {"count": india + us, "india": india, "us": us}
