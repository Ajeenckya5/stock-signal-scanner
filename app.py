from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scanner import scan_tickers, INDICATOR_CATALOG
from ticker_data import get_nifty50_tickers, load_all_tickers
from markets import search_tickers, ticker_count, universe_symbols, benchmark_symbol
from signals_store import add_to_lists, get_lists, clear_lists
from rlhf import record_feedback, get_stats, reset_weights
from chart_analysis import build_chart_payload
from ohlc import CANDLE_SPECS, LONGTERM_CANDLES, normalize_candle, fetch_ohlcv
from quant_engine import RESEARCH_CATALOG, compute_quant_bundle, try_fundamentals
from intraday_routes import router as intraday_router
import autopilot


@asynccontextmanager
async def lifespan(app: FastAPI):
    autopilot.start()
    yield
    autopilot.stop()


STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Broadtape | 24/7 US + India insights desk", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ajeenckya5.github.io",
        "https://predi-stock.onrender.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(intraday_router, prefix="/intraday")


class TrainRequest(BaseModel):
    ticker: str
    country: str = "in"


class TrainResponse(BaseModel):
    ticker: str
    checkpoint_path: str
    daily_pickle_path: str


class AgentRequest(BaseModel):
    ticker: str
    country: str = "in"
    checkpoint_path: Optional[str] = None


class AgentResponse(BaseModel):
    ticker: str
    action: str
    expected_return: float
    buy_threshold: float
    sell_threshold: float
    last_bar: str
    bars_used: int


class ScanRequest(BaseModel):
    tickers: Optional[List[str]] = None
    period: Optional[str] = None
    filter_action: Optional[str] = None
    universe: Optional[str] = None  # mix | nifty50 | all | us_mega | sp100 | nasdaq100 | sp500
    save_to_lists: bool = False  # add results to predefined BUY/SELL lists
    # Subset of indicator ids for composite score (see GET /indicators/catalog). None = use all.
    indicators: Optional[List[str]] = None
    candle: str = "24h"  # 24h | 1mo | 3mo | 6mo
    market: Optional[str] = None  # all | us | in (search only)


@app.get("/")
@app.get("/index.html")
def serve_index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/intraday")
@app.get("/intraday.html")
def serve_intraday():
    return FileResponse(Path(__file__).parent / "static" / "intraday.html")


@app.get("/desk.css")
def serve_desk_css():
    return FileResponse(STATIC / "desk.css", media_type="text/css")


@app.get("/search")
def search_endpoint(q: str = "", limit: int = 20, market: str = "all"):
    """Search US + Indian stocks by symbol or company name."""
    results = search_tickers(q, limit=limit, market=market)
    return {"tickers": results}


@app.get("/tickers/count")
def ticker_count_endpoint(market: str = "all"):
    """Return ticker counts by market."""
    return ticker_count(market)


@app.get("/tickers/nifty50")
def nifty50_endpoint():
    """Return all Nifty 50 tickers for scan."""
    return {"tickers": get_nifty50_tickers(), "count": 50}


@app.get("/universes")
def universes_endpoint():
    return {
        "universes": [
            {"id": "liquid", "label": "Liquid mix (fast, 18 names)"},
            {"id": "trained", "label": "Trained set (Nifty 50 + S&P 100)"},
            {"id": "mix", "label": "Nifty 50 + US mega-caps"},
            {"id": "nifty50", "label": "Nifty 50 (India)"},
            {"id": "nifty_next50", "label": "Nifty Next 50"},
            {"id": "all", "label": "All Indian names in DB"},
            {"id": "us_mega", "label": "US mega-caps"},
            {"id": "sp100", "label": "S&P 100"},
            {"id": "nasdaq100", "label": "Nasdaq-100"},
            {"id": "sp500", "label": "S&P 500 (slow)"},
        ]
    }


@app.get("/candles")
def candles_endpoint():
    return {
        "candles": [
            {"id": k, "label": CANDLE_SPECS[k]["label"], "ann": CANDLE_SPECS[k]["ann"]}
            for k in LONGTERM_CANDLES
        ]
    }


@app.get("/research/catalog")
def research_catalog_endpoint():
    return {"papers": RESEARCH_CATALOG}


@app.get("/research/{ticker}")
def research_endpoint(ticker: str, candle: str = "24h", period: Optional[str] = None):
    """Full quant dossier + optional fundamentals (slow)."""
    cndl = normalize_candle(candle)
    df, meta = fetch_ohlcv(ticker, candle=cndl, period=period)
    if df.empty:
        raise HTTPException(status_code=400, detail=meta.get("error") or "no data")
    bdf, _ = fetch_ohlcv(benchmark_symbol(ticker), candle=cndl, period=period)
    bench = bdf["close"] if not bdf.empty else None
    bundle = compute_quant_bundle(df, ticker, candle=cndl, benchmark=bench)
    bundle["fundamentals"] = try_fundamentals(ticker)
    bundle["papers"] = RESEARCH_CATALOG
    return bundle


@app.get("/indicators/catalog")
def indicators_catalog_endpoint():
    """Ids and labels for manual indicator selection (scoring)."""
    return {"indicators": INDICATOR_CATALOG}


@app.get("/chart/{ticker}")
def chart_live_endpoint(
    ticker: str,
    response: Response,
    period: str = "2y",
    interval: str = "1d",
    fwd_days: int = 5,
    candle: str = "24h",
):
    """
    OHLCV candles + pattern markers & S/R lines + historical forward-edge prediction.
    candle: 24h | 1mo | 3mo | 6mo
    """
    response.headers["Cache-Control"] = "no-store, max-age=0"
    payload = build_chart_payload(
        ticker,
        period=period,
        interval=interval,
        fwd_days=max(1, min(fwd_days, 20)),
        candle=candle,
    )
    err = payload.get("error")
    if err:
        raise HTTPException(status_code=400, detail=str(err))
    return payload


@app.get("/lists")
def lists_endpoint():
    """Return predefined BUY / SELL lists from stored scan results."""
    return get_lists()


@app.delete("/lists")
def clear_lists_endpoint():
    """Clear stored BUY/SELL lists."""
    clear_lists()
    return {"status": "cleared"}


class FeedbackRequest(BaseModel):
    ticker: str
    action: str  # BUY | SELL
    was_correct: bool
    factors_used: Optional[List[str]] = None


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    """RLHF: Record if prediction was correct. Algo improves from feedback."""
    record_feedback(
        ticker=req.ticker,
        action=req.action.upper(),
        was_correct=req.was_correct,
        factors_present=req.factors_used,
    )
    return get_stats()


@app.get("/feedback/stats")
def feedback_stats_endpoint():
    """Return RLHF stats: feedback count, accuracy, learned weights."""
    return get_stats()


@app.post("/feedback/reset")
def feedback_reset_endpoint():
    """Reset RLHF weights to defaults."""
    return {"weights": reset_weights()}


@app.post("/scan")
def scan_endpoint(req: ScanRequest):
    try:
        tickers = req.tickers
        candle = normalize_candle(req.candle)
        period = req.period
        if not period:
            period = "max" if candle in ("1mo", "3mo", "6mo") else "2y"
        if not tickers:
            if req.universe == "nifty50":
                tickers = [t["symbol"] for t in get_nifty50_tickers()]
            elif req.universe == "all":
                df = load_all_tickers()
                tickers = df["symbol"].astype(str).tolist()
            else:
                tickers = universe_symbols(req.universe)
        results = scan_tickers(
            tickers=tickers,
            period=period,
            filter_action=req.filter_action,
            indicators=req.indicators,
            candle=candle,
            universe=req.universe,
        )
        if req.save_to_lists and results:
            add_to_lists(results)
        return {"results": results, "count": len(results), "candle": candle}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest):
    try:
        from service import train_for_ticker
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="ML training requires torch/transformers. Use local install with full requirements.txt"
        )
    try:
        ckpt, pkl = train_for_ticker(req.ticker, req.country)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return TrainResponse(
        ticker=req.ticker.upper(),
        checkpoint_path=ckpt,
        daily_pickle_path=pkl,
    )


@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: AgentRequest):
    try:
        from service import agent_predict_once_service
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="ML agent requires torch/transformers. Use local install with full requirements.txt"
        )
    ckpt = req.checkpoint_path or f"news_stock_mdn_{req.ticker.upper()}.pt"
    try:
        result = agent_predict_once_service(req.ticker, req.country, ckpt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return AgentResponse(**result)


class DeskConfig(BaseModel):
    enabled: Optional[bool] = None
    universe: Optional[str] = None
    candle: Optional[str] = None
    interval_sec: Optional[int] = None
    watchlist: Optional[List[str]] = None
    indicators: Optional[List[str]] = None
    intraday_enabled: Optional[bool] = None
    max_names: Optional[int] = None


class DeskRun(BaseModel):
    kind: str = "full"


@app.get("/desk/snapshot")
def desk_snapshot():
    return autopilot.get_snapshot()


@app.get("/desk/status")
def desk_status():
    return autopilot.get_status()


@app.post("/desk/config")
def desk_config(req: DeskConfig):
    patch = req.model_dump(exclude_unset=True) if hasattr(req, "model_dump") else req.dict(exclude_unset=True)
    return autopilot.update_config(patch)


@app.post("/desk/run")
def desk_run(req: DeskRun):
    kind = (req.kind or "full").lower()
    if kind not in ("full", "pulse", "long", "intraday", "live", "quotes"):
        raise HTTPException(status_code=400, detail="kind must be full, pulse, long, intraday, live, or quotes")
    autopilot.request_run(kind)
    return {"status": "queued", "kind": kind}


@app.get("/model")
def model_report():
    from pred_model import get_report
    return get_report()


@app.post("/model/train")
def model_train():
    from pred_model import start_train_async
    return start_train_async()


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
