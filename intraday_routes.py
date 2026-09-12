"""Intraday 5-minute API routes. Mounted at /intraday on the long-term app, or at / on port 8001."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from intraday_engine import analyze_intraday, scan_intraday
from markets import search_tickers, universe_symbols

router = APIRouter()


class IntraScanRequest(BaseModel):
    tickers: Optional[List[str]] = None
    universe: Optional[str] = "intraday"
    filter_action: Optional[str] = None


@router.get("/search")
def intra_search(q: str = "", limit: int = 20, market: str = "all"):
    return {"tickers": search_tickers(q, limit=limit, market=market)}


@router.post("/scan")
def intra_scan(req: IntraScanRequest):
    try:
        tickers = req.tickers
        if not tickers:
            tickers = universe_symbols(req.universe or "intraday")
        results = scan_intraday(tickers, filter_action=req.filter_action)
        return {"results": results, "count": len(results), "candle": "5m"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/chart/{ticker}")
def intra_chart(ticker: str):
    payload = analyze_intraday(ticker)
    if payload.get("action") == "ERROR":
        raise HTTPException(status_code=400, detail=payload.get("error") or "no data")
    return payload
