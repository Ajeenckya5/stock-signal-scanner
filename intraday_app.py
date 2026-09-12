"""
Dedicated 5-minute intraday app.

    python intraday_app.py
    → http://localhost:8001
"""

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from intraday_routes import router
import autopilot


@asynccontextmanager
async def lifespan(app: FastAPI):
    autopilot.start()
    yield
    autopilot.stop()


STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Broadtape | Intraday 5-minute desk", lifespan=lifespan)
app.include_router(router)


class DeskConfig(BaseModel):
    enabled: Optional[bool] = None
    universe: Optional[str] = None
    candle: Optional[str] = None
    interval_sec: Optional[int] = None
    watchlist: Optional[List[str]] = None
    indicators: Optional[List[str]] = None
    intraday_enabled: Optional[bool] = None


class DeskRun(BaseModel):
    kind: str = "full"


@app.get("/")
def serve_index():
    return FileResponse(STATIC / "intraday.html")


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
    k = (req.kind or "full").lower()
    if k not in ("full", "pulse", "long", "intraday"):
        raise HTTPException(status_code=400, detail="kind must be full, pulse, long, or intraday")
    autopilot.request_run(k)
    return {"status": "queued", "kind": k}


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("intraday_app:app", host="0.0.0.0", port=8001, reload=True)
