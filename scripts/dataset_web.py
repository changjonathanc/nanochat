#!/usr/bin/env python3
"""
Dataset visualization web server.

Run with:

python -m scripts.dataset_web
"""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from nanochat.data_viewer import DataUnavailable, StageDataProvider


parser = argparse.ArgumentParser(description="NanoChat dataset visualization server")
parser.add_argument("-p", "--port", type=int, default=8080, help="Port to run the server on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
parser.add_argument("--seed", type=int, default=None, help="Seed for sampling randomness")
args = parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = StageDataProvider(seed=args.seed)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve the dataset viewer UI."""
    ui_path = Path("nanochat") / "dataset_ui.html"
    with ui_path.open("r", encoding="utf-8") as handle:
        html = handle.read()
    return HTMLResponse(content=html)


@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for reuse in the viewer."""
    logo_path = Path("nanochat") / "logo.svg"
    return FileResponse(logo_path, media_type="image/svg+xml")


@app.get("/api/stages")
async def list_stages():
    provider: StageDataProvider = app.state.provider
    return JSONResponse({"stages": provider.list_stages()})


@app.get("/api/random-example")
async def random_example(stage: str, dataset: Optional[str] = None):
    provider: StageDataProvider = app.state.provider
    try:
        record = provider.sample(stage, dataset_key=dataset)
    except DataUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(record)


@app.get("/health")
async def health():
    """Simple readiness probe."""
    provider_ready = hasattr(app.state, "provider")
    return {"status": "ok", "ready": provider_ready}


if __name__ == "__main__":
    import uvicorn

    print("Starting dataset visualization server")
    print(f"Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
