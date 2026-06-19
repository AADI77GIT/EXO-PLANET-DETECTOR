from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.database import init_db
from app.ml.autoencoder import load_autoencoder
from app.ml.cnn_classifier import initialize_classifier
from app.routers import health_router, pipeline_router, results_router, train_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    load_autoencoder()
    initialize_classifier()
    yield


limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app = FastAPI(title="ExoPlanet Detection API", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}"})


app.include_router(pipeline_router)
app.include_router(results_router)
app.include_router(health_router)
app.include_router(train_router)


@app.get("/")
async def root():
    return {"service": "ExoPlanet Detection API", "docs": "/docs", "health": "/api/health"}
