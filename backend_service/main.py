from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import initialize_database
from routes.i4c_dispatch import router as i4c_dispatch_router
from routes.threat_analytics import router as threat_analytics_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    """
    await initialize_database()
    yield


app = FastAPI(
    title="SwarVed AI - National DPI Interoperability Backend",
    description=(
        "FastAPI backend for automated cyber incident registration "
        "and live threat analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(i4c_dispatch_router)
app.include_router(threat_analytics_router)


@app.get("/", tags=["Health"])
async def health_check() -> dict[str, str]:
    """
    Basic service health endpoint.
    """
    return {
        "status": "ONLINE",
        "service": "SwarVed AI National DPI Interoperability Backend",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )