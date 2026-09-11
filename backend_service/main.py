from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    CORS_ALLOW_ORIGINS,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_RELOAD,
)
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
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
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
        "service": APP_TITLE,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=SERVER_RELOAD,
    )