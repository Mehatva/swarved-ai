from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from database import insert_incident
from routes.threat_analytics import connection_manager
from schemas.incident_schema import CyberIncidentPayload


router = APIRouter(
    prefix="/api/v1/i4c",
    tags=["I4C Dispatch"],
)


def generate_i4c_reference_number() -> str:
    """
    Generate a unique simulated I4C reference number.

    Format:
        I4C-YYYY-XXXXXXXX

    Example:
        I4C-2026-A1B2C3D4
    """
    year = datetime.now(timezone.utc).year
    unique_suffix = uuid4().hex[:8].upper()

    return f"I4C-{year}-{unique_suffix}"


@router.post(
    "/dispatch",
    status_code=status.HTTP_201_CREATED,
)
async def dispatch_incident(payload: CyberIncidentPayload) -> dict[str, str]:
    """
    Register a cyber incident with the simulated National Cyber Crime
    Helpline Gateway.

    The incident is validated by Pydantic and persisted to SQLite.
    """
    i4c_reference_number = generate_i4c_reference_number()

    incident = payload.model_dump(mode="json")

    incident["incident_id"] = str(payload.incident_id)
    incident["timestamp"] = payload.timestamp.isoformat()
    incident["i4c_reference_number"] = i4c_reference_number

    try:
        await insert_incident(incident)
        await connection_manager.broadcast(incident)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register incident in the local database.",
        ) from exc

    return {
        "status": "SUCCESS",
        "i4c_reference_number": i4c_reference_number,
        "message": (
            "Incident registered at National Cyber Crime "
            "Helpline Gateway."
        ),
    }