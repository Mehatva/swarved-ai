from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.types import UUID4


class CyberIncidentPayload(BaseModel):
    """
    Payload sent by the SwarVed AI Android client when a suspected
    cyber incident is detected.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    incident_id: UUID4

    timestamp: datetime

    caller_id: Annotated[
        str,
        Field(min_length=1, max_length=255),
    ]

    receiver_device_hash: Annotated[
        str,
        Field(min_length=1, max_length=255),
    ]

    synthetic_voice_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    audio_fingerprint_hash: Annotated[
        str,
        Field(min_length=1, max_length=255),
    ]

    gps_lat: float = Field(
        ge=-90.0,
        le=90.0,
    )

    gps_lon: float = Field(
        ge=-180.0,
        le=180.0,
    )

    threat_category: str = Field(
        default="AI_VOICE_CLONING_DIGITAL_ARREST",
        min_length=1,
        max_length=100,
    )

    @field_validator("timestamp")
    @classmethod
    def validate_and_normalize_timestamp(
        cls,
        value: datetime,
    ) -> datetime:
        """
        Require timezone information and normalize the timestamp to UTC.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "timestamp must contain timezone information"
            )

        return value.astimezone(timezone.utc)