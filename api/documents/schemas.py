import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from api.documents.models import DocStatus


class DocumentOut(BaseModel):
    """Document fields safe to expose to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: DocStatus
    created_at: datetime
    updated_at: datetime
