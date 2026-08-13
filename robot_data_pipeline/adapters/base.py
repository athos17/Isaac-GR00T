from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from robot_data_pipeline.models import AdaptedPayload, StreamConfig


class MessageAdapter(ABC):
    @abstractmethod
    def adapt(self, message: Any, stream: StreamConfig) -> AdaptedPayload:
        """Extract the fields used by canonicalization and raw QA."""
