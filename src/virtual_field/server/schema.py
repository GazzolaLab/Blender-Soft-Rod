from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = 1


def make_message(message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "type": message_type,
        "payload": payload,
    }


def validate_message(message: dict[str, Any]) -> None:
    if (
        "version" not in message
        or "type" not in message
        or "payload" not in message
    ):
        raise ValueError("message must contain version, type, and payload")
    if message["version"] != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {message['version']}")
    if not isinstance(message["type"], str):
        raise ValueError("message type must be a string")
    if not isinstance(message["payload"], dict):
        raise ValueError("message payload must be a dictionary")
