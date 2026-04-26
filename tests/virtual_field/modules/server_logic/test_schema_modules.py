import pytest

from virtual_field.server.schema import make_message, validate_message

pytestmark = pytest.mark.modules


def test_schema_round_trip() -> None:
    message = make_message("hello", {"client": "test"})
    validate_message(message)


def test_schema_rejects_invalid_version() -> None:
    message = {"version": 99, "type": "hello", "payload": {}}
    with pytest.raises(ValueError, match="unsupported protocol version"):
        validate_message(message)
