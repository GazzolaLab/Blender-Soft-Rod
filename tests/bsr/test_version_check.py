import re


def test_version():
    import bsr

    assert hasattr(bsr, "version")

    # Check if the version is a string
    assert isinstance(bsr.version, str)

    # Check if the version string is in the correct format
    assert re.match(r"\d+\.\d+\.\d+", bsr.version)
