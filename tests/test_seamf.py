import pytest
from sea_ingest.seamf import read_seamf, read_seamf_meta
from pathlib import Path

ALL_DEMO_FILES = {
    "example_v1.sigmf",
    "example_v2.sigmf",
    "example_v3.sigmf",
    "example_v4.sigmf",
    "example_v5.sigmf",
    "example_v6.sigmf",
}

DATA_DIRECTORY = Path("demos/data")

def test_read_seamf():
    for file in ALL_DEMO_FILES:
        data = read_seamf(DATA_DIRECTORY / file, tz="America/New_York")
        assert "psd" in data

def test_read_seamf_meta():
    for file in ALL_DEMO_FILES:
        metadata = read_seamf_meta(DATA_DIRECTORY / file, tz="America/New_York")
        assert hasattr(metadata, "global_")

def test_read_seamf_meta_autotimezone():
    for file in ALL_DEMO_FILES:
        # Data before v4 cannot be loaded without manual timezone info
        if "v1" in file or "v2" in file or "v3" in file:
            with pytest.raises(ValueError):
                metadata = read_seamf_meta(DATA_DIRECTORY / file)
            continue
        metadata = read_seamf_meta(DATA_DIRECTORY / file)
        assert hasattr(metadata, "timezone")
