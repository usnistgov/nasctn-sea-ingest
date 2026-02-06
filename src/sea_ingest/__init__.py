"""
sea_ingest
==========

The `sea_ingest` package provides tools for quickly and conveniently loading NASCTN SEA sensor data files.
These files are outputs from SEA sensors produced after edge compute analysis, following the SigMF format.
The package supports loading sensor metadata from both prototype- and production-era SEA sensors.

The main functionalities include:
- Parsing data products and metadata from `.sigmf` files.
- Packaging data into pandas or Dask DataFrame objects for analysis.
- Handling sensor metadata and data products efficiently.

Modules:
- seamf: Functions for reading and processing SEA sensor data files.
- util: Utility functions for handling timestamps and DataFrame initialization.
- ziparchive: Functions for reading SEA sensor data from zip archives.

Functions:
- read_seamf: Unpacks a sensor data file into a dictionary of numpy or pandas objects.
- read_seamf_meta: Reads metadata from a sensor data file.
- _iso_to_datetime: Converts ISO 8601 timestamp strings to timezone-aware datetimes.
- localize_timestamps: Localizes timestamps in DataFrames to a specified timezone.
- trace: Indexing shortcut for dictionaries of SEA pandas DataFrame data tables.
- read_seamf_zipfile: Reads SEA sensor data files from a zip archive.
- read_seamf_zipfile_as_ddf: Reads SEA sensor data files from a zip archive as Dask DataFrames.
- read_seamf_zipfile_as_delayed: Reads SEA sensor data files from a zip archive as delayed Dask objects.
- restore_multiindex: Restores a multi-index in a DataFrame.
"""

from .seamf import read_seamf, read_seamf_meta
from .util import _iso_to_datetime, localize_timestamps, trace
from .ziparchive import (
    read_seamf_zipfile,
    read_seamf_zipfile_as_ddf,
    read_seamf_zipfile_as_delayed,
    restore_multiindex,
)

__version__ = "1.0.1"
