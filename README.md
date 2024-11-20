# NASCTN SEA Ingest

This repository contains `sea_ingest`, a Python package which can quickly and conveniently load many NASCTN SEA sensor data files. These are the outputs from SEA sensors produced after edge compute analysis, which follow the [SigMF](https://github.com/sigmf/sigmf) format. The `sea_ingest` package supports loading sensor metadata from both prototype- and production-era SEA sensors.

This tool parses data products and metadata from `.sigmf` files and packages them into [pandas](https://pandas.pydata.org/) or [Dask](https://www.dask.org/) DataFrame objects. From there, the data can be analyzed directly, or piped into files or databases using Dask multiprocessing hooks.

## Background

The National Advanced Spectrum and Communications Test Network ([NASCTN](https://www.nist.gov/ctl/nasctn/about)) aims to collect the data required to ascertain the effectiveness of the Citizens Broadband Radio Service (CBRS) sharing ecosystem through the Sharing Ecosystem Assessment ([SEA](https://www.nist.gov/programs-projects/cbrs-sharing-ecosystem-assessment)) project. Towards this end, NASCTN has designed and deployed a number of RF sensors across the country in key locations of interest. The sensors record IQ waveforms and process them at the edge into various informative data products, which are then pushed back to a central NASCTN data repository. Detailed information about the project, sensor design, and data processing algorithm is available in the [NASCTN SEA CBRS Task 2 Draft Test Plan](https://www.nist.gov/document/nasctn-cbrs-sea-task-2-draft-test-plan-december-2023).

## Data Format

The data files produced by SEA sensors conform to the [SigMF](https://github.com/sigmf/sigmf) standard, and use the [extension namespace defined by NTIA](https://github.com/NTIA/sigmf-ns-ntia). Each `.sigmf` file is a tarball containing a binary `.sigmf-data` file and a JSON `.sigmf-meta` file. The data file contains LZMA-compressed numerical data, the results of the edge compute data processing algorithm. The metadata file contains the necessary information to parse the data file, along with sensor diagnostics and other information. The easiest way to parse and interact with these files is using this tool, which transparently maps the `.sigmf` file contents into DataFrames.

## Installation

This makes the `sea_ingest` module available for import from your own python scripts.

```bash
pip install git+https://github.com/usnistgov/nasctn-sea-ingest@0.6.4
```

## Usage

Once installed, the module is importable in any Python program. For some usage examples,
see the Jupyter notebooks provided in [`demos`](https://github.com/usnistgov/nasctn-sea-ingest/tree/main/demos).
