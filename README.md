# NASCTN SEA Ingest


[![GitHub release (latest SemVer)][latest-release-semver-badge]][github-releases]
[![GitHub all releases][github-download-count-badge]][github-releases]
[![GitHub issues][github-issue-count-badge]][github-issues]

[latest-release-semver-badge]: https://img.shields.io/github/v/release/usnistgov/nasctn-sea-ingest?display_name=tag&sort=semver
[github-releases]: https://github.com/usnistgov/nasctn-sea-ingest/releases
[github-download-count-badge]: https://img.shields.io/github/downloads/usnistgov/nasctn-sea-ingest/total
[github-issue-count-badge]: https://img.shields.io/github/issues/usnistgov/nasctn-sea-ingest
[github-issues]: https://github.com/usnistgov/nasctn-sea-ingest/issues

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

## Quick Start Guide

Here is a quick example to get you started:

```python
from sea_ingest import read_seamf

# Path to your .sigmf file
file_path = "path/to/your/file.sigmf"

# Read the data
data = read_seamf(file_path)

# Access, for example, the PSD data. In a Jupyter notebook, skip the "print()"!
print(data["psd"])
```

## Usage

Once installed, the module is importable in any Python program. For some usage examples,
see the Jupyter notebooks provided in [`demos`](https://github.com/usnistgov/nasctn-sea-ingest/tree/main/demos).

## Development

To contribute to this repository, clone it and install the package with development dependencies:

```cmd
git clone https://github.com/usnistgov/nasctn-sea-ingest
cd nasctn-sea-ingest
pip install .[dev]
```

### Packaging New Releases

Once you've made changes and wish to issue a new release, increment the version number and build a wheel. The backend used by this project, [Hatchling](https://github.com/pypa/hatch/tree/master/backend), makes this easy. To change the version number:

```cmd
hatchling version major  # 1.0.0 -> 2.0.0
hatchling version minor  # 1.0.0 -> 1.1.0
hatchling version micro  # 1.0.0 -> 1.0.1
hatchling version "X.X.X"  # 1.0.0 -> X.X.X
```

Then, to build a wheel and source distribution:

```cmd
hatchling build
```

### Running Tests

Unit tests are included to test the `read_seamf` and `read_seamf_meta` functions by loading the included example data files. The testing packages [pytest](https://docs.pytest.org/en/7.1.x/) and [tox](https://tox.wiki/en/latest/) are installed with the development dependencies, and can be used as follows.

```cmd
pytest          # faster, but less thorough
tox             # test code in virtual environments for multiple versions of Python
tox --recreate  # To recreate the virtual environments used for testing
```

## Disclaimer

"Certain commercial equipment, instruments, or materials (or suppliers, or software, ...) are identified in this paper to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the materials or equipment identified are necessarily the best available for the purpose.
