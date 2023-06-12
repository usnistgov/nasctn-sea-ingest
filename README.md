# NASCTN SEA Ingest

This repository is focused on fast and convenient loading of many "seamf" SEA sensor output data files. These are the outputs from SEA sensors produced after edge compute analysis, with formatting based on SIGMF. Currently, `sea_ingest` supports file metadata versions {0.1, 0.2, 0.3, 0.4}.

The data products and metadata are packaged into pandas or dask DataFrame objects. From there, the data can be analyzed directly, or piped into files or databases using dask multiprocessing hooks.

See also:
- [sea-data-product](https://github.com/NTIA/sea-data-product): the latest usage and clear reference implementation for reading the seamf data format


### Install as a module
This makes the `sea_ingest` module available for import from your own python scripts.

#### For coexistence with conda or mamba package managers
The idea here is to [re-use pre-existing base libraries when possible](https://www.anaconda.com/blog/using-pip-in-a-conda-environment) to minimize interaction problems between the two package managers.

```python
pip install --upgrade-strategy only-if-needed git+https://github.com/usnistgov/nasctn-sea-ingest
```

#### For pip-only ("bare python") package management
```python
pip install git+https://github.com/usnistgov/nasctn-sea-ingest
```

### Setting up the development environment to run notebooks or develop nasctn-sea-ingest
The following apply if you'd like to clone the project to develop the 

1. Clone this repository:

   ```bash
   git clone https://github.com/usnistgov/nasctn-sea-ingest
   ```

2. Environment setup:
   - Make sure you've installed python 3.8 or newer making sure to include `pip` for base package management (for example, with `conda` or `miniconda`)
   - Make sure you've installed `pdm`, which is used for dependency management isolated from other projects. This only needs to be done once in your python environment (not once per project). To do so, run the following in a command line environment:

      ```bash
      pip install pdm
      ```

      _At this point, close and reopen open a new one to apply updated command line variables_
   - Install the project environment with `pdm`:

      ```bash
      pdm use      
      pdm install
      ```

## Usage
See [`demos`](https://github.com/usnistgov/nasctn-sea-ingest/tree/main/demos).