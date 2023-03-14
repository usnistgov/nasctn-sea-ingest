# NASCTN SEA Ingest

This repository is focused on aggregation of "seamf" SEA sensor output data files.

See also:
- [sea-data-product](https://github.com/NTIA/sea-data-product): the latest usage and clear reference implementation for reading the seamf data format


### Installation as a module to call from your own code

```python
pip install git+https://github.com/usnistgov/nasctn-sea-ingest
```

This installs the `sea_ingest` module.

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