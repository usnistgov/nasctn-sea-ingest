# NASCTN SEA Ingest

This repository is focused on aggregation of "seamf" SEA sensor output data files.

See also:
- [sea-data-product](https://github.com/NTIA/sea-data-product): the latest usage and clear reference implementation for reading the seamf data format


## Environment Setup


1. Clone this repository:

   ```bash
   git clone https://github.com/usnistgov/nasctn-sea-ingest
   ```

2. Environment setup:
   - Make sure you've installed python 3.8 or newer making sure to include `pip` for base package management (for example, with `conda` or `miniconda`)
   - Make sure you've installed `pdm`, which is used for dependency management isolated from other projects. To do so, run the following in a command line environment:

      ```bash
      pip install pdm
      ```

      _At this point, close and reopen open a new one to apply updated command line variables_

      ```bash
      pdm config install.cache on
      ```
   - Install the project environment with `pdm`:

      ```bash
      pdm use      
      pdm install
      ```

