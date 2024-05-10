# Fluctuation matching

[![PyPI](https://img.shields.io/pypi/v/fluctmatch.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/fluctmatch.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/fluctmatch)][pypi status]
[![License](https://img.shields.io/pypi/l/fluctmatch)][license]

[![Read the documentation at https://fluctmatch.readthedocs.io/](https://img.shields.io/readthedocs/fluctmatch/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/tclick/fluctmatch/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/tclick/fluctmatch/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](ruff)
[ [MDAnalysis](https://img.shields.io/badge/Powered%20by-MDAnalysis-orange.svg)][mdanalysis]

[pypi status]: https://pypi.org/project/fluctmatch/
[read the docs]: https://fluctmatch.readthedocs.io/
[tests]: https://github.com/tclick/fluctmatch/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/tclick/fluctmatch
[pre-commit]: https://github.com/pre-commit/pre-commit
[ruff]: https://github.com/astral-sh/ruff
[mdanalysis]: https://www.mdanalysis.org

## Introduction

Fluctuation matching is a different approach to protein structural analysis.
Typically, an elastic network model (ENM) creates springs between coarse-grain
sites and then uses normal mode analysis (NMA) to determine the vibrational
information that occurs within the structure; most ENMs emply study proteins
representing each residue at the C-alpha position. Fluctuation matching has been
programmed to incorporate more molecules of interest, including nucleic acids,
solvent, and ions. Residues can either be represented on the alpha-carbon; an
alpha-carbon and the sidechain; or the amino group, carboxyl group, and
sidechain.

Previous versions of fluctuation matching strictly employed CHARMM for all
calculations (average structure, initial bond statistics, and NMA). Furthermore,
the directory structure was such that all average structures were used to
determine the springs available within the system. The current code base as been
completely retooled allowing for easier definitions of additional coarse-grain
models, additions to analysis, and implementation of other MD packages. Because
MDAnalysis 2.0+ has also been employed (compared with 0.10.0 for fluctmatch
2.0), greater improvements have been made in the efficiency of the code.

## Features

- Align trajectories to a reference to remove translation and rotation.
- Create initial subdirectories for each window of fluctuation matching.
- Split trajectories with a certain number of frames per trajectory.
- Convert an all-atom molecular dynamics (MD) trajectory to a coarse-grain (CG) model.
- Create the topology files to represent the elastic network model (ENM).
- Run fluctuation matching for each window.
- Analyze the data per window.
- Combine the data into a table for further analysis.
- Perform JADE independent component analysis (ICA) on the accumulated data.
- Perform cluster analysis of the data.

## Requirements

- Python >=3.11
- MDAnalysis >= 2.0
- click
- loguru
- pandas
- parmed
- CHARMM >= 36

## Installation

You can install _Fluctuation matching_ via [pip] from [PyPI]:

```console
$ pip install fluctmatch
```

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
_Fluctuation matching_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/tclick/fluctmatch/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/tclick/fluctmatch/blob/main/LICENSE.md
[contributor guide]: https://github.com/tclick/fluctmatch/blob/main/CONTRIBUTING.md
[command-line reference]: https://fluctmatch.readthedocs.io/en/latest/usage.html
