# python-project

> ⚠️ This project is in early development. APIs and CLI commands may change without notice.

`python-project` is a command-line tool for scaffolding molecular dynamics (MD) simulation projects. It automates the creation of standardised directory structures and generates input files for the [Amber](https://ambermd.org/) MD package, along with documentation stubs for each stage of a simulation workflow.

---

## Features

- **Directory scaffolding** — generate a consistent, reproducible project layout for MD simulations
- **Amber input files** — write `.in` input scripts for Amber MD stages (minimisation, equilibration, production, etc.)
- **README generation** — populate subdirectories with pre-filled `README.md` stubs describing their purpose
- **Extensible subcommand architecture** — new simulation engines and workflows can be added as plugins

---

## Requirements

- Python 3.13+
- [Typer](https://typer.tiangolo.com/)
- [Rich](https://rich.readthedocs.io/)

---

## Installation

`python-project` is not yet published to PyPI. Install directly from source:

```bash
git clone https://github.com/tclick/python-project.git
cd python-project
pip install -e .
```

---

## Usage

```bash
# Show available subcommands
python-project --help

# Scaffold a new simulation project directory
python-project scaffold --name my_simulation

# Write Amber input scripts into an existing project
python-project amber write-inputs --project ./my_simulation
```

> Command names and options are subject to change during early development.

---

## Project Layout

A scaffolded project typically looks like:

```
my_simulation/
├── README.md
├── 00_prep/          # System preparation (topology, coordinates)
├── 01_min/           # Energy minimisation
├── 02_equil/         # Equilibration
├── 03_prod/          # Production runs
└── analysis/         # Post-simulation analysis
```

Each subdirectory contains a `README.md` describing its purpose and expected inputs/outputs.

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for notes on working with the codebase.

---

## License

`python-project` is free software released under the [GNU General Public License v3.0](LICENSE). You are free to use, modify, and distribute it under the same terms.
