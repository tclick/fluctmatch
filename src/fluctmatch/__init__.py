# ------------------------------------------------------------------------------
#  Copyright (c) 2013-2026 Timothy H. Click
#  Project: fluctmatch
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
#  Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
#  any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for more details.
#
#  You should have received a copy of the GNU General Public License along with this program.  If not, see
#  <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------
"""Automated pipeline setup and workflow generation engine for AMBER molecular dynamics simulations.

The `fluctmatch` package provides a structured framework for managing the execution lifecycles of AMBER molecular
dynamics (MD) workflows. It abstracts complex system validation, experimental restraint determination (NMR vs. X-Ray/EM
baking specific atom selection masks like `@CA` or `@N,CA,C,O`), dynamic Slurm workstation infrastructure profiling,
and hierarchical multi-replica folder tracking.

By integrating Pydantic schemas for parameter verification, Typer for modular CLI command groups, and Rich for
interactive terminal feedback, this package guarantees that structural parameterization and execution boundaries are
caught before launching long-timescale simulations on shared multi-user workstations.

Modules
-------
commands.registry : CommandRegistry
    Scans the package tree and dynamically hooks command subsystems into the primary application framework.
config : SimulationConfig, MDStage, SlurmConfig
    Binds and processes nested user configurations, handling runtime validation against actual hardware limits.
system : SystemResources, get_system_resources
    Utilizes system APIs to inspect physical core counts, available RAM boundaries, and local NVIDIA GPU hardware.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version(distribution_name="fluctmatch")
except importlib.metadata.PackageNotFoundError:
    # Fallback default if package isn't installed in the active environment path
    __version__ = "4.0.0-dev"
