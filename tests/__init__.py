# ------------------------------------------------------------------------------
#  Copyright (c) 2026 Timothy H. Click
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
"""Test suite initialization module for the $project.Name package.

This module marks the `tests` directory as a recognizable Python package. It establishes the root test namespace and
ensure proper test discovery, package isolation, and test-environment consistency across the entire test suite.

Notes
-----
While this initialization file remains sparse to maintain test isolation and prevent unintended side effects during
module collection, it acts as the baseline configuration entry point for test runners like `pytest` or `unittest`.

For advanced runtime fixtures, global test configuration hooks, or shared cross-module testing utilities, prefer
defining them inside a dedicated `conftest.py` file or a specialized `tests.helpers` submodule rather than adding
heavy executable logic directly within this package initializer.

Examples
--------
To discover and execute all unit, integration, and functional tests registered under this namespace, run the following
command from the root directory of the repository:

$ pytest tests/
"""
