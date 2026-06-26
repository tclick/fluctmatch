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
"""Unit test suites validating package initialization parameters, metadata states, and version fallback mechanisms.

This verification tier checks the structural integrity of the fluctmatch package entrypoint interface. It focuses on
proving that the environmental setup handles normal installed runtime states as well as uninstalled development pathways
without dropping execution exceptions.
"""

import importlib.metadata
import sys


def test_version_fallback_when_package_not_installed(monkeypatch) -> None:
    """Assert that the package root gracefully falls back to a development version string when uninstalled.

    This test uses state modification arrays to simulate an execution environment where fluctmatch is missing from the active
    package distribution registry. It forces a PackageNotFoundError and validates that the fallback assignment matches the
    development branch specification string.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The standard PyTest diagnostic state modifier fixture utilized to swap runtime library execution vectors.

    Returns
    -------
    None
    """

    # Force importlib metadata parsing to raise the target structural initialization error
    def mock_version_raise(distribution_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", mock_version_raise)

    # Evict fluctmatch from active sys.modules if previously imported, forcing a fresh package evaluation sweep
    if "fluctmatch" in sys.modules:
        monkeypatch.delitem(sys.modules, "fluctmatch")

    # Re-import the root module package context to execute the defensive initialization block
    import fluctmatch

    assert fluctmatch.__version__ == "4.0.0-dev"
