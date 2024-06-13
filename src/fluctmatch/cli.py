# ---------------------------------------------------------------------------------------------------------------------
# fluctmatch
# Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
#
# This file is part of fluctmatch.
#
# Fluctmatch is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Fluctmatch is distributed in the hope that it will be useful, # but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <[1](https://www.gnu.org/licenses/)>.
#
# Reference:
# Timothy H. Click, Nixon Raj, and Jhih-Wei Chu. Simulation. Meth Enzymology. 578 (2016), 327-342,
# Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics doi:10.1016/bs.mie.2016.05.024.
# ---------------------------------------------------------------------------------------------------------------------
# pyright: reportArgumentType=false
"""Module that contains the command line app."""

from pathlib import Path

import ccl
import click
from click_help_colors import HelpColorsGroup, version_option

from fluctmatch import NAME, __copyright__, __version__


@click.group(
    name="fluctmatch",
    help=__copyright__,
    cls=HelpColorsGroup,
    help_headers_color="yellow",
    help_options_color="blue",
    context_settings={"max_content_width": 120},
)
@version_option(
    version=__version__, prog_name=NAME, version_color="blue", prog_name_color="yellow", message="%(prog)s %(version)s"
)
@click.help_option("-h", "--help", help="Show this help message and exit")
def main() -> None:
    """Console script for fluctmatch."""
    pass


plugin_folder = Path(__file__).parent.joinpath("commands").resolve()
ccl.register_commands(main, plugin_folder)
