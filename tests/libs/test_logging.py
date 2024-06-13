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
"""Tests for logging module."""

from pathlib import Path
from typing import Any, Self

from fluctmatch.libs import logging
from loguru import logger


class TestLogging:
    """Tests for logging module."""

    def test_logging(self, tmp_path: Path, capsys: Any) -> None:
        """Test logging functionality."""
        log_file = tmp_path / "test.log"
        logging.config_logger(name=__name__, logfile=log_file, level="DEBUG")

        message = "This is a debug message"
        logger.debug(message)
        _, err = capsys.readouterr()

        assert log_file.exists()
        assert log_file.stat().st_size > 0
        assert message in err
        with log_file.open() as f:
            line = f.readline()
            assert message in line

    def test_logging_with_no_file(self: Self, tmp_path: Path, capsys: Any) -> None:
        """Test logging functionality with no log file."""
        log_file = tmp_path / "test.log"

        logging.config_logger(name=__name__, level="DEBUG")
        message = "This is a debug message"
        logger.debug(message)
        _, err = capsys.readouterr()

        assert not log_file.exists()
        assert message in err
