# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2023 Timothy H. Click, Ph.D.
#
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#  Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  Neither the name of the author nor the names of its contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#  DAMAGE.
# ------------------------------------------------------------------------------
# pyright: reportInvalidTypeVarUse=false
# flake8: noqa
"""Definitions of base classes for reading and writing topology files."""

import getpass
import time
from collections.abc import Mapping
from typing import TypeVar

import MDAnalysis as mda
from MDAnalysis.coordinates.base import IOBase, _Readermeta, _Writermeta

TTopologyReaderBase = TypeVar("TTopologyReaderBase", bound="TopologyReaderBase")
TTopologyWriterBase = TypeVar("TTopologyWriterBase", bound="TopologyWriterBase")


class TopologyReaderBase(IOBase, metaclass=_Readermeta):
    """Base class for reading topology files."""

    def __init_subclass__(cls, **kwargs: Mapping) -> None:
        super().__init_subclass__(**kwargs)

    def read(self: TTopologyReaderBase) -> None:  # pragma: no cover
        """Read the file.

        Raises
        ------
        NotImplementedError
            if not overridden
        """
        msg = "Override this in each subclass"
        raise NotImplementedError(msg)


class TopologyWriterBase(IOBase, metaclass=_Writermeta):
    """Base class for writing topology files."""

    def __init_subclass__(cls, **kwargs: Mapping) -> None:
        super().__init_subclass__(**kwargs)

    def __init__(self: TTopologyWriterBase) -> None:
        self.title: str = f"""
            * Created by fluctmatch on {time.asctime(time.localtime())}
            * User: {getpass.getuser()}"""

    def write(self: TTopologyWriterBase, selection: mda.Universe | mda.AtomGroup, /) -> None:
        """Write selection at current trajectory frame to file.

        Parameters
        ----------
        selection : Universe or AtomGroup
             group of atoms to be written

        Raises
        ------
        NotImplementedError
            if not overridden
        """
        msg = "Override this in each subclass"
        raise NotImplementedError(msg)
