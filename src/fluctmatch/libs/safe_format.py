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
# pyright: reportMissingImports=false, reportInvalidTypeVarUse=false
# flake8: noqa
"""Format strings safely.

The normal way to format strings in Python can be considered unsafe. This offers a safer way to format strings. The
original code comes the blog of Armin Ronacher.

See Also
--------
https://lucumr.pocoo.org/2016/12/29/careful-with-str-format/
"""

from collections.abc import Iterator, Mapping
from string import Formatter
from typing import Any, TypeVar

# This is a necessary API but it's undocumented and moved around
# between Python releases
try:
    from _string import formatter_field_name_split
except ImportError:
    formatter_field_name_split = lambda x: x._formatter_field_name_split()

TMagicFormatMapping = TypeVar("TMagicFormatMapping", bound="MagicFormatMapping")
TSafeFormatter = TypeVar("TSafeFormatter", bound="SafeFormatter")


class MagicFormatMapping(Mapping):
    """Implement a dummy wrapper to fix a bug in the Python standard library for string formatting.

    See http://bugs.python.org/issue13598 for information about why this is necessary.
    """

    def __init__(self: TMagicFormatMapping, args, kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._last_index = 0

    def __getitem__(self: TMagicFormatMapping, key: str):
        if key == "":
            idx = self._last_index
            self._last_index += 1
            try:
                return self._args[idx]
            except LookupError:
                pass
            key = str(idx)
        return self._kwargs[key]

    def __iter__(self: TMagicFormatMapping) -> Iterator:
        return iter(self._kwargs)

    def __len__(self: TMagicFormatMapping) -> int:
        return len(self._kwargs)


class SafeFormatter(Formatter):
    def get_field(self: TSafeFormatter, field_name: str, args, kwargs) -> tuple:
        first, rest = formatter_field_name_split(field_name)
        obj = self.get_value(first, args, kwargs)
        for is_attr, i in rest:
            obj = safe_getattr(obj, i) if is_attr else obj[i]
        return obj, first


def safe_getattr(obj: Any, attr: str) -> Any:
    # Expand the logic here.  For instance on 2.x you will also need to disallow func_globals, on 3.x you will also
    # need to hide things like cr_frame and others.  So ideally have a list of objects that are entirely unsafe to
    # access.
    if attr[:1] == "_":
        raise AttributeError(attr)
    return getattr(obj, attr)


def safe_format(_string: str, *args, **kwargs) -> str:
    formatter = SafeFormatter()
    kwargs = MagicFormatMapping(args, kwargs)
    return formatter.vformat(_string, args, kwargs)
