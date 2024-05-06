# ------------------------------------------------------------------------------
#  fluctmatch
#  Copyright (c) 2013-2024 Timothy H. Click, Ph.D.
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
"""Nox sessions."""

import os
import shlex
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox

try:
    from nox_poetry import Session, session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None


package = "fluctmatch"
python_versions = ["3.12", "3.11"]
nox.needs_version = ">= 2024.4.15"
nox.options.sessions = (
    "pre-commit",
    "safety",
    "pyright",
    "tests",
    "typeguard",
    "xdoctest",
    "docs-build",
)


def activate_virtualenv_in_precommit_hooks(sessions: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        sessions: The Session object.
    """
    assert sessions.bin is not None  # nosec

    # Only patch hooks containing a reference to this session's bindir. Support
    # quoting rules for Python and bash, but strip the outermost quotes so we
    # can detect paths within the bindir, like <bindir>/python.
    bindirs = [
        bindir[1:-1] if bindir[0] in "'\"" else bindir for bindir in (repr(sessions.bin), shlex.quote(sessions.bin))
    ]

    virtualenv = sessions.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    headers = {
        # pre-commit < 2.16.0
        "python": f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {sessions.bin!r},
                os.environ.get("PATH", ""),
            ))
            """,
        # pre-commit >= 2.16.0
        "bash": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(sessions.bin)}"{os.pathsep}$PATH"
            """,
        # pre-commit >= 2.17.0 on Windows forces sh shebang
        "/bin/sh": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(sessions.bin)}"{os.pathsep}$PATH"
            """,
    }

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        if not hook.read_bytes().startswith(b"#!"):
            continue

        text = hook.read_text()

        if not any(Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text for bindir in bindirs):
            continue

        lines = text.splitlines()

        for executable, header in headers.items():
            if executable in lines[0].lower():
                lines.insert(1, dedent(header))
                hook.write_text("\n".join(lines))
                break


@session(name="pre-commit", python=python_versions[0])
def precommit(sessions: Session) -> None:
    """Lint using pre-commit."""
    args = sessions.posargs or "run --all-files --hook-stage=manual --show-diff-on-failure".split()
    sessions.install(".")
    sessions.run("pre-commit", *args, external=True)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(python=python_versions[0])
def safety(sessions: Session) -> None:
    """Scan dependencies for insecure packages."""
    requirements = sessions.poetry.export_requirements()
    sessions.install(".")
    sessions.run(*f"safety check --full-report --file={requirements}".split())


@session(python=python_versions)
def pyright(sessions: Session) -> None:
    """Type-check using pyright.

    Parameters
    ----------
    sessions: Session
        The Session object.
    """
    args = (
        sessions.posargs
        or f"-p pyproject.toml --pythonversion {sessions.python} --pythonpath={sys.executable} src".split()
    )
    sessions.install(".")
    sessions.run("basedpyright", *args)


@session(python=python_versions)
def tests(sessions: Session) -> None:
    """Run the test suite."""
    args = "--cov src --cov-report=xml --cov-config=pyproject.toml --random-order --disable-pytest-warnings".split()
    sessions.install(".")

    try:
        sessions.run("pytest", *args, *sessions.posargs)
    finally:
        if sessions.interactive:
            sessions.notify("coverage", posargs=[])


@session(python=python_versions[0])
def coverage(sessions: Session) -> None:
    """Produce the coverage report."""
    args = sessions.posargs or ["report"]

    sessions.install("coverage[toml]")

    if not sessions.posargs and any(Path().glob(".coverage.*")):
        sessions.run("coverage", "combine")

    sessions.run("coverage", *args)


@session(python=python_versions[0])
def typeguard(sessions: Session) -> None:
    """Runtime type checking using Typeguard."""
    args = "--typeguard-packages=all --random-order --disable-pytest-warnings".split()
    sessions.install(".")
    sessions.run("pytest", *args, *sessions.posargs)


@session(python=python_versions)
def xdoctest(sessions: Session) -> None:
    """Run examples with xdoctest."""
    if sessions.posargs:
        args = [package, *sessions.posargs]
    else:
        args = [f"--modname={package}", "--command=all"]
        if "FORCE_COLOR" in os.environ:
            args.append("--colored=1")

    sessions.install(".")
    sessions.install("xdoctest[colors]")
    sessions.run("python", "-m", "xdoctest", *args)


@session(name="docs-build", python=python_versions[0])
def docs_build(sessions: Session) -> None:
    """Build the documentation."""
    args = sessions.posargs or ["docs", "docs/_build"]
    if not sessions.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    sessions.install(".")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    sessions.run("sphinx-build", *args)


@session(python=python_versions[0])
def docs(sessions: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = sessions.posargs or ["--open-browser", "docs", "docs/_build"]
    sessions.install(".")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    sessions.run("sphinx-autobuild", *args)
