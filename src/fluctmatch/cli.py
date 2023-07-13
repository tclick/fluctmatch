"""Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

You might be tempted to import things from __main__ later, but that will cause
problems: the code will get executed twice:

    - When you run `python -mmdtab` python will execute
      ``__main__.py`` as a script. That means there won't be any
      ``mdta.__main__`` in ``sys.modules``.
    - When you import __main__ it will get executed again (as a module) because
      there's no ``mdta.__main__`` in ``sys.modules``.

Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import click
from click_extra import help_option, version_option

from . import __copyright__, __version__

CONTEXT_SETTINGS = {
    "auto_envvar_prefix": "COMPLEX",
    "show_default": True,
}

TComplexCLI = TypeVar("TComplexCLI", bound="ComplexCLI")


class ComplexCLI(click.Group):
    """Complex command-line options with subcommands for fluctmatch."""

    def list_commands(self: TComplexCLI, ctx: click.Context) -> list[str] | None:
        """List available commands.

        Parameters
        ----------
        ctx : `Context`
            click context

        Returns
        -------
            List of available commands
        """
        rv = []
        cmd_folder = Path(__file__).parent.joinpath("commands").resolve()

        for filename in Path(cmd_folder).iterdir():
            if filename.name.endswith(".py") and filename.name.startswith("cmd_"):
                rv.append(filename.name[4:-3])
        rv.sort()
        return rv

    def get_command(self: TComplexCLI, ctx: click.Context, name: str) -> click.Command | None:
        """Run the selected command.

        Parameters
        ----------
        ctx : `Context`
            click context
        name : str
            command name

        Returns
        -------
            The chosen command if present
        """
        try:
            mod = __import__(f"mdsetup.commands.cmd_{name}", None, None, ["cli"])
        except ImportError:
            return None
        return mod.cli


@click.command(name="mdsetup", cls=ComplexCLI, context_settings=CONTEXT_SETTINGS, help=__copyright__)
@version_option(version=__version__)
@help_option()
@click.pass_context
def main(ctx: click.Context) -> None:
    """Molecular dynamics setup main command.

    Parameters
    ----------
    ctx : `Context`
        click context
    """
