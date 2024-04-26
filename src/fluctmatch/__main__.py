"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Fluctuation Matching."""


if __name__ == "__main__":
    main(prog_name="fluctmatch")  # pragma: no cover
