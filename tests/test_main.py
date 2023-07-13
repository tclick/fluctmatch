"""Test cases for the __main__ module."""
import os

import pytest
from click.testing import CliRunner
from fluctmatch.cli import main


class TestMain:
    """Run test for main command."""

    @pytest.fixture()
    def cli_runner(self) -> CliRunner:
        """Fixture for testing `click` commands.

        Returns
        -------
        CliRunner
            CLI runner
        """
        return CliRunner()

    def test_help(self, cli_runner: CliRunner) -> None:
        """Test help output.

        GIVEN the main command
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        runner : CliRunner
            Command-line runner
        """
        result = cli_runner.invoke(main, ["-h"])

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK

    def test_main_succeeds(self, cli_runner: CliRunner) -> None:
        """Test main output.

        GIVEN the main command
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        runner : CliRunner
            Command-line runner
        """
        result = cli_runner.invoke(main)

        assert "Usage:" in result.output
        assert result.exit_code == os.EX_OK
