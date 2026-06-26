# Contributing to python-project

Thanks for your interest in contributing. `python-project` is currently a solo personal project in early development, so the contribution bar is intentionally low — bug reports, suggestions, and fixes are all welcome.

---

## Reporting Bugs

Open a [GitHub issue](https://github.com/tclick/python-project/issues) and include:

- Your Python version (`python --version`)
- Your `python-project` version or commit hash
- The exact command you ran
- The full output or error traceback
- Your operating system

---

## Suggesting Features

Open an issue with the label `enhancement`. Describe:

- What you are trying to do
- Why the current behaviour does not cover it
- Any relevant MD software or workflow context (Amber version, force field, etc.)

---

## Making Changes

### Setup

```bash
git clone https://github.com/tclick/python-project.git
cd python-project
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Code Style

- Format with [Ruff](https://docs.astral.sh/ruff/): `ruff format .`
- Lint with Ruff: `ruff check .`
- Type-check with [mypy](https://mypy-lang.org/): `mypy python-project`

These are not yet enforced in CI (there is no CI yet), but keeping to them makes diffs easier to review.

### Adding a Subcommand

`python-project` uses an auto-discovery registry. To add a new subcommand module:

1. Create `python-project/commands/your_command.py`
2. Define a `typer.Typer()` app and register it:

```python
import typer
from python-project.registry import registry

app = typer.Typer()
registry.register(name="your-command", help="What it does.")(app)

@app.command()
def your_function(...):
    ...
```

3. The command is picked up automatically — no changes to `main.py` needed.

### Submitting a Pull Request

Since this is a solo project, pull requests are accepted on a best-effort basis. Please:

- Keep changes focused — one concern per PR
- Include a brief description of what changed and why
- Reference any related issues

There is no formal review SLA.

---

## License

By contributing you agree that your contributions will be released under the [GNU General Public License v3.0](LICENSE).
