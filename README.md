# my_repo

## Installation

For development usage, we use [`poetry`](https://python-poetry.org/docs/#installing-with-the-official-installer) to handle dependency installation.
Poetry can be installed via, e.g.

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry self update
```

and ensuring that `poetry` is available in your `$PATH`

Install the dependencies:

```bash
poetry install
poetry self add poetry-plugin-export
poetry config warnings.export false
poetry run pre-commit install
```
