# DeepMTG: Building 'Magic: The Gathering' decks using LLM chains.

## Installation

For development usage, we use [`poetry`](https://python-poetry.org/docs/#installing-with-the-official-installer) to handle dependency installation.
Poetry can be installed via, e.g.

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry self update
```

and ensuring that `poetry` is available in your `$PATH`

Install [Ollama](https://ollama.com/), e.g. `curl -fsSL https://ollama.com/install.sh | sh`

Pull Ollama models, e.g.:

```bash
ollama pull MFDoom/deepseek-r1-tool-calling:8b
ollama pull phi4:latest
```

Install the dependencies:

```bash
poetry install
poetry self add poetry-plugin-export
poetry config warnings.export false
poetry run pre-commit install
```


## Data

Download these and place them in the `data` directory:

- rules [https://media.wizards.com/2024/downloads/MagicCompRules%2020241108.pdf]

Download the desrired `.json` files and place them in the `data/cards` directory:

- cards [https://mtgjson.com/downloads/all-sets/]

## Usage

To run the program, use the following command:

```bash
poetry run python scripts/build_deck.py "prompt for you deck"
```

The deck will be saved to the current directory, with a generated name.

Extra options include:

- `--deck-llm-name` to specify the LLM used for deck generation (default: `MFDoom/deepseek-r1-tool-calling:8b`)
- `--summary-llm-name` to specify the LLM used for card-summary generation (default: `phi4:latest`)

**N.B.** The first run will take a while as the cards will be summarised and embedded. After the first run, the embeddings will be cached and the process will be much faster.

**N.B.** Deck generation may be slow, depending the LLM used.

## TO DO:

- Improve initial manabase generation
- Add card-type filtering to searches
- Add post-constuction deck refinement
- Add deck-performance prediction
