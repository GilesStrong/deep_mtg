import json
import os
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from typer import Argument, Option, Typer

from deep_mtg.datamodels import DeckState
from deep_mtg.functions import build_deck, get_deck_list
from deep_mtg.tools import CardsRetriever

app = Typer()

PKG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


@app.command()
def build(
    deck_prompt: str = Argument(..., help="Deck prompt"),
    deck_llm_name: str = Option(
        "MFDoom/deepseek-r1-tool-calling:8b", help="Name of the LLM model to use for deck building"
    ),
    summary_llm_name: str = Option(
        "phi4:latest", help="Name of the LLM model to use for card & deck summary generation"
    ),
    embedding_name: str = Option("snowflake-arctic-embed2", help="Name of the embeddings model to use for RAG search"),
    sets_path: Path = Option(None, help="Path to the sets file"),
) -> None:
    print(f"Building deck for {deck_prompt}")
    print(f"Using {deck_llm_name} for deck building")
    print(f"Using {summary_llm_name} for deck summary generation")
    print(f"Using {embedding_name} for embeddings")
    if sets_path is None:
        sets_path = PKG_DIR / "../../data/cards"
    print(f"Using {sets_path} for sets")

    deck_llm = ChatOllama(model=deck_llm_name, num_ctx=8192)
    summary_llm = ChatOllama(model=summary_llm_name)
    embeddings = OllamaEmbeddings(model=embedding_name)

    deck_state = DeckState(
        prompt=deck_prompt,
        cards=[],
        n_cards=0,
        n_lands=0,
        n_creatures=0,
        n_other=0,
        n_enchantments=0,
        n_instants=0,
        n_sorceries=0,
        n_artifacts=0,
        current_analysis="",
        name="",
    )

    card_retriever = CardsRetriever(sets_path=sets_path, llm=summary_llm, embeddings=embeddings)
    deck_state = build_deck(deck_state, llm=deck_llm, cards_retriever=card_retriever)

    print("Deck built!")
    print("Deck summary:")
    print(f"Deck name: {deck_state['name']}")
    deck_contents, deck_str = get_deck_list(deck_state)
    print(deck_str)
    print("\n", deck_state["current_analysis"])

    savename = deck_state["name"].replace(" ", "_")
    print(f"Saving deck to {savename}.json")

    with open(f"{savename}.json", "w") as f:
        json.dump({"deck_state": deck_state, "deck_contents": deck_contents, "deck_str": deck_str}, f)


if __name__ == "__main__":
    app()
