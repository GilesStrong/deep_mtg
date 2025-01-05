from pathlib import Path
from typing import Optional, Type

from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

__all__ = ["RulesRetriever", "CardsRetriever"]


class StrQuery(BaseModel):
    query: str = Field(description="Retrieval query")


class ScalableQuery(BaseModel):
    query: str = Field(description="Retrieval query")
    k: int = Field(description="Number of results to return")
    # score_threshold: float = Field(default=0.0, description="Minimum similarity score to return. Float between zero and one.")


class RulesRetriever(BaseTool):
    rules_path: Path
    embeddings: OllamaEmbeddings
    vector_store: None | InMemoryVectorStore = None

    name: str = "RulesRetriever"
    description: str = "Provides relevant information for the latest Magic: The Gathering rules, as of November 2024."
    args_schema: Type[BaseModel] = StrQuery

    def __init__(self, rules_path: Path, embeddings: OllamaEmbeddings):
        super().__init__(rules_path=rules_path, embeddings=embeddings)
        self.create_storage()

    def create_storage(self) -> None:
        self.vector_store = InMemoryVectorStore(self.embeddings)
        loader = PyPDFLoader(str(self.rules_path))
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)
        print(f"Loaded {len(all_splits)} document splits.")
        self.vector_store.add_documents(documents=all_splits)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> tuple[str, list]:
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join((f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs)
        return serialized, retrieved_docs


class CardsRetriever(BaseTool):
    sets_path: Path
    embeddings: OllamaEmbeddings
    vector_store: None | InMemoryVectorStore = None
    extraction_schema: str = """
        .data.cards[] | {
            colors: .colors,
            convertedManaCost: .convertedManaCost,
            keywords: .keywords,
            manaCost: .manaCost,
            name: .name,
            power: .power,
            rarity: .rarity,
            subtypes: .subtypes,
            supertypes: .supertypes,
            text: .text,
            toughness: .toughness,
            types: .types
        }
    """

    name: str = "SetsRetriever"
    description: str = "Provides relevant information for cards in Magic."
    args_schema: Type[BaseModel] = ScalableQuery

    def __init__(self, sets_path: Path, embeddings: OllamaEmbeddings):
        super().__init__(sets_path=sets_path, embeddings=embeddings)
        self.create_storage()

    def create_storage(self) -> None:
        self.vector_store = InMemoryVectorStore(self.embeddings)
        for s in self.sets_path.glob("*.json"):
            print(f"Loading {s}...")
            loader = JSONLoader(s, self.extraction_schema, text_content=False)
            cards = loader.load()

            # Remove duplicates and basic lands
            filtered_cards = []
            filtered_hashes = []
            for card in cards:
                # card_dict = json.loads(card.page_content)
                # if card_dict["name"] in ["Plains", "Island", "Swamp", "Mountain", "Forest"]:
                #     continue
                if (h := hash(card.page_content)) not in filtered_hashes:
                    filtered_cards.append(card)
                    filtered_hashes.append(h)

            self.vector_store.add_documents(documents=filtered_cards)
            print(f"Loaded {len(filtered_cards)} cards from set {s}.")

    def _run(
        self, query: str, k: int, score_threshold: float = 0.0, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> list[str]:
        """Retrieve information related to a query."""
        retrieved_cards = self.vector_store.similarity_search_with_score(query, k=k)
        if k <= 2:
            score_threshold = 0.0
        filtered_cards = [card[0].page_content for card in retrieved_cards if card[1] > score_threshold]
        return filtered_cards
