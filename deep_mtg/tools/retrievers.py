import json
from pathlib import Path
from typing import Optional, Type

from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tqdm import tqdm

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
    recreate_storage: bool = False

    name: str = "RulesRetriever"
    description: str = "Provides relevant information for the latest Magic: The Gathering rules, as of November 2024."
    args_schema: Type[BaseModel] = StrQuery

    def __init__(self, rules_path: Path, embeddings: OllamaEmbeddings, recreate_storage: bool = False):
        super().__init__(rules_path=rules_path, embeddings=embeddings, recreate_storage=recreate_storage)
        self.create_storage()

    def create_storage(self) -> None:
        if self.recreate_storage or not (self.rules_path.parent / "rules.vec").exists():
            self.vector_store = InMemoryVectorStore(self.embeddings)
            loader = PyPDFLoader(str(self.rules_path))
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
            all_splits = text_splitter.split_documents(docs)
            print(f"Loaded {len(all_splits)} document splits.")
            self.vector_store.add_documents(documents=all_splits)
            print(f"Dumping vectors to disk {self.rules_path.parent / 'rules.vec'}...")
            self.vector_store.dump(str(self.rules_path.parent / "rules.vec"))

        else:
            print(f"Loading vectors from disk {self.rules_path.parent / 'rules.vec'}...")
            self.vector_store = InMemoryVectorStore(self.embeddings).load(
                str(self.rules_path.parent / "rules.vec"), self.embeddings
            )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> tuple[str, list]:
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join((f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs)
        return serialized, retrieved_docs


class CardsRetriever(BaseTool):
    sets_path: Path
    llm: ChatOllama
    embeddings: OllamaEmbeddings
    summary_prompt: ChatPromptTemplate
    card_vector_store: None | InMemoryVectorStore = None
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
    recreate_storage: bool = False

    name: str = "SetsRetriever"
    description: str = "Provides relevant information for cards in Magic."
    args_schema: Type[BaseModel] = ScalableQuery

    def __init__(self, sets_path: Path, llm: ChatOllama, embeddings: OllamaEmbeddings, recreate_storage: bool = False):
        summary_prompt = ChatPromptTemplate.from_messages(
            (
                [
                    (
                        "system",
                        "You are an expert Magic: The Gathering player."
                        "You are helping a new player to understand what various cards do from a high-level perspective."
                        "When provided with a card, you should write a concise summary of the card."
                        "You can assume that the player understands the basic rules of the game"
                        "Include basic keyword terms like 'flying' or 'trample', but do not explain what they mean."
                        "Do not include details of rarity or set information."
                        "Rather than quantifying attributes of the card, instead use qualitative terms like 'strong' or 'weak' to describe the card."
                        "Include the name of the card in the summary."
                        "Include details of the card's role in that, strengths, and weaknesses."
                        "Include the mana colors of the card, along with the a qualitative description of the mana cost."
                        "Do not return anything other than the summary of the card."
                        "Make sure to check that your summary accurately reflects the card.",
                    ),
                    ("user", "{card}"),
                ]
            )
        )
        super().__init__(
            sets_path=sets_path,
            llm=llm,
            embeddings=embeddings,
            summary_prompt=summary_prompt,
            recreate_storage=recreate_storage,
        )
        self.create_storage()

    def create_embeddings(self) -> None:
        print(f"Creating embeddings for all cards in {self.sets_path}...")
        for json_path in tqdm(list(self.sets_path.glob("*.json"))):
            vec_path = json_path.with_suffix(".vec")
            if self.recreate_storage or not vec_path.exists():
                print(f"Embedding {json_path}...")
                loader = JSONLoader(json_path, self.extraction_schema, text_content=False)
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

                # Create card summaries
                print(f"Creating summaries for {len(filtered_cards)} cards...")
                for card in tqdm(filtered_cards):
                    card_dict = json.loads(card.page_content)
                    if "land" in card_dict["types"]:
                        summary = card_dict["text"]
                    else:
                        summary = self.llm.invoke(self.summary_prompt.invoke({"card": card.page_content})).content
                    # clean up the summary
                    if "</think>" in summary:
                        summary = summary[summary.rfind("</think>") + 8 :]
                    summary = summary.replace("\n", " ")
                    summary = summary.replace('"', "")
                    summary = summary.replace("\\", " ")
                    summary = summary.replace("\\\\", "")
                    summary = summary.replace("\\n", " ")
                    summary = summary.strip()
                    card.page_content = '{"summary": "' + summary + '", ' + card.page_content[1:]

                tmp_vector_store = InMemoryVectorStore(self.embeddings)
                tmp_vector_store.add_documents(documents=filtered_cards)
                print(f"Embedded {len(filtered_cards)} cards from set {json_path}.")

                print(f"Dumping vectors to disk {vec_path}...")
                tmp_vector_store.dump(str(vec_path))

        vec_data = {}
        for vec_path in self.sets_path.glob("*.vec"):
            if vec_path.name == "all_cards.vec":
                continue
            print(f"Loading {vec_path}...")
            with open(vec_path, "r") as f:
                tmp_data = json.load(f)
            vec_data.update(tmp_data)
        with open(self.sets_path / "all_cards.vec", "w") as f:
            json.dump(vec_data, f)

    def create_storage(self) -> None:
        self.create_embeddings()
        self.card_vector_store = InMemoryVectorStore(self.embeddings).load(
            str(self.sets_path / "all_cards.vec"), self.embeddings
        )
        print(f"Loaded {len(self.card_vector_store.store)} cards from all sets.")

    def _run(
        self, query: str, k: int, score_threshold: float = 0.0, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> list[str]:
        """Retrieve information related to a query."""
        retrieved_cards = self.card_vector_store.similarity_search_with_score(query, k=k)
        if k <= 2:
            score_threshold = 0.0

        filtered_cards = []
        for card in retrieved_cards:
            if card[1] < score_threshold:
                continue
            c = card[0].page_content
            c = c.replace("\\\\", "")
            c = c.replace("\\n", " ")
            c.replace("\n", " ")
            filtered_cards.append(c)

        return filtered_cards
