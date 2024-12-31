from pathlib import Path
from typing import Optional, Type

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

__all__ = ["RulesRetriever"]


class StrQuery(BaseModel):
    query: str = Field(description="Retrieval query")


class RulesRetriever(BaseTool):
    rules_path: str
    embeddings: OllamaEmbeddings
    vector_store: None | InMemoryVectorStore = None

    name: str = "RulesRetriever"
    description: str = "Provides relevant information for the latest Magic: The Gathering rules, as of November 2024."
    args_schema: Type[BaseModel] = StrQuery

    def __init__(self, rules_path: Path | str, embed_model: str = "snowflake-arctic-embed2"):
        super().__init__(rules_path=str(rules_path), embeddings=OllamaEmbeddings(model=embed_model))
        self.create_storage()

    def create_storage(self) -> None:
        self.vector_store = InMemoryVectorStore(self.embeddings)
        loader = PyPDFLoader(self.rules_path)
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
