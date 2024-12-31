from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

__all__ = ["RulesRetriever"]


class RulesRetriever:
    def __init__(self, rules_path: Path | str, embed_model: str = "snowflake-arctic-embed2"):
        self.rules_path = str(rules_path)
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def create_storage(self) -> None:
        loader = PyPDFLoader(self.rules_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)
        self.vector_store.add_documents(documents=all_splits)

    @tool(response_format="content_and_artifact")
    def retrieve(self, query: str) -> tuple[str, list]:
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join((f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs)
        return serialized, retrieved_docs
