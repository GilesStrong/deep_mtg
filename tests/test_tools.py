import os
from pathlib import Path

import pytest
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from deep_mtg.tools import CardsRetriever, RulesRetriever

PKG_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def llm() -> ChatOllama:
    return ChatOllama(model="llama3.1")


@pytest.fixture
def embeddings() -> ChatOllama:
    return OllamaEmbeddings(model="snowflake-arctic-embed2")


def test_rules_retriever(llm, embeddings) -> None:
    """
    GIVEN: The rules retriever tool
    WHEN: A simple agent is created to interact with the tool
    THEN: The agent should be able to retrieve the information related to the query
    """

    memory = MemorySaver()
    rules_retriever = RulesRetriever(rules_path=PKG_DIR / "../data/MagicCompRules 20241108.pdf", embeddings=embeddings)

    rules_retriever.invoke("In Magic The Gathering, how many cards must a standard constructed deck contain?")

    agent_executor = create_react_agent(llm, [rules_retriever], checkpointer=memory)
    config = {"configurable": {"thread_id": "test_rules_retriever"}}

    input_message = "In Magic The Gathering, how many cards must a standard constructed deck contain?"

    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print()


def test_cards_retriever(llm, embeddings) -> None:
    """
    GIVEN: The cards retriever tool
    WHEN: A simple agent is created to interact with the tool
    THEN: The agent should be able to retrieve the information related to the query
    """

    memory = MemorySaver()
    cards_retriever = CardsRetriever(sets_path=PKG_DIR / "../data/cards", embeddings=embeddings, llm=llm)

    cards_retriever.invoke({"query": "Sire of seven deaths", "k": 5, "score_threshold": 0.25})

    agent_executor = create_react_agent(llm, [cards_retriever], checkpointer=memory)
    config = {"configurable": {"thread_id": "test_cards_retriever"}}

    input_message = "What is the mana cost of Omniscience?"

    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print()
