from typing import Annotated, Optional, TypedDict

__all__ = ["DeckCard", "DeckState", "LandSelector", "NLandsResponse", "ZeroIndexCardSelection"]


class DeckCard(TypedDict):
    """Description of a card in a deck"""

    name: Annotated[str, ..., "Card name"]
    types: Annotated[list[str], ..., "Card type"]
    cost: Annotated[Optional[str], ..., "Card cost, including mana colors, if applicable"]
    text: Annotated[Optional[str], ..., "Card text, if applicable"]
    power: Annotated[Optional[int], ..., "Card power, if applicable"]
    toughness: Annotated[Optional[int], ..., "Card toughness, if applicable"]
    description: Annotated[str, ..., "High-level card description"]


class DeckState(TypedDict):
    """Description of a deck"""

    prompt: Annotated[str, ..., "Deck theme prompt"]
    cards: Annotated[list[DeckCard], [], "List of cards in the deck"]
    n_cards: Annotated[int, 0, "Number of cards in the deck"]
    n_lands: Annotated[int, 0, "Number of lands in the deck"]
    n_creatures: Annotated[int, 0, "Number of creatures in the deck"]
    n_enchantments: Annotated[int, 0, "Number of enchantments in the deck"]
    n_instants: Annotated[int, 0, "Number of instants in the deck"]
    n_sorceries: Annotated[int, 0, "Number of sorceries in the deck"]
    n_artifacts: Annotated[int, 0, "Number of artifacts in the deck"]
    n_other: Annotated[int, 0, "Number of other cards in the deck"]
    current_analysis: Annotated[str, "", "Expert analysis of the current strengths and weaknesses of the deck"]


class LandSelector(TypedDict):
    """Description of the number of lands of particular colors to add to a deck"""

    n_green: Annotated[int, ..., "Number of basic green lands to add"]
    n_blue: Annotated[int, ..., "Number of basic blue lands to add"]
    n_red: Annotated[int, ..., "Number of basic red lands to add"]
    n_white: Annotated[int, ..., "Number of basic white lands to add"]
    n_black: Annotated[int, ..., "Number of basic black lands to add"]
    n_green_white: Annotated[int, ..., "Number of green-white dual lands to add"]
    n_blue_white: Annotated[int, ..., "Number of blue-white dual lands to add"]
    n_black_white: Annotated[int, ..., "Number of black-white dual lands to add"]
    n_red_white: Annotated[int, ..., "Number of red-white dual lands to add"]
    n_green_blue: Annotated[int, ..., "Number of green-blue dual lands to add"]
    n_black_blue: Annotated[int, ..., "Number of black-blue dual lands to add"]
    n_red_blue: Annotated[int, ..., "Number of red-blue dual lands to add"]
    n_green_black: Annotated[int, ..., "Number of green-black dual lands to add"]
    n_red_black: Annotated[int, ..., "Number of red-black dual lands to add"]


class NLandsResponse(TypedDict):
    """Description of the number of lands to add to a deck"""

    n_lands: Annotated[int, ..., "Number of lands to add"]


class ZeroIndexCardSelection(TypedDict):
    """Index of a card to add to a deck from a selection of possible cards"""

    index: Annotated[int, ..., "Zero-indexed index of the card to be added"]
    reason: Annotated[Optional[str], ..., "Reason for selecting the card"]
