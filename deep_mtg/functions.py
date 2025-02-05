import json
from collections import defaultdict
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tqdm import tqdm

from .datamodels import DeckCard, DeckState, LandSelector, NLandsResponse, ZeroIndexCardSelection
from .tools import CardsRetriever

__all__ = [
    "build_deck",
    "append_card",
    "build_initial_manabase",
    "build_final_manabase",
    "get_deck_list",
    "add_card",
    "search_card",
    "name_deck",
]


def build_deck(deck_state: DeckState, llm: ChatOllama, cards_retriever: CardsRetriever) -> DeckState:
    print("\n\n Beginning deck building process \n\n")
    deck_state = build_initial_manabase(deck_state, n=18, llm=llm)
    for _ in tqdm(range(60 - 24)):
        deck_state = add_card(deck_state, llm, cards_retriever)
    deck_state = build_final_manabase(deck_state, llm, cards_retriever)
    for _ in tqdm(range(60 - deck_state["n_cards"])):
        deck_state = add_card(deck_state, llm, cards_retriever)
    deck_state = name_deck(deck_state, llm)
    return deck_state


def append_card(deck_state: DeckState, card: DeckCard, llm: Optional[ChatOllama]) -> DeckState:
    deck_state["cards"].append(card)
    deck_state["n_cards"] += 1
    if "land" in card["types"] or "Land" in card["types"]:
        deck_state["n_lands"] += 1
    elif "creature" in card["types"] or "Creature" in card["types"]:
        deck_state["n_creatures"] += 1
    elif "enchantment" in card["types"] or "Enchantment" in card["types"]:
        deck_state["n_enchantments"] += 1
    elif "instant" in card["types"] or "Instant" in card["types"]:
        deck_state["n_instants"] += 1
    elif "sorcery" in card["types"] or "Sorcery" in card["types"]:
        deck_state["n_sorceries"] += 1
    elif "artifact" in card["types"] or "Artifact" in card["types"]:
        deck_state["n_artifacts"] += 1
    else:
        deck_state["n_other"] += 1

    deck_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Magic: The Gathering deck builder. "
                "You are advising on building a high-competitative 60-card deck with a user-provided theme. "
                "Given the current state of the deck, provide an analysis of the current stregths and weaknesses of the deck. "
                "Along with possible strategies that could be built or reinforced. "
                "The deck may not be complete, but eventually it will contain 60 cards. "
                "Your expert analysis will be used to guide the next steps in building the deck. "
                "Do not output the names of cards to be added, your role is simple to provide an analysis of the current deck. ",
            ),
            (
                "user",
                "The deck theme is {prompt}. "
                "The current deck consists of {n_cards} cards, {n_lands} lands, {n_creatures} creatures, and {n_other} other cards. "
                "The cards present are:\n{deck}.\n "
                "Please give a think about what are the current strengths and weaknesses of the deck. ",
            ),
        ]
    )

    if llm is not None:
        deck_state["current_analysis"] = llm.invoke(  # type: ignore [typeddict-item]
            deck_analysis_prompt.invoke(
                {
                    "prompt": deck_state["prompt"],
                    "n_cards": deck_state["n_cards"],
                    "n_lands": deck_state["n_lands"],
                    "n_creatures": deck_state["n_creatures"],
                    "n_other": deck_state["n_other"],
                    "deck": get_deck_list(deck_state)[1],
                }
            )
        ).content
    else:
        deck_state["current_analysis"] = ""
    return deck_state


def name_deck(deck_state: DeckState, llm: ChatOllama) -> DeckState:
    deck_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Magic: The Gathering deck builder with a vivi imagination. "
                "You are advising on building a high-competitative 60-card deck with a user-provided theme. "
                "Given the current state of the deck, provide an appropriate, and inspiring name for the deck. "
                "The name should reflect the theme of the deck, and the strategy that the deck is built around. "
                "The name should be catchy and memorable, and should inspire confidence in the deck. "
                "The name should be unique and not already used by another deck. "
                "The name should be no more than 50 characters long. "
                "The name shoudl strike fear into the hearts of your opponents, and inspire your allies. "
                "You should only print the name of the deck and nothing else. ",
            ),
            (
                "user",
                "The deck theme is {prompt}. "
                "The current deck consists of {n_cards} cards, {n_lands} lands, {n_creatures} creatures, and {n_other} other cards. "
                "The cards present are:\n{deck}.\n "
                "Please bestow a grand name upon this deck!. ",
            ),
        ]
    )

    name = llm.invoke(  # type: ignore [typeddict-item]
        deck_analysis_prompt.invoke(
            {
                "prompt": deck_state["prompt"],
                "n_cards": deck_state["n_cards"],
                "n_lands": deck_state["n_lands"],
                "n_creatures": deck_state["n_creatures"],
                "n_other": deck_state["n_other"],
                "deck": get_deck_list(deck_state)[1],
            }
        )
    ).content
    # clean up name
    name = name.replace("\n", " ")
    name = name.replace(":", "")
    name = name.replace(";", "")
    name = name.replace("*", "")
    name = name.replace("/", "")
    name = name.replace("\\", "")
    name = name.replace("<think>", "")
    name = name.replace(".", "")
    name = name.replace("_", " ")
    name = name.strip()

    deck_state["name"] = name
    return deck_state


def build_initial_manabase(deck_state: DeckState, n: int, llm: ChatOllama) -> DeckState:
    land_selector_llm = llm.with_structured_output(LandSelector)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Magic: The Gathering deck builder. "
                "You are advising on building a high-competitative 60-card deck with a user-provided theme. "
                "You are working with a fellow expert deck builder to build a deck. "
                "Your role is to suggest the best combination of lands to form the initial manabase of the deck. "
                "You should add a total of {n_lands} lands to the deck. "
                "These can be basic lands or dual lands. "
                "Since you will be providing the initial manabase, you should consider the colour requirements of the deck given the theme. "
                "Your colleague will then add the remaining cards to the deck, after which you will be able to finalise the manabase of the deck. "
                "If no lands of a particular type are needed, you should output 0 for that type. "
                "Remeber that the more colors you add to the deck, the more dual lands you should include. "
                "When choosing which colors to include, consider the trade-off between ease of play of and card variety: more colors allow for more card variety, but make the deck harder to play. "
                "Think very carefully about the manabase, and do not add colors haphazardly. ",
            ),
            ("user", "The deck theme is {prompt}. " "Please output how lands of each type I should add. "),
        ]
    )

    backup_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Magic: The Gathering deck builder. "
                "You are advising on building a high-competitative 60-card deck with a user-provided theme. "
                "You are working with a fellow expert deck builder to build a deck. "
                "Your role is to suggest the best combination of mana colors to form the initial manabase of the deck given the theme. "
                "You should only output the mana colors, and not any other information. "
                "\n"
                "Examples:\n"
                "- life-gain vampires: black and white\n"
                "- goblin tribal: red\n"
                "- Azorius control: blue and white\n"
                "- mono-green elves: green\n"
                "- Rakdos sacrifice: black and red\n"
                "- Izzet spells: blue and red\n"
                "- Grixis control: blue, black, and red\n"
                "- Sultai midrange: black, green, and blue\n"
                "- Jeskai control: blue, white, and red\n",
            ),
            ("user", "The deck theme is {prompt}. " "Please output the mana colors I should use. "),
        ]
    )

    response = None
    fails = 0
    print("\n\n Building initial manabase \n\n")
    while response is None:
        if fails >= 5:
            print("Failed to correctly output the initial manabase after 5 attempts, relaxing constraints")
            n_colors = 0
            while n_colors == 0:
                backup_response = llm.invoke(
                    backup_prompt_template.invoke({"prompt": deck_state["prompt"], "n_lands": n})
                ).content.lower()
                colors = []
                for color in ["green", "red", "black", "blue", "white"]:
                    if color in backup_response:
                        colors.append(color)
                n_colors = len(colors)

            response = LandSelector(
                n_green=0,
                n_blue=0,
                n_red=0,
                n_white=0,
                n_black=0,
                n_green_white=0,
                n_blue_white=0,
                n_black_white=0,
                n_red_white=0,
                n_green_blue=0,
                n_black_blue=0,
                n_red_blue=0,
                n_green_black=0,
                n_red_black=0,
            )
            if n_colors == 1:
                n_each = n
                n_duals = 0
            else:
                n_duals = 3
                n_each = (n - n_duals) // n_colors

            for i in range(n_colors):
                response[f"n_{colors[i]}"] = n_each  # type: ignore [literal-required]
                for j in range(i + 1, n_colors):
                    response[f"n_{colors[i]}_{colors[j]}"] = n_duals  # type: ignore [literal-required]
            break

        print(f"Attempt {fails + 1} \n")
        response = land_selector_llm.invoke(prompt_template.invoke({"prompt": deck_state["prompt"], "n_lands": n}))  # type: ignore [assignment]
        fails += 1

    print(f"\n\nInitial manabase:, {response}\n")
    for count in response:
        land = count[2:]
        for _ in range(response[count]):  # type: ignore [literal-required]
            deck_state = append_card(
                deck_state,
                DeckCard(
                    name=f"{land} land",
                    types=["land"],
                    cost=None,
                    text=f"{land} land",
                    power=0,
                    toughness=0,
                    description=f"{land} land",
                ),
                llm=None,
            )

    return deck_state


def build_final_manabase(deck_state: DeckState, llm: ChatOllama, card_retriever: CardsRetriever) -> DeckState:
    n_lands_llm = llm.with_structured_output(NLandsResponse)

    n_lands_query_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Magic: The Gathering deck builder. "
                "You are advising on building a high-competitative 60-card deck with a user-provided theme. "
                "You are working with a fellow expert deck builder to build a deck. "
                "They have put together a deck with a total of {n_cards} cards, {n_lands} lands, {n_creatures} creatures, and {n_other} other cards. "
                "Your role is to finalize manabase of the deck by suggesting how many more lands should be added. "
                "To do this, you should consider the current state of the deck, the metagame, and the deck theme. "
                "Most 60-card decks run between 20 and 26 lands, with 24 being standard. "
                "However, the exact number of lands can vary depending on the deck's theme and strategy. "
                "Cheaper, faster decks may run fewer lands, while slower, more expensive decks may run more lands. "
                "Remaining slots after adding the number of lands you suggest will be filled with other cards. ",
            ),
            (
                "user",
                "The deck theme is {prompt}. "
                "The current deck consists of {n_cards} cards, {n_lands} lands, {n_creatures} creatures, and {n_other} other cards. "
                "The cards present are:\n{deck}.\n "
                "In total, the deck currently contains {n_cards} cards, leaving {n_left} free slots. "
                "Please suggest how many of these remaining slots should be lands. ",
            ),
        ]
    )

    n_remaining_lands = None
    fails = 0
    print("\n\n Building final manabase \n\n")
    while n_remaining_lands is None:
        if fails >= 5:
            print("Failed to correctly output the number of lands after 5 attempts, setting total lands to 24")
            n_remaining_lands = 24 - deck_state["n_lands"]
            break

        print(f"Attempt {fails + 1} \n")
        response = n_lands_llm.invoke(
            n_lands_query_prompt_template.invoke(
                {
                    "prompt": deck_state["prompt"],
                    "n_cards": deck_state["n_cards"],
                    "n_lands": deck_state["n_lands"],
                    "n_creatures": deck_state["n_creatures"],
                    "n_other": deck_state["n_other"],
                    "deck": get_deck_list(deck_state)[1],
                    "n_left": 60 - deck_state["n_cards"],
                }
            )
        )
        if response is not None:
            n_remaining_lands = response["n_lands"]  # type: ignore [index]
        if n_remaining_lands < 0 or n_remaining_lands > 60 - deck_state["n_cards"]:
            n_remaining_lands = None
        fails += 1

    land_query_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Magic: The Gathering deck builder. "
                "You are advising on building a high-competitative 60-card deck with a user-provided theme. "
                "You are working with a fellow expert deck builder to build a deck. "
                "They have put together a prototype deck, and your role is to finalize manabase of the deck by incrementally adding lands. "
                "You will do this by describing the next land to be added to the deck. "
                "The desired card doesn't need to actually exist in the game, I will find the closest card that matches your description. "
                "\n"
                "Example descriptions: "
                "\nA dual land that taps for green and white mana and can be used to fix your mana base"
                "\nA land that can produce creature tokens"
                "\nA land that can be sacrificed to search for another card"
                "\nA land can attack and block like a creature"
                "\n"
                "Only describe one card at a time. "
                "Do not return anything other than the description of the card. "
                "Do not mention the name of the card or the theme of the deck. "
                "Please also describe its role in the deck. "
                "Rather than quantifying attributes of the card, instead use qualitative terms like 'strong' or 'weak' to describe the card. "
                "\n"
                "When choosing the card, consider the following: "
                "\n1. The card should be a good fit for the deck theme. "
                "\n2. The card should be a good fit for the current state of the deck. "
                "\n3. The card should be a good fit for the current metagame. "
                "\n4. What is the card's role in the deck? "
                "\n5. What is the card's impact on the game? "
                "\n6. What is the card's synergy with other cards in the deck? "
                "\n7. What is the card's synergy with the deck's strategy? "
                "\n8. What current weaknesses in the deck does the card address? "
                "\n9. What current strengths in the deck does the card enhance? ",
            ),
            (
                "user",
                "The deck theme is {prompt}. "
                "The current deck consists of {n_cards} cards, {n_lands} lands, {n_creatures} creatures, and {n_other} other cards. "
                "The cards present are:\n{deck}.\n "
                "In total, you will be able to add in {n_remaining_lands} more lands to the deck, inclduing this one. "
                "Please give a description of what land card should be added to the deck. ",
            ),
        ]
    )

    print(f"\n\nAdding {n_remaining_lands} lands to the deck:\n")
    for _ in tqdm(range(n_remaining_lands)):
        desired_card: str = llm.invoke(  # type: ignore [assignment]
            land_query_prompt_template.invoke(
                {
                    "prompt": deck_state["prompt"],
                    "n_cards": deck_state["n_cards"],
                    "n_lands": deck_state["n_lands"],
                    "n_creatures": deck_state["n_creatures"],
                    "n_other": deck_state["n_other"],
                    "deck": get_deck_list(deck_state)[1],
                    "n_remaining_lands": n_remaining_lands,
                }
            )
        ).content
        card = search_card(deck_state, desired_card, llm, card_retriever)
        deck_state = append_card(deck_state, card, llm)
        n_remaining_lands -= 1

    return deck_state


def get_deck_list(deck_state: DeckState) -> tuple[dict[str, dict[str, int]], str]:
    deck_contents: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    type_counts: dict[str, int] = defaultdict(int)
    production: list[str] = []
    requirements: list[str] = []

    for card in deck_state["cards"]:
        entry = f"{card['name']}: {card['description']}"
        types = ",".join(card["types"]).lower()

        if card["cost"] is not None:
            if "{B}" in card["cost"]:
                requirements.append("black")
            if "{U}" in card["cost"]:
                requirements.append("blue")
            if "{R}" in card["cost"]:
                requirements.append("red")
            if "{W}" in card["cost"]:
                requirements.append("white")
            if "{G}" in card["cost"]:
                requirements.append("green")

        if "land" in types:
            if "{B}" in card["text"] or "black" in card["name"].lower():
                production.append("black")
            if "{U}" in card["text"] or "blue" in card["name"].lower():
                production.append("blue")
            if "{R}" in card["text"] or "red" in card["name"].lower():
                production.append("red")
            if "{W}" in card["text"] or "white" in card["name"].lower():
                production.append("white")
            if "{G}" in card["text"] or "green" in card["name"].lower():
                production.append("green")
            deck_contents["lands"][entry] += 1
            type_counts["lands"] += 1
        elif "creature" in types:
            deck_contents["creatures"][entry] += 1
            type_counts["creatures"] += 1

        elif "sorcery" in types:
            deck_contents["sorceries"][entry] += 1
            type_counts["sorceries"] += 1
        elif "instant" in types:
            deck_contents["instants"][entry] += 1
            type_counts["instants"] += 1
        elif "enchantment" in types:
            deck_contents["enchantments"][entry] += 1
            type_counts["enchantments"] += 1
        elif "artifact" in types:
            deck_contents["artifacts"][entry] += 1
            type_counts["artifacts"] += 1
        else:
            deck_contents["other"][entry] += 1
            type_counts["other"] += 1

    requirements = list(set(requirements))
    production = list(set(production))

    deck_str = ""
    for card_type, cards in deck_contents.items():
        deck_str += f"\n\n## {card_type}: {type_counts[card_type]}"
        for c, count in cards.items():
            deck_str += f"\n- {count} x {c}"
    deck_str += "\n"
    deck_str += f"\n## Mana production: {" ".join(production)}"
    deck_str += f"\n## Mana requirements: {" ".join(requirements)}\n"

    return deck_contents, deck_str


def add_card(deck_state: DeckState, llm: ChatOllama, cards_retriever: CardsRetriever) -> DeckState:
    prompt = deck_state["prompt"]
    if deck_state["n_cards"] == 0:
        prompt = prompt + " but it is currently empty"

    deck_contents, deck_str = get_deck_list(deck_state)
    print("Current deck:", deck_str)
    print("\nCurrent analysis:", deck_state["current_analysis"])

    card_advisor_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Magic: The Gathering deck builder. "
                "You are advising on building a high-competitative 60-card deck with a user-provided theme. "
                "Given the current state of the deck, provide a high-level description of what card should be added next. "
                "If the deck is empty, suggest a starting card that would be a good fit for the theme. "
                "The desired card doesn't need to actually exist in the game, I will find the closest card that matches your description. "
                "\n"
                "Example descriptions: "
                "\nA cheap, weak, red creature with flying and haste that can be played early in the game"
                "\nA 3-4 mana white sorcery that destroys all creatures which can be used to clear the board in the mid-game"
                "\nA costly blue sorcery that lets you draw multiple cards and can be used to refill your hand in the late game"
                "\nA land that taps for green or black mana and can be used to fix your mana base and to surveil when played"
                "\n"
                "Only describe one card at a time. "
                "Do not return anything other than the description of the card. "
                "Do not mention the name of the card or the theme of the deck. "
                "Please also describe its role in the deck. "
                "Rather than quantifying attributes of the card, instead use qualitative terms like 'strong' or 'weak' to describe the card. "
                "Remember, the deck should have 60 cards, including lands. "
                "\nA fellow expert has provided some analysis of the current deck, which you can use to guide your decision. "
                "You could try address any of the identified weaknesses or enhance any of the identified strengths or strategies. "
                "\n"
                "When choosing the card, consider the following: "
                "\n1. The card should be a good fit for the deck theme. "
                "\n2. The card should be a good fit for the current state of the deck. "
                "\n3. The card should be a good fit for the current metagame. "
                "\n4. What is the card's role in the deck? "
                "\n5. What is the card's impact on the game? "
                "\n6. What is the card's synergy with other cards in the deck? "
                "\n7. What is the card's synergy with the deck's strategy? "
                "\n8. What current weaknesses in the deck does the card address? "
                "\n9. What current strengths in the deck does the card enhance? "
                "\n10. What is the card's mana cost and colour? "
                "\n11. Can the card be cast using the current manabase, or via other effects (e.g. reanimation)?"
                "\n",
                # "You can use the CardsRetriever tool to find cards that match the description. "
            ),
            (
                "user",
                "The deck theme is {prompt}. "
                "The current deck consists of {n_cards} cards, {n_lands} lands, {n_creatures} creatures, and {n_other} other cards. "
                "The cards present are:\n{deck}.\n "
                "\nAnalysis of the current deck: {current_analysis}. "
                "Please give a description of what card should be added to the deck. ",
            ),
        ]
    )
    desired_card: str = llm.invoke(  # type: ignore [assignment]
        card_advisor_prompt.invoke(
            {
                "prompt": deck_state["prompt"],
                "n_cards": deck_state["n_cards"],
                "n_lands": deck_state["n_lands"],
                "n_creatures": deck_state["n_creatures"],
                "n_other": deck_state["n_other"],
                "deck": deck_str,
                "current_analysis": deck_state["current_analysis"],
            }
        )
    ).content

    print("Recommended card:", desired_card)
    card = search_card(deck_state, desired_card, llm, cards_retriever)
    deck_state = append_card(deck_state, card, llm)
    return deck_state


def search_card(deck_state: DeckState, desired_card: str, llm: ChatOllama, cards_retriever: CardsRetriever) -> DeckCard:
    """
    TODO: Add metadata filtering
    """
    card_selector_llm = llm.with_structured_output(ZeroIndexCardSelection)

    deck_contents, deck_str = get_deck_list(deck_state)
    k = 5
    matching_cards: list[str] = []
    while len(matching_cards) < 5:
        matching_cards = cards_retriever.invoke({"query": desired_card, "k": k})
        matching_cards_dicts = [json.loads(c) for c in matching_cards]
        # filter out cards that are already in the deck 4 times
        matching_cards = [
            c for i, c in enumerate(matching_cards) if f'4 x {matching_cards_dicts[i]["name"]}' not in deck_str
        ]
        k += max(1, 5 - len(matching_cards))

    matching_cards_dicts = [json.loads(c) for c in matching_cards]  # recreate in case of filtering
    matching_cards_str = "".join([f'\n\nindex {i}: {c["summary"]}' for i, c in enumerate(matching_cards_dicts)])

    print(f"Searching for card: {desired_card}")
    print(f"Found these possible matches: {matching_cards_str}")

    card_selector_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Magic: The Gathering deck builder. "
                "You are advising on building a high-competitative 60-card deck with a user-provided theme. "
                "You are working with a fellow expert deck builder to build a deck. "
                "Your colleague has provided a high-level description of a card that should be added to the deck. "
                "You will be provided with a list of cards that match the description. "
                "You need to select the best card to add to the deck. "
                "The card should be a good fit for the deck theme, the current state of the deck, and the current metagame. "
                "Remeber that a deck cannot have more than 4 copies of a card, unless it is a basic land. "
                "Also consider the manabase of the deck when choosing the card. "
                "However, also consider that the deck may be able to cards not in the current manabase via reanimation effects, etc. "
                "\nA fellow expert has provided some analysis of the current deck, which you can use to guide your decision. "
                "You could try address any of the identified weaknesses or enhance any of the identified strengths or strategies. "
                "\n"
                "When choosing the card, consider the following: "
                "\n1. The card should be a good fit for the deck theme. "
                "\n2. The card should be a good fit for the current state of the deck. "
                "\n3. The card should be a good fit for the current metagame. "
                "\n4. What is the card's role in the deck? "
                "\n5. What is the card's impact on the game? "
                "\n6. What is the card's synergy with other cards in the deck? "
                "\n7. What is the card's synergy with the deck's strategy? "
                "\n8. What current weaknesses in the deck does the card address? "
                "\n9. What current strengths in the deck does the card enhance? "
                "\n10. What is the card's mana cost and colour? "
                "\n11. Can the card be cast using the current colors in the manabase?"
                "\n",
            ),
            (
                "user",
                "The deck theme is {prompt}. "
                "The current deck consists of {n_cards} cards, {n_lands} lands, {n_creatures} creatures, and {n_other} other cards. "
                "The cards present are:\n{deck}.\n"
                "The description of the card that should be added to the deck is: {description}. "
                "The list of cards that match the description is:\n{cards}.\n"
                "The current analysis of the deck is: {current_analysis}. "
                "Please output the index of the most suitable card to add, using zero-indexing. ",
            ),
        ]
    )
    selector_response: dict | None = None
    card_index = None
    fails = 0
    print(f"\nSelecting card from {len(matching_cards)} possible matches")
    while selector_response is None:
        if fails >= 5:
            print("Failed to correctly select a card after 5 attempts, selecting the zeroth card")
            card_index = 0
            break

        print(f"Attempt {fails + 1}")
        selector_response = card_selector_llm.invoke(  # type: ignore [assignment]
            card_selector_prompt.invoke(
                {
                    "prompt": deck_state["prompt"],
                    "n_cards": deck_state["n_cards"],
                    "n_lands": deck_state["n_lands"],
                    "n_creatures": deck_state["n_creatures"],
                    "n_other": deck_state["n_other"],
                    "cards": matching_cards_str,
                    "deck": deck_str,
                    "description": desired_card,
                    "current_analysis": deck_state["current_analysis"],
                }
            )
        )
        if selector_response is not None and selector_response["index"] > len(matching_cards):
            selector_response = None
        fails += 1
    if card_index is None:
        card_index = selector_response["index"]
        print(f'Selected card index: {card_index}, reason: {selector_response["reason"]}')

    card = matching_cards[card_index]
    print("Matching card:", card)

    card_dict = json.loads(card)
    return DeckCard(
        name=card_dict["name"],
        types=card_dict["types"],
        cost=card_dict["manaCost"],
        text=card_dict["text"],
        power=card_dict["power"],
        toughness=card_dict["toughness"],
        description=card_dict["summary"],
    )
