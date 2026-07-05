"""Book recommendation agent.

LangGraph StateGraph pipeline over the Pinecone `books` index:

    extract -> search -> (relax -> search)* -> rerank -> respond
                \\-> smalltalk                    \\-> no_results

Recommendations are grounded strictly in retrieved index results.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from backend.app.schemas import ChatMessage
from backend.app.services.pinecone_utils import search_books

EXTRACT_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o"

TOP_K = 15
MAX_RECOMMENDATIONS = 6
MAX_RELAX_ATTEMPTS = 3  # attempt 0 = all filters ... attempt 3 = unfiltered

# Only tokens produced inside these nodes are streamed to the user.
STREAMED_NODES = {"respond", "smalltalk"}


class SearchSpec(BaseModel):
    """Structured search parameters extracted from the conversation."""

    semantic_query: str = Field(
        description=(
            "Rich retrieval phrase capturing topic, genre, mood and themes, e.g. "
            "'gothic obsession revenge dark classic'. Genre and topic belong here, "
            "never in filters. Never empty."
        )
    )
    authors: list[str] = Field(
        default_factory=list,
        description=(
            "Only when the user explicitly names an author. Exact catalog format "
            "'Last, First', e.g. 'Twain, Mark' or 'Shakespeare, William'."
        ),
    )
    languages: list[str] = Field(
        default_factory=list,
        description="ISO 639-1 codes, only when the user constrains language, e.g. 'in French' -> ['fr'].",
    )
    year_from: int | None = Field(
        default=None, description="Earliest first-publication year, only if explicitly constrained."
    )
    year_to: int | None = Field(
        default=None, description="Latest first-publication year, e.g. 'before 1900' -> 1900."
    )
    is_book_request: bool = Field(
        description="False when the last user message is small talk or not a book request."
    )


class AgentState(TypedDict):
    history: list[BaseMessage]
    spec: SearchSpec | None
    attempt: int
    dropped_filters: list[str]
    results: list[dict[str, Any]]


EXTRACT_PROMPT = """\
You extract structured book-search parameters for a catalog of ~1000 classic
Project Gutenberg books. Analyze the WHOLE conversation (follow-ups like
'something shorter by the same author' inherit constraints from earlier turns)
and fill the SearchSpec.

Rules:
- semantic_query: always filled; describe topic, genre, mood, themes. If the user
  references a book ('like Frankenstein'), describe its themes (gothic, science,
  moral dilemmas) rather than relying on the title.
- authors / languages / year_from / year_to: only for EXPLICIT constraints.
- is_book_request: false for greetings, small talk, or non-book questions.
"""

ANSWER_PROMPT = """\
You are a concise, warm book-recommendation assistant for a classic-literature
catalog (Project Gutenberg).

The numbered list below contains the ONLY books you may mention or recommend.
They were retrieved from the catalog for this specific request:

{books}

Rules:
- Recommend the 3-5 best matches for the user's request. Format each as
  **Title** — Author (year if known), followed by 1-2 sentences on why it fits
  THIS request, grounded in the summary and subjects above.
- Never mention, invent, or allude to any book outside the list.
- If nothing above truly fits, say so honestly and present the closest options as such.
{caveat}- Answer in the user's language. Use markdown. Stay under ~250 words.
"""

SMALLTALK_PROMPT = """\
You are a book-recommendation assistant for a catalog of classic Project
Gutenberg books. The user is not asking for a book recommendation right now.
Reply briefly and warmly, and steer the conversation toward what kind of book
they might enjoy. Do not name or recommend any specific book.
"""

NO_RESULTS_MESSAGE = (
    "I couldn't find any books in the catalog matching that request — even after "
    "relaxing the filters. The catalog holds ~1000 classic Project Gutenberg "
    "titles, so try different topic words, or drop the author/language/year "
    "constraint."
)


def _to_lc_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
    mapping = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
    return [mapping[m.role](content=m.content) for m in messages]


def _build_filters(spec: SearchSpec, attempt: int) -> tuple[dict[str, Any] | None, list[str]]:
    """Progressively relax filters: year drops first (field is missing on some
    docs), then language, then author (the user's most intentional constraint)."""
    filters: dict[str, Any] = {}
    dropped: list[str] = []

    if spec.year_from is not None or spec.year_to is not None:
        if attempt < 1:
            year: dict[str, float] = {}
            if spec.year_from is not None:
                year["$gte"] = spec.year_from
            if spec.year_to is not None:
                year["$lte"] = spec.year_to
            filters["first_publish_year"] = year
        else:
            dropped.append("publication year")
    if spec.languages:
        if attempt < 2:
            filters["languages"] = {"$in": spec.languages}
        else:
            dropped.append("language")
    if spec.authors:
        if attempt < 3:
            filters["authors"] = {"$in": spec.authors}
        else:
            dropped.append("author")

    return filters or None, dropped


def _format_books(matches: list[dict[str, Any]]) -> str:
    lines = []
    for i, match in enumerate(matches, 1):
        md = match["metadata"]
        authors = ", ".join(md.get("authors") or []) or "Unknown author"
        year = md.get("first_publish_year")
        year_note = f", first published ~{int(year)}" if year else ""
        subjects = "; ".join((md.get("subjects") or [])[:5]) or "n/a"
        downloads = int(md.get("download_count") or 0)
        summary = (md.get("chunk_text") or "").replace("\n", " ").strip()[:500]
        lines.append(
            f'{i}. "{md.get("title", "Untitled")}" — {authors} '
            f"(language: {', '.join(md.get('languages') or ['?'])}{year_note}, "
            f"{downloads} downloads, relevance {match['score']:.3f})\n"
            f"   Subjects: {subjects}\n"
            f"   Summary: {summary}"
        )
    return "\n".join(lines)


# --- graph nodes -----------------------------------------------------------


async def extract(state: AgentState) -> dict[str, Any]:
    llm = ChatOpenAI(model=EXTRACT_MODEL, temperature=0).with_structured_output(SearchSpec)
    spec = await llm.ainvoke([SystemMessage(content=EXTRACT_PROMPT), *state["history"]])
    return {"spec": spec, "attempt": 0}


async def search(state: AgentState) -> dict[str, Any]:
    spec = state["spec"]
    assert spec is not None
    if state["attempt"] == 0:
        get_stream_writer()("_Searching the catalog…_\n\n")
    filters, dropped = _build_filters(spec, state["attempt"])
    results = await asyncio.to_thread(
        search_books, query_text=spec.semantic_query, filters=filters, top_k=TOP_K
    )
    return {"results": results, "dropped_filters": dropped}


def relax(state: AgentState) -> dict[str, Any]:
    return {"attempt": state["attempt"] + 1}


def rerank(state: AgentState) -> dict[str, Any]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for match in sorted(state["results"], key=lambda m: m["score"], reverse=True):
        title = (match["metadata"].get("title") or "").strip().lower()
        if title and title in seen:
            continue
        seen.add(title)
        unique.append(match)
    return {"results": unique[:MAX_RECOMMENDATIONS]}


async def respond(state: AgentState) -> dict[str, Any]:
    caveat = ""
    if state["dropped_filters"]:
        constraints = " and ".join(state["dropped_filters"])
        caveat = (
            f"- The catalog had no exact match for the requested {constraints}; "
            "be transparent that these are the closest alternatives.\n"
        )
    system = ANSWER_PROMPT.format(books=_format_books(state["results"]), caveat=caveat)
    llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0.4)
    await llm.ainvoke([SystemMessage(content=system), *state["history"]])
    return {}


async def smalltalk(state: AgentState) -> dict[str, Any]:
    llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0.7)
    await llm.ainvoke([SystemMessage(content=SMALLTALK_PROMPT), *state["history"]])
    return {}


def no_results(state: AgentState) -> dict[str, Any]:
    get_stream_writer()(NO_RESULTS_MESSAGE)
    return {}


# --- routing ---------------------------------------------------------------


def route_after_extract(state: AgentState) -> Literal["search", "smalltalk"]:
    spec = state["spec"]
    return "search" if spec is not None and spec.is_book_request else "smalltalk"


def route_after_search(state: AgentState) -> Literal["rerank", "relax", "no_results"]:
    if state["results"]:
        return "rerank"
    if state["attempt"] >= MAX_RELAX_ATTEMPTS:
        return "no_results"
    return "relax"


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract", extract)
    graph.add_node("search", search)
    graph.add_node("relax", relax)
    graph.add_node("rerank", rerank)
    graph.add_node("respond", respond)
    graph.add_node("smalltalk", smalltalk)
    graph.add_node("no_results", no_results)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges("extract", route_after_extract)
    graph.add_conditional_edges("search", route_after_search)
    graph.add_edge("relax", "search")
    graph.add_edge("rerank", "respond")
    graph.add_edge("respond", END)
    graph.add_edge("smalltalk", END)
    graph.add_edge("no_results", END)
    return graph.compile()


GRAPH = _build_graph()


async def stream_book_agent_response(messages: list[ChatMessage]) -> AsyncIterator[str]:
    """Candidate task entrypoint: stream the agent's answer as text chunks."""
    state: AgentState = {
        "history": _to_lc_messages(messages),
        "spec": None,
        "attempt": 0,
        "dropped_filters": [],
        "results": [],
    }
    try:
        async for mode, payload in GRAPH.astream(state, stream_mode=["custom", "messages"]):
            if mode == "custom":
                yield str(payload)
            else:  # "messages": (message_chunk, metadata)
                chunk, metadata = payload
                if (
                    metadata.get("langgraph_node") in STREAMED_NODES
                    and isinstance(chunk.content, str)
                    and chunk.content
                ):
                    yield chunk.content
    except Exception as exc:  # SSE protocol has no error frame — surface as text
        yield f"\n\n⚠️ Something went wrong while generating recommendations: {exc}"
