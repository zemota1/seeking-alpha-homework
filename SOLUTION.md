# Solution notes

My implementation lives entirely in `backend/app/services/book_agent.py`. It turns a
natural-language request into a short, streamed set of recommendations drawn **only** from
the Pinecone `books` index, and handles follow-up turns, small talk, and empty results
gracefully.

## Approach

I modelled the agent as a small LangGraph state machine rather than a single prompt, so
each concern (understanding the request, retrieving, grounding the answer) is an isolated,
testable step:

```
extract -> search -> (relax -> search)* -> rerank -> respond
        \-> smalltalk                    \-> no_results
```

- **extract** — one LLM call with structured output parses the *whole* conversation into a
  `SearchSpec`: a semantic query plus optional author/language/year filters and an
  `is_book_request` flag. Parsing the full history means a follow-up like "something
  shorter by the same author" inherits earlier constraints.
- **search** — runs `search_books()` (the provided helper) with the filters for the
  current attempt. The helper is synchronous, so it's offloaded with `asyncio.to_thread`
  to keep the event loop free.
- **relax** — if a filtered search comes back empty, filters are dropped and the search
  retried (see below).
- **rerank** — sorts by relevance score, de-duplicates by title, and trims to the top
  matches. I used score-based reranking rather than a dedicated reranker model; for a
  ~1000-book catalog the retrieval scores are already a good signal.
- **respond** — streams the final answer, grounded strictly in the retrieved books.

`smalltalk` and `no_results` are dead-end branches for non-book messages and genuinely
empty searches.

## Filters vs. semantic query

Author, language, and publication year become real Pinecone metadata filters, because
those are precise constraints the user stated explicitly. Genre, topic, mood, and "books
like X" go into the **semantic query** instead of a `subjects` filter — the catalog's
subject vocabulary is inconsistent, so semantic retrieval generalises far better than
exact subject matching. When the user references a title ("like Frankenstein"), the
extractor is instructed to describe its themes (gothic, science, moral dilemmas) rather
than lean on the title string.

## Progressive relaxation

If a filtered search returns nothing, filters are relaxed in order of fragility:
publication year first (the field is missing on many documents), then language, then author
(the constraint the user cared about most). Relaxation skips any rung that wouldn't
actually change the query, and an unfiltered search that still returns nothing stops
immediately instead of retrying an identical request. When filters are dropped, the final
answer says so, so the user knows the results are the closest available alternatives rather
than exact matches.

## Grounding

Two measures keep the agent from recommending books it didn't retrieve:

1. The answer prompt is handed the retrieved books as the **only** set it may mention, and
   explicitly told never to invent or allude to a title outside that list.
2. Only tokens from the `respond`/`smalltalk` nodes are streamed to the user — the
   extractor's structured-output tokens never leak into the reply.

Client-supplied `system` messages are also demoted to plain user content, so only the
agent's own prompts carry system authority.

## Streaming

`stream_book_agent_response(messages)` is an `AsyncIterator[str]`; the FastAPI layer wraps
each chunk into SSE. A brief status line ("_Searching the catalog…_") is emitted before
retrieval, then the answer streams token by token as the model produces it.

## Models & running

Extraction uses `gpt-5.4-nano` (fast, cheap, structured), and the answer uses
`gpt-5.4-mini`. The clients are created lazily and cached, so importing the module doesn't
require an API key. Set `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env`, then:

```bash
make setup    # venv + deps + .env
make verify   # Python, deps, Pinecone connectivity
make run      # http://127.0.0.1:8000
```
