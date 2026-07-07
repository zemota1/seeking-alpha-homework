# Notes

Quick notes on how the agent works. Everything lives in
`backend/app/services/book_agent.py`.

It's a small LangGraph graph:

```
extract -> search -> rerank -> respond
              |
              (relax -> search again if nothing came back)
```

plus a smalltalk branch (for "hi", "thanks", etc.) and a no_results branch.

- **extract**: one LLM call reads the whole chat history and pulls out a search
  spec - a semantic query plus optional author / language / year, and a flag for
  whether it's even a book request. Reading the full history means follow-ups
  like "something shorter by the same author" still work.
- **search**: calls the provided `search_books()` with those filters. It's a sync
  function so I run it in a thread.
- **relax**: if a filtered search comes back empty, drop a filter and try again.
  Order is year first (lots of books have no year in the data), then language,
  then author. It skips filters the user never set, and if an unfiltered search
  is still empty it just stops.
- **rerank**: sort by score, drop duplicate titles, keep the top 6.
- **respond**: streams the answer. It's only allowed to mention books that were
  actually retrieved.

Filters vs. semantic query: author/language/year are exact, so they go to
Pinecone as metadata filters. Genre, mood and "something like Frankenstein" go
into the semantic query instead - the subject tags in the data are too messy to
filter on reliably.

Grounding: the answer prompt gets the retrieved books as the only list it may
mention, and the extract call's tokens are never streamed - only respond/smalltalk
reach the user. Client-sent "system" messages are treated as normal user text.

Model: OpenAI, `gpt-5.4-nano` for the extraction and `gpt-5.4-mini` for the
answer. Clients are cached and built lazily so importing the module doesn't need
a key.

To run: put `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env`, then `make run`
(or `make setup` first for the venv).
