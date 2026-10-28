# JobMatch

A RAG-based resume/job-description analyzer. Paste or upload a CV, and
JobMatch finds the junior dev jobs it semantically matches best, then uses an
LLM to report which technologies those jobs ask for that the CV is missing,
and suggests concrete wording improvements - grounded in text that's actually
in the CV, not invented.

## Architecture

```
Job postings (data/jobs.json)
        |
        v
scripts/ingest_jobs.py  --embed ("passage: ...")-->  Supabase jobs table (pgvector)
                                                              |
CV (pasted or uploaded)                                      |
        |                                                     |
        v                                                     |
jobmatch/cv_parser.py (PDF/DOCX/TXT -> normalized text)        |
        |                                                     |
        v                                                     |
jobmatch/retrieval.py  --embed ("query: ...")-->  match_jobs() RPC  <--------
        |
        v
jobmatch/analysis.py  -->  Groq (JSON mode, pydantic-validated, retried)
        |
        v
app.py (Streamlit UI)  /  scripts/analyze_cv.py (CLI)
```

- **Ingestion** (`scripts/ingest_jobs.py`): validates `data/jobs.json` with
  pydantic, embeds every posting in one batch, and upserts into Supabase
  (`on_conflict="url"`, so re-running is safe).
- **Embeddings** (`jobmatch/embeddings.py`): loads `intfloat/multilingual-e5-base`
  once (cached) and exposes a single `embed_texts()` used by both ingestion
  and retrieval.
- **pgvector search** (`sql/`, `jobmatch/retrieval.py`): Supabase stores a
  `vector(768)` column with an HNSW/cosine index; `find_similar_jobs()` embeds
  the CV, calls the `match_jobs` Postgres RPC, then fetches `description`/
  `location` in a second query (the RPC only returns summary columns).
- **Groq analysis** (`jobmatch/analysis.py`): sends the CV + matched job
  descriptions to Groq in JSON mode, validates the response with pydantic,
  and retries on failure (see Design decisions below).
- **UI**: `app.py` (Streamlit, upload or paste a CV) and
  `scripts/analyze_cv.py` (CLI) are both thin presentation layers over the
  `jobmatch` package - all retrieval/parsing/AI logic lives there, not in
  the UI code.

## Tech stack

- **Python 3.12**, **Streamlit** (UI), **FastAPI/uvicorn** (reserved for a
  future API layer)
- **Supabase** (Postgres + `pgvector`) for storage and similarity search
- **sentence-transformers** (`intfloat/multilingual-e5-base`, 768-dim) for
  embeddings
- **Groq** (`openai/gpt-oss-120b` by default) for CV gap analysis, JSON mode
- **pydantic** for validating job records, LLM output, and search results
- **pypdf** / **python-docx** for CV file parsing
- **pytest** for unit and integration tests

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in real values for `SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`, and `GROQ_API_KEY` (`GROQ_MODEL` is optional).

Run the SQL migrations in `sql/` **in order** (`001_create_jobs.sql`, then
`002_add_location.sql`) via the Supabase SQL Editor before ingesting any data.

### Ingesting job postings

Put job postings in `data/jobs.json` (see that file for the schema), then:

```
.venv\Scripts\python.exe scripts\ingest_jobs.py
```

### Analyzing a CV (command line)

Put your CV as plain text in `data/cv.txt` (gitignored - it's personal data),
then:

```
.venv\Scripts\python.exe scripts\analyze_cv.py
```

### Running the web UI

```
.venv\Scripts\streamlit.exe run app.py
```

This starts a local Streamlit server (by default at `http://localhost:8501`)
where you can upload a PDF/DOCX/TXT CV or paste text, pick how many jobs to
match against, and get the same analysis as the CLI in a readable, interactive
report.

## Tests

```
.venv\Scripts\python.exe -m pytest
```

This runs the **unit tests** (`tests/test_retrieval.py`,
`tests/test_analysis.py`, `tests/test_cv_parser.py`) - all Supabase/Groq
calls are mocked, so no network access or credentials are needed.

`tests/test_search_integration.py` is marked `integration` and calls the
*real* `find_similar_jobs()` against your actual Supabase database. It skips
automatically (with a clear reason) if `SUPABASE_URL` /
`SUPABASE_SERVICE_ROLE_KEY` aren't set in `.env`. To run only the integration
test:

```
.venv\Scripts\python.exe -m pytest -m integration
```

To run everything except it:

```
.venv\Scripts\python.exe -m pytest -m "not integration"
```

## Key design decisions

- **e5 `query:`/`passage:` prefixes.** `intfloat/multilingual-e5-base`
  requires asymmetric prefixes to produce good similarity scores: job
  postings are embedded with `"passage: {title}\n{company}\n{description}"`
  (`scripts/ingest_jobs.py`), and a CV is embedded with `"query: {chunk}"`
  (`jobmatch/retrieval.py`). Mixing these up silently degrades search quality
  without raising an error, so it's easy to get wrong.
- **CV chunking for long documents.** The e5 model only reads ~512 tokens.
  `find_similar_jobs()` splits a long CV into word-bounded chunks, embeds each
  with the `query:` prefix, averages the resulting vectors, and re-normalizes
  - rather than silently truncating the CV to whatever fits in one pass.
- **pydantic validation with retry, not silent failure.** `analyze_cv()`
  asks Groq for JSON, validates it against a pydantic schema (exactly 3
  wording improvements, well-typed missing-technology entries), and also
  checks that every `original` wording-improvement quote is actually a
  (whitespace-normalized) substring of the CV. Either kind of failure feeds
  the error back to the model and retries (`MAX_ATTEMPTS = 3`) rather than
  returning bad data; if all attempts fail, it raises a clear error instead
  of guessing.
- **Whitespace normalization for text comparison.** PDF/DOCX extraction
  frequently breaks lines mid-sentence. `jobmatch/cv_parser.normalize_whitespace()`
  collapses whitespace runs so that line-wrapped CV text still matches
  cleanly against a model's verbatim quote of it.
- **Never inventing experience.** The system prompt in `jobmatch/analysis.py`
  explicitly forbids listing a "missing" technology that isn't actually
  requested by the matched jobs, and forbids wording improvements that add
  any skill, technology, or achievement the candidate didn't already state.
  The `original`-text substring check above enforces the second rule in code,
  not just via instruction-following.
- **Separation of concerns.** All retrieval/parsing/AI logic lives in the
  `jobmatch` package; `app.py` and `scripts/*.py` only handle input/output and
  call into it. This is what makes the package independently unit-testable
  (mocked clients, no UI) and kept the Streamlit file focused on presentation.
