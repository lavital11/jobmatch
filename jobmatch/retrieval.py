"""Find jobs in Supabase that best match a candidate's CV."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

from jobmatch.embeddings import embed_texts
from jobmatch.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# The e5 model truncates input at ~512 tokens. A CV can easily run longer than
# that, so instead of silently losing the tail of the document we split it
# into word-based chunks, embed each chunk separately, average the resulting
# vectors, and re-normalize. 350 words keeps each chunk (plus the "query: "
# prefix) comfortably under the 512-token limit even for CVs with long words.
MAX_CHUNK_WORDS = 350


class MatchedJob(BaseModel):
    id: int
    title: str
    company: str | None = None
    url: str | None = None
    location: str | None = None
    description: str
    similarity: float


def _chunk_text(text: str, max_words: int | None = None) -> list[str]:
    if max_words is None:
        max_words = MAX_CHUNK_WORDS
    words = text.split()
    if not words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _embed_query(cv_text: str) -> list[float]:
    chunks = _chunk_text(cv_text)
    prefixed = [f"query: {chunk}" for chunk in chunks]
    vectors = embed_texts(prefixed)

    if len(vectors) == 1:
        return vectors[0]

    averaged = np.mean(np.array(vectors), axis=0)
    norm = np.linalg.norm(averaged)
    if norm > 0:
        averaged = averaged / norm
    return averaged.tolist()


def _get_supabase_client():
    from supabase import create_client

    load_dotenv(ENV_PATH)
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ConfigurationError(
            "SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY are not set in .env."
        )
    return create_client(url, key)


def find_similar_jobs(cv_text: str, match_count: int = 5, client=None) -> list[MatchedJob]:
    """Embed `cv_text` and return the most similar jobs stored in Supabase."""
    client = client or _get_supabase_client()
    query_embedding = _embed_query(cv_text)

    rpc_response = client.rpc(
        "match_jobs",
        {"query_embedding": query_embedding, "match_count": match_count},
    ).execute()
    matches = rpc_response.data or []
    if not matches:
        return []

    # match_jobs() only returns id/title/company/url/similarity (see
    # sql/001_create_jobs.sql); description and location live on the jobs
    # table, so fetch them in a second query keyed by the matched ids.
    ids = [match["id"] for match in matches]
    details_response = (
        client.table("jobs").select("id,description,location").in_("id", ids).execute()
    )
    details_by_id = {row["id"]: row for row in details_response.data or []}

    results = [
        MatchedJob(
            id=match["id"],
            title=match["title"],
            company=match.get("company"),
            url=match.get("url"),
            location=details_by_id.get(match["id"], {}).get("location"),
            description=details_by_id.get(match["id"], {}).get("description", ""),
            similarity=match["similarity"],
        )
        for match in matches
    ]
    results.sort(key=lambda job: job.similarity, reverse=True)
    return results
