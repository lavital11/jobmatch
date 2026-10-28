"""Integration test: find_similar_jobs() against the real Supabase database.

Unlike tests/test_retrieval.py (fully mocked), this hits Supabase for real
over the network. It's marked `integration` (registered in pyproject.toml)
and skips automatically with a clear reason when Supabase credentials aren't
configured, so the regular test run never requires network access.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from jobmatch.retrieval import MatchedJob, find_similar_jobs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

pytestmark = pytest.mark.integration

# Short, generic, non-personal sample CV - deliberately not a real resume.
SAMPLE_CV = (
    "Junior developer with hands-on experience in Python, Java, Node.js, and SQL. "
    "Built small web applications and REST APIs, worked with relational databases, "
    "and collaborated in an agile team using Git for version control."
)


def _missing_supabase_credentials() -> bool:
    return not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))


@pytest.mark.skipif(
    _missing_supabase_credentials(),
    reason="SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY not set in .env - skipping integration test",
)
def test_find_similar_jobs_against_real_supabase():
    match_count = 5

    results = find_similar_jobs(SAMPLE_CV, match_count=match_count)

    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= match_count

    for job in results:
        assert isinstance(job, MatchedJob)
        assert job.title
        assert job.url
        assert job.description
        assert -1.0 <= job.similarity <= 1.0

    similarities = [job.similarity for job in results]
    assert similarities == sorted(similarities, reverse=True)
