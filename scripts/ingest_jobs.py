"""Validate data/jobs.json, embed each job, and upsert into Supabase's jobs table."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jobmatch.embeddings import EMBEDDING_MODEL_NAME, embed_texts  # noqa: E402

ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"

ENV_EXAMPLE_CONTENT = (
    "SUPABASE_URL=https://your-project-ref.supabase.co\n"
    "SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here\n"
    "GROQ_API_KEY=your-groq-api-key-here\n"
    "GROQ_MODEL=openai/gpt-oss-120b\n"
)


class JobRecord(BaseModel):
    title: str
    company: str | None = None
    url: str
    location: str | None = None
    description: str
    source: str

    @field_validator("title", "description", "url", "source")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value


def load_credentials() -> tuple[str, str]:
    if not ENV_PATH.exists():
        if not ENV_EXAMPLE_PATH.exists():
            ENV_EXAMPLE_PATH.write_text(ENV_EXAMPLE_CONTENT, encoding="utf-8")
            print(f"Created {ENV_EXAMPLE_PATH} with placeholder values.")
        print(f"No .env file found at {ENV_PATH}.")
        print("Copy .env.example to .env and fill in your real SUPABASE_URL and "
              "SUPABASE_SERVICE_ROLE_KEY, then re-run this script.")
        sys.exit(1)

    load_dotenv(ENV_PATH)
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    missing = [name for name, value in (("SUPABASE_URL", url), ("SUPABASE_SERVICE_ROLE_KEY", key)) if not value]
    if missing:
        print(f"Missing required environment variable(s) in .env: {', '.join(missing)}")
        sys.exit(1)

    print("Loaded Supabase credentials from .env.")
    return url, key  # type: ignore[return-value]


def load_and_validate_jobs() -> list[JobRecord]:
    if not JOBS_PATH.exists():
        print(f"Could not find {JOBS_PATH}. Run the job-collection step first.")
        sys.exit(1)

    try:
        raw_jobs = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{JOBS_PATH} is not valid JSON: {exc}")
        sys.exit(1)

    print(f"Loaded {len(raw_jobs)} raw job entries from {JOBS_PATH}")

    valid_jobs: list[JobRecord] = []
    for i, raw in enumerate(raw_jobs):
        try:
            valid_jobs.append(JobRecord(**raw))
        except ValidationError as exc:
            bad_url = raw.get("url", "<no url>") if isinstance(raw, dict) else "<malformed entry>"
            print(f"Skipping job #{i} ({bad_url}): {exc}")

    if not valid_jobs:
        print("No valid jobs to ingest after validation. Aborting.")
        sys.exit(1)

    print(f"{len(valid_jobs)}/{len(raw_jobs)} jobs passed validation.")
    return valid_jobs


def build_passage_text(job: JobRecord) -> str:
    # e5 models require the "passage: " prefix for documents being indexed.
    return f"passage: {job.title}\n{job.company or ''}\n{job.description}"


def embed_jobs(jobs: list[JobRecord]) -> list[list[float]]:
    texts = [build_passage_text(job) for job in jobs]
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME} ...")
    print(f"Embedding {len(texts)} jobs in one batch ...")
    try:
        return embed_texts(texts)
    except Exception as exc:
        print(f"Failed to embed jobs with {EMBEDDING_MODEL_NAME}: {exc}")
        sys.exit(1)


def upsert_jobs(url: str, key: str, jobs: list[JobRecord], embeddings: list[list[float]]) -> int:
    from supabase import create_client

    records = [
        {
            "title": job.title,
            "company": job.company,
            "description": job.description,
            "url": job.url,
            "location": job.location,
            "embedding": embedding,
        }
        for job, embedding in zip(jobs, embeddings)
    ]

    try:
        client = create_client(url, key)
        response = client.table("jobs").upsert(records, on_conflict="url").execute()
    except Exception as exc:
        print(f"Failed to upsert jobs into Supabase: {exc}")
        sys.exit(1)

    return len(response.data or [])


def main() -> None:
    supabase_url, supabase_key = load_credentials()
    jobs = load_and_validate_jobs()
    embeddings = embed_jobs(jobs)

    print(f"Upserting {len(jobs)} rows into Supabase 'jobs' table (on_conflict=url) ...")
    upserted_count = upsert_jobs(supabase_url, supabase_key, jobs, embeddings)
    print(f"Upserted {upserted_count} rows.")


if __name__ == "__main__":
    main()
