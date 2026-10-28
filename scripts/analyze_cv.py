"""CLI: read data/cv.txt, find similar jobs, run the Groq gap analysis, print a report."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from jobmatch.analysis import analyze_cv  # noqa: E402
from jobmatch.retrieval import find_similar_jobs  # noqa: E402

CV_PATH = PROJECT_ROOT / "data" / "cv.txt"
ENV_PATH = PROJECT_ROOT / ".env"


def main() -> None:
    load_dotenv(ENV_PATH)

    if not CV_PATH.exists():
        print(f"Could not find {CV_PATH}. Add your CV as plain text there, then re-run this script.")
        sys.exit(1)

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set in .env. Add it (see .env.example), then re-run this script.")
        sys.exit(1)

    cv_text = CV_PATH.read_text(encoding="utf-8").strip()
    if not cv_text:
        print(f"{CV_PATH} is empty.")
        sys.exit(1)

    print("Finding similar jobs ...")
    try:
        jobs = find_similar_jobs(cv_text, match_count=5)
    except Exception as exc:
        print(f"Failed to find similar jobs: {exc}")
        sys.exit(1)

    if not jobs:
        print("No matching jobs found in Supabase. Did you run scripts/ingest_jobs.py?")
        sys.exit(1)

    print(f"Found {len(jobs)} matching jobs. Running Groq analysis ...")
    try:
        analysis = analyze_cv(cv_text, jobs)
    except Exception as exc:
        print(f"CV analysis failed: {exc}")
        sys.exit(1)

    print("\n=== Matched Jobs ===")
    for job in jobs:
        print(f"- [{job.similarity:.3f}] {job.title} @ {job.company or 'Unknown company'} "
              f"({job.location or 'Unknown location'})")
        print(f"  {job.url}")

    print("\n=== Missing Technologies ===")
    if not analysis.missing_technologies:
        print("(none found)")
    for tech in analysis.missing_technologies:
        titles = ", ".join(tech.job_titles)
        print(f"- {tech.name} (mentioned in {tech.mentioned_in} job(s): {titles})")

    print("\n=== Wording Improvements ===")
    for i, improvement in enumerate(analysis.wording_improvements, start=1):
        print(f"{i}. Original: {improvement.original}")
        print(f"   Improved: {improvement.improved}")
        print(f"   Reason:   {improvement.reason}")


if __name__ == "__main__":
    main()
