"""Groq-powered CV gap analysis against matched job descriptions."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from jobmatch.cv_parser import normalize_whitespace
from jobmatch.errors import ConfigurationError
from jobmatch.retrieval import MatchedJob

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Verified against this project's live Groq account (client.models.list())
# and against the real CV+jobs prompt, not just a trivial ping. 5-attempt
# reliability check against a real CV: openai/gpt-oss-120b succeeded 4/5
# (80%) - the failures were either Groq's own server-side JSON validator
# rejecting the generation ("json_validate_failed", empty failed_generation)
# or an incomplete pydantic-schema response, not a rate limit. The
# alternatives tried were worse, not better: allam-2-7b failed 0/3 (it fell
# into degenerate token-repetition loops and separately hit a
# context_length_exceeded error), and qwen/qwen3.8-27b is structurally
# unusable on this account's free tier - it needs ~2048 output tokens per
# request but the enforced output-tokens-per-minute limit is 1000, so it
# will eventually always hit a 429. llama-3.3-70b-versatile no longer exists
# on this account at all (404 model_not_found). Given ~80% single-attempt
# reliability, MAX_ATTEMPTS=3 below pushes the overall failure rate to
# roughly 0.2**3 =~ 0.8%. Override the model via GROQ_MODEL in .env if needed.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Retries before giving up - see reliability note above.
MAX_ATTEMPTS = 3

# Keeps the prompt within context limits regardless of how long a posting is.
MAX_DESCRIPTION_CHARS = 1500

SYSTEM_PROMPT = """You are a CV gap-analysis assistant for JobMatch, a tool that compares a \
candidate's CV against job descriptions they are a good semantic match for.

Rules you must follow exactly:
1. "missing_technologies" must only list technologies, tools, or skills that are explicitly \
requested in the provided job descriptions and that do NOT already appear anywhere in the \
candidate's CV (including synonyms/equivalent phrasing already present). Never invent a \
technology that isn't actually mentioned in the job descriptions.
2. "wording_improvements" must contain EXACTLY 3 items. Each "original" field must be an \
exact, verbatim substring copied from the CV text given to you - do not paraphrase it. Each \
"improved" field must rewrite that same text more clearly/strongly without adding any skill, \
technology, achievement, or experience the candidate did not already state. Never invent or \
embellish the candidate's experience.

Respond with ONLY a single JSON object, no prose before or after it, matching exactly this \
shape:
{
  "missing_technologies": [
    {"name": "string", "mentioned_in": 0, "job_titles": ["string"]}
  ],
  "wording_improvements": [
    {"original": "string", "improved": "string", "reason": "string"}
  ]
}"""


class MissingTechnology(BaseModel):
    name: str
    mentioned_in: int
    job_titles: list[str]


class WordingImprovement(BaseModel):
    original: str
    improved: str
    reason: str


class CVAnalysis(BaseModel):
    missing_technologies: list[MissingTechnology]
    wording_improvements: list[WordingImprovement] = Field(min_length=3, max_length=3)


def _load_groq_credentials() -> tuple[str, str]:
    load_dotenv(ENV_PATH)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not set in .env. Add it (see .env.example) and retry."
        )
    model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return api_key, model


def _build_user_prompt(cv_text: str, jobs: list[MatchedJob]) -> str:
    jobs_block = "\n\n".join(
        f"Job {i + 1}: {job.title} at {job.company or 'Unknown company'}\n"
        f"{job.description[:MAX_DESCRIPTION_CHARS]}"
        for i, job in enumerate(jobs)
    )
    return f"CANDIDATE CV:\n{cv_text}\n\nMATCHED JOB DESCRIPTIONS:\n{jobs_block}"


def _request_analysis(client, model: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return response.choices[0].message.content


def _normalize_mentioned_in(analysis: CVAnalysis) -> CVAnalysis:
    # Models sometimes report "mentioned_in" inconsistently with the
    # "job_titles" they listed for the same entry. job_titles is the concrete,
    # checkable list, so it's the source of truth; mentioned_in is just its count.
    fixed = [
        tech.model_copy(update={"mentioned_in": len(tech.job_titles)})
        for tech in analysis.missing_technologies
    ]
    return analysis.model_copy(update={"missing_technologies": fixed})


def _find_unmatched_original(analysis: CVAnalysis, normalized_cv: str) -> str | None:
    """Return the first wording_improvements[].original not found in the CV, or None.

    Comparison is whitespace-normalized on both sides: PDF/DOCX extraction
    breaks lines mid-sentence, so comparing raw strings would reject a
    perfectly valid verbatim quote just because its line breaks don't land in
    the same place as the source CV text.
    """
    for improvement in analysis.wording_improvements:
        if normalize_whitespace(improvement.original) not in normalized_cv:
            return improvement.original
    return None


def analyze_cv(
    cv_text: str,
    jobs: list[MatchedJob],
    client=None,
    model: str | None = None,
) -> CVAnalysis:
    """Ask Groq to compare `cv_text` against `jobs` and return a validated CVAnalysis."""
    if client is None or model is None:
        api_key, env_model = _load_groq_credentials()
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key)
        if model is None:
            model = env_model

    from groq import APIStatusError

    normalized_cv = normalize_whitespace(cv_text)
    user_prompt = _build_user_prompt(cv_text, jobs)

    def _attempt(prompt: str) -> tuple[CVAnalysis | None, Exception | None]:
        # A bad Groq API response (e.g. the model's own server-side JSON
        # validator rejecting a generation, so no content comes back at all)
        # is treated the same as a pydantic validation failure: worth
        # retrying with the error fed back to the model.
        try:
            raw = _request_analysis(client, model, prompt)
        except APIStatusError as exc:
            return None, exc
        try:
            parsed = CVAnalysis.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            return None, exc

        unmatched = _find_unmatched_original(parsed, normalized_cv)
        if unmatched is not None:
            return None, ValueError(
                "A wording_improvements[].original value was not found (even after "
                f"whitespace normalization) anywhere in the CV text: {unmatched!r}"
            )

        return _normalize_mentioned_in(parsed), None

    prompt = user_prompt
    last_error: Exception | None = None
    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        result, error = _attempt(prompt)
        if result is not None:
            return result
        last_error = error
        prompt = (
            f"{user_prompt}\n\n"
            f"Your previous response (attempt {attempt_num}) failed validation with this "
            f"error:\n{error}\n"
            "Return ONLY a corrected JSON object matching the schema exactly. Every "
            "wording_improvements[].original value must be an exact substring of the "
            "CANDIDATE CV text above (whitespace/line-break differences are fine, but do "
            "not paraphrase or invent text)."
        )

    raise RuntimeError(
        f"Groq response failed validation {MAX_ATTEMPTS} times; aborting. "
        f"Last error: {last_error}"
    ) from last_error
