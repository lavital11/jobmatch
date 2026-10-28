from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jobmatch.analysis import CVAnalysis, MissingTechnology, WordingImprovement, analyze_cv
from jobmatch.retrieval import MatchedJob

# Must contain, verbatim, every "original" quote used by VALID_JSON below -
# analyze_cv now rejects responses whose "original" text isn't actually in the CV.
CV_TEXT = "Experience: worked on stuff and did testing. Also helped team deliver features."

VALID_JSON = """
{
  "missing_technologies": [
    {"name": "Docker", "mentioned_in": 2, "job_titles": ["Backend Dev", "DevOps Jr"]}
  ],
  "wording_improvements": [
    {"original": "worked on stuff", "improved": "built and maintained REST APIs", "reason": "more specific"},
    {"original": "did testing", "improved": "wrote unit and integration tests", "reason": "clearer scope"},
    {"original": "helped team", "improved": "collaborated with a 4-person team on feature delivery", "reason": "quantifies impact"}
  ]
}
"""

# A line-wrapped version of the same CV (what PDF/DOCX extraction often produces
# before normalization) - used to prove whitespace-only differences are tolerated.
CV_TEXT_LINE_WRAPPED = "Experience: worked on\nstuff and did testing.\nAlso helped\nteam deliver features."

UNMATCHED_ORIGINAL_JSON = """
{
  "missing_technologies": [],
  "wording_improvements": [
    {"original": "worked on stuff", "improved": "built and maintained REST APIs", "reason": "more specific"},
    {"original": "did testing", "improved": "wrote unit and integration tests", "reason": "clearer scope"},
    {"original": "this text is not in the cv at all", "improved": "x", "reason": "y"}
  ]
}
"""

INVALID_JSON = '{"missing_technologies": [], "wording_improvements": []}'  # only 2 required, has 0


def make_client(contents: list[str]) -> SimpleNamespace:
    state = {"calls": []}

    def create(**kwargs):
        state["calls"].append(kwargs)
        content = contents[len(state["calls"]) - 1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client._state = state
    return client


SAMPLE_JOBS = [
    MatchedJob(
        id=1,
        title="Backend Dev",
        company="Acme",
        url="https://a",
        location="Tel Aviv",
        description="Looking for Docker and Python experience.",
        similarity=0.8,
    )
]


def test_missing_technology_model_fields():
    tech = MissingTechnology(name="Docker", mentioned_in=3, job_titles=["A", "B"])
    assert tech.name == "Docker"
    assert tech.mentioned_in == 3
    assert tech.job_titles == ["A", "B"]


def test_wording_improvement_requires_three_items():
    with pytest.raises(ValidationError):
        CVAnalysis(
            missing_technologies=[],
            wording_improvements=[
                WordingImprovement(original="a", improved="b", reason="c"),
            ],
        )


def test_analyze_cv_happy_path_single_call():
    client = make_client([VALID_JSON])

    result = analyze_cv(CV_TEXT, SAMPLE_JOBS, client=client, model="test-model")

    assert isinstance(result, CVAnalysis)
    assert len(result.wording_improvements) == 3
    assert result.missing_technologies[0].name == "Docker"
    assert len(client._state["calls"]) == 1
    sent_kwargs = client._state["calls"][0]
    assert sent_kwargs["response_format"] == {"type": "json_object"}
    assert sent_kwargs["model"] == "test-model"


def test_analyze_cv_accepts_original_despite_line_wrapped_cv():
    # The CV text itself has mid-sentence line breaks (as PDF/DOCX extraction
    # often produces); the model's "original" is a clean single-line quote.
    # Whitespace-normalized comparison must still match.
    client = make_client([VALID_JSON])

    result = analyze_cv(CV_TEXT_LINE_WRAPPED, SAMPLE_JOBS, client=client, model="test-model")

    assert isinstance(result, CVAnalysis)
    assert len(client._state["calls"]) == 1


def test_analyze_cv_retries_once_on_invalid_response_then_succeeds():
    client = make_client([INVALID_JSON, VALID_JSON])

    result = analyze_cv(CV_TEXT, SAMPLE_JOBS, client=client, model="test-model")

    assert isinstance(result, CVAnalysis)
    assert len(client._state["calls"]) == 2
    # The retry prompt must include the validation error so the model can self-correct.
    retry_user_message = client._state["calls"][1]["messages"][1]["content"]
    assert "validation" in retry_user_message.lower()


def test_analyze_cv_retries_when_original_not_found_in_cv():
    client = make_client([UNMATCHED_ORIGINAL_JSON, VALID_JSON])

    result = analyze_cv(CV_TEXT, SAMPLE_JOBS, client=client, model="test-model")

    assert isinstance(result, CVAnalysis)
    assert len(client._state["calls"]) == 2
    retry_user_message = client._state["calls"][1]["messages"][1]["content"]
    assert "not found" in retry_user_message.lower()


def test_analyze_cv_normalizes_mentioned_in_to_job_titles_length():
    inconsistent_json = """
    {
      "missing_technologies": [
        {"name": "Docker", "mentioned_in": 0, "job_titles": ["Backend Dev", "DevOps Jr"]}
      ],
      "wording_improvements": [
        {"original": "worked on stuff", "improved": "b", "reason": "c"},
        {"original": "did testing", "improved": "e", "reason": "f"},
        {"original": "helped team", "improved": "h", "reason": "i"}
      ]
    }
    """
    client = make_client([inconsistent_json])

    result = analyze_cv(CV_TEXT, SAMPLE_JOBS, client=client, model="test-model")

    assert result.missing_technologies[0].mentioned_in == 2


def test_analyze_cv_raises_clear_error_after_max_attempts():
    client = make_client([INVALID_JSON, INVALID_JSON, INVALID_JSON])

    with pytest.raises(RuntimeError, match="failed validation 3 times"):
        analyze_cv(CV_TEXT, SAMPLE_JOBS, client=client, model="test-model")

    assert len(client._state["calls"]) == 3
