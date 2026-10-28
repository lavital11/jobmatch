from jobmatch.retrieval import MatchedJob, _chunk_text, _embed_query, find_similar_jobs


class FakeResult:
    def __init__(self, data):
        self.data = data


class FakeTableQuery:
    def __init__(self, data):
        self._data = data

    def select(self, *_args, **_kwargs):
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def execute(self):
        return FakeResult(self._data)


class FakeRpcQuery:
    def __init__(self, data):
        self._data = data

    def execute(self):
        return FakeResult(self._data)


class FakeSupabaseClient:
    def __init__(self, rpc_data, table_data):
        self._rpc_data = rpc_data
        self._table_data = table_data
        self.rpc_calls = []

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        return FakeRpcQuery(self._rpc_data)

    def table(self, name):
        assert name == "jobs"
        return FakeTableQuery(self._table_data)


def test_chunk_text_short_returns_single_chunk():
    text = "Python developer with FastAPI experience."
    chunks = _chunk_text(text, max_words=350)
    assert chunks == [text]


def test_chunk_text_splits_long_text_into_word_bounded_chunks():
    words = [f"word{i}" for i in range(900)]
    text = " ".join(words)

    chunks = _chunk_text(text, max_words=350)

    assert len(chunks) == 3
    assert chunks[0] == " ".join(words[0:350])
    assert chunks[1] == " ".join(words[350:700])
    assert chunks[2] == " ".join(words[700:900])
    # No words lost or reordered across chunk boundaries.
    assert " ".join(chunks).split() == words


def test_embed_query_single_chunk_uses_query_prefix(monkeypatch):
    captured = {}

    def fake_embed_texts(texts):
        captured["texts"] = texts
        return [[1.0, 0.0, 0.0]]

    monkeypatch.setattr("jobmatch.retrieval.embed_texts", fake_embed_texts)

    vector = _embed_query("Short CV text")

    assert captured["texts"] == ["query: Short CV text"]
    assert vector == [1.0, 0.0, 0.0]


def test_embed_query_multi_chunk_averages_and_renormalizes(monkeypatch):
    def fake_embed_texts(texts):
        assert len(texts) == 2
        # Two orthogonal unit vectors; average has norm < 1 before renormalizing.
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr("jobmatch.retrieval.embed_texts", fake_embed_texts)
    monkeypatch.setattr("jobmatch.retrieval.MAX_CHUNK_WORDS", 2)

    vector = _embed_query("aa bb cc dd")

    expected = 1 / (2 ** 0.5)
    assert abs(vector[0] - expected) < 1e-9
    assert abs(vector[1] - expected) < 1e-9
    norm = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_find_similar_jobs_merges_rpc_and_table_data_and_sorts(monkeypatch):
    monkeypatch.setattr("jobmatch.retrieval.embed_texts", lambda texts: [[0.1, 0.2, 0.3]])

    rpc_data = [
        {"id": 1, "title": "Junior Dev", "company": "Acme", "url": "https://a", "similarity": 0.5},
        {"id": 2, "title": "Backend Dev", "company": "Beta", "url": "https://b", "similarity": 0.9},
    ]
    table_data = [
        {"id": 1, "description": "Build APIs", "location": "Tel Aviv"},
        {"id": 2, "description": "Scale services", "location": "Haifa"},
    ]
    client = FakeSupabaseClient(rpc_data, table_data)

    results = find_similar_jobs("some cv text", match_count=5, client=client)

    assert [job.id for job in results] == [2, 1]
    assert all(isinstance(job, MatchedJob) for job in results)
    assert results[0].description == "Scale services"
    assert results[0].location == "Haifa"
    assert client.rpc_calls[0][0] == "match_jobs"
    assert client.rpc_calls[0][1]["match_count"] == 5


def test_find_similar_jobs_returns_empty_list_when_no_matches(monkeypatch):
    monkeypatch.setattr("jobmatch.retrieval.embed_texts", lambda texts: [[0.1, 0.2, 0.3]])
    client = FakeSupabaseClient(rpc_data=[], table_data=[])

    results = find_similar_jobs("cv text", client=client)

    assert results == []
