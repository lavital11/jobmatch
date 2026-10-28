"""Shared exception types for the jobmatch package."""


class ConfigurationError(RuntimeError):
    """Raised when required configuration (e.g. .env values) is missing or invalid.

    A subclass of RuntimeError so existing `except RuntimeError` handling still
    works, but callers that need to tell "you forgot to set something up" apart
    from "the request itself failed" can catch this specifically.
    """
