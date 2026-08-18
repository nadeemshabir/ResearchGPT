"""Tests for tolerant parsing of environment values.

`python-dotenv` and Docker's `--env-file` do not read the same file the same
way. dotenv strips inline comments and surrounding whitespace; Docker passes
the raw string. A `.env` line that works locally can therefore kill a container
on startup, which is exactly what happened:

    EMBEDDING_MODEL= all-MiniLM-L6-v2 #

became `sentence-transformers/ all-MiniLM-L6-v2 # ` inside the image and failed
with "Repo id must use alphanumeric chars".
"""

import pytest

from src.config import Settings


def build(**env: str) -> Settings:
    """A Settings built from explicit values, bypassing any real .env."""
    return Settings(_env_file=None, **env)  # type: ignore[arg-type,call-arg]


def test_leading_and_trailing_whitespace_is_stripped() -> None:
    assert build(embedding_model="  all-MiniLM-L6-v2  ").embedding_model == ("all-MiniLM-L6-v2")


def test_an_inline_comment_is_dropped() -> None:
    """Docker passes it through verbatim; dotenv does not. Both must agree."""
    assert build(embedding_model="all-MiniLM-L6-v2 # the default").embedding_model == (
        "all-MiniLM-L6-v2"
    )


def test_the_exact_failing_value_is_recovered() -> None:
    """The literal string that took the container down."""
    assert build(embedding_model=" all-MiniLM-L6-v2 # ").embedding_model == ("all-MiniLM-L6-v2")


def test_a_bare_trailing_hash_is_dropped() -> None:
    assert build(embedding_model="all-MiniLM-L6-v2 #").embedding_model == ("all-MiniLM-L6-v2")


def test_a_hash_without_preceding_whitespace_is_kept() -> None:
    """Only ` #` starts a comment, so values containing # survive.

    Keys and URL fragments legitimately contain one, and silently truncating a
    credential would be a far worse failure than the one this fixes.
    """
    assert build(collection_name="papers#2024").collection_name == "papers#2024"


def test_provider_names_are_cleaned_before_the_literal_check() -> None:
    """A padded value would otherwise fail validation rather than be corrected."""
    assert build(llm_provider=" gemini ").llm_provider == "gemini"


def test_numeric_values_still_parse_when_padded() -> None:
    settings = build(chunk_size=" 800 ", semantic_weight=" 0.5 ")

    assert settings.chunk_size == 800
    assert settings.semantic_weight == pytest.approx(0.5)


def test_cleaning_does_not_disturb_ordinary_values() -> None:
    assert build(embedding_model="all-mpnet-base-v2").embedding_model == ("all-mpnet-base-v2")
