import pytest

from aciids.sentiment import aggregate_daily_sentiment, chunk_text


def test_chunk_text_splits_long_strings() -> None:
    chunks = chunk_text("abcdefghij", max_length=4)
    assert chunks == ["abcd", "efgh", "ij"]


def test_chunk_text_rejects_non_positive_lengths() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", max_length=0)


def test_aggregate_daily_sentiment_summarizes_scores() -> None:
    summary = aggregate_daily_sentiment(
        [
            {
                "date": "2026-03-20T09:00:00",
                "labels": ["positive", "neutral"],
                "scores": [0.8, 0.5],
            },
            {
                "date": "2026-03-20T12:00:00",
                "labels": ["negative"],
                "scores": [0.7],
            },
        ]
    )

    assert len(summary) == 1
    record = summary[0]
    assert record["date"] == "2026-03-20"
    assert record["total_articles"] == 2
    assert record["positive_chunks"] == 1
    assert record["neutral_chunks"] == 1
    assert record["negative_chunks"] == 1
    assert record["average_sentiment_score"] == pytest.approx((0.8 + 0.5 + 0.7) / 3)


def test_aggregate_daily_sentiment_validates_alignment() -> None:
    with pytest.raises(ValueError):
        aggregate_daily_sentiment(
            [{"date": "2026-03-20", "labels": ["positive"], "scores": [0.1, 0.2]}]
        )
