"""Utilities for chunking text and summarizing sentiment experiment outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def chunk_text(text: str, max_length: int = 512) -> list[str]:
    """Split text into fixed-size chunks by character count."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not text:
        return []
    return [text[index : index + max_length] for index in range(0, len(text), max_length)]


def aggregate_daily_sentiment(scored_articles: Iterable[dict]) -> list[dict[str, float | int | str]]:
    """Aggregate sentiment outputs into a daily summary structure.

    Each article dict is expected to include:
    - `date`: ISO datetime string or date string
    - `scores`: iterable of confidence scores (one per chunk/model output)
    - `labels`: iterable of labels aligned with `scores` (one per chunk/model output)

    The ``positive_chunks``, ``neutral_chunks``, and ``negative_chunks`` fields in
    the returned records count the number of chunk-level label predictions (not
    the number of distinct articles) for each sentiment class.
    """

    daily_metrics: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "total_articles": 0,
            "positive_chunks": 0,
            "neutral_chunks": 0,
            "negative_chunks": 0,
            "total_score": 0.0,
            "total_chunks": 0,
        }
    )

    for article in scored_articles:
        article_date = str(article["date"]).split("T", maxsplit=1)[0]
        labels = list(article.get("labels", []))
        scores = list(article.get("scores", []))

        if len(labels) != len(scores):
            raise ValueError("labels and scores must have the same length")

        metrics = daily_metrics[article_date]
        metrics["total_articles"] += 1
        metrics["total_score"] += sum(scores)
        metrics["total_chunks"] += len(scores)

        for label in labels:
            normalized_label = label.lower()
            if normalized_label == "positive":
                metrics["positive_chunks"] += 1
            elif normalized_label == "neutral":
                metrics["neutral_chunks"] += 1
            elif normalized_label == "negative":
                metrics["negative_chunks"] += 1
            else:
                raise ValueError(f"unsupported sentiment label: {label}")

    summary: list[dict[str, float | int | str]] = []
    for article_date, metrics in sorted(daily_metrics.items()):
        average_score = metrics["total_score"] / metrics["total_chunks"] if metrics["total_chunks"] else 0.0
        summary.append(
            {
                "date": article_date,
                "total_articles": int(metrics["total_articles"]),
                "positive_chunks": int(metrics["positive_chunks"]),
                "neutral_chunks": int(metrics["neutral_chunks"]),
                "negative_chunks": int(metrics["negative_chunks"]),
                "average_sentiment_score": average_score,
            }
        )

    return summary
