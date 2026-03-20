# Research code organization

This directory contains the original research-oriented implementation for the ACIIDS 2025 paper.

## Folders
- `pipeline/` — main end-to-end scripts connecting preprocessing, pricing, and RL trading.
- `analysis/` — standalone analysis scripts for Heston and GARCH experiments.
- `pricing_engine/` — model training and pricing experiments that were part of the broader implementation work.
- `sentiment_analysis/` — scripts for data cleaning and sentiment processing.
- `archive/` — earlier snapshots and duplicate experiment variants kept for provenance, but moved out of the main surface area to reduce clutter.

## Notes
The research code is intentionally preserved close to the original paper implementation. The lightweight `src/aciids` package exists to expose a smaller, testable subset of the functionality without rewriting the whole project into a different architecture.
