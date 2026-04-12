# AGENTS.md

Guidance for coding agents working in `chi_relevance_client`.

## Scope

- Make changes only inside this project unless explicitly asked.
- Keep behavior browser-only (no new backend/service dependencies) unless explicitly requested.

## Quick Start

```bash
npm install
npm run dev
```

Build check:

```bash
npm run build
```

## Project Layout

- `src/main.js`: main UI and interaction flow.
- `src/score-worker.js`: background scoring pipeline.
- `src/shared/`: shared text/scoring/semantic utilities.
- `public/data/`: schedule index and precomputed semantic embedding artifacts.
- `public/models/`: local Transformers.js model assets.
- `scripts/`: index and semantic embedding build utilities.

## Guardrails

- Preserve local-first behavior (PDF parsing, ranking, and semantic scoring run in-browser).
- Do not reintroduce browser scraping of Google Scholar pages.
- Keep semantic model paths compatible with:
  - `public/models/onnx-community/all-MiniLM-L6-v2-ONNX`
- Prefer minimal, targeted edits; avoid broad refactors unless requested.

## Regeneration Commands

- Rebuild lexical schedule index:
  - `npm run build:index`
- Rebuild precomputed semantic schedule embeddings:
  - `npm run build:semantic-index`
- Verify parity script:
  - `npm run verify:parity`

## Validation Before Handoff

- Run `npm run build` after any JS/CSS/UI change.
- If scoring logic changes, also run `npm run verify:parity`.
- Summarize what changed, what was run, and any remaining risks.
