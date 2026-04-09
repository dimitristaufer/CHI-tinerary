# CHI Relevance Client (No Backend)

This is a static web app where all runtime computation happens in the browser:

- Upload up to 5 PDFs
- Extract text with `pdf.js` client-side
- Run TF-IDF + cosine relevance scoring in a Web Worker
- Add custom boost keywords (comma/newline separated) and auto-refresh ranking
- Show top 20 by default with a `Load more` button
- Render top agenda points with title, authors, abstract, room, building, day, and relevance score

No Flask/API server is required for computation.

## Project layout

- `public/data/schedule_index.json`: prebuilt schedule token index
- `src/main.js`: UI + PDF extraction pipeline
- `src/score-worker.js`: scoring worker
- `src/shared/scoring.js`: scoring logic aligned with `score_chi_schedule_relevance.py`
- `scripts/build_schedule_index.py`: rebuilds the static schedule index from `chi_2026_schedule_export.csv`
- `scripts/verify_parity.mjs`: compares JS scoring top results against Python baseline output

## Install

```bash
cd chi_relevance_client
npm install
```

## Rebuild schedule index (if source CSV changes)

```bash
npm run build:index
```

## Development

```bash
npm run dev
```

## Production build

```bash
npm run build
```

Outputs static assets to `dist/`. Deploy `dist/` on any static host.

## Verification

This compares the JS scorer (fed with `pdftotext`-extracted texts for parity) against `chi_2026_schedule_relevance_sorted.csv`.

```bash
npm run verify:parity
```
# CHI-tinerary
