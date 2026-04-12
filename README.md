# CHI Relevance Client

Browser-only CHI agenda ranking app.

## Current State

- Input sources:
  - PDFs (text extracted locally with `pdfjs-dist`)
  - OpenAlex author lookup -> selectable profile abstracts
  - Manual keywords
- Inputs can be combined in a single run.
- Matching modes:
  - `tfidf`
  - `semantic`
  - `hybrid`
- Semantic/hybrid scoring runs in a Web Worker with `@huggingface/transformers`.
- Semantic mode supports:
  - bundled local model files under `public/models/...`
  - in-browser model download/cache from the UI
- Worker uses precomputed CHI semantic vectors first and falls back to local runtime embedding only if needed.
- Results include relevance scores, mode labeling, and `.ics` calendar export buttons.

## Run

```bash
npm install
npm run dev
```

Build:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

## Regeneration Scripts

```bash
npm run build:index
npm run build:semantic-index
npm run verify:parity
```

## Required Artifacts

- Schedule index:
  - `public/data/schedule_index.json`
- Precomputed semantic schedule vectors:
  - `public/data/schedule_semantic_embeddings_q4.json`
  - `public/data/schedule_semantic_embeddings_q4.bin`
- Semantic model path:
  - `public/models/onnx-community/all-MiniLM-L6-v2-ONNX`

## Notes

- Google Scholar pages are not scraped in browser-only mode.
- ResearchGate URLs are only used as name hints for OpenAlex lookup.
