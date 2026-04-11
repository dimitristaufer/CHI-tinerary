# CHI Relevance Client

```bash
npm install
npm run dev
```

```bash
npm run build
```

`dist/` is deployed to GitHub Pages via Actions on push.

## Inputs

- Upload up to 10 PDFs and extract text locally in-browser.
- Or provide a Google Scholar / ResearchGate profile URL, fetch abstracts, select relevant entries, and rank from those.
- You can combine both PDFs and selected profile abstracts in one run.
- Profile fetching is browser-only. On static GitHub Pages deployments, Scholar/ResearchGate may block direct scraping.
- For better browser-only reliability, use **Author name (optional)**. The app uses OpenAlex metadata as fallback when direct profile scraping is blocked.

## Matching Modes

The app now supports:

- `TF-IDF (Lexical)`
- `Semantic Embeddings`
- `Hybrid (TF-IDF + Semantic)`

`Semantic` and `Hybrid` run in-browser via `@huggingface/transformers` inside the scoring worker and are configured for local model files only.

## Local Semantic Model Files

Place the model files at:

`public/models/onnx-community/all-MiniLM-L6-v2-ONNX`

At minimum, include the model config/tokenizer files and ONNX weights expected by Transformers.js for that model path.

In the app UI, use **Download semantic model** to fetch/cache the model in-browser (with progress). If semantic assets are already bundled in the deployment, the button switches to **Model ready**.

The prefetch button can download from Hugging Face when local `/models/...` assets are not present in the current deployment.

## Precomputed CHI Semantic Embeddings

To avoid embedding all CHI schedule rows at runtime, precompute and ship schedule embeddings:

```bash
npm run build:semantic-index
```

This writes:

- `public/data/schedule_semantic_embeddings_q4.bin`
- `public/data/schedule_semantic_embeddings_q4.json`

The scoring worker loads these first in Semantic/Hybrid mode and falls back to runtime schedule embedding only if they are missing or incompatible.
