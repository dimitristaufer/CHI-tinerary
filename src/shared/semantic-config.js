export const SEMANTIC_MODEL_ID = 'onnx-community/all-MiniLM-L6-v2-ONNX';
export const SEMANTIC_MODEL_DTYPE = 'q4';
export const SEMANTIC_MODEL_SUBFOLDER = 'onnx';
export const SEMANTIC_REMOTE_HOST = 'https://huggingface.co';
export const SEMANTIC_MODEL_CACHE_KEY = 'transformers-cache';

function normalizedBaseUrl(rawBase) {
  let base = String(rawBase || '/').trim();
  if (!base.startsWith('/')) base = `/${base}`;
  if (!base.endsWith('/')) base = `${base}/`;
  return base;
}

function appBaseUrl() {
  const candidate =
    typeof import.meta !== 'undefined' &&
    import.meta &&
    import.meta.env &&
    typeof import.meta.env.BASE_URL === 'string'
      ? import.meta.env.BASE_URL
      : '/';
  return normalizedBaseUrl(candidate);
}

function baseUrlPath(pathName) {
  const base = appBaseUrl();
  const normalized = String(pathName || '').replace(/^\/+/, '');
  return `${base}${normalized}`;
}

export const SEMANTIC_LOCAL_MODEL_PUBLIC_DIR = 'models';
export const SEMANTIC_LOCAL_MODEL_ROOT = baseUrlPath(SEMANTIC_LOCAL_MODEL_PUBLIC_DIR).replace(/\/$/, '');
export const SEMANTIC_SCHEDULE_EMBEDDINGS_META_URL = baseUrlPath('data/schedule_semantic_embeddings_q4.json');
export const SEMANTIC_SCHEDULE_EMBEDDINGS_BIN_URL = baseUrlPath('data/schedule_semantic_embeddings_q4.bin');
export const SEMANTIC_REQUIRED_FILES = Object.freeze([
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.txt',
  'onnx/model_q4.onnx',
  'onnx/model_q4.onnx_data',
]);

export function semanticModelPath(fileName) {
  const normalized = String(fileName || '').replace(/^\/+/, '');
  return `${SEMANTIC_LOCAL_MODEL_ROOT}/${SEMANTIC_MODEL_ID}/${normalized}`;
}

export function semanticModelUrl(fileName, base = globalThis.location?.origin) {
  const path = semanticModelPath(fileName);
  if (!base) return path;
  return new URL(path, base).toString();
}

export function semanticRemoteModelUrl(fileName) {
  const normalized = String(fileName || '').replace(/^\/+/, '');
  return `${SEMANTIC_REMOTE_HOST}/${SEMANTIC_MODEL_ID}/resolve/main/${normalized}`;
}
