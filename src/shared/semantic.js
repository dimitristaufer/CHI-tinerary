import { env, pipeline } from '@huggingface/transformers';
import {
  SEMANTIC_LOCAL_MODEL_ROOT,
  SEMANTIC_MODEL_DTYPE,
  SEMANTIC_MODEL_ID,
  SEMANTIC_MODEL_SUBFOLDER,
} from './semantic-config.js';

export const DEFAULT_SEMANTIC_MODEL_ID = SEMANTIC_MODEL_ID;
export const DEFAULT_LOCAL_MODEL_ROOT = SEMANTIC_LOCAL_MODEL_ROOT;
export const DEFAULT_SEMANTIC_MODEL_DTYPE = SEMANTIC_MODEL_DTYPE;
export const DEFAULT_SEMANTIC_MODEL_SUBFOLDER = SEMANTIC_MODEL_SUBFOLDER;
const DEFAULT_BATCH_SIZE = 24;
const WORK_CHUNK_CHAR_LIMIT = 1200;
const WORK_CHUNK_MAX_COUNT = 12;
const WORK_CHUNK_MIN_SIZE = 200;
const SCHEDULE_TEXT_CHAR_LIMIT = 2000;

let extractorPromise = null;
let extractorMeta = null;
let extractorConfigKey = null;

export function resetSemanticModelState() {
  extractorPromise = null;
  extractorMeta = null;
  extractorConfigKey = null;
}

function compactWhitespace(text) {
  return String(text || '').replace(/\s+/g, ' ').trim();
}

function truncateText(text, maxChars) {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars);
}

function prepareScheduleRowText(row) {
  const title = compactWhitespace(row?.title || '');
  const authors = compactWhitespace(row?.authors || '');
  const abstract = compactWhitespace(row?.abstract || '');
  const sessionType = compactWhitespace(row?.session_type || '');
  const composed = [title, authors, abstract, sessionType].filter(Boolean).join('. ');
  return truncateText(composed, SCHEDULE_TEXT_CHAR_LIMIT);
}

function splitSegmentToFit(segment, maxChars) {
  if (segment.length <= maxChars) return [segment];
  const chunks = [];
  for (let cursor = 0; cursor < segment.length; cursor += maxChars) {
    chunks.push(segment.slice(cursor, cursor + maxChars));
  }
  return chunks;
}

function splitTextIntoChunks(text, { maxChars = WORK_CHUNK_CHAR_LIMIT, maxChunks = WORK_CHUNK_MAX_COUNT } = {}) {
  const cleaned = compactWhitespace(text);
  if (!cleaned) return ['No content'];
  if (cleaned.length <= maxChars) return [cleaned];

  const sentences = cleaned
    .replace(/([.!?])\s+/g, '$1\n')
    .split(/\n+/)
    .map((segment) => segment.trim())
    .filter(Boolean);

  const chunks = [];
  let current = '';
  const appendChunk = () => {
    if (!current) return;
    chunks.push(current.trim());
    current = '';
  };

  for (const sentence of sentences) {
    const pieces = splitSegmentToFit(sentence, maxChars);
    for (const piece of pieces) {
      if (!current) {
        current = piece;
        continue;
      }
      const next = `${current} ${piece}`;
      if (next.length <= maxChars) {
        current = next;
      } else {
        appendChunk();
        current = piece;
      }
    }
  }
  appendChunk();

  const filtered = chunks.filter((chunk) => chunk.length >= WORK_CHUNK_MIN_SIZE || chunks.length === 1);
  return filtered.slice(0, maxChunks);
}

function normalizeVector(vec) {
  let normSq = 0;
  for (let i = 0; i < vec.length; i += 1) {
    normSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(normSq);
  if (!norm) return vec;
  for (let i = 0; i < vec.length; i += 1) {
    vec[i] /= norm;
  }
  return vec;
}

function averageVectorsNormalized(vectors, dimension) {
  if (!vectors.length) {
    return new Float32Array(dimension);
  }
  const dim = vectors[0]?.length || dimension;
  const avg = new Float32Array(dim);
  for (const vec of vectors) {
    for (let i = 0; i < dim; i += 1) {
      avg[i] += vec[i];
    }
  }
  const inv = 1 / vectors.length;
  for (let i = 0; i < dim; i += 1) {
    avg[i] *= inv;
  }
  return normalizeVector(avg);
}

function configKeyFor({ modelId, localModelRoot, allowRemoteModels, dtype, subfolder }) {
  return `${modelId}::${localModelRoot}::${allowRemoteModels ? '1' : '0'}::${dtype || ''}::${subfolder || ''}`;
}

function selectDeviceCandidates() {
  const hasWebGpu = typeof navigator !== 'undefined' && navigator && 'gpu' in navigator;
  return hasWebGpu ? ['webgpu', 'wasm'] : ['wasm'];
}

function extractErrorMessage(error) {
  if (!error) return 'unknown error';
  if (error instanceof Error) return error.message;
  return String(error);
}

async function createExtractor({ modelId, localModelRoot, allowRemoteModels, progressCallback, dtype, subfolder }) {
  env.allowLocalModels = true;
  env.allowRemoteModels = Boolean(allowRemoteModels);
  env.localModelPath = localModelRoot;

  const attempts = [];
  const candidates = selectDeviceCandidates();
  for (const device of candidates) {
    try {
      const extractor = await pipeline('feature-extraction', modelId, {
        device,
        local_files_only: !allowRemoteModels,
        progress_callback: typeof progressCallback === 'function' ? progressCallback : undefined,
        dtype,
        subfolder,
      });
      return {
        extractor,
        device,
      };
    } catch (error) {
      attempts.push(`${device}: ${extractErrorMessage(error)}`);
    }
  }

  throw new Error(
    `Could not load semantic model '${modelId}'. Tried devices ${candidates.join(', ')}. ` +
      `Place model files under '${localModelRoot}/${modelId}'. Details: ${attempts.join(' | ')}`
  );
}

async function ensureExtractor({
  modelId = DEFAULT_SEMANTIC_MODEL_ID,
  localModelRoot = DEFAULT_LOCAL_MODEL_ROOT,
  allowRemoteModels = false,
  progressCallback = null,
  dtype = DEFAULT_SEMANTIC_MODEL_DTYPE,
  subfolder = DEFAULT_SEMANTIC_MODEL_SUBFOLDER,
} = {}) {
  const key = configKeyFor({ modelId, localModelRoot, allowRemoteModels, dtype, subfolder });
  if (extractorPromise && extractorConfigKey === key) {
    const loaded = await extractorPromise;
    if (typeof progressCallback === 'function') {
      progressCallback({ status: 'ready', model: modelId, progress: 100 });
    }
    return loaded;
  }

  extractorConfigKey = key;
  extractorMeta = null;
  const loadPromise = createExtractor({
    modelId,
    localModelRoot,
    allowRemoteModels,
    progressCallback,
    dtype,
    subfolder,
  }).then((loaded) => {
    extractorMeta = {
      modelId,
      localModelRoot,
      device: loaded.device,
      allowRemoteModels: Boolean(allowRemoteModels),
      dtype,
      subfolder,
    };
    return loaded;
  });
  extractorPromise = loadPromise;

  try {
    const loaded = await loadPromise;
    if (typeof progressCallback === 'function') {
      progressCallback({ status: 'ready', model: modelId, progress: 100 });
    }
    return loaded;
  } catch (error) {
    if (extractorConfigKey === key) {
      extractorPromise = null;
      extractorMeta = null;
      extractorConfigKey = null;
    }
    throw error;
  }
}

function rowsFromTensor(output) {
  const data = output?.data;
  if (!data || typeof data.length !== 'number') {
    throw new Error('Embedding output is missing tensor data.');
  }
  const dims = Array.isArray(output?.dims) ? output.dims : [1, data.length];
  const dim = dims[dims.length - 1];
  const count = dims.length > 1 ? dims[0] : 1;
  const vectors = new Array(count);
  for (let row = 0; row < count; row += 1) {
    const start = row * dim;
    const end = start + dim;
    const vec = new Float32Array(dim);
    vec.set(data.subarray(start, end));
    vectors[row] = vec;
  }
  return vectors;
}

export function buildScheduleSemanticTexts(scheduleRows) {
  return scheduleRows.map((row) => {
    const text = prepareScheduleRowText(row);
    return text || 'Untitled CHI session';
  });
}

export function buildWorkChunkPlan(worksTexts, customKeywords = []) {
  const normalizedKeywords = Array.isArray(customKeywords)
    ? customKeywords
        .map((entry) => compactWhitespace(entry))
        .filter(Boolean)
    : [];
  const normalizedWorksTexts = Array.isArray(worksTexts)
    ? worksTexts
        .map((entry) => compactWhitespace(entry))
        .filter(Boolean)
    : [];
  const hasWorkTexts = normalizedWorksTexts.length > 0;
  const sourceTexts = hasWorkTexts
    ? normalizedWorksTexts
    : normalizedKeywords.length
      ? [`Focus topics: ${normalizedKeywords.join(', ')}.`]
      : [];
  const keywordSuffix = hasWorkTexts && normalizedKeywords.length > 0 ? ` Focus topics: ${normalizedKeywords.join(', ')}.` : '';

  const allChunks = [];
  const spans = [];

  for (const sourceText of sourceTexts) {
    const base = compactWhitespace(sourceText);
    const text = keywordSuffix ? `${base} ${keywordSuffix}`.trim() : base;
    const chunks = splitTextIntoChunks(text, {
      maxChars: WORK_CHUNK_CHAR_LIMIT,
      maxChunks: WORK_CHUNK_MAX_COUNT,
    });

    const start = allChunks.length;
    for (const chunk of chunks) {
      allChunks.push(chunk);
    }
    spans.push({ start, count: chunks.length });
  }

  return { chunks: allChunks, spans };
}

export function reduceChunkVectorsToWorkVectors(chunkVectors, spans) {
  const fallbackDim = chunkVectors[0]?.length || 384;
  return spans.map((span) => {
    if (!span.count) return new Float32Array(fallbackDim);
    const vectors = chunkVectors.slice(span.start, span.start + span.count);
    return averageVectorsNormalized(vectors, fallbackDim);
  });
}

export async function preloadSemanticModel(options = {}) {
  const {
    modelId = DEFAULT_SEMANTIC_MODEL_ID,
    localModelRoot = DEFAULT_LOCAL_MODEL_ROOT,
    allowRemoteModels = false,
    progressCallback = null,
    dtype = DEFAULT_SEMANTIC_MODEL_DTYPE,
    subfolder = DEFAULT_SEMANTIC_MODEL_SUBFOLDER,
  } = options;

  const loaded = await ensureExtractor({
    modelId,
    localModelRoot,
    allowRemoteModels,
    progressCallback,
    dtype,
    subfolder,
  });

  return {
    model: extractorMeta,
    device: loaded.device,
  };
}

export async function embedTexts(texts, options = {}) {
  if (!Array.isArray(texts) || texts.length === 0) {
    return {
      vectors: [],
      model: extractorMeta,
    };
  }

  const {
    batchSize = DEFAULT_BATCH_SIZE,
    onBatch = null,
    modelId = DEFAULT_SEMANTIC_MODEL_ID,
    localModelRoot = DEFAULT_LOCAL_MODEL_ROOT,
    allowRemoteModels = false,
    progressCallback = null,
    dtype = DEFAULT_SEMANTIC_MODEL_DTYPE,
    subfolder = DEFAULT_SEMANTIC_MODEL_SUBFOLDER,
  } = options;
  const normalizedTexts = texts.map((text) => compactWhitespace(text) || 'No content');
  const { extractor } = await ensureExtractor({
    modelId,
    localModelRoot,
    allowRemoteModels,
    progressCallback,
    dtype,
    subfolder,
  });

  const vectors = new Array(normalizedTexts.length);
  for (let cursor = 0; cursor < normalizedTexts.length; cursor += batchSize) {
    const batch = normalizedTexts.slice(cursor, cursor + batchSize);
    const output = await extractor(batch, { pooling: 'mean', normalize: true });
    const rows = rowsFromTensor(output);
    for (let i = 0; i < rows.length; i += 1) {
      vectors[cursor + i] = rows[i];
    }
    if (typeof onBatch === 'function') {
      onBatch({
        processed: Math.min(cursor + batch.length, normalizedTexts.length),
        total: normalizedTexts.length,
      });
    }
  }

  return {
    vectors,
    model: extractorMeta,
  };
}
