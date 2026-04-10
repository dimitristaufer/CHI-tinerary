import { scoreSchedule } from './shared/scoring.js';
import {
  buildScheduleSemanticTexts,
  buildWorkChunkPlan,
  embedTexts,
  preloadSemanticModel,
  reduceChunkVectorsToWorkVectors,
} from './shared/semantic.js';
import {
  SEMANTIC_LOCAL_MODEL_ROOT,
  SEMANTIC_SCHEDULE_EMBEDDINGS_BIN_URL,
  SEMANTIC_SCHEDULE_EMBEDDINGS_META_URL,
  SEMANTIC_MODEL_DTYPE,
  SEMANTIC_MODEL_ID,
  SEMANTIC_MODEL_SUBFOLDER,
} from './shared/semantic-config.js';

let scheduleIndex = null;
let semanticScheduleCache = null;

const HYBRID_SEMANTIC_WEIGHT = 0.55;
const VALID_MATCHING_MODES = new Set(['tfidf', 'semantic', 'hybrid']);
const SEMANTIC_OPTIONS = {
  modelId: SEMANTIC_MODEL_ID,
  localModelRoot: SEMANTIC_LOCAL_MODEL_ROOT,
  allowRemoteModels: false,
  dtype: SEMANTIC_MODEL_DTYPE,
  subfolder: SEMANTIC_MODEL_SUBFOLDER,
};

const SEMANTIC_PREFETCH_OPTIONS = {
  ...SEMANTIC_OPTIONS,
  allowRemoteModels: true,
};

function normalizeMatchingMode(rawMode) {
  const normalized = String(rawMode || 'tfidf')
    .trim()
    .toLowerCase();
  return VALID_MATCHING_MODES.has(normalized) ? normalized : 'tfidf';
}

function postProgress(requestId, message, extra = {}) {
  self.postMessage({
    type: 'progress',
    requestId,
    message,
    ...extra,
  });
}

function modelLoadProgressMessage(info) {
  const status = String(info?.status || '');
  if (status === 'progress_total') {
    const percent = Number(info.progress);
    if (Number.isFinite(percent)) {
      return `Downloading semantic model (${Math.round(percent)}%)`;
    }
    return 'Downloading semantic model...';
  }
  if (status === 'download' && info.file) {
    return `Downloading model file: ${info.file}`;
  }
  if (status === 'initiate' && info.file) {
    return `Preparing model file: ${info.file}`;
  }
  if (status === 'done' && info.file) {
    return `Cached model file: ${info.file}`;
  }
  if (status === 'ready') {
    return 'Semantic model ready.';
  }
  return 'Preparing semantic model...';
}

function toFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function relayModelLoadProgress(requestId, info) {
  const total = toFiniteNumber(info?.total);
  const loaded = toFiniteNumber(info?.loaded);
  let percent = toFiniteNumber(info?.progress);
  if (percent == null && total && loaded != null) {
    percent = (loaded / total) * 100;
  }
  if (percent != null) {
    percent = Math.max(0, Math.min(100, percent));
  }

  postProgress(requestId, modelLoadProgressMessage(info), {
    stage: 'model_download',
    rawStatus: info?.status || '',
    file: info?.file || '',
    loaded: loaded ?? undefined,
    total: total ?? undefined,
    percent: percent ?? undefined,
  });
}

function createBatchProgressReporter(requestId, label) {
  let lastPercent = -1;
  return ({ processed, total }) => {
    if (!total) return;
    const percent = Math.max(0, Math.min(100, Math.round((processed / total) * 100)));
    if (percent === lastPercent) return;
    if (percent !== 100 && percent - lastPercent < 3) return;
    lastPercent = percent;
    postProgress(requestId, `${label} (${percent}%)`, {
      processed,
      total,
      percent,
    });
  };
}

function dotProduct(left, right) {
  let dot = 0;
  const len = Math.min(left.length, right.length);
  for (let i = 0; i < len; i += 1) {
    dot += left[i] * right[i];
  }
  return dot;
}

function centroidFromWorkVectors(workVectors) {
  if (!workVectors.length) {
    return new Float32Array(384);
  }
  const dim = workVectors[0]?.length || 384;
  const centroid = new Float32Array(dim);
  for (const vector of workVectors) {
    for (let i = 0; i < dim; i += 1) {
      centroid[i] += vector[i];
    }
  }
  const inv = 1 / workVectors.length;
  let normSq = 0;
  for (let i = 0; i < dim; i += 1) {
    centroid[i] *= inv;
    normSq += centroid[i] * centroid[i];
  }
  const norm = Math.sqrt(normSq);
  if (!norm) return centroid;
  for (let i = 0; i < dim; i += 1) {
    centroid[i] /= norm;
  }
  return centroid;
}

function computeSemanticScores(scheduleVectors, workVectors) {
  const centroid = centroidFromWorkVectors(workVectors);
  const scoresByIndex = new Array(scheduleVectors.length);

  for (let rowIndex = 0; rowIndex < scheduleVectors.length; rowIndex += 1) {
    const scheduleVec = scheduleVectors[rowIndex];
    const centroidSim = dotProduct(scheduleVec, centroid);

    let maxWorkSim = 0;
    for (const workVec of workVectors) {
      const sim = dotProduct(scheduleVec, workVec);
      if (sim > maxWorkSim) {
        maxWorkSim = sim;
      }
    }

    const semanticScore = 100 * (0.7 * centroidSim + 0.3 * maxWorkSim);
    scoresByIndex[rowIndex] = {
      semanticScore,
      centroidSim,
      maxWorkSim,
    };
  }

  return scoresByIndex;
}

function normalizeTopN(value, maxRows) {
  if (Number.isFinite(value)) {
    return Math.max(1, Math.min(maxRows, Math.floor(value)));
  }
  return maxRows;
}

function rankRows(rows, topN) {
  rows.sort((a, b) => b.relevance_score - a.relevance_score);
  rows.forEach((row, idx) => {
    row.relevance_rank = idx + 1;
  });
  return rows.slice(0, normalizeTopN(topN, rows.length));
}

async function loadPrecomputedScheduleVectors(requestId) {
  postProgress(requestId, 'Loading precomputed semantic schedule embeddings...');
  const [metaResp, binResp] = await Promise.all([
    fetch(SEMANTIC_SCHEDULE_EMBEDDINGS_META_URL),
    fetch(SEMANTIC_SCHEDULE_EMBEDDINGS_BIN_URL),
  ]);
  if (!metaResp.ok) {
    throw new Error(`Missing precomputed semantic metadata (${metaResp.status}).`);
  }
  if (!binResp.ok) {
    throw new Error(`Missing precomputed semantic vectors (${binResp.status}).`);
  }

  const meta = await metaResp.json();
  const expectedRows = scheduleIndex?.rows?.length || 0;
  const rowCount = Number(meta?.row_count);
  const dimension = Number(meta?.dimension);
  if (!Number.isInteger(rowCount) || !Number.isInteger(dimension) || rowCount <= 0 || dimension <= 0) {
    throw new Error('Invalid precomputed semantic metadata.');
  }
  if (rowCount !== expectedRows) {
    throw new Error(`Precomputed semantic row_count mismatch (${rowCount} vs ${expectedRows}).`);
  }

  const arrayBuffer = await binResp.arrayBuffer();
  if (arrayBuffer.byteLength % 4 !== 0) {
    throw new Error('Invalid precomputed semantic vector binary format.');
  }
  const flat = new Float32Array(arrayBuffer);
  if (flat.length !== rowCount * dimension) {
    throw new Error(
      `Precomputed semantic vector length mismatch (${flat.length} vs expected ${rowCount * dimension}).`
    );
  }

  const vectors = new Array(rowCount);
  for (let idx = 0; idx < rowCount; idx += 1) {
    const start = idx * dimension;
    vectors[idx] = flat.subarray(start, start + dimension);
  }

  return {
    vectors,
    model: {
      modelId: meta.model_id || SEMANTIC_MODEL_ID,
      dtype: meta.dtype || SEMANTIC_MODEL_DTYPE,
      source: 'precomputed',
      generated_at: meta.generated_at || null,
    },
  };
}

async function ensureSemanticScheduleVectors(requestId) {
  const rowCount = scheduleIndex?.rows?.length || 0;
  if (semanticScheduleCache && semanticScheduleCache.rowCount === rowCount) {
    return semanticScheduleCache;
  }

  try {
    const precomputed = await loadPrecomputedScheduleVectors(requestId);
    semanticScheduleCache = {
      rowCount,
      vectors: precomputed.vectors,
      model: precomputed.model,
    };
    postProgress(requestId, 'Loaded precomputed semantic schedule index.', {
      stage: 'semantic_schedule_precomputed',
    });
    return semanticScheduleCache;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    postProgress(requestId, 'Precomputed semantic index unavailable. Building CHI semantic index locally...', {
      stage: 'semantic_schedule_fallback',
      reason: message,
    });
  }

  postProgress(requestId, 'Loading local semantic model...');
  const model = await preloadSemanticModel({
    ...SEMANTIC_OPTIONS,
    progressCallback: (info) => relayModelLoadProgress(requestId, info),
  });
  const scheduleTexts = buildScheduleSemanticTexts(scheduleIndex.rows || []);
  postProgress(requestId, 'Building semantic vectors for CHI schedule...');

  const { vectors, model: embedModel } = await embedTexts(scheduleTexts, {
    ...SEMANTIC_OPTIONS,
    onBatch: createBatchProgressReporter(requestId, 'Building local CHI semantic index'),
  });

  semanticScheduleCache = {
    rowCount,
    vectors,
    model: embedModel || model?.model || null,
  };
  postProgress(requestId, 'Semantic schedule index ready.');
  return semanticScheduleCache;
}

function buildCombinedRows(tfidfResult, semanticScoresByIndex, matchingMode, topN) {
  const baseRows = Array.isArray(tfidfResult.rows_all) ? tfidfResult.rows_all : [];
  const rows = baseRows.map((row) => {
    const semantic = semanticScoresByIndex[row.row_index] || {
      semanticScore: 0,
      centroidSim: 0,
      maxWorkSim: 0,
    };
    const tfidfScore = row.relevance_score_tfidf ?? row.relevance_score ?? 0;
    const semanticScore = semantic.semanticScore;

    let finalScore = tfidfScore;
    if (matchingMode === 'semantic') {
      finalScore = semanticScore;
    } else if (matchingMode === 'hybrid') {
      finalScore = HYBRID_SEMANTIC_WEIGHT * semanticScore + (1 - HYBRID_SEMANTIC_WEIGHT) * tfidfScore;
    }

    return {
      ...row,
      relevance_mode: matchingMode,
      relevance_score_tfidf: tfidfScore,
      relevance_score_tfidf_pretty: tfidfScore.toFixed(2),
      relevance_score_semantic: semanticScore,
      relevance_score_semantic_pretty: semanticScore.toFixed(2),
      relevance_semantic_centroid_sim: semantic.centroidSim,
      relevance_semantic_max_work_sim: semantic.maxWorkSim,
      relevance_score: finalScore,
      relevance_score_pretty: finalScore.toFixed(2),
    };
  });

  return {
    totalMatches: rows.length,
    rows: rankRows(rows, topN),
    keywords: tfidfResult.keywords || [],
    workSummaries: tfidfResult.workSummaries || [],
  };
}

self.onmessage = async (event) => {
  const { type, requestId } = event.data || {};

  if (type === 'init') {
    scheduleIndex = event.data.scheduleIndex;
    semanticScheduleCache = null;
    self.postMessage({ type: 'ready', rowCount: scheduleIndex?.row_count || 0 });
    return;
  }

  if (type === 'run') {
    if (!scheduleIndex) {
      self.postMessage({
        type: 'error',
        requestId,
        error: 'Worker not initialized with schedule index.',
      });
      return;
    }

    try {
      const started = performance.now();
      const matchingMode = normalizeMatchingMode(event.data.matchingMode);

      const tfidfResult = scoreSchedule({
        worksTexts: event.data.worksTexts || [],
        workNames: event.data.workNames || [],
        topN: event.data.topN,
        customKeywords: event.data.customKeywords || [],
        scheduleIndex,
      });

      let result = {
        ...tfidfResult,
      };
      delete result.rows_all;
      if (matchingMode === 'semantic' || matchingMode === 'hybrid') {
        const semanticSchedule = await ensureSemanticScheduleVectors(requestId);
        const workChunkPlan = buildWorkChunkPlan(event.data.worksTexts || [], event.data.customKeywords || []);

        postProgress(requestId, 'Embedding uploaded papers...');
        const { vectors: workChunkVectors, model } = await embedTexts(workChunkPlan.chunks, {
          ...SEMANTIC_OPTIONS,
          onBatch: createBatchProgressReporter(requestId, 'Embedding uploaded papers'),
        });

        const workVectors = reduceChunkVectorsToWorkVectors(workChunkVectors, workChunkPlan.spans);
        const semanticScoresByIndex = computeSemanticScores(semanticSchedule.vectors, workVectors);
        result = buildCombinedRows(tfidfResult, semanticScoresByIndex, matchingMode, event.data.topN);
        result.semantic_model = model || semanticSchedule.model || null;
      }

      result.matching_mode = matchingMode;
      const elapsedMs = performance.now() - started;
      self.postMessage({ type: 'result', requestId, result, elapsedMs });
    } catch (error) {
      self.postMessage({
        type: 'error',
        requestId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  if (type === 'prefetch-model') {
    try {
      const started = performance.now();
      postProgress(requestId, 'Preparing semantic model download...', { stage: 'model_download' });
      const loaded = await preloadSemanticModel({
        ...SEMANTIC_PREFETCH_OPTIONS,
        progressCallback: (info) => relayModelLoadProgress(requestId, info),
      });
      const elapsedMs = performance.now() - started;
      self.postMessage({
        type: 'result',
        requestId,
        command: 'prefetch-model',
        result: {
          ok: true,
          semantic_model: loaded.model || null,
        },
        elapsedMs,
      });
    } catch (error) {
      self.postMessage({
        type: 'error',
        requestId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  if (type === 'probe-model') {
    try {
      const loaded = await preloadSemanticModel({
        ...SEMANTIC_OPTIONS,
      });
      self.postMessage({
        type: 'result',
        requestId,
        command: 'probe-model',
        result: {
          ok: true,
          available: true,
          semantic_model: loaded.model || null,
        },
      });
    } catch (error) {
      self.postMessage({
        type: 'result',
        requestId,
        command: 'probe-model',
        result: {
          ok: true,
          available: false,
          error: error instanceof Error ? error.message : String(error),
        },
      });
    }
  }

};
