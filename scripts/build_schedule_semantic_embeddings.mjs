import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { env, pipeline } from '@huggingface/transformers';
import { buildScheduleSemanticTexts } from '../src/shared/semantic.js';
import {
  SEMANTIC_LOCAL_MODEL_ROOT,
  SEMANTIC_MODEL_DTYPE,
  SEMANTIC_MODEL_ID,
  SEMANTIC_MODEL_SUBFOLDER,
} from '../src/shared/semantic-config.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');

const scheduleIndexPath = path.join(appRoot, 'public', 'data', 'schedule_index.json');
const outBinPath = path.join(appRoot, 'public', 'data', 'schedule_semantic_embeddings_q4.bin');
const outMetaPath = path.join(appRoot, 'public', 'data', 'schedule_semantic_embeddings_q4.json');

const BATCH_SIZE = 48;

function resolveLocalModelRoot() {
  const trimmed = String(SEMANTIC_LOCAL_MODEL_ROOT || '/models').replace(/^\/+/, '');
  return path.resolve(appRoot, 'public', trimmed);
}

function assertFileExists(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found at ${filePath}`);
  }
}

function rowsFromTensor(output) {
  const data = output?.data;
  const dims = output?.dims;
  if (!data || !Array.isArray(dims) || dims.length < 2) {
    throw new Error('Unexpected embedding tensor output.');
  }
  const count = dims[0];
  const dim = dims[dims.length - 1];
  const vectors = new Array(count);
  for (let i = 0; i < count; i += 1) {
    const start = i * dim;
    const end = start + dim;
    const vec = new Float32Array(dim);
    vec.set(data.subarray(start, end));
    vectors[i] = vec;
  }
  return { vectors, dim };
}

async function main() {
  assertFileExists(scheduleIndexPath, 'schedule_index.json');
  const scheduleIndex = JSON.parse(fs.readFileSync(scheduleIndexPath, 'utf8'));
  const rows = Array.isArray(scheduleIndex.rows) ? scheduleIndex.rows : [];
  if (!rows.length) {
    throw new Error('No schedule rows found.');
  }

  const localModelRoot = resolveLocalModelRoot();
  assertFileExists(path.join(localModelRoot, SEMANTIC_MODEL_ID, 'config.json'), 'Local semantic model');

  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = localModelRoot;

  console.log(`Loading embedding model '${SEMANTIC_MODEL_ID}' from ${localModelRoot}...`);
  const extractor = await pipeline('feature-extraction', SEMANTIC_MODEL_ID, {
    local_files_only: true,
    dtype: SEMANTIC_MODEL_DTYPE,
    subfolder: SEMANTIC_MODEL_SUBFOLDER,
  });

  const texts = buildScheduleSemanticTexts(rows);
  console.log(`Embedding ${texts.length} schedule rows in batches of ${BATCH_SIZE}...`);

  const allVectors = [];
  let dimension = 0;
  for (let cursor = 0; cursor < texts.length; cursor += BATCH_SIZE) {
    const batch = texts.slice(cursor, cursor + BATCH_SIZE);
    const output = await extractor(batch, { pooling: 'mean', normalize: true });
    const { vectors, dim } = rowsFromTensor(output);
    if (!dimension) {
      dimension = dim;
    } else if (dimension !== dim) {
      throw new Error(`Embedding dimension mismatch (${dimension} vs ${dim}).`);
    }
    for (const vec of vectors) {
      allVectors.push(vec);
    }
    const processed = Math.min(cursor + batch.length, texts.length);
    const percent = Math.round((processed / texts.length) * 100);
    console.log(`Processed ${processed}/${texts.length} (${percent}%)`);
  }

  const flat = new Float32Array(allVectors.length * dimension);
  for (let i = 0; i < allVectors.length; i += 1) {
    flat.set(allVectors[i], i * dimension);
  }

  fs.mkdirSync(path.dirname(outBinPath), { recursive: true });
  fs.writeFileSync(outBinPath, Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength));

  const metadata = {
    version: 1,
    model_id: SEMANTIC_MODEL_ID,
    dtype: SEMANTIC_MODEL_DTYPE,
    row_count: allVectors.length,
    dimension,
    format: 'float32le',
    generated_at: new Date().toISOString(),
    source_schedule_version: scheduleIndex.version || null,
  };
  fs.writeFileSync(outMetaPath, `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');

  const mb = (flat.byteLength / (1024 * 1024)).toFixed(2);
  console.log(`Wrote embeddings: ${outBinPath} (${mb} MB)`);
  console.log(`Wrote metadata: ${outMetaPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
