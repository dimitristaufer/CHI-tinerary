import { counterFromText } from './text.js';

function buildIdf(docFreq, numDocs) {
  const idf = Object.create(null);
  for (const [term, freq] of Object.entries(docFreq)) {
    idf[term] = Math.log((1 + numDocs) / (1 + freq)) + 1;
  }
  return idf;
}

function vectorizeCounter(counter, idf) {
  const vec = Object.create(null);
  let normSq = 0;
  for (const [term, count] of Object.entries(counter)) {
    const idfWeight = idf[term] || 0;
    const weight = (1 + Math.log(count)) * idfWeight;
    if (weight > 0) {
      vec[term] = weight;
      normSq += weight * weight;
    }
  }
  return { vec, norm: Math.sqrt(normSq) };
}

function vectorizePackedTokenPairs(tokenPairs, idf) {
  const vec = Object.create(null);
  let normSq = 0;
  for (const [term, count] of tokenPairs) {
    const idfWeight = idf[term] || 0;
    const weight = (1 + Math.log(count)) * idfWeight;
    if (weight > 0) {
      vec[term] = weight;
      normSq += weight * weight;
    }
  }
  return { vec, norm: Math.sqrt(normSq) };
}

function cosineSimilarity(leftVec, leftNorm, rightVec, rightNorm) {
  if (!leftNorm || !rightNorm) return 0;
  let a = leftVec;
  let b = rightVec;
  if (Object.keys(a).length > Object.keys(b).length) {
    a = rightVec;
    b = leftVec;
  }
  let dot = 0;
  for (const [term, value] of Object.entries(a)) {
    dot += value * (b[term] || 0);
  }
  return dot / (leftNorm * rightNorm);
}

function topKeywords(workCounters, idf, topN = 15) {
  const accum = Object.create(null);
  for (const counter of workCounters) {
    for (const [term, count] of Object.entries(counter)) {
      accum[term] = (accum[term] || 0) + (1 + Math.log(count)) * (idf[term] || 0);
    }
  }
  return Object.entries(accum)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([term, weight]) => ({ term, weight }));
}

export function scoreSchedule({
  worksTexts,
  scheduleIndex,
  topN = 20,
  workNames = [],
  customKeywords = [],
}) {
  if (!Array.isArray(worksTexts) || worksTexts.length === 0) {
    throw new Error('No work texts provided.');
  }
  if (!scheduleIndex || !Array.isArray(scheduleIndex.rows) || !scheduleIndex.schedule_doc_freq) {
    throw new Error('Invalid schedule index payload.');
  }

  const scheduleRows = scheduleIndex.rows;
  const workCounters = worksTexts.map((text) => counterFromText(text));

  if (Array.isArray(customKeywords) && customKeywords.length > 0) {
    const keywordCounter = counterFromText(customKeywords.join(' '));
    const keywordBoost = 6;
    for (const workCounter of workCounters) {
      for (const [term, count] of Object.entries(keywordCounter)) {
        workCounter[term] = (workCounter[term] || 0) + count * keywordBoost;
      }
    }
  }

  const docFreq = { ...scheduleIndex.schedule_doc_freq };
  for (const workCounter of workCounters) {
    for (const term of Object.keys(workCounter)) {
      docFreq[term] = (docFreq[term] || 0) + 1;
    }
  }

  const numDocs = scheduleRows.length + workCounters.length;
  const idf = buildIdf(docFreq, numDocs);

  const workVectors = workCounters.map((counter) => vectorizeCounter(counter, idf));

  const centroid = Object.create(null);
  for (const { vec } of workVectors) {
    for (const [term, weight] of Object.entries(vec)) {
      centroid[term] = (centroid[term] || 0) + weight;
    }
  }
  const workCount = workVectors.length || 1;
  let centroidNormSq = 0;
  for (const term of Object.keys(centroid)) {
    centroid[term] /= workCount;
    centroidNormSq += centroid[term] * centroid[term];
  }
  const centroidNorm = Math.sqrt(centroidNormSq);

  const rows = [];
  for (const [rowIndex, row] of scheduleRows.entries()) {
    const { vec, norm } = vectorizePackedTokenPairs(row.tokens || [], idf);

    const centroidSim = cosineSimilarity(vec, norm, centroid, centroidNorm);
    let maxWorkSim = 0;
    for (const workVec of workVectors) {
      const sim = cosineSimilarity(vec, norm, workVec.vec, workVec.norm);
      if (sim > maxWorkSim) {
        maxWorkSim = sim;
      }
    }

    const score = 100 * (0.7 * centroidSim + 0.3 * maxWorkSim);
    rows.push({
      row_index: rowIndex,
      title: row.title || '',
      authors: row.authors || '',
      abstract: row.abstract || '',
      room: row.room || '',
      building: row.building || '',
      day: row.day || '',
      session_type: row.session_type || '',
      start_date_unix_ms: row.start_date_unix_ms ?? null,
      end_date_unix_ms: row.end_date_unix_ms ?? null,
      relevance_mode: 'tfidf',
      relevance_score_tfidf: score,
      relevance_score_tfidf_pretty: score.toFixed(2),
      relevance_score: score,
      relevance_score_pretty: score.toFixed(2),
      relevance_centroid_sim: centroidSim,
      relevance_max_work_sim: maxWorkSim,
    });
  }

  const rowsAll = rows.map((row) => ({ ...row }));
  rows.sort((a, b) => b.relevance_score - a.relevance_score);
  rows.forEach((row, idx) => {
    row.relevance_rank = idx + 1;
  });

  const normalizedTopN = Number.isFinite(topN) ? Math.max(1, Math.floor(topN)) : rows.length;
  const resultRows = rows.slice(0, Math.min(normalizedTopN, rows.length));

  return {
    totalMatches: rows.length,
    rows_all: rowsAll,
    rows: resultRows,
    keywords: topKeywords(workCounters, idf, 15),
    workSummaries: worksTexts.map((text, idx) => ({
      name: workNames[idx] || `Work ${idx + 1}`,
      chars: text.length,
      token_count: Object.values(workCounters[idx]).reduce((acc, count) => acc + count, 0),
      unique_terms: Object.keys(workCounters[idx]).length,
    })),
  };
}
