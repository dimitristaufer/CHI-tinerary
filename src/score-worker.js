import { scoreSchedule } from './shared/scoring.js';

let scheduleIndex = null;

self.onmessage = (event) => {
  const { type, requestId } = event.data || {};

  if (type === 'init') {
    scheduleIndex = event.data.scheduleIndex;
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
      const result = scoreSchedule({
        worksTexts: event.data.worksTexts || [],
        workNames: event.data.workNames || [],
        topN: event.data.topN,
        customKeywords: event.data.customKeywords || [],
        scheduleIndex,
      });
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
};
