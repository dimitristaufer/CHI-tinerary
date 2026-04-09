import './styles.css';
import * as pdfjsLib from 'pdfjs-dist/build/pdf.mjs';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const MAX_PDFS = 10;
const EXTRACTION_CONCURRENCY = 2;
const DEFAULT_VISIBLE_RESULTS = 20;
const LOAD_MORE_STEP = 20;
const KEYWORD_DEBOUNCE_MS = 350;
const ICS_PROD_ID = '-//CHI-tinerary//EN';
const FALLBACK_EVENT_DURATION_MS = 30 * 60 * 1000;
const CONFERENCE_TIMEZONE = 'Europe/Madrid';
const SCHEDULE_WALL_CLOCK_TIMEZONE = 'UTC';
const DAY_PILL_CLASS_MAP = {
  monday: 'day-pill-monday',
  tuesday: 'day-pill-tuesday',
  wednesday: 'day-pill-wednesday',
  thursday: 'day-pill-thursday',
  friday: 'day-pill-friday',
  saturday: 'day-pill-saturday',
  sunday: 'day-pill-sunday',
};
const startDateFormatter = new Intl.DateTimeFormat(undefined, {
  weekday: 'short',
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  timeZone: SCHEDULE_WALL_CLOCK_TIMEZONE,
});
const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: 'numeric',
  minute: '2-digit',
  timeZone: SCHEDULE_WALL_CLOCK_TIMEZONE,
});

const form = document.getElementById('analyze-form');
const fileInput = document.getElementById('pdfs');
const fileSelection = document.getElementById('file-selection');
const keywordInput = document.getElementById('custom-keywords');
const boostedKeywordsList = document.getElementById('boosted-keywords-list');
const runBtn = document.getElementById('run-btn');
const loadMoreBtn = document.getElementById('load-more-btn');
const statusLine = document.getElementById('status-line');
const statusList = document.getElementById('status-list');
const resultsPanel = document.getElementById('results-panel');
const resultsActions = document.getElementById('results-actions');

const scoreWorker = new Worker(new URL('./score-worker.js', import.meta.url), { type: 'module' });

let requestSeq = 0;
const pendingRequests = new Map();

let scheduleRowCount = 0;
let extractedContext = null;
let latestRows = [];
let visibleResults = DEFAULT_VISIBLE_RESULTS;
let scoringRunSeq = 0;
let keywordDebounceHandle = null;
let isExtracting = false;

scoreWorker.onmessage = (event) => {
  const payload = event.data || {};

  if (payload.type === 'ready') {
    setStatus(`Schedule index ready (${payload.rowCount.toLocaleString()} rows).`);
    return;
  }

  if (payload.type === 'result' || payload.type === 'error') {
    let requestId = payload.requestId;
    if (requestId == null && pendingRequests.size === 1) {
      requestId = pendingRequests.keys().next().value;
    }
    const pending = pendingRequests.get(requestId);
    if (!pending) return;
    pendingRequests.delete(requestId);

    if (payload.type === 'error') {
      pending.reject(new Error(payload.error || 'Unknown worker error'));
    } else {
      pending.resolve(payload);
    }
  }
};

function setStatus(text) {
  statusLine.textContent = text;
}

function appendStatusItem(text) {
  const li = document.createElement('li');
  li.textContent = text;
  statusList.appendChild(li);
  return li;
}

function clearResults() {
  resultsPanel.innerHTML = '';
  resultsActions.classList.add('hidden');
  latestRows = [];
  visibleResults = DEFAULT_VISIBLE_RESULTS;
}

function clearStatusItems() {
  statusList.innerHTML = '';
}

async function loadScheduleIndex() {
  const resp = await fetch('./data/schedule_index.json');
  if (!resp.ok) {
    throw new Error(`Could not load schedule index (${resp.status}).`);
  }
  return resp.json();
}

function validateUploads(files) {
  if (!files.length) throw new Error('Upload at least one PDF.');
  if (files.length > MAX_PDFS) throw new Error(`Upload at most ${MAX_PDFS} PDFs.`);

  for (const file of files) {
    const isPdfByType = file.type === 'application/pdf';
    const isPdfByName = file.name.toLowerCase().endsWith('.pdf');
    if (!isPdfByType && !isPdfByName) {
      throw new Error(`File '${file.name}' is not a PDF.`);
    }
  }
}

function parseCustomKeywords(raw) {
  const seen = new Set();
  const keywords = [];
  const parts = String(raw || '')
    .split(/[\n,]/g)
    .map((part) => part.trim())
    .filter(Boolean);

  for (const entry of parts) {
    const key = entry.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    keywords.push(entry);
  }

  return keywords.slice(0, 50);
}

function pageTextFromItems(items) {
  const arranged = [];
  for (const item of items) {
    if (!item || typeof item.str !== 'string') continue;
    const text = item.str.trim();
    if (!text) continue;
    arranged.push({
      str: text,
      x: Array.isArray(item.transform) ? item.transform[4] || 0 : 0,
      y: Array.isArray(item.transform) ? item.transform[5] || 0 : 0,
    });
  }

  arranged.sort((a, b) => {
    const yDiff = b.y - a.y;
    if (Math.abs(yDiff) > 2) return yDiff;
    return a.x - b.x;
  });

  return arranged.map((entry) => entry.str).join(' ');
}

async function extractPdfText(file, onProgress) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const loadingTask = pdfjsLib.getDocument({ data: bytes });
  const doc = await loadingTask.promise;

  const chunks = [];
  for (let pageNo = 1; pageNo <= doc.numPages; pageNo += 1) {
    const page = await doc.getPage(pageNo);
    const content = await page.getTextContent({ normalizeWhitespace: true });
    chunks.push(pageTextFromItems(content.items));
    onProgress(pageNo, doc.numPages);
    page.cleanup();
  }

  await doc.cleanup();
  await doc.destroy();
  return chunks.join('\n');
}

async function extractAllPdfs(files) {
  const results = new Array(files.length);
  let cursor = 0;
  let active = 0;

  return new Promise((resolve, reject) => {
    const launch = () => {
      if (cursor >= files.length && active === 0) {
        resolve(results);
        return;
      }

      while (active < EXTRACTION_CONCURRENCY && cursor < files.length) {
        const idx = cursor;
        const file = files[cursor];
        cursor += 1;
        active += 1;

        const statusItem = appendStatusItem(`Extracting ${file.name} (0%)`);

        extractPdfText(file, (pageNo, totalPages) => {
          const pct = Math.round((pageNo / totalPages) * 100);
          statusItem.textContent = `Extracting ${file.name} (${pct}%)`;
        })
          .then((text) => {
            results[idx] = text;
            statusItem.textContent = `Extracted ${file.name} (${text.length.toLocaleString()} chars)`;
          })
          .catch((err) => {
            statusItem.textContent = `Failed ${file.name}: ${err.message}`;
            reject(err);
          })
          .finally(() => {
            active -= 1;
            launch();
          });
      }
    };

    launch();
  });
}

function runWorkerScore({ worksTexts, workNames, customKeywords }) {
  const requestId = ++requestSeq;
  const timeoutMs = 120000;
  const topN = scheduleRowCount || 10000;

  return new Promise((resolve, reject) => {
    const timeoutHandle = window.setTimeout(() => {
      if (!pendingRequests.has(requestId)) return;
      pendingRequests.delete(requestId);
      reject(new Error(`Scoring timed out after ${Math.round(timeoutMs / 1000)}s.`));
    }, timeoutMs);

    pendingRequests.set(requestId, {
      resolve: (value) => {
        window.clearTimeout(timeoutHandle);
        resolve(value);
      },
      reject: (error) => {
        window.clearTimeout(timeoutHandle);
        reject(error);
      },
    });

    scoreWorker.postMessage({
      type: 'run',
      requestId,
      worksTexts,
      workNames,
      customKeywords,
      topN,
    });
  });
}

function asEpochMs(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatStartDate(epochMs) {
  if (epochMs == null) return 'n/a';
  return startDateFormatter.format(new Date(epochMs));
}

function formatTimeRange(startMs, endMs) {
  if (startMs == null) return 'n/a';
  const startText = timeFormatter.format(new Date(startMs));
  if (endMs == null || endMs <= startMs) return startText;
  const endText = timeFormatter.format(new Date(endMs));
  return `${startText} - ${endText}`;
}

function dayPillClass(day) {
  const normalized = String(day || '')
    .trim()
    .toLowerCase();
  return DAY_PILL_CLASS_MAP[normalized] || 'day-pill-other';
}

function pad2(value) {
  return String(value).padStart(2, '0');
}

function formatUtcIcsDateTime(epochMs) {
  const dt = new Date(epochMs);
  return `${dt.getUTCFullYear()}${pad2(dt.getUTCMonth() + 1)}${pad2(dt.getUTCDate())}T${pad2(dt.getUTCHours())}${pad2(dt.getUTCMinutes())}${pad2(dt.getUTCSeconds())}Z`;
}

function formatFloatingIcsDateTime(epochMs) {
  const dt = new Date(epochMs);
  return `${dt.getUTCFullYear()}${pad2(dt.getUTCMonth() + 1)}${pad2(dt.getUTCDate())}T${pad2(dt.getUTCHours())}${pad2(dt.getUTCMinutes())}${pad2(dt.getUTCSeconds())}`;
}

function escapeIcsText(value) {
  return String(value || '')
    .replaceAll('\\', '\\\\')
    .replaceAll('\r\n', '\n')
    .replaceAll('\n', '\\n')
    .replaceAll(';', '\\;')
    .replaceAll(',', '\\,');
}

function foldIcsLine(line) {
  const text = String(line || '');
  if (text.length <= 73) return text;

  let folded = text.slice(0, 73);
  let cursor = 73;
  while (cursor < text.length) {
    folded += `\r\n ${text.slice(cursor, cursor + 72)}`;
    cursor += 72;
  }
  return folded;
}

function slugifyForFilename(value) {
  return String(value || '')
    .normalize('NFKD')
    .replace(/[^\x00-\x7F]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 72);
}

function buildCalendarPayload(row, rowIndex) {
  const startMs = asEpochMs(row.start_date_unix_ms);
  const endMs = asEpochMs(row.end_date_unix_ms);
  if (startMs == null) {
    throw new Error('This session is missing a start time in the schedule data.');
  }

  const normalizedEndMs = endMs && endMs > startMs ? endMs : startMs + FALLBACK_EVENT_DURATION_MS;
  const summary = String(row.title || `CHI session #${row.relevance_rank || rowIndex + 1}`);
  const location = [row.room, row.building].filter(Boolean).join(', ');
  const descriptionParts = [];
  if (row.authors) descriptionParts.push(`Authors: ${row.authors}`);
  if (row.day) descriptionParts.push(`Day: ${row.day}`);
  if (row.session_type) descriptionParts.push(`Session type: ${row.session_type}`);
  if (row.abstract) descriptionParts.push(row.abstract);
  const description = descriptionParts.join('\n\n');

  const uidSlug = slugifyForFilename(`${summary}-${startMs}-${rowIndex + 1}`) || `chi-${rowIndex + 1}`;
  const dtStamp = formatUtcIcsDateTime(Date.now());
  const dtStart = formatFloatingIcsDateTime(startMs);
  const dtEnd = formatFloatingIcsDateTime(normalizedEndMs);

  const lines = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    `PRODID:${ICS_PROD_ID}`,
    'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH',
    'BEGIN:VEVENT',
    `UID:${uidSlug}@chi-relevance-client`,
    `DTSTAMP:${dtStamp}`,
    `DTSTART;TZID=${CONFERENCE_TIMEZONE}:${dtStart}`,
    `DTEND;TZID=${CONFERENCE_TIMEZONE}:${dtEnd}`,
    `SUMMARY:${escapeIcsText(summary)}`,
  ];

  if (location) {
    lines.push(`LOCATION:${escapeIcsText(location)}`);
  }
  if (description) {
    lines.push(`DESCRIPTION:${escapeIcsText(description)}`);
  }

  lines.push('END:VEVENT', 'END:VCALENDAR');

  const icsText = lines.map(foldIcsLine).join('\r\n');
  const fileBase = slugifyForFilename(summary) || `chi-session-${rowIndex + 1}`;

  return {
    icsText,
    fileName: `${fileBase}.ics`,
    summary,
  };
}

function updateFileSelectionText() {
  const files = Array.from(fileInput.files || []);
  if (!files.length) {
    fileSelection.textContent = 'No files selected yet.';
    return;
  }
  if (files.length === 1) {
    fileSelection.textContent = files[0].name;
    return;
  }
  fileSelection.textContent = `${files.length} files selected`;
}

function renderBoostedKeywords() {
  const keywords = parseCustomKeywords(keywordInput.value);
  boostedKeywordsList.innerHTML = '';

  if (!keywords.length) {
    const empty = document.createElement('p');
    empty.className = 'muted boosted-keywords-empty';
    empty.textContent = 'No boosted keywords yet.';
    boostedKeywordsList.appendChild(empty);
    return;
  }

  for (const keyword of keywords) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'boosted-keyword-chip';
    chip.dataset.keyword = keyword;
    chip.setAttribute('aria-label', `Remove boosted keyword: ${keyword}`);

    const text = document.createElement('span');
    text.textContent = keyword;

    const remove = document.createElement('span');
    remove.className = 'chip-remove';
    remove.setAttribute('aria-hidden', 'true');
    remove.textContent = 'x';

    chip.append(text, remove);
    boostedKeywordsList.appendChild(chip);
  }
}

async function openCalendarImport(icsText, fileName, summary) {
  const blob = new Blob([icsText], { type: 'text/calendar;charset=utf-8' });
  const isLikelyMobile = window.matchMedia('(pointer: coarse)').matches;
  if (isLikelyMobile && typeof File !== 'undefined' && navigator.share && navigator.canShare) {
    const shareFile = new File([blob], fileName, { type: 'text/calendar' });
    let canShareFiles = false;
    try {
      canShareFiles = navigator.canShare({ files: [shareFile] });
    } catch {
      canShareFiles = false;
    }
    if (canShareFiles) {
      try {
        await navigator.share({
          title: `Add to calendar: ${summary}`,
          files: [shareFile],
        });
        return 'shared';
      } catch (error) {
        if (error && error.name === 'AbortError') {
          return 'cancelled';
        }
      }
    }
  }

  const objectUrl = URL.createObjectURL(blob);
  const downloadLink = document.createElement('a');
  downloadLink.href = objectUrl;
  downloadLink.download = fileName;
  document.body.appendChild(downloadLink);
  downloadLink.click();
  downloadLink.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
  return 'downloaded';
}

function renderResults(rows, startIndex = 0) {
  resultsPanel.innerHTML = rows
    .map(
      (row, idx) => {
        const startMs = asEpochMs(row.start_date_unix_ms);
        const endMs = asEpochMs(row.end_date_unix_ms);
        const formattedStartDate = formatStartDate(startMs);
        const formattedTimeRange = formatTimeRange(startMs, endMs);
        const dayText = row.day || 'n/a';
        const dayClass = dayPillClass(dayText);
        return `
      <article class="result-card">
        <header>
          <p class="rank">#${row.relevance_rank}</p>
          <p class="score">${row.relevance_score_pretty}%</p>
        </header>
        <h2>${escapeHtml(row.title || 'Untitled')}</h2>
        <p class="authors">${escapeHtml(row.authors || 'No authors listed')}</p>
        <dl class="facts">
          <div><dt>Room</dt><dd>${escapeHtml(row.room || 'n/a')}</dd></div>
          <div><dt>Building</dt><dd>${escapeHtml(row.building || 'n/a')}</dd></div>
          <div><dt>Start</dt><dd>${escapeHtml(formattedStartDate)}</dd></div>
          <div><dt>Time</dt><dd>${escapeHtml(formattedTimeRange)}</dd></div>
          <div><dt>Day</dt><dd><span class="day-pill ${dayClass}">${escapeHtml(dayText)}</span></dd></div>
        </dl>
        <p class="abstract">${escapeHtml(row.abstract || 'No abstract provided.')}</p>
        <div class="card-actions">
          <button type="button" class="calendar-btn" data-row-index="${startIndex + idx}">Add to calendar (.ics)</button>
        </div>
      </article>
    `
      }
    )
    .join('');
}

function updateLoadMoreControls() {
  const remaining = latestRows.length - visibleResults;
  if (remaining > 0) {
    const next = Math.min(LOAD_MORE_STEP, remaining);
    loadMoreBtn.textContent = `Load more (${next} more)`;
    resultsActions.classList.remove('hidden');
  } else {
    resultsActions.classList.add('hidden');
  }
}

function renderResultsPage() {
  renderResults(latestRows.slice(0, visibleResults), 0);
  updateLoadMoreControls();
}

async function runScoringFromExtracted(statusText, skipDoneStatus = false) {
  if (!extractedContext) {
    throw new Error('Upload PDFs and run analysis first.');
  }

  const runId = ++scoringRunSeq;
  const customKeywords = parseCustomKeywords(keywordInput.value);
  if (statusText) {
    setStatus(statusText);
  }

  const { result } = await runWorkerScore({
    worksTexts: extractedContext.worksTexts,
    workNames: extractedContext.workNames,
    customKeywords,
  });

  if (runId !== scoringRunSeq) {
    return;
  }

  latestRows = result.rows || [];
  visibleResults = Math.min(DEFAULT_VISIBLE_RESULTS, latestRows.length || DEFAULT_VISIBLE_RESULTS);

  renderResultsPage();

  if (!skipDoneStatus) {
    const shown = Math.min(visibleResults, latestRows.length);
    setStatus(`Done. Showing ${shown.toLocaleString()} of ${latestRows.length.toLocaleString()} results.`);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

resultsPanel.addEventListener('click', async (event) => {
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;

  const calendarBtn = target.closest('.calendar-btn');
  if (!calendarBtn) return;

  const rowIndex = Number(calendarBtn.dataset.rowIndex);
  if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= latestRows.length) {
    return;
  }

  const row = latestRows[rowIndex];
  calendarBtn.disabled = true;

  try {
    const { icsText, fileName, summary } = buildCalendarPayload(row, rowIndex);
    const result = await openCalendarImport(icsText, fileName, summary);
    if (result === 'cancelled') return;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    appendStatusItem(`Calendar error: ${message}`);
  } finally {
    calendarBtn.disabled = false;
  }
});

boostedKeywordsList.addEventListener('click', (event) => {
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;
  const chip = target.closest('.boosted-keyword-chip');
  if (!chip) return;

  const keyword = String(chip.dataset.keyword || '').trim();
  if (!keyword) return;

  const keywordLower = keyword.toLowerCase();
  const nextKeywords = parseCustomKeywords(keywordInput.value).filter((entry) => entry.toLowerCase() !== keywordLower);
  keywordInput.value = nextKeywords.join(', ');
  keywordInput.dispatchEvent(new Event('input', { bubbles: true }));
});

fileInput.addEventListener('change', () => {
  updateFileSelectionText();
});

loadMoreBtn.addEventListener('click', () => {
  visibleResults = Math.min(latestRows.length, visibleResults + LOAD_MORE_STEP);
  renderResultsPage();
});

keywordInput.addEventListener('input', () => {
  renderBoostedKeywords();
  if (keywordDebounceHandle) {
    window.clearTimeout(keywordDebounceHandle);
  }

  keywordDebounceHandle = window.setTimeout(async () => {
    if (!extractedContext || isExtracting) return;

    try {
      await runScoringFromExtracted(null, true);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(message);
      appendStatusItem(`Error: ${message}`);
    }
  }, KEYWORD_DEBOUNCE_MS);
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearStatusItems();
  clearResults();

  try {
    const files = Array.from(fileInput.files || []);
    validateUploads(files);

    runBtn.disabled = true;
    isExtracting = true;
    extractedContext = null;

    setStatus('Extracting text from PDFs with pdf.js...');
    const worksTexts = await extractAllPdfs(files);

    extractedContext = {
      worksTexts,
      workNames: files.map((f) => f.name),
    };

    isExtracting = false;
    await runScoringFromExtracted('Computing TF-IDF relevance in scoring worker...');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    appendStatusItem(`Error: ${message}`);
  } finally {
    isExtracting = false;
    runBtn.disabled = false;
  }
});

(async () => {
  renderBoostedKeywords();
  updateFileSelectionText();
  try {
    const scheduleIndex = await loadScheduleIndex();
    scheduleRowCount = scheduleIndex.row_count || 0;
    scoreWorker.postMessage({ type: 'init', scheduleIndex });
    setStatus(`Loaded schedule index (${scheduleRowCount.toLocaleString()} rows). Ready.`);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  }
})();
