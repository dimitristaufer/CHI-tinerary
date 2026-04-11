import './styles.css';
import * as pdfjsLib from 'pdfjs-dist/build/pdf.mjs';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';
import { semanticModelUrl } from './shared/semantic-config.js';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const MAX_PDFS = 10;
const MAX_PROFILE_ABSTRACTS = 20;
const MAX_PROFILE_DETAIL_FETCHES = 24;
const PROFILE_FETCH_TIMEOUT_MS = 15000;
const EXTRACTION_CONCURRENCY = 2;
const DEFAULT_VISIBLE_RESULTS = 20;
const LOAD_MORE_STEP = 20;
const KEYWORD_DEBOUNCE_MS = 350;
const CALENDAR_DOWNLOADS_STORAGE_KEY = 'chi_tinerary_downloaded_calendar_items_v1';
const MATCHING_MODES = new Set(['tfidf', 'semantic', 'hybrid']);
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
const profileUrlInput = document.getElementById('profile-url');
const fetchProfileBtn = document.getElementById('fetch-profile-btn');
const profileFetchStatus = document.getElementById('profile-fetch-status');
const profileAbstractsPanel = document.getElementById('profile-abstracts-panel');
const profileAbstractsList = document.getElementById('profile-abstracts-list');
const profileAbstractsCount = document.getElementById('profile-abstracts-count');
const selectAllAbstractsBtn = document.getElementById('select-all-abstracts-btn');
const clearAllAbstractsBtn = document.getElementById('clear-all-abstracts-btn');
const keywordInput = document.getElementById('custom-keywords');
const matchingModeInput = document.getElementById('matching-mode');
const downloadModelBtn = document.getElementById('download-model-btn');
const modelDownloadStatus = document.getElementById('model-download-status');
const modelDownloadProgressWrap = document.getElementById('model-download-progress-wrap');
const modelDownloadProgress = document.getElementById('model-download-progress');
const modelDownloadProgressText = document.getElementById('model-download-progress-text');
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
let latestMatchingMode = 'tfidf';
let fetchedProfileAbstracts = [];
const selectedProfileAbstractIds = new Set();
let modelDownloadRequestId = null;
let modelDownloadBusy = false;
let semanticModelAvailable = false;
let modelAvailabilityRefreshSeq = 0;
const semanticFallbackNotedRequests = new Set();
const downloadedCalendarItems = loadDownloadedCalendarItems();
const DEV_PROFILE_FETCH_PROXY_PATH = '/__fetch';
const rateLimitedProfileHosts = new Set();
const profileFetchCache = new Map();

scoreWorker.onmessage = (event) => {
  const payload = event.data || {};

  if (payload.type === 'ready') {
    setStatus(`Schedule index ready (${payload.rowCount.toLocaleString()} rows).`);
    return;
  }

  if (payload.type === 'progress') {
    const pending = pendingRequests.get(payload.requestId);
    if (pending && payload.message) {
      setStatus(payload.message);
    }

    if (payload.stage === 'semantic_schedule_fallback' && payload.requestId != null) {
      if (!semanticFallbackNotedRequests.has(payload.requestId)) {
        semanticFallbackNotedRequests.add(payload.requestId);
        const detail = payload.reason ? ` (${payload.reason})` : '';
        appendStatusItem(`Precomputed CHI semantic embeddings were unavailable; using local fallback${detail}`);
      }
    }

    if (payload.requestId === modelDownloadRequestId && payload.stage === 'model_download') {
      setModelDownloadUi({
        busy: true,
        statusText: payload.message || 'Downloading semantic model...',
        progressPercent: Number.isFinite(payload.percent) ? payload.percent : null,
        loadedBytes: payload.loaded,
        totalBytes: payload.total,
      });
    }
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

function validateUploads(files, { allowEmpty = false } = {}) {
  if (!files.length && !allowEmpty) throw new Error('Upload at least one PDF or select profile abstracts.');
  if (files.length > MAX_PDFS) throw new Error(`Upload at most ${MAX_PDFS} PDFs.`);

  for (const file of files) {
    const isPdfByType = file.type === 'application/pdf';
    const isPdfByName = file.name.toLowerCase().endsWith('.pdf');
    if (!isPdfByType && !isPdfByName) {
      throw new Error(`File '${file.name}' is not a PDF.`);
    }
  }
}

function normalizeProfileUrl(rawUrl) {
  const raw = String(rawUrl || '').trim();
  if (!raw) {
    throw new Error('Enter a Google Scholar or ResearchGate profile URL.');
  }

  const prefixed = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`;
  let parsed;
  try {
    parsed = new URL(prefixed);
  } catch {
    throw new Error('Profile URL is not valid.');
  }

  const host = parsed.hostname.toLowerCase();
  const isScholar = host.includes('scholar.google.');
  const isResearchGate = host === 'researchgate.net' || host.endsWith('.researchgate.net');

  if (!isScholar && !isResearchGate) {
    throw new Error('Only Google Scholar and ResearchGate profile URLs are supported.');
  }

  if (isScholar && !parsed.pathname.includes('/citations')) {
    throw new Error('Google Scholar URL must point to a profile page containing /citations.');
  }

  if (isResearchGate && parsed.pathname.includes('/publication/')) {
    throw new Error('Use a ResearchGate profile URL, not a publication URL.');
  }

  return parsed.toString();
}

function scholarDetailUrlRegex() {
  return /https?:\/\/scholar\.google\.[^\s"'<>)]*\/citations\?[^"'<>)]*view_op=view_citation[^"'<>)]*/gi;
}

function parseJsonLdAbstracts(doc) {
  if (!doc) return [];
  const results = [];
  const stack = [];
  for (const node of doc.querySelectorAll('script[type="application/ld+json"]')) {
    const raw = node.textContent || '';
    if (!raw.trim()) continue;
    try {
      stack.push(JSON.parse(raw));
    } catch {
      // Ignore malformed script tags.
    }
  }

  while (stack.length) {
    const current = stack.pop();
    if (!current) continue;
    if (Array.isArray(current)) {
      for (const item of current) stack.push(item);
      continue;
    }
    if (typeof current === 'object') {
      for (const [key, value] of Object.entries(current)) {
        if ((key.toLowerCase() === 'abstract' || key.toLowerCase() === 'description') && typeof value === 'string') {
          results.push(value);
        } else if (value && typeof value === 'object') {
          stack.push(value);
        }
      }
    }
  }

  return results;
}

function compactWhitespace(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractPublicationYear(value) {
  const text = String(value || '');
  const match = text.match(/\b(18|19|20)\d{2}\b/);
  if (!match) return null;
  const year = Number(match[0]);
  if (!Number.isInteger(year)) return null;
  return year;
}

function sortByPublicationYearDesc(items) {
  return items
    .map((item, idx) => ({ item, idx }))
    .sort((left, right) => {
      const leftYear = Number(left.item?.publicationYear);
      const rightYear = Number(right.item?.publicationYear);
      const leftHasYear = Number.isFinite(leftYear);
      const rightHasYear = Number.isFinite(rightYear);
      if (leftHasYear && rightHasYear && leftYear !== rightYear) {
        return rightYear - leftYear;
      }
      if (leftHasYear && !rightHasYear) return -1;
      if (!leftHasYear && rightHasYear) return 1;
      return left.idx - right.idx;
    })
    .map((entry) => entry.item);
}

function looksLikeRealAbstract(text) {
  const normalized = compactWhitespace(text);
  if (normalized.length < 45) return false;
  const lower = normalized.toLowerCase();
  const blockedFragments = [
    'request full-text',
    'discover researchgate',
    'researchgate has not been able to resolve',
    'no abstract available',
  ];
  return !blockedFragments.some((fragment) => lower.includes(fragment));
}

function slugToName(input) {
  return decodeURIComponent(String(input || ''))
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function parseResearchGateProfileName(profileUrl) {
  const parsed = new URL(profileUrl);
  const parts = parsed.pathname.split('/').filter(Boolean);
  const profileIndex = parts.findIndex((part) => part.toLowerCase() === 'profile');
  if (profileIndex >= 0 && parts[profileIndex + 1]) {
    return slugToName(parts[profileIndex + 1]);
  }
  if (parts[0]) {
    return slugToName(parts[0]);
  }
  return '';
}

function reconstructOpenAlexAbstract(invertedIndex) {
  if (!invertedIndex || typeof invertedIndex !== 'object') return '';
  const positioned = [];
  for (const [token, indexes] of Object.entries(invertedIndex)) {
    if (!Array.isArray(indexes)) continue;
    for (const index of indexes) {
      if (!Number.isInteger(index)) continue;
      positioned.push({ index, token });
    }
  }
  if (!positioned.length) return '';
  positioned.sort((a, b) => a.index - b.index);
  return compactWhitespace(
    positioned
      .map((entry) => entry.token)
      .join(' ')
      .replace(/\s+([,.;:!?])/g, '$1')
  );
}

function authorMatchScore(author, targetName) {
  const normalizedTarget = compactWhitespace(targetName).toLowerCase();
  const normalizedAuthor = compactWhitespace(author?.display_name || '').toLowerCase();
  if (!normalizedTarget || !normalizedAuthor) return -1;
  const tokens = normalizedTarget.split(' ').filter(Boolean);
  let score = 0;
  for (const token of tokens) {
    if (normalizedAuthor.includes(token)) score += 3;
  }
  if (normalizedAuthor === normalizedTarget) score += 8;
  if (String(author?.orcid || '').length > 0) score += 1;
  if (Number.isFinite(author?.works_count)) score += Math.min(4, Math.log10((author.works_count || 0) + 1));
  if (Number.isFinite(author?.relevance_score)) score += Math.min(4, Math.log10((author.relevance_score || 0) + 1));
  return score;
}

async function fetchOpenAlexJson(url) {
  const response = await fetch(url, { method: 'GET' });
  if (!response.ok) {
    throw new Error(`OpenAlex request failed (${response.status})`);
  }
  return response.json();
}

async function fetchResearchGateViaOpenAlex(profileUrl) {
  const profileName = parseResearchGateProfileName(profileUrl);
  if (!profileName || profileName.length < 3) {
    return [];
  }

  const authorSearchUrl = `https://api.openalex.org/authors?search=${encodeURIComponent(profileName)}&per-page=6`;
  const authorPayload = await fetchOpenAlexJson(authorSearchUrl);
  const candidates = Array.isArray(authorPayload?.results) ? authorPayload.results : [];
  if (!candidates.length) {
    return [];
  }

  const ranked = candidates
    .map((author) => ({ author, score: authorMatchScore(author, profileName) }))
    .sort((left, right) => right.score - left.score);
  const best = ranked[0];
  if (!best || best.score < 3) {
    return [];
  }

  const authorId = String(best.author?.id || '').split('/').pop();
  if (!authorId) {
    return [];
  }

  const worksUrl = `https://api.openalex.org/works?filter=author.id:${encodeURIComponent(authorId)}&per-page=40&sort=publication_year:desc`;
  const worksPayload = await fetchOpenAlexJson(worksUrl);
  const works = Array.isArray(worksPayload?.results) ? worksPayload.results : [];

  const results = [];
  const seen = new Set();
  for (const work of works) {
    const title = compactWhitespace(work?.display_name || work?.title || 'Untitled Research Output');
    const abstract = reconstructOpenAlexAbstract(work?.abstract_inverted_index);
    if (!looksLikeRealAbstract(abstract)) continue;

    const sourceUrl =
      work?.primary_location?.landing_page_url ||
      work?.doi ||
      work?.id ||
      profileUrl;

    const fingerprint = `${title.toLowerCase()}::${abstract.slice(0, 170).toLowerCase()}`;
    if (seen.has(fingerprint)) continue;
    seen.add(fingerprint);

    results.push({
      title,
      abstract,
      sourceUrl,
      source: 'researchgate_openalex',
      publicationYear: Number.isFinite(work?.publication_year) ? work.publication_year : null,
    });
    if (results.length >= MAX_PROFILE_ABSTRACTS) break;
  }

  return results;
}

function toJinaMirrorVariants(targetUrl) {
  const parsed = new URL(targetUrl);
  const noProtocol = `${parsed.host}${parsed.pathname}${parsed.search}${parsed.hash}`;
  return [
    `https://r.jina.ai/http://${noProtocol}`,
    `https://r.jina.ai/${targetUrl}`,
  ];
}

function hostOfUrl(urlValue) {
  try {
    return new URL(urlValue).hostname.toLowerCase();
  } catch {
    return '';
  }
}

function isRateLimitedHost(urlValue) {
  const host = hostOfUrl(urlValue);
  return host ? rateLimitedProfileHosts.has(host) : false;
}

function buildFetchRequestUrl(targetUrl) {
  if (!import.meta.env.DEV) {
    return targetUrl;
  }
  const proxied = new URL(DEV_PROFILE_FETCH_PROXY_PATH, window.location.origin);
  proxied.searchParams.set('url', targetUrl);
  return proxied.toString();
}

async function fetchPageTextVariants(targetUrl) {
  const attempted = new Set();
  const variants = import.meta.env.DEV
    ? [targetUrl]
    : [targetUrl, ...toJinaMirrorVariants(targetUrl)];
  const pages = [];

  for (const variant of variants) {
    if (attempted.has(variant)) continue;
    attempted.add(variant);
    if (isRateLimitedHost(variant)) continue;
    try {
      const requestUrl = buildFetchRequestUrl(variant);
      if (profileFetchCache.has(requestUrl)) {
        const cached = profileFetchCache.get(requestUrl);
        if (cached?.ok) {
          pages.push({
            url: variant,
            text: cached.text,
            viaProxy: variant !== targetUrl,
          });
        }
        continue;
      }

      const abortController = new AbortController();
      const timeoutHandle = window.setTimeout(() => abortController.abort(), PROFILE_FETCH_TIMEOUT_MS);
      let response;
      try {
        response = await fetch(requestUrl, { method: 'GET', signal: abortController.signal });
      } finally {
        window.clearTimeout(timeoutHandle);
      }
      if (response.status === 429) {
        rateLimitedProfileHosts.add(hostOfUrl(variant));
        profileFetchCache.set(requestUrl, { ok: false, status: response.status });
        continue;
      }
      if (!response.ok) {
        profileFetchCache.set(requestUrl, { ok: false, status: response.status });
        continue;
      }
      const text = await response.text();
      if (!text) continue;
      profileFetchCache.set(requestUrl, { ok: true, text, status: response.status });
      pages.push({
        url: variant,
        text,
        viaProxy: variant !== targetUrl,
      });
    } catch {
      // Ignore and try fallback variant.
    }
  }
  return pages;
}

function parseHtml(text) {
  const snippet = String(text || '').trim().slice(0, 400).toLowerCase();
  if (!snippet) return null;
  const htmlHint = snippet.startsWith('<!doctype html') || snippet.startsWith('<html') || snippet.includes('<body') || snippet.includes('<head');
  if (!htmlHint) return null;
  return new DOMParser().parseFromString(text, 'text/html');
}

function collectScholarDetailLinks(pageText, doc) {
  const links = new Set();

  if (doc) {
    const anchors = doc.querySelectorAll('a.gsc_a_at, a[href*="view_op=view_citation"]');
    for (const anchor of anchors) {
      const href = anchor.getAttribute('data-href') || anchor.getAttribute('href') || '';
      if (!href.includes('view_op=view_citation')) continue;
      const absolute = href.startsWith('http') ? href : `https://scholar.google.com${href}`;
      links.add(absolute.replace(/&amp;/g, '&'));
    }
  }

  const absoluteMatches = pageText.match(scholarDetailUrlRegex()) || [];
  for (const match of absoluteMatches) {
    links.add(match.replace(/&amp;/g, '&'));
  }

  const relativeMatches = pageText.match(/\/citations\?[^"'<>)]*view_op=view_citation[^"'<>)]*/gi) || [];
  for (const rel of relativeMatches) {
    links.add(`https://scholar.google.com${rel.replace(/&amp;/g, '&')}`);
  }

  return Array.from(links);
}

function extractScholarEntryFromPage(pageText, doc, fallbackUrl) {
  let title = '';
  let abstract = '';
  let sourceUrl = fallbackUrl;
  let publicationYear = null;

  if (doc) {
    const titleNode = doc.querySelector('#gsc_oci_title');
    title = compactWhitespace(titleNode?.textContent || '');
    const abstractNode = doc.querySelector('#gsc_oci_descr');
    abstract = compactWhitespace(abstractNode?.textContent || '');

    if (!abstract) {
      for (const row of doc.querySelectorAll('.gs_scl')) {
        const field = compactWhitespace(row.querySelector('.gsc_oci_field')?.textContent || '').toLowerCase();
        const value = compactWhitespace(row.querySelector('.gsc_oci_value')?.textContent || '');
        if ((field === 'description' || field === 'abstract') && value) {
          abstract = value;
          break;
        }
      }
    }

    for (const row of doc.querySelectorAll('.gs_scl')) {
      const field = compactWhitespace(row.querySelector('.gsc_oci_field')?.textContent || '').toLowerCase();
      const value = compactWhitespace(row.querySelector('.gsc_oci_value')?.textContent || '');
      if (!value) continue;
      if (field.includes('publication date') || field === 'year' || field === 'date') {
        const year = extractPublicationYear(value);
        if (year) {
          publicationYear = year;
          break;
        }
      }
    }

    const titleLink = doc.querySelector('#gsc_oci_title a')?.getAttribute('href');
    if (titleLink) {
      sourceUrl = titleLink.startsWith('http') ? titleLink : `https://scholar.google.com${titleLink}`;
    }
  }

  if (!title) {
    const titleMatch = pageText.match(/(?:^|\n)\s*Title\s*[:\-]\s*(.+)/i);
    if (titleMatch) {
      title = compactWhitespace(titleMatch[1]);
    }
  }

  if (!abstract) {
    const match = pageText.match(/(?:^|\n)\s*(?:Description|Abstract)\s*[:\-]\s*([\s\S]{40,2500}?)(?:\n[A-Z][A-Za-z ]{2,25}\s*[:\-]|\n{2,}|$)/i);
    if (match) {
      abstract = compactWhitespace(match[1]);
    }
  }

  if (!publicationYear) {
    publicationYear =
      extractPublicationYear(pageText.match(/(?:^|\n)\s*(?:Publication date|Year|Date)\s*[:\-]\s*([^\n]+)/i)?.[1]) ||
      null;
  }

  if (!looksLikeRealAbstract(abstract)) return null;

  return {
    title: title || 'Untitled Scholar Entry',
    abstract,
    sourceUrl,
    source: 'google_scholar',
    publicationYear,
  };
}

function normalizeResearchGatePublicationUrl(url) {
  return url
    .replace(/^https?:\/\/r\.jina\.ai\/https?:\/\//i, 'https://')
    .replace(/^https?:\/\/r\.jina\.ai\/http:\/\//i, 'https://')
    .split('?')[0];
}

function collectResearchGatePublicationLinks(pageText, doc) {
  const links = new Set();

  if (doc) {
    for (const anchor of doc.querySelectorAll('a[href*="/publication/"]')) {
      const href = anchor.getAttribute('href') || '';
      if (!href.includes('/publication/')) continue;
      const absolute = href.startsWith('http') ? href : `https://www.researchgate.net${href}`;
      links.add(normalizeResearchGatePublicationUrl(absolute));
    }
  }

  const absoluteMatches = pageText.match(/https?:\/\/(?:www\.)?researchgate\.net\/publication\/[^\s"'<>)]*/gi) || [];
  for (const match of absoluteMatches) {
    links.add(normalizeResearchGatePublicationUrl(match));
  }

  const relativeMatches = pageText.match(/\/publication\/[A-Za-z0-9_\-./%]+/g) || [];
  for (const rel of relativeMatches) {
    links.add(normalizeResearchGatePublicationUrl(`https://www.researchgate.net${rel}`));
  }

  return Array.from(links);
}

function extractResearchGateEntryFromPage(pageText, doc, publicationUrl) {
  let title = '';
  let abstract = '';
  let publicationYear = null;

  if (doc) {
    const ogTitle = doc.querySelector('meta[property="og:title"]')?.getAttribute('content') || '';
    title = compactWhitespace(ogTitle) || compactWhitespace(doc.querySelector('h1')?.textContent || '');

    const candidates = [
      ...parseJsonLdAbstracts(doc),
      doc.querySelector('meta[property="og:description"]')?.getAttribute('content') || '',
      doc.querySelector('meta[name="description"]')?.getAttribute('content') || '',
    ]
      .map((entry) => compactWhitespace(entry))
      .filter(Boolean)
      .sort((a, b) => b.length - a.length);

    for (const candidate of candidates) {
      if (looksLikeRealAbstract(candidate)) {
        abstract = candidate;
        break;
      }
    }

    const dateCandidates = [
      doc.querySelector('meta[name="citation_publication_date"]')?.getAttribute('content') || '',
      doc.querySelector('meta[name="citation_date"]')?.getAttribute('content') || '',
      doc.querySelector('meta[property="article:published_time"]')?.getAttribute('content') || '',
      doc.querySelector('meta[name="dc.date"]')?.getAttribute('content') || '',
      doc.querySelector('meta[name="dc.date.issued"]')?.getAttribute('content') || '',
    ];
    for (const candidate of dateCandidates) {
      const year = extractPublicationYear(candidate);
      if (year) {
        publicationYear = year;
        break;
      }
    }
  }

  if (!abstract) {
    const escaped = pageText.match(/"abstract"\s*:\s*"([^"]{40,5000})"/i) || pageText.match(/"description"\s*:\s*"([^"]{40,5000})"/i);
    if (escaped) {
      const decoded = escaped[1]
        .replace(/\\n/g, ' ')
        .replace(/\\"/g, '"')
        .replace(/\\u[0-9a-fA-F]{4}/g, (sequence) => String.fromCharCode(Number.parseInt(sequence.slice(2), 16)));
      if (looksLikeRealAbstract(decoded)) {
        abstract = compactWhitespace(decoded);
      }
    }
  }

  if (!publicationYear) {
    publicationYear =
      extractPublicationYear(pageText.match(/"(?:publicationYear|datePublished|published_time|datePublishedRaw)"\s*:\s*"([^"]+)"/i)?.[1]) ||
      extractPublicationYear(pageText.match(/(?:Published|Publication date)\s*[:\-]?\s*([^\n<]+)/i)?.[1]) ||
      null;
  }

  if (!looksLikeRealAbstract(abstract)) return null;

  return {
    title: title || 'Untitled ResearchGate Entry',
    abstract,
    sourceUrl: publicationUrl,
    source: 'researchgate',
    publicationYear,
  };
}

async function fetchScholarProfileAbstracts(profileUrl) {
  const detailLinks = new Set();
  for (const cstart of [0, 20]) {
    const pageUrl = new URL(profileUrl);
    pageUrl.searchParams.set('cstart', String(cstart));
    pageUrl.searchParams.set('pagesize', '20');
    pageUrl.searchParams.set('sortby', 'pubdate');
    const pages = await fetchPageTextVariants(pageUrl.toString());
    for (const page of pages) {
      const doc = parseHtml(page.text);
      const links = collectScholarDetailLinks(page.text, doc);
      for (const link of links) {
        detailLinks.add(link);
        if (detailLinks.size >= MAX_PROFILE_ABSTRACTS) break;
      }
      if (detailLinks.size >= MAX_PROFILE_ABSTRACTS) break;
    }
    if (detailLinks.size >= MAX_PROFILE_ABSTRACTS) break;
  }

  const results = [];
  const seen = new Set();
  let attempts = 0;
  for (const link of detailLinks) {
    if (attempts >= MAX_PROFILE_DETAIL_FETCHES) break;
    if (rateLimitedProfileHosts.size > 0 && results.length > 0) break;
    attempts += 1;
    const pages = await fetchPageTextVariants(link);
    for (const page of pages) {
      const doc = parseHtml(page.text);
      const item = extractScholarEntryFromPage(page.text, doc, link);
      if (!item) continue;
      const fingerprint = `${item.title.toLowerCase()}::${item.abstract.slice(0, 160).toLowerCase()}`;
      if (seen.has(fingerprint)) break;
      seen.add(fingerprint);
      results.push(item);
      break;
    }
    if (results.length >= MAX_PROFILE_ABSTRACTS) break;
  }

  return sortByPublicationYearDesc(results);
}

async function fetchResearchGateProfileAbstracts(profileUrl) {
  try {
    const openAlexResults = await fetchResearchGateViaOpenAlex(profileUrl);
    if (openAlexResults.length) {
      return {
        sourceLabel: 'ResearchGate (OpenAlex metadata)',
        abstracts: openAlexResults,
      };
    }
  } catch {
    // Fall through to direct scraping attempts.
  }

  const publicationLinks = new Set();
  const profilePages = [profileUrl];
  if (!profileUrl.includes('/publications')) {
    profilePages.push(`${profileUrl.replace(/\/$/, '')}/publications`);
  }

  for (const pageUrl of profilePages) {
    const pages = await fetchPageTextVariants(pageUrl);
    for (const page of pages) {
      const doc = parseHtml(page.text);
      const links = collectResearchGatePublicationLinks(page.text, doc);
      for (const link of links) {
        publicationLinks.add(link);
        if (publicationLinks.size >= MAX_PROFILE_ABSTRACTS) break;
      }
      if (publicationLinks.size >= MAX_PROFILE_ABSTRACTS) break;
    }
    if (publicationLinks.size >= MAX_PROFILE_ABSTRACTS) break;
  }

  const results = [];
  const seen = new Set();
  let attempts = 0;
  for (const publicationUrl of publicationLinks) {
    if (attempts >= MAX_PROFILE_DETAIL_FETCHES) break;
    if (rateLimitedProfileHosts.size > 0 && results.length > 0) break;
    attempts += 1;
    const pages = await fetchPageTextVariants(publicationUrl);
    for (const page of pages) {
      const doc = parseHtml(page.text);
      const item = extractResearchGateEntryFromPage(page.text, doc, publicationUrl);
      if (!item) continue;
      const fingerprint = `${item.title.toLowerCase()}::${item.abstract.slice(0, 160).toLowerCase()}`;
      if (seen.has(fingerprint)) break;
      seen.add(fingerprint);
      results.push(item);
      break;
    }
    if (results.length >= MAX_PROFILE_ABSTRACTS) break;
  }

  const sortedResults = sortByPublicationYearDesc(results);

  return {
    sourceLabel: 'ResearchGate',
    abstracts: sortedResults,
  };
}

async function fetchProfileAbstracts(profileUrl) {
  const parsed = new URL(profileUrl);
  const host = parsed.hostname.toLowerCase();
  if (host.includes('scholar.google.')) {
    const abstracts = await fetchScholarProfileAbstracts(profileUrl);
    return { sourceLabel: 'Google Scholar', abstracts };
  }
  if (host === 'researchgate.net' || host.endsWith('.researchgate.net')) {
    const result = await fetchResearchGateProfileAbstracts(profileUrl);
    if (Array.isArray(result)) {
      return { sourceLabel: 'ResearchGate', abstracts: result };
    }
    return {
      sourceLabel: result?.sourceLabel || 'ResearchGate',
      abstracts: Array.isArray(result?.abstracts) ? result.abstracts : [],
    };
  }
  throw new Error('Only Google Scholar and ResearchGate profile URLs are supported.');
}

function updateProfileFetchStatus(text, isError = false) {
  if (!profileFetchStatus) return;
  profileFetchStatus.textContent = text;
  profileFetchStatus.classList.toggle('is-error', Boolean(isError));
}

function getSelectedProfileAbstracts() {
  if (!fetchedProfileAbstracts.length || !selectedProfileAbstractIds.size) return [];
  return fetchedProfileAbstracts.filter((item) => selectedProfileAbstractIds.has(item.id));
}

function updateProfileAbstractSelectionCount() {
  if (!profileAbstractsCount) return;
  const selected = getSelectedProfileAbstracts().length;
  profileAbstractsCount.textContent = `${selected} of ${fetchedProfileAbstracts.length} selected`;
}

function renderProfileAbstracts() {
  if (!profileAbstractsPanel || !profileAbstractsList) return;
  if (!fetchedProfileAbstracts.length) {
    profileAbstractsPanel.classList.add('hidden');
    profileAbstractsList.innerHTML = '';
    updateProfileAbstractSelectionCount();
    return;
  }

  profileAbstractsPanel.classList.remove('hidden');
  profileAbstractsList.innerHTML = fetchedProfileAbstracts
    .map((item) => {
      const checked = selectedProfileAbstractIds.has(item.id) ? 'checked' : '';
      return `
        <label class="profile-abstract-item">
          <input type="checkbox" data-abstract-id="${escapeHtml(item.id)}" ${checked} />
          <div class="profile-abstract-body">
            <p class="profile-abstract-title">${escapeHtml(item.title || 'Untitled')}</p>
            <p class="profile-abstract-source">${escapeHtml(item.sourceLabel || '')}</p>
            <p class="profile-abstract-text">${escapeHtml(item.abstract || '')}</p>
          </div>
        </label>
      `;
    })
    .join('');
  updateProfileAbstractSelectionCount();
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

function normalizeMatchingMode(value) {
  const mode = String(value || 'tfidf')
    .trim()
    .toLowerCase();
  return MATCHING_MODES.has(mode) ? mode : 'tfidf';
}

function matchingModeLabel(mode) {
  if (mode === 'semantic') return 'Semantic';
  if (mode === 'hybrid') return 'Hybrid';
  return 'TF-IDF';
}

function scoringStatusText(mode) {
  if (mode === 'semantic') return 'Computing semantic relevance in scoring worker...';
  if (mode === 'hybrid') return 'Computing hybrid relevance (TF-IDF + semantic) in scoring worker...';
  return 'Computing TF-IDF relevance in scoring worker...';
}

function modeNeedsSemanticModel(mode = normalizeMatchingMode(matchingModeInput?.value)) {
  return mode === 'semantic' || mode === 'hybrid';
}

function updateRunButtonAvailability() {
  if (!runBtn) return;
  if (isExtracting) {
    runBtn.disabled = true;
    return;
  }

  const needsSemanticModel = modeNeedsSemanticModel();
  const disabled = modelDownloadBusy || (needsSemanticModel && !semanticModelAvailable);
  runBtn.disabled = disabled;

  if (needsSemanticModel && !semanticModelAvailable) {
    runBtn.title = 'Download the semantic model, or use a deployment that bundles /models files.';
  } else {
    runBtn.removeAttribute('title');
  }
}

function formatBytes(bytes) {
  const numeric = Number(bytes);
  if (!Number.isFinite(numeric) || numeric < 0) return '';
  if (numeric < 1024) return `${Math.round(numeric)} B`;
  if (numeric < 1024 * 1024) return `${(numeric / 1024).toFixed(1)} KB`;
  return `${(numeric / (1024 * 1024)).toFixed(1)} MB`;
}

async function isUrlReachable(url) {
  try {
    const headResponse = await fetch(url, { method: 'HEAD', cache: 'no-store' });
    if (headResponse.ok) return true;
    if (headResponse.status !== 405 && headResponse.status !== 501) {
      return false;
    }
  } catch {
    // Fallback to GET below.
  }

  try {
    const getResponse = await fetch(url, { method: 'GET', cache: 'no-store' });
    return getResponse.ok;
  } catch {
    return false;
  }
}

async function isBundledSemanticModelAvailable() {
  const requiredLocalChecks = ['config.json', 'onnx/model_q4.onnx_data'];
  for (const file of requiredLocalChecks) {
    const url = semanticModelUrl(file, window.location.origin);
    const reachable = await isUrlReachable(url);
    if (!reachable) return false;
  }
  return true;
}

function setModelDownloadUi({
  busy = modelDownloadBusy,
  available = semanticModelAvailable,
  statusText = '',
  progressPercent = null,
  loadedBytes = null,
  totalBytes = null,
} = {}) {
  modelDownloadBusy = Boolean(busy);
  const modelReady = Boolean(available);
  if (downloadModelBtn) {
    downloadModelBtn.disabled = modelDownloadBusy || modelReady;
    downloadModelBtn.classList.toggle('is-ready', modelReady);
  }

  if (modelDownloadBusy) {
    if (downloadModelBtn) downloadModelBtn.textContent = 'Downloading model...';
  } else if (modelReady) {
    if (downloadModelBtn) downloadModelBtn.textContent = 'Model ready';
  } else {
    if (downloadModelBtn) downloadModelBtn.textContent = 'Download semantic model';
  }

  if (statusText && modelDownloadStatus) {
    modelDownloadStatus.textContent = statusText;
  }

  updateRunButtonAvailability();

  if (progressPercent == null) {
    if (modelDownloadProgressWrap) modelDownloadProgressWrap.classList.add('hidden');
    if (modelDownloadProgress) modelDownloadProgress.value = 0;
    if (modelDownloadProgressText) modelDownloadProgressText.textContent = '0%';
    return;
  }

  const normalized = Math.max(0, Math.min(100, Number(progressPercent)));
  if (modelDownloadProgressWrap) modelDownloadProgressWrap.classList.remove('hidden');
  if (modelDownloadProgress) modelDownloadProgress.value = normalized;

  const loaded = formatBytes(loadedBytes);
  const total = formatBytes(totalBytes);
  if (loaded && total) {
    if (modelDownloadProgressText) {
      modelDownloadProgressText.textContent = `${Math.round(normalized)}% (${loaded}/${total})`;
    }
  } else {
    if (modelDownloadProgressText) modelDownloadProgressText.textContent = `${Math.round(normalized)}%`;
  }
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

function runWorkerScore({ worksTexts, workNames, customKeywords, matchingMode }) {
  const requestId = ++requestSeq;
  const normalizedMode = normalizeMatchingMode(matchingMode);
  const timeoutMs = normalizedMode === 'tfidf' ? 120000 : 900000;
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
      matchingMode: normalizedMode,
      topN,
    });
  });
}

function startWorkerPrefetchModel() {
  const requestId = ++requestSeq;
  const timeoutMs = 1800000;

  const promise = new Promise((resolve, reject) => {
    const timeoutHandle = window.setTimeout(() => {
      if (!pendingRequests.has(requestId)) return;
      pendingRequests.delete(requestId);
      reject(new Error(`Model download timed out after ${Math.round(timeoutMs / 1000)}s.`));
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
      type: 'prefetch-model',
      requestId,
    });
  });

  return { requestId, promise };
}

function startWorkerProbeModelAvailability() {
  const requestId = ++requestSeq;
  const timeoutMs = 120000;

  const promise = new Promise((resolve, reject) => {
    const timeoutHandle = window.setTimeout(() => {
      if (!pendingRequests.has(requestId)) return;
      pendingRequests.delete(requestId);
      reject(new Error(`Model availability probe timed out after ${Math.round(timeoutMs / 1000)}s.`));
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
      type: 'probe-model',
      requestId,
    });
  });

  return { requestId, promise };
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
    .replaceAll('\r', '')
    .replaceAll('\\', '\\\\')
    .replaceAll('\n', '\\n')
    .replaceAll(';', '\\;')
    .replaceAll(',', '\\,')
    .replaceAll(/\u0000/g, '');
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
  const calendarName = 'CHI-tinerary';

  const lines = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    `PRODID:${ICS_PROD_ID}`,
    'CALSCALE:GREGORIAN',
    'METHOD:PUBLISH',
    `X-WR-CALNAME:${escapeIcsText(calendarName)}`,
    `X-WR-TIMEZONE:${CONFERENCE_TIMEZONE}`,
    'BEGIN:VEVENT',
    `UID:${uidSlug}@chi-relevance-client`,
    `DTSTAMP:${dtStamp}`,
    `CREATED:${dtStamp}`,
    `LAST-MODIFIED:${dtStamp}`,
    'SEQUENCE:0',
    'STATUS:CONFIRMED',
    'TRANSP:OPAQUE',
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

  const icsText = `${lines.map(foldIcsLine).join('\r\n')}\r\n`;
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
    fileSelection.textContent = 'No PDFs selected yet.';
    return;
  }
  if (files.length === 1) {
    fileSelection.textContent = files[0].name;
    return;
  }
  fileSelection.textContent = `${files.length} files selected`;
}

function normalizeCalendarKeyPart(value) {
  return String(value || '')
    .trim()
    .toLowerCase();
}

function calendarItemKey(row) {
  const payload = {
    title: normalizeCalendarKeyPart(row?.title),
    start: asEpochMs(row?.start_date_unix_ms),
    end: asEpochMs(row?.end_date_unix_ms),
    room: normalizeCalendarKeyPart(row?.room),
    building: normalizeCalendarKeyPart(row?.building),
  };
  return JSON.stringify(payload);
}

function loadDownloadedCalendarItems() {
  try {
    const raw = window.localStorage.getItem(CALENDAR_DOWNLOADS_STORAGE_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((entry) => typeof entry === 'string'));
  } catch {
    return new Set();
  }
}

function persistDownloadedCalendarItems() {
  try {
    window.localStorage.setItem(CALENDAR_DOWNLOADS_STORAGE_KEY, JSON.stringify(Array.from(downloadedCalendarItems)));
  } catch {
    // Ignore storage write failures (e.g., private mode quota restrictions).
  }
}

function hasDownloadedCalendarItem(row) {
  return downloadedCalendarItems.has(calendarItemKey(row));
}

function markCalendarItemDownloaded(row) {
  downloadedCalendarItems.add(calendarItemKey(row));
  persistDownloadedCalendarItems();
}

function setCalendarButtonDownloadedState(button, downloaded) {
  if (!(button instanceof HTMLElement)) return;
  button.classList.toggle('is-added', downloaded);
  button.setAttribute('data-downloaded', downloaded ? 'true' : 'false');
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

async function refreshModelCacheUi({ forceUi = false } = {}) {
  if (modelDownloadBusy && !forceUi) {
    updateRunButtonAvailability();
    return {
      bundled: false,
      probeAvailable: semanticModelAvailable,
      available: semanticModelAvailable,
      probeError: '',
    };
  }

  const refreshSeq = ++modelAvailabilityRefreshSeq;
  let bundled = false;
  let probeAvailable = false;
  let probeError = '';

  try {
    bundled = await isBundledSemanticModelAvailable();
  } catch {
    bundled = false;
  }

  if (!bundled) {
    try {
      const task = startWorkerProbeModelAvailability();
      const payload = await task.promise;
      const probeResult = payload?.result || {};
      probeAvailable = Boolean(probeResult.available);
      if (!probeAvailable && probeResult.error) {
        probeError = String(probeResult.error);
      }
    } catch (error) {
      probeAvailable = false;
      probeError = error instanceof Error ? error.message : String(error);
    }
  } else {
    probeAvailable = true;
  }

  const available = bundled || probeAvailable;
  const snapshot = {
    bundled,
    probeAvailable,
    available,
    probeError,
  };

  if (refreshSeq !== modelAvailabilityRefreshSeq) {
    return snapshot;
  }

  semanticModelAvailable = available;

  if (bundled) {
    setModelDownloadUi({
      busy: false,
      available: true,
      statusText: 'Semantic model files are bundled in this deployment. You can run Semantic/Hybrid now.',
    });
    return snapshot;
  }

  if (probeAvailable) {
    setModelDownloadUi({
      busy: false,
      available: true,
      statusText: 'Semantic model is already available in this browser session. You can run Semantic/Hybrid now.',
    });
    return snapshot;
  }

  setModelDownloadUi({
    busy: false,
    available: false,
    statusText: 'Semantic model not available yet in this browser. Click "Download semantic model".',
  });

  return snapshot;
}

async function openCalendarImport(icsText, fileName, summary) {
  void summary;
  const blob = new Blob([icsText], { type: 'text/calendar;charset=utf-8' });
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

function formatPercent(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 'n/a';
  return `${numeric.toFixed(2)}%`;
}

function scoreBreakdownText(row) {
  const mode = normalizeMatchingMode(row.relevance_mode || latestMatchingMode);
  const tfidf = Number(row.relevance_score_tfidf);
  const semantic = Number(row.relevance_score_semantic);

  if (mode === 'hybrid' && Number.isFinite(tfidf) && Number.isFinite(semantic)) {
    return `TF-IDF ${tfidf.toFixed(1)}% | Semantic ${semantic.toFixed(1)}%`;
  }
  if (mode === 'semantic' && Number.isFinite(tfidf)) {
    return `TF-IDF baseline ${tfidf.toFixed(1)}%`;
  }
  return '';
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
        const modeLabel = matchingModeLabel(normalizeMatchingMode(row.relevance_mode || latestMatchingMode));
        const scoreBreakdown = scoreBreakdownText(row);
        const isCalendarAdded = hasDownloadedCalendarItem(row);
        return `
      <article class="result-card">
        <header>
          <p class="rank">#${row.relevance_rank}</p>
          <div class="score-wrap">
            <p class="score">${escapeHtml(formatPercent(row.relevance_score))}</p>
            <p class="score-mode">${escapeHtml(modeLabel)}</p>
            ${scoreBreakdown ? `<p class="score-breakdown">${escapeHtml(scoreBreakdown)}</p>` : ''}
          </div>
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
          <button
            type="button"
            class="calendar-btn ${isCalendarAdded ? 'is-added' : ''}"
            data-row-index="${startIndex + idx}"
            data-downloaded="${isCalendarAdded ? 'true' : 'false'}"
          >
            Add to calendar (.ics)
            ${isCalendarAdded ? '<span class="calendar-check" aria-hidden="true">✓</span>' : ''}
          </button>
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
    throw new Error('Upload PDFs or select profile abstracts, then run analysis.');
  }

  const runId = ++scoringRunSeq;
  const customKeywords = parseCustomKeywords(keywordInput.value);
  const matchingMode = normalizeMatchingMode(matchingModeInput?.value);
  if (modeNeedsSemanticModel(matchingMode) && !semanticModelAvailable) {
    throw new Error('Semantic model is not available. Click "Download semantic model" first.');
  }
  setStatus(statusText || scoringStatusText(matchingMode));

  const { result } = await runWorkerScore({
    worksTexts: extractedContext.worksTexts,
    workNames: extractedContext.workNames,
    customKeywords,
    matchingMode,
  });

  if (runId !== scoringRunSeq) {
    return;
  }

  latestMatchingMode = normalizeMatchingMode(result.matching_mode || matchingMode);
  latestRows = result.rows || [];
  visibleResults = Math.min(DEFAULT_VISIBLE_RESULTS, latestRows.length || DEFAULT_VISIBLE_RESULTS);

  renderResultsPage();

  if (!skipDoneStatus) {
    const shown = Math.min(visibleResults, latestRows.length);
    setStatus(
      `Done (${matchingModeLabel(latestMatchingMode)}). Showing ${shown.toLocaleString()} of ${latestRows.length.toLocaleString()} results.`
    );
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
    markCalendarItemDownloaded(row);
    setCalendarButtonDownloadedState(calendarBtn, true);
    if (!calendarBtn.querySelector('.calendar-check')) {
      const check = document.createElement('span');
      check.className = 'calendar-check';
      check.setAttribute('aria-hidden', 'true');
      check.textContent = '✓';
      calendarBtn.appendChild(check);
    }
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

if (fetchProfileBtn) {
  fetchProfileBtn.addEventListener('click', async () => {
    try {
      const normalizedUrl = normalizeProfileUrl(profileUrlInput?.value);
      rateLimitedProfileHosts.clear();
      profileFetchCache.clear();
      fetchProfileBtn.disabled = true;
      updateProfileFetchStatus('Fetching abstracts from profile...');
      appendStatusItem(`Fetching profile abstracts from ${normalizedUrl}`);

      const { sourceLabel, abstracts } = await fetchProfileAbstracts(normalizedUrl);
      if (!abstracts.length) {
        fetchedProfileAbstracts = [];
        selectedProfileAbstractIds.clear();
        renderProfileAbstracts();
        if (rateLimitedProfileHosts.size > 0) {
          updateProfileFetchStatus(
            `No abstracts returned before source rate limit. Wait a bit and retry for ${sourceLabel}.`,
            true
          );
        } else {
          updateProfileFetchStatus(`No abstracts were found on this ${sourceLabel} profile.`, true);
        }
        return;
      }

      fetchedProfileAbstracts = abstracts.slice(0, MAX_PROFILE_ABSTRACTS).map((item, idx) => ({
        id: `${idx + 1}`,
        title: compactWhitespace(item.title).slice(0, 240),
        abstract: compactWhitespace(item.abstract).slice(0, 3000),
        sourceUrl: item.sourceUrl || '',
        source: item.source || '',
        sourceLabel,
      }));

      selectedProfileAbstractIds.clear();
      for (const item of fetchedProfileAbstracts) {
        selectedProfileAbstractIds.add(item.id);
      }

      renderProfileAbstracts();
      const partialSuffix =
        rateLimitedProfileHosts.size > 0
          ? ' Source rate-limited; showing partial results.'
          : '';
      updateProfileFetchStatus(
        `Fetched ${fetchedProfileAbstracts.length} abstracts from ${sourceLabel}. Select relevant abstracts and click Run.${partialSuffix}`
      );
      appendStatusItem(`Fetched ${fetchedProfileAbstracts.length} abstracts from ${sourceLabel}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      fetchedProfileAbstracts = [];
      selectedProfileAbstractIds.clear();
      renderProfileAbstracts();
      updateProfileFetchStatus(message, true);
      appendStatusItem(`Profile fetch error: ${message}`);
    } finally {
      fetchProfileBtn.disabled = false;
    }
  });
}

if (selectAllAbstractsBtn) {
  selectAllAbstractsBtn.addEventListener('click', () => {
    selectedProfileAbstractIds.clear();
    for (const item of fetchedProfileAbstracts) {
      selectedProfileAbstractIds.add(item.id);
    }
    renderProfileAbstracts();
  });
}

if (clearAllAbstractsBtn) {
  clearAllAbstractsBtn.addEventListener('click', () => {
    selectedProfileAbstractIds.clear();
    renderProfileAbstracts();
  });
}

if (profileAbstractsList) {
  profileAbstractsList.addEventListener('change', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const id = String(target.dataset.abstractId || '');
    if (!id) return;
    if (target.checked) {
      selectedProfileAbstractIds.add(id);
    } else {
      selectedProfileAbstractIds.delete(id);
    }
    updateProfileAbstractSelectionCount();
  });
}

fileInput.addEventListener('change', () => {
  updateFileSelectionText();
});

loadMoreBtn.addEventListener('click', () => {
  visibleResults = Math.min(latestRows.length, visibleResults + LOAD_MORE_STEP);
  renderResultsPage();
});

if (downloadModelBtn) {
  downloadModelBtn.addEventListener('click', async () => {
    if (modelDownloadBusy) return;

    const availability = await refreshModelCacheUi({ forceUi: true });
    if (availability.available) {
      setStatus('Semantic model is already available.');
      return;
    }

    setModelDownloadUi({
      busy: true,
      available: false,
      statusText: 'Starting semantic model download...',
      progressPercent: 0,
    });

    try {
      const task = startWorkerPrefetchModel();
      modelDownloadRequestId = task.requestId;
      await task.promise;
      const updated = await refreshModelCacheUi({ forceUi: true });
      if (updated.available) {
        setStatus('Semantic model is ready for Semantic/Hybrid matching.');
      } else {
        setStatus('Model download finished, but readiness could not be confirmed. Try Download again.');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setModelDownloadUi({
        busy: false,
        available: false,
        statusText: `Model download failed: ${message}`,
      });
      setStatus(message);
      appendStatusItem(`Error: ${message}`);
    } finally {
      modelDownloadRequestId = null;
      if (modelDownloadBusy) {
        modelDownloadBusy = false;
        updateRunButtonAvailability();
      }
    }
  });
}

window.addEventListener('focus', () => {
  void refreshModelCacheUi();
});

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    void refreshModelCacheUi();
  }
});

keywordInput.addEventListener('input', () => {
  renderBoostedKeywords();
  if (keywordDebounceHandle) {
    window.clearTimeout(keywordDebounceHandle);
  }

  keywordDebounceHandle = window.setTimeout(async () => {
    if (!extractedContext || isExtracting) return;
    const mode = normalizeMatchingMode(matchingModeInput?.value);
    if (modeNeedsSemanticModel(mode) && !semanticModelAvailable) return;

    try {
      await runScoringFromExtracted(null, true);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(message);
      appendStatusItem(`Error: ${message}`);
    }
  }, KEYWORD_DEBOUNCE_MS);
});

matchingModeInput.addEventListener('change', async () => {
  updateRunButtonAvailability();
  if (!extractedContext || isExtracting) return;
  const mode = normalizeMatchingMode(matchingModeInput.value);
  if (modeNeedsSemanticModel(mode) && !semanticModelAvailable) {
    setStatus('Semantic model is not available. Download it first or deploy bundled /models files.');
    return;
  }

  try {
    await runScoringFromExtracted(`Recomputing with ${matchingModeLabel(mode)} matching...`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    appendStatusItem(`Error: ${message}`);
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearStatusItems();
  clearResults();

  try {
    const files = Array.from(fileInput.files || []);
    const selectedAbstracts = getSelectedProfileAbstracts();
    validateUploads(files, { allowEmpty: selectedAbstracts.length > 0 });

    runBtn.disabled = true;
    isExtracting = true;
    extractedContext = null;

    let worksTexts = [];
    const workNames = [];

    if (files.length) {
      setStatus('Extracting text from PDFs with pdf.js...');
      const pdfWorksTexts = await extractAllPdfs(files);
      worksTexts = worksTexts.concat(pdfWorksTexts);
      workNames.push(...files.map((f) => f.name));
    }

    if (selectedAbstracts.length) {
      const prefix = files.length ? 'Combining selected profile abstracts with extracted PDF text...' : 'Preparing selected profile abstracts...';
      setStatus(prefix);
      worksTexts = worksTexts.concat(selectedAbstracts.map((item) => item.abstract));
      workNames.push(
        ...selectedAbstracts.map((item, idx) => `profile_abstract_${idx + 1}: ${item.title || 'Untitled'}`)
      );
      appendStatusItem(`Included ${selectedAbstracts.length} selected profile abstracts.`);
    }

    extractedContext = {
      worksTexts,
      workNames,
    };

    isExtracting = false;
    await runScoringFromExtracted();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    appendStatusItem(`Error: ${message}`);
  } finally {
    isExtracting = false;
    updateRunButtonAvailability();
  }
});

(async () => {
  renderBoostedKeywords();
  renderProfileAbstracts();
  updateFileSelectionText();
  latestMatchingMode = normalizeMatchingMode(matchingModeInput?.value);
  await refreshModelCacheUi();
  updateRunButtonAvailability();
  try {
    const scheduleIndex = await loadScheduleIndex();
    scheduleRowCount = scheduleIndex.row_count || 0;
    scoreWorker.postMessage({ type: 'init', scheduleIndex });
    setStatus(`Loaded schedule index (${scheduleRowCount.toLocaleString()} rows). Ready.`);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error));
  }
})();
