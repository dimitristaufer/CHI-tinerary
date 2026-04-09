import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { scoreSchedule } from '../src/shared/scoring.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..', '..');

const scheduleIndexPath = path.join(repoRoot, 'chi_relevance_client', 'public', 'data', 'schedule_index.json');
const worksDir = path.join(repoRoot, 'Works');
const baselineCsv = path.join(repoRoot, 'chi_2026_schedule_relevance_sorted.csv');

function findPdftotext() {
  const candidates = ['/opt/homebrew/bin/pdftotext', '/usr/local/bin/pdftotext', 'pdftotext'];
  for (const cmd of candidates) {
    try {
      execFileSync(cmd, ['-v'], { stdio: 'ignore' });
      return cmd;
    } catch {
      // continue
    }
  }
  throw new Error('Could not find pdftotext for verification.');
}

function loadBaselineTopTitles(limit) {
  const py = `
import csv, json
from pathlib import Path
p = Path(r'''${baselineCsv}''')
with p.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print(json.dumps([r.get('title','') for r in rows[:${limit}]]))
`.trim();

  const output = execFileSync('python3', ['-c', py], { cwd: repoRoot, encoding: 'utf8' });
  return JSON.parse(output);
}

function main() {
  if (!fs.existsSync(scheduleIndexPath)) {
    throw new Error(`Missing schedule index: ${scheduleIndexPath}`);
  }
  if (!fs.existsSync(baselineCsv)) {
    throw new Error(`Missing baseline CSV: ${baselineCsv}`);
  }

  const scheduleIndex = JSON.parse(fs.readFileSync(scheduleIndexPath, 'utf8'));
  const pdftotext = findPdftotext();

  const workFiles = fs
    .readdirSync(worksDir)
    .filter((name) => name.toLowerCase().endsWith('.pdf'))
    .sort();
  if (!workFiles.length) {
    throw new Error(`No PDFs found in ${worksDir}`);
  }

  const worksTexts = workFiles.map((name) => {
    const fullPath = path.join(worksDir, name);
    return execFileSync(pdftotext, [fullPath, '-'], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 128,
    });
  });

  const topN = 50;
  const jsResult = scoreSchedule({
    worksTexts,
    workNames: workFiles,
    scheduleIndex,
    topN,
  });

  const pyTitles = loadBaselineTopTitles(topN);
  const jsTitles = jsResult.rows.map((r) => r.title || '');

  const overlap = jsTitles.filter((t) => pyTitles.includes(t)).length;
  const exactTop10 = jsTitles.slice(0, 10).filter((t, idx) => t === pyTitles[idx]).length;

  console.log(`Compared top ${topN} against Python baseline.`);
  console.log(`Top-${topN} overlap: ${overlap}/${topN}`);
  console.log(`Exact match in top-10 positions: ${exactTop10}/10`);

  console.log('Top 5 JS titles:');
  jsTitles.slice(0, 5).forEach((title, idx) => {
    console.log(`${idx + 1}. ${title}`);
  });

  if (overlap < 40) {
    throw new Error(`Parity check failed: overlap too low (${overlap}/${topN}).`);
  }
}

main();
