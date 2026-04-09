export const STOPWORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
  'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'could', 'did', 'do',
  'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he',
  'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its',
  'itself', 'just', 'llm', 'llms', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off',
  'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so',
  'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
  'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
  'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves',
  'acm', 'accepted', 'affiliation', 'al', 'arxiv', 'author', 'authors', 'conference', 'copyright', 'date', 'department',
  'doi', 'edition', 'et', 'figure', 'figures', 'http', 'https', 'institute', 'isbn', 'issn', 'journal', 'org', 'page',
  'pages', 'preprint', 'proceedings', 'section', 'table', 'tables', 'university', 'volume', 'vol', 'workshop', 'www',
  'staufer', 'morehouse', 'hartmann', 'berendt',
]);

const TOKEN_RE = /[a-z][a-z\-]{2,}/g;

export function tokenize(text) {
  const lowered = (text || '').toLowerCase();
  const tokens = lowered.match(TOKEN_RE) || [];
  const filtered = [];
  for (const token of tokens) {
    if (STOPWORDS.has(token)) continue;
    if (/^\d+$/.test(token)) continue;
    filtered.push(token);
  }
  return filtered;
}

export function counterFromTokens(tokens) {
  const counts = Object.create(null);
  for (const token of tokens) {
    counts[token] = (counts[token] || 0) + 1;
  }
  return counts;
}

export function counterFromText(text) {
  return counterFromTokens(tokenize(text));
}
