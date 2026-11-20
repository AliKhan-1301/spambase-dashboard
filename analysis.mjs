// Spambase: validate, clean, analyse and model. Writes data/spambase_clean.csv and dashboard-data.js.
// Run: node analysis.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';

// ---------- load + validate ----------
const raw = readFileSync('data/spambase_raw.csv', 'utf8').trim().split(/\r?\n/);
const header = raw[0].split(',');
assert.equal(header.length, 58, 'expected 57 features + class');
assert.equal(header[57], 'class');
const FEATS = header.slice(0, 57);
const rows = raw.slice(1).map((line, i) => {
  const v = line.split(',').map(Number);
  assert.equal(v.length, 58, `row ${i + 2} has ${v.length} fields`);
  assert.ok(v.every(Number.isFinite), `row ${i + 2} has a non-numeric value`);
  return v;
});
const issues = {
  rows_raw: rows.length,
  missing_or_non_numeric: 0, // asserted above
  bad_labels: rows.filter(r => r[57] !== 0 && r[57] !== 1).length,
  negative_values: rows.filter(r => r.some(x => x < 0)).length,
  pct_over_100: rows.filter(r => r.slice(0, 54).some(x => x > 100)).length,
};
assert.equal(issues.bad_labels, 0); assert.equal(issues.negative_values, 0); assert.equal(issues.pct_over_100, 0);

// ---------- refine: drop exact duplicates (same 57 features and label) ----------
const seen = new Set(), clean = [];
for (const r of rows) { const k = r.join(','); if (!seen.has(k)) { seen.add(k); clean.push(r); } }
// same features but different labels: impossible to classify, report them
const byFeat = new Map(); for (const r of clean) { const k = r.slice(0, 57).join(','); byFeat.set(k, (byFeat.get(k) || new Set()).add(r[57])); }
issues.duplicates_removed = rows.length - clean.length;
issues.conflicting_labels = [...byFeat.values()].filter(s => s.size > 1).length;
issues.rows_clean = clean.length;
writeFileSync('data/spambase_clean.csv', [header.join(','), ...clean.map(r => r.join(','))].join('\n') + '\n');

// ---------- descriptive analysis ----------
const isSpam = r => r[57] === 1;
const spam = clean.filter(isSpam), ham = clean.filter(r => !isSpam(r));
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
const label = f => f.replace('word_freq_', '').replace('char_freq_', 'char ').replace('capital_run_length_', 'CAPS ')
  .replace('%3B', ';').replace('%28', '(').replace('%5B', '[').replace('%21', '!').replace('%24', '$').replace('%23', '#');
const kind = f => f.startsWith('word') ? 'word' : f.startsWith('char') ? 'character' : 'capitals';
const y = clean.map(r => r[57]);
const corr = (x, t) => { const mx = mean(x), mt = mean(t); let a = 0, b = 0, c = 0; for (let i = 0; i < x.length; i++) { a += (x[i] - mx) * (t[i] - mt); b += (x[i] - mx) ** 2; c += (t[i] - mt) ** 2; } return b && c ? a / Math.sqrt(b * c) : 0; };

const features = FEATS.map((f, j) => {
  const s = spam.map(r => r[j]), h = ham.map(r => r[j]), all = clean.map(r => r[j]);
  return { feature: f, label: label(f), kind: kind(f),
    spam_present: s.filter(v => v > 0).length / s.length, ham_present: h.filter(v => v > 0).length / h.length,
    spam_mean: mean(s), ham_mean: mean(h), corr: corr(all.map(Math.log1p), y) };
});

// ---------- model: logistic regression on log1p + standardised features, stratified 70/30 split ----------
let seed = 42; const rand = () => { seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const split = g => { const s = shuffle(g.slice()), k = Math.round(s.length * .7); return [s.slice(0, k), s.slice(k)]; };
const [sTr, sTe] = split(spam), [hTr, hTe] = split(ham), train = shuffle([...sTr, ...hTr]), test = [...sTe, ...hTe];
const X = r => r.slice(0, 57).map(Math.log1p);
const mu = FEATS.map((_, j) => mean(train.map(r => Math.log1p(r[j])))), sd = FEATS.map((_, j) => std(train.map(r => Math.log1p(r[j]))) || 1);
const Z = r => X(r).map((v, j) => (v - mu[j]) / sd[j]);
const Xtr = train.map(Z), ytr = train.map(r => r[57]);
const sig = z => 1 / (1 + Math.exp(-z));
let w = new Array(57).fill(0), b = 0; const lr = .5, l2 = 1e-3, n = Xtr.length;
for (let it = 0; it < 1500; it++) {
  const gw = new Array(57).fill(0); let gb = 0;
  for (let i = 0; i < n; i++) { const e = sig(b + Xtr[i].reduce((s, v, j) => s + v * w[j], 0)) - ytr[i]; gb += e; for (let j = 0; j < 57; j++) gw[j] += e * Xtr[i][j]; }
  for (let j = 0; j < 57; j++) w[j] -= lr * (gw[j] / n + l2 * w[j]); b -= lr * gb / n;
}
const score = r => sig(b + Z(r).reduce((s, v, j) => s + v * w[j], 0));
const auc = (sc, lab) => { const idx = sc.map((s, i) => [s, lab[i]]).sort((p, q) => p[0] - q[0]); let r = 0, np = 0, nn = 0, sumR = 0;
  for (let i = 0; i < idx.length; ) { let k = i; while (k < idx.length && idx[k][0] === idx[i][0]) k++; const avg = (i + k + 1) / 2; for (let m = i; m < k; m++) if (idx[m][1]) { sumR += avg; np++; } else nn++; i = k; }
  return (sumR - np * (np + 1) / 2) / (np * nn); };
const metrics = (pred, lab) => { let tp = 0, fp = 0, tn = 0, fn = 0; pred.forEach((p, i) => p ? (lab[i] ? tp++ : fp++) : (lab[i] ? fn++ : tn++));
  return { tp, fp, tn, fn, accuracy: (tp + tn) / lab.length, precision: tp / (tp + fp || 1), recall: tp / (tp + fn || 1), f1: 2 * tp / (2 * tp + fp + fn) }; };
const yte = test.map(r => r[57]), ste = test.map(score);
const model = { ...metrics(ste.map(s => s >= .5), yte), auc: auc(ste, yte), train_rows: train.length, test_rows: test.length,
  train_accuracy: metrics(train.map(r => score(r) >= .5), ytr).accuracy };
// baselines a reader can follow: always "not spam", and one rule
const best1 = features.map((f, j) => ({ j, f, m: metrics(test.map(r => r[j] > 0), yte) })).sort((p, q) => q.m.accuracy - p.m.accuracy)[0];
const baselines = { majority: metrics(yte.map(() => false), yte), one_rule: { feature: best1.f.label, ...best1.m } };
features.forEach((f, j) => { f.coef = w[j]; });

// self-checks
assert.ok(Math.abs(auc([.1, .4, .35, .8], [0, 0, 1, 1]) - .75) < 1e-9, 'auc');
assert.ok(model.auc > .9 && model.accuracy > baselines.one_rule.accuracy, 'model should beat the one-rule baseline');
assert.equal(spam.length + ham.length, clean.length);

// ---------- feature explorer: share of emails per log1p bin, zeros kept as their own bin ----------
const hist = FEATS.map((f, j) => { const vals = clean.map(r => Math.log1p(r[j])), max = Math.max(...vals) || 1, B = 14;
  const edges = Array.from({ length: B + 1 }, (_, k) => k / B * max);
  const count = grp => { const c = new Array(B + 1).fill(0); grp.forEach(r => { const v = Math.log1p(r[j]); if (v === 0) c[0]++; else c[Math.min(B, 1 + Math.floor(v / max * B))]++; }); return c.map(x => +(x / grp.length).toFixed(4)); };
  return { labels: ['0', ...edges.slice(0, B).map((e, k) => `${(Math.expm1(e)).toPrecision(2)}–${(Math.expm1(edges[k + 1])).toPrecision(2)}`)], spam: count(spam), ham: count(ham) }; });

const r4 = x => +x.toFixed(4);
const out = {
  generated: new Date().toISOString().slice(0, 10), issues,
  counts: { spam: spam.length, ham: ham.length, spam_raw: rows.filter(isSpam).length, ham_raw: rows.filter(r => !isSpam(r)).length },
  features: features.map(f => ({ ...f, spam_present: r4(f.spam_present), ham_present: r4(f.ham_present), spam_mean: r4(f.spam_mean), ham_mean: r4(f.ham_mean), corr: r4(f.corr), coef: r4(f.coef) })),
  model: Object.fromEntries(Object.entries(model).map(([k, v]) => [k, typeof v === 'number' && !Number.isInteger(v) ? r4(v) : v])),
  baselines, hist, test: { score: ste.map(s => +s.toFixed(4)), label: yte },
};
writeFileSync('dashboard-data.js', `// generated by analysis.mjs on ${out.generated}\nconst DATA = ${JSON.stringify(out)};\n`);
console.log(JSON.stringify({ issues, counts: out.counts, model: out.model, baselines: { majority: r4(baselines.majority.accuracy), one_rule: [baselines.one_rule.feature, r4(baselines.one_rule.accuracy)] } }, null, 1));
const top = (k, dir) => features.slice().sort((p, q) => dir * (q.spam_present - q.ham_present) - dir * (p.spam_present - p.ham_present)).slice(0, k).map(f => `${f.label} ${Math.round(f.spam_present * 100)}% vs ${Math.round(f.ham_present * 100)}%`);
console.log('spam tells:', top(8, 1).join(' | ')); console.log('ham tells:', top(8, -1).join(' | '));
console.log('top coefs:', features.slice().sort((p, q) => q.coef - p.coef).slice(0, 6).map(f => `${f.label} ${f.coef.toFixed(2)}`).join(', '), '| most negative:', features.slice().sort((p, q) => p.coef - q.coef).slice(0, 6).map(f => `${f.label} ${f.coef.toFixed(2)}`).join(', '));
const caps = ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'].map(c => features.find(f => f.feature === c));
console.log('caps:', caps.map(f => `${f.label} spam ${f.spam_mean.toFixed(1)} ham ${f.ham_mean.toFixed(1)}`).join(' | '));
