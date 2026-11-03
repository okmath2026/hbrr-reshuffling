// Section "Real data": l2-regularized logistic regression on LIBSVM data sets, with RR, RR with averaged
// iterates, HBRR, NAG-RR and SAGA. Every method is tuned over the same kind of grid.
// Runs in a browser (optionally inside a Web Worker) or in Node.js; exports into the global object G.
// f(x) = (1/n) sum_i log(1 + exp(-y_i a_i.x)) + (lam/2)|x|^2, with a constant feature appended to every a_i.
(function (G) {
  // ---------- data ----------
  // LIBSVM text -> CSR; the larger label becomes +1, the other -1; a bias feature of value 1 is appended
  function parseLibsvm(text) {
    const lab = [], ptr = [0], idx = [], val = []; let d = 0;
    for (const line of text.split('\n')) {
      const t = line.trim(); if (!t) continue; const parts = t.split(/\s+/); lab.push(parseFloat(parts[0]));
      for (let k = 1; k < parts.length; k++) {
        const c = parts[k].indexOf(':'), j = parseInt(parts[k].slice(0, c), 10) - 1, v = parseFloat(parts[k].slice(c + 1));
        if (v !== 0) { idx.push(j); val.push(v); if (j + 1 > d) d = j + 1; }
      }
      ptr.push(idx.length);
    }
    const top = Math.max(...lab), n = lab.length, P = new Int32Array(n + 1), I = new Int32Array(idx.length + n), V = new Float64Array(idx.length + n);
    let q = 0;
    for (let i = 0; i < n; i++) { for (let k = ptr[i]; k < ptr[i + 1]; k++) { I[q] = idx[k]; V[q++] = val[k]; } I[q] = d; V[q++] = 1; P[i + 1] = q; }
    return { n, d: d + 1, y: Float64Array.from(lab, l => l === top ? 1 : -1), ptr: P, idx: I, val: V };
  }
  const logLoss = m => m > 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m));
  function margin(D, i, x) { let s = 0; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) s += D.val[k] * x[D.idx[k]]; return D.y[i] * s; }
  // scalar s_i with grad of the i-th loss term = s_i a_i
  const coef = (D, i, x) => -D.y[i] / (1 + Math.exp(margin(D, i, x)));
  function value(D, lam, x) { let f = 0; for (let i = 0; i < D.n; i++) f += logLoss(margin(D, i, x)); let r = 0; for (const v of x) r += v * v; return f / D.n + lam * r / 2; }

  // ---------- constants and the minimizer ----------
  // L = lam_max(A^T A)/(4n) + lam by power iteration; Lmax = max_i |a_i|^2/4 + lam
  function constants(D, lam) {
    let v = new Float64Array(D.d).fill(1), ev = 0;
    for (let it = 0; it < 300; it++) {
      const w = new Float64Array(D.d);
      for (let i = 0; i < D.n; i++) { let s = 0; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) s += D.val[k] * v[D.idx[k]]; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) w[D.idx[k]] += s * D.val[k]; }
      let nr = 0; for (const t of w) nr += t * t; nr = Math.sqrt(nr); ev = nr; for (let j = 0; j < D.d; j++) v[j] = w[j] / nr;
    }
    let mx = 0; for (let i = 0; i < D.n; i++) { let s = 0; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) s += D.val[k] * D.val[k]; mx = Math.max(mx, s); }
    return { L: ev / (4 * D.n) + lam, Lmax: mx / 4 + lam, mu: lam };
  }
  // largest eigenvalue of the Hessian at x (power iteration); with lam it gives the local condition number at x*
  function hessTop(D, lam, x) {
    const w = new Float64Array(D.n); for (let i = 0; i < D.n; i++) { const p = 1 / (1 + Math.exp(margin(D, i, x))); w[i] = p * (1 - p); }
    let v = new Float64Array(D.d).fill(1), ev = 0;
    for (let it = 0; it < 300; it++) {
      const u = new Float64Array(D.d);
      for (let i = 0; i < D.n; i++) { let s = 0; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) s += D.val[k] * v[D.idx[k]]; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) u[D.idx[k]] += w[i] * s * D.val[k] / D.n; }
      let nr = 0; for (let j = 0; j < D.d; j++) { u[j] += lam * v[j]; nr += u[j] * u[j]; } nr = Math.sqrt(nr); ev = nr; for (let j = 0; j < D.d; j++) v[j] = u[j] / nr;
    }
    return ev;
  }
  // damped Newton with a dense Cholesky solve (d is at most a few hundred)
  function newton(D, lam) {
    const d = D.d, x = new Float64Array(d); let f = value(D, lam, x);
    for (let it = 0; it < 60; it++) {
      const g = new Float64Array(d), H = new Float64Array(d * d);
      for (let i = 0; i < D.n; i++) {
        const m = margin(D, i, x), p = 1 / (1 + Math.exp(m)), s = -D.y[i] * p, w = p * (1 - p);
        for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) {
          const a = D.idx[k], va = D.val[k]; g[a] += s * va;
          for (let l = D.ptr[i]; l < D.ptr[i + 1]; l++) H[a * d + D.idx[l]] += w * va * D.val[l];
        }
      }
      for (let j = 0; j < d; j++) { g[j] = g[j] / D.n + lam * x[j]; for (let l = 0; l < d; l++) H[j * d + l] /= D.n; H[j * d + j] += lam; }
      for (let j = 0; j < d; j++) { // Cholesky H = R R^T in place (lower part)
        for (let l = 0; l < j; l++) { let s = H[j * d + l]; for (let k = 0; k < l; k++) s -= H[j * d + k] * H[l * d + k]; H[j * d + l] = s / H[l * d + l]; }
        let s = H[j * d + j]; for (let k = 0; k < j; k++) s -= H[j * d + k] * H[j * d + k]; H[j * d + j] = Math.sqrt(s);
      }
      const u = new Float64Array(d);
      for (let j = 0; j < d; j++) { let s = g[j]; for (let k = 0; k < j; k++) s -= H[j * d + k] * u[k]; u[j] = s / H[j * d + j]; }
      for (let j = d - 1; j >= 0; j--) { let s = u[j]; for (let k = j + 1; k < d; k++) s -= H[k * d + j] * u[k]; u[j] = s / H[j * d + j]; }
      let t = 1, xn, fn;
      for (;;) { xn = x.map((v, j) => v - t * u[j]); fn = value(D, lam, xn); if (fn <= f || t < 1e-8) break; t /= 2; }
      x.set(xn); const dec = f - fn; f = Math.min(f, fn); if (dec < 1e-17) break;
    }
    return { x, fstar: f, f0: value(D, lam, new Float64Array(d)) };
  }

  // ---------- methods ----------
  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function shuffle(p, r) { for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; } }
  // job = { method, a, b, seed, T }; returns the relative error (f - f*)/(f(0) - f*) after each of T epochs
  //  'rr'   : x+ = x - a grad f_i(x)
  //  'ema'  : RR with step a, output y+ = b y + (1 - b) x+
  //  'hb'   : HBRR in averaging form with effective step a and momentum b: z+ = z - a grad f_i(x), x+ = b x + (1 - b) z+
  //  'nag'  : x+ = y - a grad f_i(y), y+ = x+ + b (x+ - x)
  //  'saga' : SAGA with step a and uniform i.i.d. sampling, n steps per epoch
  function run(D, lam, opt, job) {
    if (job.part) return runBlocks(D, lam, opt, job);
    if (job.B > 1) return runBatch(D, lam, opt, job);
    const { n, d } = D, a = job.a, b = job.b, r = rng(job.seed * 7919 + 13), perm = Int32Array.from({ length: n }, (_, i) => i);
    const x = new Float64Array(d), z = new Float64Array(d), y = new Float64Array(d), tab = new Float64Array(job.method === 'saga' ? n : 0), mean = new Float64Array(d);
    const out = new Float64Array(job.T), den = opt.f0 - opt.fstar;
    for (let t = 0; t < job.T; t++) {
      if (job.method !== 'saga') shuffle(perm, r);
      for (let k = 0; k < n; k++) {
        if (job.method === 'rr' || job.method === 'ema') {
          const i = perm[k], s = coef(D, i, x);
          for (let j = 0; j < d; j++) x[j] *= 1 - a * lam;
          for (let q = D.ptr[i]; q < D.ptr[i + 1]; q++) x[D.idx[q]] -= a * s * D.val[q];
          if (job.method === 'ema') for (let j = 0; j < d; j++) y[j] = b * y[j] + (1 - b) * x[j];
        } else if (job.method === 'hb') {
          const i = perm[k], s = coef(D, i, x);
          for (let j = 0; j < d; j++) z[j] -= a * lam * x[j];
          for (let q = D.ptr[i]; q < D.ptr[i + 1]; q++) z[D.idx[q]] -= a * s * D.val[q];
          for (let j = 0; j < d; j++) x[j] = b * x[j] + (1 - b) * z[j];
        } else if (job.method === 'nag') { // y holds the extrapolated point
          const i = perm[k], s = coef(D, i, y);
          for (let j = 0; j < d; j++) { const xo = x[j]; x[j] = y[j] - a * lam * y[j]; z[j] = xo; }
          for (let q = D.ptr[i]; q < D.ptr[i + 1]; q++) x[D.idx[q]] -= a * s * D.val[q];
          for (let j = 0; j < d; j++) y[j] = x[j] + b * (x[j] - z[j]);
        } else {
          const i = Math.floor(r() * n), s = coef(D, i, x), ds = s - tab[i];
          for (let j = 0; j < d; j++) x[j] -= a * (mean[j] + lam * x[j]);
          for (let q = D.ptr[i]; q < D.ptr[i + 1]; q++) { x[D.idx[q]] -= a * ds * D.val[q]; mean[D.idx[q]] += ds * D.val[q] / n; }
          tab[i] = s;
        }
      }
      const e = (value(D, lam, job.method === 'ema' ? y : x) - opt.fstar) / den;
      out[t] = e;
      if (!(e < 1e3)) { out.fill(Infinity, t); break; }
    }
    return out;
  }
  // mini-batch version (job.B > 1): every epoch reshuffles the samples and takes ceil(n/B) steps with the average
  // gradient of consecutive blocks of B samples, so the gradient noise still sums to zero over an epoch
  function runBatch(D, lam, opt, job) {
    const { n, d } = D, a = job.a, b = job.b, B = job.B, nb = Math.ceil(n / B), r = rng(job.seed * 7919 + 13), perm = Int32Array.from({ length: n }, (_, i) => i);
    const x = new Float64Array(d), z = new Float64Array(d), y = new Float64Array(d), g = new Float64Array(d);
    const out = new Float64Array(job.T), den = opt.f0 - opt.fstar;
    for (let t = 0; t < job.T; t++) {
      shuffle(perm, r);
      for (let k = 0; k < nb; k++) {
        const lo = k * B, hi = Math.min(n, lo + B), w = 1 / (hi - lo), p = job.method === 'nag' ? y : x;
        g.fill(0);
        for (let q = lo; q < hi; q++) { const i = perm[q], s = coef(D, i, p) * w; for (let u = D.ptr[i]; u < D.ptr[i + 1]; u++) g[D.idx[u]] += s * D.val[u]; }
        for (let j = 0; j < d; j++) g[j] += lam * p[j];
        if (job.method === 'rr') for (let j = 0; j < d; j++) x[j] -= a * g[j];
        else if (job.method === 'ema') for (let j = 0; j < d; j++) { x[j] -= a * g[j]; y[j] = b * y[j] + (1 - b) * x[j]; }
        else if (job.method === 'hb') for (let j = 0; j < d; j++) { z[j] -= a * g[j]; x[j] = b * x[j] + (1 - b) * z[j]; }
        else for (let j = 0; j < d; j++) { const xo = x[j]; x[j] = y[j] - a * g[j]; y[j] = x[j] + b * (x[j] - xo); }
      }
      const e = (value(D, lam, job.method === 'ema' ? y : x) - opt.fstar) / den;
      out[t] = e;
      if (!(e < 1e3)) { out.fill(Infinity, t); break; }
    }
    return out;
  }
  // ---------- fixed partitions into nb blocks (components f_k = average loss of block k) ----------
  // mode 'random': random split; 'label': sorted by label (each block mostly one class);
  // 'pca': sorted by the projection on the top principal direction of the features
  const partCache = new WeakMap();
  function partition(D, mode, nb) {
    if (!partCache.has(D)) partCache.set(D, new Map());
    const cache = partCache.get(D), key = mode + ':' + nb; if (cache.has(key)) return cache.get(key);
    const n = D.n, r = rng(12345), ord = Int32Array.from({ length: n }, (_, i) => i); shuffle(ord, r);
    if (mode === 'label') { const o = Array.from(ord); o.sort((i, j) => D.y[i] - D.y[j]); ord.set(o); }
    if (mode === 'pca') {
      const d = D.d, mean = new Float64Array(d); let v = new Float64Array(d);
      for (let i = 0; i < n; i++) for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) mean[D.idx[k]] += D.val[k] / n;
      for (let j = 0; j < d; j++) v[j] = Math.sin(j + 1);
      for (let it = 0; it < 200; it++) {
        const w = new Float64Array(d); let mv = 0; for (let j = 0; j < d; j++) mv += mean[j] * v[j];
        for (let i = 0; i < n; i++) { let s = 0; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) s += D.val[k] * v[D.idx[k]]; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) w[D.idx[k]] += s * D.val[k] / n; }
        let nr = 0; for (let j = 0; j < d; j++) { w[j] -= mean[j] * mv; nr += w[j] * w[j]; } nr = Math.sqrt(nr); for (let j = 0; j < d; j++) v[j] = w[j] / nr;
      }
      const proj = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let k = D.ptr[i]; k < D.ptr[i + 1]; k++) s += D.val[k] * v[D.idx[k]]; proj[i] = s; }
      const o = Array.from(ord); o.sort((i, j) => proj[i] - proj[j]); ord.set(o);
    }
    const lo = Int32Array.from({ length: nb + 1 }, (_, k) => Math.round(k * n / nb)), P = { ord, lo };
    cache.set(key, P); return P;
  }
  // RR over the nb blocks of a fixed partition (job.part, job.nb): every epoch reshuffles the order of the blocks
  function runBlocks(D, lam, opt, job) {
    const { d } = D, a = job.a, b = job.b, nb = job.nb, { ord, lo } = partition(D, job.part, nb), r = rng(job.seed * 7919 + 13), pb = Int32Array.from({ length: nb }, (_, i) => i);
    const x = new Float64Array(d), z = new Float64Array(d), y = new Float64Array(d), g = new Float64Array(d);
    const out = new Float64Array(job.T), den = opt.f0 - opt.fstar;
    for (let t = 0; t < job.T; t++) {
      shuffle(pb, r);
      for (let k = 0; k < nb; k++) {
        const c = pb[k], w = 1 / (lo[c + 1] - lo[c]), p = job.method === 'nag' ? y : x;
        g.fill(0);
        for (let q = lo[c]; q < lo[c + 1]; q++) { const i = ord[q], s = coef(D, i, p) * w; for (let u = D.ptr[i]; u < D.ptr[i + 1]; u++) g[D.idx[u]] += s * D.val[u]; }
        for (let j = 0; j < d; j++) g[j] += lam * p[j];
        if (job.method === 'rr') for (let j = 0; j < d; j++) x[j] -= a * g[j];
        else if (job.method === 'ema') for (let j = 0; j < d; j++) { x[j] -= a * g[j]; y[j] = b * y[j] + (1 - b) * x[j]; }
        else if (job.method === 'hb') for (let j = 0; j < d; j++) { z[j] -= a * g[j]; x[j] = b * x[j] + (1 - b) * z[j]; }
        else for (let j = 0; j < d; j++) { const xo = x[j]; x[j] = y[j] - a * g[j]; y[j] = x[j] + b * (x[j] - xo); }
      }
      const e = (value(D, lam, job.method === 'ema' ? y : x) - opt.fstar) / den;
      out[t] = e;
      if (!(e < 1e3)) { out.fill(Infinity, t); break; }
    }
    return out;
  }
  // bias diagnostic for block RR started at x*: share of f at the averaged iterate in the mean error (an upper bound
  // on the bias share), averaged over seeds and over the epochs after burn
  function biasShareBlocks(P, lam, part, nb, aL, E, burn, seeds) {
    const D = P.D, d = D.d, a = aL / P.C.L, { ord, lo } = partition(D, part, nb), mean = new Float64Array(d), g = new Float64Array(d); let cnt = 0, tot = 0;
    for (let sd = 1; sd <= seeds; sd++) {
      const x = Float64Array.from(P.opt.x), r = rng(sd * 7919 + 13), pb = Int32Array.from({ length: nb }, (_, i) => i);
      for (let t = 0; t < E; t++) {
        shuffle(pb, r);
        for (let k = 0; k < nb; k++) {
          const c = pb[k], w = 1 / (lo[c + 1] - lo[c]); g.fill(0);
          for (let q = lo[c]; q < lo[c + 1]; q++) { const i = ord[q], s = coef(D, i, x) * w; for (let u = D.ptr[i]; u < D.ptr[i + 1]; u++) g[D.idx[u]] += s * D.val[u]; }
          for (let j = 0; j < d; j++) x[j] -= a * (g[j] + lam * x[j]);
        }
        if (t >= burn) { for (let j = 0; j < d; j++) mean[j] += x[j]; tot += value(D, lam, x) - P.opt.fstar; cnt++; }
      }
    }
    for (let j = 0; j < d; j++) mean[j] /= cnt;
    const den = P.opt.f0 - P.opt.fstar, fb = (value(D, lam, mean) - P.opt.fstar) / den, ft = tot / cnt / den;
    return { total: ft, ofMean: fb, share: fb / ft };
  }
  // stationary error of block RR ('rr'), averaged RR ('ema') and HBRR ('hb') started at x* with zero velocity, at
  // effective step aL/L and memory m (1 - b = 1/m): mean of (f - f*)/(f(0) - f*) over seeds and over epochs >= burn
  function floorBlocks(P, lam, part, nb, method, aL, m, E, burn, seeds) {
    const D = P.D, d = D.d, a = aL / P.C.L, b = 1 - 1 / m, { ord, lo } = partition(D, part, nb), g = new Float64Array(d), den = P.opt.f0 - P.opt.fstar; let cnt = 0, tot = 0;
    for (let sd = 1; sd <= seeds; sd++) {
      const x = Float64Array.from(P.opt.x), z = Float64Array.from(P.opt.x), y = Float64Array.from(P.opt.x), r = rng(sd * 7919 + 13), pb = Int32Array.from({ length: nb }, (_, i) => i);
      for (let t = 0; t < E; t++) {
        shuffle(pb, r);
        for (let k = 0; k < nb; k++) {
          const c = pb[k], w = 1 / (lo[c + 1] - lo[c]); g.fill(0);
          for (let q = lo[c]; q < lo[c + 1]; q++) { const i = ord[q], s = coef(D, i, x) * w; for (let u = D.ptr[i]; u < D.ptr[i + 1]; u++) g[D.idx[u]] += s * D.val[u]; }
          if (method === 'hb') for (let j = 0; j < d; j++) { z[j] -= a * (g[j] + lam * x[j]); x[j] = b * x[j] + (1 - b) * z[j]; }
          else for (let j = 0; j < d; j++) { x[j] -= a * (g[j] + lam * x[j]); if (method === 'ema') y[j] = b * y[j] + (1 - b) * x[j]; }
        }
        if (t >= burn) { tot += (value(D, lam, method === 'ema' ? y : x) - P.opt.fstar) / den; cnt++; }
      }
    }
    return tot / cnt;
  }
  // first epoch from which the error stays at most eps up to the budget, or Infinity
  function hit(err, eps) { let t = err.length; while (t > 0 && err[t - 1] <= eps) t--; return t < err.length ? t + 1 : Infinity; }
  const median = v => { const s = [...v].sort((p, q) => p - q); return s[(s.length - 1) >> 1]; };

  // grids: steps in units of 1/L; memories m = 1/(1 - b) relative to n (m = 10 is the usual b = 0.9).
  // HBRR and NAG-RR are both indexed by the effective step gamma = alpha/(1 - b) (NAG-RR runs with alpha = gamma/m).
  // When kappa > n, critical damping needs memories beyond 100 n, and the grid extends to 1000 n.
  function grids(n, L, Lmax, mu) {
    const pw = (lo, hi) => { const v = []; for (let k = lo; k <= hi; k++) v.push(2 ** k / L); return v; };
    const mem = [10, n / 30, n / 10, n / 3, n, 3 * n, 10 * n, 30 * n, 100 * n].concat(L / mu > n ? [300 * n, 1000 * n] : []).filter(m => m >= 10);
    const jobs = [];
    for (const a of pw(-14, 1)) jobs.push({ method: 'rr', a, b: 0, m: 0 });
    for (const a of pw(-12, 1)) for (const m of mem) jobs.push({ method: 'ema', a, b: 1 - 1 / m, m });
    for (const a of pw(-10, 3)) for (const m of mem) jobs.push({ method: 'hb', a, b: 1 - 1 / m, m });
    for (const a of pw(-10, 3)) for (const m of mem) jobs.push({ method: 'nag', a: a / m, b: 1 - 1 / m, m, g: a });
    for (const c of [1 / 8, 1 / 4, 1 / 2, 1, 2]) jobs.push({ method: 'saga', a: c / (3 * Lmax), b: 0, m: 0 });
    return jobs;
  }
  // mini-batch grid with nb = ceil(n/B) steps per epoch; memories relative to nb
  function gridsBatch(n, B, L) {
    const nb = Math.ceil(n / B), pw = (lo, hi) => { const v = []; for (let k = lo; k <= hi; k++) v.push(2 ** k / L); return v; };
    const mem = [...new Set([10, nb / 10, nb / 3, nb, 3 * nb, 10 * nb, 30 * nb, 100 * nb, 300 * nb].filter(m => m >= 10))];
    const jobs = [];
    for (const a of pw(-8, 5)) jobs.push({ method: 'rr', a, b: 0, m: 0, B });
    for (const a of pw(-8, 5)) for (const m of mem) jobs.push({ method: 'ema', a, b: 1 - 1 / m, m, B });
    for (const a of pw(-8, 5)) for (const m of mem) jobs.push({ method: 'hb', a, b: 1 - 1 / m, m, B });
    for (const a of pw(-8, 5)) for (const m of mem) jobs.push({ method: 'nag', a: a / m, b: 1 - 1 / m, m, g: a, B });
    return jobs;
  }
  // grid for RR over the nb blocks of a fixed partition; memories relative to nb
  function gridsBlocks(part, nb, L) {
    const pw = (lo, hi) => { const v = []; for (let k = lo; k <= hi; k++) v.push(2 ** k / L); return v; };
    const mem = [nb / 3, nb, 3 * nb, 10 * nb, 30 * nb, 100 * nb, 300 * nb, 1000 * nb, 3000 * nb].filter(m => m >= 1.5);
    const jobs = [];
    for (const a of pw(-8, 2)) jobs.push({ method: 'rr', a, b: 0, m: 0, part, nb });
    for (const a of pw(-8, 2)) for (const m of mem) jobs.push({ method: 'ema', a, b: 1 - 1 / m, m, part, nb });
    for (const a of pw(-8, 2)) for (const m of mem) jobs.push({ method: 'hb', a, b: 1 - 1 / m, m, part, nb });
    for (const a of pw(-8, 2)) for (const m of mem) jobs.push({ method: 'nag', a: a / m, b: 1 - 1 / m, m, g: a, part, nb });
    return jobs;
  }
  // results: [{ job, err: [per seed] }]; best median epochs to eps per method, and per (method, memory)
  function summarize(results, eps) {
    const best = {}, byMem = {};
    for (const R of results) {
      const h = median(R.err.map(e => hit(e, eps))), j = R.job;
      if (!best[j.method] || h < best[j.method].T) best[j.method] = { T: h, a: j.g || j.a, m: j.m };
      const k = j.method + ':' + j.m; if (!byMem[k] || h < byMem[k].T) byMem[k] = { T: h, a: j.g || j.a };
    }
    return { best, byMem };
  }
  // tuned median error after B epochs, per method
  function atBudget(results, B) {
    const best = {};
    for (const R of results) { const v = median(R.err.map(e => e[B - 1])), j = R.job; if (!best[j.method] || v < best[j.method].err) best[j.method] = { err: v, a: j.g || j.a, m: j.m }; }
    return best;
  }

  // bias diagnostic: RR started at x* with step aL/L; returns the error at the average of the epoch-boundary iterates
  // (over seeds and over epochs >= burn) and its ratio to the mean error; the ratio bounds the share of the bias
  function biasShare(P, lam, aL, E, burn, seeds) {
    const D = P.D, n = D.n, d = D.d, a = aL / P.C.L, mean = new Float64Array(d); let cnt = 0, tot = 0;
    for (let sd = 1; sd <= seeds; sd++) {
      const x = Float64Array.from(P.opt.x), p = Int32Array.from({ length: n }, (_, i) => i);
      let st = (sd * 2654435761) >>> 0; const r = () => { st ^= st << 13; st >>>= 0; st ^= st >>> 17; st ^= st << 5; st >>>= 0; return st / 4294967296; };
      for (let t = 0; t < E; t++) {
        for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const q = p[i]; p[i] = p[j]; p[j] = q; }
        for (let k = 0; k < n; k++) { const i = p[k], s = coef(D, i, x); for (let j = 0; j < d; j++) x[j] *= 1 - a * lam; for (let u = D.ptr[i]; u < D.ptr[i + 1]; u++) x[D.idx[u]] -= a * s * D.val[u]; }
        if (t >= burn) { for (let j = 0; j < d; j++) mean[j] += x[j]; tot += value(D, lam, x) - P.opt.fstar; cnt++; }
      }
    }
    for (let j = 0; j < d; j++) mean[j] /= cnt;
    const den = P.opt.f0 - P.opt.fstar, fb = (value(D, lam, mean) - P.opt.fstar) / den, ft = tot / cnt / den;
    return { total: ft, ofMean: fb, share: fb / ft };
  }
  // the paper used (aL, E, burn, seeds) = (0.0625, 12, 4, 3) and (0.5, 12, 4, 3) for mushrooms, (0.0625, 10, 3, 3) and
  // (0.004, 10, 3, 3) for a9a, (0.0625, 200, 50, 4) and (0.004, 400, 100, 4) for australian

  // ---------- browser driver ----------
  // data set text -> { D, C, opt }
  function prepare(text, lam) { const D = parseLibsvm(text); return { D, C: constants(D, lam), opt: newton(D, lam) }; }
  // runs the whole grid for one data set in nw Web Workers created from src, the source text of this file
  // B: mini-batch size, or { part, nb } for a fixed partition into blocks
  function runGrid(src, P, lam, T, seeds, nw, prog, B) {
    return runJobs(src, P, lam, B && B.part ? gridsBlocks(B.part, B.nb, P.C.L) : B > 1 ? gridsBatch(P.D.n, B, P.C.L) : grids(P.D.n, P.C.L, P.C.Lmax, P.C.mu), T, seeds, nw, prog);
  }
  // runs an explicit list of jobs (the local re-tuning of Section 6.7 uses the best grid point of each method and its
  // neighbours: steps times 1/2, 1, 2 and memories times 1/3, 1, 3)
  function runJobs(src, P, lam, grid, T, seeds, nw, prog) {
    const R = grid.map(job => ({ job, err: [] })), queue = [];
    grid.forEach((job, gi) => seeds.forEach((s, si) => queue.push({ ...job, seed: s, T, i: gi * 16 + si })));
    prog.total = queue.length; prog.done = 0;
    const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    return Promise.all(Array.from({ length: nw }, () => new Promise(res => {
      const w = new Worker(url), next = () => { const J = queue.shift(); if (!J) { w.terminate(); res(); } else w.postMessage({ cmd: 'run', jobs: [J] }); };
      w.onmessage = e => { const M = e.data; if (M.err) { R[M.i >> 4].err[M.i & 15] = Float32Array.from(M.err); prog.done++; } if (M.ready || M.done) next(); };
      w.postMessage({ cmd: 'init', D: P.D, lam, opt: { fstar: P.opt.fstar, f0: P.opt.f0 } });
    }))).then(() => R);
  }

  const API = { parseLibsvm, value, constants, hessTop, newton, run, hit, median, grids, gridsBatch, gridsBlocks, summarize, atBudget, biasShare, partition, biasShareBlocks, floorBlocks, prepare, runGrid, runJobs };
  G.LR = API;
  // Web Worker protocol: {cmd: 'init', D, lam, opt} then {cmd: 'run', jobs}; posts {i, err} per run
  if (typeof WorkerGlobalScope !== 'undefined' && G instanceof WorkerGlobalScope) {
    let S = null;
    G.onmessage = e => {
      const M = e.data;
      if (M.cmd === 'init') { S = M; G.postMessage({ ready: true }); return; }
      for (const J of M.jobs) G.postMessage({ i: J.i, err: Array.from(run(S.D, S.lam, S.opt, J)) });
      G.postMessage({ done: true });
    };
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
