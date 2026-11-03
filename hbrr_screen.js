// HBRR / NAG-RR / RR epoch-operator screening on quadratic finite sums.
// Exact expectations by enumerating all n! permutations (or all n^n i.i.d. sequences).
// Error state w = (x - x*, x_prev - x*);  one step  w+ = M_i w + c_i;  one epoch  w+ = Phi_pi w + r_pi.
// Mean-square dynamics:  S' = E[Phi S Phi^T] + E[Phi m r^T] + E[r m^T Phi^T] + E[r r^T],  m' = E[Phi] m + E[r].
// Per-epoch MSE contraction = rho(E[Phi (x) Phi]).
// Runs in any JS engine: call  HBRR.run(cfg)  (async; progress in HBRR.status, results in HBRR.results).
(function (G) {
  'use strict';
  const d = 2, D = 2 * d, DD = D * D;

  function lin(a, b, k) { const r = []; for (let i = 0; i < k; i++) r.push(a + (b - a) * i / (k - 1)); return r; }
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }

  // ---------- dense linear algebra ----------
  function matmul(A, B, N) {
    const C = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) {
      const a = A[i * N + k]; if (a === 0) continue;
      for (let j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
    }
    return C;
  }
  function fro(A) { let s = 0; for (let i = 0; i < A.length; i++) s += A[i] * A[i]; return Math.sqrt(s); }
  // Gelfand formula rho = lim ||A^m||^{1/m} via repeated normalized squaring (robust for non-normal A).
  function specRad(A, N) {
    let B = Float64Array.from(A); let s = fro(B); if (s === 0) return 0;
    for (let i = 0; i < B.length; i++) B[i] /= s;
    let ell = Math.log(s), m = 1;
    for (let k = 0; k < 45; k++) {
      B = matmul(B, B, N); s = fro(B); if (s === 0) return 0;
      for (let i = 0; i < B.length; i++) B[i] /= s;
      ell = 2 * ell + Math.log(s); m *= 2;
    }
    return Math.exp(ell / m);
  }
  function solve(A, b, N) { // Gaussian elimination, partial pivoting
    const M = Float64Array.from(A), x = Float64Array.from(b);
    for (let c = 0; c < N; c++) {
      let p = c; for (let r = c + 1; r < N; r++) if (Math.abs(M[r * N + c]) > Math.abs(M[p * N + c])) p = r;
      if (p !== c) { for (let j = 0; j < N; j++) { const t = M[c * N + j]; M[c * N + j] = M[p * N + j]; M[p * N + j] = t; } const t = x[c]; x[c] = x[p]; x[p] = t; }
      const piv = M[c * N + c];
      for (let r = c + 1; r < N; r++) {
        const f = M[r * N + c] / piv; if (f === 0) continue;
        for (let j = c; j < N; j++) M[r * N + j] -= f * M[c * N + j];
        x[r] -= f * x[c];
      }
    }
    for (let r = N - 1; r >= 0; r--) { let s = x[r]; for (let j = r + 1; j < N; j++) s -= M[r * N + j] * x[j]; x[r] = s / M[r * N + r]; }
    return x;
  }
  function isPosDef(X) { // Cholesky on the symmetric part of a D x D matrix stored row-major
    const A = new Float64Array(DD);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) A[i * D + j] = (X[i * D + j] + X[j * D + i]) / 2;
    for (let j = 0; j < D; j++) {
      let s = A[j * D + j]; for (let k = 0; k < j; k++) s -= A[j * D + k] ** 2;
      if (!(s > 0)) return false;
      const ljj = Math.sqrt(s); A[j * D + j] = ljj;
      for (let i = j + 1; i < D; i++) { let t = A[i * D + j]; for (let k = 0; k < j; k++) t -= A[i * D + k] * A[j * D + k]; A[i * D + j] = t / ljj; }
    }
    return true;
  }
  function iMinus(A, N) { const B = new Float64Array(N * N); for (let i = 0; i < N * N; i++) B[i] = -A[i]; for (let i = 0; i < N; i++) B[i * N + i] += 1; return B; }

  // ---------- instances: f_i(x) = 1/2 (a_i^T x - y_i)^2,  a_i = (g1, eps*g2), scaled so L = max_i ||a_i||^2 = 1 ----------
  function makeInstance(n, kappaTarget, seed) {
    const r = mulberry32(seed * 1000003 + n * 7919);
    const g1 = [], g2 = [], y = [];
    for (let i = 0; i < n; i++) { g1.push(gauss(r)); g2.push(gauss(r)); y.push(gauss(r)); }
    function stats(eps) {
      let p = 0, q = 0, s = 0, L = 0;
      for (let i = 0; i < n; i++) { const a0 = g1[i], a1 = eps * g2[i]; p += a0 * a0; q += a0 * a1; s += a1 * a1; L = Math.max(L, a0 * a0 + a1 * a1); }
      p /= n; q /= n; s /= n;
      const lmin = (p + s) / 2 - Math.sqrt(((p - s) / 2) ** 2 + q * q), lmax = (p + s) / 2 + Math.sqrt(((p - s) / 2) ** 2 + q * q);
      return { L, lmin, lmax };
    }
    let eps = 1;
    const st1 = stats(1);
    if (st1.L / st1.lmin < kappaTarget) {
      let lo = -8, hi = 0;
      for (let it = 0; it < 80; it++) { const mid = (lo + hi) / 2, st = stats(10 ** mid); if (st.L / st.lmin > kappaTarget) lo = mid; else hi = mid; }
      eps = 10 ** ((lo + hi) / 2);
    }
    const st = stats(eps), L = st.L;
    const A = [], b = [], Abar = new Float64Array(d * d), bbar = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      const a = [g1[i], eps * g2[i]];
      const Ai = new Float64Array(d * d), bi = new Float64Array(d);
      for (let p = 0; p < d; p++) { bi[p] = a[p] * y[i] / L; for (let q = 0; q < d; q++) Ai[p * d + q] = a[p] * a[q] / L; }
      A.push(Ai); b.push(bi);
      for (let k = 0; k < d * d; k++) Abar[k] += Ai[k] / n;
      for (let k = 0; k < d; k++) bbar[k] += bi[k] / n;
    }
    const xstar = solve(Abar, bbar, d);
    const gstar = A.map((Ai, i) => { const g = new Float64Array(d); for (let p = 0; p < d; p++) { let s = -b[i][p]; for (let q = 0; q < d; q++) s += Ai[p * d + q] * xstar[q]; g[p] = s; } return g; });
    return { n, A, b, Abar, xstar, gstar, mu: st.lmin / L, kappa: L / st.lmin, kappaBar: st.lmax / st.lmin, E0: xstar.reduce((s, v) => s + v * v, 0) };
  }
  function fullBatchInstance(inst) { // every component replaced by the average: deterministic full-gradient reference
    const o = Object.assign({}, inst);
    o.A = inst.A.map(() => inst.Abar); o.gstar = inst.A.map(() => new Float64Array(d));
    return o;
  }

  // ---------- one-step maps ----------
  function buildSteps(inst, method, alpha, beta) {
    const Ms = [], cs = [];
    for (let i = 0; i < inst.n; i++) {
      const M = new Float64Array(DD), Ai = inst.A[i];
      for (let p = 0; p < d; p++) for (let q = 0; q < d; q++) {
        const I = p === q ? 1 : 0, a = Ai[p * d + q];
        if (method === 'NAG') { const B = I - alpha * a; M[p * D + q] = (1 + beta) * B; M[p * D + d + q] = -beta * B; }
        else { M[p * D + q] = (1 + beta) * I - alpha * a; M[p * D + d + q] = -beta * I; }
      }
      for (let p = 0; p < d; p++) M[(d + p) * D + p] = 1;
      const c = new Float64Array(D); for (let p = 0; p < d; p++) c[p] = -alpha * inst.gstar[i][p];
      Ms.push(M); cs.push(c);
    }
    return { Ms, cs };
  }

  // ---------- exact epoch moments: sampling 'RR' (all perms), 'IID' (all n^n sequences), 'FIX' (order 0..n-1) ----------
  function moments(Ms, cs, n, sampling) {
    const T = new Float64Array(DD * DD), K = new Float64Array(DD * D), Q = new Float64Array(DD), EP = new Float64Array(DD), Er = new Float64Array(D);
    const Ph = [], R = []; for (let k = 0; k <= n; k++) { Ph.push(new Float64Array(DD)); R.push(new Float64Array(D)); }
    for (let i = 0; i < D; i++) Ph[0][i * D + i] = 1;
    const used = new Uint8Array(n); let cnt = 0;
    function leaf(P, r) {
      cnt++;
      for (let i = 0; i < D; i++) for (let k = 0; k < D; k++) {
        const Pik = P[i * D + k]; if (Pik === 0) continue;
        for (let j = 0; j < D; j++) {
          const base = (i * D + j) * DD + k * D;
          for (let l = 0; l < D; l++) T[base + l] += Pik * P[j * D + l];
          K[(i * D + j) * D + k] += Pik * r[j];
        }
      }
      for (let i = 0; i < D; i++) { Er[i] += r[i]; for (let j = 0; j < D; j++) { Q[i * D + j] += r[i] * r[j]; EP[i * D + j] += P[i * D + j]; } }
    }
    function rec(k) {
      if (k === n) { leaf(Ph[n], R[n]); return; }
      const P0 = Ph[k], r0 = R[k], P1 = Ph[k + 1], r1 = R[k + 1];
      for (let i = 0; i < n; i++) {
        if (sampling === 'RR' && used[i]) continue;
        if (sampling === 'FIX' && i !== k) continue;
        used[i] = 1;
        const M = Ms[i], c = cs[i];
        for (let p = 0; p < D; p++) {
          for (let q = 0; q < D; q++) { let s = 0; for (let t = 0; t < D; t++) s += M[p * D + t] * P0[t * D + q]; P1[p * D + q] = s; }
          let s = c[p]; for (let t = 0; t < D; t++) s += M[p * D + t] * r0[t]; r1[p] = s;
        }
        rec(k + 1); used[i] = 0;
      }
    }
    rec(0);
    for (const a of [T, K, Q, EP, Er]) for (let i = 0; i < a.length; i++) a[i] /= cnt;
    return { T, K, Q, EP, Er };
  }
  function cross(K, m) { // vec(E[Phi m r^T] + E[r m^T Phi^T])
    const h = new Float64Array(DD);
    for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) {
      let s = 0; for (let k = 0; k < D; k++) s += (K[(i * D + j) * D + k] + K[(j * D + i) * D + k]) * m[k];
      h[i * D + j] = s;
    }
    return h;
  }
  function xErr(S) { let e = 0; for (let i = 0; i < d; i++) e += S[i * D + i]; return e; }

  function evaluate(inst, method, sampling, alpha, beta) {
    const { Ms, cs } = buildSteps(inst, method, alpha, beta);
    const mo = moments(Ms, cs, inst.n, sampling);
    const rho = specRad(mo.T, DD);
    // with residuals, the MSE also carries the mean's mode (rate rho(E Phi) <= sqrt(rho(T))), so the affine rate is the max
    const rhoEff = Math.max(rho, specRad(mo.EP, D));
    // Mean-square stability, decided algebraically: T(X) = E[Phi X Phi^T] is a positive map on the PSD cone, so
    // rho(T) < 1 iff the solution of X = T(X) + I is positive definite. (The Gelfand estimate rho above is not used
    // for this: it overshoots 1 for stable but nearly defective maps, e.g. critically damped HB with beta -> 1.)
    // rho(E Phi)^2 <= rho(T), so the mean is then stable too.
    let errInf = Infinity;
    const eye = new Float64Array(DD); for (let i = 0; i < D; i++) eye[i * D + i] = 1;
    const stable = isPosDef(solve(iMinus(mo.T, DD), eye, DD));
    if (stable) {
      const mInf = solve(iMinus(mo.EP, D), mo.Er, D);
      const h = cross(mo.K, mInf); for (let i = 0; i < DD; i++) h[i] += mo.Q[i];
      const v = Math.abs(xErr(solve(iMinus(mo.T, DD), h, DD)));
      if (isFinite(v)) errInf = v;
    }
    return { rho, rhoEff, errInf, mo };
  }
  function initState(inst) {
    const w0 = new Float64Array(D); for (let p = 0; p < d; p++) { w0[p] = -inst.xstar[p]; w0[d + p] = -inst.xstar[p]; } // x0 = 0, v0 = 0
    const S = new Float64Array(DD); for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) S[i * D + j] = w0[i] * w0[j];
    return { S, m: w0 };
  }
  function stepMoments(mo, S, m) {
    const S1 = new Float64Array(DD), h = cross(mo.K, m);
    for (let a = 0; a < DD; a++) { let s = h[a] + mo.Q[a]; for (let b = 0; b < DD; b++) s += mo.T[a * DD + b] * S[b]; S1[a] = s; }
    const m1 = new Float64Array(D);
    for (let i = 0; i < D; i++) { let s = mo.Er[i]; for (let k = 0; k < D; k++) s += mo.EP[i * D + k] * m[k]; m1[i] = s; }
    return { S: S1, m: m1 };
  }
  // Epochs until E||x_t - x*||^2 <= eps and stays there, by stepping the affine moment map z = (vec S, m, 1) -> G z.
  // (Repeated squaring of G is inaccurate when beta -> 1: the HB maps become nearly defective.)
  // "Stays": after the last crossing, the error must remain below eps for max(200, t, 5*mem) further epochs, where
  // mem = 1/(n(1-beta)) is the momentum memory in epochs: for beta -> 1 the second moment can dip below eps and then
  // grow by many orders of magnitude (non-normal transient) before it finally decays.
  // Returns Infinity if not reached within cap epochs (used as a pruning budget).
  function exactEpochs(inst, mo, eps, cap, mem) {
    const win = t => Math.max(200, t, 5 * (mem || 0));
    const Z = DD + D + 1, G = new Float64Array(Z * Z);
    for (let a = 0; a < DD; a++) {
      for (let b = 0; b < DD; b++) G[a * Z + b] = mo.T[a * DD + b];
      const i = Math.floor(a / D), j = a % D;
      for (let k = 0; k < D; k++) G[a * Z + DD + k] = mo.K[(i * D + j) * D + k] + mo.K[(j * D + i) * D + k];
      G[a * Z + Z - 1] = mo.Q[a];
    }
    for (let i = 0; i < D; i++) { for (let k = 0; k < D; k++) G[(DD + i) * Z + DD + k] = mo.EP[i * D + k]; G[(DD + i) * Z + Z - 1] = mo.Er[i]; }
    G[Z * Z - 1] = 1;
    const { S, m } = initState(inst);
    let z = new Float64Array(Z), y = new Float64Array(Z); z.set(S, 0); z.set(m, DD); z[Z - 1] = 1;
    let last = -1;
    for (let t = 0; t <= cap + win(cap); t++) {
      if (xErr(z) > eps) { last = t; if (t > cap) return Infinity; }
      else if (t - last > win(last)) return last + 1;
      for (let i = 0; i < Z; i++) { let s = 0; const o = i * Z; for (let j = 0; j < Z; j++) s += G[o + j] * z[j]; y[i] = s; }
      const tmp = z; z = y; y = tmp;
    }
    return Infinity;
  }
  // rho(T) by normalized power iteration on S -> T S from S = I (T maps the PSD cone into itself); used to verify specRad
  function rateByStepping(T, steps) {
    let S = new Float64Array(DD), S1 = new Float64Array(DD); for (let i = 0; i < D; i++) S[i * D + i] = 1;
    let logn = 0, half = 0;
    for (let t = 1; t <= steps; t++) {
      let nn = 0;
      for (let i = 0; i < DD; i++) { let s = 0; for (let j = 0; j < DD; j++) s += T[i * DD + j] * S[j]; S1[i] = s; nn += s * s; }
      nn = Math.sqrt(nn); if (nn === 0) return 0;
      for (let i = 0; i < DD; i++) S[i] = S1[i] / nn;
      logn += Math.log(nn); if (t === steps / 2) half = logn;
    }
    return Math.exp((logn - half) / (steps / 2));
  }

  // ---------- parameter search ----------
  function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
  function optimize(inst, spec, cfg) {
    const cache = new Map();
    const oneD = spec.beta0;
    const f = (u, w) => {
      u = clamp(u, cfg.uMin, cfg.uMax); w = oneD ? 0 : clamp(w, cfg.wMin, 0);
      const key = u.toFixed(5) + ',' + w.toFixed(5);
      let e = cache.get(key);
      if (!e) {
        const r = evaluate(inst, spec.method, spec.sampling, 10 ** u, 1 - 10 ** w);
        e = { u, w, rho: r.rho, rhoEff: r.rhoEff, errInf: r.errInf, mo: r.mo, ep: {}, epCap: {} };
        cache.set(key, e);
      }
      return e;
    };
    // epoch count, computed lazily with the current best as budget (a capped Infinity is still "worse than best")
    const bestEp = {};
    const epochsOf = (e, er) => {
      const eps = er * inst.E0;
      if (!(e.errInf < eps)) return Infinity; // unstable points never settle below eps in exactEpochs
      const budget = Math.min(cfg.cap, bestEp[er] === undefined ? cfg.cap : bestEp[er]);
      if (e.ep[er] !== undefined && (e.ep[er] < Infinity || e.epCap[er] >= budget)) return e.ep[er];
      const v = exactEpochs(inst, e.mo, eps, budget, 10 ** (-e.w) / inst.n);
      e.ep[er] = v; e.epCap[er] = budget;
      if (v < (bestEp[er] === undefined ? Infinity : bestEp[er])) bestEp[er] = v;
      return v;
    };
    for (const u of cfg.uGrid) for (const w of (oneD ? [0] : cfg.wGrid)) f(u, w);
    const objectives = [{ name: 'rate', val: e => (e.rho < 1 ? e.rho : Infinity) }];
    // exact epochs until the expected squared error stays below eps (transient growth of non-normal maps included)
    for (const er of cfg.epsRel) objectives.push({ name: 'eps' + er, val: e => epochsOf(e, er) });
    const out = {};
    for (const ob of objectives) {
      let best = null, bv = Infinity;
      // scan the grid fastest-rate first (points with too large a residual return Infinity at once), so the budget tightens early
      const order = [...cache.values()].sort((a, b) => a.rhoEff - b.rhoEff);
      for (const e of order) { const v = ob.val(e); if (v < bv) { bv = v; best = e; } }
      if (!best) { out[ob.name] = null; continue; }
      let u = best.u, w = best.w, su = 0.25, sw = 0.25;
      for (let it = 0; it < 300 && su > 0.003; it++) {
        const cands = oneD ? [[u + su, w], [u - su, w]] :
          [[u + su, w], [u - su, w], [u, w + sw], [u, w - sw], [u + su, w + sw], [u - su, w - sw], [u + su, w - sw], [u - su, w + sw]];
        let moved = false;
        for (const [cu, cw] of cands) { const e = f(cu, cw), v = ob.val(e); if (v < bv) { bv = v; u = e.u; w = e.w; moved = true; } }
        if (!moved) { su /= 2; sw /= 2; }
      }
      const e = f(u, w), rec = { alpha: 10 ** u, beta: 1 - 10 ** w, rho: e.rho, rhoEff: e.rhoEff, errInf: e.errInf, value: bv };
      if (ob.name === 'rate') {
        rec.N = -1 / Math.log(e.rho); // epochs per e-fold of the MSE (homogeneous part)
        rec.rhoCheck = rateByStepping(e.mo.T, 20000);
      } else rec.epochs = bv;
      out[ob.name] = rec;
    }
    out.evals = cache.size;
    return out;
  }

  // Tuning used for Tables 2-4: T(eps) is rugged in (alpha, beta), so a single pattern search can stop early.
  // Stage 1: 29 x 17 grid, pattern searches from the 5 best grid points. Stage 2: 25 x 25 local grid of half-width 0.6
  // decades around the stage-1 optimum, pattern searches from its 10 best points. Returns the better of the two stages.
  function tuneRefined(inst, spec, er) {
    const eps = er * inst.E0, oneD = spec.beta0, cache = new Map(); let best = Infinity;
    const f = (u, w) => {
      u = clamp(u, -6, 1); w = oneD ? 0 : Math.min(0, w); const key = u.toFixed(5) + ',' + w.toFixed(5); let e = cache.get(key);
      if (!e) { const r = evaluate(inst, spec.method, spec.sampling, 10 ** u, 1 - 10 ** w); e = { u, w, rhoEff: r.rhoEff, errInf: r.errInf, mo: r.mo }; cache.set(key, e); }
      const budget = Math.min(3e6, best === Infinity ? 3e6 : Math.ceil(1.5 * best));
      if (e.ep === undefined || (e.ep === Infinity && e.cap < budget)) {
        e.ep = !(e.errInf < eps) ? Infinity : exactEpochs(inst, e.mo, eps, budget, 10 ** (-e.w) / inst.n); e.cap = budget;
        if (e.ep < best) best = e.ep;
      }
      return e;
    };
    const pattern = (s0, h0, hMin) => {
      let u = s0.u, w = s0.w, bv = s0.ep, su = h0;
      while (su > hMin) {
        const c = oneD ? [[u + su, w], [u - su, w]] : [[u + su, w], [u - su, w], [u, w + su], [u, w - su], [u + su, w + su], [u - su, w - su], [u + su, w - su], [u - su, w + su]];
        let moved = false;
        for (const [cu, cw] of c) { const e = f(cu, cw); if (e.ep < bv) { bv = e.ep; u = e.u; w = e.w; moved = true; } }
        if (!moved) su /= 2;
      }
      return { v: bv, u, w };
    };
    const top = k => [...cache.values()].filter(e => e.ep < Infinity).sort((a, b) => a.ep - b.ep).slice(0, k);
    const grid = []; for (const u of lin(-6, 1, 29)) for (const w of (oneD ? [0] : lin(-4, 0, 17))) grid.push([u, w]);
    grid.map(([u, w]) => cache.get(u.toFixed(5) + ',' + w.toFixed(5)) || f(u, w));
    let gb = { v: Infinity };
    for (const s0 of top(5)) { const p = pattern(s0, 0.25, 0.003); if (p.v < gb.v) gb = p; }
    if (!oneD && gb.v < Infinity) {
      const c0 = gb; for (let i = 0; i < 25; i++) for (let k = 0; k < 25; k++) f(c0.u - 0.6 + 1.2 * i / 24, c0.w - 0.6 + 1.2 * k / 24);
      for (const s0 of top(10)) { const p = pattern(s0, 0.05, 0.002); if (p.v < gb.v) gb = p; }
    }
    return { epochs: gb.v, alpha: 10 ** gb.u, beta: 1 - 10 ** gb.w };
  }

  // Exact epochs per e-fold N = -1/ln rho(T) for Table 4 (bottom): rho(T) < lam iff X = T(X)/lam + I has a positive
  // definite solution (T is positive on the PSD cone), so rho is located by bisection on log10(1 - lam).
  function rateExact(T) {
    const below = lam => { const B = new Float64Array(DD * DD); for (let i = 0; i < DD * DD; i++) B[i] = -T[i] / lam; for (let i = 0; i < DD; i++) B[i * DD + i] += 1;
      const eye = new Float64Array(DD); for (let i = 0; i < D; i++) eye[i * D + i] = 1; const X = solve(B, eye, DD); return X.every(isFinite) && isPosDef(X); };
    if (!below(1)) return Infinity;
    let lo = -14, hi = 0; for (let it = 0; it < 50; it++) { const mid = (lo + hi) / 2; if (below(1 - 10 ** mid)) lo = mid; else hi = mid; }
    return -1 / Math.log(1 - 10 ** lo);
  }
  // minimise N over (alpha, beta): 29 x 21 grid (log10(1-beta) in [-5, 0]), pattern searches from the 5 best points,
  // then a 15 x 15 local grid of half-width 0.4 decades and pattern searches from its 5 best points
  function tuneRate(inst, spec) {
    const oneD = spec.beta0, cache = new Map();
    const f = (u, w) => { u = clamp(u, -6, 1); w = oneD ? 0 : clamp(w, -5, 0); const key = u.toFixed(5) + ',' + w.toFixed(5); let e = cache.get(key);
      if (!e) { const r = evaluate(inst, spec.method, spec.sampling, 10 ** u, 1 - 10 ** w); e = { u, w, N: rateExact(r.mo.T) }; cache.set(key, e); } return e; };
    for (const u of lin(-6, 1, 29)) for (const w of (oneD ? [0] : lin(-5, 0, 21))) f(u, w);
    const pattern = (s0, h0, hMin) => { let u = s0.u, w = s0.w, bv = s0.N, su = h0;
      while (su > hMin) { const c = oneD ? [[u + su, w], [u - su, w]] : [[u + su, w], [u - su, w], [u, w + su], [u, w - su], [u + su, w + su], [u - su, w - su], [u + su, w - su], [u - su, w + su]];
        let moved = false; for (const [cu, cw] of c) { const e = f(cu, cw); if (e.N < bv) { bv = e.N; u = e.u; w = e.w; moved = true; } } if (!moved) su /= 2; }
      return { v: bv, u, w }; };
    const top = k => [...cache.values()].filter(e => e.N < Infinity).sort((a, b) => a.N - b.N).slice(0, k);
    let gb = { v: Infinity }; for (const s0 of top(5)) { const p = pattern(s0, 0.25, 0.002); if (p.v < gb.v) gb = p; }
    if (!oneD) { const c0 = gb; for (let i = 0; i < 15; i++) for (let k = 0; k < 15; k++) f(c0.u - 0.4 + 0.8 * i / 14, c0.w - 0.4 + 0.8 * k / 14);
      for (const s0 of top(5)) { const p = pattern(s0, 0.05, 0.001); if (p.v < gb.v) gb = p; } }
    return { N: gb.v, alpha: 10 ** gb.u, beta: 1 - 10 ** gb.w };
  }

  const SPECS = [
    { label: 'RR', method: 'HB', sampling: 'RR', beta0: true },
    { label: 'HBRR', method: 'HB', sampling: 'RR' },
    { label: 'NAG-RR', method: 'NAG', sampling: 'RR' },
    { label: 'SGD-iid', method: 'HB', sampling: 'IID', beta0: true, iid: true },
    { label: 'SHB-iid', method: 'HB', sampling: 'IID', iid: true },
    { label: 'GD-full', method: 'HB', sampling: 'FIX', beta0: true, full: true },
    { label: 'HB-full', method: 'HB', sampling: 'FIX', full: true },
  ];

  // ---------- Monte Carlo validation of the moment recursion ----------
  function mcCheck(method, alpha, beta, n, kappa, epochs, runs) {
    const inst = makeInstance(n, kappa, 1);
    const r = mulberry32(42);
    let acc = 0;
    for (let run = 0; run < runs; run++) {
      let x = new Float64Array(d), xp = new Float64Array(d);
      for (let t = 0; t < epochs; t++) {
        const perm = [...Array(n).keys()];
        for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
        for (const i of perm) {
          const y = new Float64Array(d);
          for (let p = 0; p < d; p++) y[p] = method === 'NAG' ? x[p] + beta * (x[p] - xp[p]) : x[p];
          const g = new Float64Array(d);
          for (let p = 0; p < d; p++) { let s = -inst.b[i][p]; for (let q = 0; q < d; q++) s += inst.A[i][p * d + q] * y[q]; g[p] = s; }
          const xn = new Float64Array(d);
          for (let p = 0; p < d; p++) xn[p] = method === 'NAG' ? y[p] - alpha * g[p] : x[p] - alpha * g[p] + beta * (x[p] - xp[p]);
          xp = x; x = xn;
        }
      }
      for (let p = 0; p < d; p++) acc += (x[p] - inst.xstar[p]) ** 2;
    }
    const mc = acc / runs;
    const mo = evaluate(inst, method, 'RR', alpha, beta).mo;
    let { S, m } = initState(inst);
    for (let t = 0; t < epochs; t++) ({ S, m } = stepMoments(mo, S, m));
    return { method, alpha, beta, n, kappa, epochs, runs, mc, exact: xErr(S), relDiff: Math.abs(mc - xErr(S)) / xErr(S) };
  }

  function makeCfg(user) {
    const cfg = Object.assign({
      ns: [4, 5, 6, 7], kappas: [10, 30, 100, 300, 1000], seeds: [1, 2, 3], epsRel: [1e-3, 1e-6], iidMaxN: 4, iidKappaMax: 100,
      uMin: -6, uMax: 1, wMin: -4, cap: 3000000,
    }, user || {});
    cfg.uGrid = lin(cfg.uMin, cfg.uMax, 29); cfg.wGrid = lin(cfg.wMin, 0, 17);
    return cfg;
  }
  // instance from explicit component Hessians A_i (d x d, row-major) and linear terms b_i:  f_i(x) = 1/2 x^T A_i x - b_i^T x
  function customInstance(As, bs) {
    const n = As.length, Abar = new Float64Array(d * d), bbar = new Float64Array(d);
    let L = 0;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < d * d; k++) Abar[k] += As[i][k] / n;
      for (let k = 0; k < d; k++) bbar[k] += bs[i][k] / n;
      const p = As[i][0], q = As[i][1], s = As[i][3];
      L = Math.max(L, (p + s) / 2 + Math.sqrt(((p - s) / 2) ** 2 + q * q));
    }
    const A = As.map(a => Float64Array.from(a, v => v / L)), b = bs.map(v => Float64Array.from(v, w => w / L));
    for (let k = 0; k < d * d; k++) Abar[k] /= L;
    for (let k = 0; k < d; k++) bbar[k] /= L;
    const p = Abar[0], q = Abar[1], s = Abar[3];
    const lmin = (p + s) / 2 - Math.sqrt(((p - s) / 2) ** 2 + q * q), lmax = (p + s) / 2 + Math.sqrt(((p - s) / 2) ** 2 + q * q);
    const xstar = solve(Abar, bbar, d);
    const gstar = A.map((Ai, i) => { const g = new Float64Array(d); for (let r = 0; r < d; r++) { let t = -b[i][r]; for (let c = 0; c < d; c++) t += Ai[r * d + c] * xstar[c]; g[r] = t; } return g; });
    return { n, A, b, Abar, xstar, gstar, mu: lmin, kappa: 1 / lmin, kappaBar: lmax / lmin, E0: xstar.reduce((t, v) => t + v * v, 0) };
  }

  const H = G.HBRR = { status: 'idle', results: [], mcCheck, makeInstance, customInstance, evaluate, SPECS, tuneRefined, rateExact, tuneRate,
    optimize: (inst, spec, user) => optimize(inst, spec, makeCfg(user)) };
  H.run = async function (user) {
    const cfg = makeCfg(user);
    H.results = []; H.cfg = cfg; const t0 = Date.now();
    for (const n of cfg.ns) for (const kappa of cfg.kappas) for (const seed of cfg.seeds) {
      const inst = makeInstance(n, kappa, seed), fb = fullBatchInstance(inst);
      for (const spec of SPECS) {
        if (spec.iid && (n > cfg.iidMaxN || kappa > cfg.iidKappaMax)) continue;
        H.status = `n=${n} kappa=${kappa} seed=${seed} ${spec.label}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`;
        await new Promise(res => setTimeout(res, 0));
        const o = optimize(spec.full ? fb : inst, spec, cfg);
        H.results.push({ n, kappaTarget: kappa, seed, kappa: inst.kappa, kappaBar: inst.kappaBar, label: spec.label, ...o });
      }
    }
    H.status = `done in ${((Date.now() - t0) / 1000).toFixed(0)}s`;
    return H.results;
  };
})(typeof window !== 'undefined' ? window : globalThis);
