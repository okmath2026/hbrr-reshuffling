// Section "Control: averaging the RR iterates": RR with an exponential moving average of its output,
// y+ = b y + (1 - b) x+, compared with HBRR, and HBRR whose iterates are averaged in the same way (Section 6.9).
// Exact computations (no sampling).
// Runs in a browser or in Node.js; exports into the global object G.
(function (G) {
  // ---------- common Hessian (instance of Table 1): exact Tbar from Theorem 3.3 ----------
  // n = 6, A = diag(1, 1/kappa), sigma_l^2 = (0.95, 0.56), e0 = (-1, -1), eps = 2e-4.
  // method 'hb': heavy ball (alpha, b = beta); method 'ema': RR with step alpha and output EMA weight 1 - b;
  // method 'hbema': heavy ball (alpha, b = beta) with output y+ = c y + (1 - c) x+ (c = 0 gives 'hb', b = 0 gives 'ema').
  function tbarCommon(method, alpha, b, kappa, maxEp, c) {
    const n = 6, as = [1, 1 / kappa], sig = [0.95, 0.56], eps = 2e-4;
    const st = as.map((a, l) => method === 'hb' ? { a, sg: sig[l], h: 1, hp: 0, s0: -1, s1: -1, N: 0 }
      : method === 'hbema' ? { a, sg: sig[l], h: 1, hp: 0, s0: -1, s1: -1, f: 1 - c, y: -1, N: 0 }
      : { a, q: 1 - alpha * a, sg: sig[l], f: 1 - b, qm: 1, x: -1, y: -1, N: 0 });
    let last = -1;
    for (let t = 0; t < maxEp; t++) {
      let err = 0, amp = 0;
      for (const S of st) {
        let sum = 0; const blk = [];
        if (method === 'hb') {
          err += S.s0 * S.s0 + S.N; const p = 1 + b - alpha * S.a;
          for (let j = 0; j < n; j++) { blk.push(S.h); sum += S.h; const hn = p * S.h - b * S.hp; S.hp = S.h; S.h = hn; const sn = p * S.s0 - b * S.s1; S.s1 = S.s0; S.s0 = sn; }
          amp += Math.abs(S.h) + Math.abs(S.hp) + Math.abs(S.s0);
        } else if (method === 'hbema') { // noise response of the output: f_m = c f_{m-1} + (1 - c) h_m, f_0 = 1 - c
          err += S.y * S.y + S.N; const p = 1 + b - alpha * S.a;
          for (let j = 0; j < n; j++) { blk.push(S.f); sum += S.f; const hn = p * S.h - b * S.hp; S.hp = S.h; S.h = hn; S.f = c * S.f + (1 - c) * hn;
            const sn = p * S.s0 - b * S.s1; S.s1 = S.s0; S.s0 = sn; S.y = c * S.y + (1 - c) * sn; }
          amp += Math.abs(S.h) + Math.abs(S.hp) + Math.abs(S.s0) + Math.abs(S.f) + Math.abs(S.y);
        } else { // noise response of the averaged output: f_m = b f_{m-1} + (1 - b) q^m, f_0 = 1 - b
          err += S.y * S.y + S.N;
          for (let j = 0; j < n; j++) { blk.push(S.f); sum += S.f; S.qm *= S.q; S.f = b * S.f + (1 - b) * S.qm; S.x *= S.q; S.y = b * S.y + (1 - b) * S.x; }
          amp += Math.abs(S.f) + Math.abs(S.y);
        }
        let v = 0; for (const x of blk) v += (x - sum / n) ** 2;
        S.N += alpha * alpha * S.sg * v * n / (n - 1);
        if (S.N > eps) return Infinity;
      }
      if (err > eps) last = t;
      if (t > 2000 && t > 3 * last && amp < 1e-7) return last + 1;
    }
    return last < maxEp - 1 ? last + 1 : Infinity;
  }
  // tune averaged RR over (log10 alpha in [uMin, uMax], log10(1 - b) in [-5, -1]): 17 x 17 grid, pattern searches
  // from the 5 best points; the paper used uMin = -1.5, uMax = 0.29
  function tuneCommonEMA(kappa, uMin, uMax) {
    let best = Infinity; const cache = new Map();
    const f = (u, w) => {
      u = Math.min(u, Math.log10(1.98)); w = Math.min(0, w); const k = u.toFixed(4) + ',' + w.toFixed(4); if (cache.has(k)) return cache.get(k);
      const T = tbarCommon('ema', 10 ** u, 1 - 10 ** w, kappa, best === Infinity ? 600000 : Math.ceil(1.5 * best) + 10); if (T < best) best = T;
      const e = { u, w, T }; cache.set(k, e); return e;
    };
    for (let i = 0; i <= 16; i++) for (let k = 0; k <= 16; k++) f(uMin + (uMax - uMin) * i / 16, -5 + 4 * k / 16);
    let gb = { T: Infinity };
    for (const s of [...cache.values()].filter(e => e.T < Infinity).sort((a, b) => a.T - b.T).slice(0, 5)) {
      let u = s.u, w = s.w, bv = s.T, h = 0.1;
      while (h > 0.002) { let mv = false; for (const [du, dw] of [[h, 0], [-h, 0], [0, h], [0, -h], [h, h], [-h, -h], [h, -h], [-h, h]]) { const e = f(u + du, w + dw); if (e.T < bv) { bv = e.T; u = e.u; w = e.w; mv = true; } } if (!mv) h /= 2; }
      if (bv < gb.T) gb = { T: bv, u, w };
    }
    gb.Tlong = tbarCommon('ema', 10 ** gb.u, 1 - 10 ** gb.w, kappa, 4000000); gb.alpha = 10 ** gb.u; gb.oneMinusB = 10 ** gb.w;
    return gb;
  }
  // tune HBRR with averaged iterates over (u, v, w) = log10 of (alpha, 1 - beta, 1 - c): 13 x 11 x 10 grid over
  // u in [-5.5, 0.2], v in [-5, 0], w in [-5.5, -1] (v = 0 is averaged RR), pattern searches from the 5 best points
  // and from the given starts (e.g. the tuned HBRR and averaged RR with a long output memory)
  function tuneCommonHBEMA(kappa, starts) {
    let best = Infinity; const cache = new Map();
    const f = (u, v, w) => {
      u = Math.min(u, Math.log10(1.98)); v = Math.min(0, v); w = Math.min(0, w); const k = u.toFixed(4) + ',' + v.toFixed(4) + ',' + w.toFixed(4); if (cache.has(k)) return cache.get(k);
      const T = tbarCommon('hbema', 10 ** u, 1 - 10 ** v, kappa, best === Infinity ? 600000 : Math.ceil(1.5 * best) + 10, 1 - 10 ** w); if (T < best) best = T;
      const e = { u, v, w, T }; cache.set(k, e); return e;
    };
    for (let i = 0; i <= 12; i++) for (let j = 0; j <= 10; j++) for (let k = 0; k <= 9; k++) f(-5.5 + 5.7 * i / 12, -5 + 5 * j / 10, -5.5 + 4.5 * k / 9);
    const dirs = []; for (const a of [-1, 0, 1]) for (const b of [-1, 0, 1]) for (const c of [-1, 0, 1]) if (a || b || c) dirs.push([a, b, c]);
    const st = [...cache.values()].filter(e => e.T < Infinity).sort((a, b) => a.T - b.T).slice(0, 5).concat((starts || []).map(s => f(s[0], s[1], s[2])));
    let gb = { T: Infinity };
    for (const s of st) {
      let { u, v, w } = s, bv = s.T, h = 0.1;
      while (h > 0.002) { let mv = false; for (const [a, b, c] of dirs) { const e = f(u + a * h, v + b * h, w + c * h); if (e.T < bv) { bv = e.T; u = e.u; v = e.v; w = e.w; mv = true; } } if (!mv) h /= 2; }
      if (bv < gb.T) gb = { T: bv, u, v, w };
    }
    gb.alpha = 10 ** gb.u; gb.oneMinusBeta = 10 ** gb.v; gb.oneMinusC = 10 ** gb.w;
    gb.Tlong = tbarCommon('hbema', gb.alpha, 1 - gb.oneMinusBeta, kappa, 4000000, 1 - gb.oneMinusC);
    return gb;
  }

  // ---------- heterogeneous least squares (instances of Table 2, same generator as hbrr_screen.js) ----------
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function makeInstance(n, kappaTarget, seed) {
    const r = mulberry32(seed * 1000003 + n * 7919), g1 = [], g2 = [], y = [];
    for (let i = 0; i < n; i++) { g1.push(gauss(r)); g2.push(gauss(r)); y.push(gauss(r)); }
    const stats = eps => { let p = 0, q = 0, s = 0, L = 0; for (let i = 0; i < n; i++) { const a0 = g1[i], a1 = eps * g2[i]; p += a0 * a0; q += a0 * a1; s += a1 * a1; L = Math.max(L, a0 * a0 + a1 * a1); }
      p /= n; q /= n; s /= n; return { L, lmin: (p + s) / 2 - Math.sqrt(((p - s) / 2) ** 2 + q * q) }; };
    let eps = 1; const st1 = stats(1);
    if (st1.L / st1.lmin < kappaTarget) { let lo = -8, hi = 0; for (let it = 0; it < 80; it++) { const mid = (lo + hi) / 2; if (stats(10 ** mid).L / stats(10 ** mid).lmin > kappaTarget) lo = mid; else hi = mid; } eps = 10 ** ((lo + hi) / 2); }
    const st = stats(eps), L = st.L, A = [], b = [], Ab = [0, 0, 0, 0], bb = [0, 0];
    for (let i = 0; i < n; i++) { const a = [g1[i], eps * g2[i]], Ai = [a[0] * a[0] / L, a[0] * a[1] / L, a[1] * a[0] / L, a[1] * a[1] / L], bi = [a[0] * y[i] / L, a[1] * y[i] / L];
      A.push(Ai); b.push(bi); for (let k = 0; k < 4; k++) Ab[k] += Ai[k] / n; bb[0] += bi[0] / n; bb[1] += bi[1] / n; }
    const det = Ab[0] * Ab[3] - Ab[1] * Ab[2], xs = [(bb[0] * Ab[3] - Ab[1] * bb[1]) / det, (Ab[0] * bb[1] - Ab[2] * bb[0]) / det];
    return { n, A, b, xs, kappa: L / st.lmin, E0: xs[0] ** 2 + xs[1] ** 2 };
  }
  const grad0 = (I, i) => { const A = I.A[i]; return [A[0] * I.xs[0] + A[1] * I.xs[1] - I.b[i][0], A[2] * I.xs[0] + A[3] * I.xs[1] - I.b[i][1]]; };
  // affine step maps w+ = M_i w + c_i on R^4. Averaged RR: w = (e, y); heavy ball: w = (e, e_prev).
  function mapsEMA(I, alpha, b) {
    const Ms = [], cs = [];
    for (let i = 0; i < I.n; i++) {
      const A = I.A[i], g = grad0(I, i), B = [1 - alpha * A[0], -alpha * A[1], -alpha * A[2], 1 - alpha * A[3]], M = new Float64Array(16);
      M[0] = B[0]; M[1] = B[1]; M[4] = B[2]; M[5] = B[3];
      M[8] = (1 - b) * B[0]; M[9] = (1 - b) * B[1]; M[10] = b; M[12] = (1 - b) * B[2]; M[13] = (1 - b) * B[3]; M[15] = b;
      Ms.push(M); cs.push([-alpha * g[0], -alpha * g[1], -(1 - b) * alpha * g[0], -(1 - b) * alpha * g[1]]);
    }
    return { Ms, cs };
  }
  function mapsHB(I, alpha, b) {
    const Ms = [], cs = [];
    for (let i = 0; i < I.n; i++) {
      const A = I.A[i], g = grad0(I, i), M = new Float64Array(16);
      M[0] = 1 + b - alpha * A[0]; M[1] = -alpha * A[1]; M[4] = -alpha * A[2]; M[5] = 1 + b - alpha * A[3]; M[2] = -b; M[7] = -b; M[8] = 1; M[13] = 1;
      Ms.push(M); cs.push([-alpha * g[0], -alpha * g[1], 0, 0]);
    }
    return { Ms, cs };
  }
  // HBRR with averaged iterates: w = (e, e_prev, y) on R^6, y+ = c y + (1 - c) e+; the error is the block (4, 5)
  function mapsHBEMA(I, alpha, b, c) {
    const Ms = [], cs = [];
    for (let i = 0; i < I.n; i++) {
      const A = I.A[i], g = grad0(I, i), M = new Float64Array(36);
      M[0] = 1 + b - alpha * A[0]; M[1] = -alpha * A[1]; M[6] = -alpha * A[2]; M[7] = 1 + b - alpha * A[3]; M[2] = -b; M[9] = -b; M[12] = 1; M[19] = 1;
      for (let q = 0; q < 4; q++) { M[24 + q] = (1 - c) * M[q]; M[30 + q] = (1 - c) * M[6 + q]; } M[28] = c; M[35] = c;
      Ms.push(M); cs.push([-alpha * g[0], -alpha * g[1], 0, 0, -(1 - c) * alpha * g[0], -(1 - c) * alpha * g[1]]);
    }
    return { Ms, cs };
  }
  // exact epoch moments over all n! permutations; the state dimension d is that of the maps (4 or 6)
  function moments(Ms, cs, n) {
    const d = cs[0].length, d2 = d * d, T = new Float64Array(d2 * d2), K = new Float64Array(d2 * d), Q = new Float64Array(d2), EP = new Float64Array(d2), Er = new Float64Array(d), used = new Uint8Array(n); let cnt = 0;
    function rec(k, P, r) {
      if (k === n) {
        cnt++;
        for (let i = 0; i < d; i++) for (let kk = 0; kk < d; kk++) { const Pik = P[i * d + kk]; if (Pik === 0) continue;
          for (let j = 0; j < d; j++) { const base = (i * d + j) * d2 + kk * d; for (let l = 0; l < d; l++) T[base + l] += Pik * P[j * d + l]; K[(i * d + j) * d + kk] += Pik * r[j]; } }
        for (let i = 0; i < d; i++) { Er[i] += r[i]; for (let j = 0; j < d; j++) { Q[i * d + j] += r[i] * r[j]; EP[i * d + j] += P[i * d + j]; } }
        return;
      }
      for (let i = 0; i < n; i++) {
        if (used[i]) continue; used[i] = 1;
        const M = Ms[i], c = cs[i], P1 = new Float64Array(d2), r1 = new Float64Array(d);
        for (let p = 0; p < d; p++) { for (let q = 0; q < d; q++) { let s = 0; for (let t = 0; t < d; t++) s += M[p * d + t] * P[t * d + q]; P1[p * d + q] = s; }
          let s = c[p]; for (let t = 0; t < d; t++) s += M[p * d + t] * r[t]; r1[p] = s; }
        rec(k + 1, P1, r1); used[i] = 0;
      }
    }
    const P0 = new Float64Array(d2); for (let i = 0; i < d; i++) P0[i * d + i] = 1; rec(0, P0, new Float64Array(d));
    for (const a of [T, K, Q, EP, Er]) for (let i = 0; i < a.length; i++) a[i] /= cnt;
    return { T, K, Q, EP, Er, d };
  }
  function solve(A, b, N) {
    const M = Float64Array.from(A), x = Float64Array.from(b);
    for (let c = 0; c < N; c++) {
      let p = c; for (let r = c + 1; r < N; r++) if (Math.abs(M[r * N + c]) > Math.abs(M[p * N + c])) p = r;
      if (p !== c) { for (let j = 0; j < N; j++) { const t = M[c * N + j]; M[c * N + j] = M[p * N + j]; M[p * N + j] = t; } const t = x[c]; x[c] = x[p]; x[p] = t; }
      const piv = M[c * N + c]; for (let r = c + 1; r < N; r++) { const f = M[r * N + c] / piv; if (f === 0) continue; for (let j = c; j < N; j++) M[r * N + j] -= f * M[c * N + j]; x[r] -= f * x[c]; }
    }
    for (let r = N - 1; r >= 0; r--) { let s = x[r]; for (let j = r + 1; j < N; j++) s -= M[r * N + j] * x[j]; x[r] = s / M[r * N + r]; }
    return x;
  }
  const stationaryMean = mo => { const d = mo.d, C = new Float64Array(d * d); for (let i = 0; i < d * d; i++) C[i] = -mo.EP[i]; for (let i = 0; i < d; i++) C[i * d + i] += 1; return solve(C, mo.Er, d); };
  // stationary error of the block (o, o+1); Infinity if not mean-square stable
  function errInf(mo, o) {
    const d = mo.d, d2 = d * d, B = new Float64Array(d2 * d2); for (let i = 0; i < d2 * d2; i++) B[i] = -mo.T[i]; for (let i = 0; i < d2; i++) B[i * d2 + i] += 1;
    const eye = new Float64Array(d2); for (let i = 0; i < d; i++) eye[i * d + i] = 1;
    const X = solve(B, eye, d2), A = new Float64Array(d2); for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) A[i * d + j] = (X[i * d + j] + X[j * d + i]) / 2;
    for (let j = 0; j < d; j++) { let s = A[j * d + j]; for (let k = 0; k < j; k++) s -= A[j * d + k] ** 2; if (!(s > 0)) return Infinity; const l = Math.sqrt(s); A[j * d + j] = l;
      for (let i = j + 1; i < d; i++) { let t = A[i * d + j]; for (let k = 0; k < j; k++) t -= A[i * d + k] * A[j * d + k]; A[i * d + j] = t / l; } }
    const m = stationaryMean(mo), h = new Float64Array(d2);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let s = mo.Q[i * d + j]; for (let k = 0; k < d; k++) s += (mo.K[(i * d + j) * d + k] + mo.K[(j * d + i) * d + k]) * m[k]; h[i * d + j] = s; }
    const S = solve(B, h, d2), v = S[o * d + o] + S[(o + 1) * d + o + 1];
    return isFinite(v) && v >= 0 ? v : Infinity;
  }
  const bias2 = (mo, o) => { const m = stationaryMean(mo); return m[o] ** 2 + m[o + 1] ** 2; };
  // first epoch from which the exact error of block o stays <= eps (checked for 3 last + 2000 epochs)
  function tbar(I, mo, eps, cap, o) {
    const d = mo.d, d2 = d * d; let S = new Float64Array(d2), m = new Float64Array(d); const w0 = Array.from({ length: d }, (_, i) => -I.xs[i % 2]);
    for (let i = 0; i < d; i++) { m[i] = w0[i]; for (let j = 0; j < d; j++) S[i * d + j] = w0[i] * w0[j]; }
    let last = -1;
    for (let t = 0; t < cap; t++) {
      const err = S[o * d + o] + S[(o + 1) * d + o + 1]; if (!(err < 1e30)) return Infinity;
      if (err > eps) last = t; else if (t > 3 * last + 2000) return last + 1;
      const S1 = new Float64Array(d2), m1 = new Float64Array(d);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let s = mo.Q[i * d + j]; for (let k = 0; k < d; k++) s += (mo.K[(i * d + j) * d + k] + mo.K[(j * d + i) * d + k]) * m[k];
        const base = (i * d + j) * d2; for (let a = 0; a < d2; a++) s += mo.T[base + a] * S[a]; S1[i * d + j] = s; }
      for (let i = 0; i < d; i++) { let s = mo.Er[i]; for (let k = 0; k < d; k++) s += mo.EP[i * d + k] * m[k]; m1[i] = s; }
      S = S1; m = m1;
    }
    return Infinity;
  }
  // tuned T(eps) on one heterogeneous instance by pattern search in log10 of the parameters, from the given starts:
  //  'rr' [u], 'ema' [u, w], 'hb' [u, v], 'hbema' [u, v, w], with alpha = 10^u, 1 - beta = 10^v, 1 - (output weight) = 10^w
  function tuneHet(I, method, starts, eps, cap) {
    const n = I.n, cache = new Map(); let best = Infinity;
    const maps = p => method === 'rr' ? mapsEMA(I, 10 ** p[0], 0) : method === 'ema' ? mapsEMA(I, 10 ** p[0], 1 - 10 ** p[1])
      : method === 'hb' ? mapsHB(I, 10 ** p[0], 1 - 10 ** p[1]) : mapsHBEMA(I, 10 ** p[0], 1 - 10 ** p[1], 1 - 10 ** p[2]);
    const o = method === 'hb' ? 0 : method === 'hbema' ? 4 : 2;
    const f = p => {
      p = p.map((x, i) => i ? Math.min(0, x) : x); const k = p.map(x => x.toFixed(4)).join(','); if (cache.has(k)) return cache.get(k);
      // tbar certifies T only after 3 T + 2000 epochs, so the cap for T <= 1.5 best is 4.5 best + 2010
      const mm = maps(p), T = tbar(I, moments(mm.Ms, mm.cs, n), eps, best === Infinity ? cap : Math.min(cap, Math.ceil(4.5 * best) + 2010), o); if (T < best) best = T;
      const e = { p, T }; cache.set(k, e); return e;
    };
    const dim = starts[0].length, dirs = [];
    (function gen(v) { if (v.length === dim) { if (v.some(x => x)) dirs.push(v); return; } for (const s of [-1, 0, 1]) gen(v.concat(s)); })([]);
    let gb = { T: Infinity };
    for (const s of starts) {
      let e0 = f(s), p = e0.p, bv = e0.T, h = 0.2;
      while (h > 0.004) { let mv = false; for (const dv of dirs) { const e = f(p.map((x, i) => x + dv[i] * h)); if (e.T < bv) { bv = e.T; p = e.p; mv = true; } } if (!mv) h /= 2; }
      if (bv < gb.T) gb = { T: bv, p };
    }
    gb.evals = cache.size;
    return gb;
  }
  // RR step at which the RR stationary error equals 1e-6 |x*|^2; then the stationary errors of averaged RR and of HBRR
  // (same effective step, 1 - beta = w), relative to RR, and the bias share of the RR error
  function floorStudy(n, kappa, seed) {
    const I = makeInstance(n, kappa, seed), target = 1e-6 * I.E0;
    const rr = a => { const mm = mapsEMA(I, a, 0); return moments(mm.Ms, mm.cs, n); };
    let lo = -6, hi = Math.log10(1.9); for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; if (errInf(rr(10 ** mid), 2) < target) lo = mid; else hi = mid; }
    const a = 10 ** lo, mo = rr(a), F = errInf(mo, 2), out = { n, kappa: I.kappa, nAlpha: n * a, biasShare: bias2(mo, 2) / F };
    for (const w of [0.1, 0.01, 0.001]) {
      const e = mapsEMA(I, a, 1 - w), h = mapsHB(I, a * w, 1 - w);
      out['avg' + w] = errInf(moments(e.Ms, e.cs, n), 2) / F; out['hb' + w] = errInf(moments(h.Ms, h.cs, n), 0) / F;
    }
    return out;
  }
  // as floorStudy, for HBRR with averaged iterates: stationary errors relative to RR of HBRR (effective step = RR step,
  // 1 - beta = w) and of HBRR with output weight 1 - c = w (same memory) and w / 10 (ten times longer), w in {0.01, 0.001}
  function floorStudyCombined(n, kappa, seed) {
    const I = makeInstance(n, kappa, seed), target = 1e-6 * I.E0;
    const rr = a => { const mm = mapsEMA(I, a, 0); return moments(mm.Ms, mm.cs, n); };
    let lo = -6, hi = Math.log10(1.9); for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; if (errInf(rr(10 ** mid), 2) < target) lo = mid; else hi = mid; }
    const a = 10 ** lo, mo = rr(a), F = errInf(mo, 2), out = { n, kappa: I.kappa, nAlpha: n * a, biasShare: bias2(mo, 2) / F };
    for (const w of [0.01, 0.001]) {
      const e = mapsEMA(I, a, 1 - w), h = mapsHB(I, a * w, 1 - w), c1 = mapsHBEMA(I, a * w, 1 - w, 1 - w), c2 = mapsHBEMA(I, a * w, 1 - w, 1 - w / 10);
      out['avg' + w] = errInf(moments(e.Ms, e.cs, n), 2) / F; out['hb' + w] = errInf(moments(h.Ms, h.cs, n), 0) / F;
      out['hbe' + w] = errInf(moments(c1.Ms, c1.cs, n), 4) / F; out['hbe10_' + w] = errInf(moments(c2.Ms, c2.cs, n), 4) / F;
    }
    return out;
  }
  // HBRR with averaged iterates with the parameters set by the rule of Section 6.9 from muHat = fac * mu:
  // (1 - beta)^2 = r alpha muHat and 1 - c = q (1 - beta), r = 2.5, q = 0.6; only alpha is searched (41-point grid of
  // log10 alpha in [uMin, uMax], then a 1-D pattern search). Common Hessian: instance of Table 1, mu = 1 / kappa.
  function ruleCommon(kappa, fac, uMin, uMax, r = 2.5, q = 0.6) {
    const muHat = fac / kappa; let best = { T: Infinity };
    const f = u => { const a = 10 ** u, omb = Math.sqrt(r * a * muHat); if (omb >= 1) return Infinity;
      return tbarCommon('hbema', a, 1 - omb, kappa, best.T === Infinity ? 600000 : Math.ceil(1.5 * best.T) + 10, 1 - q * omb); };
    for (let i = 0; i <= 40; i++) { const u = uMin + (uMax - uMin) * i / 40, T = f(u); if (T < best.T) best = { T, u }; }
    for (let h = 0.05; h > 0.001; h /= 2) { let mv = true; while (mv) { mv = false; for (const d of [-h, h]) { const T = f(best.u + d); if (T < best.T) { best = { T, u: best.u + d }; mv = true; } } } }
    best.alpha = 10 ** best.u; return best;
  }
  // the same rule on a heterogeneous instance (mu = 1 / kappa of the instance), started from the given values of log10 alpha
  function ruleHet(n, kappaTarget, seed, fac, starts, eps, cap, r = 2.5, q = 0.6) {
    const I = makeInstance(n, kappaTarget, seed), muHat = fac / I.kappa, e = eps * I.E0, cache = new Map(); let best = Infinity;
    const f = u => { const k = u.toFixed(4); if (cache.has(k)) return cache.get(k); const a = 10 ** u, omb = Math.sqrt(r * a * muHat); let T = Infinity;
      if (omb < 1) { const mm = mapsHBEMA(I, a, 1 - omb, 1 - q * omb); T = tbar(I, moments(mm.Ms, mm.cs, n), e, best === Infinity ? cap : Math.min(cap, Math.ceil(4.5 * best) + 2010), 4); }
      if (T < best) best = T; cache.set(k, T); return T; };
    let gb = { T: Infinity };
    for (const s of starts) { let u = s, bv = f(u); for (let h = 0.2; h > 0.004; h /= 2) { let mv = true; while (mv) { mv = false; for (const d of [-h, h]) { const T = f(u + d); if (T < bv) { bv = T; u += d; mv = true; } } } } if (bv < gb.T) gb = { T: bv, u }; }
    return { n, kappa: I.kappa, fac, T: gb.T, alpha: 10 ** gb.u };
  }
  Object.assign(G, { avgRuleCommon: ruleCommon, avgRuleHet: ruleHet, avgTbarCommon: tbarCommon, avgTuneCommon: tuneCommonEMA, avgTuneCommonHBEMA: tuneCommonHBEMA, avgMakeInstance: makeInstance, avgMapsEMA: mapsEMA, avgMapsHB: mapsHB,
    avgMapsHBEMA: mapsHBEMA, avgMoments: moments, avgErrInf: errInf, avgTbar: tbar, avgTuneHet: tuneHet, avgFloorStudy: floorStudy, avgFloorStudyCombined: floorStudyCombined });
})(typeof window !== 'undefined' ? window : globalThis);
