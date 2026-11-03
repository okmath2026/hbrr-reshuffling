// Monte Carlo experiments for HBRR at large n (Section "Large-n Monte Carlo experiments").
// Least-squares components f_i(x) = 1/2 (a_i^T x - y_i)^2 in R^d, rescaled to L = max_i |a_i|^2 = 1.
// Runs in a browser or in Node.js; exports into the global object G.
(function (G) {
  function rng(seed) { let s = (seed >>> 0) || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  function solve(H, b) {
    const d = b.length, M = H.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < d; c++) {
      let p = c; for (let r = c + 1; r < d; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < d; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= d; k++) M[r][k] -= f * M[c][k]; }
    }
    return M.map((row, i) => row[d] / row[i]);
  }
  function lambdaMin(H) { // inverse power iteration
    const d = H.length; let v = new Array(d).fill(1), lam = 0;
    for (let it = 0; it < 100; it++) {
      const w = solve(H, v), nw = Math.hypot(...w); v = w.map(x => x / nw);
      lam = 0; for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += H[i][j] * v[j]; lam += s * v[i]; }
    }
    return lam;
  }
  // a_i = D g_i with D_jj = scale^{-j/(2(d-1))}, y_i = a_i^T x_true + 0.5 N(0,1)
  function makeLS(n, d, scale, seed) {
    const r = rng(seed), D = [], xt = [];
    for (let j = 0; j < d; j++) { D.push(Math.pow(scale, -0.5 * j / (d - 1))); xt.push(gauss(r)); }
    const A = [], y = [];
    for (let i = 0; i < n; i++) {
      const a = []; for (let j = 0; j < d; j++) a.push(D[j] * gauss(r));
      let s = 0; for (let j = 0; j < d; j++) s += a[j] * xt[j]; A.push(a); y.push(s + 0.5 * gauss(r));
    }
    let L = 0; for (const a of A) L = Math.max(L, a.reduce((s, x) => s + x * x, 0));
    const sc = 1 / Math.sqrt(L); for (const a of A) for (let j = 0; j < d; j++) a[j] *= sc; for (let i = 0; i < n; i++) y[i] *= sc;
    const H = [], bb = new Array(d).fill(0); for (let i = 0; i < d; i++) H.push(new Array(d).fill(0));
    for (let i = 0; i < n; i++) for (let p = 0; p < d; p++) { bb[p] += A[i][p] * y[i] / n; for (let q = 0; q < d; q++) H[p][q] += A[i][p] * A[i][q] / n; }
    const xs = solve(H, bb), mu = lambdaMin(H);
    let s2 = 0;
    for (let i = 0; i < n; i++) { let res = -y[i]; for (let j = 0; j < d; j++) res += A[i][j] * xs[j]; for (let j = 0; j < d; j++) s2 += (A[i][j] * res) ** 2; }
    const Af = new Float64Array(n * d); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) Af[i * d + j] = A[i][j];
    return { n, d, A: Af, y: Float64Array.from(y), xs: Float64Array.from(xs), mu, L: 1, sig2: s2 / n, kappa: 1 / mu };
  }
  // instance whose realised kappa = L/mu matches the target (bisection on the scale)
  function makeK(n, d, target, seed) {
    let lo = 1, hi = 10 * target;
    for (let it = 0; it < 25; it++) { const mid = Math.sqrt(lo * hi); if (makeLS(n, d, mid, seed).kappa > target) hi = mid; else lo = mid; }
    return makeLS(n, d, Math.sqrt(lo * hi), seed);
  }
  // C independent chains of HBRR(alpha, beta) from x0 with zero velocity; rec(t, mean over chains of |x^(t)-x*|^2)
  function runMC(I, alpha, beta, x0, epochs, C, seed, rec) {
    const { n, d, A, y, xs } = I, r = rng(seed), perm = new Int32Array(n);
    const X = new Float64Array(C * d), V = new Float64Array(C * d);
    for (let c = 0; c < C; c++) for (let j = 0; j < d; j++) X[c * d + j] = x0[j];
    for (let t = 0; t < epochs; t++) {
      for (let c = 0; c < C; c++) {
        for (let i = 0; i < n; i++) perm[i] = i;
        for (let i = n - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)), q = perm[i]; perm[i] = perm[k]; perm[k] = q; }
        const o = c * d;
        for (let k = 0; k < n; k++) {
          const ai = perm[k] * d; let res = -y[perm[k]];
          for (let j = 0; j < d; j++) res += A[ai + j] * X[o + j];
          for (let j = 0; j < d; j++) { const v = beta * V[o + j] - alpha * A[ai + j] * res; V[o + j] = v; X[o + j] += v; }
        }
      }
      let m = 0; for (let c = 0; c < C; c++) for (let j = 0; j < d; j++) m += (X[c * d + j] - xs[j]) ** 2;
      rec(t + 1, m / C);
    }
  }
  // stationary floor: start at x*, burn in 8 relaxation times, then average over K epochs
  function floorMC(I, alpha, beta, rate, C, seed, mult) {
    const mix = Math.ceil(1 / (rate * I.n)), B = 8 * mix, K = Math.max(Math.round(40 * mix * (mult || 1)), 400);
    let acc = 0, cnt = 0;
    runMC(I, alpha, beta, I.xs, B + K, C, seed, (t, m) => { if (t > B) { acc += m; cnt++; } });
    return acc / cnt;
  }
  // critically damped parameters of Section 5 for a given alpha
  function critical(alpha, mu) { const r = Math.sqrt(mu * alpha); return { alpha, beta: (1 - r) / (1 + r), rate: r }; }
  Object.assign(G, { MCrng: rng, makeLS, makeK, runMC, floorMC, critical });
})(typeof window !== 'undefined' ? window : globalThis);
