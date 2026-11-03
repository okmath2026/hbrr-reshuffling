// Stress tests of Conjecture 7.1 with strongly heterogeneous Hessians (Section "Large-n Monte Carlo experiments")
// and the switching example of Section "Open problems".
// Runs in a browser or in Node.js; exports into the global object G.
(function (G) {
  function rng(seed) { let s = (seed >>> 0) || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
  // A_i = u_i u_i^T in R^3, |u_i| = 1 = L, u_i = (c cos phi_i, c sin phi_i, +-sqrt(mu)), c = sqrt(1-mu):
  // Abar = diag((1-mu)/2, (1-mu)/2, mu), and every A_i differs from Abar by a matrix of norm about L/2.
  function mkInst(n, kappa) {
    const mu = 1 / kappa, s = Math.sqrt(mu), c = Math.sqrt(1 - mu), U = [];
    for (let i = 0; i < n; i++) { const ph = 2 * Math.PI * (i + 0.5) / n, sg = ((i >> 1) & 1) ? 1 : -1; U.push([c * Math.cos(ph), c * Math.sin(ph), sg * s]); }
    const Ab = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const u of U) for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) Ab[p][q] += u[p] * u[q] / n;
    return { n, U: Float64Array.from(U.flat()), Ab, mu };
  }
  // critical damping with prescribed effective step gamma = alpha/(1-beta)
  function params(gam, mu) {
    let a = 4 * gam * gam * mu;
    for (let it = 0; it < 50; it++) { const r = Math.sqrt(a * mu); a = Math.pow(2 * Math.sqrt(mu) * gam / (1 + r), 2); }
    const r = Math.sqrt(a * mu); return { alpha: a, beta: (1 - r) / (1 + r), r };
  }
  function shuffle(perm, r) { const n = perm.length; for (let i = 0; i < n; i++) perm[i] = i; for (let i = n - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)), q = perm[i]; perm[i] = perm[k]; perm[k] = q; } }
  // one HBRR step on (e, f = previous e); grad_i(e) = A_i e + g_i, or Abar e + g_i if common
  function step(I, st, i, alpha, beta, Gn, common) {
    const U = I.U, Ab = I.Ab, o = 3 * i, e = st.e, f = st.f; let a0, a1, a2;
    if (common) { a0 = Ab[0][0] * e[0] + Ab[0][1] * e[1] + Ab[0][2] * e[2]; a1 = Ab[1][0] * e[0] + Ab[1][1] * e[1] + Ab[1][2] * e[2]; a2 = Ab[2][0] * e[0] + Ab[2][1] * e[1] + Ab[2][2] * e[2]; }
    else { const d = U[o] * e[0] + U[o + 1] * e[1] + U[o + 2] * e[2]; a0 = d * U[o]; a1 = d * U[o + 1]; a2 = d * U[o + 2]; }
    const g = Gn ? [Gn[o], Gn[o + 1], Gn[o + 2]] : [0, 0, 0];
    const n0 = (1 + beta) * e[0] - beta * f[0] - alpha * (a0 + g[0]), n1 = (1 + beta) * e[1] - beta * f[1] - alpha * (a1 + g[1]), n2 = (1 + beta) * e[2] - beta * f[2] - alpha * (a2 + g[2]);
    st.f = e; st.e = [n0, n1, n2];
  }
  // noiseless HBRR: slope of log mean |e|^2 per epoch over the second half of `relax` relaxation times, divided by -2nr
  function decay(I, gam, C, relax, seed) {
    const p = params(gam, I.mu), r = rng(seed), perm = new Int32Array(I.n), T = Math.ceil(relax / (I.n * p.r));
    const S = []; for (let c = 0; c < C; c++) { const e = [r() - 0.5, r() - 0.5, r() - 0.5]; S.push({ e, f: e.slice() }); }
    const log = [];
    for (let t = 0; t < T; t++) {
      for (const st of S) { shuffle(perm, r); for (let k = 0; k < I.n; k++) step(I, st, perm[k], p.alpha, p.beta, null, false); }
      let m = 0; for (const st of S) m += st.e[0] ** 2 + st.e[1] ** 2 + st.e[2] ** 2; log.push(Math.log(m / C));
    }
    const h = Math.floor(T / 2); return ((log[T - 1] - log[h]) / (T - 1 - h)) / (-2 * I.n * p.r);
  }
  // stationary floor with centred Gaussian g_i, divided by gamma^2 (1-beta) n^2 sigma_*^2;
  // common = true replaces every A_i by Abar (common-Hessian comparison)
  function floorH(I, gam, C, seed, common) {
    const n = I.n, p = params(gam, I.mu), r = rng(seed), perm = new Int32Array(n), Gn = new Float64Array(3 * n);
    for (let i = 0; i < 3 * n; i++) { let u = 0; while (u === 0) u = r(); Gn[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
    for (let j = 0; j < 3; j++) { let m = 0; for (let i = 0; i < n; i++) m += Gn[3 * i + j] / n; for (let i = 0; i < n; i++) Gn[3 * i + j] -= m; }
    let s2 = 0; for (let i = 0; i < 3 * n; i++) s2 += Gn[i] * Gn[i] / n;
    const mix = Math.ceil(1 / (n * p.r)), B = 8 * mix, K = Math.max(40 * mix, 200);
    let acc = 0, cnt = 0;
    for (let c = 0; c < C; c++) {
      const st = { e: [0, 0, 0], f: [0, 0, 0] };
      for (let t = 0; t < B + K; t++) {
        shuffle(perm, r); for (let k = 0; k < n; k++) step(I, st, perm[k], p.alpha, p.beta, Gn, common);
        if (t >= B) { acc += st.e[0] ** 2 + st.e[1] ** 2 + st.e[2] ** 2; cnt++; }
      }
    }
    return acc / cnt / (gam * gam * (1 - p.beta) * n * n * s2);
  }
  // scalar heavy ball, critically damped for mu (L = 1), curvature mu while moving outward and L while moving inward;
  // returns log10 of |x| + |x - x_prev|/sqrt(alpha) after 200 relaxation times
  function switching(kappa, gam) {
    const mu = 1 / kappa, p = params(gam, mu), steps = Math.ceil(200 / p.r); let x = 0, xp = -1e-3;
    for (let k = 0; k < steps; k++) { const v = x - xp, h = (x * v > 0) ? mu : 1, xn = x + p.beta * v - p.alpha * h * x; xp = x; x = xn; }
    return Math.log10(Math.abs(x) + Math.abs(x - xp) / Math.sqrt(p.alpha));
  }
  // floor with offsets correlated with the Hessians, g_i = (A_i - Abar) w (shifted component minimizers);
  // normalised as in floorH, common = true replaces every A_i by Abar
  function floorCorr(I, gam, C, seed, common, w) {
    const n = I.n, U = I.U, Ab = I.Ab, p = params(gam, I.mu), r = rng(seed), perm = new Int32Array(n), Gn = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
      const o = 3 * i, d = U[o] * w[0] + U[o + 1] * w[1] + U[o + 2] * w[2];
      for (let a = 0; a < 3; a++) Gn[o + a] = d * U[o + a] - (Ab[a][0] * w[0] + Ab[a][1] * w[1] + Ab[a][2] * w[2]);
    }
    let s2 = 0; for (let i = 0; i < 3 * n; i++) s2 += Gn[i] * Gn[i] / n;
    const mix = Math.ceil(1 / (n * p.r)), B = 8 * mix, K = Math.max(40 * mix, 200);
    let acc = 0, cnt = 0;
    for (let c = 0; c < C; c++) {
      const st = { e: [0, 0, 0], f: [0, 0, 0] };
      for (let t = 0; t < B + K; t++) {
        shuffle(perm, r); for (let k = 0; k < n; k++) step(I, st, perm[k], p.alpha, p.beta, Gn, common);
        if (t >= B) { acc += st.e[0] ** 2 + st.e[1] ** 2 + st.e[2] ** 2; cnt++; }
      }
    }
    return acc / cnt / (gam * gam * (1 - p.beta) * n * n * s2);
  }
  // instance (5.x) of Theorem 5.3 with L = 1 at large n: x1-curvature 1 with offset +1 (i < n/2) or 0 with offset -1;
  // the x2 coordinate (curvature mu, no noise) only fixes beta. common = true uses curvature 1/2 for every component.
  function floorRRbias(n, mu, gam, C, seed, common) {
    const p = params(gam, mu), r = rng(seed), perm = new Int32Array(n), m = n / 2;
    const mix = Math.ceil(1 / (n * p.r)), B = 8 * mix, K = Math.max(40 * mix, 200);
    let acc = 0, cnt = 0;
    for (let c = 0; c < C; c++) {
      let e = 0, f = 0;
      for (let t = 0; t < B + K; t++) {
        shuffle(perm, r);
        for (let k = 0; k < n; k++) { const i = perm[k], a = common ? 0.5 : (i < m ? 1 : 0), g = i < m ? 1 : -1, en = (1 + p.beta) * e - p.beta * f - p.alpha * (a * e + g); f = e; e = en; }
        if (t >= B) { acc += e * e; cnt++; }
      }
    }
    return acc / cnt / (gam * gam * (1 - p.beta) * n * n);
  }
  Object.assign(G, { mkHetInst: mkInst, critParams: params, hetDecay: decay, hetFloor: floorH, hetFloorCorr: floorCorr, rrBiasFloor: floorRRbias, hbSwitching: switching });
})(typeof window !== 'undefined' ? window : globalThis);
