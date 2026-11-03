// Checks for Section 6 and Appendix B of hbrr_paper.tex.
//
// Part 1 (Table tab:hetsep) needs the HBRR engine of hbrr_screen.js loaded first.
// Parts 2-3 (logistic instances) are standalone and enumerate all n! permutations.
//
// Usage in a browser console or node:
//   runSep(4, [1e-2,1e-3,1e-4,1e-5,1e-6], 'RR');   // Table tab:hetsep, one row
//   checkOneEpoch();        // inequality (eq:general) at random states
//   checkInjection();       // one-epoch injection from x*, scaled by L a^2 n^3 sigma*^2
//   checkIntermediate();    // eq:am, Step 4, Lemma gradE, Lemma drift
//   checkBias();            // bounds on b and v (Step 5), cancellation-free recursion
//   checkLinear();          // Step 5: split identity and the full linear-term bound
//   checkSDJS();            // Lemma B.1, one deterministic step

(function (G) {
  'use strict';

  // ---------- Part 1: heterogeneous instance (eq:hetinst), translated to x* = (1,1) ----------
  function mk(n, mu, g) {
    const As = [], bs = [], m = n / 2;
    for (let i = 0; i < n; i++) {
      if (i < m) { As.push([1, 0, 0, mu]); bs.push([1 + g, mu]); }
      else { As.push([0, 0, 0, mu]); bs.push([-g, mu]); }
    }
    return G.HBRR.customInstance(As, bs);
  }
  function runSep(n, epsRel, lab, extra) {
    const inst = mk(n, 0.01, 0.3);
    const spec = G.HBRR.SPECS.find(s => s.label === lab);
    const o = G.HBRR.optimize(inst, spec, Object.assign({ epsRel, uMin: -9, uMax: 0.5, wMin: -5.5, cap: 2e6 }, extra || {}));
    const pick = f => epsRel.map(e => o['eps' + e] && f(o['eps' + e]));
    return {
      n, lab,
      T: pick(r => r.epochs),
      nalpha: pick(r => n * r.alpha),
      oneMinusBeta: pick(r => 1 - r.beta),
      critical: pick(r => 2 * Math.sqrt(0.01 * r.alpha)),
    };
  }

  // ---------- Part 2: regularized logistic instances ----------
  function rng(s) { return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  function gauss(R) { const u = R() || 1e-12, v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function perms(n) {
    const out = [], a = [...Array(n).keys()];
    (function p(k) {
      if (k === n) { out.push(a.slice()); return; }
      for (let i = k; i < n; i++) { [a[k], a[i]] = [a[i], a[k]]; p(k + 1); [a[k], a[i]] = [a[i], a[k]]; }
    })(0);
    return out;
  }
  const sig = z => 1 / (1 + Math.exp(-z));
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const nrm = a => Math.hypot(a[0], a[1]);

  // f_i(x) = log(1+exp(a_i.x)) - y_i a_i.x + lam/2 |x|^2,  mu = lam,  L = max |a_i|^2/4 + lam
  function inst(n, seed, lam) {
    const R = rng(seed), A = [], Y = [];
    for (let i = 0; i < n; i++) { A.push([2 * gauss(R), 2 * gauss(R)]); Y.push(R() < 0.5 ? 0 : 1); }
    const gi = (i, x) => { const z = A[i][0] * x[0] + A[i][1] * x[1], c = sig(z) - Y[i]; return [c * A[i][0] + lam * x[0], c * A[i][1] + lam * x[1]]; };
    const g = x => { const s = [0, 0]; for (let i = 0; i < n; i++) { const q = gi(i, x); s[0] += q[0]; s[1] += q[1]; } return [s[0] / n, s[1] / n]; };
    const H = x => {
      const h = [[lam, 0], [0, lam]];
      for (let i = 0; i < n; i++) { const z = A[i][0] * x[0] + A[i][1] * x[1], w = sig(z) * (1 - sig(z)) / n; for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) h[a][b] += w * A[i][a] * A[i][b]; }
      return h;
    };
    let x = [0, 0];
    for (let it = 0; it < 100; it++) {
      const q = g(x), h = H(x), det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
      x = [x[0] - (h[1][1] * q[0] - h[0][1] * q[1]) / det, x[1] - (-h[1][0] * q[0] + h[0][0] * q[1]) / det];
    }
    let L = 0; for (let i = 0; i < n; i++) L = Math.max(L, (A[i][0] ** 2 + A[i][1] ** 2) / 4 + lam);
    let s2 = 0; for (let i = 0; i < n; i++) { const q = gi(i, x); s2 += q[0] ** 2 + q[1] ** 2; } s2 /= n;
    // gradient difference grad f_i(u+d) - grad f_i(u) without cancellation
    const chi = (i, u, d) => {
      const z = A[i][0] * u[0] + A[i][1] * u[1], dz = A[i][0] * d[0] + A[i][1] * d[1], a = z + dz;
      const ds = z <= 0 ? Math.exp(z) * Math.expm1(dz) / ((1 + Math.exp(a)) * (1 + Math.exp(z)))
        : -Math.exp(-a) * Math.expm1(-dz) / ((1 + Math.exp(-a)) * (1 + Math.exp(-z)));
      return [ds * A[i][0] + lam * d[0], ds * A[i][1] + lam * d[1]];
    };
    return { n, gi, g, chi, xs: x, L, mu: lam, s2 };
  }

  // f(y) - f* as an integral of the gradient (avoids cancellation)
  const GL = [[-0.9602898565, 0.1012285363], [-0.7966664774, 0.2223810345], [-0.5255324099, 0.3137066459], [-0.1834346425, 0.3626837834],
    [0.1834346425, 0.3626837834], [0.5255324099, 0.3137066459], [0.7966664774, 0.2223810345], [0.9602898565, 0.1012285363]];
  function gap(I, y) {
    const d = sub(y, I.xs); let s = 0;
    for (const [t0, w] of GL) { const t = (t0 + 1) / 2, q = I.g([I.xs[0] + t * d[0], I.xs[1] + t * d[1]]); s += w / 2 * (q[0] * d[0] + q[1] * d[1]); }
    return s;
  }
  function params(I, alpha) {
    const r = Math.sqrt(I.mu * alpha);
    return { r, beta: (1 - r) / (1 + r), th: (1 + r) / (1 - r), vt: 2 * Math.sqrt(I.mu) / (1 - r) };
  }
  function lyap(I, alpha, P, y, x) {
    const dx = sub(x, y), w = [dx[0] / Math.sqrt(alpha) + P.vt * (x[0] - I.xs[0]), dx[1] / Math.sqrt(alpha) + P.vt * (x[1] - I.xs[1])];
    return { E: P.th * gap(I, y) + (dx[0] ** 2 + dx[1] ** 2) / (4 * alpha) + (w[0] ** 2 + w[1] ** 2) / 4, w, dx };
  }
  function hatTraj(I, alpha, P, y, x) {
    let hp = y.slice(), hc = x.slice(); const hat = [x.slice()];
    for (let j = 0; j < I.n; j++) { const q = I.g(hc); const hn = [hc[0] + P.beta * (hc[0] - hp[0]) - alpha * q[0], hc[1] + P.beta * (hc[1] - hp[1]) - alpha * q[1]]; hp = hc; hc = hn; hat.push(hc.slice()); }
    return hat;
  }
  // deviations d_0..d_n via the recursion of Lemma B.3 (no cancellation)
  function deviations(I, alpha, P, hat, p) {
    let dm = [0, 0], dc = [0, 0]; const d = [[0, 0]];
    for (let j = 0; j < I.n; j++) {
      const u = hat[j], ch = I.chi(p[j], u, dc), gi = I.gi(p[j], u), gf = I.g(u);
      const z = [ch[0] + gi[0] - gf[0], ch[1] + gi[1] - gf[1]];
      const dn = [(1 + P.beta) * dc[0] - P.beta * dm[0] - alpha * z[0], (1 + P.beta) * dc[1] - P.beta * dm[1] - alpha * z[1]];
      dm = dc; dc = dn; d.push(dc);
    }
    return d;
  }
  function sigmaHat(I, x) {
    let s = 0; const gx = I.g(x);
    for (let i = 0; i < I.n; i++) { const q = I.gi(i, x); s += (q[0] - gx[0]) ** 2 + (q[1] - gx[1]) ** 2; }
    return s / I.n;
  }
  function randomState(I, alpha, R, spread) {
    const sc = Math.pow(10, -4 + 4 * R());
    const x = [I.xs[0] + sc * gauss(R), I.xs[1] + sc * gauss(R)];
    const f1 = spread ? Math.pow(10, 2 * R() - 1) : 1, f2 = spread ? Math.pow(10, 2 * R() - 1) : 1;
    const y = [x[0] + sc * Math.sqrt(alpha) * gauss(R) * f1, x[1] + sc * Math.sqrt(alpha) * gauss(R) * f2];
    return { x, y };
  }
  const C0 = 1e-3, C1 = 40;

  // E[E(S_n)] for one epoch from (y,x)
  function epochE(I, alpha, P, y, x, Ps) {
    const hat = hatTraj(I, alpha, P, y, x); let EE = 0;
    for (const p of Ps) {
      const d = deviations(I, alpha, P, hat, p);
      const yy = [hat[I.n - 1][0] + d[I.n - 1][0], hat[I.n - 1][1] + d[I.n - 1][1]], xx = [hat[I.n][0] + d[I.n][0], hat[I.n][1] + d[I.n][1]];
      EE += lyap(I, alpha, P, yy, xx).E;
    }
    return EE / Ps.length;
  }

  // ---------- Part 3: the checks ----------
  function checkOneEpoch() {
    const worst = {};
    for (const n of [3, 4, 5, 6]) {
      const Ps = perms(n);
      for (let seed = 1; seed <= 6; seed++) {
        const I = inst(n, seed * 97 + n, 0.05);
        for (const k of [1, 10, 100]) {
          const alpha = k * C0 * I.mu / (I.L ** 2 * n * n), P = params(I, alpha), R = rng(seed * 13 + k + n);
          for (let s = 0; s < 40; s++) {
            const { x, y } = randomState(I, alpha, R, false);
            const E0 = lyap(I, alpha, P, y, x).E, EE = epochE(I, alpha, P, y, x, Ps);
            const ratio = (EE - (1 - n * P.r / 8) * E0) / (C1 * I.L * alpha * alpha * n ** 3 * I.s2);
            worst[k] = Math.max(worst[k] ?? -Infinity, ratio);
          }
        }
      }
    }
    return worst; // must be <= 1 at k = 1
  }

  function checkInjection() {
    const out = {};
    for (const n of [3, 4, 5, 6, 7]) {
      const Ps = perms(n); let mn = Infinity, mx = 0;
      for (let seed = 1; seed <= 8; seed++) {
        const I = inst(n, seed * 97 + n, 0.05);
        for (const k of [1, 0.1, 0.01]) {
          const alpha = k * C0 * I.mu / (I.L ** 2 * n * n), P = params(I, alpha);
          const v = epochE(I, alpha, P, I.xs.slice(), I.xs.slice(), Ps) / (I.L * alpha * alpha * n ** 3 * I.s2);
          mn = Math.min(mn, v); mx = Math.max(mx, v);
        }
      }
      out[n] = [mn, mx];
    }
    return out;
  }

  function forEachState(ks, cb) {
    for (const n of [3, 4, 5, 6]) {
      const Ps = perms(n);
      for (let seed = 1; seed <= 6; seed++) {
        const I = inst(n, seed * 97 + n, 0.05);
        for (const k of ks) {
          const alpha = k * C0 * I.mu / (I.L ** 2 * n * n), P = params(I, alpha), R = rng(seed * 31 + n + k * 7);
          for (let s = 0; s < 25; s++) { const { x, y } = randomState(I, alpha, R, true); cb(I, alpha, P, Ps, y, x); }
        }
      }
    }
  }

  function checkIntermediate() {
    const W = { A2: 0, step4: 0, gradG: 0, gradX: 0 }; let driftOK = true;
    forEachState([1, 0.1, 0.01], (I, alpha, P, Ps, y, x) => {
      const n = I.n, L = I.L, mu = I.mu, e = lyap(I, alpha, P, y, x), Delta = 8 * Math.sqrt(alpha * e.E);
      const hat = hatTraj(I, alpha, P, y, x);
      for (let j = 0; j <= n; j++) if (nrm(sub(hat[j], hat[0])) > j * Delta * (1 + 1e-9) + 1e-15) driftOK = false;
      const sh2 = sigmaHat(I, x), a2 = new Array(n + 1).fill(0); let vv = 0;
      for (const p of Ps) {
        const d = deviations(I, alpha, P, hat, p);
        for (let j = 0; j <= n; j++) a2[j] += (d[j][0] ** 2 + d[j][1] ** 2) / Ps.length;
        const dv = sub(d[n], d[n - 1]); vv += (dv[0] ** 2 + dv[1] ** 2) / Ps.length;
      }
      const A2 = Math.max(...a2);
      W.A2 = Math.max(W.A2, A2 / (4.1 * sh2 * alpha ** 2 * n ** 3 + 0.23 * alpha ** 2 * L ** 2 * Delta ** 2 * n ** 6));
      W.step4 = Math.max(W.step4, (3 / (4 * alpha)) * vv / (6 * mu * sh2 * alpha ** 2 * n ** 3 + 2.25 * alpha * L ** 2 * n ** 4 * Delta ** 2 + 2.25 * alpha * L ** 2 * n ** 2 * A2));
      const gy = I.g(y), Gv = [P.th * gy[0] + P.vt / 2 * e.w[0], P.th * gy[1] + P.vt / 2 * e.w[1]];
      const gx = [e.dx[0] / (2 * alpha) + (1 / (2 * Math.sqrt(alpha)) + P.vt / 2) * e.w[0], e.dx[1] / (2 * alpha) + (1 / (2 * Math.sqrt(alpha)) + P.vt / 2) * e.w[1]];
      W.gradG = Math.max(W.gradG, nrm(Gv) / (3.8 * L / Math.sqrt(mu) * Math.sqrt(e.E)));
      W.gradX = Math.max(W.gradX, nrm(gx) / (2.2 * Math.sqrt(e.E / alpha)));
    });
    return { W, driftOK }; // all ratios must be <= 1
  }

  function checkBias() {
    const W = { v: 0, b: 0 };
    forEachState([1, 0.1, 0.01], (I, alpha, P, Ps, y, x) => {
      const n = I.n, L = I.L, E0 = lyap(I, alpha, P, y, x).E, Delta = 8 * Math.sqrt(alpha * E0), sh = Math.sqrt(sigmaHat(I, x));
      const hat = hatTraj(I, alpha, P, y, x); let b = [0, 0], v = [0, 0];
      for (const p of Ps) {
        const d = deviations(I, alpha, P, hat, p);
        b[0] += d[n - 1][0]; b[1] += d[n - 1][1]; v[0] += d[n][0] - d[n - 1][0]; v[1] += d[n][1] - d[n - 1][1];
      }
      b = [b[0] / Ps.length, b[1] / Ps.length]; v = [v[0] / Ps.length, v[1] / Ps.length];
      W.v = Math.max(W.v, nrm(v) / (0.58 * alpha ** 2 * L * sh * n ** 2.5 + 0.085 * alpha ** 2 * L ** 2 * Delta * n ** 4));
      W.b = Math.max(W.b, nrm(b) / (0.36 * alpha ** 2 * L * sh * n ** 3.5 + 0.035 * alpha ** 2 * L ** 2 * Delta * n ** 5));
    });
    return W; // must be <= 1
  }

  // Step 5: the split identity <grad E, E delta> = <G,b> + <grad_x E, v> and the full linear-term bound
  function checkLinear() {
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
    let worst = 0, worstIdent = 0;
    forEachState([1, 0.1, 0.01], (I, alpha, P, Ps, y, x) => {
      const n = I.n, E0 = lyap(I, alpha, P, y, x).E, hat = hatTraj(I, alpha, P, y, x);
      let b = [0, 0], dn = [0, 0];
      for (const p of Ps) {
        const d = deviations(I, alpha, P, hat, p);
        b[0] += d[n - 1][0] / Ps.length; b[1] += d[n - 1][1] / Ps.length; dn[0] += d[n][0] / Ps.length; dn[1] += d[n][1] / Ps.length;
      }
      const v = sub(dn, b), e = lyap(I, alpha, P, hat[n - 1], hat[n]), gy = I.g(hat[n - 1]), sa = Math.sqrt(alpha);
      const gradY = [P.th * gy[0] - e.dx[0] / (2 * alpha) - e.w[0] / (2 * sa), P.th * gy[1] - e.dx[1] / (2 * alpha) - e.w[1] / (2 * sa)];
      const gradX = [e.dx[0] / (2 * alpha) + (1 / (2 * sa) + P.vt / 2) * e.w[0], e.dx[1] / (2 * alpha) + (1 / (2 * sa) + P.vt / 2) * e.w[1]];
      const lin = dot(gradY, b) + dot(gradX, dn), split = dot([gradY[0] + gradX[0], gradY[1] + gradX[1]], b) + dot(gradX, v);
      worstIdent = Math.max(worstIdent, Math.abs(lin - split) / (Math.abs(lin) + 1e-300));
      const om = I.L * sa * n / Math.sqrt(I.mu), nr = n * P.r;
      const rhs = (nr / 40) * E0 + 1.05 * I.L * alpha * alpha * n ** 3 * sigmaHat(I, x) + (1.5 * om ** 3 + 1.07 * om ** 4) * nr * E0;
      worst = Math.max(worst, Math.abs(lin) / rhs);
    });
    return { worst, worstIdent }; // worst must be <= 1; worstIdent is rounding error
  }

  // Lemma B.1: one deterministic heavy-ball step contracts E by 1/(1+r/4)
  function checkSDJS() {
    let worst = 0, cnt = 0;
    for (let seed = 1; seed <= 20; seed++) {
      const I = inst(4, 1000 + seed, 0.05);
      for (const k of [1, 0.1, 0.01]) {
        const alpha = k * I.mu / (16 * I.L ** 2), P = params(I, alpha), R = rng(seed * 7 + k * 1e3);
        for (let s = 0; s < 200; s++) {
          const sc = Math.pow(10, -3 + 4 * R());
          const x = [I.xs[0] + sc * gauss(R), I.xs[1] + sc * gauss(R)];
          const y = [x[0] + sc * Math.sqrt(alpha) * gauss(R) * Math.pow(10, 2 * R() - 1), x[1] + sc * Math.sqrt(alpha) * gauss(R) * Math.pow(10, 2 * R() - 1)];
          const q = I.g(x), xp = [x[0] + P.beta * (x[0] - y[0]) - alpha * q[0], x[1] + P.beta * (x[1] - y[1]) - alpha * q[1]];
          worst = Math.max(worst, (1 + P.r / 4) * lyap(I, alpha, P, x, xp).E / lyap(I, alpha, P, y, x).E); cnt++;
        }
      }
    }
    return { cnt, worst }; // must be <= 1
  }

  Object.assign(G, { mk, runSep, checkOneEpoch, checkInjection, checkIntermediate, checkBias, checkLinear, checkSDJS });
})(typeof globalThis !== 'undefined' ? globalThis : this);
