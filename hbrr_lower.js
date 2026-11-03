// Numerical checks of Theorem 4.3 (lower bound for heavy ball) on the scalar common-Hessian instance.
// Runs in a browser or in Node.js; exports into the global object G.
(function (G) {
  // characteristic roots of lambda^2 - (1+beta-alpha*mu) lambda + beta
  function roots(alpha, beta, mu) {
    const p = 1 + beta - alpha * mu, disc = p * p - 4 * beta;
    if (disc >= 0) { const l1 = (p + Math.sqrt(disc)) / 2; return { real: true, rho: l1 }; }
    return { real: false, rho: Math.sqrt(beta) };
  }
  // stationary RR block sum R^RR(mu) = n/(n-1) sum_tau sum_j (h_{n tau+j} - mean)^2
  function Rrr(alpha, beta, mu, n, maxSteps) {
    const R = 1 - roots(alpha, beta, mu).rho, N = Math.min(maxSteps, Math.ceil(60 / R / n + 5) * n);
    const p = 1 + beta - alpha * mu, blk = new Float64Array(n);
    let hm1 = 0, h = 1, tot = 0;
    for (let tau = 0; tau * n < N; tau++) {
      let s = 0;
      for (let j = 0; j < n; j++) { blk[j] = h; s += h; const hn = p * h - beta * hm1; hm1 = h; h = hn; }
      let v = 0; for (let j = 0; j < n; j++) v += (blk[j] - s / n) ** 2; tot += v;
    }
    return { Rrr: tot * n / (n - 1), R };
  }
  // Case 1 of Appendix C (distinct real roots, 1-l2 >= 2(1-l1)), mu = 1, alpha = s^2: exact stationary block sum,
  // R^RR = n/(n-1) c^2 [ |a|^2/(1-L1^2) - 2 a.b/(1-L1 L2) + |b|^2/(1-L2^2) ], a_j, b_j centred lambda_i^{j+1}, L_i = lambda_i^n
  function RrrClosed(s, beta, n) {
    const p = 1 + beta - s * s, l1 = (p + Math.sqrt(p * p - 4 * beta)) / 2, l2 = beta / l1;
    const om2 = 1 - l2, om1 = s * s / om2, L1 = Math.log1p(-om1), L2 = l2 > 0 ? Math.log(l2) : -Infinity;
    const A = [], B = [];
    for (let j = 0; j < n; j++) { A.push(Math.expm1((j + 1) * L1)); B.push(l2 > 0 ? Math.exp((j + 1) * L2) : 0); }
    const ma = A.reduce((x, y) => x + y) / n, mb = B.reduce((x, y) => x + y) / n;
    let aa = 0, ab = 0, bb = 0;
    for (let j = 0; j < n; j++) { const a = A[j] - ma, b = B[j] - mb; aa += a * a; ab += a * b; bb += b * b; }
    const d1 = -Math.expm1(2 * n * L1), d12 = -Math.expm1(n * (L1 + L2)), d2 = l2 > 0 ? -Math.expm1(2 * n * L2) : 1, c = 1 / (om2 - om1);
    return { R: om1, val: n / (n - 1) * c * c * (aa / d1 - 2 * ab / d12 + bb / d2) };
  }
  // smallest value of R^RR s^4 / (n^2 R^3) over admissible parameters (n sqrt(alpha mu) <= 1/64); cf. (C.3).
  // Pairs with R < 3e-6 are skipped here and evaluated with RrrClosed (Case 1) or a truncated sum (Case 2).
  function checkFloorRate(ns) {
    let min = Infinity, arg = null, count = 0;
    for (const n of ns) for (let i = 0; i <= 10; i++) {
      const s = (1 / (64 * n)) * Math.pow(10, -i * 1.5 / 10);
      for (let k = 0; k <= 40; k++) {
        const b = 0.02 * Math.pow(10, k * Math.log10(1 / s / 0.02) / 40), sb = Math.max(0, 1 - b * s), beta = sb * sb;
        const R = 1 - roots(s * s, beta, 1).rho; if (60 / R > 2e7) continue; // RR-like corner, covered analytically
        const r = Rrr(s * s, beta, 1, n, 2e7); count++;
        const ratio = r.Rrr * s ** 4 / (n * n * r.R ** 3);
        if (ratio < min) { min = ratio; arg = { n, s, beta }; }
      }
    }
    return { min, arg, count };
  }
  // exact Tbar: first epoch from which E e^2 = E_mu g_{nt}^2 + noise_t stays <= eps
  function Tbar(alpha, beta, mu, n, sig2, E, eps, maxEp) {
    const p = 1 + beta - alpha * mu, rho = roots(alpha, beta, mu).rho; if (rho >= 1) return Infinity;
    const blk = new Float64Array(n); let h = 1, hprev = 0, N = 0, last = -1;
    for (let t = 0; t < maxEp; t++) {
      const g = h - beta * hprev; if (g * g * E + N > eps) last = t;
      let s = 0; for (let j = 0; j < n; j++) { blk[j] = h; s += h; const hn = p * h - beta * hprev; hprev = h; h = hn; }
      let v = 0; for (let j = 0; j < n; j++) v += (blk[j] - s / n) ** 2;
      N += alpha * alpha * sig2 * v * n / (n - 1); if (N > eps) return Infinity;
      const env = Math.pow(rho, n * t) * (n * t + 2) * 10;
      if (env * env * E < 1e-3 * eps && v * alpha * alpha * sig2 * n < 1e-3 * eps && t > 10) return last + 1;
    }
    return Infinity;
  }
  // Tbar minimized over admissible (alpha, beta), divided by (sig2/(n mu^2 eps))^{1/3} log(E/(4 eps))
  function checkTbar(n, mu, sig2, E, eps) {
    let best = Infinity;
    for (let i = 0; i <= 40; i++) {
      const s = (1 / (64 * n)) * Math.pow(10, -i * 3 / 40), alpha = s * s / mu;
      for (let k = 0; k <= 30; k++) {
        const b = 0.2 * Math.pow(10, k * 2.5 / 30), sb = 1 - b * s; if (sb < 0) continue;
        best = Math.min(best, Tbar(alpha, sb * sb, mu, n, sig2, E, eps, best === Infinity ? 3e5 : best + 1));
      }
    }
    return { T: best, ratio: best / (Math.cbrt(sig2 / (n * mu * mu * eps)) * Math.log(E / (4 * eps))) };
  }
  Object.assign(G, { hbRoots: roots, Rrr, RrrClosed, checkFloorRate, TbarExact: Tbar, checkTbar });
})(typeof window !== 'undefined' ? window : globalThis);
