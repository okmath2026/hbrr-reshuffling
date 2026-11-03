// Table 1 (common-Hessian instances): exact Tbar(eps) from Theorem 3.3 and its tuning.
// Instance: n = 6, A = diag(1, 1/kappa), x* = (1, 1), x0 = 0 with zero velocity, noise variances sigma_l^2 = (0.95, 0.56)
// along the two eigenvectors (only these enter Theorem 3.3), eps = 1e-4 |x*|^2.
// Runs in a browser or in Node.js; exports into the global object G.
(function (G) {
  const N = 6, SIG = [0.95, 0.56], EPS = 2e-4;
  // method 'hb' (beta = 0 gives RR) or 'nag'; first epoch from which the exact error stays <= EPS, or Infinity
  function tbar(method, alpha, beta, kappa, maxEp) {
    const as = [1, 1 / kappa];
    const st = as.map((a, l) => {
      const M = method === 'nag' ? [(1 + beta) * (1 - alpha * a), -beta * (1 - alpha * a)] : [1 + beta - alpha * a, -beta];
      return { M, h: 1, hp: 0, s0: -1, s1: -1, N: 0, sg: SIG[l] };
    });
    let last = -1;
    for (let t = 0; t < maxEp; t++) {
      let err = 0, amp = 0;
      for (const S of st) {
        err += S.s0 * S.s0 + S.N;
        let sum = 0; const blk = [];
        for (let j = 0; j < N; j++) {
          blk.push(S.h); sum += S.h;
          const hn = S.M[0] * S.h + S.M[1] * S.hp; S.hp = S.h; S.h = hn;
          const sn = S.M[0] * S.s0 + S.M[1] * S.s1; S.s1 = S.s0; S.s0 = sn;
        }
        let v = 0; for (const x of blk) v += (x - sum / N) ** 2;
        S.N += alpha * alpha * S.sg * v * N / (N - 1); // block B_tau of Theorem 3.3
        if (S.N > EPS) return Infinity;
        amp += Math.abs(S.h) + Math.abs(S.hp) + Math.abs(S.s0) + Math.abs(S.s1);
      }
      if (err > EPS) last = t;
      if (t > 2000 && t > 3 * last && amp < 1e-6) return last + 1;
    }
    return last < maxEp - 1 ? last + 1 : Infinity;
  }
  // grid over (log10 alpha, log10 of (1-beta)^2/(alpha mu)) followed by pattern search; RR is a 1-D search over alpha.
  // The optimum is re-evaluated over 4e6 epochs (field Tlong).
  function tune(method, kappa, laMin, laMax) {
    const mu = 1 / kappa; let best = { T: Infinity };
    const f = (la, lr) => {
      const cap = best.T === Infinity ? 400000 : Math.ceil(2 * best.T) + 10, alpha = 10 ** la;
      if (method === 'rr') return tbar('hb', alpha, 0, kappa, cap);
      const omb = Math.sqrt(10 ** lr * alpha * mu); if (omb >= 1) return Infinity;
      return tbar(method, alpha, 1 - omb, kappa, cap);
    };
    if (method === 'rr') {
      for (let i = 0; i <= 40; i++) { const la = laMin + (laMax - laMin) * i / 40, T = f(la); if (T < best.T) best = { T, la }; }
      for (let h = 0.03; h > 0.0005; h /= 2) for (const d of [-h, h]) { const T = f(best.la + d); if (T < best.T) best = { T, la: best.la + d }; }
    } else {
      for (let i = 0; i <= 20; i++) for (let k = 0; k <= 14; k++) {
        const la = laMin + (laMax - laMin) * i / 20, lr = -0.6 + 1.4 * k / 14, T = f(la, lr); if (T < best.T) best = { T, la, lr };
      }
      for (let h = 0.06; h > 0.002; h /= 2) {
        let imp = true;
        while (imp) {
          imp = false;
          for (const [da, dr] of [[h, 0], [-h, 0], [0, h], [0, -h], [h, h], [-h, -h], [h, -h], [-h, h]]) {
            const T = f(best.la + da, best.lr + dr); if (T < best.T) { best = { T, la: best.la + da, lr: best.lr + dr }; imp = true; }
          }
        }
      }
    }
    const alpha = 10 ** best.la; best.alpha = alpha;
    if (method === 'rr') best.Tlong = tbar('hb', alpha, 0, kappa, 4000000);
    else { best.omb = Math.sqrt(10 ** best.lr * alpha * mu); best.ratio = 10 ** best.lr; best.Tlong = tbar(method, alpha, 1 - best.omb, kappa, 4000000); }
    return best;
  }
  // Misspecified strong convexity: the momentum is set from mu_hat = c mu by 1 - beta = sqrt(ratio alpha mu_hat), with the
  // ratio of the tuned HBRR at c = 1, and only the step is re-tuned (1-D search as for RR).
  function misspec(kappa, c, ratio, laMin, laMax) {
    const mu = c / kappa; let best = { T: Infinity };
    const f = la => { const alpha = 10 ** la, omb = Math.sqrt(ratio * alpha * mu); if (omb >= 1) return Infinity;
      return tbar('hb', alpha, 1 - omb, kappa, best.T === Infinity ? 400000 : Math.ceil(2 * best.T) + 10); };
    for (let i = 0; i <= 40; i++) { const la = laMin + (laMax - laMin) * i / 40, T = f(la); if (T < best.T) best = { T, la }; }
    for (let h = 0.03; h > 0.0005; h /= 2) for (const d of [-h, h]) { const T = f(best.la + d); if (T < best.T) best = { T, la: best.la + d }; }
    best.alpha = 10 ** best.la; best.omb = Math.sqrt(ratio * best.alpha * mu);
    return best;
  }
  // Table 1 used laMin/laMax = (-4.8, -2.3) for HBRR with kappa <= 1000, (-5.3, -3.3) above; (-4.8, -2.0) and (-5.3, -3.0)
  // for NAG-RR; (-3.5, -1.0) for RR (RR for kappa >= 1000 by a 1-D scan of alpha in [0.042, 0.046]).
  Object.assign(G, { commonTbar: tbar, commonTune: tune, commonMisspec: misspec });
})(typeof window !== 'undefined' ? window : globalThis);
