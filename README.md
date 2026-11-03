# hbrr-reshuffling

Code for the paper *Heavy-Ball Momentum Close to One Filters the Noise of Random Reshuffling* (Qingyu Zhang).

Every file is plain JavaScript without dependencies. It runs in a browser console or in Node.js and exports its functions into the global object (`window` or `globalThis`).

| File | Reproduces |
|---|---|
| `hbrr_common.js` | Table 1: exact epoch complexity on common-Hessian quadratics from the mean-square formula (`commonTbar`, `commonTune`). |
| `hbrr_screen.js` | Tables 2–5: exact mean-square moments by enumerating all permutations, the stability test, and the tuning (`HBRR.tuneRefined` for epoch counts, `HBRR.tuneRate` and `HBRR.rateExact` for rates). |
| `hbrr_checks.js` | The separating instance (Table 6) with `hbrr_screen.js` loaded, the one-epoch checks of the general theory, and the numerical checks of the proofs in Appendix B. |
| `hbrr_lower.js` | The numerical checks of the heavy-ball lower bound (Theorem 4.3, Appendix C). |
| `hbrr_mc.js` | The large-n Monte Carlo experiments (Tables 7 and 8). |
| `hbrr_hetero.js` | The strongly heterogeneous and correlated-offset experiments, and the switching example of Section 7. |
| `hbrr_averaging.js` | The comparison with RR whose iterates are averaged (Section 6.6), on common-Hessian and heterogeneous instances. |

Example in Node.js:

```js
require('./hbrr_common.js');
console.log(commonTune('hb', 100, -4.5, -2.5)); // tuned HBRR for kappa = 100
```

License: MIT (see `LICENSE`).
