# hbrr-reshuffling

Code for the paper *Heavy-Ball Momentum Close to One Filters the Noise of Random Reshuffling* (Qingyu Zhang).

Every file is plain JavaScript without dependencies. It runs in a browser console or in Node.js and exports its functions into the global object (`window` or `globalThis`).

| File | Reproduces |
|---|---|
| `hbrr_common.js` | Table 1: exact epoch complexity on common-Hessian quadratics from the mean-square formula (`commonTbar`, `commonTune`), and the misspecified-μ computations (`commonMisspec`). |
| `hbrr_screen.js` | Tables 2–4 and 11: exact mean-square moments by enumerating all permutations, the stability test, and the tuning (`HBRR.tuneRefined` for epoch counts, `HBRR.tuneRate` and `HBRR.rateExact` for rates). |
| `hbrr_checks.js` | The separating instance (Table 12) with `hbrr_screen.js` loaded, the one-epoch checks of the general theory, and the numerical checks of the proofs in Appendix B. |
| `hbrr_lower.js` | The numerical checks of the heavy-ball lower bound (Theorem 4.4, Appendix C). |
| `hbrr_mc.js` | The large-n Monte Carlo experiments (Tables 13 and 14). |
| `hbrr_hetero.js` | The strongly heterogeneous and correlated-offset experiments, and the switching example of Section 7. |
| `hbrr_averaging.js` | The comparison with RR whose iterates are averaged (Section 6.6), on common-Hessian and heterogeneous instances. |
| `hbrr_logreg.js` | The logistic-regression experiments on LIBSVM data (Sections 6.7–6.9): RR, averaged RR, HBRR, HBRR with averaged iterates (`'hbema'`; the two-stage local tuning of Table 10 is `LR.gridsCombined`), NAG-RR and SAGA, single-sample, mini-batch and over fixed blocks (`LR.partition`: random, label or pca splits, or natural splits by a one-hot coded attribute, `'attr:lo:hi'`, found with `LR.onehotGroups`), the local re-tuning (`LR.runJobs`), the stationary errors over blocks (`LR.floorBlocks`), the local condition number (`LR.hessTop`) and the bias diagnostics (`LR.biasShare`, `LR.biasShareBlocks`). In a browser, `LR.runGrid` runs a grid in parallel Web Workers; the data files are read from the LIBSVM repository (australian_scale, mushrooms, a9a, w8a). |

Example in Node.js:

```js
require('./hbrr_common.js');
console.log(commonTune('hb', 100, -4.5, -2.5)); // tuned HBRR for kappa = 100
```

License: MIT (see `LICENSE`).
