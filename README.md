# CS4267 Assignment 2 – kNN from scratch

## Files
- `knn.py` – full program (data loading, split, normalization, evaluation, driver)
- `test_knn.py` – checks for the two modules you implement
- `bcwdisc.data.mb.csv` – **add this** (from D2L) next to `knn.py`

## The two kNN modules
1. `compute_distances` – Euclidean distance from one test sample to all training samples.
   Hint: broadcasting `X_train - x_test`, square, sum across columns, sqrt.
2. `assign_class` – indices of the k smallest distances (`np.argsort`), look up their
   labels, majority vote. With labels ±1 the vote is the sign of the sum.

Run `python3 test_knn.py` to check them, then `python3 knn.py`.

## Report checklist
- [ ] Cover page (name, CS4267, Assignment 2, date)
- [ ] Abstract ≤ 100 words: objective, what you did, what you found
- [ ] Normalization decision, backed by the feature-range printout
- [ ] Table: accuracy for k = 1,3,5,7,9, raw vs normalized
- [ ] Confusion matrix for each k (at least the normalized run)
- [ ] Discussion: effect of normalization, effect of k, false negatives (missed malignant)
- [ ] Lessons learned, experience, future work (e.g. cross-validation, other distance metrics, weighted votes)
- [ ] Source code printout
- [ ] Zip of source + PDF report uploaded separately to D2L
