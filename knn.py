import os
import sys
import numpy as np

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bcwdisc.data.mb.csv")
K_VALUES = [1, 3, 5, 7, 9]
TRAIN_FRACTION = 0.70
SEED = 42                    
POSITIVE = 1
NEGATIVE = -1    


def load_data(path):
    """Return X and y"""
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)
    return X, y


def print_feature_ranges(X):
    """Print min / max / range of every column"""
    mins, maxs = X.min(axis=0), X.max(axis=0)
    print(f"{'Feature':>8} {'Min':>12} {'Max':>12} {'Range':>12}")
    for i, (lo, hi) in enumerate(zip(mins, maxs)):
        print(f"{i + 1:>8} {lo:>12.4f} {hi:>12.4f} {hi - lo:>12.4f}")
    ranges = maxs - mins
    print(f"\nLargest range / smallest range = {ranges.max() / ranges.min():,.0f}x")


def stratified_split(X, y, train_fraction=TRAIN_FRACTION, seed=SEED):
    """Shuffle each class"""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        cut = int(round(train_fraction * len(idx)))
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def minmax_fit(X_train):
    mins = X_train.min(axis=0)
    spans = X_train.max(axis=0) - mins
    spans[spans == 0] = 1.0   # guard against constant columns
    return mins, spans


def minmax_apply(X, mins, spans):
    return (X - mins) / spans


def compute_distances(X_train, x_test):
    """
    Distance from ONE test sample to EVERY training sample.

    X_train : (n_train, d) array
    x_test  : (d,) array
    returns : (n_train,) array, entry i = Euclidean distance between x_test and X_train[i]
    """
    diff = X_train - x_test                  
    return np.sqrt(np.sum(diff ** 2, axis=1)) 


def assign_class(distances, y_train, k):
    nearest = np.argsort(distances)[:k]      
    votes = y_train[nearest]                 
    return POSITIVE if votes.sum() > 0 else NEGATIVE  


def knn_predict(X_train, y_train, X_test, k):
    """Classify every test sample using Module 1 then Module 2."""
    return np.array([assign_class(compute_distances(X_train, x), y_train, k) for x in X_test])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred):
    """Returns dict with TP, FN, FP, TN """
    return {
        "TP": int(np.sum((y_true == POSITIVE) & (y_pred == POSITIVE))),
        "FN": int(np.sum((y_true == POSITIVE) & (y_pred == NEGATIVE))),
        "FP": int(np.sum((y_true == NEGATIVE) & (y_pred == POSITIVE))),
        "TN": int(np.sum((y_true == NEGATIVE) & (y_pred == NEGATIVE))),
    }


def print_confusion(cm):
    print("                    Predicted")
    print("                 Malig(1)  Benign(-1)")
    print(f"Actual Malig(1)  {cm['TP']:>8}  {cm['FN']:>10}")
    print(f"Actual Benign(-1){cm['FP']:>8}  {cm['TN']:>10}")


def run_experiment(X_train, X_test, y_train, y_test, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    summary = []
    for k in K_VALUES:
        y_pred = knn_predict(X_train, y_train, X_test, k)
        acc = accuracy(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        summary.append((k, acc, cm))
        print(f"\nk = {k}   accuracy = {acc:.4f}  ({acc * 100:.2f}%)")
        print_confusion(cm)
    return summary


def print_summary(raw, norm):
    print(f"\n{'=' * 60}\nSUMMARY (accuracy, FN = missed malignant)\n{'=' * 60}")
    print(f"{'k':>3} | {'Raw acc':>8} {'Raw FN':>7} | {'Norm acc':>8} {'Norm FN':>7}")
    for (k, a1, c1), (_, a2, c2) in zip(raw, norm):
        print(f"{k:>3} | {a1:>8.4f} {c1['FN']:>7} | {a2:>8.4f} {c2['FN']:>7}")

def main():
    if not os.path.exists(DATA_FILE):
        sys.exit(f"Dataset not found: {DATA_FILE}\nPut bcwdisc.data.mb.csv next to knn.py.")

    X, y = load_data(DATA_FILE)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Malignant (1): {np.sum(y == 1)}   Benign (-1): {np.sum(y == -1)}\n")
    print_feature_ranges(X)

    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"\nTrain: {len(y_train)}   Test: {len(y_test)}")

    raw = run_experiment(X_train, X_test, y_train, y_test, "WITHOUT normalization")

    mins, spans = minmax_fit(X_train)
    norm = run_experiment(minmax_apply(X_train, mins, spans), minmax_apply(X_test, mins, spans),
                          y_train, y_test, "WITH min-max normalization (fit on train only)")

    print_summary(raw, norm)


if __name__ == "__main__":
    main()
