"""Quick checks for your two modules. Run:  python3 test_knn.py"""
import numpy as np
from knn import compute_distances, assign_class, knn_predict

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return cond

ok = True
X_train = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [10.0, 10.0], [0.5, 0.0]])
y_train = np.array([-1, 1, -1, 1, 1])

d = compute_distances(X_train, np.array([0.0, 0.0]))
ok &= check("distance shape", d.shape == (5,))
ok &= check("distance values", np.allclose(d, [0.0, 5.0, np.sqrt(2), np.sqrt(200), 0.5]))

# nearest to origin in order: idx0(-1), idx4(+1), idx2(-1), idx1(+1), idx3(+1)
ok &= check("k=1 vote", assign_class(d, y_train, 1) == -1)
ok &= check("k=3 vote", assign_class(d, y_train, 3) == -1)
ok &= check("k=5 vote", assign_class(d, y_train, 5) == 1)

X_test = np.array([[0.1, 0.1], [9.0, 9.0]])
ok &= check("knn_predict", list(knn_predict(X_train, y_train, X_test, 1)) == [-1, 1])

print("\nAll good!" if ok else "\nSome checks failed.")
