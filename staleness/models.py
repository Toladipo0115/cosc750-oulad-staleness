"""Model definitions and hyper-parameter grids.

B0 majority-class floor, B1 logistic regression (course baseline),
B2 RBF-kernel SVM whose decision values are Platt-scaled on the grouped,
student-purged inner folds like B3 (course baseline), B3 calibrated
histogram gradient boosting (the main model of Wu, Duan and Luo 2026 and of
Wu, Jia and Pham 2026, re-implemented here; their code is not used).

Grids are fixed before any test fold is seen and are searched on inner
folds only.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class SVCScore(SVC):
    """RBF-SVM without scikit-learn's internal Platt step (whose internal folds ignore
    presentation and student grouping). predict_proba returns the logistic of the decision
    value, so the external sigmoid calibration in experiment.fit_tuned, fitted on the
    grouped, student-purged inner out-of-fold scores, is exactly Platt scaling."""

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])


SVM_MAX_TRAIN = 6000  # RBF-SVM training is O(n^2) to O(n^3); stratified subsample
SVM_MAX_TRAIN_LARGE = 12000  # Stage 4 sensitivity run (B2L)


class AgeInteractions(BaseEstimator, TransformerMixin):
    """Append age-by-feature products so a linear model can re-weight each
    feature by how old its source is. Expects the last two columns to be the
    source ages (a_V, a_A); V features interact with a_V, A features with a_A."""

    def __init__(self, n_vle=6, n_asm=3):
        self.n_vle, self.n_asm = n_vle, n_asm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        a_v, a_a = X[:, -2:-1], X[:, -1:]
        v = X[:, :self.n_vle]
        a = X[:, self.n_vle:self.n_vle + self.n_asm]
        return np.hstack([X, v * a_v, a * a_a])


def make_model(name, with_ages=False, seed=0, n_vle=6, n_asm=3):
    """Return (estimator, param_grid). n_vle and n_asm locate the source columns for
    the age interactions when features are dropped (Stage 4 recency ablation)."""
    if name == "B0":
        return DummyClassifier(strategy="prior"), {}
    if name == "B1":
        steps = [("impute", SimpleImputer(strategy="median"))]
        if with_ages:
            steps.append(("age_x", AgeInteractions(n_vle, n_asm)))
        steps += [("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000))]
        return Pipeline(steps), {"clf__C": [0.0003, 0.001, 0.01, 0.1, 1.0]}
    if name in ("B2", "B2L"):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", SVCScore(kernel="rbf", random_state=seed))]), \
            {"clf__C": [0.3, 1.0, 3.0], "clf__gamma": ["scale", 0.03]}
    if name == "B3":
        return HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                              random_state=seed), \
            {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15],
             "l2_regularization": [1.0, 10.0]}
    if name == "B3W":  # Stage 4: wider grid for B3 (the Stage 3 grid's most regularized corner won)
        return HistGradientBoostingClassifier(max_iter=200, early_stopping=False,
                                              random_state=seed), \
            {"learning_rate": [0.01, 0.03, 0.1], "max_leaf_nodes": [3, 7, 15],
             "l2_regularization": [1.0, 10.0, 30.0], "min_samples_leaf": [20, 100]}
    raise ValueError(name)


CALIBRATE = {"B0": False, "B1": False, "B2": True, "B2L": True, "B3": True, "B3W": True}  # B2: external Platt scaling
