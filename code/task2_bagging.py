import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from task1_hddt import (
    HDDT,
    SEEDS,
    compute_metrics,
    preprocess_train_test,
    resolve_dataset_path,
)

T_VALUES = [11, 31, 51, 101]
METRIC_COLS = [
    "Accuracy",
    "Precision (+1)",
    "Recall (+1)",
    "F1-score (+1)",
    "AUC-ROC",
    "G-mean",
]


class BaggingImbalanced:
    def __init__(
        self,
        base_learner="hddt",
        T=31,
        max_depth=3,
        min_samples_split=10,
        random_state=42,
    ):
        if base_learner not in {"hddt", "dt"}:
            raise ValueError("base_learner must be either 'hddt' or 'dt'.")
        self.base_learner = base_learner
        self.T = T
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if not set(np.unique(y)).issubset({-1, 1}):
            raise ValueError("BaggingImbalanced expects labels in {-1, +1}.")

        rng = np.random.default_rng(self.random_state)
        minority_indices = np.flatnonzero(y == 1)
        majority_indices = np.flatnonzero(y == -1)

        if len(minority_indices) == 0 or len(majority_indices) == 0:
            raise ValueError("Both minority (+1) and majority (-1) classes are required.")
        if len(majority_indices) < len(minority_indices):
            raise ValueError("This undersampling implementation assumes +1 is the minority class.")

        self.estimators_ = []
        minority_size = len(minority_indices)

        for _ in range(self.T):
            sampled_minority = rng.choice(
                minority_indices,
                size=minority_size,
                replace=True,
            )
            sampled_majority = rng.choice(
                majority_indices,
                size=minority_size,
                replace=False,
            )
            sampled_indices = np.concatenate([sampled_minority, sampled_majority])
            rng.shuffle(sampled_indices)

            estimator = self._make_base_estimator(rng)
            estimator.fit(X[sampled_indices], y[sampled_indices])
            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.column_stack([estimator.predict(X) for estimator in self.estimators_])
        vote_sum = votes.sum(axis=1)
        return np.where(vote_sum >= 0, 1, -1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probabilities = [
            self._positive_probability(estimator, X)
            for estimator in self.estimators_
        ]
        return np.mean(np.column_stack(probabilities), axis=1)

    def _make_base_estimator(self, rng):
        if self.base_learner == "hddt":
            return HDDT(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
            )
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )

    @staticmethod
    def _positive_probability(estimator, X):
        proba = estimator.predict_proba(X)
        if proba.ndim == 1:
            return proba

        classes = list(estimator.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(X.shape[0], dtype=float)


def run_bagging_experiments(
    df,
    T=31,
    base_learner="hddt",
    max_depth=3,
    seeds=SEEDS,
    min_samples_split=10,
):
    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, preprocessing_info = preprocess_train_test(df, seed=seed)
        model = BaggingImbalanced(
            base_learner=base_learner,
            T=T,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics.update({
            "seed": seed,
            "T": T,
            "base_learner": base_learner,
            "max_depth": max_depth,
            "n_features": len(preprocessing_info["selected_features"]),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_by_t(results):
    return (
        results
        .groupby("T")[METRIC_COLS]
        .agg(["mean", "std"])
        .reset_index()
    )


def format_mean_std_table(results, group_col):
    summary = (
        results
        .groupby(group_col)[METRIC_COLS]
        .agg(["mean", "std"])
    )

    formatted = pd.DataFrame(index=summary.index)
    for metric in METRIC_COLS:
        formatted[metric] = [
            f"{mean:.4f} +/- {std:.4f}"
            for mean, std in zip(summary[(metric, "mean")], summary[(metric, "std")])
        ]
    return formatted.reset_index()


def run_ensemble_size_experiment(df, T_values=T_VALUES, seeds=SEEDS):
    all_results = []
    for T in T_values:
        results = run_bagging_experiments(
            df,
            T=T,
            base_learner="hddt",
            max_depth=3,
            seeds=seeds,
        )
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)


def run_base_learner_comparison(df, T=51, max_depth=3, seeds=SEEDS):
    all_results = []
    for base_learner in ["hddt", "dt"]:
        results = run_bagging_experiments(
            df,
            T=T,
            base_learner=base_learner,
            max_depth=max_depth,
            seeds=seeds,
        )
        all_results.append(results)
    return pd.concat(all_results, ignore_index=True)


def plot_t_experiment(results):
    plot_summary = results.groupby("T")[["F1-score (+1)", "G-mean"]].mean().reindex(T_VALUES)

    plt.figure(figsize=(8, 5))
    plt.plot(plot_summary.index, plot_summary["G-mean"], marker="o", label="G-mean")
    plt.plot(plot_summary.index, plot_summary["F1-score (+1)"], marker="s", label="F1-score (+1)")
    plt.xlabel("Number of estimators (T)")
    plt.ylabel("Mean score across 10 runs")
    plt.title("Bagging with undersampling: effect of ensemble size")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
  
    plt.show()
    return plt.gca()


def main(dataset_path=None):
    data_path = resolve_dataset_path(dataset_path)
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path} with shape {df.shape}")

    print("\nExperiment 3.3: effect of ensemble size T with HDDT base learner.")
    t_results = run_ensemble_size_experiment(df)
    print(format_mean_std_table(t_results, "T").to_string(index=False))

    print("\nExperiment 3.4: HDDT vs. standard sklearn DecisionTree base learner.")
    comparison_results = run_base_learner_comparison(df)
    comparison_table = format_mean_std_table(comparison_results, "base_learner")
    comparison_table["base_learner"] = comparison_table["base_learner"].map({
        "hddt": "HDDT (yours)",
        "dt": "Standard DT (sklearn)",
    })
    print(comparison_table.to_string(index=False))
    
    t_results = run_ensemble_size_experiment(df, T_values=T_VALUES, seeds=SEEDS)
    ax = plot_t_experiment(t_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: Bagging with undersampling")
    parser.add_argument("dataset", nargs="?", default=None, help="Path to Covid.csv")
    args = parser.parse_args()
    main(args.dataset)
