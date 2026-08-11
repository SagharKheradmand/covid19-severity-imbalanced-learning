import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

from task1_hddt import (
    HDDT,
    SEEDS,
    compute_metrics,
    preprocess_train_test,
    resolve_dataset_path,
    run_hddt_experiments,
)
from task2_bagging import (
    METRIC_COLS,
    run_bagging_experiments,
)

BOOSTING_T_VALUES = [10, 25, 50, 100]
TABLE_FORMAT = "github"


class AdaBoostM1:
    def __init__(self, T=50, learning_rate=1.0, random_state=42):
        self.T = T
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.alphas_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if not set(np.unique(y)).issubset({-1, 1}):
            raise ValueError("AdaBoostM1 expects labels in {-1, +1}.")

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1.0 / n_samples)

        self.estimators_ = []
        self.alphas_ = []

        for _ in range(self.T):
            stump = DecisionTreeClassifier(
                max_depth=1,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            stump.fit(X, y, sample_weight=weights)

            predictions = stump.predict(X)
            incorrect = predictions != y
            error = np.sum(weights * incorrect)

            if error <= 1e-12:
                alpha = self.learning_rate * 0.5 * np.log((1.0 - 1e-12) / 1e-12)
                self.estimators_.append(stump)
                self.alphas_.append(alpha)
                break

            if error >= 0.5:
                break

            alpha = self.learning_rate * 0.5 * np.log((1.0 - error) / error)
            weights *= np.exp(-alpha * y * predictions)
            weights /= weights.sum()

            self.estimators_.append(stump)
            self.alphas_.append(alpha)

        if not self.estimators_:
            stump = DecisionTreeClassifier(max_depth=1, random_state=self.random_state)
            stump.fit(X, y)
            self.estimators_.append(stump)
            self.alphas_.append(1.0)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.zeros(X.shape[0], dtype=float)
        for alpha, estimator in zip(self.alphas_, self.estimators_):
            scores += alpha * estimator.predict(X)
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = np.clip(scores, -30, 30)
        return 1.0 / (1.0 + np.exp(-2.0 * scores))


def smote_oversample(X, y, k=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(random_state)

    minority_indices = np.flatnonzero(y == 1)
    majority_indices = np.flatnonzero(y == -1)
    n_minority = len(minority_indices)
    n_majority = len(majority_indices)

    if n_minority == 0 or n_majority == 0:
        raise ValueError("Both classes are required for SMOTE.")
    if n_minority >= n_majority:
        return X.copy(), y.copy()

    minority_samples = X[minority_indices]
    n_synthetic = n_majority - n_minority
    effective_k = min(k, n_minority - 1)

    if effective_k < 1:
        synthetic_samples = minority_samples[
            rng.choice(n_minority, size=n_synthetic, replace=True)
        ]
    else:
        distances = np.linalg.norm(
            minority_samples[:, None, :] - minority_samples[None, :, :],
            axis=2,
        )
        neighbor_indices = np.argsort(distances, axis=1)[:, 1:effective_k + 1]

        synthetic_samples = np.empty((n_synthetic, X.shape[1]), dtype=float)
        for i in range(n_synthetic):
            base_index = rng.integers(0, n_minority)
            neighbor_index = rng.choice(neighbor_indices[base_index])
            lam = rng.random()
            synthetic_samples[i] = (
                minority_samples[base_index]
                + lam * (minority_samples[neighbor_index] - minority_samples[base_index])
            )

    X_resampled = np.vstack([X, synthetic_samples])
    y_resampled = np.concatenate([y, np.ones(n_synthetic, dtype=int)])
    return X_resampled, y_resampled


def run_adaboost_experiments(df, T=50, use_smote=False, seeds=SEEDS):
    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, preprocessing_info = preprocess_train_test(df, seed=seed)
        X_fit, y_fit = X_train, y_train

        if use_smote:
            X_fit, y_fit = smote_oversample(X_train, y_train, k=5, random_state=seed)

        model = AdaBoostM1(T=T, learning_rate=1.0, random_state=seed)
        model.fit(X_fit, y_fit)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics.update({
            "seed": seed,
            "T": T,
            "method": "AdaBoost + SMOTE" if use_smote else "AdaBoost (no SMOTE)",
            "use_smote": use_smote,
            "n_estimators_used": len(model.estimators_),
            "n_train_after_smote": len(y_fit),
            "n_features": len(preprocessing_info["selected_features"]),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def run_smote_comparison(df, T=50, seeds=SEEDS):
    no_smote = run_adaboost_experiments(df, T=T, use_smote=False, seeds=seeds)
    with_smote = run_adaboost_experiments(df, T=T, use_smote=True, seeds=seeds)
    return pd.concat([no_smote, with_smote], ignore_index=True)


def run_boosting_round_experiment(df, T_values=BOOSTING_T_VALUES, seeds=SEEDS):
    rows = []
    for T in T_values:
        for seed in seeds:
            X_train, X_test, y_train, y_test, preprocessing_info = preprocess_train_test(df, seed=seed)
            X_fit, y_fit = smote_oversample(X_train, y_train, k=5, random_state=seed)

            model = AdaBoostM1(T=T, learning_rate=1.0, random_state=seed)
            model.fit(X_fit, y_fit)

            train_pred = model.predict(X_fit)
            test_pred = model.predict(X_test)
            test_score = model.predict_proba(X_test)
            test_metrics = compute_metrics(y_test, test_pred, test_score)

            row = {
                "seed": seed,
                "T": T,
                "train_error": np.mean(train_pred != y_fit),
                "test_error": np.mean(test_pred != y_test),
                "n_estimators_used": len(model.estimators_),
                "n_features": len(preprocessing_info["selected_features"]),
            }
            row.update(test_metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_grouped(results, group_col):
    return (
        results
        .groupby(group_col)[METRIC_COLS]
        .agg(["mean", "std"])
        .reset_index()
    )


def format_mean_std_table(results, group_col, metric_cols=None):
    if metric_cols is None:
        metric_cols = METRIC_COLS

    summary = results.groupby(group_col)[metric_cols].agg(["mean", "std"])
    formatted = pd.DataFrame(index=summary.index)
    for metric in metric_cols:
        formatted[metric] = [
            f"{mean:.4f} +/- {std:.4f}"
            for mean, std in zip(summary[(metric, "mean")], summary[(metric, "std")])
        ]
    return formatted.reset_index()


def table_to_text(table, floatfmt=".4f"):
    if tabulate is None:
        return table.to_string(index=False)
    return tabulate(
        table,
        headers="keys",
        tablefmt=TABLE_FORMAT,
        showindex=False,
        floatfmt=floatfmt,
    )


def plot_boosting_rounds(round_results):
    plot_summary = (
        round_results
        .groupby("T")[["train_error", "test_error", "F1-score (+1)"]]
        .mean()
        .reindex(BOOSTING_T_VALUES)
    )

    plt.figure(figsize=(8, 5))
    plt.plot(plot_summary.index, plot_summary["train_error"], marker="o", label="Train error")
    plt.plot(plot_summary.index, plot_summary["test_error"], marker="s", label="Test error")
    plt.plot(plot_summary.index, plot_summary["F1-score (+1)"], marker="^", label="F1 (+1)")
    plt.xlabel("Number of boosting rounds (T)")
    plt.ylabel("Mean score across 10 runs")
    plt.title("AdaBoost + SMOTE: effect of boosting rounds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()
    return plt.gca()


def build_full_comparison(df, seeds=SEEDS):
    hddt_results = []
    for depth in [None, 2, 3, 4, 5]:
        result = run_hddt_experiments(df, max_depth=depth, seeds=seeds)
        result["method"] = "HDDT (Task 1)"
        result["config"] = f"max_depth={depth}"
        hddt_results.append(result)
    hddt_all = pd.concat(hddt_results, ignore_index=True)

    bag_hddt_results = []
    for T in [11, 31, 51, 101]:
        result = run_bagging_experiments(
            df,
            T=T,
            base_learner="hddt",
            max_depth=3,
            seeds=seeds,
        )
        result["method"] = "Bagging + HDDT (Task 2)"
        result["config"] = f"T={T}, max_depth=3"
        bag_hddt_results.append(result)
    bag_hddt_all = pd.concat(bag_hddt_results, ignore_index=True)

    bag_dt_all = run_bagging_experiments(
        df,
        T=51,
        base_learner="dt",
        max_depth=3,
        seeds=seeds,
    )
    bag_dt_all["method"] = "Bagging + DT (Task 2)"
    bag_dt_all["config"] = "T=51, max_depth=3"

    ada_results = []
    for T in [10, 25, 50, 100]:
        result = run_adaboost_experiments(df, T=T, use_smote=False, seeds=seeds)
        result["method"] = "AdaBoost (Task 3)"
        result["config"] = f"T={T}"
        ada_results.append(result)
    ada_all = pd.concat(ada_results, ignore_index=True)

    ada_smote_results = []
    for T in [10, 25, 50, 100]:
        result = run_adaboost_experiments(df, T=T, use_smote=True, seeds=seeds)
        result["method"] = "AdaBoost + SMOTE (Task 3)"
        result["config"] = f"T={T}"
        ada_smote_results.append(result)
    ada_smote_all = pd.concat(ada_smote_results, ignore_index=True)

    all_results = pd.concat(
        [hddt_all, bag_hddt_all, bag_dt_all, ada_all, ada_smote_all],
        ignore_index=True,
    )

    rows = []
    for method, method_results in all_results.groupby("method"):
        config_summary = (
            method_results
            .groupby("config")[["F1-score (+1)", "G-mean", "AUC-ROC"]]
            .mean()
            .sort_values(["F1-score (+1)", "G-mean"], ascending=False)
        )
        best_config = config_summary.index[0]
        best_row = config_summary.iloc[0]
        rows.append({
            "Method": method,
            "Best Config": best_config,
            "F1 (+1)": best_row["F1-score (+1)"],
            "G-mean": best_row["G-mean"],
            "AUC-ROC": best_row["AUC-ROC"],
        })

    return pd.DataFrame(rows).sort_values("F1 (+1)", ascending=False)


def main(dataset_path=None):
    data_path = resolve_dataset_path(dataset_path)
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path} with shape {df.shape}")

    print("\nExperiment A: AdaBoost vs. AdaBoost + SMOTE.")
    smote_comparison = run_smote_comparison(df, T=50)
    print(table_to_text(format_mean_std_table(smote_comparison, "method")))

    print("\nExperiment B: number of boosting rounds with SMOTE.")
    round_results = run_boosting_round_experiment(df)
    round_table = format_mean_std_table(
        round_results,
        "T",
        metric_cols=["train_error", "test_error", "F1-score (+1)"],
    )
    print(table_to_text(round_table))

    ax = plot_boosting_rounds(round_results)
    full_comparison = build_full_comparison(df, seeds=SEEDS)
    print(table_to_text(full_comparison))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3: AdaBoost for imbalanced data")
    parser.add_argument("dataset", nargs="?", default=None, help="Path to Covid.csv")
    args = parser.parse_args()
    main(args.dataset)
