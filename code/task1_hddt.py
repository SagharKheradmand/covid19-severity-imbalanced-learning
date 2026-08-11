import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Label"
DEFAULT_RANDOM_STATE = 42
DEPTH_VALUES = [None, 2, 3, 4, 5]
SEEDS = list(range(10))


def resolve_dataset_path(path=None):
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend([
        Path("Covid.csv"),
        Path("../Covid.csv"),
        Path(__file__).resolve().parent / "Covid.csv",
        Path(__file__).resolve().parent.parent / "Covid.csv",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find Covid.csv. Pass the dataset path as an argument.")


def kendall_feature_selection(X_train, y_train, threshold=0.5):
    kendall_corr = X_train.corr(method="kendall").abs().fillna(0.0)
    target_corr = X_train.apply(lambda col: col.corr(y_train, method="kendall")).abs().fillna(0.0)

    removed_features = set()
    removal_records = []
    columns = list(X_train.columns)
    candidate_pairs = []

    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1:]:
            tau_abs = kendall_corr.loc[feature_a, feature_b]
            if tau_abs > threshold:
                candidate_pairs.append((tau_abs, feature_a, feature_b))

    candidate_pairs.sort(reverse=True)

    for tau_abs, feature_a, feature_b in candidate_pairs:
        if feature_a in removed_features or feature_b in removed_features:
            continue

        corr_a = target_corr.loc[feature_a]
        corr_b = target_corr.loc[feature_b]

        if corr_a >= corr_b:
            kept, removed = feature_a, feature_b
            kept_corr, removed_corr = corr_a, corr_b
        else:
            kept, removed = feature_b, feature_a
            kept_corr, removed_corr = corr_b, corr_a

        removed_features.add(removed)
        removal_records.append({
            "removed_feature": removed,
            "kept_feature": kept,
            "abs_kendall_between_features": tau_abs,
            "removed_abs_corr_with_label": removed_corr,
            "kept_abs_corr_with_label": kept_corr,
        })

    selected_features = [col for col in columns if col not in removed_features]
    removal_report = pd.DataFrame(removal_records)
    return selected_features, removal_report, kendall_corr, target_corr


def preprocess_train_test(df, seed=DEFAULT_RANDOM_STATE, test_size=0.30, kendall_threshold=0.5):
    feature_cols = [col for col in df.columns if col != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    numerical_features = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 2
    ]
    categorical_features = [col for col in feature_cols if col not in numerical_features]
    continuous_features = [
        col for col in numerical_features
        if df[col].nunique(dropna=True) > 10
    ]

    median_imputer = SimpleImputer(strategy="median")
    mode_imputer = SimpleImputer(strategy="most_frequent")

    X_train_imputed = X_train_raw.copy()
    X_test_imputed = X_test_raw.copy()

    if numerical_features:
        X_train_imputed[numerical_features] = median_imputer.fit_transform(
            X_train_raw[numerical_features]
        )
        X_test_imputed[numerical_features] = median_imputer.transform(
            X_test_raw[numerical_features]
        )

    if categorical_features:
        X_train_imputed[categorical_features] = mode_imputer.fit_transform(
            X_train_raw[categorical_features]
        )
        X_test_imputed[categorical_features] = mode_imputer.transform(
            X_test_raw[categorical_features]
        )

    selected_features, removal_report, _, _ = kendall_feature_selection(
        X_train_imputed,
        y_train,
        threshold=kendall_threshold,
    )

    X_train_selected = X_train_imputed[selected_features].copy()
    X_test_selected = X_test_imputed[selected_features].copy()

    continuous_selected = [col for col in continuous_features if col in selected_features]
    scaler = StandardScaler()

    X_train_preprocessed = X_train_selected.copy()
    X_test_preprocessed = X_test_selected.copy()

    if continuous_selected:
        X_train_preprocessed[continuous_selected] = scaler.fit_transform(
            X_train_selected[continuous_selected]
        )
        X_test_preprocessed[continuous_selected] = scaler.transform(
            X_test_selected[continuous_selected]
        )

    preprocessing_info = {
        "selected_features": selected_features,
        "removed_features": removal_report,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "continuous_scaled": continuous_selected,
    }

    return (
        X_train_preprocessed.to_numpy(dtype=float),
        X_test_preprocessed.to_numpy(dtype=float),
        y_train.to_numpy(dtype=int),
        y_test.to_numpy(dtype=int),
        preprocessing_info,
    )


@dataclass
class HDDTNode:
    prediction: int
    positive_probability: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["HDDTNode"] = None
    right: Optional["HDDTNode"] = None

    @property
    def is_leaf(self):
        return self.feature_index is None


class HDDT:
    def __init__(self, min_samples_split=10, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if not set(np.unique(y)).issubset({-1, 1}):
            raise ValueError("HDDT expects labels in {-1, +1}.")
        self.n_features_in_ = X.shape[1]
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root_) for row in X], dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_positive_probability(row, self.root_) for row in X], dtype=float)

    def _build_tree(self, X, y, depth):
        positive_count = np.sum(y == 1)
        negative_count = np.sum(y == -1)
        prediction = 1 if positive_count >= negative_count else -1
        positive_probability = positive_count / len(y)

        node = HDDTNode(
            prediction=prediction,
            positive_probability=positive_probability,
        )

        if self._should_stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, _ = split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _should_stop(self, y, depth):
        if len(y) < self.min_samples_split:
            return True
        if np.all(y == y[0]):
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _best_split(self, X, y):
        total_positive = np.sum(y == 1)
        total_negative = np.sum(y == -1)
        if total_positive == 0 or total_negative == 0:
            return None

        best_feature = None
        best_threshold = None
        best_score = -np.inf

        for feature_index in range(X.shape[1]):
            values = np.sort(np.unique(X[:, feature_index]))
            if len(values) < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                p_left = np.sum(y[left_mask] == 1)
                n_left = np.sum(y[left_mask] == -1)
                p_right = total_positive - p_left
                n_right = total_negative - n_left

                score = (
                    (np.sqrt(p_left / total_positive) - np.sqrt(n_left / total_negative)) ** 2
                    + (np.sqrt(p_right / total_positive) - np.sqrt(n_right / total_negative)) ** 2
                )

                if score > best_score:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_score = score

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_score

    def _predict_one(self, row, node):
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def _predict_positive_probability(self, row, node):
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.positive_probability


def compute_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    g_mean = np.sqrt(sensitivity * specificity)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (+1)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall (+1)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-score (+1)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_score),
        "G-mean": g_mean,
    }


def run_hddt_experiments(df, max_depth=None, seeds=SEEDS, min_samples_split=10):
    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, preprocessing_info = preprocess_train_test(df, seed=seed)
        model = HDDT(min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics.update({
            "seed": seed,
            "max_depth": "None" if max_depth is None else max_depth,
            "n_features": len(preprocessing_info["selected_features"]),
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_metrics(results):
    metric_cols = [
        "Accuracy",
        "Precision (+1)",
        "Recall (+1)",
        "F1-score (+1)",
        "AUC-ROC",
        "G-mean",
    ]
    return pd.DataFrame({
        "Metric": metric_cols,
        "Mean (10 runs)": results[metric_cols].mean().values,
        "Std Dev": results[metric_cols].std(ddof=1).values,
    })


def run_pruning_experiment(df, depth_values=DEPTH_VALUES, seeds=SEEDS):
    all_results = []
    for depth in depth_values:
        depth_results = run_hddt_experiments(df, max_depth=depth, seeds=seeds)
        all_results.append(depth_results)
    pruning_results = pd.concat(all_results, ignore_index=True)

    metric_cols = [
        "Accuracy",
        "Precision (+1)",
        "Recall (+1)",
        "F1-score (+1)",
        "AUC-ROC",
        "G-mean",
    ]
    pruning_summary = (
        pruning_results
        .groupby("max_depth")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    return pruning_results, pruning_summary


def _max_depth_sort_key(depth):
    depth_label = str(depth).lower()
    if pd.isna(depth) or depth_label in {"none", "nan"}:
        return (1, float("inf"))
    return (0, float(depth))


def plot_pruning_results(pruning_results):
    plot_summary = (
        pruning_results
        .groupby("max_depth", dropna=False)[["F1-score (+1)", "G-mean"]]
        .mean()
    )
    plot_summary = plot_summary.loc[sorted(plot_summary.index, key=_max_depth_sort_key)]

    x_labels = [str(depth) for depth in plot_summary.index]
    x = np.arange(len(x_labels))

    plt.figure(figsize=(8, 5))
    plt.plot(x, plot_summary["G-mean"], marker="o", label="G-mean")
    plt.plot(x, plot_summary["F1-score (+1)"], marker="s", label="F1-score (+1)")
    plt.xticks(x, x_labels)
    plt.xlabel("max_depth")
    plt.ylabel("Mean score across 10 runs")
    plt.title("HDDT pruning experiment")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gca()


def main(dataset_path=None):
    np.random.seed(DEFAULT_RANDOM_STATE)
    data_path = resolve_dataset_path(dataset_path)
    df = pd.read_csv(data_path)

    print(f"Loaded dataset: {data_path} with shape {df.shape}")
    print("Running default HDDT evaluation with max_depth=None...")
    default_results = run_hddt_experiments(df, max_depth=None)
    print(summarize_metrics(default_results).round(4).to_string(index=False))

    print(f"\nRunning pruning experiment for max_depth in {DEPTH_VALUES}...")
    pruning_results, pruning_summary = run_pruning_experiment(df)
    print(pruning_summary.round(4).to_string(index=False))

    best_by_gmean = (
        pruning_results
        .groupby("max_depth")[["G-mean", "F1-score (+1)"]]
        .mean()
        .sort_values(["G-mean", "F1-score (+1)"], ascending=False)
        .head(1)
    )
    print("\nBest depth by mean G-mean:")
    print(best_by_gmean.round(4).to_string())

    ax = plot_pruning_results(pruning_results)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1: Hellinger Distance Decision Tree")
    parser.add_argument("dataset", nargs="?", default=None, help="Path to Covid.csv")
    args = parser.parse_args()
    main(args.dataset)
