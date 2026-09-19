import numpy as np


def reconstruction_errors(autoencoder, X):
    """
    Per-sample reconstruction error (mean squared error per row).
    This is the anomaly score: higher error = more anomalous.
    """
    reconstruction = autoencoder.forward(X)
    errors = np.mean((X - reconstruction) ** 2, axis=1)
    return errors


def choose_threshold(val_errors_benign, method="percentile", percentile=95, n_std=3):
    """
    Pick a decision threshold using ONLY benign validation errors
    (never test data — that would leak information).

    - 'percentile': flag anything above the Nth percentile of benign
      validation error as anomalous. Simple, robust to outliers.
    - 'std': mean + n_std * std of benign validation error. More
      sensitive to the assumption that errors are roughly normal.
    """
    if method == "percentile":
        return np.percentile(val_errors_benign, percentile)
    elif method == "std":
        return np.mean(val_errors_benign) + n_std * np.std(val_errors_benign)
    else:
        raise ValueError(f"Unknown method: {method}")


def classification_metrics(y_true, y_pred):
    """
    Manual precision / recall / F1 / accuracy / confusion matrix.
    Convention: 1 = anomaly/attack, 0 = benign/normal.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    true_negative = np.sum((y_pred == 0) & (y_true == 0))
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    false_negative = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (true_positive + true_negative) / len(y_true)

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0 else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0 else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_positive": int(true_positive),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
        },
    }


def roc_auc(y_true, scores):
    """
    Manual ROC-AUC via the rank-sum (Mann-Whitney U) method —
    avoids needing sklearn and avoids sweeping thresholds by hand.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return float("nan")  # AUC undefined with only one class present

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    sum_ranks_pos = np.sum(ranks[y_true == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def evaluate(autoencoder, X_val, Y_val, X_test, Y_test, threshold_method="percentile", percentile=95):
    """
    Full evaluation pipeline. Returns a dict summary and prints a report.
    """
    # 1. Threshold chosen from BENIGN validation reconstruction error only
    X_val_benign = X_val[Y_val == 0]
    val_errors_benign = reconstruction_errors(autoencoder, X_val_benign)
    threshold = choose_threshold(val_errors_benign, method=threshold_method, percentile=percentile)

    # 2. Score the held-out test set (mixed benign + attack, never seen before)
    test_errors = reconstruction_errors(autoencoder, X_test)
    test_predictions = (test_errors > threshold).astype(int)

    # 3. Metrics
    metrics = classification_metrics(Y_test, test_predictions)
    auc = roc_auc(Y_test, test_errors)

    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Threshold method:      {threshold_method} (p{percentile})" if threshold_method == "percentile" else threshold_method)
    print(f"Threshold value:       {threshold:.8e}")
    print(f"Test samples:          {len(Y_test)}  "
          f"(benign: {np.sum(Y_test == 0)}, attack: {np.sum(Y_test == 1)})")
    print("-" * 50)
    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(f"Precision:             {metrics['precision']:.4f}")
    print(f"Recall:                {metrics['recall']:.4f}")
    print(f"F1 Score:              {metrics['f1']:.4f}")
    print(f"ROC-AUC:               {auc:.4f}")
    print("-" * 50)
    cm = metrics["confusion_matrix"]
    print("Confusion Matrix:")
    print(f"                  Predicted Benign   Predicted Attack")
    print(f"Actual Benign     {cm['true_negative']:<18} {cm['false_positive']}")
    print(f"Actual Attack     {cm['false_negative']:<18} {cm['true_positive']}")
    print("=" * 50)

    return {
        "threshold": threshold,
        "test_errors": test_errors,
        "test_predictions": test_predictions,
        "metrics": metrics,
        "roc_auc": auc,
    }


def threshold_sweep(autoencoder, X_val, Y_val, X_test, Y_test, percentiles=(80, 85, 90, 95, 97, 99)):
    """
    Evaluate multiple thresholds WITHOUT re-running the forward pass
    each time (errors are computed once, thresholds just move).

    Prints a compact table and returns the results so you can pick
    the operating point that fits your report's priorities.

    F2 is included alongside F1 because in intrusion detection, missing
    an attack (false negative) is usually costlier than a false alarm
    (false positive) -- F2 weights recall 4x more heavily than precision,
    which better reflects that asymmetry than F1's equal weighting.
    """
    X_val_benign = X_val[Y_val == 0]
    val_errors_benign = reconstruction_errors(autoencoder, X_val_benign)
    test_errors = reconstruction_errors(autoencoder, X_test)  # computed once, reused

    rows = []
    print(f"{'Percentile':>10} | {'Threshold':>12} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>6} | {'F2':>6}")
    print("-" * 78)

    for p in percentiles:
        threshold = np.percentile(val_errors_benign, p)
        predictions = (test_errors > threshold).astype(int)
        m = classification_metrics(Y_test, predictions)

        precision, recall = m["precision"], m["recall"]
        f2 = (
            5 * precision * recall / (4 * precision + recall)
            if (4 * precision + recall) > 0 else 0.0
        )

        rows.append({"percentile": p, "threshold": threshold, **m, "f2": f2})

        print(
            f"{p:>10} | {threshold:>12.4e} | {m['accuracy']:>8.4f} | "
            f"{precision:>9.4f} | {recall:>8.4f} | {m['f1']:>6.4f} | {f2:>6.4f}"
        )

    best_f1 = max(rows, key=lambda r: r["f1"])
    best_f2 = max(rows, key=lambda r: r["f2"])
    print("-" * 78)
    print(f"Best F1 at percentile {best_f1['percentile']}  (F1={best_f1['f1']:.4f})")
    print(f"Best F2 at percentile {best_f2['percentile']}  (F2={best_f2['f2']:.4f})  <- weights recall higher, likely more relevant for security")

    return rows


def plot_training_curves(train_history, val_history, save_path="training_curves.png"):
    """
    Plot train/val loss over epochs. Requires matplotlib.
    Run this to visually confirm convergence and where the plateau/
    early-stop point actually was.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(train_history, label="Train Loss")
    plt.plot(val_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training History")
    plt.legend()
    plt.yscale("log")  # log scale makes late-stage plateaus visible
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved {save_path}")