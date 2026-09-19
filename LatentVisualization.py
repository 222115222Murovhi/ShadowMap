
import numpy as np
import matplotlib.pyplot as plt


def plot_latent_space(autoencoder, X_train, X_test, Y_test,
                      max_points=10000, seed=42, save_path=None):

    labels = np.asarray(Y_test).reshape(-1)
    if len(labels) != len(X_test) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Y_test must contain one 0/1 label per test row.")
    if max_points < 2 or len(X_train) < 2 or len(X_test) == 0:
        raise ValueError("Need at least two training rows, one test row, and max_points >= 2.")

    rng = np.random.default_rng(seed)
    train_ids = rng.choice(len(X_train), min(max_points, len(X_train)), replace=False)
    test_ids = rng.choice(len(X_test), min(max_points, len(X_test)), replace=False)

    # Encode a bounded sample to keep memory use low on large flow datasets.
    train_latent = np.asarray(autoencoder.encoder.forward(X_train[train_ids]),
                              dtype=np.float64).copy()
    test_latent = np.asarray(autoencoder.encoder.forward(X_test[test_ids]),
                             dtype=np.float64).copy()
    if train_latent.ndim != 2 or train_latent.shape[1] < 2:
        raise ValueError("The encoder must produce at least two latent features.")
    if not np.isfinite(train_latent).all() or not np.isfinite(test_latent).all():
        raise ValueError("Latent vectors contain NaN or infinity; check data/model.")

    # Centre on the training mean; learn the two highest-variance directions.
    mean = train_latent.mean(axis=0)
    centered = train_latent - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values ** 2
    if variance.sum() == 0:
        raise ValueError("Training latent vectors are constant; PCA is undefined.")
    explained = variance[:2] / variance.sum()
    points = (test_latent - mean) @ vt[:2].T
    sampled_labels = labels[test_ids]

    fig, ax = plt.subplots(figsize=(10, 7))
    for label, name, colour in [(0, "Normal", "tab:blue"),
                                 (1, "Attack", "tab:red")]:
        mask = sampled_labels == label
        ax.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.45,
                   c=colour, label=f"{name} ({mask.sum():,})", edgecolors="none")
    ax.set_xlabel(f"Principal component 1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"Principal component 2 ({explained[1]:.1%} variance)")
    ax.set_title("Latent space — test flows coloured by actual labels")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    print(f"PCA fitted on {len(train_ids):,} training flows; "
          f"showing {len(test_ids):,} test flows.")
    print(f"Two-dimensional view retains {explained.sum():.1%} of training latent variance.")
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax
