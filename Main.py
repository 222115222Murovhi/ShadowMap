# %%
# Imports and model setup
from pathlib import Path
from datetime import datetime
import numpy as np
import json

from activation import ReLU
import DatasetManager as dm
import Optimizer as opt
import Loss as loss
import DenseLayer as dl
import Sequential as seq
import Autoencoder as ae
import Trainer as tr
from LatentVisualization import plot_latent_space

# %%
# 2. Load dataset and preprocess

dataset_manager = dm.DatasetManager(r"C:\Users\PHATHUTSHEDZO MUROVH\OneDrive - University of Johannesburg\Desktop\ShadowMap\data")
X_train, X_val, Y_val, X_test, Y_test, final_features_list = dataset_manager.load()

# %%
# 3. Define the autoencoder architecture
np.random.seed(42)
input_size = X_train.shape[1]

encoder = seq.Sequential([
    dl.DenseLayer(input_size, 32),
    ReLU(),
    dl.DenseLayer(32, 16),
    ReLU(),
    dl.DenseLayer(16, 8)    
])

decoder = seq.Sequential([
    dl.DenseLayer(8, 16),
    ReLU(),
    dl.DenseLayer(16, 32),
    ReLU(),
    dl.DenseLayer(32, input_size)
])

autoencoder = ae.Autoencoder(
    encoder=encoder,
    decoder=decoder
)

# Use the corrected Adam class with a timestep counter per layer.
optimizer = opt.Adam(learning_rate=0.001)
loss_function = loss.MSE()

# %%
# 4. Train model
trainer = tr.Trainer(
    autoencoder=autoencoder,
    optimizer=optimizer,
    loss_function=loss_function,
    batch_size=256,
    max_epochs=700,
    lr_patience=14,
    early_stop_patience=20,
    min_delta=1e-3,           # Relative improvement of 0.1%.
)
# Separate folder for each run, so previous results aren't overwritten.
run_dir = Path("runs") / datetime.now().strftime("basic_ae_%Y%m%d_%H%M%S")
run_dir.mkdir(parents=True, exist_ok=True)

trainer.fit(X_train, X_val, Y_val)

autoencoder.save(run_dir / "model.npz")


# %%
# 5. Evaluate model
import evaluate as ev

results = ev.evaluate(
    autoencoder=autoencoder,
    X_val=X_val,
    Y_val=Y_val,
    X_test=X_test,
    Y_test=Y_test,
    threshold_method="percentile",
    percentile=95,
)
# Small, readable summary for comparing models.
summary = {
    "model": "Basic AE",
    "threshold_method": "benign_validation_percentile",
    "percentile": 95,
    "threshold": float(results["threshold"]),
    "metrics": results["metrics"],
    "roc_auc": float(results["roc_auc"]),
    "epochs": len(trainer.train_history),
}

with open(run_dir / "evaluation.json", "w") as file:
    json.dump(
        summary,
        file,
        indent=4,
        default=lambda value: value.item(),
    )

# Save individual predictions and errors for later analysis.
np.savez_compressed(
    run_dir / "evaluation_arrays.npz",
    labels=Y_test,
    predictions=results["test_predictions"],
    errors=results["test_errors"],
    train_loss=np.asarray(trainer.train_history),
    val_loss=np.asarray(trainer.val_history),
)

ev.plot_training_curves(
    trainer.train_history,
    trainer.val_history,
    save_path=run_dir / "training_curves.png",
)

plot_latent_space(
    autoencoder,
    X_train,
    X_test,
    Y_test,
    save_path=run_dir / "latent_space.png",
)

print(f"Everything saved in: {run_dir.resolve()}")
# %%
