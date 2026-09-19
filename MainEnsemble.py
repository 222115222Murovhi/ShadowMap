"""Run beside the existing project modules: python MainEnsemble.py."""
from pathlib import Path
from datetime import datetime
import json
import time
import numpy as np
from activation import ReLU
from DenseLayer import DenseLayer
from Sequential import Sequential
from Autoencoder import Autoencoder
from SparseAutoencoder import SparseAutoencoder
from Trainer import Trainer
from Optimizer import Adam
from Loss import MSE
from DatasetManager import DatasetManager
from Ensemble import Ensemble, bootstrap_indices
import evaluate as ev

DATA_PATH = r"C:\Users\PHATHUTSHEDZO MUROVH\OneDrive - University of Johannesburg\Desktop\ShadowMap\data"
MAX_EPOCHS = 700
MEMBERS = [
    dict(name='basic', seed=101, corruption=0.0),
    dict(name='denoising', seed=202, corruption=0.2),
    dict(name='sparse', seed=303, corruption=0.0,
         sparsity_weight=1e-3, decoder_weight_penalty=1e-4),
]


class MaskingNoise:
    # Same corruption rule as your denoising AE; separate random generator.
    def __init__(self, rate, seed):
        self.rate = rate
        self.rng = np.random.default_rng(seed)

    def corrupt(self, X):
        return X * (self.rng.random(X.shape) >= self.rate)


def build_model(input_size, settings):
    np.random.seed(settings['seed'])
    encoder = Sequential([DenseLayer(input_size, 32), ReLU(),
                          DenseLayer(32, 16), ReLU(), DenseLayer(16, 8)])
    decoder = Sequential([DenseLayer(8, 16), ReLU(),
                          DenseLayer(16, 32), ReLU(), DenseLayer(32, input_size)])
    if settings['name'] == 'sparse':
        return SparseAutoencoder(encoder, decoder,
            sparsity_weight=settings['sparsity_weight'],
            decoder_weight_penalty=settings['decoder_weight_penalty'])
    return Autoencoder(encoder, decoder)


def load_ensemble(folder):
    """Reload without retraining: returns ensemble and its saved config."""
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text())
    models = []
    for settings in config['members']:
        model = build_model(config['input_size'], settings)
        model.load(folder / (settings['name'] + '.npz'))
        models.append(model)
    ensemble = Ensemble(models)
    ensemble.load_calibration(folder / 'calibration.npz')
    return ensemble, config


def main():
    # Fail before training if old module versions are still installed.
    if not hasattr(Autoencoder, 'load') or not hasattr(Sequential, 'set_training'):
        raise RuntimeError('Use the corrected Autoencoder and your updated Sequential.')
    if not isinstance(Adam().t, dict):
        raise RuntimeError('Use corrected Adam with per-layer timestep counters.')
    manager = DatasetManager(DATA_PATH)
    X_train, X_val, Y_val, X_test, Y_test, features = manager.load()
    folder = Path('runs') / datetime.now().strftime('ensemble_%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    trainer_settings = dict(batch_size=256, max_epochs=MAX_EPOCHS,
                            lr_patience=14, early_stop_patience=20, min_delta=1e-3)
    config = dict(input_size=X_train.shape[1], members=MEMBERS,
                  architecture=[X_train.shape[1], 32, 16, 8, 16, 32, X_train.shape[1]],
                  features=list(map(str, features)), trainer=trainer_settings,
                  learning_rate=0.001, percentile=95,
                  aggregation='mean of scores divided by member benign-validation p95',
                  preprocessing='Use the original processed cache; scaler is not included.')
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    models = []
    for settings in MEMBERS:
        name, seed = settings['name'], settings['seed']
        print(f'\nTraining bagged member: {name}', flush=True)
        model = build_model(X_train.shape[1], settings)
        indices = bootstrap_indices(len(X_train), seed + 1000)
        np.save(folder / f'{name}_bootstrap_indices.npy', indices)
        corruption = MaskingNoise(settings['corruption'], seed + 2000) if settings['corruption'] else None
        trainer = Trainer(model, Adam(learning_rate=0.001), MSE(),
                          corruption=corruption, **trainer_settings)
        started = time.perf_counter()
        trainer.fit(X_train[indices], X_val, Y_val)
        seconds = time.perf_counter() - started
        model.set_training(False)
        model.save(folder / f'{name}.npz')
        np.savez_compressed(folder / f'{name}_history.npz',
                            train=trainer.train_history, validation=trainer.val_history,
                            training_seconds=seconds)
        models.append(model)
    ensemble = Ensemble(models)
    validation_members = ensemble.member_scores(X_val)
    ensemble.calibrate(validation_members[Y_val == 0])
    ensemble.save_calibration(folder / 'calibration.npz')
    test_members = ensemble.member_scores(X_test)
    validation_scores = ensemble.combine(validation_members)
    test_scores = ensemble.combine(test_members)
    np.savez_compressed(folder / 'scores.npz', validation_members=validation_members,
                        test_members=test_members, validation_scores=validation_scores,
                        test_scores=test_scores, validation_labels=Y_val, test_labels=Y_test)
    # Compare ensemble and newly bagged members at their own calibrated p95.
    summary = {}
    for split, scores, members, labels in [
        ('validation', validation_scores, validation_members, Y_val),
        ('test', test_scores, test_members, Y_test)]:
        summary[split] = {'ensemble': ev.classification_metrics(labels, (scores > ensemble.threshold).astype(int))}
        for i, settings in enumerate(MEMBERS):
            summary[split][settings['name']] = ev.classification_metrics(
                labels, (members[:, i] > ensemble.scales[i]).astype(int))
    (folder / 'evaluation.json').write_text(json.dumps(summary, indent=2, default=lambda x: x.item()))
    print(json.dumps(summary, indent=2, default=lambda x: x.item()))
    print(f'All models, calibration, histories and results saved: {folder.resolve()}')


if __name__ == '__main__':
    main()
