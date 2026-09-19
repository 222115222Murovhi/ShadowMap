import numpy as np
import time
import copy


class Trainer:

    def __init__(
        self,
        autoencoder,
        optimizer,
        loss_function,
        batch_size=256,
        max_epochs=5000,
        lr_patience=20,
        early_stop_patience=40,
        lr_factor=0.5,
        min_delta=1e-4,      # now RELATIVE (fraction), not absolute
        min_lr=1e-6,
        lr_cooldown=5,        # epochs to wait after an LR cut before it can fire again
        corruption=None,
    ):

        self.autoencoder = autoencoder
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.lr_patience = lr_patience
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.lr_cooldown = lr_cooldown
        self.corruption = corruption

    def create_batches(self, X):
        # so the network doesn't see the same ordering every epoch.
        indices = np.random.permutation(len(X))

        for start in range(0, len(X), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield X[batch_indices]

    def train_epoch(self, X_train):

        total_loss = 0.0
        total_samples = 0

        self.autoencoder.set_training(True)

        for X_batch in self.create_batches(X_train):

            if self.corruption is None:
                network_input = X_batch
            else:
                network_input = self.corruption.corrupt(X_batch)

            reconstruction = self.autoencoder.forward(network_input)
            loss = self.loss_function.forward(X_batch, reconstruction)
            loss += self.autoencoder.regularization_loss()      
            gradient = self.loss_function.backward(X_batch, reconstruction)
            self.autoencoder.backward(gradient)

            for layer in self.autoencoder.encoder.layers:
                if layer.trainable:
                    self.optimizer.update(layer)

            for layer in self.autoencoder.decoder.layers:
                if layer.trainable:
                    self.optimizer.update(layer)

            # Any model-specific weight constraint belongs in that model,
            # not unconditionally in the shared trainer.

            batch_size = len(X_batch)
            total_loss += loss * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    def calculate_validation_loss(self, X_val):
        self.autoencoder.set_training(False)
        reconstruction = self.autoencoder.forward(X_val)
        return self.loss_function.forward(X_val, reconstruction)

    def _is_improvement(self, val_loss, best_val_loss):
     
        if best_val_loss == float("inf"):
            return True
        required_loss = best_val_loss * (1 - self.min_delta)
        return val_loss < required_loss

    def fit(self, X_train, X_val, Y_val):
        X_val_benign = X_val[Y_val == 0]

        print("Starting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val_benign)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Maximum epochs: {self.max_epochs}")
        print(f"Initial learning rate: {self.optimizer.learning_rate}")
        print(f"LR reduction factor: {self.lr_factor}")
        print(f"LR patience: {self.lr_patience}")
        print(f"Early stop patience: {self.early_stop_patience}")
        print(f"Min delta (relative): {self.min_delta}")

        best_val_loss = float("inf")
        best_parameters = None

        # epochs_since_best: counts up from the last TRUE best epoch.
        #   Only resets to 0 when a genuine improvement happens.
        #   Drives early stopping. LR reductions do NOT reset this.
        epochs_since_best = 0

        # epochs_since_lr_action: counts epochs since we last either
        #   improved OR cut the LR. Drives the LR scheduler only.
        epochs_since_lr_action = 0

        self.train_history = []
        self.val_history = []

        for epoch in range(self.max_epochs):

            start_time = time.time()

            train_loss = self.train_epoch(X_train)
            val_loss = self.calculate_validation_loss(X_val_benign)

            epoch_time = time.time() - start_time

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)

            improved = self._is_improvement(val_loss, best_val_loss)

            print(
                f"Epoch {epoch + 1:4d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {self.optimizer.learning_rate:.8f} | "
                f"Time: {epoch_time:.2f}s"
            )

            if improved:
                best_val_loss = val_loss
                epochs_since_best = 0
                epochs_since_lr_action = 0
                best_parameters = self.autoencoder.get_parameters()

                print(f"    ✓ New best validation loss: {best_val_loss:.6f}")
            else:
                epochs_since_best += 1
                epochs_since_lr_action += 1

                print(
                    f"    No improvement "
                    f"(early-stop: {epochs_since_best}/{self.early_stop_patience}, "
                    f"lr: {epochs_since_lr_action}/{self.lr_patience})"
                )

            # --------------------------------------------------
            # EARLY STOPPING — based purely on epochs_since_best,
            # unaffected by LR cuts along the way.
            # --------------------------------------------------
            if epochs_since_best >= self.early_stop_patience:
                print("\nEarly stopping triggered.")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print(f"Best epoch: {epoch + 1 - epochs_since_best}")
                break

            # --------------------------------------------------
            # LEARNING-RATE SCHEDULER — independent counter,
            # with a short cooldown so it can't fire every epoch
            # right after a cut.
            # --------------------------------------------------
            if epochs_since_lr_action >= self.lr_patience:
                current_lr = self.optimizer.learning_rate

                if current_lr > self.min_lr:
                    self.reduce_learning_rate()
                    epochs_since_lr_action = -self.lr_cooldown
                    # negative value effectively adds a grace period
                    # before this scheduler can trigger again
                else:
                    print("\nLearning rate has reached minimum.")
                    print("Early stopping triggered.")
                    break

        if best_parameters is not None:
            print("\nRestoring best model parameters...")
            self.autoencoder.set_parameters(best_parameters)
            print("✓ Best model restored.")

        print("\nTraining complete.")

        return self.train_history, self.val_history

    def reduce_learning_rate(self):
        old_lr = self.optimizer.learning_rate
        new_lr = old_lr * self.lr_factor
        new_lr = max(new_lr, self.min_lr)
        self.optimizer.learning_rate = new_lr

        print(f"    ↓ Learning rate reduced: {old_lr:.8f} → {new_lr:.8f}")

        return new_lr
