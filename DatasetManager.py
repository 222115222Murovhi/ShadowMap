import os
import numpy as np
import fileLoader as fl


class DatasetManager:

    def __init__(self, data_path, processed_dir="processed"):

        self.data_path = data_path
        self.processed_dir = processed_dir

        os.makedirs(self.processed_dir, exist_ok=True)

    def load(self):

        X_train_path = os.path.join(
            self.processed_dir, "X_train.npy"
        )

        X_val_path = os.path.join(
            self.processed_dir, "X_val.npy"
        )

        Y_val_path = os.path.join(
            self.processed_dir, "Y_val.npy"
        )

        X_test_path = os.path.join(
            self.processed_dir, "X_test.npy"
        )

        Y_test_path = os.path.join(
            self.processed_dir, "Y_test.npy"
        )

        features_path = os.path.join(
            self.processed_dir, "features.npy"
        )

        # Check whether processed data already exists
        files_exist = all(
            os.path.exists(path)
            for path in [
                X_train_path,
                X_val_path,
                Y_val_path,
                X_test_path,
                Y_test_path,
                features_path
            ]
        )

        # --------------------------------------------------
        # LOAD CACHED DATA
        # --------------------------------------------------

        if files_exist:

            print("📂 Loading processed dataset...")

            X_train = np.load(X_train_path)
            X_val = np.load(X_val_path)
            Y_val = np.load(Y_val_path)
            X_test = np.load(X_test_path)
            Y_test = np.load(Y_test_path)
            features = np.load(
                features_path,
                allow_pickle=True
            )

        # --------------------------------------------------
        # PROCESS RAW DATA
        # --------------------------------------------------

        else:

            print(" No processed dataset found.")
            print(" Running preprocessing pipeline...")

            (
                X_train,
                X_val,
                Y_val,
                X_test,
                Y_test,
                features
            ) = fl.clean_and_split_dataset(
                self.data_path
            )

            print("💾 Saving processed dataset...")

            np.save(X_train_path, X_train)
            np.save(X_val_path, X_val)
            np.save(Y_val_path, Y_val)
            np.save(X_test_path, X_test)
            np.save(Y_test_path, Y_test)
            np.save(features_path, features)

            print("✅ Dataset saved.")

        print("\nDataset ready:")
        print("X_train:", X_train.shape)
        print("X_val:  ", X_val.shape)
        print("X_test: ", X_test.shape)

        return X_train, X_val, Y_val, X_test, Y_test, features