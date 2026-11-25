"""Classical estimators and a compact EEGNet-inspired neural network."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_classical_models():
    """Build estimators; feature scaling is fitted within each CV training fold."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def make_csp_lda():
    """Return a fold-fitted CSP + LDA pipeline for raw EEG epochs."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return Pipeline(
        [
            ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True)),
            ("classifier", LinearDiscriminantAnalysis()),
        ]
    )


class EEGNet:
    """Small EEGNet-style model and training helper using CPU by default."""

    @staticmethod
    def create_model(n_channels: int, n_times: int, n_classes: int = 2):
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
                    nn.BatchNorm2d(8),
                    nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ELU(),
                    nn.AvgPool2d((1, 4)),
                    nn.Dropout(0.25),
                    nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
                    nn.Conv2d(16, 16, 1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ELU(),
                    nn.AvgPool2d((1, 8)),
                    nn.Dropout(0.25),
                    nn.Flatten(),
                )
                with torch.no_grad():
                    flattened = self.features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
                self.classifier = nn.Linear(flattened, n_classes)

            def forward(self, inputs):
                return self.classifier(self.features(inputs))

        return Network()

    @classmethod
    def fit_predict(
        cls,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        test_data: np.ndarray,
        epochs: int = 30,
        seed: int = 42,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit on training participants and return test labels/probabilities."""
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        if epochs < 1:
            raise ValueError("EEGNet epochs must be at least one.")
        torch.manual_seed(seed)
        torch.set_num_threads(1)

        train_data = np.asarray(train_data, dtype=np.float32)
        test_data = np.asarray(test_data, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.int64)
        if (
            train_data.ndim != 3
            or test_data.ndim != 3
            or train_data.shape[1:] != test_data.shape[1:]
            or len(train_data) == 0
            or len(test_data) == 0
            or len(train_labels) != len(train_data)
        ):
            raise ValueError("EEGNet expects compatible nonempty train and test epoch arrays.")
        if set(np.unique(train_labels)) != {0, 1}:
            raise ValueError("EEGNet training labels must contain classes 0 and 1.")
        if not np.isfinite(train_data).all() or not np.isfinite(test_data).all():
            raise ValueError("EEGNet input data contains non-finite values.")
        mean = train_data.mean(axis=(0, 2), keepdims=True)
        scale = train_data.std(axis=(0, 2), keepdims=True)
        scale[scale < 1e-8] = 1.0
        train_data = (train_data - mean) / scale
        test_data = (test_data - mean) / scale

        train_tensor = torch.from_numpy(train_data[:, None, :, :])
        labels_tensor = torch.from_numpy(train_labels)
        loader = DataLoader(
            TensorDataset(train_tensor, labels_tensor),
            batch_size=min(batch_size, len(train_labels)),
            shuffle=True,
        )
        model = cls.create_model(train_data.shape[1], train_data.shape[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            for batch, labels in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch), labels)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(test_data[:, None, :, :]))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities.argmax(axis=1), probabilities[:, 1]
