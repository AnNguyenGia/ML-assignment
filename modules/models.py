"""Các mô hình phân loại tự cài đặt: Gaussian Naive Bayes và MLP (PyTorch)."""
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .config import DEVICE, SEED


def _to_dense(X) -> np.ndarray:
    """Chuyển ma trận thưa (nếu có) thành mảng numpy dense."""
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


# ============================================================================
# Model 1: Gaussian Naive Bayes
# ============================================================================
class NaiveBayes:
    """Gaussian Naive Bayes tự cài đặt (làm mịn phương sai giống scikit-learn).

    ``predict`` xử lý theo batch để không bị tràn RAM với đặc trưng nhiều chiều
    (ví dụ TF-IDF 20.000 chiều).
    """

    def __init__(self, var_smoothing: float = 1e-9, batch_size: int = 64):
        self.var_smoothing = var_smoothing
        self.batch_size = batch_size

    def fit(self, X, y):
        X, y = _to_dense(X).astype(np.float64), np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]
        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.prior_ = np.zeros(n_classes)

        # epsilon tỉ lệ với phương sai lớn nhất để tránh chia cho 0
        epsilon = self.var_smoothing * np.max(np.var(X, axis=0))

        for idx, k in enumerate(self.classes_):
            X_k = X[y == k]
            self.mean_[idx] = X_k.mean(axis=0)
            self.var_[idx] = X_k.var(axis=0) + epsilon
            self.prior_[idx] = X_k.shape[0] / X.shape[0]
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log P(c) + sum_j log N(x_j | mean_cj, var_cj) cho mọi lớp c."""
        out = np.zeros((X.shape[0], len(self.classes_)))
        for idx in range(len(self.classes_)):
            log_norm = -0.5 * np.sum(np.log(2 * np.pi * self.var_[idx]))
            sq = ((X - self.mean_[idx]) ** 2) / self.var_[idx]
            out[:, idx] = np.log(self.prior_[idx]) + log_norm - 0.5 * sq.sum(axis=1)
        return out

    def predict(self, X) -> np.ndarray:
        preds = []
        n = X.shape[0]
        for start in range(0, n, self.batch_size):
            batch = _to_dense(X[start : start + self.batch_size]).astype(np.float64)
            jll = self._joint_log_likelihood(batch)
            preds.append(self.classes_[np.argmax(jll, axis=1)])
        return np.concatenate(preds)


# ============================================================================
# Model 3: MLP
# ============================================================================
class MLP(nn.Module):
    """MLP nhiều lớp: [Linear -> ReLU -> Dropout -> BatchNorm] x N -> Linear.

    ``forward`` trả về *logits* (không softmax) vì ``CrossEntropyLoss`` đã gồm softmax.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (1024, 512, 256, 128, 64, 32),
        dropout: float = 0.5,
        device: str = DEVICE,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout), nn.BatchNorm1d(h)]
            prev = h
        self.features = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev, output_dim)

        self.history_loss: list = []
        self.history_accuracy: list = []
        self.to(device)

    def forward(self, X):
        return self.output_layer(self.features(X))

    def fit(
        self,
        X,
        y,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 64,
        verbose_every: int = 10,
        seed: int = SEED,
    ):
        """Huấn luyện bằng Adam + mini-batch. X có thể là mảng dense hoặc ma trận thưa."""
        y = np.asarray(y)
        n = X.shape[0]
        generator = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.history_loss, self.history_accuracy = [], []
        for epoch in range(1, epochs + 1):
            self.train()
            perm = torch.randperm(n, generator=generator).numpy()
            total_loss, correct, seen = 0.0, 0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) < 2:  # BatchNorm cần >= 2 mẫu trong batch
                    continue
                xb = torch.tensor(_to_dense(X[idx]), dtype=torch.float32, device=self.device)
                yb = torch.tensor(y[idx], dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(idx)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                seen += len(idx)

            self.history_loss.append(total_loss / seen)
            self.history_accuracy.append(correct / seen)
            if verbose_every and (epoch % verbose_every == 0 or epoch == 1 or epoch == epochs):
                print(
                    f"Epoch {epoch:>3}/{epochs} | loss {self.history_loss[-1]:.4f}"
                    f" | acc {self.history_accuracy[-1]:.4f}"
                )
        return self

    @torch.no_grad()
    def predict_proba(self, X, batch_size: int = 256) -> np.ndarray:
        self.eval()
        outs = []
        for start in range(0, X.shape[0], batch_size):
            xb = torch.tensor(
                _to_dense(X[start : start + batch_size]), dtype=torch.float32, device=self.device
            )
            outs.append(torch.softmax(self(xb), dim=1).cpu().numpy())
        return np.vstack(outs)

    def predict(self, X, batch_size: int = 256) -> np.ndarray:
        return self.predict_proba(X, batch_size).argmax(axis=1)
