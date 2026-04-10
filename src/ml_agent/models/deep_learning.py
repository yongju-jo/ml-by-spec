"""
Deep Learning 모델 래퍼 (PyTorch 기반 MLP).
TabNet, FT-Transformer는 추후 추가 예정.
spec: 03_model_spec.md §1.3
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_agent.models.base import BaseModel, Task


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPModel(BaseModel):
    """
    PyTorch MLP.

    Parameters
    ----------
    hidden_dims : list[int]
        각 hidden layer의 유닛 수. 기본 [256, 128].
    dropout : float
        Dropout 비율. 기본 0.1.
    lr : float
        학습률. 기본 1e-3.
    batch_size : int
        기본 256.
    max_epochs : int
        기본 100. Early stopping으로 조기 종료 가능.
    patience : int
        Early stopping patience. 기본 10.
    """

    name = "mlp"
    family = "dl"

    def __init__(
        self,
        task: Task,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(task)
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self._net: Optional[_MLP] = None
        self._n_classes: int = 1
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPModel":
        torch.manual_seed(self.seed)
        X_np = X.values.astype(np.float32)
        y_np = y.values

        if self.task == "classification":
            classes = np.unique(y_np)
            self._n_classes = len(classes)
            y_np = y_np.astype(np.int64)
        else:
            y_np = y_np.astype(np.float32)

        output_dim = self._n_classes if self._n_classes > 2 else 1
        self._net = _MLP(X_np.shape[1], self.hidden_dims, output_dim, self.dropout).to(self._device)

        X_t = torch.tensor(X_np, device=self._device)
        y_t = torch.tensor(y_np, device=self._device)

        # 90/10 내부 split for early stopping
        n_val = max(1, int(len(X_t) * 0.1))
        X_tr, X_val = X_t[:-n_val], X_t[-n_val:]
        y_tr, y_val = y_t[:-n_val], y_t[-n_val:]

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = (
            nn.CrossEntropyLoss() if self._n_classes > 2
            else nn.BCEWithLogitsLoss() if self.task == "classification"
            else nn.MSELoss()
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_cnt = 0

        for epoch in range(self.max_epochs):
            self._net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self._net(xb).squeeze(-1)
                if self.task == "classification" and self._n_classes == 2:
                    loss = criterion(out, yb.float())
                else:
                    loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            # Early stopping
            self._net.eval()
            with torch.no_grad():
                val_out = self._net(X_val).squeeze(-1)
                if self.task == "classification" and self._n_classes == 2:
                    val_loss = criterion(val_out, y_val.float()).item()
                else:
                    val_loss = criterion(val_out, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break

        self._net.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._net.eval()
        X_t = torch.tensor(X.values.astype(np.float32), device=self._device)
        with torch.no_grad():
            out = self._net(X_t).squeeze(-1)

        if self.task == "classification":
            if self._n_classes > 2:
                return out.argmax(dim=1).cpu().numpy()
            return (torch.sigmoid(out) > 0.5).long().cpu().numpy()
        return out.cpu().numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task == "regression":
            raise NotImplementedError
        self._net.eval()
        X_t = torch.tensor(X.values.astype(np.float32), device=self._device)
        with torch.no_grad():
            out = self._net(X_t).squeeze(-1)
        if self._n_classes > 2:
            return torch.softmax(out, dim=1).cpu().numpy()
        pos = torch.sigmoid(out).cpu().numpy()
        return np.column_stack([1 - pos, pos])

    def get_params(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }
