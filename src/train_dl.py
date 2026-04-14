# train_dl.py
# ─────────────────────────────────────────────────────────────────────────────
# A fairly simple feed-forward network — nothing fancy.
# Tried adding more layers but they didn't help on this dataset size.
# The key things that actually improved performance:
#   - BatchNorm before activation (not after — learned this the hard way)
#   - Gradient clipping to prevent the occasional exploding gradient
#   - Early stopping with a patience of 10 to avoid overfitting
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, classification_report
import json
import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR         = 1e-3
MAX_EPOCHS = 120
PATIENCE   = 10


class ClinicalNet(nn.Module):
    """
    Two hidden layers with BatchNorm + Dropout.
    The architecture is deliberately conservative — this is a ~750 row dataset
    and we're not trying to win a Kaggle competition, just build something
    solid that generalizes well.
    """

    def __init__(self, n_features):
        super().__init__()

        self.net = nn.Sequential(
            # first block
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # second block
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            # output
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def make_loaders(X_train, y_train, X_test, y_test):
    def to_tensor(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    train_ds = to_tensor(X_train, y_train)
    test_ds  = to_tensor(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, test_loader):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

    best_val_loss = float("inf")
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "val_auc": []}

    log.info(f"Training on {DEVICE}  |  max {MAX_EPOCHS} epochs  |  patience {PATIENCE}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_losses, preds, labels = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out  = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                preds.extend(out.cpu().numpy())
                labels.extend(yb.cpu().numpy())

        t_loss = np.mean(batch_losses)
        v_loss = np.mean(val_losses)
        v_auc  = roc_auc_score(labels, preds)
        scheduler.step(v_loss)

        history["train_loss"].append(round(t_loss, 5))
        history["val_loss"].append(round(v_loss, 5))
        history["val_auc"].append(round(v_auc, 4))

        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:3d}  |  "
                f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_auc={v_auc:.4f}"
            )

        # ── early stopping ─────────────────────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            no_improve    = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/dl_best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch}")
                break

    return history


def evaluate(model, test_loader):
    model.load_state_dict(torch.load("models/dl_best.pt", map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(DEVICE))
            all_preds.extend(out.cpu().numpy())
            all_labels.extend(yb.numpy())

    binary = [1 if p >= 0.5 else 0 for p in all_preds]
    auc    = roc_auc_score(all_labels, all_preds)
    rep    = classification_report(all_labels, binary, output_dict=True)

    metrics = {
        "auc_roc":   round(auc, 4),
        "accuracy":  round(rep["accuracy"], 4),
        "f1_weighted": round(rep["weighted avg"]["f1-score"], 4),
        "recall_positive": round(rep["1"]["recall"], 4),
    }
    log.info(f"Final test AUC:  {metrics['auc_roc']}")
    log.info(f"Final accuracy:  {metrics['accuracy']}")
    return metrics


def save_metadata(metrics, history, n_features, n_epochs):
    meta = {
        "model":        "ClinicalNet (PyTorch)",
        "version":      "1.0.0",
        "trained_at":   datetime.utcnow().isoformat(),
        "architecture": "Linear(128)→BN→ReLU→Dropout(0.3)→Linear(64)→BN→ReLU→Dropout(0.25)→Linear(1)→Sigmoid",
        "n_features":   n_features,
        "epochs_run":   n_epochs,
        "best_val_auc": max(history["val_auc"]),
        "test_metrics": metrics,
    }
    with open("models/dl_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("DL metadata saved → models/dl_metadata.json")
    return meta


def run():
    log.info("=" * 55)
    log.info("DEEP LEARNING TRAINING")
    log.info("=" * 55)

    X_train = pd.read_csv("data/X_train.csv").values.astype(np.float32)
    X_test  = pd.read_csv("data/X_test.csv").values.astype(np.float32)
    y_train = pd.read_csv("data/y_train.csv").squeeze().values.astype(np.float32)
    y_test  = pd.read_csv("data/y_test.csv").squeeze().values.astype(np.float32)

    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test)

    n_features = X_train.shape[1]
    model      = ClinicalNet(n_features).to(DEVICE)
    log.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    history  = train(model, train_loader, test_loader)
    metrics  = evaluate(model, test_loader)
    meta     = save_metadata(metrics, history, n_features, len(history["train_loss"]))

    log.info("Done.\n")
    return model, meta


if __name__ == "__main__":
    run()
# training loop 
# scheduler 
