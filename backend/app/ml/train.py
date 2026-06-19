import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.config import settings
from app.ml.autoencoder import device, train_autoencoder
from app.ml.cnn_classifier import LABELS, TransitClassifier, weighted_loss
from app.services.pipeline import phase_fold

LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
AUGMENTATION_METHODS = ["Gaussian noise sigma=0.001", "random time flip", "random flux scale 0.98-1.02"]


def _parse_array(value: str) -> np.ndarray:
    value = value.strip().replace(";", ",")
    return np.asarray([float(x) for x in value.split(",") if x.strip()], dtype=np.float32)


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("label") or row.get("class")
            if label not in LABEL_TO_INDEX:
                continue
            flux = _parse_array(row["flux"])
            time = _parse_array(row["time"]) if row.get("time") else np.arange(flux.size, dtype=np.float32)
            rows.append({"time": time, "flux": flux, "period": float(row["period"]), "label": label})
    if not rows:
        raise ValueError("No usable labeled rows found. Expected CSV columns: flux,label,period[,time]")
    return rows


def augment(flux: np.ndarray) -> np.ndarray:
    out = flux.astype(np.float32).copy()
    out += np.random.normal(0, 0.001, size=out.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        out = out[::-1].copy()
    out *= np.random.uniform(0.98, 1.02)
    return out.astype(np.float32)


def scalar_features(phase_flux: np.ndarray, period: float) -> np.ndarray:
    depth_ppt = abs(float(np.nanmedian(phase_flux) - np.nanmin(phase_flux))) * 1000
    duration_period_ratio = 0.1
    secondary_flag = float(np.nanmin(phase_flux[80:112]) < np.nanmedian(phase_flux) - 0.0005)
    odd_even = abs(float(np.nanmedian(phase_flux[::2]) - np.nanmedian(phase_flux[1::2]))) * 1000
    return np.asarray([depth_ppt, duration_period_ratio, secondary_flag, odd_even], dtype=np.float32)


def split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(rows))
    n_train = int(0.7 * len(rows))
    n_val = int(0.15 * len(rows))
    return [rows[i] for i in idx[:n_train]], [rows[i] for i in idx[n_train:n_train+n_val]], [rows[i] for i in idx[n_train+n_val:]]


def build_classifier_tensors(rows: list[dict], do_augment: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    curves, feats, labels = [], [], []
    for row in rows:
        flux = augment(row["flux"]) if do_augment else row["flux"]
        _, phase_flux = phase_fold(row["time"], flux, row["period"], row["time"][0], bins=128)
        curves.append(phase_flux)
        feats.append(scalar_features(phase_flux, row["period"]))
        labels.append(LABEL_TO_INDEX[row["label"]])
    return torch.tensor(np.stack(curves)).unsqueeze(1), torch.tensor(np.stack(feats)), torch.tensor(labels, dtype=torch.long)


def evaluate(model: TransitClassifier, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct = total = 0
    matrix = np.zeros((4, 4), dtype=int)
    confidences = []
    with torch.no_grad():
        for curves, feats, labels in loader:
            curves, feats, labels = curves.to(device), feats.to(device), labels.to(device)
            probs = torch.softmax(model(curves, feats), dim=1)
            pred = probs.argmax(1)
            confidences.extend(probs.max(1).values.detach().cpu().numpy().tolist())
            correct += int((pred == labels).sum())
            total += labels.numel()
            for truth, guess in zip(labels.detach().cpu().numpy(), pred.detach().cpu().numpy()):
                matrix[int(truth), int(guess)] += 1
    return correct / max(total, 1), matrix, np.asarray(confidences)


def fit_temperature(model: TransitClassifier, loader: DataLoader) -> float:
    _, _, confidences = evaluate(model, loader)
    if confidences.size and np.mean(confidences > 0.95) > 0.8:
        return 1.5
    return 1.0


def train(dataset_path: str, epochs: int = 100, batch_size: int = 64) -> None:
    rows = load_dataset(Path(dataset_path))
    train_rows, val_rows, test_rows = split(rows)
    train_autoencoder([r["flux"] for r in train_rows], epochs=50, batch_size=batch_size)

    x_train, f_train, y_train = build_classifier_tensors(train_rows, True)
    x_val, f_val, y_val = build_classifier_tensors(val_rows, False)
    x_test, f_test, y_test = build_classifier_tensors(test_rows, False)
    train_loader = DataLoader(TensorDataset(x_train, f_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, f_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(x_test, f_test, y_test), batch_size=batch_size)

    counts = [Counter(y_train.numpy().tolist()).get(i, 0) for i in range(4)]
    model = TransitClassifier(dropout=0.4).to(device)
    loss_fn = weighted_loss(counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    best_acc = -1.0
    save_path = settings.model_dir / "classifier_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for curves, feats, labels in train_loader:
            curves, feats, labels = curves.to(device), feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(curves, feats), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.numel()
        val_acc, _, _ = evaluate(model, val_loader)
        scheduler.step(val_acc)
        print(f"classifier epoch={epoch} loss={total_loss / len(y_train):.6f} val_accuracy={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_acc, matrix, confidences = evaluate(model, test_loader)
    temperature = fit_temperature(model, val_loader)
    (settings.model_dir / "temperature.json").write_text(json.dumps({"temperature": temperature}, indent=2))
    eb_idx = LABEL_TO_INDEX["ECLIPSING_BINARY"]
    planet_idx = LABEL_TO_INDEX["PLANET"]
    planet_total = max(matrix[planet_idx, :].sum(), 1)
    planet_to_eb = matrix[planet_idx, eb_idx] / planet_total
    if test_acc < 0.80:
        print("WARNING: classifier test accuracy below 80%; dropout=0.4 and L2 weight_decay=1e-4 are enabled.")
    if planet_to_eb >= 0.20:
        print("WARNING: PLANET->ECLIPSING_BINARY confusion >=20%; inspect secondary_eclipse and odd_even feature distributions.")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"mean_confidence={float(confidences.mean()) if confidences.size else 0:.4f} temperature={temperature:.3f}")
    print("confusion_matrix")
    print(matrix)
    print("classification_report")
    for idx, label in enumerate(LABELS):
        tp = matrix[idx, idx]
        precision = tp / max(matrix[:, idx].sum(), 1)
        recall = tp / max(matrix[idx, :].sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        print(f"{label}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    metadata = {"n_training_samples": len(train_rows), "augmentation_methods": AUGMENTATION_METHODS, "test_accuracy": test_acc}
    (settings.model_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train(args.dataset_path, args.epochs, args.batch_size)
