from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LightCurveAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(size=128, mode="linear", align_corners=False),
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=256, mode="linear", align_corners=False),
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=512, mode="linear", align_corners=False),
            nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.decoder(self.encoder(x))
        return y[..., : x.shape[-1]]


def segment_flux(flux: np.ndarray, window: int = 512, overlap: float = 0.5) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float32)
    step = max(1, int(window * (1 - overlap)))
    if flux.size < window:
        padded = np.pad(flux, (0, window - flux.size), mode="edge")
        return padded.reshape(1, window)
    starts = list(range(0, flux.size - window + 1, step))
    if starts[-1] != flux.size - window:
        starts.append(flux.size - window)
    return np.stack([flux[start : start + window] for start in starts]).astype(np.float32)


def train_autoencoder(flux_arrays: list[np.ndarray], epochs: int = 50, lr: float = 1e-3, batch_size: int = 64, save_path: Path | None = None) -> LightCurveAutoencoder:
    save_path = save_path or settings.model_dir / "autoencoder_best.pt"
    segments = np.concatenate([segment_flux(flux) for flux in flux_arrays], axis=0)
    dataset = TensorDataset(torch.tensor(segments).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LightCurveAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = model.loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch.size(0)
        avg = total / len(dataset)
        print(f"autoencoder epoch={epoch} loss={avg:.6f}")
        if avg < best:
            best = avg
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
    return model


def load_autoencoder() -> LightCurveAutoencoder:
    model = LightCurveAutoencoder().to(device)
    path = settings.model_dir / "autoencoder_best.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def denoise(flux_norm: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux_norm, dtype=np.float32)
    window = 512
    step = 256
    model = load_autoencoder()
    output = np.zeros_like(flux, dtype=np.float32)
    weights = np.zeros_like(flux, dtype=np.float32)
    if flux.size == 0:
        return flux
    if flux.size < window:
        padded = np.pad(flux, (0, window - flux.size), mode="edge")
        with torch.no_grad():
            recon = model(torch.tensor(padded, device=device).view(1, 1, window)).view(-1).detach().cpu().numpy()[: flux.size]
        return recon.astype(np.float32)
    starts = list(range(0, flux.size - window + 1, step))
    if starts[-1] != flux.size - window:
        starts.append(flux.size - window)
    with torch.no_grad():
        for start in starts:
            chunk = flux[start : start + window]
            recon = model(torch.tensor(chunk, device=device).view(1, 1, window)).view(-1).detach().cpu().numpy()
            output[start : start + window] += recon[:window]
            weights[start : start + window] += 1
    weights[weights == 0] = 1
    denoised = output / weights
    if denoised.shape[0] != flux.shape[0]:
        raise RuntimeError("Autoencoder denoise length mismatch")
    return denoised
