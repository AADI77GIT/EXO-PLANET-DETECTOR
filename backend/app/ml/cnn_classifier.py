import json
import numpy as np
import torch
from torch import nn

from app.config import settings

LABELS = ["PLANET", "ECLIPSING_BINARY", "FALSE_POSITIVE", "STARSPOT"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransitClassifier(nn.Module):
    def __init__(self, dropout: float = 0.4) -> None:
        super().__init__()
        self.curve_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.feature_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )

    def forward(self, curve: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        a = self.curve_branch(curve)
        b = self.feature_branch(features)
        return self.classifier(torch.cat([a, b], dim=1))


def weighted_loss(class_counts: list[int]) -> nn.CrossEntropyLoss:
    counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=weights)


def load_temperature() -> float:
    path = settings.model_dir / "temperature.json"
    if not path.exists():
        return 1.0
    try:
        value = float(json.loads(path.read_text()).get("temperature", 1.0))
        return max(value, 1e-3)
    except Exception:
        return 1.0


class ClassifierService:
    def __init__(self) -> None:
        self.model = TransitClassifier().to(device)
        self.loaded = False
        self.temperature = load_temperature()
        path = settings.model_dir / "classifier_best.pt"
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=device))
            self.loaded = True
        self.model.eval()

    def predict(self, phase_folded: np.ndarray, scalar_features: np.ndarray) -> dict:
        if len(phase_folded) != 128:
            raise ValueError("phase_folded curve must have length 128")
        curve = torch.tensor(phase_folded, dtype=torch.float32, device=device).view(1, 1, 128)
        features = torch.tensor(scalar_features, dtype=torch.float32, device=device).view(1, 4)
        with torch.no_grad():
            logits = self.model(curve, features) / self.temperature
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return {"label": LABELS[idx], "confidence": float(probs[idx]), "probabilities": dict(zip(LABELS, map(float, probs)))}


_classifier: ClassifierService | None = None


def get_classifier() -> ClassifierService:
    global _classifier
    if _classifier is None:
        _classifier = ClassifierService()
    return _classifier


def initialize_classifier() -> None:
    get_classifier()
