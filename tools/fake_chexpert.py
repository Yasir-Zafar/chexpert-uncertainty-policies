"""Build a tiny CheXpert-shaped dataset for local development (not part of the submission).

The layout mirrors the Kaggle CheXpert-small mirror: ``<out>/train.csv``, ``<out>/valid.csv`` and
JPEGs under ``<out>/train/patientXXXXX/studyN/viewK_<frontal|lateral>.jpg``. The ``Path`` column
keeps the official ``CheXpert-v1.0-small/`` prefix, so the notebook's prefix stripping is
exercised.

Labels follow the official encoding: train uses 1.0 / 0.0 / -1.0 / blank, valid uses 0.0 / 1.0
only. Uncertainty rates are higher than in the real data so that a ~10% validation split still
holds at least 50 uncertain images per studied pathology (the qualitative subset needs that).
Each studied pathology brightens its own image region when positive, so a model trained on
fake data can beat chance.

Usage: python tools/fake_chexpert.py --out fake_data/chexpert
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PREFIX = "CheXpert-v1.0-small"
OBSERVATIONS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
COLUMNS = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", *OBSERVATIONS]
# Uncertain rate per observation; the five studied pathologies get explicit rates.
UNCERTAIN_RATE = {
    "Atelectasis": 0.15,
    "Cardiomegaly": 0.10,
    "Consolidation": 0.12,
    "Edema": 0.10,
    "Pleural Effusion": 0.10,
}
# Image region (row slice, col slice as fractions) brightened when a studied pathology is positive.
SIGNAL_REGION = {
    "Atelectasis": (0.55, 0.85, 0.10, 0.40),
    "Cardiomegaly": (0.45, 0.80, 0.35, 0.65),
    "Consolidation": (0.20, 0.50, 0.10, 0.40),
    "Edema": (0.20, 0.50, 0.60, 0.90),
    "Pleural Effusion": (0.75, 0.95, 0.60, 0.90),
}


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("fake_data/chexpert"))
    p.add_argument("--patients", type=int, default=4000, help="training patients")
    p.add_argument("--valid-patients", type=int, default=60)
    p.add_argument("--height", type=int, default=80)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def train_label(rng: np.random.Generator, obs: str) -> str:
    """One training label in the official CSV encoding: '1.0', '0.0', '-1.0' or blank."""
    u = UNCERTAIN_RATE.get(obs, 0.05)
    choice = rng.choice(4, p=[0.15, 0.20, u, 0.65 - u])
    return ["1.0", "0.0", "-1.0", ""][choice]


def valid_label(rng: np.random.Generator) -> str:
    """One radiologist-consensus label: '1.0' or '0.0', never uncertain or blank."""
    return "1.0" if rng.random() < 0.25 else "0.0"


def make_image(rng: np.random.Generator, labels: dict[str, str], h: int, w: int) -> np.ndarray:
    """Noisy grayscale image; positive (and half of uncertain) findings brighten their region."""
    img = rng.normal(110, 25, size=(h, w))
    img += np.linspace(-20, 20, w)[None, :]
    for obs, (r0, r1, c0, c1) in SIGNAL_REGION.items():
        value = labels[obs]
        if value == "1.0" or (value == "-1.0" and rng.random() < 0.5):
            img[int(r0 * h) : int(r1 * h), int(c0 * w) : int(c1 * w)] += 40
    return np.clip(img, 0, 255).astype(np.uint8)


def write_split(
    rng: np.random.Generator,
    out: Path,
    split: str,
    patient_ids: range,
    h: int,
    w: int,
) -> int:
    """Write ``<split>.csv`` and its JPEGs; return the number of rows."""
    rows = []
    for pid in patient_ids:
        sex = rng.choice(["Male", "Female"])
        age = int(rng.integers(18, 91))
        n_studies = 1 if split == "valid" else int(rng.integers(1, 4))
        for study in range(1, n_studies + 1):
            views = ["frontal"]
            if rng.random() < 0.4:
                views.append("lateral")
            if split == "train" and rng.random() < 0.1:
                views.insert(1, "frontal")  # some studies have two frontal views
            for k, view in enumerate(views, start=1):
                if split == "valid":
                    labels = {obs: valid_label(rng) for obs in OBSERVATIONS}
                else:
                    labels = {obs: train_label(rng, obs) for obs in OBSERVATIONS}
                if view == "frontal":
                    # A few frontal rows have a view position other than AP/PA, as in the real CSV.
                    ap_pa = "LL" if rng.random() < 0.01 else rng.choice(["AP", "PA"], p=[0.7, 0.3])
                else:
                    ap_pa = ""
                rel = f"{split}/patient{pid:05d}/study{study}/view{k}_{view}.jpg"
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), make_image(rng, labels, h, w))
                rows.append(
                    {
                        "Path": f"{PREFIX}/{rel}",
                        "Sex": sex,
                        "Age": age,
                        "Frontal/Lateral": view.capitalize(),
                        "AP/PA": ap_pa,
                        **labels,
                    }
                )
    with open(out / f"{split}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    """Generate the fake dataset."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    n_train = write_split(
        rng, args.out, "train", range(1, args.patients + 1), args.height, args.width
    )
    first_valid = args.patients + 1
    n_valid = write_split(
        rng,
        args.out,
        "valid",
        range(first_valid, first_valid + args.valid_patients),
        args.height,
        args.width,
    )
    print(f"fake CheXpert written to {args.out}: train.csv {n_train} rows, valid.csv {n_valid}")


if __name__ == "__main__":
    main()
