"""
evaluate.py

Loads a trained checkpoint and reports classification metrics (accuracy, per-class
precision/recall/F1, confusion matrix) on windowed sequences built from a features CSV.
This is the same evaluation family the base paper uses (PASCAL VOC mAP / precision /
recall), adapted to the sequence-classification setting.

Usage:
    python evaluate.py --checkpoint models/best_model.pt --features data/processed/features.csv
"""

import argparse
import sys
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from dataset import build_windows, normalize_features, personalize_dataframe, DrowsinessSequenceDataset  # noqa: E402
from model import DrowsinessLSTM  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


CLASS_NAMES = ["Alert", "Drowsy", "Highly Drowsy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    df = pd.read_csv(args.features)
    if ckpt.get("personalize", True):
        df = personalize_dataframe(df)
    X, y = build_windows(df, cfg["window"]["sequence_length"], cfg["window"]["stride"])
    mean = np.array(ckpt["feature_mean"])
    std = np.array(ckpt["feature_std"])
    X_norm, _, _ = normalize_features(X, mean, std)

    ds = DrowsinessSequenceDataset(X_norm, y)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"])

    model = DrowsinessLSTM(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        bidirectional=cfg["model"]["bidirectional"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
