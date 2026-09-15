"""
Downstream evaluation script for CSCC-PAG (Stage 2 / adapted target domain).

Evaluates a trained CLAM-SB grading model on extracted slide features and
reports Accuracy (ACC), Balanced Accuracy (BACC) and macro F1-Score.

Real usage (features must be pre-extracted with the RSDA-adapted encoder):
    python eval.py \
        --feat_dir /path/to/slide_features \
        --csv /path/to/labels.csv \
        --ckpt /path/to/clam_sb.pth \
        --embed_dim 384

Smoke test with synthetic bags:
    python eval.py --demo
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.clam import CLAM_SB

try:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
except ImportError:  # pragma: no cover
    accuracy_score = balanced_accuracy_score = f1_score = None


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, feats, labels, device, n_classes=3):
    """
    Run bag-level inference for every slide.

    Args:
        model: CLAM_SB.
        feats: list of (N_i, feat_dim) tensors (one per slide).
        labels: list of int, ground-truth class ids.
        device: torch device.
    Returns:
        y_true, y_pred (numpy arrays), y_prob (N, n_classes)
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for feat, lab in zip(feats, labels):
        feat = feat.to(device).float()
        # IMPORTANT: instance_eval=False -> bag-level only (no instance labels)
        logits, Y_prob, Y_hat, _, _ = model(
            feat, instance_eval=False
        )
        y_true.append(int(lab))
        y_pred.append(int(Y_hat.item()))
        y_prob.append(Y_prob.squeeze(0).detach().cpu().numpy())

    return (
        np.array(y_true),
        np.array(y_pred),
        np.stack(y_prob, axis=0),
    )


def compute_metrics(y_true, y_pred):
    """ACC, BACC, macro-F1 (+ weighted F1)."""
    if accuracy_score is None:
        raise ImportError("scikit-learn required: pip install scikit-learn")
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


# ---------------------------------------------------------------------------
def load_real(feat_dir, csv_path, label_map=None):
    """
    Load slide features (.pt / .h5) and labels from a csv.

    csv must contain at least: slide_id, label
    feature file naming: <slide_id>.pt  (or .h5 with key 'features')
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    feats, labels, ids = [], [], []

    for _, row in df.iterrows():
        sid = str(row["slide_id"]).strip()
        pt = os.path.join(feat_dir, sid + ".pt")
        h5 = os.path.join(feat_dir, sid + ".h5")

        if os.path.exists(pt):
            f = torch.load(pt, map_location="cpu")
            if isinstance(f, dict):
                f = f.get("features", list(f.values())[0])
        elif os.path.exists(h5):
            import h5py
            with h5py.File(h5, "r") as fh:
                f = torch.from_numpy(fh["features"][:])
        else:
            print(f"  [warn] feature not found, skip: {sid}")
            continue

        lab = row["label"]
        if label_map is not None:
            lab = label_map[lab]

        feats.append(f.float())
        labels.append(int(lab))
        ids.append(sid)

    return feats, labels, ids


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CSCC-PAG downstream evaluation")
    ap.add_argument("--feat_dir", type=str, default=None,
                    help="dir with per-slide feature .pt/.h5 files")
    ap.add_argument("--csv", type=str, default=None,
                    help="csv with columns [slide_id, label]")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="trained CLAM-SB checkpoint")
    ap.add_argument("--n_classes", type=int, default=3)
    ap.add_argument("--embed_dim", type=int, default=384,
                    help="patch feature dim (384 for ViT-S, 1024 if used)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic bags, no data needed")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---------------- model ----------------
    model = CLAM_SB(
        n_classes=args.n_classes,
        dropout=0.0,          # deterministic at eval
        subtyping=True,
        embed_dim=args.embed_dim,
    ).to(device)

    if args.demo:
        # -------- synthetic bags --------
        torch.manual_seed(0)
        n_slides, feat_dim = 24, args.embed_dim
        feats = [torch.randn(torch.randint(50, 200, (1,)).item(), feat_dim)
                 for _ in range(n_slides)]
        labels = [int(i % args.n_classes) for i in range(n_slides)]
        print(f"[demo] {n_slides} synthetic slides, random-init CLAM-SB.")
    else:
        if not (args.feat_dir and args.csv and args.ckpt):
            raise ValueError(
                "Real eval needs --feat_dir, --csv and --ckpt (or use --demo)."
            )
        # default Chinese grading label map (Well / Moderately / Poorly)
        label_map = {"高分化": 0, "中分化": 1, "低分化": 2,
                     "Well": 0, "Moderately": 1, "Poorly": 2}
        feats, labels, ids = load_real(args.feat_dir, args.csv, label_map)

        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[eval] loaded {len(feats)} slides from {args.feat_dir}")

    # ---------------- inference ----------------
    y_true, y_pred, y_prob = evaluate(model, feats, labels, device,
                                      n_classes=args.n_classes)
    m = compute_metrics(y_true, y_pred)

    print("\n" + "=" * 60)
    print("CSCC-PAG downstream evaluation (target domain)")
    print("=" * 60)
    print(f"  slides        : {len(y_true)}")
    print(f"  ACC           : {m['acc']:.4f}")
    print(f"  BACC          : {m['bacc']:.4f}")
    print(f"  F1 (macro)    : {m['f1_macro']:.4f}")
    print(f"  F1 (weighted) : {m['f1_weighted']:.4f}")
    print("=" * 60)

    # per-class support
    for c in range(args.n_classes):
        n_true = int((y_true == c).sum())
        n_pred = int((y_pred == c).sum())
        print(f"  class {c}: true={n_true}, pred={n_pred}")


if __name__ == "__main__":
    main()
