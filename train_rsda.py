"""
RSDA domain-adaptation training script (Stage 3 of CSCC-PAG).

Real-data usage:
    python train_rsda.py \
        --source_img_dir /path/to/source_patches \
        --target_img_dir /path/to/target_patches \
        --pretrained /path/to/spc_pretrained.pth \
        --epochs 50 --batch_size 32

Smoke test without any local data (recommended first run):
    python train_rsda.py --demo

The --demo flag synthesizes dummy 256x256 patch tensors, builds a tiny
K-means buffer and runs forward + backward passes end-to-end.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.rsda import RSDA
from utils.kmeans_buffer import (
    build_prototype_buffer,
    extract_source_features,
    MixedRatioDataset,
    PrototypeBufferDataset,
)


# ---------------------------------------------------------------------------
def build_demo_data(device, n_source=200, n_target=360, feat_dim=384):
    """
    Synthesize dummy data so the pipeline runs with zero local data.

    Returns:
        feats: (n_source, feat_dim) random 'SPC' features
        src_items: list of random image tensors (3,256,256)
        target_ds: TensorDataset of random target images
    """
    torch.manual_seed(0)
    feats = torch.randn(n_source, feat_dim).numpy().astype("float32")
    src_items = [torch.rand(3, 256, 256) for _ in range(n_source)]
    tgt_imgs = torch.rand(n_target, 3, 256, 256)
    target_ds = TensorDataset(tgt_imgs)
    return feats, src_items, target_ds


def build_real_data(args, device, transform):
    """
    Build source/target loaders from image folders.
    Assumes standard `ImageFolder` layout (any sub-dir structure is fine,
    labels are unused because RSDA is self-supervised).
    """
    from torchvision.datasets import ImageFolder

    src_ds = ImageFolder(args.source_img_dir, transform=transform)
    tgt_ds = ImageFolder(args.target_img_dir, transform=transform)

    src_loader = DataLoader(
        src_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    # keep paths aligned with the (unshuffled) feature order
    src_paths = [s[0] for s in src_ds.samples]

    target_ds = TensorDataset  # placeholder; we use ImageFolder directly below
    return src_loader, src_paths, tgt_ds


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RSDA domain adaptation training")

    # ---- data ----
    ap.add_argument("--source_img_dir", type=str, default=None,
                    help="source-domain patch folder (ImageFolder layout)")
    ap.add_argument("--target_img_dir", type=str, default=None,
                    help="target-domain patch folder (ImageFolder layout)")
    ap.add_argument("--demo", action="store_true",
                    help="run with synthetic dummy patches (no data needed)")

    # ---- model ----
    ap.add_argument("--arch", type=str, default="vit_small",
                    choices=["vit_small", "vit_base"])
    ap.add_argument("--mask_ratio", type=float, default=0.75,
                    help="MIM mask ratio (paper: 0.75)")
    ap.add_argument("--lambda_fd", type=float, default=0.1,
                    help="weight of feature-distillation loss (paper: 0.1)")
    ap.add_argument("--pretrained", type=str, default=None,
                    help="SPC pretrained checkpoint (teacher & student init)")

    # ---- prototype buffer ----
    ap.add_argument("--k", type=int, default=100, help="K-means clusters")
    ap.add_argument("--per_cluster", type=int, default=500,
                    help="prototypes per cluster (K*per_cluster=50k)")

    # ---- optimization ----
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)

    # ---- misc ----
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--num_demo_steps", type=int, default=2,
                    help="steps to run in --demo mode")
    ap.add_argument("--save_every", type=int, default=10)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("CSCC-PAG  |  Stage-3 RSDA domain adaptation")
    print(f"device={device}  arch={args.arch}  mask_ratio={args.mask_ratio}  "
          f"lambda_fd={args.lambda_fd}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) Build model
    # ------------------------------------------------------------------
    model = RSDA(
        arch=args.arch,
        mask_ratio=args.mask_ratio,
        lambda_fd=args.lambda_fd,
    ).to(device)

    if args.pretrained is not None and os.path.isfile(args.pretrained):
        missing_t, missing_s = model.load_pretrained_weights(args.pretrained)
        print(f"[init] loaded SPC weights from {args.pretrained} "
              f"(teacher missing={len(missing_t)}, student missing={len(missing_s)})")
    else:
        print("[init] random init (no --pretrained given / file missing)")

    # ------------------------------------------------------------------
    # 2) Build prototype buffer + mixed loader
    # ------------------------------------------------------------------
    if args.demo:
        # -------- synthetic data path (no I/O, no real images) --------
        print("\n[demo] building synthetic dummy patches ...")
        feats, src_items, target_ds = build_demo_data(device)
        selected, info = build_prototype_buffer(
            feats, src_items, k=5, per_cluster=10, seed=0
        )
        print(f"[demo] buffer: {info}")

        src_ds = PrototypeBufferDataset(selected)
        mixed_ds = MixedRatioDataset(src_ds, target_ds)
        loader = DataLoader(mixed_ds, batch_size=8, shuffle=False)
        max_steps = args.num_demo_steps

    else:
        # -------- real data path --------
        if args.source_img_dir is None or args.target_img_dir is None:
            raise ValueError(
                "Real training needs --source_img_dir and --target_img_dir "
                "(or use --demo)."
            )
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

        src_loader, src_paths, tgt_ds = build_real_data(args, device, transform)

        print("\n[data] extracting source features with the frozen SPC encoder ...")
        feats, _ = extract_source_features(model.teacher, src_loader, device)
        feats = feats[: len(src_paths)] if len(src_paths) == feats.shape[0] else feats

        selected, info = build_prototype_buffer(
            feats, src_paths, k=args.k, per_cluster=args.per_cluster,
            seed=0, save_path=os.path.join(args.out_dir, "buffer.json"),
        )
        print(f"[data] buffer: {info}")

        src_ds = PrototypeBufferDataset(selected, transform=transform)
        mixed_ds = MixedRatioDataset(src_ds, tgt_ds)
        loader = DataLoader(
            mixed_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        max_steps = None

    print(f"[data] mixed dataset size = {len(mixed_ds)} "
          f"(source:target = 1:9 inside every batch)")

    # ------------------------------------------------------------------
    # 3) Optimizer
    # ------------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay
    )

    # ------------------------------------------------------------------
    # 4) Training loop
    # ------------------------------------------------------------------
    model.train()
    model.teacher.eval()   # teacher always frozen / eval

    step = 0
    t0 = time.time()
    print("\n[train] starting ...")

    for epoch in range(args.epochs if max_steps is None else 1):
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)

            out = model(x)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            step += 1
            if step % 10 == 0 or step == 1:
                print(
                    f"  step {step:5d} | loss={out['loss'].item():.4f} "
                    f"(mim={out['loss_mim'].item():.4f}, "
                    f"fd={out['loss_fd'].item():.4f}) "
                    f"| {time.time() - t0:.1f}s"
                )

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

        # periodic checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = os.path.join(args.out_dir, f"rsda_epoch{epoch+1}.pth")
            torch.save(
                {"student": model.student.state_dict(),
                 "decoder": model.decoder.state_dict()},
                ckpt,
            )
            print(f"  [save] {ckpt}")

    # ------------------------------------------------------------------
    # 5) Demo summary / final save
    # ------------------------------------------------------------------
    final = os.path.join(args.out_dir, "rsda_final.pth")
    torch.save(
        {"student": model.student.state_dict(),
         "decoder": model.decoder.state_dict()},
        final,
    )
    print(f"\n[done] forward+backward OK. Saved: {final}")
    if args.demo:
        print("[demo] smoke test PASSED - run without --demo for real training.")


if __name__ == "__main__":
    main()
