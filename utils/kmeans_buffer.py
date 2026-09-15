"""
K-means Prototype Buffer & mixed (1:9) sampling for RSDA.

Pipeline:
    1. Encode every source-domain patch with the (frozen) SPC encoder -> features.
    2. Run K-means (K=100) in the source feature space.
    3. From each cluster, keep the 500 patches closest to the centroid
       -> buffer capacity 100 * 500 = 50,000 prototype patches.
    4. During RSDA training, each mini-batch mixes source prototypes and
       unlabeled target patches at a 1:9 ratio.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover
    KMeans = None


# ---------------------------------------------------------------------------
# 1) Feature extraction over the source domain
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_source_features(encoder, dataloader, device="cuda"):
    """
    Encode all source patches with the frozen SPC encoder.

    Args:
        encoder: SPCViT (or any module with .forward_tokens()).
        dataloader: yields (images, ...) batches of (B, 3, 256, 256).
        device: compute device.
    Returns:
        feats: (M, D) float32 numpy array of mean-pooled patch features.
        items: list of identifiers (paths) aligned with feats rows.
    """
    encoder.eval()
    feats, items = [], []

    for batch in dataloader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device, non_blocking=True)
        f = encoder.extract_feature(imgs)            # (B, D)
        feats.append(f.detach().cpu().numpy())

        if isinstance(batch, (list, tuple)) and len(batch) > 2:
            items.extend(list(batch[2]))             # batch provides paths

    if len(feats) == 0:
        raise RuntimeError("Empty source dataloader - cannot build buffer.")

    feats = np.concatenate(feats, axis=0).astype(np.float32)
    return feats, items


# ---------------------------------------------------------------------------
# 2) K-means + representative prototype selection
# ---------------------------------------------------------------------------
def build_prototype_buffer(feats, items, k=100, per_cluster=500, seed=0,
                           save_path=None):
    """
    Cluster source features and select the most representative patches.

    Args:
        feats: (M, D) source features.
        items: list of length M, each element is either an image path (str)
               or an already-loaded image tensor (used in --demo mode).
        k: number of clusters (paper: 100).
        per_cluster: prototypes per cluster (paper: 500).
        seed: random seed.
        save_path: optional .json path to dump the selected item list.

    Returns:
        selected: list of selected items (subset of `items`),
                  length <= k * per_cluster (limited by cluster population).
        info: dict with cluster statistics.
    """
    if KMeans is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    M = feats.shape[0]
    k = int(min(k, M))
    per_cluster = int(min(per_cluster, max(1, M // k)))

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(feats)
    centers = kmeans.cluster_centers_

    selected, cluster_sizes = [], []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        # distance of each member to its centroid
        d = np.linalg.norm(feats[idx] - centers[c], axis=1)
        order = np.argsort(d)                      # ascending: closest first
        take = idx[order[:per_cluster]]
        selected.extend([items[i] for i in take])
        cluster_sizes.append(len(take))

    info = {
        "num_source_samples": int(M),
        "num_clusters": k,
        "per_cluster": per_cluster,
        "buffer_size": len(selected),
        "cluster_sizes": cluster_sizes,
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(
                {
                    "items": [s if isinstance(s, str) else f"demo_{i}"
                              for i, s in enumerate(selected)],
                    "info": info,
                },
                f,
                indent=2,
            )

    return selected, info


# ---------------------------------------------------------------------------
# 3) Buffer dataset
# ---------------------------------------------------------------------------
class PrototypeBufferDataset(Dataset):
    """
    Dataset over the selected prototype patches.

    Each item is either:
      - a path (str)  -> lazily loaded from disk (real data), or
      - a tensor      -> returned directly (synthetic / --demo mode).
    """

    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]

        if torch.is_tensor(it):
            img = it.float()
        else:
            from PIL import Image
            img = Image.open(it).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                import torchvision.transforms as T
                img = T.ToTensor()(img)

        return img


# ---------------------------------------------------------------------------
# 4) Mixed 1:9 sampling dataset
# ---------------------------------------------------------------------------
class MixedRatioDataset(Dataset):
    """
    Interleave source prototypes and (unlabeled) target patches at 1:9.

    Index mapping: for every 10 consecutive indices, index 0 -> source,
    indices 1..9 -> target. This yields exactly a 1:9 source:target ratio
    inside every mini-batch produced by a standard DataLoader.

    Length = 10 * min(len(source), len(target) // 9) so both sides stay in range.
    """

    def __init__(self, source_ds, target_ds):
        self.source_ds = source_ds
        self.target_ds = target_ds
        n_src = len(source_ds)
        n_tgt = len(target_ds)
        self.n_blocks = min(n_src, max(1, n_tgt // 9))
        self.length = self.n_blocks * 10

    def __len__(self):
        return self.length

    @staticmethod
    def _as_image(item):
        """
        Normalize a dataset item to a plain image tensor.
        Handles datasets returning (image, label) tuples (e.g. ImageFolder,
        TensorDataset) as well as datasets returning bare tensors.
        """
        if isinstance(item, (tuple, list)):
            return item[0]
        return item

    def __getitem__(self, i):
        block, off = divmod(i, 10)
        if off == 0:
            item = self.source_ds[block % len(self.source_ds)]
        else:
            t_idx = (block * 9 + (off - 1)) % len(self.target_ds)
            item = self.target_ds[t_idx]
        return self._as_image(item)
