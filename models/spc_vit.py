"""
SPC (Self-supervised Pathology Cell) ViT feature extractor + MAE reconstruction head.

Stage 1 of CSCC-PAG:
    - Backbone: Vision Transformer (ViT-Small / ViT-Base), patch_size=16,
      input resolution 256x256 -> 16x16 = 256 patches.
    - The SPC encoder is pretrained by masked auto-encoding (MIM) on pathology
      tiles and then frozen to serve as the RSDA Teacher.

Author: CSCC-PAG
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def patchify(imgs, patch_size=16):
    """
    Split images into non-overlapped patches and flatten them.

    Args:
        imgs: (B, 3, H, W) tensor, H = W = 256 (divisible by patch_size).
        patch_size: int, default 16.

    Returns:
        (B, num_patches, patch_size**2 * 3) flattened patches.
        num_patches = (H // patch_size) * (W // patch_size)
    """
    assert imgs.shape[2] == imgs.shape[3], "expect square image"
    p = patch_size
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    # (B, 3, h, p, w, p) -> (B, h, w, p, p, 3) -> (B, h*w, p*p*3)
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def unpatchify(x, patch_size=16):
    """
    Inverse of `patchify`: rebuild images from flattened patches.

    Args:
        x: (B, num_patches, patch_size**2 * 3)
    Returns:
        (B, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs


def random_masking(x, mask_ratio=0.75):
    """
    Perform per-sample random masking by shuffling tokens (MAE strategy).

    Args:
        x: (B, N, D) token sequence (N = 256 patch tokens).
        mask_ratio: fraction of tokens to mask (paper: 0.75).

    Returns:
        x_visible: (B, N_vis, D) kept tokens, N_vis = round(N * (1 - mask_ratio))
        mask:      (B, N) binary, 1 = masked (removed), 0 = visible
        ids_restore: (B, N) indices that restore the original token order
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)          # uniform [0, 1)
    ids_shuffle = torch.argsort(noise, dim=1)          # ascending
    ids_restore = torch.argsort(ids_shuffle, dim=1)    # inverse permutation

    ids_keep = ids_shuffle[:, :len_keep]
    x_visible = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )

    # binary mask: 1 = removed/masked, 0 = kept
    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_visible, mask, ids_restore


# ---------------------------------------------------------------------------
# SPC ViT encoder
# ---------------------------------------------------------------------------
class SPCViT(nn.Module):
    """
    SPC feature extractor: a ViT backbone that returns patch-level tokens.

    The encoder is used in three places:
      1. Frozen Teacher  -> provides source-domain features f_T(x) for L_FD.
      2. Student encoder -> adapted to target domain, shares the same structure.
      3. Feature extractor for K-means prototype buffer & downstream MIL.

    Args:
        arch: 'vit_small' | 'vit_base' (patch16, 256 resolution).
        embed_dim: feature dim (ViT-S: 384, ViT-B: 768). If None, auto from arch.
        drop_path_rate: stochastic depth.
    """

    # (timm model name, default embed_dim)
    # NOTE: timm >=1.0 has no native *_256 variant; we build the *_224 model
    #       with img_size=256 so the positional embedding matches 16x16 patches.
    ARCH_TO_TIMM = {
        "vit_small": ("vit_small_patch16_224", 384),
        "vit_base": ("vit_base_patch16_224", 768),
    }

    def __init__(self, arch="vit_small", embed_dim=None, drop_path_rate=0.1,
                 img_size=256):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm is required for SPCViT. Install via `pip install timm`."
            )
        if arch not in self.ARCH_TO_TIMM:
            raise ValueError(f"arch must be one of {list(self.ARCH_TO_TIMM)}")

        timm_name, default_dim = self.ARCH_TO_TIMM[arch]
        self.embed_dim = embed_dim or default_dim

        # num_classes=0 -> no classification head; global_pool='' -> keep tokens
        self.backbone = timm.create_model(
            timm_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
            img_size=img_size,          # 256 -> 16x16 = 256 patches
        )
        # timm ViT exposes the patch embedding stride / grid for sanity checks
        self.patch_size = 16
        self.num_patches = (256 // self.patch_size) ** 2  # 256

    def forward_tokens(self, x):
        """
        Full (unmasked) forward -> patch tokens (B, num_patches, embed_dim).
        The CLS token (if any) is dropped.
        """
        feats = self.backbone.forward_features(x)
        # timm ViT returns (B, num_patches + 1, D) with CLS at index 0
        if feats.dim() == 3 and feats.shape[1] == self.num_patches + 1:
            feats = feats[:, 1:, :]
        return feats

    def forward(self, x, mask_ratio=None):
        """
        Args:
            x: (B, 3, 256, 256)
            mask_ratio: if not None, return (visible_tokens, mask, ids_restore)
                        for MAE; otherwise return full patch tokens.
        """
        tokens = self.forward_tokens(x)  # (B, N, D)

        if mask_ratio is not None:
            x_visible, mask, ids_restore = random_masking(tokens, mask_ratio)
            return x_visible, mask, ids_restore

        return tokens

    def extract_feature(self, x):
        """
        Global (bag / slide-level) representation: mean-pooled patch tokens.
        Used for feature distillation & K-means buffer construction.
        Returns (B, embed_dim).
        """
        tokens = self.forward_tokens(x)      # (B, N, D)
        return tokens.mean(dim=1)            # (B, D)


# ---------------------------------------------------------------------------
# MAE reconstruction head (for L_MIM)
# ---------------------------------------------------------------------------
class MAEDecoder(nn.Module):
    """
    Lightweight transformer decoder that reconstructs pixel values of the
    masked patches from the visible-token latents (standard MAE pipeline).

    Inputs are the *visible* tokens produced by SPCViT(mask_ratio=0.75) plus
    learnable [MASK] tokens; output predicts the RGB values of every patch.

    Args:
        embed_dim: encoder dim (384 for ViT-S).
        decoder_dim: width of the decoder.
        decoder_depth: number of transformer blocks.
        num_heads: attention heads.
        num_patches: total patches (256).
        patch_pixels: pixels per patch (16*16*3 = 768).
    """

    def __init__(
        self,
        embed_dim=384,
        decoder_dim=256,
        decoder_depth=4,
        num_heads=8,
        num_patches=256,
        patch_pixels=768,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_pixels = patch_pixels

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # predict pixels of one patch
        self.decoder_pred = nn.Linear(decoder_dim, patch_pixels, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.decoder_pred.weight)

    def forward(self, x_visible, ids_restore):
        """
        Args:
            x_visible:  (B, N_vis, embed_dim) tokens kept after masking.
            ids_restore: (B, N) index map restoring original order.
        Returns:
            pred: (B, num_patches, patch_pixels) reconstructed patch pixels.
        """
        B, N_vis, _ = x_visible.shape
        N = self.num_patches

        # project visible tokens to decoder width
        x = self.decoder_embed(x_visible)  # (B, N_vis, decoder_dim)

        # append learnable mask tokens for the removed positions
        n_mask = N - N_vis
        mask_tokens = self.mask_token.repeat(B, n_mask, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, decoder_dim)

        # restore original patch order, then add positional embedding
        x_full = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2])
        )
        x_full = x_full + self.decoder_pos_embed

        x_full = self.decoder_blocks(x_full)
        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)  # (B, N, patch_pixels)
        return pred


def mim_reconstruction_loss(pred, imgs, mask, patch_size=16, norm_pix=True):
    """
    Compute the MIM loss: MSE between reconstruction and the *original*
    patches at the masked positions (standard MAE objective).

    Args:
        pred:  (B, N, patch_pixels) decoder output.
        imgs:  (B, 3, H, W) original images.
        mask:  (B, N) binary mask, 1 = masked position (to be predicted).
        patch_size: 16.
        norm_pix: whether to normalize target patches (zero-mean unit-var),
                  which stabilizes MAE training.
    Returns:
        scalar MSE loss (mean over masked patches only).
    """
    target = patchify(imgs, patch_size)  # (B, N, patch_pixels)

    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, unbiased=True, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)          # (B, N) per-patch MSE

    # average only over masked (removed) patches
    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
    return loss
