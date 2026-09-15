"""
RSDA: Representation-Shift Domain Adaptation (Stage 3 core of CSCC-PAG).

Teacher-Student framework:
    - Teacher : frozen SPC ViT (source-domain pretrained weights).
    - Student : same architecture, updated by gradient to adapt to target domain.
    - Prototype Buffer: source-domain K-means prototypes mixed with unlabeled
      target patches at a 1:9 ratio each mini-batch.

Composite objective:
    L_total = L_MIM + lambda * L_FD,  lambda = 0.1
      L_MIM : MAE pixel-reconstruction MSE on 75% randomly masked patches.
      L_FD  : feature-distillation MSE ||f_S(x) - f_T(x)||^2 (no masking),
              preserving source discriminability / avoiding catastrophic forgetting.
"""

import torch
import torch.nn as nn

from .spc_vit import SPCViT, MAEDecoder, mim_reconstruction_loss


class RSDA(nn.Module):
    """
    RSDA teacher-student model.

    Args:
        arch: ViT backbone name ('vit_small' | 'vit_base').
        mask_ratio: fraction of masked patches for MIM (default 0.75).
        lambda_fd: weight of the feature-distillation loss (default 0.1).
        decoder_dim / decoder_depth / num_heads: MAE decoder hyper-params.
    """

    def __init__(
        self,
        arch="vit_small",
        mask_ratio=0.75,
        lambda_fd=0.1,
        decoder_dim=256,
        decoder_depth=4,
        num_heads=8,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.lambda_fd = lambda_fd

        # ---- Teacher (frozen source model) ----
        self.teacher = SPCViT(arch=arch)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # ---- Student (adapted to target domain) ----
        self.student = SPCViT(arch=arch)

        # ---- MAE reconstruction head attached to the student ----
        self.decoder = MAEDecoder(
            embed_dim=self.student.embed_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            num_patches=self.student.num_patches,  # 256
            patch_pixels=16 * 16 * 3,              # 768
        )

    # ------------------------------------------------------------------
    def load_pretrained_weights(self, ckpt_path):
        """
        Load SPC pretrained weights into BOTH teacher and student
        (student then diverges via adaptation).
        """
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
        missing_t, _ = self.teacher.load_state_dict(state, strict=False)
        missing_s, _ = self.student.load_state_dict(state, strict=False)
        return missing_t, missing_s

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        One RSDA training step on a mini-batch of (mixed) patches.

        Args:
            x: (B, 3, 256, 256) images (source prototypes + target patches).

        Returns:
            dict with:
                loss      : L_total = L_MIM + lambda * L_FD
                loss_mim  : reconstruction MSE on masked patches
                loss_fd   : feature-distillation MSE
                pred      : (B, N, 768) reconstructed patch pixels (for viz)
                mask      : (B, N) binary mask
        """
        # ---------- 1) MIM: mask 75% -> encode visible -> reconstruct ----------
        x_visible, mask, ids_restore = self.student(x, mask_ratio=self.mask_ratio)
        pred = self.decoder(x_visible, ids_restore)                  # (B,N,768)
        loss_mim = mim_reconstruction_loss(pred, x, mask, patch_size=16)

        # ---------- 2) FD: full (unmasked) image through teacher & student -----
        with torch.no_grad():
            self.teacher.eval()
            f_t = self.teacher(x)                 # (B, N, D) teacher tokens
        f_s = self.student(x)                     # (B, N, D) student tokens
        loss_fd = torch.nn.functional.mse_loss(f_s, f_t)

        # ---------- 3) composite loss ----------
        loss = loss_mim + self.lambda_fd * loss_fd

        return {
            "loss": loss,
            "loss_mim": loss_mim,
            "loss_fd": loss_fd,
            "pred": pred,
            "mask": mask,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(self, x):
        """
        Slide/patch-level feature extraction with the ADAPTED student encoder.
        Used to build MIL bags for downstream CLAM training / evaluation.

        Args:
            x: (B, 3, 256, 256) patches.
        Returns:
            (B, num_patches, embed_dim) patch tokens (student features).
        """
        self.student.eval()
        return self.student(x)
