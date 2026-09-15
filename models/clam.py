"""
CLAM-SB: Clustering-constrained Attention Multiple Instance Learning (Single Branch).

Stage 2 of CSCC-PAG: downstream WSI-level 3-class grading
(Well / Moderately / Poorly differentiated) from patch features.

Structure:
    feature (N, feat_dim)
      -> Linear(feat_dim, 512) -> ReLU -> Dropout
      -> Attn_Net_Gated(L=512, D=256)   # tanh(.) * sigmoid(.) gating
      -> softmax over N instances       # attention weights A
      -> weighted sum -> bag embedding M (1, 512)
      -> classifier Linear(512, n_classes)
    Optional instance-level clustering branch (instance_classifiers) when
    instance_eval=True.

Reference: Lu et al., "Data Efficient and Weakly Supervised Computational
Pathology on Whole Slide Images", Nature Biomedical Engineering 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn_Net_Gated(nn.Module):
    """
    Gated attention network: A = sigmoid(W_b x) * tanh(W_a x) -> W_c A.

    Args:
        L: input feature dim (512)
        D: hidden dim (256)
        dropout: dropout probability
        n_classes: number of attention heads / output dims (1 for CLAM-SB)
    """

    def __init__(self, L=512, D=256, dropout=0.25, n_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D), nn.Tanh()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D), nn.Sigmoid()
        )
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)                # element-wise gate
        A = self.attention_c(A)     # (N, n_classes)
        return A, x


class CLAM_SB(nn.Module):
    """
    Single-branch CLAM.

    Args:
        n_classes: number of bag classes (3 for grading).
        dropout: dropout prob in the feature projector.
        subtyping: if True, use per-class instance classifiers (n_classes of
                   them, each binary); if False, a single binary instance classifier.
        embed_dim: input patch feature dim (1024 by default, 384 if from ViT-S).
        size: internal dims [input, hidden(512), attn_hidden(256)].
    """

    def __init__(self, n_classes=3, dropout=0.25, subtyping=True,
                 embed_dim=1024, size=None):
        super().__init__()
        if size is None:
            size = [embed_dim, 512, 256]

        self.n_classes = n_classes
        self.subtyping = subtyping
        self.embed_dim = embed_dim
        self.size = size

        # feature projector + gated attention
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        fc.append(Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1))
        self.attention_net = nn.Sequential(*fc)

        # bag-level classifier
        self.classifiers = nn.Linear(size[1], n_classes)

        # instance-level clustering branch
        if subtyping:
            self.instance_classifiers = nn.ModuleList(
                [nn.Linear(size[1], 2) for _ in range(n_classes)]
            )
        else:
            self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2)])

        # used only when instance_eval=True (not needed for bag-level adaptation)
        self.instance_loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        """
        Instance-level evaluation (clustering pseudo-labels).
        A: (1, N) attention, h: (N, 512) instance embeddings.
        """
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, 1)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, 1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(1, device)
        n_targets = self.create_negative_targets(1, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        """Instance evaluation for the out-of-class (negative) branch."""
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, 1)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(1, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    # ------------------------------------------------------------------
    def forward(self, h, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        """
        Args:
            h: (N, feat_dim) instance features of ONE slide.
            label: (1,) bag label, required only if instance_eval=True.
            instance_eval: whether to compute the instance-level clustering loss.
                           IMPORTANT: keep False during bag-level fine-tuning,
                           otherwise it requires instance pseudo-labels.
            return_features: also return the bag embedding M.
            attention_only: return raw attention only.

        Returns:
            logits, Y_prob, Y_hat, A_raw, results_dict
        """
        device = h.device
        # A: (N, 1) attention logits, h: (N, 512) projected embeddings
        A, h = self.attention_net(h)
        h = h.float()

        A = torch.transpose(A, 1, 0)      # (1, N)
        A_raw = A
        A = F.softmax(A, dim=1)           # softmax over instances

        if attention_only:
            return A

        M = torch.mm(A, h)                # (1, 512) bag embedding

        logits = self.classifiers(M)      # (1, n_classes)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        results_dict = {}

        # ---- instance-level branch (optional) ----
        if instance_eval:
            results_dict["instance_loss"] = 0.0
            for c in range(self.n_classes):
                if label is None:
                    raise ValueError("instance_eval=True requires a bag label.")
                if self.subtyping:
                    if label.item() == c:
                        instance_loss, preds, targets = self.inst_eval(
                            A, h, self.instance_classifiers[c]
                        )
                    else:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A, h, self.instance_classifiers[c]
                        )
                else:
                    instance_loss, preds, targets = (
                        self.inst_eval(A, h, self.instance_classifiers[0])
                        if label.item() == 1
                        else self.inst_eval_out(A, h, self.instance_classifiers[0])
                    )
                results_dict["instance_loss"] = (
                    results_dict["instance_loss"] + 0.5 * instance_loss
                )
            results_dict["instance_loss"] /= self.n_classes

        if return_features:
            results_dict["features"] = M

        return logits, Y_prob, Y_hat, A_raw, results_dict
