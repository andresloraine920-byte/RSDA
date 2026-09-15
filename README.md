# CSCC-PAG: Enhancing Cervical Squamous Cell Carcinoma Diagnosis under Distribution Shift via Domain Adaptive Representation Learning

Official implementation of **CSCC-PAG** – a three-stage framework
(**P**re-training, **A**daptation, **G**eneralization) for robust cervical
squamous cell carcinoma (CSCC) diagnosis across multi-center distribution shift.

The core domain-adaptation engine is **RSDA (Representation-Shift Domain
Adaptation)**, a teacher–student self-supervised framework that aligns a
source-pretrained pathology feature extractor to an unlabeled target domain
using a K-means prototype buffer, masked image modeling (MIM) and feature
distillation (FD).

---

## 1. Framework overview

```
Stage 1 (Pre-training)          Stage 2 (Downstream)          Stage 3 (Adaptation)
----------------------          --------------------          --------------------
SPC: ViT-S/B + MAE       -->   CLAM-SB MIL grading    <--    RSDA: Teacher-Student
256x256 pathology tiles         3-class grading               + K-means buffer (K=100)
self-supervised MIM             (Well/Moderately/Poorly)      + MIM (75% mask)
                                TransUNet segmentation        + FD (lambda=0.1)
                                                              1:9 source:target mix
```

### Stage 3 – RSDA objective

```
L_total = L_MIM + lambda * L_FD ,   lambda = 0.1

L_MIM : MAE pixel-reconstruction MSE over 75% randomly masked target patches.
L_FD  : || f_S(x) - f_T(x) ||^2  (feature distillation between the adapted
        Student and the frozen source Teacher, preventing catastrophic forgetting).
```

A K-means prototype buffer (**K=100 clusters × 500 patches = 50,000 prototypes**)
is built in the source feature space; every mini-batch mixes source prototypes
and unlabeled target patches at a **1:9** ratio.

---

## 2. Repository structure

```
RSDA/
├── models/
│   ├── spc_vit.py       # SPC ViT encoder + MAE reconstruction head
│   ├── rsda.py          # RSDA teacher-student, MIM + FD losses
│   └── clam.py          # CLAM-SB gated-attention MIL (3-class grading)
├── utils/
│   └── kmeans_buffer.py # K-means prototype buffer & 1:9 mixed sampler
├── train_rsda.py        # Stage-3 domain-adaptation training (--demo supported)
├── eval.py              # Downstream evaluation (ACC / BACC / F1)
├── requirements.txt
└── README.md
```

---

## 3. Environment

Tested with **Python 3.8** and **PyTorch 1.10** (also runs on newer versions).

```bash
conda create -n csccpag python=3.8 -y
conda activate csccpag
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `timm` (ViT backbones),
`scikit-learn` (K-means), `openslide-python`, `staintools`, `h5py`.

---

## 4. Quick start: demo (no data required)

Verify the whole pipeline (forward + backward) with synthetic patches:

```bash
python train_rsda.py --demo
```

This builds a tiny K-means buffer from dummy tensors, runs the MIM + FD
losses and saves a checkpoint to `./outputs/rsda_final.pth`.
A `[demo] smoke test PASSED` message confirms the installation is correct.

Evaluate the downstream head on synthetic slide bags:

```bash
python eval.py --demo
```

---

## 5. Real training & evaluation

### 5.1 Stage 3 – RSDA adaptation

```bash
python train_rsda.py \
    --source_img_dir /data/source_patches \
    --target_img_dir /data/target_patches \
    --pretrained /path/to/spc_pretrained.pth \
    --arch vit_small \
    --epochs 50 --batch_size 32 --lr 1e-4 \
    --k 100 --per_cluster 500 --lambda_fd 0.1
```

* `--source_img_dir` / `--target_img_dir` follow the `ImageFolder` layout
  (labels are ignored – RSDA is fully self-supervised).
* The script automatically extracts source features with the frozen teacher,
  builds the 50k prototype buffer, and trains the student.

### 5.2 Stage 2 – downstream grading

Extract slide features with the adapted encoder, then evaluate CLAM-SB:

```bash
python eval.py \
    --feat_dir /data/slide_features \
    --csv /data/labels.csv \
    --ckpt /path/to/clam_sb.pth \
    --embed_dim 384
```

* `--feat_dir` : one `.pt`/`.h5` feature file per slide, named `<slide_id>.pt`.
* `--csv`      : columns `slide_id,label`
  (label ∈ {`高分化`/`中分化`/`低分化`} or {`Well`,`Moderately`,`Poorly`}).
* Outputs **ACC**, **BACC**, **macro F1** and per-class support.

---

## 6. Pretrained weights

> **Note.** The SPC pretrained backbone and the adapted CSCC-PAG weights are
> available from the authors **upon reasonable request** for research purposes.
>
> Placeholder download link (to be replaced upon publication):
> `https://github.com/andresloraine920-byte/RSDA/releases`
>
> To load a checkpoint into the RSDA teacher/student:
> ```bash
> python train_rsda.py --pretrained <path/to/spc_pretrained.pth> ...
> ```

---

## 7. Citation

If you find this code useful, please cite:

```bibtex
@article{csccpag2026,
  title   = {Enhancing Cervical Squamous Cell Carcinoma Diagnosis under
             Distribution Shift via Domain Adaptive Representation Learning},
  author  = {Anonymous Authors},
  journal = {Under Review},
  year    = {2026},
  note    = {CSCC-PAG / RSDA official implementation}
}
```

---

## 8. License

This project is released for academic research only.
