# Architectural and Pipeline Decisions

## Human-Specified Decisions

1. **Backbone and Checkpoint**: Use DINOv3 ViT-B/16 loaded from a local clone of `facebookresearch/dinov3` with weights `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
2. **Input Channels and Per-Channel Tokenization**: Process 4-channel inputs (RGB brightfield plus 1 BODIPY fluorescence channel) built from RGB PNGs under `data/raw/`. Before images have no fluorescence image (BODIPY not yet added), so their fluorescence channel is zero. Backlight and RGB weights are fixed within the before set and within the after set but differ between the two; BODIPY RGB weights are fixed across all after images. Average DINOv3's pretrained RGB patch-embedding weights across input channels to initialize a 1-channel projection, and tokenize each of the 4 channels independently through `TE`.
3. **Architecture and Three-Stage Training Order**:
   - **Stage 1**: Train a `BatchTopK` sparse autoencoder (`SAE`) with tied decoder bias (`b_dec`) and an auxiliary dead-latent reconstruction loss (`TopK_aux` minimizing $e - \hat{e}$ on $e = x - \hat{x}$ computed from dead latents only) while keeping `TE` frozen.
   - **Stage 2**: Train a Transformer decoder (`TD`) over `x_pred` with `TE` and `SAE` frozen to visualize the morphological effect of sparse latent activations.
   - **Stage 3**: Train a linear classifier (`LC`) directly on spatially pooled `z` (with no intermediate cross-channel convolution) while keeping `TE` and `SAE` frozen.
4. **Inference Sparsity**: Remove `BatchTopK` during inference (`infer.py`).
5. **Dataset Conditions, Pairing, and Splits**:
   - Organize data into `before`, `control` (`DD` DMSO + DMSO, `DF` DMSO + ferrostatin, `RF` RSL3 + ferrostatin), and `test` (`RD` RSL3 + DMSO, the only ferroptotic condition), with comparable sample counts per condition. DMSO is diluted in the same ratio as the other compounds.
   - Pair `before` images with their corresponding post-treatment images.
   - Split train, validation, and test sets by well groups.
6. **Scripts and Hyperparameters**:
   - Implement channel separation in a Windows batch file (`colorsep.bat`) using ImageMagick.
   - Implement Cellpose-SAM segmentation in `segment.py` reading from `data/raw/`.
   - Provide minimal single-file PyTorch scripts for training (`train.py`) and inference (`infer.py`) with every hyperparameter, including model dimensions, left empty (`None`).
7. **Inference Outputs**: Export `LC` predictions, pooled sparse latents `z`, baseline `TD` reconstructions, and latent-intervened `TD` reconstructions on `test` (`RD`).
8. **Cell Crops Keep Cell Size**: Do not resize cell crops. Square each Cellpose bounding box with the surrounding image pixels, so crop sizes differ between cells.

## Derived Architecture and Pipeline Assumptions

### 1. Plate Layout and File Naming
- One 24-well plate; wells 1–24 numbered row-major over 4 rows × 6 columns, columns in order `0.1L`, `0.1R`, `0.5L`, `0.5R`, `1.0L`, `1.0R`; each row holds one condition code, with the row order set in `segment.py` `row_codes`.
- Concentrations are pooled into one class per code; a well is identified as `<code>_<side><conc>` (e.g. `DD_L0.5`); each well is one split group.
- Raw files sit directly in `data/raw/` (no subfolders), named `<phase><code>_<side><conc>_<fov>.png` with `<conc>` one of `0.1`, `0.5`, `1.0`, or `<phase>W<well>_<fov>.png`; phase `b` = before brightfield, `a` = after brightfield, `f` = after BODIPY. `<fov>` indexes the same field before and after. `W<well>` files are skipped until `row_codes` is set; other names are skipped and printed.
- `colorsep.bat` writes `data/sep/<stem>_c0.png`, `<stem>_c1.png`, `<stem>_c2.png` (R, G, B) for each RGB file in `data/raw/`.
- `segment.py` writes `data/crops/{before|control|test}/<code>/<well>_<fov>_cell<idx>.pt` containing `{"image": Tensor[4, S, S], "centroid": Tensor[2], "area": Tensor[]}`, where `S` is a multiple of 16 that varies per cell and `area` is the mask pixel count.

### 2. Channel Separation and Intensity Conversion
- `colorsep.bat` runs `magick <file> -alpha off -channel RGB -separate -define png:color-type=0 <stem>_c%d.png`, keeping the input bit depth; `segment.py` reads 8- or 16-bit from the PNG mode.
- Brightfield pixel = backlight × gain_c × transmittance_c. Each channel is divided by its acquisition-set background (median over images of per-image channel medians), divided by `bf_white`, and clipped to [0, 1], leaving transmittance independent of backlight and RGB weights. One constant per set, not per image, because per-image medians shift with cell density, which differs by condition.
- BODIPY channel = `fluo_weights · (R, G, B)`, divided by one global constant (largest per-image 99.9th percentile over all after BODIPY images), clipped to [0, 1]; intensities stay comparable across after images.
- Failure mode: saturated pixels break the multiplicative brightfield model; BODIPY 581/591 C11 shifts emission from red to green on oxidation, so one linear channel cannot carry the oxidised/reduced ratio.

### 3. Segmentation, Pairing, and Splits
- `segment.py` runs `cellpose.models.CellposeModel(pretrained_model="cpsam")` on the 3 brightfield channels (`channel_axis=0`), with the `cpsam` weights in Cellpose's default `<home>/.cellpose/models/`, pads each mask's bounding box by `pad_px`, and cuts a square window from the 4-channel stack with no resizing: the side is the padded box's longer side rounded up to a multiple of the 16-pixel patch size, the window is centred on the box, and it is shifted back inside the image where it would stick out (such cells sit off-centre). Cells whose square would be larger than the image are skipped. Neighbouring cells inside the window are not masked out. After brightfield images without a matching BODIPY file are skipped.
- Failure mode: masks cut by the image border understate cell size; such cells are kept.
- After cell groups pair with before cell groups of the same `(well, fov)` at the nearest mask centroid within `max_pair_dist_px`.
- `RD` wells (before and after) are excluded from all training; control wells are split into train and validation per condition by `val_well_frac`.
- Failure mode: stage drift or cell migration between before and after imaging breaks centroid correspondence; unmatched crops are skipped during Stage 3 training and omitted from paired inference.

### 4. DINOv3 Backbone (`TE`)
- Meta `.pth` checkpoints store a raw `state_dict` without model definitions, so `TE` builds `dinov3_vitb16` from `dinov3_repo_dir` via `torch.hub.load(..., source="local")` before loading `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` from `<home>/Documents/models/` (`Path.home()`, so no username is hard-coded).
- `TE` strips `[CLS]` and register tokens via `forward_features(flat)["x_norm_patchtokens"]` and remains frozen across all three training stages.

### 5. Inference Threshold Calibration (`SAE`)
- Removing `BatchTopK` without a threshold leaves unconstrained `ReLU` activations (`h = F.relu(encoder(x - b_dec))`), shifting the sparsity distribution seen by `LC` and `TD`.
- `train.py` tracks the running mean `theta` of the minimum positive retained activation per training batch so Stage 3 and `infer.py` apply `JumpReLU` (`h * (h > theta)`) for batch-size-independent sparsity. Stages 1 and 2 keep `BatchTopK`.

### 6. Decoder and Classifier Tensor Routing (`TD` and `LC`)
- Stage 2 trains `TD` (Transformer encoder blocks over `x_pred` followed by linear projection and `PixelShuffle(patch_size)`) per channel to reconstruct the 4-channel crop.
- Stage 3 averages each crop's patch-level `z` over its own `(H_p, W_p)` grid, computes the paired difference `z_after - z_before` per channel, concatenates all 4 channels into shape `(B, 4 * d_sae)`, and trains `LC` across `DD`, `DF`, and `RF`. Because `TE` and `SAE` are frozen, each crop's pooled latents are computed once and reused every epoch.
- Averaging over patches removes cell size from the pooled latents; `LC` does not read `area`.

### 7. Variable Crop Sizes
- Crops of different sizes cannot be stacked into one tensor. Stages 1 and 2 group crops by side length and draw each batch from one group; Stage 3 computes latents group by group; `infer.py` processes one before/after pair at a time.
- Failure mode: a side length with few crops yields small batches, so `BatchTopK` there behaves closer to per-crop TopK.