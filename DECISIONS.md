# Architectural and Pipeline Decisions

## Human-Specified Decisions

1. **Backbone and Checkpoint**: Use DINOv3 ViT-B/16 loaded from a local clone of `facebookresearch/dinov3` with weights `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
2. **Input Channels and Per-Channel Tokenization**: Images are RGB only; there are no separate fluorescence (BODIPY) images, so inputs have 3 channels (R, G, B). Average DINOv3's pretrained RGB patch-embedding weights across input channels to initialize a 1-channel projection, and tokenize each channel independently through `TE`.
3. **Architecture and Three-Stage Training Order**:
   - **Stage 1**: Train a `BatchTopK` sparse autoencoder (`SAE`) with tied decoder bias (`b_dec`) and an auxiliary dead-latent reconstruction loss (`TopK_aux` minimizing $e - \hat{e}$ on $e = x - \hat{x}$ computed from dead latents only) while keeping `TE` frozen.
   - **Stage 2**: Train a Transformer decoder (`TD`) over `x_pred` with `TE` and `SAE` frozen to visualize the morphological effect of sparse latent activations.
   - **Stage 3**: Train a linear classifier (`LC`) directly on spatially pooled `z` (with no intermediate cross-channel convolution) while keeping `TE` and `SAE` frozen.
4. **Inference Sparsity**: Remove `BatchTopK` during inference (`infer.py`).
5. **Dataset Conditions, Pairing, and Splits**:
   - Organize data into `before` and `after`, each by condition: `DD` DMSO + DMSO, `DF` DMSO + ferrostatin, `RF` RSL3 + ferrostatin (not ferroptotic), and `RD` RSL3 + DMSO (the only ferroptotic condition), with comparable sample counts per condition. DMSO is diluted in the same ratio as the other compounds.
   - Pair `before` images with their corresponding post-treatment images once both exist for the same experiment (see decision 15).
   - Split by well groups (see decision 12).
6. **Scripts and Hyperparameters**:
   - `segment.py` runs the data steps (decision 13) and Cellpose-SAM segmentation, reading the raw RGB JPEGs directly; `colorsep.bat` (ImageMagick channel separation) is no longer used.
   - Provide minimal single-file PyTorch scripts for training (`train.py`) and inference (`infer.py`) with every hyperparameter, including model dimensions, left empty (`None`).
7. **Inference Outputs**: Export `LC` predictions, pooled sparse latents `z`, baseline `TD` reconstructions, and latent-intervened `TD` reconstructions on the held-out wells of every fold.
8. **Cell Crops Keep Cell Size**: Do not resize cell crops. Square each Cellpose bounding box with the surrounding image pixels, so crop sizes differ between cells. Segment single cells, not cell groups.
9. **Per-Image Normalisation**: Normalise each input image's mean RGB and contrast, centring each channel at 0.5 (128 on an 8-bit scale).
10. **Augmentation Before Cellpose**: Augment whole images after normalisation and before Cellpose segmentation, with rotation by multiples of 90° plus flips, Gaussian noise, Gaussian blur, and realistic microscope artefacts (chromatic aberration, uneven illumination). No crop augmentation.
11. **Concentration 1.0 Dropped**: Every cell at concentration 1.0 was already dead, so the 1.0 wells are not used. The remaining wells keep their 24-well plate numbers (1–4, 7–10, 13–16, 19–22).
12. **Binary Labels and Leave-One-Well-Out**: `RD` is the only ferroptotic condition and must be in training. `LC` predicts ferroptotic (`RD`) versus not ferroptotic (`DD`, `DF`, `RF`). Evaluation is leave-one-well-out over 4 folds: each well of every condition is held out once, and all three stages are retrained for each fold.
13. **Data Folders**: Under `data/`, raw images stay in `0_before_raw/` and `0_after_raw/` (one subfolder per experiment); renamed copies go to `1_<phase>_rename/` as `exp<name>_<phase>W<well>_<index>.<ext>`, with one index shared by all images in alphabetical order; preprocessed and augmented images go to `2_<phase>_prepaug/`; per-cell crops go to `3_<phase>_crops/`.
14. **Magnification**: Use 20x images at their native scale. exp3's images are labelled "40x" but were taken at 20x. 4x and 10x images are dropped. A single cell is about 120 × 120 px including dendrites.
15. **Before/After Pairing Pending**: The current images do not cover both phases in any experiment (exp1 has only after images; exp2 and exp3 only before images). Until more images arrive, train with what exists: `LC` classifies single after cells, and before cells train only the `SAE` and `TD`.
16. **Plate Rows**: The plate rows hold `RD`, `RF`, `DD`, `DF` from top to bottom, in every experiment.
17. **LC Pooling, Mean and Max Side by Side**: For each channel, `LC` reads the SAE latents averaged over the crop's patches and their maxima over the patches, joined into one vector of 2 × 3 × `d_sae` values (all means first, then all maxima).
18. **Hyperparameters from Published Work**: The empty hyperparameters are filled with values from published work (see Hyperparameters). `seed` is 42. Epochs are set for a first run of about 40 minutes on the RTX 5070 Ti.

## Derived Architecture and Pipeline Assumptions

### 1. Plate Layout and File Naming
- One 24-well plate per experiment; wells 1–24 numbered row-major over 4 rows × 6 columns, columns in order `0.1L`, `0.1R`, `0.5L`, `0.5R`, `1.0L`, `1.0R`; rows `RD`, `RF`, `DD`, `DF` (`plate.py`). The `1.0` columns (wells 5, 6, 11, 12, 17, 18, 23, 24) are renamed but not preprocessed.
- Concentrations are pooled into one class per code; a well is identified as `<experiment>_<code>_<side><conc>` (e.g. `exp1_DD_L0.5`); each well is one split group.
- Raw names in use: exp1 `<code>_<conc><side>_<magnification>_<number>` (e.g. `DD_0.1L_20x_39`), exp2 `bW<well>_<magnification>_<number>`, exp3 `well<well>_<magnification>_<number>`. Names that fit none of these are printed and skipped.
- Renaming copies every kept raw file in alphabetical order of its path under `data/` (`0_after_raw/...` before `0_before_raw/...`) and numbers them with one 3-digit index, e.g. `exp1_afterW13_012.jpg`; the experiment name is the raw subfolder's name up to the first `_`. `data/1_rename_map.csv` lists each new name with its raw path. Each run of `segment.py` deletes and rewrites the `1_`, `2_` and `3_` folders.
- `2_<phase>_prepaug/` holds `<name>.png` (normalised) and `<name>-aug<copy>.png` (augmented copies) as 8-bit RGB PNGs.
- `3_<phase>_crops/` holds `<image name>_cell<idx>.pt` containing `{"image": Tensor[3, S, S], "centroid": Tensor[2], "area": Tensor[]}`, where `S` is a multiple of 16 that varies per cell and `area` is the mask pixel count.

### 2. Intensity Conversion
- `segment.py` reads each JPEG as 8-bit RGB scaled to [0, 1]. Each of R, G, B in each image is shifted and scaled to mean `bf_mean` (0.5) and standard deviation `bf_std`, then clipped to [0, 1]. Cellpose segments these normalised images.
- Failure mode: per-image normalisation also removes real brightness and contrast differences between images, such as those caused by different cell density; values beyond `bf_mean ± 0.5` clip. Saving the `2_` images as 8-bit PNG rounds intensities to steps of 1/255.

### 3. Segmentation and Splits
- `segment.py` runs `cellpose.models.CellposeModel(pretrained_model="cpsam")` on each `2_` image (`channel_axis=0`), with the `cpsam` weights in Cellpose's default `<home>/.cellpose/models/`, pads each mask's bounding box by `pad_px`, and cuts a square window with no resizing: the side is the padded box's longer side rounded up to a multiple of the 16-pixel patch size, the window is centred on the box, and it is shifted back inside the image where it would stick out (such cells sit off-centre). Cells whose square would be larger than the image are skipped. Neighbouring cells inside the window are not masked out.
- Failure mode: masks cut by the image border understate cell size; such cells are kept. Cellpose gives each pixel to one cell, so where neurites overlap they are split between cells; masks measured on sample images were smaller than the ~120 px cell size including dendrites (median long side about 40–65 px).
- Folds follow plate columns: fold k holds out the wells of column `FOLD_COLUMNS[k]` (`L0.1`, `R0.1`, `L0.5`, `R0.5`) in all four conditions and all experiments, and trains on the other three columns. Training uses crops of both phases for Stages 1–2 and after crops for Stage 3; held-out wells contribute their original after crops only. Each fold's checkpoint is `checkpoints/fold<k>.pt`; `infer.py` runs each fold's models on that fold's held-out after cells.
- Failure mode: exp1 has no 20x after images of `DF_L0.5` and `DF_R0.5`, so folds 3 and 4 have no held-out `DF` after cells.

### 4. DINOv3 Backbone (`TE`)
- Meta `.pth` checkpoints store a raw `state_dict` without model definitions, so `TE` builds `dinov3_vitb16` from `dinov3_repo_dir` via `torch.hub.load(..., source="local")` before loading `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` from `<home>/Documents/models/` (`Path.home()`, so no username is hard-coded).
- `TE` strips `[CLS]` and register tokens via `forward_features(flat)["x_norm_patchtokens"]` and remains frozen across all three training stages.

### 5. Inference Threshold Calibration (`SAE`)
- Removing `BatchTopK` without a threshold leaves unconstrained `ReLU` activations (`h = F.relu(encoder(x - b_dec))`), shifting the sparsity distribution seen by `LC` and `TD`.
- `train.py` tracks the running mean `theta` of the minimum positive retained activation per training batch so Stage 3 and `infer.py` apply `JumpReLU` (`h * (h > theta)`) for batch-size-independent sparsity. Stages 1 and 2 keep `BatchTopK`.

### 6. Decoder and Classifier Tensor Routing (`TD` and `LC`)
- Stage 2 trains `TD` (Transformer encoder blocks over `x_pred` followed by linear projection and `PixelShuffle(patch_size)`) per channel to reconstruct the 3-channel crop.
- Stage 3 takes the mean and the maximum of each after crop's patch-level `z` over its own `(H_p, W_p)` grid, for each of the 3 channels, and joins them into shape `(B, 2 * 3 * d_sae)` (means, then maxima); `LC` is trained on it with 2 outputs (not ferroptotic, ferroptotic). With one `RD` condition against three others, the cross-entropy weights each class by 1 / (its share of training cells × 2), so both classes count equally. Pooled latents use the `theta` threshold and are computed once, since `TE`, `SAE` and the crops are fixed.
- Neither the mean nor the maximum over patches keeps cell size; `LC` does not read `area`.

### 7. Variable Crop Sizes
- Crops of different sizes cannot be stacked into one tensor. Stages 1 and 2 group crops by side length and draw each batch from one group; Stage 3 computes latents group by group; `infer.py` processes one cell at a time.
- Failure mode: a side length with few crops yields small batches, so `BatchTopK` there behaves closer to per-crop TopK.

### 8. Augmentation (`segment.py`, before Cellpose)
- Order per image: load, normalise, save, then save `aug_copies` augmented copies; Cellpose later segments the original and every copy on its own. Every well is a training well in three of the four folds, so every condition is augmented.
- Steps, in order: rotation by 0/90/180/270° and a mirror flip half of the time; chromatic aberration, shifting R and B against G by up to `aug_ca_max_shift_px` pixels with edge pixels repeated; uneven illumination, multiplying the image by a linear ramp across it in a random direction with slope up to `aug_illum_max_slope`; Gaussian blur with sigma up to `aug_blur_max_sigma`; Gaussian noise with standard deviation up to `aug_noise_max_std`. Each strength is drawn uniformly from 0 to its maximum from a generator seeded by `seed`, the image name and the copy number; the result is clipped to [0, 1].
- `train.py` uses augmented copies only while a well is a training well; for its held-out fold only the original crops are used. Training itself does no further augmentation.
- Failure mode: augmentation is fixed when `segment.py` runs, so every epoch sees the same `1 + aug_copies` versions of each image; Cellpose runs `1 + aug_copies` times per image.

## Hyperparameters

Chosen from published work on request; `seed` (42) was set by hand. Batch sizes count crops, not tokens. Run speed on the RTX 5070 Ti has not been measured.

### `segment.py`

| Setting | Value | Source |
|---|---|---|
| `flow_threshold` | 0.4 | Cellpose `CellposeModel.eval` default (installed version 4.2.1.1) |
| `cellprob_threshold` | 0.0 | Cellpose default |
| `batch_size` | 8 | Cellpose default |
| `min_mask_pixels` | 15 | Cellpose `min_size` default, so this filter removes nothing that Cellpose keeps |
| `pad_px` | 16 | R-CNN context padding p = 16 (Girshick et al., 2014). R-CNN counts it in the 227 × 227 warped region; here it is native pixels, one DINOv3 patch |

### `train.py`, Stage 1 (SAE)

Defaults of the BatchTopK reference implementation (Bussmann et al., 2024), which trains on GPT-2 small activations of width 768, the same width as DINOv3 ViT-B/16 tokens.

| Setting | Value | Source |
|---|---|---|
| `d_sae` | 12288 | BatchTopK default (16 × 768) |
| `topk` | 32 | BatchTopK default |
| `topk_aux` | 512 | BatchTopK default; Gao et al. (2024) choose a power of two near half the input width |
| `aux_weight` | 1/32 | BatchTopK default; Gao et al. (2024) |
| `dead_steps` | 5 | BatchTopK default (5 batches); Gao et al. (2024) use 10 million tokens |
| `sae_lr` | 3e-4 | BatchTopK default |
| `sae_batch_size` | 32 | BatchTopK uses 4,096 tokens per batch. At the median Cellpose crop sizes of six sample images (48 to 147 tokens per crop over 3 channels), 32 crops hold about 1,500 to 4,700 tokens; 32 crops of the largest size (272 px) hold about 27,700 |
| `sae_epochs` | 1 | First 40-minute run; BatchTopK trains on 1 billion tokens |

### Stage 2 (TD)

Default MAE decoder (He et al., 2022): 8 blocks of width 512, 16 heads, MLP ratio 4, base learning rate 1.5e-4 per 256 images, scaled linearly with batch size.

| Setting | Value | Source |
|---|---|---|
| `td_dim` | 512 | MAE decoder |
| `td_heads` | 16 | MAE decoder |
| `td_layers` | 8 | MAE decoder |
| `td_mlp_dim` | 2048 | MAE decoder (4 × 512) |
| `td_batch_size` | 32 | Same as `sae_batch_size`; 96 channel images per batch |
| `td_lr` | 1.5e-4 × 96 / 256 = 5.625e-5 | MAE linear scaling rule, each channel image counted as one image |
| `td_epochs` | 1 | First 40-minute run |

### Stage 3 (LC)

| Setting | Value | Source |
|---|---|---|
| `lc_batch_size` | 32 | Close to the batch of 30 in CellDeathPred (Schorpp et al., 2023); also the batch for computing pooled latents |
| `lc_lr` | 1e-3 | Adam default step size (Kingma & Ba, 2015) |
| `lc_weight_decay` | 0.0 | MAE linear probing (He et al., 2022) |
| `lc_epochs` | 90 | MAE linear probing (He et al., 2022) |

### Differences from the sources

- SAE: tokens are not scaled to unit norm (BatchTopK scales its inputs); decoder columns are not renormalised and `b_dec` starts at zero (Gao et al. renormalise them and start the bias at the geometric median of the data); the optimiser is AdamW with PyTorch defaults (betas 0.9 and 0.999, weight decay 0.01), while BatchTopK uses Adam with betas 0.9 and 0.99.
- TD: AdamW with PyTorch defaults at a constant learning rate; MAE uses betas 0.9 and 0.95, weight decay 0.05 and cosine decay with warm-up, and adds positional embeddings to the decoder input.
- LC: AdamW instead of LARS, and no BatchNorm on the inputs (MAE's linear probe adds BatchNorm without affine parameters). LC has 6 × 12288 = 73,728 inputs, more than there are training after cells, so with no weight decay it can overfit.

## References

- Bussmann, B., Leask, P., & Nanda, N. (2024). *BatchTopK sparse autoencoders* (arXiv:2412.06410). arXiv. https://doi.org/10.48550/arXiv.2412.06410
- Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., & Wu, J. (2024). *Scaling and evaluating sparse autoencoders* (arXiv:2406.04093). arXiv. https://doi.org/10.48550/arXiv.2406.04093
- Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 580–587). https://doi.org/10.1109/CVPR.2014.81
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. arXiv:2111.06377
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations*. arXiv:1412.6980
- Pachitariu, M., Rariden, M., & Stringer, C. (2025). *Cellpose-SAM: Superhuman generalization for cellular segmentation* [Preprint]. bioRxiv. https://doi.org/10.1101/2025.04.28.651001
- Schorpp, K., Bessadok, A., Biibosunov, A., Rothenaigner, I., Strasser, S., Peng, T., & Hadian, K. (2023). CellDeathPred: A deep learning framework for ferroptosis and apoptosis prediction based on cell painting. *Cell Death Discovery, 9*, 277. https://doi.org/10.1038/s41420-023-01559-y
