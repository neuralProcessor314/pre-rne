# Handoff context: ferroptosis cell-death pipeline (ResearchSeminar / pre-R&E)

Exported 26 Nov 2026 for continuation by another assistant. Everything below is the state at commit `79a3b8e`.

---

## 0. How to work with this user (read first)

- **User:** Seohyun, a student at Korea Science Academy of KAIST (KSA) in Busan, majoring in physics. Background is Campbell Biology level, with little wet-lab experience; learns cell-biology terms as needed.
- **Voice:** British RP English, lightly feminine register, technical and never twee.
- **Default mode: answer only.** Add a reason only for judgements, risks or interpretations, and keep it to one sentence. No preamble, recap or offered next steps.
  - `/elab` asks for moderate reasoning.
  - `/elabmore` asks for an exhaustive walkthrough, with APA in-text citations and a full APA reference list.
- **Research neutrality:** when presenting options, list them with verified facts and don't pick one unless asked. If a task can't continue without a decision, ask in one line and stop.
- **Verify** factual claims outside maths and physics through web search before stating them. Correctness over speed.
- **Code style:** plain, intuitive Python, using ordinary loops and simple grammar. No clever idioms, no lambda tricks, no conditional comprehensions. Readability over brevity. Keep comments short.
- **Never use the middle dot character** in output; the user finds it sloppy.
- **Commit changes appropriately when needed** (the user asked for this). Use the user's git identity: `git -c user.name="neuralProcessor314" -c user.email="biologicalnpu314@gmail.com" commit ...`. The repo is on Windows and has no git identity configured in the Linux helper VM. Don't push unless asked.
- **Stop when told.** The user once stopped a run in alarm, thinking training was running on their own data or PC. Say plainly where anything runs.
- **Prefers FOSS tools.** Doesn't want to "vibecode" web things.
- **Explanations** should be intuitive, not equation-heavy, but should show equations and variables when needed.

---

## 1. Project goal

Classify **ferroptotic vs non-ferroptotic** neuronal-like cells (SH-SY5Y-type neuroblastoma culture, phase-contrast/brightfield 20x images) from microscopy, with an interpretable pipeline:

`image -> Cellpose-SAM single-cell crops -> frozen DINOv3 ViT-B/16 (TE), per-channel tokens -> BatchTopK sparse autoencoder (SAE) -> linear classifier (LC) on pooled sparse latents` plus a transformer decoder (TD) that reconstructs crops from SAE output, for visualising latent interventions.

The work builds on CellDeathPred (Schorpp et al., 2023), which used EfficientNet-b0 with 4-channel cell painting, a supervised-contrastive embedder and an FC classifier.

### Biology / conditions

24-well plate, four conditions:

| Code | Treatment | Status |
|---|---|---|
| **RD** | RSL3 + DMSO | The ONLY ferroptotic condition (label 1) |
| **RF** | RSL3 + ferrostatin | Rescued, not ferroptotic (label 0) |
| **DD** | DMSO + DMSO | Control (label 0) |
| **DF** | DMSO + ferrostatin | Control (label 0) |

- **Plate rows, top to bottom:** RD, RF, DD, DF (wells 1–6, 7–12, 13–18, 19–24). The same layout is used in every experiment.
- **Columns, left to right:** 0.1L, 0.1R, 0.5L, 0.5R, 1.0L, 1.0R. Concentrations are pooled per condition.
- **Concentration 1.0 is dropped** because every cell was already dead. The 1.0 wells (5, 6, 11, 12, 17, 18, 23, 24) are dropped, and the other wells keep their plate numbers (1–4, 7–10, 13–16, 19–22).

---

## 2. Repository and environment

- **Repo:** `C:\Users\k7884\Documents\KSA\sem2\ResearchSeminar`, remote `https://github.com/neuralProcessor314/pre-rne` on branch `main`. Code is GPL-3.0.
- **Files:** `plate.py`, `segment.py`, `train.py`, `infer.py`, `DECISIONS.md` (the authoritative design record), `README.md`, `LICENSE`, `colorsep.bat` (no longer used), `pyproject.toml`, `uv.lock`, `.python-version`.
- **`.gitignore`:** `docs/`, `data/`, `.venv/`, `__pycache__/`.
- **Untracked user files:** `data_archive/` and `scale.txt` (a Korean note recording that the "40x" images were really 20x). Don't commit or touch them.
- **`docs/`** (git-ignored) holds the user's model diagrams (`PreRnE Model.svg` and others), `TODO.txt`, `calc*.txt`, `celldeathpred.pdf`, and `preprocessing_augmentation_grid_cited.png`.

### Environment (uv, Windows x64, NVIDIA GPU)

`pyproject.toml` pins:
- Python `==3.12.*`
- `cellpose[gui]==4.2.1.1`
- `torch==2.15.0` and `torchvision==0.29.2`, both from the `https://download.pytorch.org/whl/cu132` index (CUDA 13.2 builds)
- `dinov3` from git, `facebookresearch/dinov3` at commit `6876159a11b4df116f30f667f8c9888617df0751`

`[tool.uv] environments = ["sys_platform == 'win32' and platform_machine == 'AMD64'"]`, `package = false`. `uv.lock` holds 49 packages, all with Windows cp312 wheels and hashes.

```
uv sync --locked
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect 2.15.0+cu132 True
uv run python -m cellpose        # Cellpose GUI (draw masks, Ctrl+T to fine-tune cpsam)
```

Hardware:
- Desktop: RTX 5070 Ti, Windows 11.
- Laptop: RTX 5060 Mobile, 8 GB VRAM.

Model weights:
- **DINOv3 ViT-B/16 (distilled from ViT-7B, LVD-1689M):** `C:\Users\k7884\Documents\models\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (327 MB). `train.py` builds this path with `Path.home() / "Documents" / "models" / ...`. Meta's signed download links expire, and the HF mirror `facebook/dinov3-vitb16-pretrain-lvd1689m` is gated.
- **Cellpose-SAM `cpsam`:** in Cellpose's default directory, `C:\Users\k7884\.cellpose\models\cpsam` (1,233,587,898 bytes, SHA-256 `e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2`). It is BSD-3 licensed, ViT-L based, ~305M parameters.
- `dinov3_repo_dir` in `train.py` is still `None` and needs a local clone of facebookresearch/dinov3, because `TE` is built with `torch.hub.load(repo_dir, "dinov3_vitb16", source="local", weights=path)`.

### Git history (newest first)

```
79a3b8e Stage data through numbered folders; train on what exists
5e06299 Train on RD: binary labels and leave-one-well-out
03e71dd Set augmentation seed and copy count
113ad9c Augment whole images before Cellpose
47c6657 Set normalisation contrast and augmentation strengths
a33cc0b Drop concentration 1.0
a053731 Normalise brightfield per image and augment training crops
a620337 move venv. also fix transformer settings   (user's commit; deleted the old ferroptosis/ uv sub-project)
7459390 Add locked uv environment with the Cellpose GUI
3d96424 Load DINOv3 weights from ~/Documents/models
48db9cb Keep cell size: square crops without resizing
c883de3 ... d179dcb  (user's initial commits, README, LICENSE)
```

---

## 3. Data (actual files, inspected 26 Nov 2026)

All images are **1360 × 1024 RGB JPEGs**, 505 in total under `data\`.

| Folder | Experiment / date | Phase | Count | Raw name pattern | Magnification |
|---|---|---|---|---|---|
| `0_after_raw\exp1_after_20261106\` | exp1, 6 Nov | after | 94 | `<code>_<conc><side>_<mag>_<num>.jpg`, e.g. `DD_0.1L_20x_39.jpg` | 69 at 20x, 25 at 4x; includes 1.0 wells |
| `0_before_raw\exp2_before_20261118\` | exp2, 18 Nov | before | 88 | `bW<well>_20x_<num>.jpg` | 20x; wells 1–24 |
| `0_before_raw\exp3_before_20261125\` | exp3, 25 Nov | before | 323 | `well<well>_40x_<num>.jpg` | Labelled 40x but actually 20x (user confirmed from the objective) |

Facts:
- **No experiment has both before and after images,** so cell-level or field-level before/after pairing is currently impossible. The user says more images are coming.
- **exp1 has no 20x after images of DF 0.5L or DF 0.5R** (only 4x).
- **exp1 appearance:** blue-tinted phase contrast, dense.
- **exp2 appearance:** grey phase contrast, clustered neurite-bearing cells.
- **exp3 appearance:** green-tinted.
- **The after images are RGB only;** there is no separate BODIPY or fluorescence image.
- **Cell size at native 20x** (the user's estimate): a single cell is about 120 × 120 px including dendrites, with many overlaps. Cellpose-SAM measurements at default settings:

  | Image | Cells | Median area | Long side median / 90th pct / max |
  |---|---|---|---|
  | exp1 `DD_0.1L_20x_39` | 5 | 396 px | 26 / 58 / 63 px |
  | exp1 `RD_0.1L_20x_25` | 81 | 724 px | 37 / 57 / 73 px |
  | exp2 `bW13_20x_27` | 103 | 1,044 px | 54 / 110 / 166 px |
  | exp2 `bW1_20x_73` | 239 | 827 px | 47 / 98 / 190 px |
  | exp3 `well13_40x_00` | 69 | 1,205 px | 66 / 117 / 228 px |
  | exp3 `well1_40x_35` | 159 | 999 px | 52 / 92 / 170 px |

  Cellpose masks tend to cover the soma plus part of the neurites. Overlapping neurites get split between cells, and dense clusters are partly unsegmented. Each image took about 80 s on CPU.

---

## 4. Current pipeline (code at `79a3b8e`)

### `plate.py`

`CODES`, `ROW_CODES = ["RD","RF","DD","DF"]`, `COLUMNS` (6 entries including 1.0), `CONCENTRATIONS = ["0.1","0.5"]`, `well_number(code, side, conc)`, `well_position(number)`, `well_name(number)` (for example 13 → `DD_L0.1`).

### `segment.py`: three data steps; running it redoes all of them and deletes and rewrites the `1_`, `2_` and `3_` folders

1. **`rename_raw()`**, from `data/0_<phase>_raw/<exp folder>/*` to `data/1_<phase>_rename/`:
   - New name format `exp<n>_<phase>W<well>_<index>.<ext>`, e.g. `exp1_afterW13_000.jpg`.
   - The index is one 3-digit counter shared by all kept images, in alphabetical order of raw path relative to `data/` (so `0_after_raw` comes before `0_before_raw`).
   - The experiment name is the folder name up to the first `_`.
   - `parse_raw_name` understands the three raw patterns above. Magnifications kept are `20x` and `40x` (the 40x files are really 20x); 4x and 10x are dropped. Unknown names are printed and skipped.
   - Writes `data/1_rename_map.csv` with the columns new_name and raw_path.
2. **`preprocess_and_augment()`**, from `1_` to `data/2_<phase>_prepaug/`:
   - Skips 1.0 wells.
   - Per-image normalisation: each of R, G and B is shifted to mean `bf_mean`=0.5 (128/255) and scaled to std `bf_std`=0.25, then clipped to [0,1].
   - Saves `<name>.png` (8-bit RGB) plus `aug_copies`=3 augmented copies `<name>-aug<k>.png`.
   - Augmentation, applied to the whole image before Cellpose, with each strength uniform in [0, max] from a generator seeded by `seed`=42, the image name and the copy number:
     1. Rotation by 0/90/180/270° plus a mirror flip half of the time.
     2. Chromatic aberration: R and B shifted against G by up to `aug_ca_max_shift_px`=4 px, with edges replicated.
     3. Uneven illumination: the image multiplied by `1 + s*(x cosθ + y sinθ)`, where x and y run over [-1,1] across the image, s ≤ `aug_illum_max_slope`=0.5 and θ is uniform over 0–360°. Negative s isn't needed because θ+180° covers it.
     4. Gaussian blur, σ ≤ `aug_blur_max_sigma`=1.5 px.
     5. Gaussian noise, std ≤ `aug_noise_max_std`=0.06.
     6. Clip to [0,1].
   - No crop augmentation, by user decision.
3. **`segment_all()`**, from `2_` to `data/3_<phase>_crops/`:
   - Cellpose-SAM (`CellposeModel(pretrained_model="cpsam")`, `channel_axis=0`) runs on every PNG.
   - For each mask: the bounding box is padded by `pad_px`, then squared using the surrounding image pixels, **without resizing**. The side is rounded up to a multiple of 16, and the square is centred and shifted back inside the image where needed. Cells larger than the image are skipped.
   - Saves `<image name>_cell<id>.pt` = `{"image": Tensor[3,S,S], "centroid": Tensor[2], "area": Tensor[]}`, where S is a multiple of 16 and varies per cell. Neighbours inside the window are not masked.
- **Unset (`None`), required before running:** `min_mask_pixels`, `pad_px`, `flow_threshold`, `cellprob_threshold`, `batch_size`.

### `train.py`

- **Constants:** `FERROPTOTIC_CODE="RD"`, `CLASSES=["not ferroptotic","ferroptotic"]`, `CHANNELS=3`, `FOLD_COLUMNS=["L0.1","R0.1","L0.5","R0.5"]`.
- **`build_fold(data_dir, fold)`:**
  - **Leave-one-well-out by plate column:** fold k holds out column `FOLD_COLUMNS[k]` in all conditions and all experiments, which is one well per condition per experiment.
  - Returns:
    - `train_singles`: all crops of both phases from training wells, augmented copies included. Used for SAE and TD.
    - `train_cells`: after crops of training wells, augmented included, as (path, label). Used for LC.
    - `heldout_cells`: original after crops of held-out wells only, with augmented copies excluded.
- **TE:** frozen DINOv3 ViT-B/16. Its RGB patch-embed kernel is averaged into a 1-channel conv, and each channel is encoded separately (`encode_channels`). It returns `x_norm_patchtokens`, with CLS and register tokens dropped. Crops of different sizes are batched by side length (`group_by_side`, `make_batches`).
- **SAE (`BatchTopKSAE`):**
  - `b_dec` is subtracted before the encoder and added after the decoder. The decoder is initialised as the encoder transpose.
  - `encode_batch_topk` keeps the top (tokens × topk) activations across the whole batch.
  - Training loss is MSE recon + `aux_weight` × AuxK, where AuxK reconstructs the residual `e = x − x_pred` using the top `topk_aux` dead latents (dead means not fired for `dead_steps` steps).
  - Tracks `theta`, the running mean of the minimum kept activation, for inference: `encode_threshold` = JumpReLU `h*(h>theta)`.
- **TD:** `nn.TransformerEncoder` over SAE-reconstructed tokens per channel, then a linear layer to patch pixels and `PixelShuffle(16)`. MSE against the crop. It has no positional embedding of its own.
- **LC:**
  - Mean-pools each crop's z over patches per channel and concatenates the channels, giving (3×d_sae). Uses `encode_threshold`.
  - Latents are computed once.
  - `nn.Linear(3*d_sae, 2)`, trained with class-weighted cross-entropy (weight = N / (2 × count_c)).
- **`train()`:** for each of the 4 folds, trains SAE, TD and LC from scratch and saves `checkpoints/fold<k>.pt` containing cfg, classes, fold, held_out_column and the state dicts.
- **Unset (`None`):** `dinov3_repo_dir`, `seed`, `d_sae`, `topk`, `topk_aux`, `dead_steps`, `aux_weight`, the `sae_*`, `td_*` and `lc_*` batch sizes, epochs and LRs, `td_dim`/`heads`/`layers`/`mlp_dim`, and `lc_weight_decay`.

### `infer.py`

For each fold, loads `checkpoints/fold<k>.pt` (TE loaded once) and runs that fold's models on its held-out after cells, one cell at a time. It saves `heldout_results.pt` = `{"classes", "results": [...]}`, where each result holds path, fold, well (for example `exp1_DD_L0.1`), label, logits, probs, z (pooled), img_recon, and optionally img_intervened. The intervention scales latent `intervene_latent` in channel `intervene_channel` (0–2) by `intervene_scale`. No augmentation is applied at inference.

### Tests done (synthetic only; never on user data)

End-to-end tests ran in the cloud workspace with synthetic noise images named like the real files, a stand-in Cellpose and randomly initialised ViT-B. They covered renaming (alphabetical index, well mapping, 4x drop), normalisation (mean 0.50, std ~0.25), crop extraction, fold splits (no held-out leakage), all 4 folds training, and inference shapes and labels. All passed.

---

## 5. Decision record (human-specified; see `DECISIONS.md`)

1. Backbone: DINOv3 ViT-B/16 from a local clone, with weights `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
2. Inputs: RGB only (3 channels), since no BODIPY images exist. The patch-embed is averaged to one channel, and each channel is tokenised separately.
3. Stages: SAE (BatchTopK, tied b_dec, AuxK), then TD, then LC on pooled z, with TE frozen. In practice the code trains SAE first, then TD, then LC.
4. Inference: no BatchTopK; JumpReLU with θ instead.
5. Conditions as in section 1. Before/after pairing waits until both phases exist in one experiment. Splits are by well.
6. `segment.py` handles the data steps and segmentation; `colorsep.bat` is unused. Training and inference scripts leave hyperparameters as `None`.
7. Inference outputs: predictions, pooled z, TD reconstructions and intervened reconstructions on held-out wells.
8. Crops keep cell size: square windows with no resizing. Single cells, not groups. The user considered groups, but dead cells may not form groups.
9. Per-image normalisation of mean RGB and contrast, centred at 0.5.
10. Augmentation is applied before Cellpose: 90° rotations and flips, noise, blur, chromatic aberration and uneven illumination. No crop augmentation.
11. Concentration 1.0 is dropped.
12. Binary labels (RD vs rest) with leave-one-well-out over 4 folds, retraining every stage per fold.
13. Data folders `0_raw`, `1_rename`, `2_prepaug` and `3_crops`, using the naming above.
14. Native 20x, no rescaling. exp3's "40x" is really 20x. 4x and 10x are dropped. A cell is about 120 px including dendrites.
15. Pairing is pending, so LC classifies single after cells and before cells only train the SAE and TD.
16. Plate rows are RD, RF, DD, DF.

---

## 6. Open issues / known limitations / likely next steps

- **Before/after pairing:** more images are coming. Under the new naming the image index is global, so before and after photos of the same field can't be matched by name. A correspondence scheme is needed, such as a mapping file or a shared field index. The old code had centroid-based cell pairing within the same (well, fov); it was removed in `79a3b8e`.
- **Cellpose settings are unset,** so `segment.py` won't run yet.
- **Planned fine-tuning:** fine-tune Cellpose-SAM with the Cellpose GUI on the real 20x images. `segment.py` still loads `pretrained_model="cpsam"` and must point to the custom model afterwards.
- **TE input isn't ImageNet-normalised.** Crops in [0,1] are fed directly, while DINOv3 was trained on inputs normalised to roughly zero mean and unit variance. This was flagged and not fixed.
- **Mean pooling over patches removes cell size.** `area` is saved but LC doesn't read it. Max, mean+max and size-as-feature were discussed but not decided.
- **Small crops:** median Cellpose crops are about 50–70 px (3–4 patches per side), below DINOv3's smallest training crop of 112 px (7×7 patches). It runs because RoPE scales to the grid, but feature quality at small grids is untested.
- **Small held-out sets:** each held-out set is one well per condition per experiment. Folds 3 and 4 have no DF after cells (no 20x DF 0.5 after images). Training takes about 4× as long because of the four folds.
- **Augmentation is offline:** every epoch sees the same 1 + aug_copies versions. Cellpose runs 4× per image.
- **Masks cut by the image border** understate size, and those cells are kept.
- **BODIPY C11 581/591:** if fluorescence images ever arrive, note that the dye shifts emission from red to green on oxidation, so a single linear channel can't carry the ratio.
- **Earlier conceptual suggestions,** not implemented:
  - Columbus/CellProfiler feature correlation of SAE latents for interpretability.
  - ATP/viability (CellTiter-Glo) regression to separate death-modality latents from generic-toxicity latents.
  - L1 on LC weights.
  - (m, k) sweeps with FVU–sparsity frontiers.
  - Max-activating crop galleries.
  - Causal latent ablation through TD.
  - Plate-level held-out splits in multi-plate settings.

---

## 7. References (APA)

- Schorpp, K., Bessadok, A., Biibosunov, A., Rothenaigner, I., Strasser, S., Peng, T., & Hadian, K. (2023). CellDeathPred: A deep learning framework for ferroptosis and apoptosis prediction based on cell painting. *Cell Death Discovery, 9*, 277. https://doi.org/10.1038/s41420-023-01559-y
- Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., et al. (2025). *DINOv3* (arXiv:2508.10104). arXiv. https://arxiv.org/abs/2508.10104
- Pachitariu, M., Rariden, M., & Stringer, C. (2025). *Cellpose-SAM: Superhuman generalization for cellular segmentation* [Preprint]. bioRxiv. https://doi.org/10.1101/2025.04.28.651001
- Michelucci, U. (2022). *An introduction to autoencoders* (arXiv:2201.03898). arXiv. https://doi.org/10.48550/arXiv.2201.03898
- Offringa, R. (2006). *BiggeggSH-SY5Y.jpg* [Photograph of the SH-SY5Y neuroblastoma cell line]. Wikimedia Commons. https://commons.wikimedia.org/wiki/File:BiggeggSH-SY5Y.jpg (CC BY-SA 2.5; the Commons description lists the author as anonymous, while the upload note names Reid Offringa). Used for the augmentation demo grid in `docs/preprocessing_augmentation_grid_cited.png`.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998–6008.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*. arXiv:1409.0473.
