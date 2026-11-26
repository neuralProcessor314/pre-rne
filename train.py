from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from plate import well_position

# Model dimensions, paths, and training hyperparameters
CFG = {
    "dinov3_repo_dir": None,    # str: path to local facebookresearch/dinov3 clone
    "dinov3_weights": str(Path.home() / "Documents" / "models" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    "data_dir": Path("data"),          # crops are read from data/3_before_crops and data/3_after_crops
    "ckpt_dir": Path("checkpoints"),    # one checkpoint per fold: fold1.pt ... fold4.pt
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": None,               # int: RNG seed
    # Stage 1: SAE
    "d_sae": None,              # int: SAE latent dictionary size
    "topk": None,               # int: average active latents per token (BatchTopK)
    "topk_aux": None,           # int: active dead latents in auxiliary loss
    "dead_steps": None,         # int: steps without firing before latent is dead
    "aux_weight": None,         # float: weight on TopK_aux loss
    "sae_batch_size": None,     # int
    "sae_epochs": None,         # int
    "sae_lr": None,             # float
    # Stage 2: Transformer Decoder (TD)
    "td_dim": None,             # int: TD transformer hidden dimension
    "td_heads": None,           # int: TD attention heads
    "td_layers": None,          # int: TD transformer layer count
    "td_mlp_dim": None,         # int: TD feedforward dimension
    "td_batch_size": None,      # int
    "td_epochs": None,          # int
    "td_lr": None,              # float
    # Stage 3: Linear Classifier (LC)
    "lc_batch_size": None,      # int
    "lc_epochs": None,          # int
    "lc_lr": None,              # float
    "lc_weight_decay": None,    # float
}

# RD (RSL3 + DMSO) is the only ferroptotic condition: LC predicts ferroptotic (1, RD) or
# not ferroptotic (0, DD, DF, RF)
FERROPTOTIC_CODE = "RD"
CLASSES = ["not ferroptotic", "ferroptotic"]
CHANNELS = 3        # R, G, B, each tokenised by TE as its own grayscale image

# Leave-one-well-out folds: fold k holds out the wells in plate column FOLD_COLUMNS[k] of every
# condition (one well per condition, in every experiment) and trains on the other three columns
FOLD_COLUMNS = ["L0.1", "R0.1", "L0.5", "R0.5"]


# ---------------- Data ----------------

# Read a crop file name such as "exp1_afterW13_012-aug1_cell0003"
def parse_crop_name(stem):
    experiment, phase_well, image, cell = stem.split("_")
    phase, number = phase_well.split("W")
    return experiment, phase, int(number), image


# Label of a condition: 1 = ferroptotic (RD), 0 = not ferroptotic
def label_of(code):
    if code == FERROPTOTIC_CODE:
        return 1
    return 0


# Every crop in 3_before_crops and 3_after_crops, with its experiment, phase, well, condition and plate column
def list_crops(data_dir):
    crops = []
    for phase in ["before", "after"]:
        for path in sorted((data_dir / f"3_{phase}_crops").glob("*.pt")):
            experiment, file_phase, number, image = parse_crop_name(path.stem)
            code, side, conc = well_position(number)
            crops.append({
                "path": path,
                "experiment": experiment,
                "phase": phase,
                "well": experiment + "_" + code + "_" + side + conc,    # e.g. "exp1_DD_L0.1"
                "code": code,
                "column": side + conc,                                    # e.g. "L0.1"
                "augmented": "-aug" in image,                             # augmented copy made by segment.py
            })
    return crops


# Crops of one fold. Training: every crop of the other three columns' wells, augmented copies included.
# Held out: one well per condition, original images only (its augmented copies are not used).
# Before and after images are not paired yet, so LC works on single after cells.
# Returns training crops of both phases (Stages 1-2), training after cells and held-out after cells,
# each a list of (path, label).
def build_fold(data_dir, fold):
    train_singles = []
    train_cells = []
    heldout_cells = []
    for crop in list_crops(data_dir):
        held_out = crop["column"] == FOLD_COLUMNS[fold]
        if held_out and crop["augmented"]:
            continue
        if not held_out:
            train_singles.append(crop["path"])
        if crop["phase"] == "after":
            cell = (crop["path"], label_of(crop["code"]))
            if held_out:
                heldout_cells.append(cell)
            else:
                train_cells.append(cell)
    return train_singles, train_cells, heldout_cells


# Crops are not resized, so their sizes differ, and crops of different sizes cannot share a batch.
# Group crop paths by side length: {side: [paths]}
def group_by_side(paths):
    groups = {}
    for path in paths:
        side = torch.load(path, weights_only=True)["image"].shape[-1]
        if side not in groups:
            groups[side] = []
        groups[side].append(path)
    return groups


# Shuffle each size group, cut it into batches, then shuffle the order of the batches
def make_batches(groups, batch_size, rng):
    batches = []
    for side in sorted(groups):
        paths = list(groups[side])
        rng.shuffle(paths)
        for start in range(0, len(paths), batch_size):
            batches.append(paths[start:start + batch_size])
    rng.shuffle(batches)
    return batches


# Load the images of same-sized crops as one tensor (B, 4, side, side)
def load_images(paths):
    images = []
    for path in paths:
        images.append(torch.load(path, weights_only=True)["image"])
    return torch.stack(images)


# ---------------- Models ----------------

# Load frozen DINOv3 ViT-B/16 and replace its RGB patch projection with a 1-channel one
def load_frozen_dinov3_single_channel(repo_dir, weights_path, device):
    te = torch.hub.load(repo_dir, "dinov3_vitb16", source="local", weights=weights_path)

    # 1-channel projection initialised with the average of the R, G, B kernels
    old_proj = te.patch_embed.proj
    new_proj = nn.Conv2d(1, old_proj.out_channels, kernel_size=old_proj.kernel_size, stride=old_proj.stride)
    with torch.no_grad():
        new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
        new_proj.bias.copy_(old_proj.bias)
    te.patch_embed.proj = new_proj

    te.eval()
    te.requires_grad_(False)
    return te.to(device)


# Encode each channel as its own grayscale image
# imgs (B, C, H, W) -> tokens (B, C, T, D), where T = hp * wp patches
def encode_channels(te, imgs):
    b, c, h, w = imgs.shape
    patch = te.patch_embed.patch_size[0]
    hp = h // patch
    wp = w // patch
    with torch.no_grad():
        flat = imgs.reshape(b * c, 1, h, w)
        tokens = te.forward_features(flat)["x_norm_patchtokens"]    # (B*C, T, D)
    tokens = tokens.reshape(b, c, hp * wp, te.embed_dim)
    return tokens, hp, wp


# Average SAE latents over patches: (B*C*T, d_sae) -> (B, C*d_sae)
def pool_over_patches(z, b, c, t):
    z = z.reshape(b, c, t, -1)
    z = z.mean(dim=2)
    return z.reshape(b, -1)


# Sparse autoencoder with tied decoder bias, BatchTopK, TopK_aux, and a threshold for inference
class BatchTopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, topk, topk_aux, dead_steps):
        super().__init__()
        self.topk = topk
        self.topk_aux = topk_aux
        self.dead_steps = dead_steps

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())   # decoder starts as encoder transpose

        # Training statistics, saved with the model
        self.register_buffer("steps_since_active", torch.zeros(d_sae, dtype=torch.long))
        self.register_buffer("theta_sum", torch.tensor(0.0))
        self.register_buffer("theta_count", torch.tensor(0.0))

    # Non-negative latent activations before any sparsity is applied
    def preactivations(self, x):
        return F.relu(self.encoder(x - self.b_dec))

    # BatchTopK: keep the (tokens x topk) largest activations of the whole batch, zero the rest
    def encode_batch_topk(self, x):
        h = self.preactivations(x)
        n_keep = min(h.shape[0] * self.topk, h.numel())
        values, positions = torch.topk(h.flatten(), n_keep)
        z = torch.zeros_like(h.flatten())
        z[positions] = values                     # put the kept values back in place
        return z.reshape(h.shape)

    # Inference: keep activations above theta, the average smallest kept activation in training
    def encode_threshold(self, x):
        h = self.preactivations(x)
        theta = self.theta_sum / self.theta_count
        return h * (h > theta).float()

    def decode(self, z):
        return self.decoder(z) + self.b_dec

    # Stage-1 losses: reconstruction loss and dead-latent auxiliary loss
    def training_losses(self, x):
        z = self.encode_batch_topk(x)
        x_pred = self.decode(z)
        self.update_statistics(z)
        recon_loss = F.mse_loss(x_pred, x)
        aux_loss = self.aux_loss(x, x_pred)
        return recon_loss, aux_loss

    # Track how long each latent has been silent, and add this batch's smallest kept activation to theta
    def update_statistics(self, z):
        with torch.no_grad():
            fired = (z > 0).any(dim=0)
            self.steps_since_active += 1
            self.steps_since_active[fired] = 0

            kept = z[z > 0]
            if kept.numel() > 0:
                self.theta_sum += kept.min()
                self.theta_count += 1

    # TopK_aux: reconstruct the residual e = x - x_pred using only the top-k_aux dead latents
    def aux_loss(self, x, x_pred):
        dead = self.steps_since_active >= self.dead_steps
        n_dead = int(dead.sum())
        if n_dead == 0:
            return torch.tensor(0.0, device=x.device)

        e = (x - x_pred).detach()
        h = F.relu(self.encoder(e)) * dead.float()                # live latents set to 0
        k = min(self.topk_aux, n_dead)
        kth_largest = torch.topk(h, k, dim=1).values[:, -1:]       # k-th largest of each token
        z_dead = h * (h >= kth_largest).float()                    # keep each token's top k
        e_pred = self.decoder(z_dead)
        return F.mse_loss(e_pred, e)


# Transformer decoder: SAE-reconstructed patch tokens of one channel -> that channel's pixels
class TransformerDecoder(nn.Module):
    def __init__(self, d_in, patch_size, td_dim, td_heads, td_layers, td_mlp_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_proj = nn.Linear(d_in, td_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=td_dim,
            nhead=td_heads,
            dim_feedforward=td_mlp_dim,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=td_layers)
        self.out_proj = nn.Linear(td_dim, patch_size * patch_size)   # pixels of one patch
        self.shuffle = nn.PixelShuffle(patch_size)

    # tokens (B, C, T, D) -> images (B, C, hp*p, wp*p)
    def forward(self, tokens, hp, wp):
        b, c, t, d = tokens.shape
        p = self.patch_size
        x = tokens.reshape(b * c, t, d)            # each channel decoded on its own
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_proj(x)                       # (B*C, T, p*p)
        x = x.transpose(1, 2)                      # (B*C, p*p, T)
        x = x.reshape(b * c, p * p, hp, wp)        # patch grid
        x = self.shuffle(x)                        # (B*C, 1, hp*p, wp*p)
        return x.reshape(b, c, hp * p, wp * p)


# ---------------- Training ----------------

# Stage 1: train the SAE on patch tokens of every channel (TE frozen)
def train_sae(te, image_paths, device):
    d_in = te.embed_dim
    sae = BatchTopKSAE(
        d_in=d_in,
        d_sae=CFG["d_sae"],
        topk=CFG["topk"],
        topk_aux=CFG["topk_aux"],
        dead_steps=CFG["dead_steps"],
    ).to(device)
    groups = group_by_side(image_paths)
    rng = random.Random(CFG["seed"])
    optimizer = torch.optim.AdamW(sae.parameters(), lr=CFG["sae_lr"])

    sae.train()
    for epoch in range(CFG["sae_epochs"]):
        for batch_paths in make_batches(groups, CFG["sae_batch_size"], rng):
            imgs = load_images(batch_paths)
            tokens, hp, wp = encode_channels(te, imgs.to(device))
            tokens = tokens.reshape(-1, d_in)       # all patch tokens of all channels
            recon_loss, aux_loss = sae.training_losses(tokens)
            loss = recon_loss + CFG["aux_weight"] * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    sae.eval()
    sae.requires_grad_(False)
    return sae


# Stage 2: train TD to turn SAE reconstructions back into images (TE and SAE frozen)
def train_td(te, sae, image_paths, device):
    d_in = te.embed_dim
    td = TransformerDecoder(
        d_in=d_in,
        patch_size=te.patch_embed.patch_size[0],
        td_dim=CFG["td_dim"],
        td_heads=CFG["td_heads"],
        td_layers=CFG["td_layers"],
        td_mlp_dim=CFG["td_mlp_dim"],
    ).to(device)
    groups = group_by_side(image_paths)
    rng = random.Random(CFG["seed"])
    optimizer = torch.optim.AdamW(td.parameters(), lr=CFG["td_lr"])

    td.train()
    for epoch in range(CFG["td_epochs"]):
        for batch_paths in make_batches(groups, CFG["td_batch_size"], rng):
            imgs = load_images(batch_paths).to(device)
            with torch.no_grad():
                tokens, hp, wp = encode_channels(te, imgs)
                b, c, t, d = tokens.shape
                z = sae.encode_batch_topk(tokens.reshape(-1, d))
                x_pred = sae.decode(z).reshape(b, c, t, d)
            img_pred = td(x_pred, hp, wp)
            loss = F.mse_loss(img_pred, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    td.eval()
    td.requires_grad_(False)
    return td


# Patch-averaged SAE latents of each crop: {path: Tensor[CHANNELS * d_sae]}
# Uses the inference threshold (as infer.py does), so a crop's latents do not depend on
# which other crops share its batch.
def pooled_latents(te, sae, groups, batch_size, device):
    latents = {}
    with torch.no_grad():
        for side in sorted(groups):
            group = groups[side]
            for start in range(0, len(group), batch_size):
                batch_paths = group[start:start + batch_size]
                imgs = load_images(batch_paths).to(device)
                tokens, hp, wp = encode_channels(te, imgs)
                b, c, t, d = tokens.shape
                z = sae.encode_threshold(tokens.reshape(-1, d))
                pooled = pool_over_patches(z, b, c, t)
                for i in range(b):
                    latents[batch_paths[i]] = pooled[i]
    return latents


# Stage 3: train LC (ferroptotic or not) on the patch-averaged SAE latents of single after cells
# (TE and SAE frozen)
def train_lc(te, sae, cells, device):
    paths = []
    labels = []
    for path, label in cells:
        paths.append(path)
        labels.append(label)
    labels = torch.tensor(labels, device=device)

    # One RD condition against three others: weight each class by 1 / its share of the cells,
    # so both classes count equally in the loss
    class_weights = []
    for c in range(len(CLASSES)):
        count = int((labels == c).sum())
        if count == 0:
            raise ValueError(f"no training cells of class '{CLASSES[c]}'")
        class_weights.append(len(labels) / (len(CLASSES) * count))
    class_weights = torch.tensor(class_weights, device=device)

    lc = nn.Linear(CHANNELS * CFG["d_sae"], len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(lc.parameters(), lr=CFG["lc_lr"], weight_decay=CFG["lc_weight_decay"])

    # TE and SAE are frozen and crops are fixed, so every crop's latents are computed once
    latents = pooled_latents(te, sae, group_by_side(paths), CFG["lc_batch_size"], device)
    features = []
    for path in paths:
        features.append(latents[path])
    features = torch.stack(features)                          # (N, CHANNELS * d_sae)

    lc.train()
    for epoch in range(CFG["lc_epochs"]):
        order = torch.randperm(len(labels), device=device)
        for start in range(0, len(labels), CFG["lc_batch_size"]):
            idx = order[start:start + CFG["lc_batch_size"]]
            logits = lc(features[idx])
            loss = F.cross_entropy(logits, labels[idx], weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return lc


# Leave-one-well-out: for each fold, train the three stages from scratch on that fold's training
# wells and save a checkpoint; infer.py then runs each checkpoint on its held-out wells
def train():
    device = CFG["device"]
    te = load_frozen_dinov3_single_channel(CFG["dinov3_repo_dir"], CFG["dinov3_weights"], device)
    CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)

    for fold in range(len(FOLD_COLUMNS)):
        torch.manual_seed(CFG["seed"])
        train_singles, train_cells, heldout_cells = build_fold(CFG["data_dir"], fold)
        print(f"fold {fold + 1}: held-out column {FOLD_COLUMNS[fold]}, "
              f"{len(train_singles)} training crops, {len(train_cells)} training after cells, "
              f"{len(heldout_cells)} held-out after cells")

        sae = train_sae(te, train_singles, device)
        td = train_td(te, sae, train_singles, device)
        lc = train_lc(te, sae, train_cells, device)

        torch.save(
            {
                "cfg": CFG,
                "classes": CLASSES,
                "fold": fold + 1,
                "held_out_column": FOLD_COLUMNS[fold],
                "sae": sae.state_dict(),
                "td": td.state_dict(),
                "lc": lc.state_dict(),
            },
            CFG["ckpt_dir"] / f"fold{fold + 1}.pt",
        )


if __name__ == "__main__":
    train()
