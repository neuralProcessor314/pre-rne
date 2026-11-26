from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model dimensions, paths, and training hyperparameters
CFG = {
    "dinov3_repo_dir": None,    # str: path to local facebookresearch/dinov3 clone
    "dinov3_weights": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "crop_dir": Path("data/crops"),
    "ckpt_out": Path("checkpoint.pt"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "val_well_frac": None,      # float: fraction of wells per control condition for val
    "seed": None,               # int: RNG seed
    "max_pair_dist_px": None,   # float: max centroid distance for before/after pairing
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

# Condition codes: DD = DMSO + DMSO, DF = DMSO + ferrostatin, RF = RSL3 + ferrostatin, RD = RSL3 + DMSO
CONTROL_CLASSES = ["DD", "DF", "RF"]
TEST_CODE = "RD"


# ---------------- Data ----------------

# Split a crop file name such as "DD_L0.5_12_cell0003" into well "DD_L0.5" and FOV "12"
def parse_well_fov(stem):
    code, side_conc, fov, cell = stem.split("_")
    return code + "_" + side_conc, fov


# List every crop except the test condition, with its group (before/control), condition, well, and FOV
def list_crops(crop_dir):
    crops = []
    for path in sorted(crop_dir.glob("*/*/*.pt")):
        group = path.parent.parent.name
        cond = path.parent.name
        if cond == TEST_CODE:
            continue
        well, fov = parse_well_fov(path.stem)
        crops.append({"path": path, "group": group, "cond": cond, "well": well, "fov": fov})
    return crops


# Pick validation wells separately for each control condition
def choose_val_wells(crops):
    rng = random.Random(CFG["seed"])
    val_wells = []
    for cond in CONTROL_CLASSES:
        wells = []
        for crop in crops:
            if crop["cond"] == cond and crop["well"] not in wells:
                wells.append(crop["well"])
        wells.sort()
        rng.shuffle(wells)
        n_val = max(1, int(len(wells) * CFG["val_well_frac"]))
        val_wells.extend(wells[:n_val])
    return val_wells


# Group before crops by (well, fov), storing each crop's path and centroid
def index_before(before_paths):
    index = {}
    for path in before_paths:
        well, fov = parse_well_fov(path.stem)
        centroid = torch.load(path, weights_only=True)["centroid"]
        if (well, fov) not in index:
            index[(well, fov)] = []
        index[(well, fov)].append((path, centroid))
    return index


# Return the before crop with the nearest centroid, or None if none is within max_dist
def find_nearest_before(after_path, candidates, max_dist):
    after_centroid = torch.load(after_path, weights_only=True)["centroid"]
    best_path = None
    best_dist = None
    for before_path, before_centroid in candidates:
        dist = torch.norm(after_centroid - before_centroid).item()
        if best_dist is None or dist < best_dist:
            best_path = before_path
            best_dist = dist
    if best_dist is None or best_dist > max_dist:
        return None
    return best_path


# Split crops into train/val by well, and pair each control after-crop with its before-crop
def build_splits(crop_dir):
    crops = list_crops(crop_dir)
    val_wells = choose_val_wells(crops)

    # Single crops for Stages 1-2
    train_singles = []
    val_singles = []
    before_paths = []
    for crop in crops:
        if crop["well"] in val_wells:
            val_singles.append(crop["path"])
        else:
            train_singles.append(crop["path"])
        if crop["group"] == "before":
            before_paths.append(crop["path"])

    # (before, after, label) pairs for Stage 3
    before_index = index_before(before_paths)
    train_pairs = []
    val_pairs = []
    for crop in crops:
        if crop["group"] != "control":
            continue
        candidates = before_index.get((crop["well"], crop["fov"]), [])
        before_path = find_nearest_before(crop["path"], candidates, CFG["max_pair_dist_px"])
        if before_path is None:
            continue
        label = CONTROL_CLASSES.index(crop["cond"])
        pair = (before_path, crop["path"], label)
        if crop["well"] in val_wells:
            val_pairs.append(pair)
        else:
            train_pairs.append(pair)

    return train_singles, val_singles, train_pairs, val_pairs


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


# Encode each of the 4 channels as its own grayscale image
# imgs (B, 4, H, W) -> tokens (B, 4, T, D), where T = hp * wp patches
def encode_four_channels(te, imgs):
    b, c, h, w = imgs.shape
    patch = te.patch_embed.patch_size[0]
    hp = h // patch
    wp = w // patch
    with torch.no_grad():
        flat = imgs.reshape(b * c, 1, h, w)
        tokens = te.forward_features(flat)["x_norm_patchtokens"]    # (B*4, T, D)
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
            tokens, hp, wp = encode_four_channels(te, imgs.to(device))
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
                tokens, hp, wp = encode_four_channels(te, imgs)
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


# Patch-averaged SAE latents of each crop: {path: Tensor[4 * d_sae]}
# Uses the inference threshold (as infer.py does), so a crop's latents do not depend on
# which other crops share its batch. Before and after crops of one cell usually differ in size.
def pooled_latents(te, sae, paths, batch_size, device):
    groups = group_by_side(paths)
    latents = {}
    with torch.no_grad():
        for side in sorted(groups):
            group = groups[side]
            for start in range(0, len(group), batch_size):
                batch_paths = group[start:start + batch_size]
                imgs = load_images(batch_paths).to(device)
                tokens, hp, wp = encode_four_channels(te, imgs)
                b, c, t, d = tokens.shape
                z = sae.encode_threshold(tokens.reshape(-1, d))
                pooled = pool_over_patches(z, b, c, t)
                for i in range(b):
                    latents[batch_paths[i]] = pooled[i]
    return latents


# Stage 3: train LC on the before -> after change of patch-averaged SAE latents (TE and SAE frozen)
def train_lc(te, sae, pairs, device):
    # TE and SAE are frozen, so every crop's latents are computed once and reused in every epoch
    crop_paths = []
    seen = set()
    for before_path, after_path, label in pairs:
        for path in [before_path, after_path]:
            if path not in seen:
                seen.add(path)
                crop_paths.append(path)
    latents = pooled_latents(te, sae, crop_paths, CFG["lc_batch_size"], device)

    features = []
    labels = []
    for before_path, after_path, label in pairs:
        features.append(latents[after_path] - latents[before_path])
        labels.append(label)
    features = torch.stack(features)                          # (N, 4 * d_sae)
    labels = torch.tensor(labels, device=device)

    lc = nn.Linear(4 * CFG["d_sae"], len(CONTROL_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(lc.parameters(), lr=CFG["lc_lr"], weight_decay=CFG["lc_weight_decay"])

    lc.train()
    for epoch in range(CFG["lc_epochs"]):
        order = torch.randperm(len(labels), device=device)
        for start in range(0, len(labels), CFG["lc_batch_size"]):
            idx = order[start:start + CFG["lc_batch_size"]]
            logits = lc(features[idx])
            loss = F.cross_entropy(logits, labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return lc


# Run the three training stages in order and save the checkpoint
def train():
    device = CFG["device"]
    torch.manual_seed(CFG["seed"])

    te = load_frozen_dinov3_single_channel(CFG["dinov3_repo_dir"], CFG["dinov3_weights"], device)
    train_singles, val_singles, train_pairs, val_pairs = build_splits(CFG["crop_dir"])

    sae = train_sae(te, train_singles, device)
    td = train_td(te, sae, train_singles, device)
    lc = train_lc(te, sae, train_pairs, device)

    torch.save(
        {
            "cfg": CFG,
            "classes": CONTROL_CLASSES,
            "sae": sae.state_dict(),
            "td": td.state_dict(),
            "lc": lc.state_dict(),
        },
        CFG["ckpt_out"],
    )


if __name__ == "__main__":
    train()
