from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train import (
    TEST_CODE,
    BatchTopKSAE,
    TransformerDecoder,
    encode_four_channels,
    find_nearest_before,
    index_before,
    load_frozen_dinov3_single_channel,
    parse_well_fov,
    pool_over_patches,
)

# Inference paths, batch size, and optional latent-intervention settings
INFER_CFG = {
    "ckpt_path": Path("checkpoint.pt"),
    "crop_dir": Path("data/crops"),
    "out_path": Path("test_RD_results.pt"),
    "batch_size": None,              # int
    "max_pair_dist_px": None,        # float: max centroid distance for before/test pairing
    "intervene_channel": None,       # int or None: channel 0..3 to edit before TD
    "intervene_latent": None,        # int or None: latent 0..d_sae-1 to edit before TD
    "intervene_scale": None,         # float or None: multiplier (e.g. 0.0 to remove the latent)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# Test (RD) after-crops, each paired with the nearest before-crop of the same well and FOV
class TestPairedDataset(Dataset):
    def __init__(self, crop_dir, max_dist):
        before_paths = sorted((crop_dir / "before" / TEST_CODE).glob("*.pt"))
        before_index = index_before(before_paths)

        self.pairs = []
        for after_path in sorted((crop_dir / "test" / TEST_CODE).glob("*.pt")):
            well, fov = parse_well_fov(after_path.stem)
            candidates = before_index.get((well, fov), [])
            before_path = find_nearest_before(after_path, candidates, max_dist)
            if before_path is not None:
                self.pairs.append((before_path, after_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        before_path, after_path = self.pairs[idx]
        before = torch.load(before_path, weights_only=True)["image"]
        after = torch.load(after_path, weights_only=True)["image"]
        return before, after, str(after_path)


# Rebuild TE, SAE, TD, and LC from the checkpoint
def load_models(ckpt, device):
    cfg = ckpt["cfg"]
    te = load_frozen_dinov3_single_channel(cfg["dinov3_repo_dir"], cfg["dinov3_weights"], device)

    sae = BatchTopKSAE(
        d_in=te.embed_dim,
        d_sae=cfg["d_sae"],
        topk=cfg["topk"],
        topk_aux=cfg["topk_aux"],
        dead_steps=cfg["dead_steps"],
    ).to(device)
    sae.load_state_dict(ckpt["sae"])
    sae.eval()

    td = TransformerDecoder(
        d_in=te.embed_dim,
        patch_size=te.patch_embed.patch_size[0],
        td_dim=cfg["td_dim"],
        td_heads=cfg["td_heads"],
        td_layers=cfg["td_layers"],
        td_mlp_dim=cfg["td_mlp_dim"],
    ).to(device)
    td.load_state_dict(ckpt["td"])
    td.eval()

    lc = nn.Linear(4 * cfg["d_sae"], len(ckpt["classes"])).to(device)
    lc.load_state_dict(ckpt["lc"])
    lc.eval()

    return te, sae, td, lc


# Classify test pairs and export latents, predictions, and TD reconstructions
def run_inference():
    device = INFER_CFG["device"]
    ckpt = torch.load(INFER_CFG["ckpt_path"], map_location=device, weights_only=False)
    te, sae, td, lc = load_models(ckpt, device)
    d_sae = ckpt["cfg"]["d_sae"]

    dataset = TestPairedDataset(INFER_CFG["crop_dir"], INFER_CFG["max_pair_dist_px"])
    loader = DataLoader(dataset, batch_size=INFER_CFG["batch_size"], shuffle=False)

    channel = INFER_CFG["intervene_channel"]
    latent = INFER_CFG["intervene_latent"]
    scale = INFER_CFG["intervene_scale"]
    do_intervene = channel is not None and latent is not None and scale is not None

    results = []
    with torch.no_grad():
        for before, after, paths in loader:
            before = before.to(device)
            after = after.to(device)

            # Sparse latents with the threshold instead of BatchTopK
            tok_before, hp, wp = encode_four_channels(te, before)
            tok_after, hp, wp = encode_four_channels(te, after)
            b, c, t, d = tok_after.shape
            z_before_tokens = sae.encode_threshold(tok_before.reshape(-1, d))
            z_after_tokens = sae.encode_threshold(tok_after.reshape(-1, d))

            # Classify the before -> after change
            z_before = pool_over_patches(z_before_tokens, b, c, t)
            z_after = pool_over_patches(z_after_tokens, b, c, t)
            z_delta = z_after - z_before
            logits = lc(z_delta)
            probs = F.softmax(logits, dim=1)

            # Reconstruct the after image from its latents
            x_pred = sae.decode(z_after_tokens).reshape(b, c, t, d)
            img_recon = td(x_pred, hp, wp)

            # Optionally scale one latent in one channel and reconstruct again
            img_intervened = None
            if do_intervene:
                z_edit = z_after_tokens.reshape(b, c, t, d_sae).clone()
                z_edit[:, channel, :, latent] = z_edit[:, channel, :, latent] * scale
                x_edit = sae.decode(z_edit.reshape(-1, d_sae)).reshape(b, c, t, d)
                img_intervened = td(x_edit, hp, wp)

            for i in range(b):
                entry = {
                    "path": paths[i],
                    "logits": logits[i].cpu(),
                    "probs": probs[i].cpu(),
                    "z_before": z_before[i].cpu(),
                    "z_after": z_after[i].cpu(),
                    "z_delta": z_delta[i].cpu(),
                    "img_recon": img_recon[i].cpu(),
                }
                if img_intervened is not None:
                    entry["img_intervened"] = img_intervened[i].cpu()
                results.append(entry)

    torch.save({"classes": ckpt["classes"], "results": results}, INFER_CFG["out_path"])


if __name__ == "__main__":
    run_inference()
