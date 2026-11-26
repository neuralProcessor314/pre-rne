from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import (
    CLASSES,
    FOLD_COLUMNS,
    BatchTopKSAE,
    TransformerDecoder,
    build_fold,
    encode_four_channels,
    load_frozen_dinov3_single_channel,
    load_images,
    parse_well_fov,
    pool_over_patches,
)

# Inference paths and optional latent-intervention settings
INFER_CFG = {
    "ckpt_dir": Path("checkpoints"),         # fold1.pt ... fold4.pt from train.py
    "crop_dir": Path("data/crops"),
    "out_path": Path("heldout_results.pt"),
    "intervene_channel": None,       # int or None: channel 0..3 to edit before TD
    "intervene_latent": None,        # int or None: latent 0..d_sae-1 to edit before TD
    "intervene_scale": None,         # float or None: multiplier (e.g. 0.0 to remove the latent)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# Rebuild SAE, TD, and LC of one fold from its checkpoint (TE is shared by all folds)
def load_fold_models(ckpt, te, device):
    cfg = ckpt["cfg"]

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

    return sae, td, lc


# Leave-one-well-out: run each fold's models on that fold's held-out wells, and export
# predictions, true labels, latents, and TD reconstructions for every held-out pair
def run_inference():
    device = INFER_CFG["device"]
    first = torch.load(INFER_CFG["ckpt_dir"] / "fold1.pt", map_location=device, weights_only=False)
    te = load_frozen_dinov3_single_channel(first["cfg"]["dinov3_repo_dir"], first["cfg"]["dinov3_weights"], device)

    channel = INFER_CFG["intervene_channel"]
    latent = INFER_CFG["intervene_latent"]
    scale = INFER_CFG["intervene_scale"]
    do_intervene = channel is not None and latent is not None and scale is not None

    # One pair at a time: crops are not resized, so before and after crops differ in size
    results = []
    with torch.no_grad():
        for fold in range(len(FOLD_COLUMNS)):
            ckpt = torch.load(INFER_CFG["ckpt_dir"] / f"fold{fold + 1}.pt", map_location=device, weights_only=False)
            sae, td, lc = load_fold_models(ckpt, te, device)
            d_sae = ckpt["cfg"]["d_sae"]
            train_singles, train_pairs, heldout_pairs = build_fold(
                INFER_CFG["crop_dir"], fold, ckpt["cfg"]["max_pair_dist_px"]
            )

            for before_path, after_path, label in heldout_pairs:
                before = load_images([before_path]).to(device)     # (1, 4, side, side)
                after = load_images([after_path]).to(device)

                # Sparse latents with the threshold instead of BatchTopK
                tok_before, hp_before, wp_before = encode_four_channels(te, before)
                tok_after, hp, wp = encode_four_channels(te, after)
                b, c, t_before, d = tok_before.shape
                b, c, t, d = tok_after.shape
                z_before_tokens = sae.encode_threshold(tok_before.reshape(-1, d))
                z_after_tokens = sae.encode_threshold(tok_after.reshape(-1, d))

                # Classify the before -> after change
                z_before = pool_over_patches(z_before_tokens, b, c, t_before)
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

                entry = {
                    "path": str(after_path),
                    "fold": fold + 1,
                    "well": parse_well_fov(after_path.stem)[0],
                    "label": label,                         # true class: 1 ferroptotic (RD), 0 not
                    "logits": logits[0].cpu(),
                    "probs": probs[0].cpu(),
                    "z_before": z_before[0].cpu(),
                    "z_after": z_after[0].cpu(),
                    "z_delta": z_delta[0].cpu(),
                    "img_recon": img_recon[0].cpu(),
                }
                if img_intervened is not None:
                    entry["img_intervened"] = img_intervened[0].cpu()
                results.append(entry)

    torch.save({"classes": CLASSES, "results": results}, INFER_CFG["out_path"])


if __name__ == "__main__":
    run_inference()
