from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import (
    CHANNELS,
    CLASSES,
    FOLD_COLUMNS,
    POOLINGS,
    BatchTopKSAE,
    TransformerDecoder,
    build_fold,
    encode_channels,
    load_frozen_dinov3_single_channel,
    load_images,
    parse_crop_name,
    pool_over_patches,
)
from plate import well_name

# Inference paths and optional latent-intervention settings
INFER_CFG = {
    "ckpt_dir": Path("checkpoints"),         # fold1.pt ... fold4.pt from train.py
    "data_dir": Path("data"),
    "out_path": Path("heldout_results.pt"),
    "intervene_channel": None,       # int or None: channel 0..2 (R, G, B) to edit before TD
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

    lc = nn.Linear(len(POOLINGS) * CHANNELS * cfg["d_sae"], len(CLASSES)).to(device)
    lc.load_state_dict(ckpt["lc"])
    lc.eval()

    return sae, td, lc


# Leave-one-well-out: run each fold's models on that fold's held-out after cells, and export
# predictions, true labels, pooled latents, and TD reconstructions for every held-out cell
def run_inference():
    device = INFER_CFG["device"]
    first = torch.load(INFER_CFG["ckpt_dir"] / "fold1.pt", map_location=device, weights_only=False)
    te = load_frozen_dinov3_single_channel(first["cfg"]["dinov3_repo_dir"], first["cfg"]["dinov3_weights"], device)

    channel = INFER_CFG["intervene_channel"]
    latent = INFER_CFG["intervene_latent"]
    scale = INFER_CFG["intervene_scale"]
    do_intervene = channel is not None and latent is not None and scale is not None

    # One cell at a time: crops are not resized, so their sizes differ
    results = []
    with torch.no_grad():
        for fold in range(len(FOLD_COLUMNS)):
            ckpt = torch.load(INFER_CFG["ckpt_dir"] / f"fold{fold + 1}.pt", map_location=device, weights_only=False)
            sae, td, lc = load_fold_models(ckpt, te, device)
            d_sae = ckpt["cfg"]["d_sae"]
            train_singles, train_cells, heldout_cells = build_fold(INFER_CFG["data_dir"], fold)

            for path, label in heldout_cells:
                image = load_images([path]).to(device)              # (1, C, side, side)

                # Sparse latents with the threshold instead of BatchTopK
                tokens, hp, wp = encode_channels(te, image)
                b, c, t, d = tokens.shape
                z_tokens = sae.encode_threshold(tokens.reshape(-1, d))

                # Classify the cell
                z = pool_over_patches(z_tokens, b, c, t)
                logits = lc(z)
                probs = F.softmax(logits, dim=1)

                # Reconstruct the crop from its latents
                x_pred = sae.decode(z_tokens).reshape(b, c, t, d)
                img_recon = td(x_pred, hp, wp)

                # Optionally scale one latent in one channel and reconstruct again
                img_intervened = None
                if do_intervene:
                    z_edit = z_tokens.reshape(b, c, t, d_sae).clone()
                    z_edit[:, channel, :, latent] = z_edit[:, channel, :, latent] * scale
                    x_edit = sae.decode(z_edit.reshape(-1, d_sae)).reshape(b, c, t, d)
                    img_intervened = td(x_edit, hp, wp)

                experiment, phase, number, image_name = parse_crop_name(path.stem)
                entry = {
                    "path": str(path),
                    "fold": fold + 1,
                    "well": experiment + "_" + well_name(number),    # e.g. "exp1_DD_L0.1"
                    "label": label,                                   # true class: 1 ferroptotic (RD), 0 not
                    "logits": logits[0].cpu(),
                    "probs": probs[0].cpu(),
                    "z": z[0].cpu(),
                    "img_recon": img_recon[0].cpu(),
                }
                if img_intervened is not None:
                    entry["img_intervened"] = img_intervened[0].cpu()
                results.append(entry)

    torch.save({"classes": CLASSES, "results": results}, INFER_CFG["out_path"])


if __name__ == "__main__":
    run_inference()
