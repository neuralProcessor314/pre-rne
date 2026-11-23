from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from cellpose import models

from train import TEST_CODE

# Paths, plate layout, intensity conversion, and segmentation hyperparameters
CFG = {
    "sep_dir": Path("data/sep"),
    "crop_dir": Path("data/crops"),
    "row_codes": None,          # list[str]: condition code of plate rows 1-4, e.g. ["RD", "DD", "RF", "DF"]
    "fluo_weights": None,       # (wR, wG, wB): BODIPY channel = wR*R + wG*G + wB*B
    "bf_white": None,           # float: brightfield transmittance that becomes 1.0
    "crop_size": None,          # int: square crop side length, multiple of 16
    "min_mask_pixels": None,    # int: minimum cell-group mask area
    "pad_px": None,             # int: bounding-box padding in pixels
    "flow_threshold": None,     # float: Cellpose flow error threshold
    "cellprob_threshold": None, # float: Cellpose cell probability threshold
    "batch_size": None,         # int: Cellpose evaluation batch size
    "gpu": True,
}

CODES = ["RD", "DD", "RF", "DF"]
CONCENTRATIONS = ["0.1", "0.5", "1.0"]

# Plate columns from left to right: (side, concentration)
COLUMNS = [("L", "0.1"), ("R", "0.1"), ("L", "0.5"), ("R", "0.5"), ("L", "1.0"), ("R", "1.0")]


# Turn a well number 1-24 into a well name such as "RD_L0.1"
# Wells are numbered along each row of 6, top row first
def well_name_from_number(number):
    row = (number - 1) // 6
    column = (number - 1) % 6
    code = CFG["row_codes"][row]
    side, conc = COLUMNS[column]
    return code + "_" + side + conc


# Read (phase, well name, FOV) from a file name, or return None if the name does not fit
# Phase: b = before brightfield, a = after brightfield, f = after BODIPY
# Examples: "bRD_L0.1_3" -> ("b", "RD_L0.1", "3"),  "aW7_3" -> ("a", name of well 7, "3")
def parse_name(name):
    phase = name[0]
    parts = name[1:].split("_")
    if phase not in ["b", "a", "f"]:
        return None

    # Well-number form: W<number>_<fov>
    if len(parts) == 2 and parts[0].startswith("W"):
        number_text = parts[0][1:]
        fov = parts[1]
        if CFG["row_codes"] is None:
            return None
        if not number_text.isdigit() or not fov.isdigit():
            return None
        number = int(number_text)
        if number < 1 or number > 24:
            return None
        return phase, well_name_from_number(number), str(int(fov))

    # Named form: <code>_<side><conc>_<fov>
    if len(parts) == 3:
        code = parts[0]
        side = parts[1][0]
        conc = parts[1][1:]
        fov = parts[2]
        if code not in CODES or side not in ["L", "R"] or conc not in CONCENTRATIONS:
            return None
        if not fov.isdigit():
            return None
        return phase, code + "_" + side + conc, str(int(fov))   # int() drops leading zeros

    return None


# Load the three grayscale PNGs of one RGB image as a (3, H, W) array scaled to [0, 1]
def load_rgb(name):
    channels = []
    for i in range(3):
        img = Image.open(CFG["sep_dir"] / f"{name}_c{i}.png")
        if img.mode == "L":
            max_value = 255.0       # 8-bit PNG
        elif img.mode == "I;16" or img.mode == "I":
            max_value = 65535.0     # 16-bit PNG
        else:
            raise ValueError(f"unexpected PNG mode {img.mode} in {name}_c{i}.png")
        channels.append(np.array(img, dtype=np.float32) / max_value)
    return np.stack(channels)


# Load a BODIPY image as one channel: a fixed weighted sum of R, G, B
def load_bodipy(name):
    r, g, b = load_rgb(name)
    w_r, w_g, w_b = CFG["fluo_weights"]
    return w_r * r + w_g * g + w_b * b


# Sort the separated images into before, after, and BODIPY tables keyed by (well, fov)
def index_images():
    before = {}
    after = {}
    bodipy = {}
    tables = {"b": before, "a": after, "f": bodipy}

    for c0_path in sorted(CFG["sep_dir"].glob("*_c0.png")):
        name = c0_path.name[: -len("_c0.png")]
        parsed = parse_name(name)
        if parsed is None:
            print("skipped:", name)
            continue
        phase, well, fov = parsed
        table = tables[phase]
        if (well, fov) in table:
            raise ValueError(f"two images for {well} FOV {fov}: {table[(well, fov)]} and {name}")
        table[(well, fov)] = name

    print(f"before={len(before)} after={len(after)} bodipy={len(bodipy)}")
    return before, after, bodipy


# Background level of each brightfield channel for one set of images (before or after)
# Backlight and RGB weights are the same within a set, so one value per channel is enough
def set_background(names):
    medians = []
    for name in names:
        rgb = load_rgb(name)
        medians.append(np.median(rgb, axis=(1, 2)))
    return np.median(np.array(medians), axis=0)


# Divide each brightfield channel by its background, so the result is transmittance
def normalise_brightfield(rgb, background):
    for i in range(3):
        rgb[i] = rgb[i] / background[i] / CFG["bf_white"]
    return np.clip(rgb, 0.0, 1.0)


# One BODIPY scale shared by all after images: the largest per-image 99.9th percentile
def bodipy_scale(names):
    scale = 0.0
    for name in names:
        scale = max(scale, float(np.percentile(load_bodipy(name), 99.9)))
    if scale == 0.0:
        scale = 1.0
    return scale


# Segment cells on the 3 brightfield channels and save each cell group as a resized 4-channel crop
def segment_and_save(model, stack, out_dir, well, fov):
    masks, flows, styles = model.eval(
        stack[:3],
        channel_axis=0,
        batch_size=CFG["batch_size"],
        flow_threshold=CFG["flow_threshold"],
        cellprob_threshold=CFG["cellprob_threshold"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    height = stack.shape[1]
    width = stack.shape[2]

    for cell_id in range(1, int(masks.max()) + 1):
        ys, xs = np.where(masks == cell_id)
        if len(ys) < CFG["min_mask_pixels"]:
            continue

        # Padded bounding box, kept inside the image
        y0 = max(0, ys.min() - CFG["pad_px"])
        y1 = min(height, ys.max() + 1 + CFG["pad_px"])
        x0 = max(0, xs.min() - CFG["pad_px"])
        x1 = min(width, xs.max() + 1 + CFG["pad_px"])

        crop = torch.from_numpy(stack[:, y0:y1, x0:x1]).unsqueeze(0)    # (1, 4, h, w)
        crop = F.interpolate(crop, size=(CFG["crop_size"], CFG["crop_size"]), mode="bilinear", align_corners=False)
        crop = crop.squeeze(0)                                           # (4, crop_size, crop_size)

        centroid = torch.tensor([float(ys.mean()), float(xs.mean())])
        out_path = out_dir / f"{well}_{fov}_cell{cell_id:04d}.pt"
        torch.save({"image": crop, "centroid": centroid}, out_path)


# Build 4-channel images (brightfield R, G, B + BODIPY) and save cell crops for every FOV
def segment_all():
    before, after, bodipy = index_images()
    model = models.CellposeModel(gpu=CFG["gpu"], pretrained_model="cpsam")

    # Before images: brightfield and an empty BODIPY channel (dye not added yet)
    if len(before) > 0:
        background = set_background(list(before.values()))
        for (well, fov), name in sorted(before.items()):
            rgb = normalise_brightfield(load_rgb(name), background)
            empty = np.zeros_like(rgb[0])
            stack = np.stack([rgb[0], rgb[1], rgb[2], empty])
            code = well.split("_")[0]
            segment_and_save(model, stack, CFG["crop_dir"] / "before" / code, well, fov)

    # After images: brightfield and the BODIPY image of the same well and FOV
    pairs = []
    for (well, fov), name in sorted(after.items()):
        if (well, fov) in bodipy:
            pairs.append((well, fov, name, bodipy[(well, fov)]))
        else:
            print("after image without BODIPY, skipped:", name)

    if len(pairs) > 0:
        background = set_background(list(after.values()))
        bodipy_names = []
        for well, fov, name, fluo_name in pairs:
            bodipy_names.append(fluo_name)
        scale = bodipy_scale(bodipy_names)

        for well, fov, name, fluo_name in pairs:
            rgb = normalise_brightfield(load_rgb(name), background)
            fluo = np.clip(load_bodipy(fluo_name) / scale, 0.0, 1.0)
            if fluo.shape != rgb[0].shape:
                raise ValueError(f"brightfield and BODIPY sizes differ for {well} FOV {fov}")
            stack = np.stack([rgb[0], rgb[1], rgb[2], fluo])

            code = well.split("_")[0]
            if code == TEST_CODE:
                group = "test"
            else:
                group = "control"
            segment_and_save(model, stack, CFG["crop_dir"] / group / code, well, fov)


if __name__ == "__main__":
    segment_all()
