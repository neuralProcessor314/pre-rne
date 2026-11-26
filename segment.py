from pathlib import Path
import numpy as np
from PIL import Image
import torch
from cellpose import models

from train import TEST_CODE

# Paths, plate layout, intensity conversion, and segmentation hyperparameters
CFG = {
    "sep_dir": Path("data/sep"),
    "crop_dir": Path("data/crops"),
    "row_codes": None,          # list[str]: condition code of plate rows 1-4, e.g. ["RD", "DD", "RF", "DF"]
    "fluo_weights": None,       # (wR, wG, wB): BODIPY channel = wR*R + wG*G + wB*B
    "bf_mean": 0.5,             # float: mean of each brightfield channel after normalisation (0.5 = 128 of 255)
    "bf_std": None,             # float: standard deviation (contrast) of each brightfield channel after normalisation
    "min_mask_pixels": None,    # int: minimum cell-group mask area
    "pad_px": None,             # int: bounding-box padding in pixels
    "flow_threshold": None,     # float: Cellpose flow error threshold
    "cellprob_threshold": None, # float: Cellpose cell probability threshold
    "batch_size": None,         # int: Cellpose evaluation batch size
    "gpu": True,
}

PATCH_SIZE = 16     # DINOv3 ViT-B/16 patch side; crop sides must be a multiple of it

CODES = ["RD", "DD", "RF", "DF"]
CONCENTRATIONS = ["0.1", "0.5"]     # 1.0 dropped: every cell was already dead

# Plate columns from left to right: (side, concentration)
# The 1.0 columns (wells 5, 6, 11, 12, 17, 18, 23, 24) stay in the list so well numbers keep their plate positions
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
        side, conc = COLUMNS[(number - 1) % 6]
        if conc not in CONCENTRATIONS:
            return None        # well of a dropped concentration
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


# Normalise one brightfield image: each of R, G, B gets mean bf_mean and standard deviation bf_std,
# so mean colour and contrast are the same in every image
def normalise_brightfield(rgb):
    for i in range(3):
        mean = rgb[i].mean()
        std = rgb[i].std()
        if std == 0:
            std = 1.0          # a perfectly flat channel is only shifted
        rgb[i] = (rgb[i] - mean) / std * CFG["bf_std"] + CFG["bf_mean"]
    return np.clip(rgb, 0.0, 1.0)


# One BODIPY scale shared by all after images: the largest per-image 99.9th percentile
def bodipy_scale(names):
    scale = 0.0
    for name in names:
        scale = max(scale, float(np.percentile(load_bodipy(name), 99.9)))
    if scale == 0.0:
        scale = 1.0
    return scale


# Turn a padded bounding box into a square window of real image pixels (no resizing)
# The side is the box's longer side rounded up to a multiple of PATCH_SIZE; the window is
# centred on the box and moved back inside the image if it sticks out.
# Returns (top, bottom, left, right), or None if the square is larger than the image.
def square_window(y0, y1, x0, x1, height, width):
    side = max(y1 - y0, x1 - x0)
    if side % PATCH_SIZE != 0:
        side = side + PATCH_SIZE - side % PATCH_SIZE
    if side > height or side > width:
        return None

    top = (y0 + y1) // 2 - side // 2
    left = (x0 + x1) // 2 - side // 2
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    if top + side > height:
        top = height - side
    if left + side > width:
        left = width - side
    return top, top + side, left, left + side


# Segment cells on the 3 brightfield channels and save each cell group as a square 4-channel crop
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

        # Padded bounding box, then squared with the surrounding pixels
        y0 = int(ys.min()) - CFG["pad_px"]
        y1 = int(ys.max()) + 1 + CFG["pad_px"]
        x0 = int(xs.min()) - CFG["pad_px"]
        x1 = int(xs.max()) + 1 + CFG["pad_px"]
        window = square_window(y0, y1, x0, x1, height, width)
        if window is None:
            print(f"cell larger than the image, skipped: {well} FOV {fov} cell {cell_id}")
            continue
        top, bottom, left, right = window

        # copy() so the saved file holds only the crop, not the whole image behind it
        crop = torch.from_numpy(stack[:, top:bottom, left:right].copy())   # (4, side, side)

        centroid = torch.tensor([float(ys.mean()), float(xs.mean())])
        area = torch.tensor(len(ys))                                          # mask pixel count
        out_path = out_dir / f"{well}_{fov}_cell{cell_id:04d}.pt"
        torch.save({"image": crop, "centroid": centroid, "area": area}, out_path)


# Build 4-channel images (brightfield R, G, B + BODIPY) and save cell crops for every FOV
def segment_all():
    before, after, bodipy = index_images()
    model = models.CellposeModel(gpu=CFG["gpu"], pretrained_model="cpsam")

    # Before images: brightfield and an empty BODIPY channel (dye not added yet)
    if len(before) > 0:
        for (well, fov), name in sorted(before.items()):
            rgb = normalise_brightfield(load_rgb(name))
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
        bodipy_names = []
        for well, fov, name, fluo_name in pairs:
            bodipy_names.append(fluo_name)
        scale = bodipy_scale(bodipy_names)

        for well, fov, name, fluo_name in pairs:
            rgb = normalise_brightfield(load_rgb(name))
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
