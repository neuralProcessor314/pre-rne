from pathlib import Path
import csv
import math
import shutil
import zlib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from cellpose import models

from plate import CODES, COLUMNS, CONCENTRATIONS, well_number, well_position

# Data steps, all under data_dir:
#   0_<phase>_raw/<experiment folder>/<files>   raw microscope images (not changed)
#   1_<phase>_rename/                           renamed copies: exp<n>_<phase>W<well>_<index>.<ext>
#   2_<phase>_prepaug/                          normalised images and their augmented copies (PNG)
#   3_<phase>_crops/                            one .pt file per cell
# <phase> is "before" or "after". Running this script redoes all three steps; the 1_, 2_ and 3_ folders
# are deleted and written again each time.
# Cellpose and crop settings: sources in DECISIONS.md, Hyperparameters
CFG = {
    "data_dir": Path("data"),
    "bf_mean": 0.5,             # float: mean of each R, G, B channel after normalisation (0.5 = 128 of 255)
    "bf_std": 0.25,             # float: standard deviation (contrast) of each channel after normalisation
    "min_mask_pixels": 15,      # int: minimum cell mask area
    "pad_px": 16,               # int: bounding-box padding in pixels
    "flow_threshold": 0.4,      # float: Cellpose flow error threshold
    "cellprob_threshold": 0.0,  # float: Cellpose cell probability threshold
    "batch_size": 8,            # int: Cellpose evaluation batch size
    "gpu": True,
    # Augmentation, applied to whole images after normalisation and before Cellpose.
    # Each image is also saved as aug_copies augmented copies; strengths are drawn from 0 up to these maxima.
    "seed": 42,                     # int: seed for the augmentation draws
    "aug_copies": 3,                # int: augmented copies per image, besides the original
    "aug_ca_max_shift_px": 4,       # int: chromatic aberration, max shift of R and B against G, in pixels
    "aug_illum_max_slope": 0.5,     # float: uneven illumination, max brightness change from image centre to edge (0.1 = 10 %)
    "aug_blur_max_sigma": 1.5,      # float: Gaussian blur, max sigma in pixels
    "aug_noise_max_std": 0.06,      # float: Gaussian noise, max standard deviation (intensities are 0-1)
}

PHASES = ["before", "after"]
PATCH_SIZE = 16     # DINOv3 ViT-B/16 patch side; crop sides must be a multiple of it

# Magnifications kept. All kept images are 20x: exp3's files are labelled "40x" but were taken at 20x.
# 4x and 10x images are dropped.
KEEP_MAGNIFICATIONS = ["20x", "40x"]


# ---------------- Step 1: rename ----------------

# Read (well number, magnification) from a raw file name, or return None if the name does not fit.
# Names in use:  exp1 "DD_0.1L_20x_39",  exp2 "bW13_20x_27",  exp3 "well13_40x_00"
def parse_raw_name(stem):
    parts = stem.split("_")

    # <code>_<conc><side>_<magnification>_<number>
    if len(parts) == 4:
        code = parts[0]
        conc = parts[1][:-1]
        side = parts[1][-1]
        magnification = parts[2]
        if code not in CODES or (side, conc) not in COLUMNS:
            return None
        return well_number(code, side, conc), magnification

    # bW<well>_<magnification>_<number>  or  well<well>_<magnification>_<number>
    if len(parts) == 3:
        head = parts[0]
        magnification = parts[1]
        for prefix in ["bW", "well"]:
            if head.startswith(prefix):
                number_text = head[len(prefix):]
                if number_text.isdigit() and 1 <= int(number_text) <= 24:
                    return int(number_text), magnification
        return None

    return None


# Copy every raw image to 1_<phase>_rename/ as exp<n>_<phase>W<well>_<index>.<ext>.
# The index is one counter over all kept images (both phases), in alphabetical order of their raw paths.
# Writes 1_rename_map.csv with each new name and its raw path.
def rename_raw():
    data = CFG["data_dir"]
    entries = []
    for phase in PHASES:
        out_dir = data / f"1_{phase}_rename"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for path in (data / f"0_{phase}_raw").glob("*/*"):
            if path.is_file():
                entries.append((path.relative_to(data).as_posix(), phase, path))
    entries.sort()

    index = 0
    rows = [["new_name", "raw_path"]]
    for relative, phase, path in entries:
        parsed = parse_raw_name(path.stem)
        if parsed is None:
            print("name not understood, skipped:", relative)
            continue
        number, magnification = parsed
        if magnification not in KEEP_MAGNIFICATIONS:
            print(f"{magnification} image dropped:", relative)
            continue
        experiment = path.parent.name.split("_")[0]          # "exp1_after_20261106" -> "exp1"
        new_name = f"{experiment}_{phase}W{number}_{index:03d}{path.suffix.lower()}"
        shutil.copy2(path, data / f"1_{phase}_rename" / new_name)
        rows.append([new_name, relative])
        index += 1

    with open(data / "1_rename_map.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"renamed {index} images")


# ---------------- Step 2: normalise and augment ----------------

# Read a renamed or derived file name, e.g. "exp1_afterW13_012-aug1" -> ("exp1", "after", 13, "012-aug1")
def parse_image_name(stem):
    experiment, phase_well, image = stem.split("_")
    phase, number = phase_well.split("W")
    return experiment, phase, int(number), image


# Load an image as a float (3, H, W) array of R, G, B scaled to [0, 1]
def load_rgb(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return rgb.transpose(2, 0, 1).copy()


# Save a (3, H, W) array in [0, 1] as an 8-bit RGB PNG
def save_rgb(rgb, path):
    pixels = (np.clip(rgb, 0.0, 1.0).transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


# Normalise one image: each of R, G, B gets mean bf_mean and standard deviation bf_std,
# so mean colour and contrast are the same in every image
def normalise(rgb):
    for i in range(3):
        mean = rgb[i].mean()
        std = rgb[i].std()
        if std == 0:
            std = 1.0          # a perfectly flat channel is only shifted
        rgb[i] = (rgb[i] - mean) / std * CFG["bf_std"] + CFG["bf_mean"]
    return np.clip(rgb, 0.0, 1.0)


# Random generator for one augmented copy of one image, fixed by seed, image name and copy number
def copy_generator(name, copy):
    key = f"{CFG['seed']}|{name}|{copy}"
    return np.random.default_rng(zlib.crc32(key.encode()))


# Shift one channel (H, W) by (dy, dx) pixels; the uncovered edge repeats the nearest edge pixels
def shift_channel(channel, dy, dx):
    m = max(abs(dy), abs(dx))
    if m == 0:
        return channel
    h, w = channel.shape
    padded = np.pad(channel, m, mode="edge")
    return padded[m - dy:m - dy + h, m - dx:m - dx + w]


# One augmented copy of a normalised (3, H, W) image
def augment(rgb, rng):
    # Every random choice, drawn first
    turns = int(rng.integers(0, 4))
    flip = rng.random() < 0.5
    m = CFG["aug_ca_max_shift_px"]
    shift_r = (int(rng.integers(-m, m + 1)), int(rng.integers(-m, m + 1)))
    shift_b = (int(rng.integers(-m, m + 1)), int(rng.integers(-m, m + 1)))
    angle = rng.random() * 2 * math.pi
    slope = rng.random() * CFG["aug_illum_max_slope"]
    sigma = rng.random() * CFG["aug_blur_max_sigma"]
    noise_std = rng.random() * CFG["aug_noise_max_std"]

    # Rotation by 0, 90, 180 or 270 degrees, then a mirror flip (exact, no new pixels)
    out = np.rot90(rgb, turns, axes=(1, 2))
    if flip:
        out = out[:, :, ::-1]
    out = out.copy()

    # Chromatic aberration: R and B shifted a few pixels against G
    out[0] = shift_channel(out[0], shift_r[0], shift_r[1])
    out[2] = shift_channel(out[2], shift_b[0], shift_b[1])

    # Uneven illumination: multiplied by a brightness ramp in a random direction
    height = out.shape[1]
    width = out.shape[2]
    ys = np.linspace(-1.0, 1.0, height)[:, None]
    xs = np.linspace(-1.0, 1.0, width)[None, :]
    ramp = 1.0 + slope * (xs * math.cos(angle) + ys * math.sin(angle))
    out = out * ramp

    # Out-of-focus blur
    if sigma > 0:
        for c in range(3):
            out[c] = gaussian_filter(out[c], sigma, mode="reflect", truncate=3.0)

    # Camera noise
    out = out + rng.standard_normal(out.shape) * noise_std

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Normalise every renamed image of a kept concentration and save it to 2_<phase>_prepaug/
# as <name>.png, plus aug_copies augmented copies <name>-aug<copy>.png
def preprocess_and_augment():
    data = CFG["data_dir"]
    for phase in PHASES:
        out_dir = data / f"2_{phase}_prepaug"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        for path in sorted((data / f"1_{phase}_rename").glob("*")):
            experiment, file_phase, number, image = parse_image_name(path.stem)
            code, side, conc = well_position(number)
            if conc not in CONCENTRATIONS:
                continue                                  # well of a dropped concentration

            rgb = normalise(load_rgb(path))
            save_rgb(rgb, out_dir / f"{path.stem}.png")
            for copy in range(1, CFG["aug_copies"] + 1):
                augmented = augment(rgb, copy_generator(path.stem, copy))
                save_rgb(augmented, out_dir / f"{path.stem}-aug{copy}.png")
        print(f"{phase}: {len(list(out_dir.glob('*.png')))} images in {out_dir}")


# ---------------- Step 3: segment and crop ----------------

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


# Segment cells in one (3, H, W) image and save each cell as a square crop <name>_cell<id>.pt
def segment_and_save(model, rgb, out_dir, name):
    masks, flows, styles = model.eval(
        rgb,
        channel_axis=0,
        batch_size=CFG["batch_size"],
        flow_threshold=CFG["flow_threshold"],
        cellprob_threshold=CFG["cellprob_threshold"],
    )
    height = rgb.shape[1]
    width = rgb.shape[2]

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
            print(f"cell larger than the image, skipped: {name} cell {cell_id}")
            continue
        top, bottom, left, right = window

        # copy() so the saved file holds only the crop, not the whole image behind it
        crop = torch.from_numpy(rgb[:, top:bottom, left:right].copy())   # (3, side, side)

        centroid = torch.tensor([float(ys.mean()), float(xs.mean())])
        area = torch.tensor(len(ys))                                        # mask pixel count
        torch.save({"image": crop, "centroid": centroid, "area": area}, out_dir / f"{name}_cell{cell_id:04d}.pt")


# Run Cellpose on every image in 2_<phase>_prepaug/ and save per-cell crops to 3_<phase>_crops/
def segment_all():
    data = CFG["data_dir"]
    model = models.CellposeModel(gpu=CFG["gpu"], pretrained_model="cpsam")
    for phase in PHASES:
        out_dir = data / f"3_{phase}_crops"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for path in sorted((data / f"2_{phase}_prepaug").glob("*.png")):
            segment_and_save(model, load_rgb(path), out_dir, path.stem)
        print(f"{phase}: {len(list(out_dir.glob('*.pt')))} cell crops in {out_dir}")


if __name__ == "__main__":
    rename_raw()
    preprocess_and_augment()
    segment_all()
