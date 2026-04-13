"""Download and prepare the Kenney 1-Bit Pack for training.

Source: https://kenney.nl/assets/1-bit-pack (CC0 license)
Slices tilesheets into individual 16x16 sprites.
Supports both monochrome and colored variants.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ZIP_URL = "https://kenney.nl/media/pages/assets/1-bit-pack/f41b6925f0-1677578516/kenney_1-bit-pack.zip"

TILE_SIZE = 16

TILESHEETS = {
    "mono": "monochrome-transparent_packed.png",
    "colored": "colored-transparent_packed.png",
}


def download_and_extract(tmp_dir: Path) -> Path:
    """Download the zip and extract to a temp directory."""
    zip_path = tmp_dir / "kenney.zip"
    logger.info("Downloading Kenney 1-Bit Pack...")
    subprocess.run(["curl", "-L", "-o", str(zip_path), ZIP_URL], check=True)
    logger.info("Extracting...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(tmp_dir)], check=True)
    return tmp_dir


def slice_tilesheet(tilesheet_path: Path, output_dir: Path, composite_bg: tuple[int, ...] = (0, 0, 0)) -> int:
    """Slice a packed tilesheet into individual 16x16 PNGs."""
    img = Image.open(tilesheet_path).convert("RGBA")
    w, h = img.size
    cols = w // TILE_SIZE
    rows = h // TILE_SIZE

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for row in range(rows):
        for col in range(cols):
            x = col * TILE_SIZE
            y = row * TILE_SIZE
            tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))

            if tile.getextrema()[3][1] == 0:
                continue

            # Composite onto background
            bg = Image.new("RGB", tile.size, composite_bg)
            bg.paste(tile, mask=tile.split()[3])
            bg.save(output_dir / f"sprite_{row:03d}_{col:03d}.png")
            count += 1

    return count


def filter_focused(src_dir: Path, dst_dir: Path, min_density: float = 0.15, max_density: float = 0.50) -> int:
    """Filter sprites by pixel density to get a focused subset."""
    import shutil

    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in dst_dir.glob("*.png"):
        f.unlink()

    count = 0
    for p in sorted(src_dir.glob("sprite_*.png")):
        img = Image.open(p).convert("RGB")
        arr = np.array(img)
        non_black = (arr.sum(axis=2) > 10).sum()
        density = non_black / (TILE_SIZE * TILE_SIZE)
        if min_density <= density <= max_density:
            shutil.copy(p, dst_dir / p.name)
            count += 1

    return count


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Download and prepare Kenney sprite data")
    parser.add_argument("--variant", choices=["mono", "colored", "both"], default="both")
    parser.add_argument("--focused", action="store_true", default=True, help="Also create focused subset")
    args = parser.parse_args()

    variants = ["mono", "colored"] if args.variant == "both" else [args.variant]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        download_and_extract(tmp_dir)

        for variant in variants:
            sheet_name = TILESHEETS[variant]
            matches = list(tmp_dir.rglob(sheet_name))
            if not matches:
                logger.error("%s not found", sheet_name)
                continue

            sprites_dir = DATA_DIR / f"sprites{'_colored' if variant == 'colored' else ''}"
            if sprites_dir.exists() and any(sprites_dir.glob("sprite_*.png")):
                logger.info("%s already exists (%d sprites) — skipping", sprites_dir, len(list(sprites_dir.glob("sprite_*.png"))))
            else:
                count = slice_tilesheet(matches[0], sprites_dir)
                logger.info("[%s] %d sprites → %s", variant, count, sprites_dir)

            if args.focused:
                focused_dir = DATA_DIR / f"sprites{'_colored' if variant == 'colored' else ''}_focused"
                if focused_dir.exists() and any(focused_dir.glob("sprite_*.png")):
                    logger.info("%s already exists — skipping", focused_dir)
                else:
                    focused_count = filter_focused(sprites_dir, focused_dir)
                    logger.info("[%s] %d focused sprites → %s", variant, focused_count, focused_dir)


if __name__ == "__main__":
    main()
