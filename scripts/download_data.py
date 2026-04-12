"""Download and prepare the Kenney 1-Bit Pack for training.

Source: https://kenney.nl/assets/1-bit-pack (CC0 license)
Slices tilesheets into individual 16x16 sprites.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
SPRITES_DIR = DATA_DIR / "sprites"
ZIP_URL = "https://kenney.nl/media/pages/assets/1-bit-pack/f41b6925f0-1677578516/kenney_1-bit-pack.zip"

TILE_SIZE = 16

# Use monochrome-transparent packed (clean, no spacing, transparent bg)
TILESHEET_NAME = "monochrome-transparent_packed.png"


def download_and_extract(tmp_dir: Path) -> Path:
    """Download the zip and extract to a temp directory."""
    zip_path = tmp_dir / "kenney.zip"
    logger.info("Downloading Kenney 1-Bit Pack...")
    subprocess.run(["curl", "-L", "-o", str(zip_path), ZIP_URL], check=True)
    logger.info("Extracting...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(tmp_dir)], check=True)
    return tmp_dir


def find_tilesheet(extract_dir: Path) -> Path:
    """Find the target tilesheet in the extracted files."""
    matches = list(extract_dir.rglob(TILESHEET_NAME))
    if not matches:
        available = [p.name for p in extract_dir.rglob("*.png")]
        logger.error("Tilesheet %s not found. Available: %s", TILESHEET_NAME, available)
        sys.exit(1)
    return matches[0]


def slice_tilesheet(tilesheet_path: Path, output_dir: Path) -> int:
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

            # Skip fully transparent tiles
            if tile.getextrema()[3][1] == 0:
                continue

            tile.save(output_dir / f"sprite_{row:03d}_{col:03d}.png")
            count += 1

    return count


def main() -> None:
    if SPRITES_DIR.exists() and any(SPRITES_DIR.glob("sprite_*.png")):
        count = len(list(SPRITES_DIR.glob("sprite_*.png")))
        logger.info("Sprites already exist (%d files) — skipping download.", count)
        logger.info("To re-download: rm -rf %s", SPRITES_DIR)
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        download_and_extract(tmp_dir)
        tilesheet = find_tilesheet(tmp_dir)
        logger.info("Found tilesheet: %s (%s)", tilesheet.name, Image.open(tilesheet).size)

        # Clean slate
        if SPRITES_DIR.exists():
            import shutil
            shutil.rmtree(SPRITES_DIR)

        count = slice_tilesheet(tilesheet, SPRITES_DIR)
        logger.info("Done. %d sprites extracted to %s", count, SPRITES_DIR)


if __name__ == "__main__":
    main()
