"""Download and unpack the Yahoo! Learning-to-Rank Challenge dataset (C14B).

Final layout (matches the convention used by downstream code):

    data/yahoo/
    ├── set1.train.txt
    ├── set1.valid.txt
    ├── set1.test.txt
    ├── set2.train.txt
    ├── set2.valid.txt
    └── set2.test.txt

The original Hugging Face download (a nested ``.tgz`` containing a ``.tar.bz2``) is cached under ``data/yahoo/raw/`` so re-runs are no-ops.

Prerequisites:
    pip install -U huggingface_hub
    huggingface-cli login
    # Accept terms at:
    #   https://huggingface.co/datasets/YahooResearch/Yahoo-Learning-to-Rank-Challenge
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "YahooResearch/Yahoo-Learning-to-Rank-Challenge"
DATA_DIR = Path(__file__).resolve().parent / "yahoo"
RAW_DIR = DATA_DIR / "raw"

SETS = ("set1", "set2")
SPLITS = ("train", "valid", "test")
EXPECTED_FILES = [f"{s}.{sp}.txt" for s in SETS for sp in SPLITS]


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
        max_workers=4,
    )
    print(f"[download] raw files in: {path}")
    return Path(path)


def extract(raw_root: Path) -> None:
    """Unpack the nested tarballs and place six ``set*.{train,valid,test}.txt``
    files directly under ``data/yahoo/``."""
    if all((DATA_DIR / name).exists() for name in EXPECTED_FILES):
        print("[extract] all six split files already present, skipping")
        return

    outer = next(raw_root.glob("*.tgz"), None) or next(raw_root.glob("*.tar.gz"), None)
    if outer is None:
        raise FileNotFoundError(f"No outer .tgz under {raw_root}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        print(f"[extract] outer: {outer.name}")
        with tarfile.open(outer, "r:gz") as tf:
            tf.extractall(tmp_path)

        inner = next(tmp_path.rglob("ltrc_yahoo.tar.bz2"), None)
        if inner is None:
            raise FileNotFoundError("ltrc_yahoo.tar.bz2 not found after first extraction")

        print(f"[extract] inner: {inner.name}")
        with tarfile.open(inner, "r:bz2") as tf:
            tf.extractall(tmp_path)

        for name in EXPECTED_FILES:
            src = next(tmp_path.rglob(name), None)
            if src is None:
                raise FileNotFoundError(f"{name} not found after extraction")
            dst = DATA_DIR / name
            shutil.move(str(src), dst)
            print(f"[extract]   -> {dst.relative_to(DATA_DIR.parent)}")


def main() -> None:
    raw_root = download()
    extract(raw_root)
    print(f"[done] dataset ready in: {DATA_DIR}")


if __name__ == "__main__":
    main()
