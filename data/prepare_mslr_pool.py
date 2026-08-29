"""Pool MSLR-WEB30K Fold<N> into one dense npz (no train/vali/test split).

Reads Fold<N>/{train.txt, vali.txt, test.txt} from MSLR-WEB30K.zip and concatenates every query-document record into a single npz that the contextual-semibandit stream consumes directly.

Inputs:
    data/MSLR-WEB30K.zip    (the official Microsoft download, with Fold1..Fold5 inside)

Output (data/mslr/):
    mslr30k_pool.npz   keys: features, relevances, docsPerQuery, queryOrder

Usage:
    python data/prepare_mslr_pool.py                # Fold1, drop txt after
    python data/prepare_mslr_pool.py --fold 1
    python data/prepare_mslr_pool.py --keep-txt     # retain extracted .txt
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
DST = DATA / "mslr"
ZIP_DEFAULT = DATA / "MSLR-WEB30K.zip"

NUM_FEATURES = 136
MAX_DOCS_PER_QUERY = 50  # matches prepare_mslr.py / PreloadMSLR.py

FOLD_MEMBERS = ("train.txt", "vali.txt", "test.txt")


def extract_fold(zip_path: Path, fold: int, out_dir: Path) -> list[Path]:
    """Extract every Fold<N>/{train,vali,test}.txt from the zip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in FOLD_MEMBERS:
            member = f"Fold{fold}/{name}"
            local = out_dir / name
            print(f"  extracting {member} -> {local}", flush=True)
            with zf.open(member) as src, open(local, "wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            out_paths.append(local)
    return out_paths


def parse_libsvm_pool(txt_paths: list[Path], out_npz: Path) -> int:
    """Parse multiple libSVM files into one pooled npz."""
    print(f"== pass 1: count queries and docs across {len(txt_paths)} files ==",
          flush=True)
    query_to_idx: dict[int, int] = {}
    docs_per_query: dict[int, int] = {}
    order: list[int] = []
    for p in txt_paths:
        print(f"  scanning {p.name}", flush=True)
        with open(p, "r") as f:
            for line in f:
                tok = line.split()
                if not tok:
                    continue
                qid = int(tok[1].split(":", 1)[-1])
                qi = query_to_idx.get(qid)
                if qi is None:
                    qi = len(query_to_idx)
                    query_to_idx[qid] = qi
                    docs_per_query[qi] = 0
                order.append(qi)
                docs_per_query[qi] += 1

    n_queries = len(query_to_idx)
    max_docs = min(max(docs_per_query.values()), MAX_DOCS_PER_QUERY)
    print(f"  pooled {n_queries:,} queries, max docs/query={max_docs}",
          flush=True)

    docsPerQuery = np.zeros(n_queries, dtype=np.int32)
    for qi, n in docs_per_query.items():
        docsPerQuery[qi] = min(n, MAX_DOCS_PER_QUERY)
    relevances = -np.ones((n_queries, max_docs), dtype=np.int32)
    features = np.full((n_queries, max_docs, NUM_FEATURES), np.nan,
                       dtype=np.float32)

    print("== pass 2: fill dense arrays ==", flush=True)
    processed = np.zeros(n_queries, dtype=np.int32)
    for p in txt_paths:
        print(f"  parsing {p.name}", flush=True)
        with open(p, "r") as f:
            for line in f:
                tok = line.split()
                if not tok:
                    continue
                rel = int(tok[0])
                qid = int(tok[1].split(":", 1)[-1])
                qi = query_to_idx[qid]
                doc_idx = int(processed[qi])
                if doc_idx >= MAX_DOCS_PER_QUERY:
                    continue
                fvec = np.fromiter(
                    (float(x.split(":", 1)[-1]) for x in tok[2:]),
                    dtype=np.float32, count=NUM_FEATURES,
                )
                relevances[qi, doc_idx] = rel
                features[qi, doc_idx, :] = fvec
                processed[qi] += 1

    np.savez_compressed(
        out_npz,
        relevances=relevances,
        features=features,
        docsPerQuery=docsPerQuery,
        queryOrder=np.asarray(order, dtype=np.int32),
    )
    print(f"  saved {out_npz.name} "
          f"({out_npz.stat().st_size / 1e6:.1f} MB)", flush=True)
    return n_queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--zip", default=str(ZIP_DEFAULT),
                        help="path to MSLR-WEB30K.zip "
                             "(default: data/MSLR-WEB30K.zip)")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--keep-txt", action="store_true",
                        help="retain extracted .txt files after parsing")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        sys.exit(f"zip not found: {zip_path}")
    DST.mkdir(parents=True, exist_ok=True)

    txt_dir = DST / f"_fold{args.fold}_extracted"
    print(f"== extract Fold{args.fold} from {zip_path.name} ==", flush=True)
    txt_paths = extract_fold(zip_path, args.fold, txt_dir)

    parse_libsvm_pool(txt_paths, DST / "mslr30k_pool.npz")

    if not args.keep_txt:
        for p in txt_paths:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        try:
            txt_dir.rmdir()
        except OSError:
            pass

    print("done.", flush=True)


if __name__ == "__main__":
    main()
