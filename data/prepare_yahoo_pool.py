"""Pool Yahoo! LTRC set1 into one dense npz (no train/vali/test split).

Concatenates set1.{train, valid, test}.txt into a single npz with a shared dense feature index built from the union of feature ids seen in all three files.

set2 is intentionally excluded (different feature schema; standard LTR-bandit convention, e.g. Bietti et al. 2021, TF Datasets `yahoo_ltrc`).

Inputs (must already exist):
    data/yahoo/raw/set1.train.txt
    data/yahoo/raw/set1.valid.txt
    data/yahoo/raw/set1.test.txt

Output (data/yahoo/):
    yahoo_pool.npz   keys: features, relevances, docsPerQuery, queryOrder

Usage:
    python data/prepare_yahoo_pool.py                # cap 50 docs/query
    python data/prepare_yahoo_pool.py --max-docs 30
"""

from __future__ import annotations

import argparse
import sys
import numpy as np

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
DST = DATA / "yahoo"
RAW = DST / "raw"

DEFAULT_MAX_DOCS_PER_QUERY = 50

SET1_FILES = ("set1.train.txt", "set1.valid.txt", "set1.test.txt")


def collect_feature_index(paths: list[Path]) -> dict[int, int]:
    """Build a dense {libsvm_id -> col} map covering every feature id
    seen in any of `paths`."""
    seen: set[int] = set()
    for p in paths:
        print(f"  scanning {p.name} for feature ids", flush=True)
        with open(p, "r") as f:
            for line in f:
                tok = line.split()
                if not tok:
                    continue
                for kv in tok[2:]:
                    seen.add(int(kv.split(":", 1)[0]))
    fids = sorted(seen)
    return {fid: i for i, fid in enumerate(fids)}


def parse_libsvm_pool(
    txt_paths: list[Path], out_npz: Path, feat_map: dict[int, int],
    max_docs: int,
) -> int:
    """Parse multiple libSVM files into one pooled npz."""
    n_features = len(feat_map)

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
    actual_max = max(docs_per_query.values())
    stored_max = min(actual_max, max_docs)
    print(f"  pooled {n_queries:,} queries, max docs/query (raw)={actual_max}, "
          f"stored cap={stored_max}", flush=True)

    docsPerQuery = np.zeros(n_queries, dtype=np.int32)
    for qi, n in docs_per_query.items():
        docsPerQuery[qi] = min(n, max_docs)
    relevances = -np.ones((n_queries, stored_max), dtype=np.int32)
    features = np.full((n_queries, stored_max, n_features), np.nan,
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
                if doc_idx >= max_docs:
                    continue
                row = np.zeros(n_features, dtype=np.float32)
                for kv in tok[2:]:
                    fid_s, val_s = kv.split(":", 1)
                    col = feat_map.get(int(fid_s))
                    if col is not None:
                        row[col] = float(val_s)
                relevances[qi, doc_idx] = rel
                features[qi, doc_idx, :] = row
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
    parser.add_argument("--max-docs", type=int,
                        default=DEFAULT_MAX_DOCS_PER_QUERY,
                        help="cap on docs/query stored in the dense tensor "
                             "(default 50; raises memory cost if higher)")
    args = parser.parse_args()

    paths = [RAW / name for name in SET1_FILES]
    for p in paths:
        if not p.exists():
            sys.exit(f"missing input: {p}")

    DST.mkdir(parents=True, exist_ok=True)

    print("== build shared feature index from set1 (train + valid + test) ==",
          flush=True)
    feat_map = collect_feature_index(paths)
    print(f"  d = {len(feat_map)} dense features "
          f"(min libsvm id={min(feat_map)}, max={max(feat_map)})", flush=True)

    n = parse_libsvm_pool(paths, DST / "yahoo_pool.npz", feat_map,
                          args.max_docs)
    print("done.", flush=True)
    print(f"  pooled queries: {n:,}  -> yahoo_pool.npz")
    print(f"  d             : {len(feat_map)} features")


if __name__ == "__main__":
    main()
