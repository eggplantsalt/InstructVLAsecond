import json
from pathlib import Path

import numpy as np
import pandas as pd


def short_value(x, max_len=120):
    s = repr(x)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def main():
    root = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA/datasets/droid_100_hf")

    info_json = root / "meta/info.json"
    tasks_path = root / "meta/tasks.parquet"
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
    data_path = root / "data/chunk-000/file-000.parquet"

    print("=== paths ===")
    for p in [info_json, tasks_path, episodes_path, data_path]:
        print(p, "exists=", p.exists(), "size=", p.stat().st_size if p.exists() else None)

    print("\n=== meta/info.json ===")
    info = json.loads(info_json.read_text())
    for k in sorted(info.keys()):
        v = info[k]
        if isinstance(v, (dict, list)):
            print(k, "=", type(v).__name__, "len=", len(v))
        else:
            print(k, "=", v)

    if "features" in info:
        print("\n--- info['features'] ---")
        for k, v in info["features"].items():
            print(k, ":", v)

    print("\n=== tasks.parquet ===")
    tasks = pd.read_parquet(tasks_path)
    print("shape:", tasks.shape)
    print("columns:", list(tasks.columns))
    print(tasks.head(10).to_string())

    print("\n=== episodes parquet ===")
    eps = pd.read_parquet(episodes_path)
    print("shape:", eps.shape)
    print("columns:", list(eps.columns))
    print(eps.head(10).to_string())

    print("\n=== data parquet ===")
    df = pd.read_parquet(data_path)
    print("shape:", df.shape)
    print("columns:")
    for c in df.columns:
        print("  -", c, "| dtype:", df[c].dtype)

    print("\n--- first row values ---")
    row0 = df.iloc[0]
    for c in df.columns:
        print(c, "=", short_value(row0[c]))

    print("\n=== episode statistics ===")
    if "episode_index" in df.columns:
        counts = df["episode_index"].value_counts().sort_index()
        print("num episodes in data:", len(counts))
        print("first 20 episode lengths:")
        print(counts.head(20).to_string())
    else:
        print("[WARN] no episode_index column in data")

    print("\n=== likely action fields ===")
    for c in df.columns:
        if "action" in c.lower():
            val = df[c].iloc[0]
            arr = np.asarray(val)
            print(c, "first value type=", type(val), "shape=", arr.shape, "value=", arr)

    print("\n=== likely language/task fields ===")
    for c in df.columns:
        lc = c.lower()
        if "language" in lc or "instruction" in lc or "task" in lc:
            print(c, "unique sample:", df[c].dropna().head(10).tolist())

    print("\n=== likely image/video fields ===")
    for c in df.columns:
        lc = c.lower()
        if "image" in lc or "video" in lc:
            print(c, "first value:", short_value(df[c].iloc[0]))


if __name__ == "__main__":
    main()
