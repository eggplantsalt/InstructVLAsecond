import argparse
import csv
import json
from pathlib import Path


def short(x, nd=4):
    if x is None or x == "":
        return ""
    try:
        return str(round(float(x), nd))
    except Exception:
        return str(x)


def print_table(title, rows, cols):
    print("\n=== " + title + " ===")
    print("\t".join(cols))
    for r in rows:
        print("\t".join(short(r.get(c, "")) for c in cols))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_dir", required=True)
    args = ap.parse_args()

    root = Path(args.episode_dir)
    top = json.loads((root / "summary.json").read_text())
    clean = json.loads((root / "clean_baseline/summary.json").read_text())

    print("episode:", top.get("episode"))
    print("instruction:", top.get("instruction"))
    print("sample_strategy:", top.get("sample_strategy"))
    print("num_frames:", top.get("num_frames"))
    print("unnorm_key:", clean.get("unnorm_key"))
    print("clean_vs_gt_droid_scaled_l2_6d:", short(clean.get("clean_droid_scaled_vs_gt_mean_l2_6d")))
    print("clean_gt_gripper_mismatch_rate:", short(clean.get("clean_gt_gripper_mismatch_rate")))

    main_rows = read_csv(root / "main_conditions/condition_summary.csv")
    main_cols = [
        "condition",
        "mean_droid_scaled_l2_6d",
        "condition_droid_scaled_vs_gt_mean_l2_6d",
        "condition_minus_clean_vs_gt_mean_l2_6d",
        "mean_normalized_l2_6d",
        "gripper_change_rate",
        "condition_gt_gripper_mismatch_rate",
    ]
    print_table("main_conditions", main_rows, main_cols)

    extra_path = root / "extra_wrong_random/condition_summary.csv"
    if extra_path.exists():
        extra_rows = read_csv(extra_path)
        extra_cols = [
            "condition",
            "image_text",
            "mean_droid_scaled_l2_6d",
            "condition_droid_scaled_vs_gt_mean_l2_6d",
            "condition_minus_clean_vs_gt_mean_l2_6d",
            "mean_normalized_l2_6d",
            "gripper_change_rate",
            "condition_gt_gripper_mismatch_rate",
        ]
        print_table("extra_wrong_random", extra_rows, extra_cols)


if __name__ == "__main__":
    main()
