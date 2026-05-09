import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def task_index_to_text(tasks_df, task_index: int) -> str:
    """
    In this LeRobot parquet, the text task is stored as the DataFrame index,
    and the numeric task id is stored in column 'task_index'.
    """
    matches = tasks_df[tasks_df["task_index"] == task_index]
    if len(matches) == 0:
        return f"UNKNOWN_TASK_INDEX_{task_index}"
    return str(matches.index[0])


def extract_video_frames(video_path: Path, start_frame: int, num_frames: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # This works because LeRobot stores the concatenated video in the same order
    # as the global dataset frame index.
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    saved = []
    for i in range(num_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"[WARN] video ended early at local frame {i}")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        out_path = out_dir / f"frame_{i:03d}.png"
        img.save(out_path)
        saved.append(str(out_path))

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/storage/v-xiangxizheng/zy_workspace/InstructVLA/datasets/droid_100_hf")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", default="observation.images.exterior_image_1_left")
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--out_dir", default="/storage/v-xiangxizheng/zy_workspace/InstructVLA/outputs_vpi/droid100_debug/episode_000000")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = root / "data/chunk-000/file-000.parquet"
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
    tasks_path = root / "meta/tasks.parquet"
    video_path = root / "videos" / args.camera / "chunk-000/file-000.mp4"

    print("=== paths ===")
    for p in [data_path, episodes_path, tasks_path, video_path]:
        print(p, "exists=", p.exists(), "size=", p.stat().st_size if p.exists() else None)

    assert data_path.exists(), data_path
    assert episodes_path.exists(), episodes_path
    assert tasks_path.exists(), tasks_path
    assert video_path.exists(), video_path

    print("\n=== load metadata ===")
    df = pd.read_parquet(data_path)
    eps = pd.read_parquet(episodes_path)
    tasks = pd.read_parquet(tasks_path)

    ep_row = eps[eps["episode_index"] == args.episode]
    if len(ep_row) != 1:
        raise RuntimeError(f"Expected one episode row for episode {args.episode}, got {len(ep_row)}")

    ep_row = ep_row.iloc[0]
    start_idx = int(ep_row["dataset_from_index"])
    end_idx = int(ep_row["dataset_to_index"])
    length = int(ep_row["length"])

    ep_df = df[df["episode_index"] == args.episode].copy().reset_index(drop=True)
    if len(ep_df) != length:
        print(f"[WARN] ep_df len {len(ep_df)} != metadata length {length}")

    task_index = int(ep_df["task_index"].iloc[0])
    instruction = task_index_to_text(tasks, task_index)

    print("episode:", args.episode)
    print("start_idx:", start_idx)
    print("end_idx:", end_idx)
    print("length:", length)
    print("task_index:", task_index)
    print("instruction:", instruction)

    n = min(args.max_frames, len(ep_df))

    actions = np.stack(ep_df["action"].iloc[:n].to_numpy()).astype(np.float32)
    states = np.stack(ep_df["observation.state"].iloc[:n].to_numpy()).astype(np.float32)
    timestamps = ep_df["timestamp"].iloc[:n].to_numpy(dtype=np.float32)
    frame_indices = ep_df["frame_index"].iloc[:n].to_numpy(dtype=np.int64)
    global_indices = ep_df["index"].iloc[:n].to_numpy(dtype=np.int64)

    print("\n=== arrays ===")
    print("actions shape:", actions.shape)
    print("states shape:", states.shape)
    print("first action:", actions[0].tolist())
    print("first state:", states[0].tolist())
    print("timestamps first/last:", float(timestamps[0]), float(timestamps[-1]))

    print("\n=== extract frames ===")
    frame_dir = out_dir / "frames"
    frame_paths = extract_video_frames(
        video_path=video_path,
        start_frame=start_idx,
        num_frames=n,
        out_dir=frame_dir,
    )
    print("saved frames:", len(frame_paths))
    if frame_paths:
        print("first frame:", frame_paths[0])

    if len(frame_paths) != n:
        print(f"[WARN] saved {len(frame_paths)} frames but expected {n}")
        n = len(frame_paths)
        actions = actions[:n]
        states = states[:n]
        timestamps = timestamps[:n]
        frame_indices = frame_indices[:n]
        global_indices = global_indices[:n]

    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "timestamps.npy", timestamps)
    np.save(out_dir / "frame_indices.npy", frame_indices)
    np.save(out_dir / "global_indices.npy", global_indices)

    meta = {
        "dataset": "lerobot/droid_100",
        "episode": args.episode,
        "camera": args.camera,
        "instruction": instruction,
        "task_index": task_index,
        "dataset_from_index": start_idx,
        "dataset_to_index": end_idx,
        "metadata_length": length,
        "num_saved_frames": n,
        "video_path": str(video_path),
        "frame_paths": frame_paths,
        "actions_shape": list(actions.shape),
        "states_shape": list(states.shape),
        "first_action": actions[0].tolist() if n > 0 else None,
        "first_state": states[0].tolist() if n > 0 else None,
    }

    (out_dir / "episode_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")

    print("\n=== done ===")
    print("out_dir:", out_dir)
    print("meta:", out_dir / "episode_meta.json")
    print("instruction:", instruction)


if __name__ == "__main__":
    main()
