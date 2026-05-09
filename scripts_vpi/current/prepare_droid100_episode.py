import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def task_index_to_text(tasks_df, task_index: int) -> str:
    matches = tasks_df[tasks_df["task_index"] == task_index]
    if len(matches) == 0:
        return f"UNKNOWN_TASK_INDEX_{task_index}"
    return str(matches.index[0])


def sample_indices(length: int, num_frames: int, strategy: str = "uniform"):
    if length <= 0:
        raise ValueError("Episode length must be positive.")

    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    n = min(num_frames, length)

    if strategy == "uniform":
        # Cover the whole trajectory from beginning to end.
        return np.linspace(0, length - 1, n, dtype=np.int64)

    if strategy == "first":
        # Debug only. Not recommended for final benchmark.
        return np.arange(n, dtype=np.int64)

    raise ValueError(f"Unknown sample strategy: {strategy}")


def get_ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "imageio-ffmpeg is required. Install it with: python -m pip install -U imageio-ffmpeg"
        ) from e


def extract_selected_frames(video_path: Path, global_indices, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob("frame_*.png"):
        old.unlink()

    ffmpeg = get_ffmpeg_exe()

    # ffmpeg select expression for exact global frame indices.
    # Example: eq(n\,166)+eq(n\,174)+...
    select_expr = "+".join([f"eq(n\\,{int(i)})" for i in global_indices])

    output_pattern = str(out_dir / "frame_%03d.png")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "0",
        "-start_number",
        "0",
        output_pattern,
    ]

    print("=== ffmpeg command ===")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("=== ffmpeg output tail ===")
    lines = result.stdout.splitlines()
    for line in lines[-30:]:
        print(line)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")

    frame_paths = sorted(out_dir.glob("frame_*.png"))
    return frame_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/droid_100_hf")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera", default="observation.images.exterior_image_1_left")
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--sample_strategy", default="uniform", choices=["uniform", "first"])
    parser.add_argument("--out_root", default="outputs_vpi/droid100_debug")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    episode = args.episode

    data_path = root / "data/chunk-000/file-000.parquet"
    episodes_path = root / "meta/episodes/chunk-000/file-000.parquet"
    tasks_path = root / "meta/tasks.parquet"
    video_path = root / "videos" / args.camera / "chunk-000/file-000.mp4"

    out_dir = out_root / f"episode_{episode:06d}"
    frame_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== paths ===")
    for p in [data_path, episodes_path, tasks_path, video_path]:
        print(p, "exists=", p.exists(), "size=", p.stat().st_size if p.exists() else None)

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    print("\n=== load parquet ===")
    df = pd.read_parquet(data_path)
    eps = pd.read_parquet(episodes_path)
    tasks = pd.read_parquet(tasks_path)

    ep_meta = eps[eps["episode_index"] == episode]
    if len(ep_meta) != 1:
        raise RuntimeError(f"Expected one metadata row for episode {episode}, got {len(ep_meta)}")

    ep_row = ep_meta.iloc[0]
    start_idx = int(ep_row["dataset_from_index"])
    end_idx = int(ep_row["dataset_to_index"])
    length = int(ep_row["length"])

    ep_df = df[df["episode_index"] == episode].copy().reset_index(drop=True)
    if len(ep_df) != length:
        print(f"[WARN] ep_df length {len(ep_df)} != metadata length {length}")

    if len(ep_df) == 0:
        raise RuntimeError(f"No frames found for episode {episode}")

    task_index = int(ep_df["task_index"].iloc[0])
    instruction = task_index_to_text(tasks, task_index).strip()

    if instruction == "":
        raise RuntimeError(
            f"Episode {episode} has empty instruction. Please choose another episode."
        )

    local_indices = sample_indices(
        length=len(ep_df),
        num_frames=args.num_frames,
        strategy=args.sample_strategy,
    )

    global_indices = start_idx + local_indices

    print("\n=== episode info ===")
    print("episode:", episode)
    print("instruction:", instruction)
    print("task_index:", task_index)
    print("metadata length:", length)
    print("dataset_from_index:", start_idx)
    print("dataset_to_index:", end_idx)
    print("sample_strategy:", args.sample_strategy)
    print("requested num_frames:", args.num_frames)
    print("actual sampled frames:", len(local_indices))
    print("local_indices:", local_indices.tolist())
    print("global_indices:", global_indices.tolist())

    print("\n=== extract frames ===")
    frame_paths = extract_selected_frames(video_path, global_indices, frame_dir)

    print("saved frames:", len(frame_paths))
    if len(frame_paths) != len(local_indices):
        raise RuntimeError(
            f"Expected {len(local_indices)} frames, got {len(frame_paths)}. "
            "Frame extraction mismatch."
        )

    selected_df = ep_df.iloc[local_indices].copy().reset_index(drop=True)

    actions = np.stack(selected_df["action"].to_numpy()).astype(np.float32)
    states = np.stack(selected_df["observation.state"].to_numpy()).astype(np.float32)
    timestamps = selected_df["timestamp"].to_numpy(dtype=np.float32)
    frame_indices = selected_df["frame_index"].to_numpy(dtype=np.int64)
    dataset_indices = selected_df["index"].to_numpy(dtype=np.int64)

    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "timestamps.npy", timestamps)
    np.save(out_dir / "frame_indices.npy", frame_indices)
    np.save(out_dir / "local_indices.npy", local_indices)
    np.save(out_dir / "global_indices.npy", global_indices)
    np.save(out_dir / "dataset_indices.npy", dataset_indices)

    meta = {
        "dataset": "lerobot/droid_100",
        "episode": episode,
        "camera": args.camera,
        "instruction": instruction,
        "task_index": task_index,
        "metadata_length": length,
        "dataset_from_index": start_idx,
        "dataset_to_index": end_idx,
        "sample_strategy": args.sample_strategy,
        "requested_num_frames": args.num_frames,
        "num_saved_frames": len(frame_paths),
        "local_indices": local_indices.tolist(),
        "global_indices": global_indices.tolist(),
        "frame_indices": frame_indices.tolist(),
        "timestamps": timestamps.tolist(),
        "video_path": str(video_path),
        "frame_paths": [str(p) for p in frame_paths],
        "actions_shape": list(actions.shape),
        "states_shape": list(states.shape),
        "first_action": actions[0].tolist(),
        "first_state": states[0].tolist(),
    }

    (out_dir / "episode_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")

    print("\n=== done ===")
    print("out_dir:", out_dir)
    print("instruction:", instruction)
    print("actions shape:", actions.shape)
    print("states shape:", states.shape)
    print("first frame:", frame_paths[0])
    print("last frame:", frame_paths[-1])
    print("summary:", out_dir / "episode_meta.json")


if __name__ == "__main__":
    main()
