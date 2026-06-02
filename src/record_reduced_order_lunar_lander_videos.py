from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_reduced_order_lunar_lander import (
    ActorCritic,
    Config,
    RunningMeanStd,
    load_checkpoint,
    make_env,
)


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in ["1", "true", "yes", "y"]:
        return True
    if x in ["0", "false", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {x}")


def load_seeds(path: str, n_fallback: int, seed: int):
    if path:
        return np.load(path).astype(np.int64)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**31 - 1, size=n_fallback, dtype=np.int64)


@torch.no_grad()
def run_episode(cfg, model, obs_rms, reset_seed, deterministic=True, keep_frames=True):
    env = make_env(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=int(reset_seed))
    frames = []
    if keep_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    done = False
    step = 0
    total_task_reward = 0.0
    total_train_reward = 0.0
    total_variation = 0.0
    success = False
    crash = False
    out_of_bounds = False
    final_info = {}
    while not done:
        obs_in = obs
        if obs_rms is not None:
            obs_in = obs_rms.normalize(obs_in, cfg.obs_clip)
        obs_t = torch.tensor(obs_in, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        action, _, _, _ = model.get_action_and_value(obs_t, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        if keep_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        total_task_reward += float(info.get("task_reward", reward))
        total_train_reward += float(reward)
        total_variation += float(info.get("variation_cost", 0.0))
        success = bool(info.get("success", False))
        crash = bool(info.get("crash", False))
        out_of_bounds = bool(info.get("out_of_bounds", False))
        final_info = dict(info)
        done = bool(terminated or truncated)
        step += 1
    env.close()
    return {
        "seed": int(reset_seed),
        "success": bool(success),
        "crash": bool(crash),
        "out_of_bounds": bool(out_of_bounds),
        "length": int(step),
        "task_return": float(total_task_reward),
        "train_return": float(total_train_reward),
        "variation": float(total_variation),
        "final_distance_to_pad": float((float(final_info.get("final_distance_to_pad", np.nan)) if "final_distance_to_pad" in final_info else np.nan)),
        "frames": frames,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed_file", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_videos", type=int, default=5)
    p.add_argument("--scan_episodes", type=int, default=333)
    p.add_argument("--theta_limit_deg", type=float, default=20.0)
    p.add_argument("--step_penalty", type=float, default=0.02)
    p.add_argument("--timeout_penalty", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--deterministic_eval", type=str2bool, default=True)
    p.add_argument("--fallback_non_success", type=str2bool, default=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.mode = "eval"
    cfg.device = args.device
    cfg.theta_limit_deg = args.theta_limit_deg
    cfg.step_penalty = args.step_penalty
    cfg.timeout_penalty = args.timeout_penalty
    cfg.deterministic_eval = args.deterministic_eval

    probe_env = make_env(cfg, render_mode=None)
    obs, _ = probe_env.reset(seed=0)
    obs_dim = len(obs)
    probe_env.close()

    model = ActorCritic(obs_dim, cfg.hidden_size, cfg.std_mode).to(cfg.device)
    obs_rms = RunningMeanStd((obs_dim,)) if cfg.obs_norm else None
    load_checkpoint(args.checkpoint, model, optimizer=None, obs_rms=obs_rms, device=cfg.device)
    model.eval()

    seeds = load_seeds(args.seed_file, args.scan_episodes, cfg.fixed_eval_seed)[: args.scan_episodes]
    rows = []
    saved = 0
    fallback = []
    for idx, reset_seed in enumerate(seeds):
        result = run_episode(cfg, model, obs_rms, reset_seed, deterministic=args.deterministic_eval, keep_frames=True)
        frames = result.pop("frames")
        result["episode_index"] = int(idx)
        rows.append(result)
        if result["success"]:
            saved += 1
            video_path = out_dir / f"success_{saved:02d}_seed_{int(reset_seed)}.mp4"
            imageio.mimsave(video_path, frames, fps=args.fps)
            result["video_path"] = str(video_path)
            print(f"saved {video_path}")
            if saved >= args.num_videos:
                break
        elif len(fallback) < args.num_videos:
            fallback.append((result, frames))
    if saved < args.num_videos and args.fallback_non_success:
        for result, frames in fallback:
            if saved >= args.num_videos:
                break
            saved += 1
            video_path = out_dir / f"non_success_{saved:02d}_seed_{int(result['seed'])}.mp4"
            imageio.mimsave(video_path, frames, fps=args.fps)
            result["video_path"] = str(video_path)
            print(f"saved fallback {video_path}")
    summary = {
        "checkpoint": str(args.checkpoint),
        "seed_file": str(args.seed_file),
        "num_saved_videos": int(saved),
        "num_scanned": int(len(rows)),
        "success_rate_scanned": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
        "crash_rate_scanned": float(np.mean([r["crash"] for r in rows])) if rows else 0.0,
        "out_of_bounds_rate_scanned": float(np.mean([r["out_of_bounds"] for r in rows])) if rows else 0.0,
        "mean_task_return_scanned": float(np.mean([r["task_return"] for r in rows])) if rows else float("nan"),
        "mean_length_scanned": float(np.mean([r["length"] for r in rows])) if rows else float("nan"),
    }
    with (out_dir / "video_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "video_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["episode_index", "seed", "success", "crash", "out_of_bounds", "length", "task_return", "train_return", "variation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
