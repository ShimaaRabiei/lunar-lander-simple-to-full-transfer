from __future__ import annotations
import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import deploy_reduced_order_lunar_lander as deploy
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


def load_or_create_seeds(path, n, seed):
    path = Path(path)
    if path.exists():
        arr = np.load(path).astype(np.int64)
        if len(arr) < n:
            raise ValueError(f"Seed file has {len(arr)} seeds but {n} were requested: {path}")
        return arr[:n]
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    np.save(path, arr)
    return arr


def load_reduced_policy(checkpoint, cfg):
    probe_env = make_env(cfg, render_mode=None)
    obs, _ = probe_env.reset(seed=cfg.seed)
    obs_dim = len(obs)
    probe_env.close()
    model = ActorCritic(obs_dim, cfg.hidden_size, cfg.std_mode).to(cfg.device)
    obs_rms = RunningMeanStd((obs_dim,)) if cfg.obs_norm else None
    load_checkpoint(checkpoint, model, optimizer=None, obs_rms=obs_rms, device=cfg.device)
    model.eval()
    return model, obs_rms


@torch.no_grad()
def run_reduced_video(cfg, model, obs_rms, reset_seed, video_path):
    env = make_env(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=int(reset_seed))
    frames = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    done = False
    success = False
    crash = False
    out_of_bounds = False
    length = 0
    task_return = 0.0
    variation = 0.0
    while not done:
        obs_in = obs
        if obs_rms is not None:
            obs_in = obs_rms.normalize(obs_in, cfg.obs_clip)
        obs_t = torch.tensor(obs_in, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        action, _, _, _ = model.get_action_and_value(obs_t, deterministic=cfg.deterministic_eval)
        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        done = bool(terminated or truncated)
        success = bool(info.get("success", False))
        crash = bool(info.get("crash", False))
        out_of_bounds = bool(info.get("out_of_bounds", False))
        task_return += float(info.get("task_reward", reward))
        variation += float(info.get("variation_cost", 0.0))
        length += 1
    env.close()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=cfg_video_fps)
    return {
        "success": success,
        "crash": crash,
        "out_of_bounds": out_of_bounds,
        "length": length,
        "task_return": task_return,
        "variation": variation,
    }


def build_deploy_args(args):
    return Namespace(
        checkpoint=args.checkpoint,
        seed_file=args.seed_file,
        num_episodes=args.scan_episodes,
        save_dir=str(args.out_dir),
        run_name=args.run_name,
        controller=args.controller,
        scenario=args.scenario,
        device=args.device,
        seed=args.seed,
        theta_limit_deg=args.theta_limit_deg,
        variation_lambda=args.variation_lambda,
        step_penalty=args.step_penalty,
        gamma=args.gamma,
        obs_clip=args.obs_clip,
        deterministic_eval=args.deterministic_eval,
        max_episode_steps=args.max_episode_steps,
        gravity=args.gravity,
        enable_wind=False,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power,
        inner_wn=args.inner_wn,
        inner_zeta=args.inner_zeta,
        ref_wn=args.ref_wn,
        ref_zeta=args.ref_zeta,
        theta_dot_filter_alpha=args.theta_dot_filter_alpha,
        controller_deadband=args.controller_deadband,
        min_side_action=args.min_side_action,
        max_side_action=args.max_side_action,
        effort_scale=args.effort_scale,
        side_sign=args.side_sign,
        zero_side_on_contact=args.zero_side_on_contact,
        main_gain=args.main_gain,
        main_bias=args.main_bias,
        side_gain=args.side_gain,
        side_bias=args.side_bias,
        side_delay_steps=args.side_delay_steps,
        theta_measurement_bias=args.theta_measurement_bias,
        omega_measurement_bias=args.omega_measurement_bias,
        adaptive_lambda_s=2.0,
        adaptive_gamma_p=3.0,
        adaptive_gamma_d=1.0,
        adaptive_gamma_b=0.5,
        adaptive_sigma_p=0.5,
        adaptive_sigma_d=0.5,
        adaptive_sigma_b=0.2,
        adaptive_kp_min=0.0,
        adaptive_kp_max=80.0,
        adaptive_kd_min=0.0,
        adaptive_kd_max=20.0,
        adaptive_bias_min=-0.75,
        adaptive_bias_max=0.75,
        freeze_adaptation_on_contact=True,
        low_velocity_threshold=args.low_velocity_threshold,
        save_step_csv=False,
        record_videos=True,
        num_videos=args.num_pairs,
        video_fps=args.video_fps,
    )


def read_frames(path: Path):
    reader = imageio.get_reader(str(path))
    frames = [frame for frame in reader]
    reader.close()
    return frames


def fit_to_height(frame: np.ndarray, target_h: int):
    img = Image.fromarray(frame)
    w, h = img.size
    if h != target_h:
        new_w = int(round(w * target_h / h))
        img = img.resize((new_w, target_h), Image.Resampling.BILINEAR)
    return np.array(img)


def add_text_banner(frame: np.ndarray, left_title: str, right_title: str, footer: str):
    img = Image.fromarray(frame)
    w, h = img.size
    top_h = 56
    bottom_h = 34
    canvas = Image.new('RGB', (w, h + top_h + bottom_h), (255, 255, 255))
    canvas.paste(img, (0, top_h))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, w, top_h], fill=(245, 245, 245))
    draw.rectangle([0, h + top_h, w, h + top_h + bottom_h], fill=(245, 245, 245))
    mid = w // 2
    draw.line((mid, 0, mid, top_h), fill=(180, 180, 180), width=2)
    draw.line((mid, top_h, mid, h + top_h), fill=(180, 180, 180), width=2)
    draw.text((10, 10), left_title, fill=(20, 20, 20))
    draw.text((mid + 10, 10), right_title, fill=(20, 20, 20))
    draw.text((10, h + top_h + 8), footer, fill=(20, 20, 20))
    return np.array(canvas)


def combine_two_videos(reduced_path: Path, full_path: Path, out_path: Path, left_title: str, right_title: str, footer: str, fps: int):
    red_frames = read_frames(reduced_path)
    full_frames = read_frames(full_path)
    if not red_frames or not full_frames:
        raise RuntimeError('One of the temporary videos has no frames')
    target_h = min(red_frames[0].shape[0], full_frames[0].shape[0])
    red_frames = [fit_to_height(f, target_h) for f in red_frames]
    full_frames = [fit_to_height(f, target_h) for f in full_frames]
    n = max(len(red_frames), len(full_frames))
    red_last = red_frames[-1]
    full_last = full_frames[-1]
    output_frames = []
    for i in range(n):
        left = red_frames[i] if i < len(red_frames) else red_last
        right = full_frames[i] if i < len(full_frames) else full_last
        combined = np.concatenate([left, right], axis=1)
        combined = add_text_banner(combined, left_title, right_title, footer)
        output_frames.append(combined)
    imageio.mimsave(str(out_path), output_frames, fps=fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seed_file', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--run_name', default='combined_videos')
    p.add_argument('--num_pairs', type=int, default=10)
    p.add_argument('--scan_episodes', type=int, default=333)
    p.add_argument('--selection', default='both_success', choices=['first', 'both_success', 'full_success', 'reduced_success'])
    p.add_argument('--lambda_value', type=float, required=True)
    p.add_argument('--label_case', default='')
    p.add_argument('--theta_limit_deg', type=float, default=20.0)
    p.add_argument('--variation_lambda', type=float, default=0.0)
    p.add_argument('--step_penalty', type=float, default=0.02)
    p.add_argument('--timeout_penalty', type=float, default=0.0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=20260528)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--obs_clip', type=float, default=10.0)
    p.add_argument('--deterministic_eval', type=str2bool, default=True)
    p.add_argument('--max_episode_steps', type=int, default=1000)
    p.add_argument('--controller', default='pd', choices=['pd', 'mrc', 'adaptive_bias', 'adaptive_gain'])
    p.add_argument('--scenario', default='nominal', choices=['nominal', 'wind', 'bias', 'wind_bias'])
    p.add_argument('--inner_wn', type=float, required=True)
    p.add_argument('--inner_zeta', type=float, required=True)
    p.add_argument('--ref_wn', type=float, default=6.0)
    p.add_argument('--ref_zeta', type=float, default=1.0)
    p.add_argument('--theta_dot_filter_alpha', type=float, default=0.85)
    p.add_argument('--controller_deadband', type=float, default=0.04)
    p.add_argument('--min_side_action', type=float, default=0.55)
    p.add_argument('--max_side_action', type=float, default=1.0)
    p.add_argument('--effort_scale', type=float, default=1.0)
    p.add_argument('--side_sign', type=int, default=0)
    p.add_argument('--zero_side_on_contact', type=str2bool, default=True)
    p.add_argument('--main_gain', type=float, default=1.0)
    p.add_argument('--main_bias', type=float, default=0.0)
    p.add_argument('--side_gain', type=float, default=1.0)
    p.add_argument('--side_bias', type=float, default=0.0)
    p.add_argument('--side_delay_steps', type=int, default=0)
    p.add_argument('--theta_measurement_bias', type=float, default=0.0)
    p.add_argument('--omega_measurement_bias', type=float, default=0.0)
    p.add_argument('--gravity', type=float, default=-10.0)
    p.add_argument('--wind_power', type=float, default=0.0)
    p.add_argument('--turbulence_power', type=float, default=0.0)
    p.add_argument('--low_velocity_threshold', type=float, default=0.05)
    p.add_argument('--video_fps', type=int, default=50)
    p.add_argument('--keep_raw_videos', type=str2bool, default=False)
    args = p.parse_args()

    global cfg_video_fps
    cfg_video_fps = int(args.video_fps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / '_tmp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    reduced_cfg = Config()
    reduced_cfg.mode = 'eval'
    reduced_cfg.device = args.device
    reduced_cfg.theta_limit_deg = args.theta_limit_deg
    reduced_cfg.step_penalty = args.step_penalty
    reduced_cfg.timeout_penalty = args.timeout_penalty
    reduced_cfg.variation_lambda = args.variation_lambda
    reduced_cfg.deterministic_eval = args.deterministic_eval
    reduced_cfg.seed = args.seed

    reduced_model, reduced_obs_rms = load_reduced_policy(args.checkpoint, reduced_cfg)

    deploy_args = build_deploy_args(args)
    deploy_model, deploy_obs_rms, deploy_obs_dim, _ = deploy.load_policy(args.checkpoint, args.device)
    side_sign = deploy_args.side_sign if int(deploy_args.side_sign) != 0 else deploy.detect_side_sign(deploy_args)
    controller = deploy.build_controller(deploy_args, side_sign)
    mismatch = deploy.ActuatorMismatch(deploy_args.main_gain, deploy_args.main_bias, deploy_args.side_gain, deploy_args.side_bias, deploy_args.side_delay_steps)

    seeds = load_or_create_seeds(args.seed_file, args.scan_episodes, args.seed)
    rows = []
    saved = 0
    case_label = args.label_case.strip() or f'lambda={args.lambda_value:g}, wn={args.inner_wn:g}, zeta={args.inner_zeta:g}'
    left_title = f'Reduced | lambda={args.lambda_value:g}'
    right_title = f'Full | lambda={args.lambda_value:g}, wn={args.inner_wn:g}, zeta={args.inner_zeta:g}'

    for idx, seed in enumerate(seeds):
        temp_reduced_path = temp_dir / f'tmp_reduced_{idx:04d}.mp4'
        temp_full_path = temp_dir / f'tmp_full_{idx:04d}.mp4'
        red = run_reduced_video(reduced_cfg, reduced_model, reduced_obs_rms, int(seed), temp_reduced_path)
        full, _ = deploy.run_episode(deploy_args, deploy_model, deploy_obs_rms, deploy_obs_dim, int(seed), controller, mismatch, True, temp_full_path)
        keep = False
        if args.selection == 'first':
            keep = True
        elif args.selection == 'both_success':
            keep = bool(red['success'] and full['official_success'])
        elif args.selection == 'full_success':
            keep = bool(full['official_success'])
        elif args.selection == 'reduced_success':
            keep = bool(red['success'])

        row = {
            'pair': saved + 1 if keep else '',
            'episode_index': int(idx),
            'seed': int(seed),
            'keep': bool(keep),
            'reduced_success': bool(red['success']),
            'full_success': bool(full['official_success']),
            'reduced_task_return': float(red['task_return']),
            'full_common_task_return': float(full['common_task_return']),
            'full_official_return': float(full['official_return']),
            'reduced_length': int(red['length']),
            'full_length': int(full['length']),
        }

        if keep:
            saved += 1
            combined_path = out_dir / f'pair_{saved:02d}_seed_{int(seed)}_combined.mp4'
            footer = f'{case_label} | seed={int(seed)} | reduced_success={int(bool(red["success"]))} | full_success={int(bool(full["official_success"]))}'
            combine_two_videos(temp_reduced_path, temp_full_path, combined_path, left_title, right_title, footer, int(args.video_fps))
            row['combined_video'] = str(combined_path)
            if args.keep_raw_videos:
                reduced_path = out_dir / f'pair_{saved:02d}_seed_{int(seed)}_reduced.mp4'
                full_path = out_dir / f'pair_{saved:02d}_seed_{int(seed)}_full.mp4'
                temp_reduced_path.replace(reduced_path)
                temp_full_path.replace(full_path)
                row['reduced_video'] = str(reduced_path)
                row['full_video'] = str(full_path)
            else:
                if temp_reduced_path.exists():
                    temp_reduced_path.unlink()
                if temp_full_path.exists():
                    temp_full_path.unlink()
            print(f'saved combined pair {saved}: seed={int(seed)}')
        else:
            if temp_reduced_path.exists():
                temp_reduced_path.unlink()
            if temp_full_path.exists():
                temp_full_path.unlink()
        rows.append(row)
        if saved >= args.num_pairs:
            break

    with (out_dir / 'combined_video_summary.json').open('w', encoding='utf-8') as f:
        json.dump({
            'saved_pairs': saved,
            'selection': args.selection,
            'rows_scanned': len(rows),
            'lambda_value': args.lambda_value,
            'inner_wn': args.inner_wn,
            'inner_zeta': args.inner_zeta,
        }, f, indent=2)
    with (out_dir / 'combined_video_episodes.csv').open('w', newline='', encoding='utf-8') as f:
        keys = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    if temp_dir.exists():
        try:
            temp_dir.rmdir()
        except OSError:
            pass
    print(json.dumps({'saved_pairs': saved, 'out_dir': str(out_dir.resolve())}, indent=2))


if __name__ == '__main__':
    main()
