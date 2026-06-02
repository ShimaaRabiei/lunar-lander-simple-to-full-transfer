from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    x = str(x).strip().lower()
    if x in ['1', 'true', 'yes', 'y']:
        return True
    if x in ['0', 'false', 'no', 'n']:
        return False
    raise argparse.ArgumentTypeError(f'Cannot parse boolean value: {x}')


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == '':
            return default
        return float(s)
    except Exception:
        return default


def pick(row, keys, default=np.nan):
    for k in keys:
        if k in row and str(row[k]).strip() != '':
            return row[k]
    return default


def read_csv_rows(path: Path):
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parse_full_summary_rows(csv_path: Path, lambda_value: float):
    rows = read_csv_rows(csv_path)
    out = []
    for row in rows:
        lam = to_float(pick(row, ['lambda', 'lambda_value', 'variation_lambda']))
        if not np.isfinite(lam) or abs(lam - float(lambda_value)) > 1e-12:
            continue
        wn = to_float(pick(row, ['wn', 'sweep_wn', 'inner_wn']))
        zeta = to_float(pick(row, ['zeta', 'sweep_zeta', 'inner_zeta']))
        metric = to_float(pick(row, ['mean_discounted_task_return', 'mean_discounted_common_task_return']))
        success = to_float(pick(row, ['success_rate', 'official_success_rate', 'reduced_success_rule_rate']))
        final_distance = to_float(pick(row, ['mean_final_distance_to_pad', 'final_distance_to_pad']))
        out.append({
            'lambda': lam,
            'wn': wn,
            'zeta': zeta,
            'metric': metric,
            'success_rate': success,
            'final_distance': final_distance,
        })
    out = [r for r in out if np.isfinite(r['wn']) and np.isfinite(r['zeta']) and np.isfinite(r['metric'])]
    out.sort(key=lambda r: (r['metric'], r['wn'], r['zeta']))
    return out


def parse_pairs_string(s: str):
    items = []
    if not s.strip():
        return items
    for chunk in s.split(';'):
        part = chunk.strip()
        if not part:
            continue
        toks = [t.strip() for t in part.split(':')]
        if len(toks) < 2:
            raise ValueError(f'Bad pair spec: {part}')
        wn = float(toks[0])
        zeta = float(toks[1])
        label = toks[2] if len(toks) >= 3 else f'wn={wn:g}, zeta={zeta:g}'
        items.append({'wn': wn, 'zeta': zeta, 'label': label})
    return items


def choose_pairs(summary_rows, mode: str, top_k: int, bottom_k: int):
    if not summary_rows:
        raise ValueError('No rows found for the requested lambda in full_summaries.csv')
    if mode == 'all':
        selected = list(summary_rows)
    elif mode == 'best_worst':
        selected = [summary_rows[-1], summary_rows[0]]
    elif mode == 'top_bottom':
        bottom = summary_rows[:max(0, int(bottom_k))]
        top = list(reversed(summary_rows[-max(0, int(top_k)):]))
        merged = []
        seen = set()
        for r in top + bottom:
            key = (float(r['wn']), float(r['zeta']))
            if key not in seen:
                merged.append(r)
                seen.add(key)
        selected = merged
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    out = []
    best_metric = max(r['metric'] for r in summary_rows)
    worst_metric = min(r['metric'] for r in summary_rows)
    for r in selected:
        label = f"wn={r['wn']:g}, zeta={r['zeta']:g}"
        if abs(r['metric'] - best_metric) < 1e-12:
            label += ' [best]'
        if abs(r['metric'] - worst_metric) < 1e-12:
            label += ' [worst]'
        out.append({
            'wn': float(r['wn']),
            'zeta': float(r['zeta']),
            'label': label,
            'metric': float(r['metric']),
            'success_rate': float(r.get('success_rate', np.nan)),
            'final_distance': float(r.get('final_distance', np.nan)),
        })
    return out


def load_policy_reduced(checkpoint: str, cfg: Config):
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
def run_reduced_episode(cfg: Config, model, obs_rms, seed: int):
    env = make_env(cfg, render_mode=None)
    obs, _ = env.reset(seed=int(seed))
    rows = []
    task_return = 0.0
    variation = 0.0
    prev_theta_star = 0.0
    success = False
    crash = False
    out_of_bounds = False
    for t in range(cfg.max_steps):
        obs_in = obs
        if obs_rms is not None:
            obs_in = obs_rms.normalize(obs_in, cfg.obs_clip)
        obs_t = torch.tensor(obs_in, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        action, _, _, _ = model.get_action_and_value(obs_t, deterministic=cfg.deterministic_eval)
        action_np = action.squeeze(0).cpu().numpy()
        theta_star_norm = float(np.clip(action_np[1], -1.0, 1.0)) if len(action_np) > 1 else 0.0
        theta_star = math.radians(cfg.theta_limit_deg) * theta_star_norm
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        rows.append({
            'step': t,
            'x': float(next_obs[0]),
            'y': float(next_obs[1]),
            'theta_star': theta_star,
            'theta': float(next_obs[4]) if len(next_obs) > 4 else np.nan,
        })
        success = bool(info.get('success', False))
        crash = bool(info.get('crash', False))
        out_of_bounds = bool(info.get('out_of_bounds', False))
        task_return += float(info.get('task_reward', reward))
        variation += abs(float(theta_star - prev_theta_star))
        prev_theta_star = theta_star
        obs = next_obs
        if bool(terminated or truncated):
            break
    env.close()
    metrics = {
        'success': bool(success),
        'crash': bool(crash),
        'out_of_bounds': bool(out_of_bounds),
        'task_return': float(task_return),
        'theta_variation': float(variation),
        'length': len(rows),
    }
    return rows, metrics


def build_deploy_args(base_args, wn: float, zeta: float):
    return Namespace(
        checkpoint=base_args.checkpoint,
        seed_file='',
        num_episodes=1,
        save_dir=str(base_args.out_dir),
        run_name='overlay_tmp',
        controller=base_args.controller,
        scenario=base_args.scenario,
        device=base_args.device,
        seed=base_args.seed,
        theta_limit_deg=base_args.theta_limit_deg,
        variation_lambda=base_args.variation_lambda,
        step_penalty=base_args.step_penalty,
        gamma=base_args.gamma,
        obs_clip=base_args.obs_clip,
        deterministic_eval=base_args.deterministic_eval,
        max_episode_steps=base_args.max_episode_steps,
        gravity=base_args.gravity,
        enable_wind=False,
        wind_power=base_args.wind_power,
        turbulence_power=base_args.turbulence_power,
        inner_wn=wn,
        inner_zeta=zeta,
        ref_wn=base_args.ref_wn,
        ref_zeta=base_args.ref_zeta,
        theta_dot_filter_alpha=base_args.theta_dot_filter_alpha,
        controller_deadband=base_args.controller_deadband,
        min_side_action=base_args.min_side_action,
        max_side_action=base_args.max_side_action,
        effort_scale=base_args.effort_scale,
        side_sign=base_args.side_sign,
        zero_side_on_contact=base_args.zero_side_on_contact,
        main_gain=base_args.main_gain,
        main_bias=base_args.main_bias,
        side_gain=base_args.side_gain,
        side_bias=base_args.side_bias,
        side_delay_steps=base_args.side_delay_steps,
        theta_measurement_bias=base_args.theta_measurement_bias,
        omega_measurement_bias=base_args.omega_measurement_bias,
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
        low_velocity_threshold=base_args.low_velocity_threshold,
        save_step_csv=True,
        record_videos=False,
        num_videos=0,
        video_fps=base_args.video_fps,
    )


def make_overlay_static(traces, out_png: Path):
    plt.figure(figsize=(9, 7))
    for tr in traces:
        xs = [r['x'] for r in tr['rows']]
        ys = [r['y'] for r in tr['rows']]
        plt.plot(xs, ys, linewidth=2, label=tr['label'])
        if xs and ys:
            plt.scatter([xs[0]], [ys[0]], marker='o', s=45)
            plt.scatter([xs[-1]], [ys[-1]], marker='x', s=70)
    plt.axhline(0.0, linewidth=1.0)
    plt.scatter([0.0], [0.0], marker='*', s=120, label='pad')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory overlay')
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9, ncol=1, loc='best')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def make_overlay_video(traces, out_path: Path, fps: int = 25, xlim=(-1.5, 1.5), ylim=(-0.2, 1.6)):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(len(tr['rows']) for tr in traces)
    frames = []
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if len(colors) < len(traces):
        extra = list(plt.cm.tab20(np.linspace(0, 1, len(traces))))
        colors = colors + extra

    for k in range(max_len):
        fig, ax = plt.subplots(figsize=(9, 7))
        for idx, tr in enumerate(traces):
            color = colors[idx]
            rows = tr['rows']
            upto = min(k + 1, len(rows))
            xs = [r['x'] for r in rows[:upto]]
            ys = [r['y'] for r in rows[:upto]]
            ax.plot(xs, ys, linewidth=2.2, label=tr['label'], color=color)
            if xs and ys:
                ax.scatter([xs[-1]], [ys[-1]], s=42, color=color)
        ax.axhline(0.0, linewidth=1.0)
        ax.scatter([0.0], [0.0], marker='*', s=140)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Trajectory overlay, frame {k+1}/{max_len}')
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8, loc='upper right')
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    try:
        imageio.mimsave(out_path, frames, fps=fps)
        return out_path
    except Exception:
        gif_path = out_path.with_suffix('.gif')
        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--lambda_value', type=float, required=True)
    p.add_argument('--summary_csv', default='')
    p.add_argument('--pairs', default='')
    p.add_argument('--mode', choices=['best_worst', 'top_bottom', 'all'], default='best_worst')
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--bottom_k', type=int, default=1)
    p.add_argument('--seed', type=int, default=20260531)
    p.add_argument('--seed_file', default='')
    p.add_argument('--seed_index', type=int, default=0)
    p.add_argument('--theta_limit_deg', type=float, default=20.0)
    p.add_argument('--variation_lambda', type=float, default=0.0)
    p.add_argument('--step_penalty', type=float, default=0.02)
    p.add_argument('--device', default='cpu')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--obs_clip', type=float, default=10.0)
    p.add_argument('--deterministic_eval', type=str2bool, default=True)
    p.add_argument('--max_episode_steps', type=int, default=1000)
    p.add_argument('--controller', default='pd', choices=['pd', 'mrc', 'adaptive_bias', 'adaptive_gain'])
    p.add_argument('--scenario', default='nominal', choices=['nominal', 'wind', 'bias', 'wind_bias'])
    p.add_argument('--inner_wn', type=float, default=5.0)
    p.add_argument('--inner_zeta', type=float, default=0.7)
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
    p.add_argument('--video_fps', type=int, default=25)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed)
    if args.seed_file:
        arr = np.load(args.seed_file).astype(np.int64)
        if not (0 <= int(args.seed_index) < len(arr)):
            raise ValueError(f'seed_index {args.seed_index} is out of range for {args.seed_file}')
        seed = int(arr[int(args.seed_index)])

    if args.pairs.strip():
        selected = parse_pairs_string(args.pairs)
    else:
        if not args.summary_csv:
            raise ValueError('Provide --summary_csv when --pairs is not used')
        summary_rows = parse_full_summary_rows(Path(args.summary_csv), args.lambda_value)
        selected = choose_pairs(summary_rows, args.mode, args.top_k, args.bottom_k)

    reduced_cfg = Config()
    reduced_cfg.mode = 'eval'
    reduced_cfg.device = args.device
    reduced_cfg.theta_limit_deg = args.theta_limit_deg
    reduced_cfg.step_penalty = args.step_penalty
    reduced_cfg.variation_lambda = args.variation_lambda
    reduced_cfg.deterministic_eval = args.deterministic_eval
    reduced_cfg.seed = seed

    reduced_model, reduced_obs_rms = load_policy_reduced(args.checkpoint, reduced_cfg)
    reduced_rows, reduced_metrics = run_reduced_episode(reduced_cfg, reduced_model, reduced_obs_rms, seed)

    deploy_model, deploy_obs_rms, deploy_obs_dim, _ = deploy.load_policy(args.checkpoint, args.device)
    traces = [{
        'label': f'Reduced λ={args.lambda_value:g}',
        'rows': reduced_rows,
        'kind': 'reduced',
        'meta': reduced_metrics,
    }]

    saved_selected = []
    for item in selected:
        wn = float(item['wn'])
        zeta = float(item['zeta'])
        dep_args = build_deploy_args(args, wn, zeta)
        side_sign = dep_args.side_sign if int(dep_args.side_sign) != 0 else deploy.detect_side_sign(dep_args)
        controller = deploy.build_controller(dep_args, side_sign)
        mismatch = deploy.ActuatorMismatch(dep_args.main_gain, dep_args.main_bias, dep_args.side_gain, dep_args.side_bias, dep_args.side_delay_steps)
        ep, rows = deploy.run_episode(dep_args, deploy_model, deploy_obs_rms, deploy_obs_dim, seed, controller, mismatch, False, None)
        label = item.get('label', f'wn={wn:g}, zeta={zeta:g}')
        label = f"{label} | R={float(ep['discounted_common_task_return']):.1f}"
        traces.append({
            'label': label,
            'rows': rows,
            'kind': 'full',
            'meta': ep,
            'wn': wn,
            'zeta': zeta,
        })
        saved_selected.append({
            'wn': wn,
            'zeta': zeta,
            'label': item.get('label', label),
            'summary_metric': item.get('metric', np.nan),
            'episode_metric': float(ep['discounted_common_task_return']),
            'episode_success': bool(ep['official_success']),
            'episode_final_distance': float(ep['final_distance_to_pad']),
        })

    stem = f"lam{str(args.lambda_value).replace('.', 'p')}_seed{seed}"
    static_png = out_dir / f"overlay_{stem}.png"
    video_mp4 = out_dir / f"overlay_{stem}.mp4"
    make_overlay_static(traces, static_png)
    actual_video_path = make_overlay_video(traces, video_mp4, fps=int(args.video_fps))

    manifest = {
        'seed': seed,
        'lambda_value': args.lambda_value,
        'checkpoint': args.checkpoint,
        'summary_csv': args.summary_csv,
        'selected_pairs': saved_selected,
        'reduced_metrics': reduced_metrics,
        'outputs': {
            'overlay_png': str(static_png),
            'overlay_video': str(actual_video_path),
        },
    }
    (out_dir / f"overlay_{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
