from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import deploy_reduced_order_lunar_lander as dep
import train_reduced_order_lunar_lander as red


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in ["1", "true", "yes", "y"]:
        return True
    if x in ["0", "false", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {x}")


def resolve_checkpoint(path_str):
    p = Path(path_str)
    if p.is_dir():
        p = p / "final_model.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def tag_float(x):
    return ("%g" % float(x)).replace("-", "m").replace(".", "p")


def load_or_create_seeds(path, n, seed):
    p = Path(path)
    if p.exists():
        arr = np.load(p).astype(np.int64)
        if len(arr) < n:
            raise ValueError(f"Seed file has {len(arr)} seeds but {n} were requested: {p}")
        return arr[:n]
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    np.save(p, arr)
    return arr


def make_stress_conditions(seed_file, n, seed, x_min, x_max, vx_min, vx_max, out_dir):
    p = Path(seed_file)
    csv_path = p.with_suffix(".stress_initial_conditions.csv")
    if csv_path.exists():
        rows = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if len(rows) >= n:
            return [
                {"episode": int(r["episode"]), "reset_seed": int(r["reset_seed"]), "x0": float(r["x0"]), "vx0": float(r["vx0"])}
                for r in rows[:n]
            ]

    reset_seeds = load_or_create_seeds(seed_file, n, seed)
    rng = np.random.default_rng(seed + 17)
    xs = rng.uniform(float(x_min), float(x_max), size=n)
    vxs = rng.uniform(float(vx_min), float(vx_max), size=n)
    conditions = [{"episode": i, "reset_seed": int(reset_seeds[i]), "x0": float(xs[i]), "vx0": float(vxs[i])} for i in range(n)]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reset_seed", "x0", "vx0"])
        writer.writeheader()
        writer.writerows(conditions)

    out_copy = Path(out_dir) / "stress_initial_conditions.csv"
    out_copy.parent.mkdir(parents=True, exist_ok=True)
    with out_copy.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reset_seed", "x0", "vx0"])
        writer.writeheader()
        writer.writerows(conditions)

    return conditions


def stock_state_from_lander(env):
    pos = env.lander.position
    vel = env.lander.linearVelocity
    return np.asarray([
        (pos.x - red.VIEWPORT_W / red.SCALE / 2) / (red.VIEWPORT_W / red.SCALE / 2),
        (pos.y - (env.helipad_y + red.LEG_DOWN / red.SCALE)) / (red.VIEWPORT_H / red.SCALE / 2),
        vel.x * (red.VIEWPORT_W / red.SCALE / 2) / red.FPS,
        vel.y * (red.VIEWPORT_H / red.SCALE / 2) / red.FPS,
        float(env.lander.angle),
        20.0 * float(env.lander.angularVelocity) / red.FPS,
        1.0 if env.legs[0].ground_contact else 0.0,
        1.0 if env.legs[1].ground_contact else 0.0,
    ], dtype=np.float32)


def shaping_from_state(state):
    return float(
        -100.0 * math.sqrt(float(state[0]) ** 2 + float(state[1]) ** 2)
        -100.0 * math.sqrt(float(state[2]) ** 2 + float(state[3]) ** 2)
        -100.0 * abs(float(state[4]))
        +10.0 * float(state[6])
        +10.0 * float(state[7])
    )


def apply_lateral_stress(raw_env, x0, vx0):
    center_x = red.VIEWPORT_W / red.SCALE / 2
    half_width = red.VIEWPORT_W / red.SCALE / 2
    target_x_world = center_x + float(x0) * half_width
    target_vx_world = float(vx0) * red.FPS / half_width
    dx = target_x_world - float(raw_env.lander.position.x)
    bodies = [raw_env.lander] + list(raw_env.legs)
    for body in bodies:
        body.position = (float(body.position.x + dx), float(body.position.y))
        body.linearVelocity = (float(target_vx_world), float(body.linearVelocity.y))
        body.awake = True
    raw_env.lander.awake = True


def reset_reduced_env_with_stress(env, reset_seed, x0, vx0):
    obs, info = env.reset(seed=int(reset_seed))
    raw = env.env
    apply_lateral_stress(raw, x0, vx0)
    raw.prev_theta_star = 0.0
    raw.last_theta_star = 0.0
    state = raw._stock_state_with_imposed_angle(theta_star=0.0)
    raw.prev_shaping = shaping_from_state(state)
    obs = raw._reduced_obs_from_stock(state)
    return obs, info


def reset_full_env_with_stress(env, reset_seed, x0, vx0):
    obs, info = env.reset(seed=int(reset_seed))
    raw = env.unwrapped
    apply_lateral_stress(raw, x0, vx0)
    obs = stock_state_from_lander(raw)
    raw.prev_shaping = shaping_from_state(obs)
    return obs, info


def mean_bool(rows, key):
    return float(np.mean([bool(r[key]) for r in rows])) if rows else float("nan")


def mean_float(rows, key):
    return float(np.mean([float(r[key]) for r in rows])) if rows else float("nan")


def write_rows_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def reduced_config(args, lam):
    cfg = red.Config()
    cfg.mode = "eval"
    cfg.device = args.device
    cfg.theta_limit_deg = args.theta_limit_deg
    cfg.variation_lambda = float(lam)
    cfg.gamma = args.gamma
    cfg.deterministic_eval = True
    cfg.max_episode_steps = args.max_episode_steps
    if hasattr(cfg, "step_penalty"):
        cfg.step_penalty = args.step_penalty
    if hasattr(cfg, "timeout_penalty"):
        cfg.timeout_penalty = 0.0
    return cfg


def eval_reduced(args, checkpoint, lam, conditions, model, obs_rms, obs_dim, out_dir):
    cfg = reduced_config(args, lam)
    rows = []
    for c in conditions:
        env = red.make_env(cfg, render_mode=None)
        obs, _ = reset_reduced_env_with_stress(env, c["reset_seed"], c["x0"], c["vx0"])
        task_return = 0.0
        train_return = 0.0
        variation_total = 0.0
        discounted_task = 0.0
        discounted_variation = 0.0
        discounted_train = 0.0
        disc = 1.0
        final_info = {}
        done = False
        step = 0
        while not done:
            policy_obs = np.asarray(obs, dtype=np.float32)
            if obs_rms is not None:
                policy_obs = obs_rms.normalize(policy_obs, args.obs_clip)
            obs_t = torch.tensor(policy_obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_t, deterministic=True)
            action_np = action.squeeze(0).detach().cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            task_reward = float(info.get("task_reward", reward))
            train_reward = float(info.get("train_reward", reward))
            variation = float(info.get("variation_cost", 0.0))
            task_return += task_reward
            train_return += train_reward
            variation_total += variation
            discounted_task += disc * task_reward
            discounted_variation += disc * variation
            discounted_train += disc * train_reward
            disc *= args.gamma
            final_info = dict(info)
            done = bool(terminated or truncated)
            step += 1
        x, y, vx, vy = [float(v) for v in obs[:4]]
        row = {
            "lambda": float(lam),
            "episode": int(c["episode"]),
            "reset_seed": int(c["reset_seed"]),
            "stress_x0": float(c["x0"]),
            "stress_vx0": float(c["vx0"]),
            "success": bool(final_info.get("success", False)),
            "crash": bool(final_info.get("crash", False)),
            "out_of_bounds": bool(final_info.get("out_of_bounds", False)),
            "truncated": bool(final_info.get("time_limit", False)),
            "length": int(step),
            "task_return": float(task_return),
            "train_return": float(train_return),
            "discounted_task_return": float(discounted_task),
            "variation": float(variation_total),
            "discounted_variation": float(discounted_variation),
            "discounted_train_return": float(discounted_train),
            "final_distance_to_pad": float(math.sqrt(x * x + y * y)),
            "final_speed": float(math.sqrt(vx * vx + vy * vy)),
            "final_both_legs_contact": bool(final_info.get("both_legs_contact", False)),
        }
        rows.append(row)
        env.close()
    summary = {
        "lambda": float(lam),
        "episodes": len(rows),
        "success_rate": mean_bool(rows, "success"),
        "crash_rate": mean_bool(rows, "crash"),
        "out_of_bounds_rate": mean_bool(rows, "out_of_bounds"),
        "truncated_rate": mean_bool(rows, "truncated"),
        "mean_task_return": mean_float(rows, "task_return"),
        "mean_discounted_task_return": mean_float(rows, "discounted_task_return"),
        "mean_variation": mean_float(rows, "variation"),
        "mean_discounted_variation": mean_float(rows, "discounted_variation"),
        "mean_final_distance_to_pad": mean_float(rows, "final_distance_to_pad"),
        "mean_final_speed": mean_float(rows, "final_speed"),
        "final_both_legs_contact_rate": mean_bool(rows, "final_both_legs_contact"),
    }
    write_rows_csv(Path(out_dir) / f"reduced_lambda_{tag_float(lam)}_episodes.csv", rows)
    return summary, rows


def deploy_args(args, lam, wn, zeta):
    return SimpleNamespace(
        checkpoint="",
        seed_file="",
        num_episodes=args.num_episodes,
        save_dir="",
        run_name="",
        controller=args.controller,
        scenario=args.scenario,
        device=args.device,
        seed=args.seed,
        theta_limit_deg=args.theta_limit_deg,
        variation_lambda=float(lam),
        step_penalty=args.step_penalty,
        gamma=args.gamma,
        obs_clip=args.obs_clip,
        deterministic_eval=True,
        max_episode_steps=args.max_episode_steps,
        gravity=args.gravity,
        enable_wind=False,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power,
        inner_wn=float(wn),
        inner_zeta=float(zeta),
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
        theta_measurement_bias=0.0,
        omega_measurement_bias=0.0,
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
        record_videos=False,
        num_videos=0,
        video_fps=50,
    )


def eval_full(args, lam, conditions, model, obs_rms, obs_dim, wn, zeta, side_sign, out_dir):
    dargs = deploy_args(args, lam, wn, zeta)
    controller = dep.build_controller(dargs, side_sign)
    mismatch = dep.ActuatorMismatch(dargs.main_gain, dargs.main_bias, dargs.side_gain, dargs.side_bias, dargs.side_delay_steps)
    rows = []
    for c in conditions:
        env = dep.make_env(dargs, render_mode=None)
        obs, _ = reset_full_env_with_stress(env, c["reset_seed"], c["x0"], c["vx0"])
        theta0 = float(obs[4])
        omega0 = float(obs[5]) * dep.FPS / 20.0
        controller.reset(theta0, omega0)
        mismatch.reset()
        deriv_filter = dep.ThetaDerivativeFilter(1.0 / float(dep.FPS), dargs.theta_dot_filter_alpha)
        deriv_filter.reset(0.0)
        prev_theta_star = 0.0
        prev_task_shaping = None
        task_return = 0.0
        official_return = 0.0
        variation_total = 0.0
        discounted_task = 0.0
        discounted_variation = 0.0
        disc = 1.0
        distances = []
        speeds = []
        theta_errors = []
        success = False
        crash = False
        out_of_bounds = False
        truncated = False
        done = False
        step = 0
        while not done:
            policy_obs = dep.full_obs_to_policy_obs(obs, prev_theta_star, obs_dim)
            if obs_rms is not None:
                policy_obs = obs_rms.normalize(policy_obs, dargs.obs_clip)
            obs_t = torch.tensor(policy_obs, dtype=torch.float32, device=dargs.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_t, deterministic=True)
            action_np = action.squeeze(0).detach().cpu().numpy()
            main_cmd_policy = float(np.clip(action_np[0], -1.0, 1.0))
            theta_star_norm = float(np.clip(action_np[1], -1.0, 1.0))
            theta_star = math.radians(dargs.theta_limit_deg) * theta_star_norm
            theta_star_dot = deriv_filter.update(theta_star)
            side_cmd_policy, ctrl = controller(
                theta_star,
                theta_star_dot,
                float(obs[4]),
                float(obs[5]) * dep.FPS / 20.0,
                float(obs[6]),
                float(obs[7]),
            )
            main_cmd_env, side_cmd_env = mismatch(main_cmd_policy, side_cmd_policy)
            next_obs, official_reward, terminated, step_truncated, _ = env.step(np.asarray([main_cmd_env, side_cmd_env], dtype=np.float32))
            crash = bool(env.unwrapped.game_over)
            out_of_bounds = bool(abs(float(next_obs[0])) >= 1.0)
            success = bool(not env.unwrapped.lander.awake)
            failure = bool(crash or out_of_bounds)
            task_reward, prev_task_shaping = dep.compute_common_reward(next_obs, prev_task_shaping, main_cmd_env, success, failure)
            variation = abs(dep.wrap_to_pi(theta_star - prev_theta_star))
            official_return += float(official_reward)
            task_return += float(task_reward)
            variation_total += float(variation)
            discounted_task += disc * float(task_reward)
            discounted_variation += disc * float(variation)
            disc *= dargs.gamma
            distance = math.sqrt(float(next_obs[0]) ** 2 + float(next_obs[1]) ** 2)
            speed = math.sqrt(float(next_obs[2]) ** 2 + float(next_obs[3]) ** 2)
            theta_error = abs(dep.wrap_to_pi(theta_star - float(next_obs[4])))
            distances.append(distance)
            speeds.append(speed)
            theta_errors.append(theta_error)
            prev_theta_star = theta_star
            obs = next_obs
            done = bool(terminated or step_truncated)
            if done:
                truncated = bool(step_truncated and not terminated)
                break
            step += 1
        row = {
            "lambda": float(lam),
            "wn": float(wn),
            "zeta": float(zeta),
            "episode": int(c["episode"]),
            "reset_seed": int(c["reset_seed"]),
            "stress_x0": float(c["x0"]),
            "stress_vx0": float(c["vx0"]),
            "success": bool(success),
            "crash": bool(crash),
            "out_of_bounds": bool(out_of_bounds),
            "truncated": bool(truncated),
            "length": int(len(distances)),
            "official_return": float(official_return),
            "task_return": float(task_return),
            "discounted_task_return": float(discounted_task),
            "theta_star_variation": float(variation_total),
            "discounted_theta_star_variation": float(discounted_variation),
            "mean_distance_to_pad": dep.summarize(distances),
            "min_distance_to_pad": float(np.min(distances)) if distances else float("nan"),
            "final_distance_to_pad": float(distances[-1]) if distances else float("nan"),
            "mean_speed": dep.summarize(speeds),
            "final_speed": float(speeds[-1]) if speeds else float("nan"),
            "mean_abs_theta_tracking_error": dep.summarize(theta_errors),
            "max_abs_theta_tracking_error": float(np.max(theta_errors)) if theta_errors else float("nan"),
            "final_abs_theta_tracking_error": float(theta_errors[-1]) if theta_errors else float("nan"),
        }
        rows.append(row)
        env.close()
    summary = {
        "lambda": float(lam),
        "wn": float(wn),
        "zeta": float(zeta),
        "episodes": len(rows),
        "success_rate": mean_bool(rows, "success"),
        "crash_rate": mean_bool(rows, "crash"),
        "out_of_bounds_rate": mean_bool(rows, "out_of_bounds"),
        "truncated_rate": mean_bool(rows, "truncated"),
        "mean_official_return": mean_float(rows, "official_return"),
        "mean_task_return": mean_float(rows, "task_return"),
        "mean_discounted_task_return": mean_float(rows, "discounted_task_return"),
        "mean_theta_star_variation": mean_float(rows, "theta_star_variation"),
        "mean_discounted_theta_star_variation": mean_float(rows, "discounted_theta_star_variation"),
        "mean_final_distance_to_pad": mean_float(rows, "final_distance_to_pad"),
        "mean_final_speed": mean_float(rows, "final_speed"),
        "mean_mean_abs_theta_tracking_error": mean_float(rows, "mean_abs_theta_tracking_error"),
        "mean_max_abs_theta_tracking_error": mean_float(rows, "max_abs_theta_tracking_error"),
    }
    write_rows_csv(Path(out_dir) / f"full_lambda_{tag_float(lam)}_zeta_{tag_float(zeta)}_wn_{tag_float(wn)}_episodes.csv", rows)
    return summary, rows


def write_summary_files(out_dir, reduced_summaries, full_summaries):
    out_dir = Path(out_dir)
    write_rows_csv(out_dir / "reduced_summaries.csv", reduced_summaries)
    write_rows_csv(out_dir / "full_summaries.csv", full_summaries)
    (out_dir / "reduced_summaries.json").write_text(json.dumps(reduced_summaries, indent=2), encoding="utf-8")
    (out_dir / "full_summaries.json").write_text(json.dumps(full_summaries, indent=2), encoding="utf-8")


def plot_metric(full_summaries, out_dir, metric, ylabel):
    if plt is None or not full_summaries:
        return
    out_dir = Path(out_dir)
    zetas = sorted({float(r["zeta"]) for r in full_summaries})
    lambdas = sorted({float(r["lambda"]) for r in full_summaries})
    for zeta in zetas:
        plt.figure(figsize=(8, 5))
        for lam in lambdas:
            group = sorted([r for r in full_summaries if float(r["lambda"]) == lam and float(r["zeta"]) == zeta], key=lambda r: float(r["wn"]))
            xs = [float(r["wn"]) for r in group]
            ys = [float(r[metric]) for r in group]
            plt.plot(xs, ys, marker="o", linewidth=2, label=f"lambda={lam:g}")
        plt.xlabel("omega_n")
        plt.ylabel(ylabel)
        plt.title(f"lateral-stress deployment: {ylabel}, zeta={zeta:g}")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_zeta_{tag_float(zeta)}.png", dpi=180)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--lambdas", nargs="+", type=float, required=True)
    p.add_argument("--seed_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--stress_x_min", type=float, default=-0.15)
    p.add_argument("--stress_x_max", type=float, default=0.15)
    p.add_argument("--stress_vx_min", type=float, default=-0.8)
    p.add_argument("--stress_vx_max", type=float, default=0.8)
    p.add_argument("--stress_seed", type=int, default=20260601)
    p.add_argument("--wn_values", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--zeta_values", nargs="+", type=float, default=[0.7])
    p.add_argument("--controller", default="pd", choices=["pd", "mrc", "adaptive_bias", "adaptive_gain"])
    p.add_argument("--scenario", default="nominal", choices=["nominal", "wind", "bias", "wind_bias"])
    p.add_argument("--theta_limit_deg", type=float, default=20.0)
    p.add_argument("--step_penalty", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--obs_clip", type=float, default=10.0)
    p.add_argument("--max_episode_steps", type=int, default=1000)
    p.add_argument("--gravity", type=float, default=-10.0)
    p.add_argument("--wind_power", type=float, default=0.0)
    p.add_argument("--turbulence_power", type=float, default=0.0)
    p.add_argument("--inner_wn", type=float, default=1.0)
    p.add_argument("--inner_zeta", type=float, default=0.7)
    p.add_argument("--ref_wn", type=float, default=6.0)
    p.add_argument("--ref_zeta", type=float, default=1.0)
    p.add_argument("--theta_dot_filter_alpha", type=float, default=0.85)
    p.add_argument("--controller_deadband", type=float, default=0.04)
    p.add_argument("--min_side_action", type=float, default=0.55)
    p.add_argument("--max_side_action", type=float, default=1.0)
    p.add_argument("--effort_scale", type=float, default=1.0)
    p.add_argument("--side_sign", type=int, default=0)
    p.add_argument("--zero_side_on_contact", type=str2bool, default=True)
    p.add_argument("--main_gain", type=float, default=1.0)
    p.add_argument("--main_bias", type=float, default=0.0)
    p.add_argument("--side_gain", type=float, default=1.0)
    p.add_argument("--side_bias", type=float, default=0.0)
    p.add_argument("--side_delay_steps", type=int, default=0)
    p.add_argument("--low_velocity_threshold", type=float, default=0.05)
    args = p.parse_args()

    if len(args.checkpoints) != len(args.lambdas):
        raise ValueError("The number of checkpoints must match the number of lambdas.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = make_stress_conditions(
        args.seed_file,
        args.num_episodes,
        args.stress_seed,
        args.stress_x_min,
        args.stress_x_max,
        args.stress_vx_min,
        args.stress_vx_max,
        out_dir,
    )

    side_sign = args.side_sign if int(args.side_sign) != 0 else dep.detect_side_sign(deploy_args(args, args.lambdas[0], args.wn_values[0], args.zeta_values[0]))

    reduced_summaries = []
    full_summaries = []

    for checkpoint_path, lam in zip(args.checkpoints, args.lambdas):
        checkpoint = resolve_checkpoint(checkpoint_path)
        model, obs_rms, obs_dim, _ = dep.load_policy(str(checkpoint), args.device)
        print(f"evaluating reduced lambda={lam:g}")
        red_summary, _ = eval_reduced(args, checkpoint, lam, conditions, model, obs_rms, obs_dim, out_dir)
        reduced_summaries.append(red_summary)
        write_summary_files(out_dir, reduced_summaries, full_summaries)
        for zeta in args.zeta_values:
            for wn in args.wn_values:
                print(f"evaluating full lambda={lam:g}, zeta={zeta:g}, wn={wn:g}")
                full_summary, _ = eval_full(args, lam, conditions, model, obs_rms, obs_dim, wn, zeta, side_sign, out_dir)
                full_summaries.append(full_summary)
                write_summary_files(out_dir, reduced_summaries, full_summaries)

    plot_metric(full_summaries, out_dir, "mean_discounted_task_return", "discounted task return")
    plot_metric(full_summaries, out_dir, "success_rate", "success rate")
    plot_metric(full_summaries, out_dir, "mean_discounted_theta_star_variation", "discounted theta-reference variation")
    plot_metric(full_summaries, out_dir, "mean_mean_abs_theta_tracking_error", "theta tracking error")
    plot_metric(full_summaries, out_dir, "mean_final_distance_to_pad", "final distance to pad")

    final = {
        "episodes": int(args.num_episodes),
        "lambdas": [float(x) for x in args.lambdas],
        "wn_values": [float(x) for x in args.wn_values],
        "zeta_values": [float(x) for x in args.zeta_values],
        "stress_x_range": [float(args.stress_x_min), float(args.stress_x_max)],
        "stress_vx_range": [float(args.stress_vx_min), float(args.stress_vx_max)],
        "out_dir": str(out_dir.resolve()),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
