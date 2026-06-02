from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import torch

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_reduced_order_lunar_lander import ActorCritic, Config, RunningMeanStd

try:
    from gymnasium.envs.box2d.lunar_lander import LunarLander, FPS
except Exception:
    from lunar_lander import LunarLander, FPS


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ["1", "true", "yes", "y"]:
        return True
    if x in ["0", "false", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {x}")


def wrap_to_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def wn_zeta_to_kp_kd(wn: float, zeta: float) -> Tuple[float, float]:
    return float(wn) ** 2, 2.0 * float(zeta) * float(wn)


def main_power_from_cmd(main_cmd: float) -> float:
    if float(main_cmd) > 0.0:
        return float((np.clip(float(main_cmd), 0.0, 1.0) + 1.0) * 0.5)
    return 0.0


def side_power_from_cmd(side_cmd: float) -> float:
    if abs(float(side_cmd)) <= 0.5:
        return 0.0
    return float(np.clip(abs(float(side_cmd)), 0.5, 1.0))


def full_obs_to_policy_obs(full_obs: np.ndarray, prev_theta_star: float, obs_dim: int) -> np.ndarray:
    full_obs = np.asarray(full_obs, dtype=np.float32)
    if obs_dim == 7:
        return np.asarray([full_obs[0], full_obs[1], full_obs[2], full_obs[3], prev_theta_star, full_obs[6], full_obs[7]], dtype=np.float32)
    if obs_dim == 6:
        return np.asarray([full_obs[0], full_obs[1], full_obs[2], full_obs[3], full_obs[6], full_obs[7]], dtype=np.float32)
    if obs_dim == 5:
        return np.asarray([full_obs[0], full_obs[1], full_obs[2], full_obs[3], prev_theta_star], dtype=np.float32)
    if obs_dim == 4:
        return np.asarray(full_obs[:4], dtype=np.float32)
    raise ValueError(f"Unsupported reduced policy observation dimension: {obs_dim}")


def load_policy(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    obs_dim = int(state["actor.0.weight"].shape[1])
    hidden_size = int(state["actor.0.weight"].shape[0])
    cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    std_mode = str(cfg_dict.get("std_mode", "anneal"))
    model = ActorCritic(obs_dim, hidden_size, std_mode).to(device)
    model.load_state_dict(state)
    model.eval()
    obs_rms = None
    if isinstance(ckpt, dict) and ckpt.get("obs_rms") is not None:
        obs_rms = RunningMeanStd((obs_dim,))
        obs_rms.load_state_dict(ckpt["obs_rms"])
    return model, obs_rms, obs_dim, cfg_dict


class TimeLimitDeployment:
    def __init__(self, env: LunarLander, max_episode_steps: int):
        self.env = env
        self.max_episode_steps = int(max_episode_steps)
        self.elapsed_steps = 0
        self.unwrapped = env

    def reset(self, *, seed: Optional[int] = None):
        self.elapsed_steps = 0
        return self.env.reset(seed=seed)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps and not terminated:
            truncated = True
            info = dict(info)
            info["time_limit"] = True
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class ThetaDerivativeFilter:
    def __init__(self, dt: float, alpha: float):
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.reset(0.0)

    def reset(self, theta_star: float = 0.0):
        self.prev = float(theta_star)
        self.rate = 0.0
        self.initialized = False

    def update(self, theta_star: float) -> float:
        theta_star = float(theta_star)
        if not self.initialized:
            self.prev = theta_star
            self.rate = 0.0
            self.initialized = True
            return 0.0
        raw = wrap_to_pi(theta_star - self.prev) / self.dt
        self.rate = self.alpha * self.rate + (1.0 - self.alpha) * raw
        self.prev = theta_star
        return float(self.rate)


class ActuatorMapper:
    def __init__(self, deadband: float, min_side_action: float, max_side_action: float, effort_scale: float):
        self.deadband = float(deadband)
        self.min_side_action = float(min_side_action)
        self.max_side_action = float(max_side_action)
        self.effort_scale = max(float(effort_scale), 1e-6)

    def __call__(self, effort: float) -> float:
        effort = float(effort)
        if abs(effort) < self.deadband:
            return 0.0
        mag01 = np.clip(abs(effort) / self.effort_scale, 0.0, 1.0)
        mag = self.min_side_action + (self.max_side_action - self.min_side_action) * mag01
        return float(math.copysign(np.clip(mag, self.min_side_action, self.max_side_action), effort))


class PDController:
    def __init__(self, kp: float, kd: float, side_sign: int, mapper: ActuatorMapper, zero_side_on_contact: bool):
        self.kp = float(kp)
        self.kd = float(kd)
        self.side_sign = int(np.sign(side_sign)) if side_sign != 0 else 1
        self.mapper = mapper
        self.zero_side_on_contact = bool(zero_side_on_contact)

    def reset(self, theta: float = 0.0, omega: float = 0.0):
        return None

    def __call__(self, theta_star: float, theta_star_dot: float, theta: float, omega: float, c_left: float, c_right: float):
        contact = bool(c_left > 0.5 or c_right > 0.5)
        e = wrap_to_pi(theta_star - theta)
        edot = float(theta_star_dot) - float(omega)
        effort = self.kp * e + self.kd * edot
        side_cmd = self.mapper(self.side_sign * effort)
        if self.zero_side_on_contact and contact:
            side_cmd = 0.0
        return float(side_cmd), {"controller": "pd", "theta_error": float(e), "theta_error_abs": float(abs(e)), "theta_star_dot": float(theta_star_dot), "edot": float(edot), "effort": float(effort), "side_cmd": float(side_cmd), "kp": float(self.kp), "kd": float(self.kd)}


class ModelReferenceController:
    def __init__(self, kp: float, kd: float, side_sign: int, mapper: ActuatorMapper, zero_side_on_contact: bool, ref_wn: float, ref_zeta: float):
        self.kp = float(kp)
        self.kd = float(kd)
        self.side_sign = int(np.sign(side_sign)) if side_sign != 0 else 1
        self.mapper = mapper
        self.zero_side_on_contact = bool(zero_side_on_contact)
        self.ref_wn = float(ref_wn)
        self.ref_zeta = float(ref_zeta)
        self.dt = 1.0 / float(FPS)
        self.reset()

    def reset(self, theta: float = 0.0, omega: float = 0.0):
        self.theta_m = float(theta)
        self.omega_m = float(omega)
        self.initialized = False

    def update_reference_model(self, theta_star: float, theta: float, omega: float):
        if not self.initialized:
            self.theta_m = float(theta)
            self.omega_m = float(omega)
            self.initialized = True
        theta_error_m = wrap_to_pi(self.theta_m - theta_star)
        omega_dot_m = -2.0 * self.ref_zeta * self.ref_wn * self.omega_m - self.ref_wn ** 2 * theta_error_m
        self.omega_m = float(self.omega_m + self.dt * omega_dot_m)
        self.theta_m = float(wrap_to_pi(self.theta_m + self.dt * self.omega_m))
        return self.theta_m, self.omega_m

    def __call__(self, theta_star: float, theta_star_dot: float, theta: float, omega: float, c_left: float, c_right: float):
        theta_m, omega_m = self.update_reference_model(float(theta_star), float(theta), float(omega))
        contact = bool(c_left > 0.5 or c_right > 0.5)
        e = wrap_to_pi(theta_m - theta)
        edot = float(omega_m) - float(omega)
        effort = self.kp * e + self.kd * edot
        side_cmd = self.mapper(self.side_sign * effort)
        if self.zero_side_on_contact and contact:
            side_cmd = 0.0
        return float(side_cmd), {"controller": "mrc", "theta_error": float(e), "theta_error_abs": float(abs(e)), "theta_model": float(theta_m), "omega_model": float(omega_m), "theta_star_dot": float(theta_star_dot), "edot": float(edot), "effort": float(effort), "side_cmd": float(side_cmd), "kp": float(self.kp), "kd": float(self.kd), "ref_wn": float(self.ref_wn), "ref_zeta": float(self.ref_zeta)}


class AdaptiveModelReferenceController(ModelReferenceController):
    def __init__(self, kp: float, kd: float, side_sign: int, mapper: ActuatorMapper, zero_side_on_contact: bool, ref_wn: float, ref_zeta: float, mode: str, lambda_s: float, gamma_p: float, gamma_d: float, gamma_b: float, sigma_p: float, sigma_d: float, sigma_b: float, kp_min: float, kp_max: float, kd_min: float, kd_max: float, bias_min: float, bias_max: float, freeze_on_contact: bool):
        super().__init__(kp, kd, side_sign, mapper, zero_side_on_contact, ref_wn, ref_zeta)
        self.mode = str(mode)
        self.lambda_s = float(lambda_s)
        self.gamma_p = float(gamma_p)
        self.gamma_d = float(gamma_d)
        self.gamma_b = float(gamma_b)
        self.sigma_p = float(sigma_p)
        self.sigma_d = float(sigma_d)
        self.sigma_b = float(sigma_b)
        self.kp_min = float(kp_min)
        self.kp_max = float(kp_max)
        self.kd_min = float(kd_min)
        self.kd_max = float(kd_max)
        self.bias_min = float(bias_min)
        self.bias_max = float(bias_max)
        self.freeze_on_contact = bool(freeze_on_contact)
        self.kp0 = float(kp)
        self.kd0 = float(kd)
        self.reset()

    def reset(self, theta: float = 0.0, omega: float = 0.0):
        super().reset(theta, omega)
        self.kp_hat = getattr(self, "kp0", self.kp)
        self.kd_hat = getattr(self, "kd0", self.kd)
        self.bias_hat = 0.0

    def project(self):
        self.kp_hat = float(np.clip(self.kp_hat, self.kp_min, self.kp_max))
        self.kd_hat = float(np.clip(self.kd_hat, self.kd_min, self.kd_max))
        self.bias_hat = float(np.clip(self.bias_hat, self.bias_min, self.bias_max))

    def __call__(self, theta_star: float, theta_star_dot: float, theta: float, omega: float, c_left: float, c_right: float):
        theta_m, omega_m = self.update_reference_model(float(theta_star), float(theta), float(omega))
        contact = bool(c_left > 0.5 or c_right > 0.5)
        e = wrap_to_pi(theta_m - theta)
        edot = float(omega_m) - float(omega)
        s = edot + self.lambda_s * e
        effort = self.kp_hat * e + self.kd_hat * edot + self.bias_hat
        side_cmd = self.mapper(self.side_sign * effort)
        if self.zero_side_on_contact and contact:
            side_cmd = 0.0
        adaptation_active = not (self.freeze_on_contact and contact)
        if adaptation_active:
            if self.mode == "adaptive_gain":
                self.kp_hat += self.dt * (self.gamma_p * s * e - self.sigma_p * (self.kp_hat - self.kp0))
                self.kd_hat += self.dt * (self.gamma_d * s * edot - self.sigma_d * (self.kd_hat - self.kd0))
            self.bias_hat += self.dt * (self.gamma_b * s - self.sigma_b * self.bias_hat)
            self.project()
        return float(side_cmd), {"controller": self.mode, "theta_error": float(e), "theta_error_abs": float(abs(e)), "theta_model": float(theta_m), "omega_model": float(omega_m), "theta_star_dot": float(theta_star_dot), "edot": float(edot), "s": float(s), "effort": float(effort), "side_cmd": float(side_cmd), "kp": float(self.kp0), "kd": float(self.kd0), "kp_hat": float(self.kp_hat), "kd_hat": float(self.kd_hat), "bias_hat": float(self.bias_hat), "adaptation_active": bool(adaptation_active), "ref_wn": float(self.ref_wn), "ref_zeta": float(self.ref_zeta)}


class ActuatorMismatch:
    def __init__(self, main_gain: float, main_bias: float, side_gain: float, side_bias: float, side_delay_steps: int):
        self.main_gain = float(main_gain)
        self.main_bias = float(main_bias)
        self.side_gain = float(side_gain)
        self.side_bias = float(side_bias)
        self.side_delay_steps = max(0, int(side_delay_steps))
        self.reset()

    def reset(self):
        self.side_buffer = [0.0 for _ in range(self.side_delay_steps + 1)]

    def __call__(self, main_cmd: float, side_cmd: float):
        main_env = float(np.clip(self.main_gain * float(main_cmd) + self.main_bias, -1.0, 1.0))
        side_raw = float(np.clip(self.side_gain * float(side_cmd) + self.side_bias, -1.0, 1.0))
        self.side_buffer.append(side_raw)
        side_env = float(self.side_buffer.pop(0))
        return main_env, side_env


def detect_side_sign(args) -> int:
    env = LunarLander(render_mode=None, continuous=True, gravity=args.gravity, enable_wind=False, wind_power=0.0, turbulence_power=0.0)
    try:
        obs0, _ = env.reset(seed=args.seed)
        theta0 = float(obs0[4])
        obs1, _, _, _, _ = env.step(np.asarray([0.0, 1.0], dtype=np.float32))
        dtheta = wrap_to_pi(float(obs1[4]) - theta0)
        return 1 if dtheta >= 0.0 else -1
    finally:
        env.close()


def build_controller(args, side_sign: int):
    kp, kd = wn_zeta_to_kp_kd(args.inner_wn, args.inner_zeta)
    mapper = ActuatorMapper(args.controller_deadband, args.min_side_action, args.max_side_action, args.effort_scale)
    name = args.controller.lower()
    if name == "pd":
        return PDController(kp, kd, side_sign, mapper, args.zero_side_on_contact)
    if name == "mrc":
        return ModelReferenceController(kp, kd, side_sign, mapper, args.zero_side_on_contact, args.ref_wn, args.ref_zeta)
    if name in ["adaptive_bias", "adaptive_gain"]:
        return AdaptiveModelReferenceController(kp, kd, side_sign, mapper, args.zero_side_on_contact, args.ref_wn, args.ref_zeta, name, args.adaptive_lambda_s, args.adaptive_gamma_p, args.adaptive_gamma_d, args.adaptive_gamma_b, args.adaptive_sigma_p, args.adaptive_sigma_d, args.adaptive_sigma_b, args.adaptive_kp_min, args.adaptive_kp_max, args.adaptive_kd_min, args.adaptive_kd_max, args.adaptive_bias_min, args.adaptive_bias_max, args.freeze_adaptation_on_contact)
    raise ValueError(f"Unsupported controller: {args.controller}")


def make_env(args, render_mode=None):
    enable_wind = args.enable_wind
    wind_power = args.wind_power
    turbulence_power = args.turbulence_power
    if args.scenario in ["wind", "wind_bias"]:
        enable_wind = True
        if wind_power <= 0.0:
            wind_power = 15.0
        if turbulence_power <= 0.0:
            turbulence_power = 1.5
    env = LunarLander(render_mode=render_mode, continuous=True, gravity=args.gravity, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
    return TimeLimitDeployment(env, args.max_episode_steps)


def load_or_create_seeds(path: str, n: int, seed: int):
    if path:
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
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)


def compute_common_reward(obs: np.ndarray, prev_shaping: Optional[float], main_cmd_env: float, success: bool, failure: bool):
    x, y, vx, vy, theta, _, c_left, c_right = [float(v) for v in obs]
    shaping = -100.0 * math.sqrt(x * x + y * y) - 100.0 * math.sqrt(vx * vx + vy * vy) - 100.0 * abs(theta) + 10.0 * c_left + 10.0 * c_right
    if prev_shaping is None:
        reward = 0.0
    else:
        reward = shaping - float(prev_shaping)
    reward -= 0.30 * main_power_from_cmd(main_cmd_env)
    if failure:
        reward = -100.0
    if success:
        reward = +100.0
    return float(reward), float(shaping)


def summarize(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def run_episode(args, model, obs_rms, obs_dim: int, seed: int, controller, mismatch, record: bool, video_path: Optional[Path]):
    env = make_env(args, render_mode="rgb_array" if record else None)
    obs, _ = env.reset(seed=int(seed))
    theta0 = float(obs[4])
    omega0 = float(obs[5]) * FPS / 20.0
    controller.reset(theta0, omega0)
    mismatch.reset()
    deriv_filter = ThetaDerivativeFilter(1.0 / float(FPS), args.theta_dot_filter_alpha)
    deriv_filter.reset(0.0)
    prev_theta_star = 0.0
    prev_common_shaping = None
    frames = []
    if record:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    rows = []
    official_return = 0.0
    common_return = 0.0
    train_like_return = 0.0
    variation_total = 0.0
    discounted_common = 0.0
    discounted_variation = 0.0
    discounted_train_like = 0.0
    disc = 1.0
    distances = []
    speeds = []
    theta_errors = []
    theta_abs = []
    theta_star_abs = []
    theta_star_dot_abs = []
    side_cmd_abs = []
    side_power_vals = []
    main_power_vals = []
    both_contact_vals = []
    low_velocity_vals = []
    success = False
    failure = False
    crash = False
    out_of_bounds = False
    truncated = False
    for t in range(args.max_episode_steps):
        policy_obs = full_obs_to_policy_obs(obs, prev_theta_star, obs_dim)
        if obs_rms is not None:
            policy_obs = obs_rms.normalize(policy_obs, args.obs_clip)
        obs_t = torch.tensor(policy_obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs_t, deterministic=args.deterministic_eval)
        action_np = action.squeeze(0).detach().cpu().numpy()
        main_cmd_policy = float(np.clip(action_np[0], -1.0, 1.0))
        theta_star_norm = float(np.clip(action_np[1], -1.0, 1.0))
        theta_star = math.radians(args.theta_limit_deg) * theta_star_norm
        theta_star_dot = deriv_filter.update(theta_star)
        theta_meas = float(obs[4]) + args.theta_measurement_bias
        omega_meas = float(obs[5]) * FPS / 20.0 + args.omega_measurement_bias
        c_left = float(obs[6])
        c_right = float(obs[7])
        side_cmd_policy, ctrl = controller(theta_star, theta_star_dot, theta_meas, omega_meas, c_left, c_right)
        if args.scenario in ["bias", "wind_bias"]:
            args_side_bias = args.side_bias if abs(args.side_bias) > 0.0 else 0.05
            args_side_gain = args.side_gain if abs(args.side_gain - 1.0) > 1e-12 else 0.85
        else:
            args_side_bias = args.side_bias
            args_side_gain = args.side_gain
        mismatch.side_bias = float(args_side_bias)
        mismatch.side_gain = float(args_side_gain)
        main_cmd_env, side_cmd_env = mismatch(main_cmd_policy, side_cmd_policy)
        next_obs, official_reward, terminated, step_truncated, _ = env.step(np.asarray([main_cmd_env, side_cmd_env], dtype=np.float32))
        crash = bool(env.unwrapped.game_over)
        out_of_bounds = bool(abs(float(next_obs[0])) >= 1.0)
        success = bool(not env.unwrapped.lander.awake)
        failure = bool(crash or out_of_bounds)
        common_reward, prev_common_shaping = compute_common_reward(next_obs, prev_common_shaping, main_cmd_env, success, failure)
        variation = abs(wrap_to_pi(theta_star - prev_theta_star))
        train_like_reward = common_reward - args.variation_lambda * variation - args.step_penalty
        variation_total += float(variation)
        official_return += float(official_reward)
        common_return += float(common_reward)
        train_like_return += float(train_like_reward)
        discounted_common += disc * float(common_reward)
        discounted_variation += disc * float(variation)
        discounted_train_like += disc * float(train_like_reward)
        distance = math.sqrt(float(next_obs[0]) ** 2 + float(next_obs[1]) ** 2)
        speed = math.sqrt(float(next_obs[2]) ** 2 + float(next_obs[3]) ** 2)
        theta_error = wrap_to_pi(theta_star - float(next_obs[4]))
        both_contact = bool(float(next_obs[6]) > 0.5 and float(next_obs[7]) > 0.5)
        low_velocity = bool(speed < args.low_velocity_threshold)
        distances.append(distance)
        speeds.append(speed)
        theta_errors.append(abs(theta_error))
        theta_abs.append(abs(float(next_obs[4])))
        theta_star_abs.append(abs(theta_star))
        theta_star_dot_abs.append(abs(theta_star_dot))
        side_cmd_abs.append(abs(side_cmd_env))
        side_power_vals.append(side_power_from_cmd(side_cmd_env))
        main_power_vals.append(main_power_from_cmd(main_cmd_env))
        both_contact_vals.append(float(both_contact))
        low_velocity_vals.append(float(low_velocity))
        if args.save_step_csv:
            rows.append({"step": t, "x": float(next_obs[0]), "y": float(next_obs[1]), "vx": float(next_obs[2]), "vy": float(next_obs[3]), "theta": float(next_obs[4]), "omega_raw": float(next_obs[5]) * FPS / 20.0, "left_contact": float(next_obs[6]), "right_contact": float(next_obs[7]), "main_cmd_policy": main_cmd_policy, "side_cmd_policy": side_cmd_policy, "main_cmd_env": main_cmd_env, "side_cmd_env": side_cmd_env, "theta_star": theta_star, "theta_star_norm": theta_star_norm, "theta_star_dot_hat": theta_star_dot, "theta_error": theta_error, "official_reward": float(official_reward), "common_task_reward": common_reward, "variation": variation, "train_like_reward": train_like_reward, "controller_effort": float(ctrl.get("effort", 0.0)), "controller_theta_error": float(ctrl.get("theta_error", theta_error)), "controller_kp_hat": float(ctrl.get("kp_hat", ctrl.get("kp", 0.0))), "controller_kd_hat": float(ctrl.get("kd_hat", ctrl.get("kd", 0.0))), "controller_bias_hat": float(ctrl.get("bias_hat", 0.0))})
        if record:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        done = bool(terminated or step_truncated)
        prev_theta_star = theta_star
        disc *= args.gamma
        obs = next_obs
        if done:
            truncated = bool(step_truncated and not terminated)
            break
    env.close()
    if record and imageio is not None and video_path is not None and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=args.video_fps)
    ep = {"seed": int(seed), "official_success": bool(success), "reduced_success_rule": bool(success), "crash": bool(crash), "out_of_bounds": bool(out_of_bounds), "truncated": bool(truncated), "length": int(len(distances)), "official_return": float(official_return), "common_task_return": float(common_return), "train_like_return": float(train_like_return), "discounted_common_task_return": float(discounted_common), "discounted_theta_star_variation": float(discounted_variation), "discounted_train_like_return": float(discounted_train_like), "theta_star_variation": float(variation_total), "mean_distance_to_pad": summarize(distances), "min_distance_to_pad": float(np.min(distances)) if distances else float("nan"), "final_distance_to_pad": float(distances[-1]) if distances else float("nan"), "mean_speed": summarize(speeds), "final_speed": float(speeds[-1]) if speeds else float("nan"), "mean_abs_theta_tracking_error": summarize(theta_errors), "max_abs_theta_tracking_error": float(np.max(theta_errors)) if theta_errors else float("nan"), "final_abs_theta_tracking_error": float(theta_errors[-1]) if theta_errors else float("nan"), "mean_abs_theta": summarize(theta_abs), "final_abs_theta": float(theta_abs[-1]) if theta_abs else float("nan"), "mean_abs_theta_star": summarize(theta_star_abs), "mean_abs_theta_star_dot_hat": summarize(theta_star_dot_abs), "mean_abs_side_cmd": summarize(side_cmd_abs), "mean_side_power": summarize(side_power_vals), "mean_main_power": summarize(main_power_vals), "final_main_power": float(main_power_vals[-1]) if main_power_vals else float("nan"), "both_legs_contact_fraction": summarize(both_contact_vals), "final_both_legs_contact": bool(both_contact_vals[-1] > 0.5) if both_contact_vals else False, "ever_both_legs_contact": bool(np.max(both_contact_vals) > 0.5) if both_contact_vals else False, "low_velocity_fraction": summarize(low_velocity_vals), "landing_candidate": bool((both_contact_vals[-1] > 0.5 if both_contact_vals else False) and (speeds[-1] < args.low_velocity_threshold if speeds else False))}
    return ep, rows


def write_csv(path: Path, rows: Sequence[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed_file", default="")
    p.add_argument("--num_episodes", type=int, default=333)
    p.add_argument("--save_dir", default="deployment_results")
    p.add_argument("--run_name", default="deployment")
    p.add_argument("--controller", default="pd", choices=["pd", "mrc", "adaptive_bias", "adaptive_gain"])
    p.add_argument("--scenario", default="nominal", choices=["nominal", "wind", "bias", "wind_bias"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--theta_limit_deg", type=float, default=20.0)
    p.add_argument("--variation_lambda", type=float, default=0.0)
    p.add_argument("--step_penalty", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--obs_clip", type=float, default=10.0)
    p.add_argument("--deterministic_eval", type=str2bool, default=True)
    p.add_argument("--max_episode_steps", type=int, default=1000)
    p.add_argument("--gravity", type=float, default=-10.0)
    p.add_argument("--enable_wind", type=str2bool, default=False)
    p.add_argument("--wind_power", type=float, default=0.0)
    p.add_argument("--turbulence_power", type=float, default=0.0)
    p.add_argument("--inner_wn", type=float, default=5.0)
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
    p.add_argument("--theta_measurement_bias", type=float, default=0.0)
    p.add_argument("--omega_measurement_bias", type=float, default=0.0)
    p.add_argument("--adaptive_lambda_s", type=float, default=2.0)
    p.add_argument("--adaptive_gamma_p", type=float, default=3.0)
    p.add_argument("--adaptive_gamma_d", type=float, default=1.0)
    p.add_argument("--adaptive_gamma_b", type=float, default=0.5)
    p.add_argument("--adaptive_sigma_p", type=float, default=0.5)
    p.add_argument("--adaptive_sigma_d", type=float, default=0.5)
    p.add_argument("--adaptive_sigma_b", type=float, default=0.2)
    p.add_argument("--adaptive_kp_min", type=float, default=0.0)
    p.add_argument("--adaptive_kp_max", type=float, default=80.0)
    p.add_argument("--adaptive_kd_min", type=float, default=0.0)
    p.add_argument("--adaptive_kd_max", type=float, default=20.0)
    p.add_argument("--adaptive_bias_min", type=float, default=-0.75)
    p.add_argument("--adaptive_bias_max", type=float, default=0.75)
    p.add_argument("--freeze_adaptation_on_contact", type=str2bool, default=True)
    p.add_argument("--low_velocity_threshold", type=float, default=0.05)
    p.add_argument("--save_step_csv", type=str2bool, default=False)
    p.add_argument("--record_videos", type=str2bool, default=False)
    p.add_argument("--num_videos", type=int, default=5)
    p.add_argument("--video_fps", type=int, default=50)
    args = p.parse_args()
    out_dir = Path(args.save_dir) / f"{args.run_name}_{args.controller}_{args.scenario}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, obs_rms, obs_dim, ckpt_cfg = load_policy(args.checkpoint, args.device)
    side_sign = args.side_sign if int(args.side_sign) != 0 else detect_side_sign(args)
    seeds = load_or_create_seeds(args.seed_file, args.num_episodes, args.seed)
    controller = build_controller(args, side_sign)
    mismatch = ActuatorMismatch(args.main_gain, args.main_bias, args.side_gain, args.side_bias, args.side_delay_steps)
    episode_rows = []
    step_rows_all = []
    videos_saved = 0
    for i, seed in enumerate(seeds):
        record = bool(args.record_videos and videos_saved < args.num_videos)
        video_path = out_dir / "videos" / f"episode_{i:03d}_seed_{int(seed)}.mp4" if record else None
        ep, step_rows = run_episode(args, model, obs_rms, obs_dim, int(seed), controller, mismatch, record, video_path)
        ep["episode"] = int(i)
        ep["controller"] = str(args.controller)
        ep["scenario"] = str(args.scenario)
        ep["video_path"] = str(video_path) if record else ""
        if record:
            videos_saved += 1
        episode_rows.append(ep)
        if args.save_step_csv:
            for r in step_rows:
                r["episode"] = int(i)
                r["seed"] = int(seed)
            step_rows_all.extend(step_rows)
    write_csv(out_dir / "episodes.csv", episode_rows)
    if args.save_step_csv:
        write_csv(out_dir / "steps.csv", step_rows_all)
    summary_keys = ["official_success", "reduced_success_rule", "crash", "out_of_bounds", "truncated", "length", "official_return", "common_task_return", "train_like_return", "discounted_common_task_return", "discounted_theta_star_variation", "discounted_train_like_return", "theta_star_variation", "mean_distance_to_pad", "min_distance_to_pad", "final_distance_to_pad", "mean_speed", "final_speed", "mean_abs_theta_tracking_error", "max_abs_theta_tracking_error", "final_abs_theta_tracking_error", "mean_abs_theta", "final_abs_theta", "mean_abs_theta_star", "mean_abs_theta_star_dot_hat", "mean_abs_side_cmd", "mean_side_power", "mean_main_power", "final_main_power", "both_legs_contact_fraction", "final_both_legs_contact", "ever_both_legs_contact", "low_velocity_fraction", "landing_candidate"]
    summary = {"checkpoint": str(args.checkpoint), "seed_file": str(args.seed_file), "episodes": int(len(episode_rows)), "controller": str(args.controller), "scenario": str(args.scenario), "theta_limit_deg": float(args.theta_limit_deg), "side_sign": int(side_sign), "gravity": float(args.gravity), "enable_wind": bool(args.enable_wind or args.scenario in ["wind", "wind_bias"]), "wind_power": float(args.wind_power), "turbulence_power": float(args.turbulence_power), "variation_lambda": float(args.variation_lambda), "gamma": float(args.gamma)}
    for k in summary_keys:
        vals = [r[k] for r in episode_rows if k in r]
        if vals and isinstance(vals[0], bool):
            summary[k + "_rate"] = float(np.mean(vals))
        elif vals:
            summary["mean_" + k] = float(np.mean(vals))
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved deployment evaluation to: {out_dir}")


if __name__ == "__main__":
    main()
