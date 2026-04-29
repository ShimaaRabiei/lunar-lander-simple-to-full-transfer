#!/usr/bin/env python3
"""
Evaluate a reduced-policy checkpoint on:
1) the reduced surrogate environment used for training/eval, and
2) the full stock LunarLander via one or more deployment-time inner-loop controllers
   that track the RL reference theta*,
across a configurable bank of initializations.

Main features
-------------
- RL policy still uses [main thrust, theta_ref] from the checkpointed model.
- The reduced rollout uses the exact settle + direct-theta training wrapper imported
  from the aligned training file so evaluation does not drift from training assumptions.
- Transfer rollouts can use one or more controllers:
    * pd             : historical fixed-gain PD baseline
    * adaptive_bias  : fixed PD gains + adaptive disturbance / bias estimate
    * adaptive_gain  : adaptive gains + adaptive disturbance estimate
- The adaptive controllers use a filtered reference model for theta*, preserve the same
  deadzone / clipping / contact-gating interface used by the PD controller, and reset
  their internal adaptive state at the start of every rollout.
- Optional actuator mismatch can be injected in transfer only (gain, bias, delay) so the
  adaptive controller is evaluated under explicit deployment uncertainty.
- Runs N initializations (default 10, can set 100 or more).
- Saves per-rollout trajectories (CSV) for reduced and each transfer controller.
- Saves summary metrics (JSON + CSV).
- Saves plots:
    * 2D plant trajectories (x-y)
    * actions over time
    * tracking error over time
    * adaptive internal states over time
    * observation means/stds
    * aggregate bar charts across initializations
    * cross-controller comparison plots
- Optional overlay replay and GIF saving for a selected controller on the first few initializations.

Notes
-----
- Each initialization uses the same seed for reduced and every transfer controller, so the
  comparison is paired. Different initializations use seed = base_seed + init_index.
- Success and terminal conditions follow the official LunarLander code:
  crash/contact with the moon body => failure, |x| >= 1 => failure,
  and not-awake (Box2D sleep) => success.
- The reduced model uses the settle + direct-theta official-success wrapper, while the
  transfer rollout preserves the same outer-loop RL policy and only swaps the deployment
  inner-loop bridge from PD to adaptive control.
"""

from __future__ import annotations

import os

# Windows OpenMP runtime workaround. This avoids the
# 'libomp.dll / libiomp5md.dll already initialized' abort that can happen
# with torch/numpy/matplotlib stacks in some Conda environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import hashlib
import math
import time
from dataclasses import dataclass, fields
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
from gymnasium.wrappers import TimeLimit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from train_reduced_lander_official_success_settle_direct_theta_objectivefix import (
        Config,
        ActorCritic,
        RunningMeanStd,
        make_env,
        set_seed,
        FPS,
        VIEWPORT_W,
        VIEWPORT_H,
        SCALE,
    )
except ImportError:
    from train_reduced_lander_official_success_settle_direct_theta_objectivefix_evalsave import (
        Config,
        ActorCritic,
        RunningMeanStd,
        make_env,
        set_seed,
        FPS,
        VIEWPORT_W,
        VIEWPORT_H,
        SCALE,
    )

try:
    from lunar_lander import LunarLander
except ImportError:
    from gymnasium.envs.box2d.lunar_lander import LunarLander

try:
    import pygame
    from pygame import gfxdraw
except Exception:
    pygame = None
    gfxdraw = None


# -----------------------------
# Small utilities
# -----------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v in ["1", "true", "True", "yes", "y"]:
        return True
    if v in ["0", "false", "False", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool from {v}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def wn_zeta_to_kp_kd(wn: float, zeta: float) -> Tuple[float, float]:
    """For e_ddot + kd e_dot + kp e = 0 matching s^2 + 2*zeta*wn*s + wn^2."""
    kp = float(wn) ** 2
    kd = 2.0 * float(zeta) * float(wn)
    return kp, kd


def floatify_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out

def shorten_tag(name: str, max_len: int = 48) -> str:
    name = str(name).strip().replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:8]
    keep = max(8, max_len - 9)
    return f"{safe[:keep]}_{digest}"



def load_seed_list(seed_list_json: str = "", seed_list_txt: str = "") -> Optional[List[int]]:
    if seed_list_json:
        with open(seed_list_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("--seed_list_json must point to a JSON list of integers.")
        return [int(x) for x in data]

    if seed_list_txt:
        seeds: List[int] = []
        with open(seed_list_txt, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    seeds.append(int(s))
        return seeds

    return None


def safe_ratio_pct(numer: float, denom: float) -> float:
    numer = float(numer)
    denom = float(denom)
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return 100.0 * numer / denom


# -----------------------------
# Checkpoint / model helpers
# -----------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise KeyError("Checkpoint does not contain a 'config' field.")
    return ckpt


def build_cfg_from_checkpoint(ckpt: Dict[str, Any], override_seed: Optional[int]) -> Config:
    ckpt_cfg = dict(ckpt["config"])
    valid_keys = {f.name for f in fields(Config)}
    filtered_cfg = {k: v for k, v in ckpt_cfg.items() if k in valid_keys}
    cfg = Config(**filtered_cfg)
    if override_seed is not None:
        cfg.seed = int(override_seed)
    return cfg


def build_agent_from_checkpoint(cfg: Config, ckpt: Dict[str, Any], device: torch.device, obs_dim: int, act_dim: int):
    action_scale = np.array([1.0, math.radians(cfg.theta_limit_deg)], dtype=np.float32)
    agent = ActorCritic(obs_dim, act_dim, cfg, action_scale=action_scale).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


def build_obs_rms_from_checkpoint(ckpt: Dict[str, Any], obs_dim: int) -> Optional[RunningMeanStd]:
    obs_rms_state = ckpt.get("obs_rms", None)
    if obs_rms_state is None:
        return None
    rms = RunningMeanStd(shape=(obs_dim,))
    rms.load_state_dict(obs_rms_state)
    return rms


# -----------------------------
# Observation / controller helpers
# -----------------------------

def reduced_obs_from_full(full_obs: np.ndarray, include_leg_contacts: bool) -> np.ndarray:
    full_obs = np.asarray(full_obs, dtype=np.float32)
    if include_leg_contacts:
        return np.asarray([full_obs[0], full_obs[1], full_obs[2], full_obs[3], full_obs[6], full_obs[7]], dtype=np.float32)
    return np.asarray(full_obs[:4], dtype=np.float32)


def extract_angle_and_omega(full_obs: np.ndarray) -> Tuple[float, float]:
    theta = float(full_obs[4])
    omega_scaled = float(full_obs[5])
    omega_raw = omega_scaled * FPS / 20.0
    return theta, omega_raw



class SideEnginePDController:
    def __init__(
        self,
        theta_limit_deg: float,
        wn: float = 5.0,
        zeta: float = 0.7,
        min_side_action: float = 0.55,
        max_side_action: float = 1.0,
        deadband: float = 0.05,
        side_sign: int = 1,
        zero_side_on_contact: bool = False,
    ):
        self.theta_limit_rad = math.radians(theta_limit_deg)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.kp, self.kd = wn_zeta_to_kp_kd(self.wn, self.zeta)
        self.min_side_action = float(min_side_action)
        self.max_side_action = float(max_side_action)
        self.deadband = float(deadband)
        self.side_sign = int(np.sign(side_sign)) if side_sign != 0 else 1
        self.zero_side_on_contact = bool(zero_side_on_contact)

    def __call__(self, theta_ref: float, theta: float, omega_raw: float, c_left: float, c_right: float):
        if self.zero_side_on_contact and (c_left > 0.5 or c_right > 0.5):
            err = wrap_to_pi(theta_ref - theta)
            return 0.0, {
                "theta_error_rad": float(err),
                "theta_error_deg": float(math.degrees(err)),
                "omega_raw": float(omega_raw),
                "effort": 0.0,
                "side_cmd": 0.0,
                "kp": self.kp,
                "kd": self.kd,
                "wn": self.wn,
                "zeta": self.zeta,
            }

        error = wrap_to_pi(theta_ref - theta)
        effort = self.kp * error - self.kd * omega_raw
        u = self.side_sign * effort

        if abs(u) < self.deadband:
            side_cmd = 0.0
        else:
            mag = min(self.max_side_action, max(self.min_side_action, abs(u)))
            side_cmd = float(np.sign(u) * mag)

        return side_cmd, {
            "theta_error_rad": float(error),
            "theta_error_deg": float(math.degrees(error)),
            "omega_raw": float(omega_raw),
            "effort": float(effort),
            "side_cmd": float(side_cmd),
            "kp": self.kp,
            "kd": self.kd,
            "wn": self.wn,
            "zeta": self.zeta,
        }




def actuator_map_with_deadzone(u: float, deadband: float, min_side_action: float, max_side_action: float) -> float:
    if abs(float(u)) < float(deadband):
        return 0.0
    mag = min(float(max_side_action), max(float(min_side_action), abs(float(u))))
    return float(math.copysign(mag, float(u)))


class SideEngineAdaptiveController:
    """
    Online adaptive inner-loop controller for theta*-tracking during transfer evaluation.

    Modes
    -----
    - bias_only:    fixed PD gains + adaptive disturbance/bias estimate
    - gain_and_bias: adaptive gains + adaptive disturbance/bias estimate

    The controller preserves the original side-engine interface used by the PD baseline:
    deadzone, min/max side action, side sign, and optional zeroing on contact.
    """

    def __init__(
        self,
        theta_limit_deg: float,
        wn: float = 5.0,
        zeta: float = 0.7,
        min_side_action: float = 0.55,
        max_side_action: float = 1.0,
        deadband: float = 0.05,
        side_sign: int = 1,
        zero_side_on_contact: bool = False,
        mode: str = "gain_and_bias",
        ref_wn: float = 6.0,
        ref_zeta: float = 1.0,
        lambda_s: float = 2.0,
        gamma_p: float = 3.0,
        gamma_d: float = 1.0,
        gamma_b: float = 0.5,
        sigma_p: float = 0.5,
        sigma_d: float = 0.5,
        sigma_b: float = 0.2,
        kp_min: float = 0.0,
        kp_max: float = 80.0,
        kd_min: float = 0.0,
        kd_max: float = 20.0,
        bias_min: float = -0.75,
        bias_max: float = 0.75,
        freeze_adaptation_on_contact: bool = True,
    ):
        self.theta_limit_rad = math.radians(theta_limit_deg)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.kp0, self.kd0 = wn_zeta_to_kp_kd(self.wn, self.zeta)
        self.min_side_action = float(min_side_action)
        self.max_side_action = float(max_side_action)
        self.deadband = float(deadband)
        self.side_sign = int(np.sign(side_sign)) if side_sign != 0 else 1
        self.zero_side_on_contact = bool(zero_side_on_contact)

        mode = str(mode).strip().lower()
        if mode not in {"bias_only", "gain_and_bias"}:
            raise ValueError(f"Unsupported adaptive mode: {mode}")
        self.mode = mode

        self.ref_wn = float(ref_wn)
        self.ref_zeta = float(ref_zeta)
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
        self.freeze_adaptation_on_contact = bool(freeze_adaptation_on_contact)
        self.dt = 1.0 / float(FPS)
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.theta_m = 0.0
        self.omega_m = 0.0
        self.kp_hat = float(self.kp0)
        self.kd_hat = float(self.kd0)
        self.d_hat = 0.0
        self.step_count = 0

    def _project(self) -> None:
        self.kp_hat = float(np.clip(self.kp_hat, self.kp_min, self.kp_max))
        self.kd_hat = float(np.clip(self.kd_hat, self.kd_min, self.kd_max))
        self.d_hat = float(np.clip(self.d_hat, self.bias_min, self.bias_max))

    def __call__(self, theta_ref: float, theta: float, omega_raw: float, c_left: float, c_right: float):
        contact = bool((c_left > 0.5) or (c_right > 0.5))

        if not self.initialized:
            self.theta_m = float(theta)
            self.omega_m = float(omega_raw)
            self.initialized = True

        # Filter the RL reference through a second-order reference model so the
        # inner loop does not see theta* as a discontinuous command every frame.
        theta_model_error = wrap_to_pi(self.theta_m - float(theta_ref))
        omega_m_dot = -2.0 * self.ref_zeta * self.ref_wn * self.omega_m - (self.ref_wn ** 2) * theta_model_error
        self.omega_m = float(self.omega_m + self.dt * omega_m_dot)
        self.theta_m = float(wrap_to_pi(self.theta_m + self.dt * self.omega_m))

        e = float(wrap_to_pi(self.theta_m - float(theta)))
        e_omega = float(self.omega_m - float(omega_raw))
        s = float(e_omega + self.lambda_s * e)

        v_pre_sign = float(self.kp_hat * e + self.kd_hat * e_omega + self.d_hat)
        u = float(self.side_sign * v_pre_sign)
        side_cmd = actuator_map_with_deadzone(u, self.deadband, self.min_side_action, self.max_side_action)

        contact_gate_applied = False
        if self.zero_side_on_contact and contact:
            side_cmd = 0.0
            contact_gate_applied = True

        adaptation_active = not (self.freeze_adaptation_on_contact and contact)
        if adaptation_active:
            if self.mode == "gain_and_bias":
                self.kp_hat = float(self.kp_hat + self.dt * (self.gamma_p * s * e - self.sigma_p * (self.kp_hat - self.kp0)))
                self.kd_hat = float(self.kd_hat + self.dt * (self.gamma_d * s * e_omega - self.sigma_d * (self.kd_hat - self.kd0)))
            self.d_hat = float(self.d_hat + self.dt * (self.gamma_b * s - self.sigma_b * self.d_hat))
            self._project()

        self.step_count += 1
        return float(side_cmd), {
            "controller_type": "adaptive",
            "adaptive_mode": self.mode,
            "theta_error_rad": e,
            "theta_error_deg": float(math.degrees(e)),
            "omega_raw": float(omega_raw),
            "theta_model_rad": float(self.theta_m),
            "theta_model_deg": float(math.degrees(self.theta_m)),
            "omega_model": float(self.omega_m),
            "e": e,
            "e_omega": e_omega,
            "s": s,
            "effort": float(v_pre_sign),
            "v_unsat": float(u),
            "side_cmd": float(side_cmd),
            "kp": float(self.kp0),
            "kd": float(self.kd0),
            "kp_hat": float(self.kp_hat),
            "kd_hat": float(self.kd_hat),
            "d_hat": float(self.d_hat),
            "wn": float(self.wn),
            "zeta": float(self.zeta),
            "ref_wn": float(self.ref_wn),
            "ref_zeta": float(self.ref_zeta),
            "lambda_s": float(self.lambda_s),
            "adaptation_active": bool(adaptation_active),
            "contact_gate_applied": bool(contact_gate_applied),
            "contact": bool(contact),
        }


class TransferActuatorMismatch:
    """Applies optional deployment-time actuator mismatch in transfer only.

    This keeps reduced evaluation untouched while letting transfer inject explicit
    uncertainty (gain, bias, delay) so adaptive control has a meaningful target.
    """

    def __init__(
        self,
        side_gain: float = 1.0,
        side_bias: float = 0.0,
        side_delay_steps: int = 0,
        main_gain: float = 1.0,
        main_bias: float = 0.0,
    ):
        self.side_gain = float(side_gain)
        self.side_bias = float(side_bias)
        self.side_delay_steps = max(0, int(side_delay_steps))
        self.main_gain = float(main_gain)
        self.main_bias = float(main_bias)
        self.reset()

    def reset(self) -> None:
        self.side_delay_buffer = deque([0.0] * self.side_delay_steps)

    def __call__(self, main_cmd: float, side_cmd: float):
        delayed_side_cmd = float(side_cmd)
        if self.side_delay_steps > 0:
            self.side_delay_buffer.append(float(side_cmd))
            delayed_side_cmd = float(self.side_delay_buffer.popleft())

        plant_main_cmd = float(np.clip(self.main_gain * float(main_cmd) + self.main_bias, -1.0, 1.0))
        plant_side_cmd = float(np.clip(self.side_gain * delayed_side_cmd + self.side_bias, -1.0, 1.0))
        return plant_main_cmd, plant_side_cmd, {
            "controller_main_cmd": float(main_cmd),
            "controller_side_cmd": float(side_cmd),
            "delayed_side_cmd": float(delayed_side_cmd),
            "plant_main_cmd": float(plant_main_cmd),
            "plant_side_cmd": float(plant_side_cmd),
            "side_gain": float(self.side_gain),
            "side_bias": float(self.side_bias),
            "side_delay_steps": int(self.side_delay_steps),
            "main_gain": float(self.main_gain),
            "main_bias": float(self.main_bias),
        }


def controller_name_is_adaptive(name: str) -> bool:
    return str(name).strip().lower() in {"adaptive_bias", "adaptive_gain"}


def build_transfer_controller(name: str, cfg: Config, args, side_sign: int):
    key = str(name).strip().lower()
    if key == "pd":
        return SideEnginePDController(
            theta_limit_deg=cfg.theta_limit_deg,
            wn=args.wn,
            zeta=args.zeta,
            min_side_action=args.min_side_action,
            max_side_action=args.max_side_action,
            deadband=args.deadband,
            side_sign=side_sign,
            zero_side_on_contact=args.zero_side_on_contact,
        )
    if key == "adaptive_bias":
        return SideEngineAdaptiveController(
            theta_limit_deg=cfg.theta_limit_deg,
            wn=args.wn,
            zeta=args.zeta,
            min_side_action=args.min_side_action,
            max_side_action=args.max_side_action,
            deadband=args.deadband,
            side_sign=side_sign,
            zero_side_on_contact=args.zero_side_on_contact,
            mode="bias_only",
            ref_wn=args.adaptive_ref_wn,
            ref_zeta=args.adaptive_ref_zeta,
            lambda_s=args.adaptive_lambda_s,
            gamma_p=args.adaptive_gamma_p,
            gamma_d=args.adaptive_gamma_d,
            gamma_b=args.adaptive_gamma_b,
            sigma_p=args.adaptive_sigma_p,
            sigma_d=args.adaptive_sigma_d,
            sigma_b=args.adaptive_sigma_b,
            kp_min=args.adaptive_kp_min,
            kp_max=args.adaptive_kp_max,
            kd_min=args.adaptive_kd_min,
            kd_max=args.adaptive_kd_max,
            bias_min=args.adaptive_bias_min,
            bias_max=args.adaptive_bias_max,
            freeze_adaptation_on_contact=args.freeze_adaptation_on_contact,
        )
    if key == "adaptive_gain":
        return SideEngineAdaptiveController(
            theta_limit_deg=cfg.theta_limit_deg,
            wn=args.wn,
            zeta=args.zeta,
            min_side_action=args.min_side_action,
            max_side_action=args.max_side_action,
            deadband=args.deadband,
            side_sign=side_sign,
            zero_side_on_contact=args.zero_side_on_contact,
            mode="gain_and_bias",
            ref_wn=args.adaptive_ref_wn,
            ref_zeta=args.adaptive_ref_zeta,
            lambda_s=args.adaptive_lambda_s,
            gamma_p=args.adaptive_gamma_p,
            gamma_d=args.adaptive_gamma_d,
            gamma_b=args.adaptive_gamma_b,
            sigma_p=args.adaptive_sigma_p,
            sigma_d=args.adaptive_sigma_d,
            sigma_b=args.adaptive_sigma_b,
            kp_min=args.adaptive_kp_min,
            kp_max=args.adaptive_kp_max,
            kd_min=args.adaptive_kd_min,
            kd_max=args.adaptive_kd_max,
            bias_min=args.adaptive_bias_min,
            bias_max=args.adaptive_bias_max,
            freeze_adaptation_on_contact=args.freeze_adaptation_on_contact,
        )
    raise ValueError(f"Unsupported controller name: {name}")


def controller_summary_dict(controller, name: str) -> Dict[str, Any]:
    out = {
        "name": str(name),
        "class": type(controller).__name__,
        "wn": float(getattr(controller, "wn", np.nan)),
        "zeta": float(getattr(controller, "zeta", np.nan)),
        "kp": float(getattr(controller, "kp", getattr(controller, "kp0", np.nan))),
        "kd": float(getattr(controller, "kd", getattr(controller, "kd0", np.nan))),
        "deadband": float(getattr(controller, "deadband", np.nan)),
        "min_side_action": float(getattr(controller, "min_side_action", np.nan)),
        "max_side_action": float(getattr(controller, "max_side_action", np.nan)),
        "side_sign": int(getattr(controller, "side_sign", 0)),
        "zero_side_on_contact": bool(getattr(controller, "zero_side_on_contact", False)),
    }
    if isinstance(controller, SideEngineAdaptiveController):
        out.update({
            "adaptive_mode": str(controller.mode),
            "ref_wn": float(controller.ref_wn),
            "ref_zeta": float(controller.ref_zeta),
            "lambda_s": float(controller.lambda_s),
            "gamma_p": float(controller.gamma_p),
            "gamma_d": float(controller.gamma_d),
            "gamma_b": float(controller.gamma_b),
            "sigma_p": float(controller.sigma_p),
            "sigma_d": float(controller.sigma_d),
            "sigma_b": float(controller.sigma_b),
            "kp_min": float(controller.kp_min),
            "kp_max": float(controller.kp_max),
            "kd_min": float(controller.kd_min),
            "kd_max": float(controller.kd_max),
            "bias_min": float(controller.bias_min),
            "bias_max": float(controller.bias_max),
            "freeze_adaptation_on_contact": bool(controller.freeze_adaptation_on_contact),
        })
    return out

def detect_side_sign(cfg: Config, seed: int) -> int:
    env = LunarLander(
        render_mode=None,
        continuous=True,
        gravity=cfg.gravity,
        enable_wind=cfg.enable_wind,
        wind_power=cfg.wind_power,
        turbulence_power=cfg.turbulence_power,
    )
    obs, _ = env.reset(seed=seed)
    theta0 = float(obs[4])
    obs1, _, _, _, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
    theta1 = float(obs1[4])
    env.close()
    dtheta = wrap_to_pi(theta1 - theta0)
    return 1 if dtheta >= 0.0 else -1


# -----------------------------
# Geometry capture for overlay renderer
# -----------------------------

def capture_polygons(env_obj) -> Dict[str, Any]:
    def fixture_world_polys(body):
        polys = []
        for f in body.fixtures:
            shape = f.shape
            if hasattr(shape, "vertices"):
                trans = body.transform
                path = [tuple(trans * v) for v in shape.vertices]
                polys.append(path)
        return polys

    return {
        "lander": fixture_world_polys(env_obj.lander),
        "leg0": fixture_world_polys(env_obj.legs[0]),
        "leg1": fixture_world_polys(env_obj.legs[1]),
    }


def terrain_snapshot(env_obj) -> Dict[str, Any]:
    return {
        "sky_polys": [[tuple(p) for p in poly] for poly in env_obj.sky_polys],
        "helipad_x1": float(env_obj.helipad_x1),
        "helipad_x2": float(env_obj.helipad_x2),
        "helipad_y": float(env_obj.helipad_y),
    }


# -----------------------------
# Rollouts
# -----------------------------
@dataclass
class RolloutResult:
    trajectory: List[Dict[str, Any]]
    terrain: Dict[str, Any]
    return_value: float
    cost: float
    length: int
    success: bool
    crashed: bool


@torch.no_grad()
def run_reduced_episode(env, agent, obs_rms, device, deterministic: bool, seed: int) -> RolloutResult:
    obs, info = env.reset(seed=seed)
    traj: List[Dict[str, Any]] = []
    terrain = terrain_snapshot(env.unwrapped)
    done = False
    ep_return = 0.0
    ep_cost = 0.0
    step_idx = 0

    while not done:
        raw_obs = np.asarray(obs, dtype=np.float32).copy()
        obs_in = raw_obs.copy()
        if obs_rms is not None:
            obs_in = obs_rms.normalize(obs_in)
        obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic:
            _, mean_action, _, _ = agent.distribution(obs_t)
            action = mean_action.squeeze(0).detach().cpu().numpy()
        else:
            action_t, _, _, _, _, _, _ = agent.get_action(obs_t, deterministic=False)
            action = action_t.squeeze(0).detach().cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        full_state = np.asarray(info["full_state"], dtype=np.float32)
        shapes = capture_polygons(env.unwrapped)
        ctrl_dbg = info.get("controller_debug", {}) or {}

        row = {
            "step": int(step_idx),
            "obs0": float(raw_obs[0]),
            "obs1": float(raw_obs[1]),
            "obs2": float(raw_obs[2]),
            "obs3": float(raw_obs[3]),
            "obs4": float(raw_obs[4]) if raw_obs.shape[0] > 4 else np.nan,
            "obs5": float(raw_obs[5]) if raw_obs.shape[0] > 5 else np.nan,
            "x": float(full_state[0]),
            "y": float(full_state[1]),
            "vx": float(full_state[2]),
            "vy": float(full_state[3]),
            "theta_rad": float(full_state[4]),
            "omega_scaled": float(full_state[5]),
            "leg_left": float(full_state[6]),
            "leg_right": float(full_state[7]),
            "action_main": float(action[0]),
            "theta_ref_rad": float(info.get("theta_cmd_rad", math.radians(info.get("theta_cmd_deg", 0.0)))),
            "theta_ref_deg": float(info.get("theta_cmd_deg", 0.0)),
            "side_cmd": float(info.get("side_cmd", 0.0)),
            "side_power": float(info.get("side_power", 0.0)),
            "raw_side_cmd": float(info.get("raw_side_cmd", info.get("side_cmd", 0.0))),
            "settle_mode": bool(info.get("settle_mode", False)),
            "settle_ready_counter": int(info.get("settle_ready_counter", 0)),
            "settle_exit_counter": int(info.get("settle_exit_counter", 0)),
            "settle_entered": bool(info.get("settle_entered", False)),
            "settle_exited": bool(info.get("settle_exited", False)),
            "theta_imposed_pre_step": bool(info.get("theta_imposed_pre_step", False)),
            "theta_imposed_post_step": bool(info.get("theta_imposed_post_step", False)),
            "reward": float(reward),
            "cost": float(info.get("cost", 0.0)),
            "success": bool(info.get("is_success", False)),
            "crashed": bool(info.get("crashed", False)),
            "out_of_bounds": bool(abs(float(full_state[0])) >= 1.0),
            "awake": bool(info.get("awake", True)),
            "theta_error_rad": float(ctrl_dbg.get("theta_error_rad", wrap_to_pi(float(info.get("theta_cmd_rad", 0.0)) - float(full_state[4])))),
            "theta_error_deg": float(ctrl_dbg.get("theta_error_deg", math.degrees(wrap_to_pi(float(info.get("theta_cmd_rad", 0.0)) - float(full_state[4]))))),
            "omega_raw": float(ctrl_dbg.get("omega_raw", float(full_state[5]) * FPS / 20.0)),
            "effort": float(ctrl_dbg.get("effort", np.nan)),
            "kp": np.nan,
            "kd": np.nan,
            "wn": np.nan,
            "zeta": np.nan,
            "shapes": shapes,
        }
        traj.append(row)

        ep_return += float(reward)
        ep_cost += float(info.get("cost", 0.0))
        obs = next_obs
        step_idx += 1
        done = bool(terminated or truncated)

    return RolloutResult(
        trajectory=traj,
        terrain=terrain,
        return_value=float(ep_return),
        cost=float(ep_cost),
        length=len(traj),
        success=bool(traj[-1]["success"] if traj else False),
        crashed=bool(traj[-1]["crashed"] if traj else False),
    )


@torch.no_grad()
def run_transfer_episode(
    env,
    agent,
    controller,
    actuator_mismatch: TransferActuatorMismatch,
    obs_rms,
    include_leg_contacts_in_obs: bool,
    theta_limit_deg: float,
    cfg: Config,
    device,
    deterministic: bool,
    seed: int,
    max_steps: int,
) -> RolloutResult:
    if hasattr(controller, "reset"):
        try:
            controller.reset()
        except Exception:
            pass
    actuator_mismatch.reset()

    full_obs, _ = env.reset(seed=seed)
    traj: List[Dict[str, Any]] = []
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    terrain = terrain_snapshot(base_env)
    done = False
    ep_return = 0.0
    ep_cost = 0.0
    theta_limit_rad = math.radians(theta_limit_deg)
    step_idx = 0

    while not done:
        if step_idx >= max_steps:
            break
        reduced_obs = reduced_obs_from_full(full_obs, include_leg_contacts_in_obs)
        obs_in = reduced_obs.copy()
        if obs_rms is not None:
            obs_in = obs_rms.normalize(obs_in)
        obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic:
            _, mean_action, _, _ = agent.distribution(obs_t)
            action = mean_action.squeeze(0).detach().cpu().numpy()
        else:
            action_t, _, _, _, _, _, _ = agent.get_action(obs_t, deterministic=False)
            action = action_t.squeeze(0).detach().cpu().numpy()

        main_cmd = float(np.clip(action[0], -1.0, 1.0))
        theta_ref = float(np.clip(action[1], -1.0, 1.0) * theta_limit_rad)
        theta, omega_raw = extract_angle_and_omega(full_obs)
        c_left, c_right = float(full_obs[6]), float(full_obs[7])
        side_cmd, ctrl_dbg = controller(theta_ref, theta, omega_raw, c_left, c_right)
        plant_main_cmd, plant_side_cmd, plant_dbg = actuator_mismatch(main_cmd, side_cmd)

        next_full_obs, reward, terminated, truncated, _ = env.step(np.array([plant_main_cmd, plant_side_cmd], dtype=np.float32))
        full_state = np.asarray(next_full_obs, dtype=np.float32)
        shapes = capture_polygons(base_env)
        crashed = bool(base_env.game_over)
        out_of_bounds = bool(abs(float(full_state[0])) >= 1.0)
        awake = bool(base_env.lander.awake)
        success = bool((not crashed) and (not out_of_bounds) and (not awake))
        cost = float(crashed or out_of_bounds or (abs(float(full_state[0])) > float(cfg.safety_x_bound)))

        row = {
            "step": int(step_idx),
            "obs0": float(reduced_obs[0]),
            "obs1": float(reduced_obs[1]),
            "obs2": float(reduced_obs[2]),
            "obs3": float(reduced_obs[3]),
            "obs4": float(reduced_obs[4]) if reduced_obs.shape[0] > 4 else np.nan,
            "obs5": float(reduced_obs[5]) if reduced_obs.shape[0] > 5 else np.nan,
            "x": float(full_state[0]),
            "y": float(full_state[1]),
            "vx": float(full_state[2]),
            "vy": float(full_state[3]),
            "theta_rad": float(full_state[4]),
            "omega_scaled": float(full_state[5]),
            "leg_left": float(full_state[6]),
            "leg_right": float(full_state[7]),
            "action_main": float(main_cmd),
            "plant_main_cmd": float(plant_main_cmd),
            "theta_ref_rad": float(theta_ref),
            "theta_ref_deg": float(math.degrees(theta_ref)),
            "side_cmd": float(side_cmd),
            "plant_side_cmd": float(plant_side_cmd),
            "delayed_side_cmd": float(plant_dbg.get("delayed_side_cmd", plant_side_cmd)),
            "side_power": np.nan,
            "reward": float(reward),
            "cost": float(cost),
            "success": bool(success),
            "crashed": bool(crashed),
            "out_of_bounds": bool(out_of_bounds),
            "awake": bool(awake),
            "theta_error_rad": float(ctrl_dbg.get("theta_error_rad", wrap_to_pi(theta_ref - float(full_state[4])))),
            "theta_error_deg": float(ctrl_dbg.get("theta_error_deg", math.degrees(wrap_to_pi(theta_ref - float(full_state[4]))))),
            "omega_raw": float(ctrl_dbg.get("omega_raw", omega_raw)),
            "effort": float(ctrl_dbg.get("effort", np.nan)),
            "kp": float(ctrl_dbg.get("kp", np.nan)),
            "kd": float(ctrl_dbg.get("kd", np.nan)),
            "wn": float(ctrl_dbg.get("wn", np.nan)),
            "zeta": float(ctrl_dbg.get("zeta", np.nan)),
            "theta_model_rad": float(ctrl_dbg.get("theta_model_rad", np.nan)),
            "theta_model_deg": float(ctrl_dbg.get("theta_model_deg", np.nan)),
            "omega_model": float(ctrl_dbg.get("omega_model", np.nan)),
            "e": float(ctrl_dbg.get("e", ctrl_dbg.get("theta_error_rad", np.nan))),
            "e_omega": float(ctrl_dbg.get("e_omega", np.nan)),
            "s": float(ctrl_dbg.get("s", np.nan)),
            "v_unsat": float(ctrl_dbg.get("v_unsat", np.nan)),
            "kp_hat": float(ctrl_dbg.get("kp_hat", ctrl_dbg.get("kp", np.nan))),
            "kd_hat": float(ctrl_dbg.get("kd_hat", ctrl_dbg.get("kd", np.nan))),
            "d_hat": float(ctrl_dbg.get("d_hat", np.nan)),
            "adaptation_active": float(bool(ctrl_dbg.get("adaptation_active", False))),
            "controller_side_gain": float(plant_dbg.get("side_gain", np.nan)),
            "controller_side_bias": float(plant_dbg.get("side_bias", np.nan)),
            "controller_side_delay_steps": float(plant_dbg.get("side_delay_steps", np.nan)),
            "shapes": shapes,
        }
        traj.append(row)

        ep_return += float(reward)
        ep_cost += cost
        full_obs = next_full_obs
        step_idx += 1
        done = bool(terminated or truncated)

    return RolloutResult(
        trajectory=traj,
        terrain=terrain,
        return_value=float(ep_return),
        cost=float(ep_cost),
        length=len(traj),
        success=bool(traj[-1]["success"] if traj else False),
        crashed=bool(traj[-1]["crashed"] if traj else False),
    )


# -----------------------------
# Metrics / summaries
# -----------------------------

def first_leg_contact_step(traj: List[Dict[str, Any]]) -> Optional[int]:
    for row in traj:
        if row["leg_left"] > 0.5 or row["leg_right"] > 0.5:
            return int(row["step"])
    return None


def first_side_action_step(traj: List[Dict[str, Any]], threshold: float = 1e-6) -> Optional[int]:
    for row in traj:
        if abs(float(row.get("side_cmd", 0.0))) > threshold:
            return int(row["step"])
    return None


def compare_trajectories(red: RolloutResult, trn: RolloutResult) -> Dict[str, Any]:
    rtraj = red.trajectory
    ttraj = trn.trajectory
    n = min(len(rtraj), len(ttraj))
    rc = first_leg_contact_step(rtraj)
    tc = first_leg_contact_step(ttraj)
    contact_cut = n
    candidates = [x for x in [rc, tc] if x is not None]
    if candidates:
        contact_cut = min(min(candidates) + 1, n)
    if contact_cut < 1:
        contact_cut = 1

    def stack(key: str, cut: int):
        r = np.array([row[key] for row in rtraj[:cut]], dtype=np.float64)
        t = np.array([row[key] for row in ttraj[:cut]], dtype=np.float64)
        return r, t

    metrics: Dict[str, Any] = {}
    labels = ["x", "y", "vx", "vy", "theta_rad"]
    for name in labels:
        r, t = stack(name, contact_cut)
        diff = t - r
        metrics[f"precontact_rms_{name}"] = float(np.sqrt(np.mean(np.square(diff))))
        metrics[f"precontact_maxabs_{name}"] = float(np.max(np.abs(diff)))

    metrics["reduced_first_contact_step"] = rc
    metrics["transfer_first_contact_step"] = tc
    metrics["transfer_first_side_action_step"] = first_side_action_step(ttraj)
    metrics["aligned_steps"] = n
    metrics["precontact_compare_steps"] = contact_cut
    metrics["reduced_return"] = float(red.return_value)
    metrics["transfer_return"] = float(trn.return_value)
    metrics["reduced_cost"] = float(red.cost)
    metrics["transfer_cost"] = float(trn.cost)
    metrics["reduced_success"] = bool(red.success)
    metrics["transfer_success"] = bool(trn.success)
    metrics["reduced_crashed"] = bool(red.crashed)
    metrics["transfer_crashed"] = bool(trn.crashed)
    return metrics


def rollout_obs_summary(traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    obs_cols = [k for k in traj[0].keys() if k.startswith("obs")]
    out: Dict[str, Any] = {}
    for col in obs_cols:
        vals = np.asarray([row[col] for row in traj if not np.isnan(row[col])], dtype=np.float64)
        if vals.size == 0:
            out[f"{col}_mean"] = np.nan
            out[f"{col}_std"] = np.nan
        else:
            out[f"{col}_mean"] = float(np.mean(vals))
            out[f"{col}_std"] = float(np.std(vals, ddof=0))
    return out


def aggregate_episode_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = [
        "reduced_return", "transfer_return", "reduced_cost", "transfer_cost",
        "reduced_length", "transfer_length",
        "precontact_rms_x", "precontact_rms_y", "precontact_rms_vx", "precontact_rms_vy", "precontact_rms_theta_rad",
        "precontact_maxabs_x", "precontact_maxabs_y", "precontact_maxabs_vx", "precontact_maxabs_vy", "precontact_maxabs_theta_rad",
    ]
    out: Dict[str, Any] = {"num_inits": len(rows)}
    for key in numeric_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=0))

    out["reduced_success_rate"] = float(np.mean([1.0 if r["reduced_success"] else 0.0 for r in rows]))
    out["transfer_success_rate"] = float(np.mean([1.0 if r["transfer_success"] else 0.0 for r in rows]))
    out["reduced_crash_rate"] = float(np.mean([1.0 if r["reduced_crashed"] else 0.0 for r in rows]))
    out["transfer_crash_rate"] = float(np.mean([1.0 if r["transfer_crashed"] else 0.0 for r in rows]))

    out["success_pct_of_reduced"] = safe_ratio_pct(out["transfer_success_rate"], out["reduced_success_rate"])
    out["return_pct_of_reduced"] = safe_ratio_pct(out["transfer_return_mean"], out["reduced_return_mean"])
    out["cost_efficiency_pct_vs_reduced"] = safe_ratio_pct(out["reduced_cost_mean"], out["transfer_cost_mean"])
    out["crash_efficiency_pct_vs_reduced"] = safe_ratio_pct(out["reduced_crash_rate"], out["transfer_crash_rate"])
    out["length_efficiency_pct_vs_reduced"] = safe_ratio_pct(out["reduced_length_mean"], out["transfer_length_mean"])
    return out


# -----------------------------
# Saving trajectories / plots
# -----------------------------

def save_trajectory_csv(traj: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in traj:
        slim = {k: v for k, v in row.items() if k != "shapes"}
        rows.append(floatify_dict(slim))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_episode_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([floatify_dict(r) for r in rows])


def plot_xy_trajectory(red: RolloutResult, trn: RolloutResult, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    rx = [row["x"] for row in red.trajectory]
    ry = [row["y"] for row in red.trajectory]
    tx = [row["x"] for row in trn.trajectory]
    ty = [row["y"] for row in trn.trajectory]
    plt.plot(rx, ry, label="Reduced")
    plt.plot(tx, ty, label="Transfer")
    plt.scatter([rx[0], tx[0]], [ry[0], ty[0]], marker="o")
    plt.scatter([rx[-1], tx[-1]], [ry[-1], ty[-1]], marker="x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_actions(red: RolloutResult, trn: RolloutResult, out_path: Path, title: str) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    rsteps = [row["step"] for row in red.trajectory]
    tsteps = [row["step"] for row in trn.trajectory]
    axs[0].plot(rsteps, [row["action_main"] for row in red.trajectory], label="Reduced")
    axs[0].plot(tsteps, [row["action_main"] for row in trn.trajectory], label="Transfer")
    axs[0].set_ylabel("main")
    axs[1].plot(rsteps, [row["theta_ref_deg"] for row in red.trajectory], label="Reduced")
    axs[1].plot(tsteps, [row["theta_ref_deg"] for row in trn.trajectory], label="Transfer")
    axs[1].set_ylabel("theta_ref [deg]")
    axs[2].plot(rsteps, [row["side_cmd"] for row in red.trajectory], label="Reduced")
    axs[2].plot(tsteps, [row["side_cmd"] for row in trn.trajectory], label="Transfer")
    axs[2].set_ylabel("side_cmd")
    axs[2].set_xlabel("step")
    for ax in axs:
        ax.grid(True, alpha=0.3)
    axs[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tracking_error(trn: RolloutResult, out_path: Path, title: str) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    steps = [row["step"] for row in trn.trajectory]
    axs[0].plot(steps, [row["theta_error_deg"] for row in trn.trajectory])
    axs[0].set_ylabel("theta error [deg]")
    axs[1].plot(steps, [row["omega_raw"] for row in trn.trajectory])
    axs[1].set_ylabel("omega [rad/s]")
    axs[2].plot(steps, [row["side_cmd"] for row in trn.trajectory])
    axs[2].set_ylabel("side_cmd")
    axs[2].set_xlabel("step")
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_observation_stats(summary_rows: List[Dict[str, Any]], out_path: Path, title: str, prefix: str) -> None:
    obs_names = [f"obs{i}" for i in range(6)]
    means = []
    stds = []
    labels = []
    for obs in obs_names:
        key_m = f"{prefix}_{obs}_mean"
        key_s = f"{prefix}_{obs}_std"
        if key_m in summary_rows[0]:
            labels.append(obs)
            means.append(np.mean([r[key_m] for r in summary_rows]))
            stds.append(np.mean([r[key_s] for r in summary_rows]))
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_summary_bars(episode_rows: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    metrics = [
        ("reduced_return", "transfer_return", "Return"),
        ("reduced_cost", "transfer_cost", "Cost"),
        ("reduced_length", "transfer_length", "Length"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (kr, kt, lab) in zip(axs, metrics):
        rvals = np.asarray([r[kr] for r in episode_rows], dtype=np.float64)
        tvals = np.asarray([r[kt] for r in episode_rows], dtype=np.float64)
        ax.bar([0, 1], [np.mean(rvals), np.mean(tvals)], yerr=[np.std(rvals), np.std(tvals)], capsize=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Reduced", "Transfer"])
        ax.set_title(lab)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_observation_summary_csv(summary_rows: List[Dict[str, Any]], path: Path) -> None:
    if not summary_rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows([floatify_dict(r) for r in summary_rows])




def _stack_rollout_key(rollouts: List[RolloutResult], key: str, max_len: Optional[int] = None) -> np.ndarray:
    if max_len is None:
        max_len = max(len(r.trajectory) for r in rollouts) if rollouts else 0
    arr = np.full((len(rollouts), max_len), np.nan, dtype=np.float64)
    for i, rollout in enumerate(rollouts):
        vals = [row.get(key, np.nan) for row in rollout.trajectory]
        if len(vals) > 0:
            arr[i, :len(vals)] = np.asarray(vals, dtype=np.float64)
    return arr


def _nanmeanstd(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def plot_xy_aggregate(red_rollouts: List[RolloutResult], trn_rollouts: List[RolloutResult], out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1], sharex=ax_x)

    for r in red_rollouts:
        ax_xy.plot([row['x'] for row in r.trajectory], [row['y'] for row in r.trajectory], alpha=0.12, linewidth=1.0)
    for t in trn_rollouts:
        ax_xy.plot([row['x'] for row in t.trajectory], [row['y'] for row in t.trajectory], alpha=0.12, linewidth=1.0)

    max_len = 0
    if red_rollouts:
        max_len = max(max_len, max(len(r.trajectory) for r in red_rollouts))
    if trn_rollouts:
        max_len = max(max_len, max(len(t.trajectory) for t in trn_rollouts))

    red_x = _stack_rollout_key(red_rollouts, 'x', max_len=max_len)
    red_y = _stack_rollout_key(red_rollouts, 'y', max_len=max_len)
    trn_x = _stack_rollout_key(trn_rollouts, 'x', max_len=max_len)
    trn_y = _stack_rollout_key(trn_rollouts, 'y', max_len=max_len)

    mx_r, sx_r = _nanmeanstd(red_x)
    my_r, sy_r = _nanmeanstd(red_y)
    mx_t, sx_t = _nanmeanstd(trn_x)
    my_t, sy_t = _nanmeanstd(trn_y)

    ax_xy.plot(mx_r, my_r, linewidth=2.5, label='Reduced mean')
    ax_xy.plot(mx_t, my_t, linewidth=2.5, label='Transfer mean')
    if np.any(~np.isnan(mx_r)) and np.any(~np.isnan(my_r)) and np.any(~np.isnan(mx_t)) and np.any(~np.isnan(my_t)):
        ax_xy.scatter([mx_r[0], mx_t[0]], [my_r[0], my_t[0]], marker='o', s=35)
        ax_xy.scatter([mx_r[~np.isnan(mx_r)][-1], mx_t[~np.isnan(mx_t)][-1]], [my_r[~np.isnan(my_r)][-1], my_t[~np.isnan(my_t)][-1]], marker='x', s=40)
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    ax_xy.set_title('2D trajectories: all runs faint, mean bold')
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend()

    steps = np.arange(max_len)
    ax_x.plot(steps, mx_r, label='Reduced mean')
    ax_x.fill_between(steps, mx_r - sx_r, mx_r + sx_r, alpha=0.2)
    ax_x.plot(steps, mx_t, label='Transfer mean')
    ax_x.fill_between(steps, mx_t - sx_t, mx_t + sx_t, alpha=0.2)
    ax_x.set_ylabel('x')
    ax_x.set_title('x over step: mean ± std')
    ax_x.grid(True, alpha=0.3)
    ax_x.legend()

    ax_y.plot(steps, my_r, label='Reduced mean')
    ax_y.fill_between(steps, my_r - sy_r, my_r + sy_r, alpha=0.2)
    ax_y.plot(steps, my_t, label='Transfer mean')
    ax_y.fill_between(steps, my_t - sy_t, my_t + sy_t, alpha=0.2)
    ax_y.set_xlabel('step')
    ax_y.set_ylabel('y')
    ax_y.set_title('y over step: mean ± std')
    ax_y.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_timeseries_mean_std(red_rollouts: List[RolloutResult], trn_rollouts: List[RolloutResult], specs: List[Tuple[str, str]], out_path: Path, title: str) -> None:
    n = len(specs)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    if n == 1:
        axs = [axs]
    max_len = max(max(len(r.trajectory) for r in red_rollouts), max(len(t.trajectory) for t in trn_rollouts))
    steps = np.arange(max_len)
    for ax, (key, label) in zip(axs, specs):
        red_arr = _stack_rollout_key(red_rollouts, key, max_len=max_len)
        trn_arr = _stack_rollout_key(trn_rollouts, key, max_len=max_len)
        mr, sr = _nanmeanstd(red_arr)
        mt, st = _nanmeanstd(trn_arr)
        ax.plot(steps, mr, label='Reduced')
        ax.fill_between(steps, mr - sr, mr + sr, alpha=0.2)
        ax.plot(steps, mt, label='Transfer')
        ax.fill_between(steps, mt - st, mt + st, alpha=0.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axs[0].legend()
    axs[-1].set_xlabel('step')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)





def plot_single_rollouts_mean_std(rollouts: List[RolloutResult], specs: List[Tuple[str, str]], out_path: Path, title: str) -> None:
    if not rollouts:
        return
    n = len(specs)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    if n == 1:
        axs = [axs]
    max_len = max(len(r.trajectory) for r in rollouts)
    steps = np.arange(max_len)
    for ax, (key, label) in zip(axs, specs):
        arr = _stack_rollout_key(rollouts, key, max_len=max_len)
        mean_vals, std_vals = _nanmeanstd(arr)
        ax.plot(steps, mean_vals)
        ax.fill_between(steps, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axs[-1].set_xlabel('step')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_obs_mean_std_combined(summary_rows: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    obs_names = [f'obs{i}' for i in range(6)]
    labels, r_means, r_stds, t_means, t_stds = [], [], [], [], []
    for obs in obs_names:
        rk_m, rk_s = f'reduced_{obs}_mean', f'reduced_{obs}_std'
        tk_m, tk_s = f'transfer_{obs}_mean', f'transfer_{obs}_std'
        if rk_m in summary_rows[0] and tk_m in summary_rows[0]:
            labels.append(obs)
            r_means.append(np.mean([r[rk_m] for r in summary_rows]))
            r_stds.append(np.mean([r[rk_s] for r in summary_rows]))
            t_means.append(np.mean([r[tk_m] for r in summary_rows]))
            t_stds.append(np.mean([r[tk_s] for r in summary_rows]))
    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width/2, r_means, width, yerr=r_stds, capsize=4, label='Reduced')
    plt.bar(x + width/2, t_means, width, yerr=t_stds, capsize=4, label='Transfer')
    plt.xticks(x, labels)
    plt.ylabel('value')
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_final_conclusion(episode_rows: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        ('reduced_return', 'transfer_return', 'Return'),
        ('reduced_cost', 'transfer_cost', 'Cost'),
        ('reduced_length', 'transfer_length', 'Length'),
        ('precontact_rms_x', 'precontact_rms_y', 'Precontact RMS x/y'),
        ('precontact_rms_vx', 'precontact_rms_vy', 'Precontact RMS vx/vy'),
        ('precontact_rms_theta_rad', None, 'Precontact RMS theta'),
    ]
    for ax, panel in zip(axs.ravel(), panels):
        kr = panel[0]
        kt = panel[1]
        title_panel = panel[2]
        if kt is not None and title_panel.startswith('Precontact'):
            vals1 = np.asarray([r[kr] for r in episode_rows], dtype=np.float64)
            vals2 = np.asarray([r[kt] for r in episode_rows], dtype=np.float64)
            ax.bar([0,1], [np.mean(vals1), np.mean(vals2)], yerr=[np.std(vals1), np.std(vals2)], capsize=4)
            ax.set_xticks([0,1])
            ax.set_xticklabels([kr.split('_')[-1], kt.split('_')[-1]])
        elif kt is not None:
            rvals = np.asarray([r[kr] for r in episode_rows], dtype=np.float64)
            tvals = np.asarray([r[kt] for r in episode_rows], dtype=np.float64)
            ax.bar([0,1], [np.mean(rvals), np.mean(tvals)], yerr=[np.std(rvals), np.std(tvals)], capsize=4)
            ax.set_xticks([0,1])
            ax.set_xticklabels(['Reduced', 'Transfer'])
        else:
            vals = np.asarray([r[kr] for r in episode_rows], dtype=np.float64)
            ax.bar([0], [np.mean(vals)], yerr=[np.std(vals)], capsize=4)
            ax.set_xticks([0])
            ax.set_xticklabels(['theta'])
        ax.set_title(title_panel)
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# -----------------------------
# Overlay replay
# -----------------------------

def world_to_screen(p):
    return (int(round(p[0] * SCALE)), int(round(VIEWPORT_H - p[1] * SCALE)))


def draw_background(surface, terrain):
    surface.fill((255, 255, 255))
    if gfxdraw is None:
        raise RuntimeError("pygame gfxdraw unavailable")
    for poly in terrain["sky_polys"]:
        pts = [world_to_screen(p) for p in poly]
        pygame.draw.polygon(surface, (0, 0, 0), pts)
        gfxdraw.aapolygon(surface, pts, (0, 0, 0))
    for x in [terrain["helipad_x1"], terrain["helipad_x2"]]:
        xpix = int(round(x * SCALE))
        flagy1 = int(round(VIEWPORT_H - terrain["helipad_y"] * SCALE))
        flagy2 = flagy1 - 50
        pygame.draw.line(surface, (255, 255, 255), (xpix, flagy1), (xpix, flagy2), 1)
        flag_pts = [(xpix, flagy2), (xpix, flagy2 + 10), (xpix + 25, flagy2 + 5)]
        pygame.draw.polygon(surface, (204, 204, 0), flag_pts)
        gfxdraw.aapolygon(surface, flag_pts, (204, 204, 0))


def draw_shapes(surface, shapes: Dict[str, Any], fill_color, outline_color, alpha: int = 110):
    overlay = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)
    fill = (*fill_color, alpha)
    for group in ["lander", "leg0", "leg1"]:
        for poly in shapes[group]:
            pts = [world_to_screen(p) for p in poly]
            pygame.draw.polygon(overlay, fill, pts)
            gfxdraw.aapolygon(overlay, pts, outline_color)
    surface.blit(overlay, (0, 0))


def draw_text(surface, text: str, pos: Tuple[int, int], font, color=(20, 20, 20)):
    surf = font.render(text, True, color)
    surface.blit(surf, pos)


def replay_overlay(red: RolloutResult, trn: RolloutResult, title: str, realtime: bool, save_gif_path: Optional[Path] = None):
    if pygame is None:
        raise RuntimeError("pygame is required for overlay replay")

    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    n = max(len(red.trajectory), len(trn.trajectory))
    terrain = red.terrain
    frames = []

    red_color = (50, 120, 255)
    red_outline = (20, 70, 180)
    tr_color = (255, 80, 80)
    tr_outline = (180, 30, 30)

    for i in range(n):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        draw_background(surf, terrain)

        if i < len(red.trajectory):
            draw_shapes(surf, red.trajectory[i]["shapes"], red_color, red_outline, alpha=100)
        if i < len(trn.trajectory):
            draw_shapes(surf, trn.trajectory[i]["shapes"], tr_color, tr_outline, alpha=100)

        draw_text(surf, "Blue = reduced rollout", (10, 10), font, red_outline)
        draw_text(surf, "Red = transfer rollout", (10, 30), font, tr_outline)
        draw_text(surf, f"frame={i+1}/{n}", (10, 52), font)
        if i < len(red.trajectory):
            rr = red.trajectory[i]
            draw_text(surf, f"R: x={rr['x']: .3f} y={rr['y']: .3f} vy={rr['vy']: .3f} th*={rr['theta_ref_deg']: .2f}", (10, 74), font, red_outline)
        if i < len(trn.trajectory):
            tt = trn.trajectory[i]
            draw_text(surf, f"T: x={tt['x']: .3f} y={tt['y']: .3f} vy={tt['vy']: .3f} th*={tt['theta_ref_deg']: .2f} side={tt['side_cmd']: .2f}", (10, 94), font, tr_outline)

        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if save_gif_path is not None:
            arr = np.transpose(np.array(pygame.surfarray.pixels3d(screen)).copy(), (1, 0, 2))
            frames.append(arr)

        if realtime:
            clock.tick(FPS)

    if save_gif_path is not None:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(str(save_gif_path), frames, fps=FPS)
            print(f"Saved overlay GIF: {save_gif_path}")
        except Exception as e:
            print(f"Could not save GIF ({e}). Install imageio if needed.")

    hold_until = time.time() + 1.0
    while time.time() < hold_until:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                hold_until = 0
        clock.tick(30)
    pygame.quit()




def plot_adaptive_internal(transfer: RolloutResult, out_path: Path, title: str) -> None:
    if not transfer.trajectory:
        return
    steps = [row["step"] for row in transfer.trajectory]
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    axs[0].plot(steps, [row.get("kp_hat", np.nan) for row in transfer.trajectory], label="kp_hat")
    axs[0].plot(steps, [row.get("kd_hat", np.nan) for row in transfer.trajectory], label="kd_hat")
    axs[0].set_ylabel("adaptive gains")
    axs[0].legend()

    axs[1].plot(steps, [row.get("d_hat", np.nan) for row in transfer.trajectory], label="d_hat")
    axs[1].set_ylabel("bias estimate")
    axs[1].legend()

    axs[2].plot(steps, [row.get("e", np.nan) for row in transfer.trajectory], label="e")
    axs[2].plot(steps, [row.get("e_omega", np.nan) for row in transfer.trajectory], label="e_omega")
    axs[2].plot(steps, [row.get("s", np.nan) for row in transfer.trajectory], label="s")
    axs[2].set_ylabel("tracking vars")
    axs[2].legend()

    axs[3].plot(steps, [row.get("v_unsat", np.nan) for row in transfer.trajectory], label="u unsat")
    axs[3].plot(steps, [row.get("side_cmd", np.nan) for row in transfer.trajectory], label="controller side")
    axs[3].plot(steps, [row.get("plant_side_cmd", np.nan) for row in transfer.trajectory], label="plant side")
    axs[3].set_ylabel("control")
    axs[3].set_xlabel("step")
    axs[3].legend()

    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_controller_comparison_bars(controller_summaries: Dict[str, Dict[str, Any]], out_path: Path, title: str) -> None:
    names = list(controller_summaries.keys())
    if not names:
        return
    metrics = [
        ("transfer_return_mean", "Return"),
        ("transfer_cost_mean", "Cost"),
        ("transfer_success_rate", "Success rate"),
        ("precontact_rms_theta_rad_mean", "Precontact RMS theta"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    x = np.arange(len(names))
    for ax, (key, label) in zip(axs, metrics):
        vals = [controller_summaries[n].get(key, np.nan) for n in names]
        err_key = key.replace("_mean", "_std") if key.endswith("_mean") else None
        errs = [controller_summaries[n].get(err_key, 0.0) if err_key is not None else 0.0 for n in names]
        ax.bar(x, vals, yerr=errs, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)



def apply_regime_preset(args):
    regime = str(getattr(args, "regime", "custom")).strip().lower()
    if regime in {"", "custom", "nominal"}:
        return args
    if regime == "light_mismatch":
        args.transfer_side_gain = 0.90
        args.transfer_side_bias = 0.00
        args.transfer_side_delay_steps = 0
        args.eval_enable_wind = False
    elif regime == "delay1":
        args.transfer_side_gain = 1.00
        args.transfer_side_bias = 0.00
        args.transfer_side_delay_steps = 1
        args.eval_enable_wind = False
    elif regime == "biased":
        args.transfer_side_gain = 0.90
        args.transfer_side_bias = 0.05
        args.transfer_side_delay_steps = 0
        args.eval_enable_wind = False
    elif regime == "windy":
        args.transfer_side_gain = 1.00
        args.transfer_side_bias = 0.00
        args.transfer_side_delay_steps = 0
        args.eval_enable_wind = True
        if getattr(args, 'eval_wind_power', None) is not None:
            args.eval_wind_power = max(float(args.eval_wind_power), 15.0)
        if getattr(args, 'eval_turbulence_power', None) is not None:
            args.eval_turbulence_power = max(float(args.eval_turbulence_power), 1.5)
    elif regime == "hard":
        args.transfer_side_gain = 0.75
        args.transfer_side_bias = 0.05
        args.transfer_side_delay_steps = 1
        args.eval_enable_wind = True
        if getattr(args, 'eval_wind_power', None) is not None:
            args.eval_wind_power = max(float(args.eval_wind_power), 15.0)
        if getattr(args, 'eval_turbulence_power', None) is not None:
            args.eval_turbulence_power = max(float(args.eval_turbulence_power), 1.5)
    return args

# -----------------------------
# Main evaluation loop
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Bank evaluation of reduced vs transfer rollouts with PD and adaptive deployment controllers")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inits", type=int, default=10)
    p.add_argument("--seed_list_json", type=str, default="", help="Optional JSON file containing an explicit list of seeds to evaluate.")
    p.add_argument("--seed_list_txt", type=str, default="", help="Optional TXT file containing one seed per line.")
    p.add_argument("--deterministic", type=str2bool, default=True)
    p.add_argument("--controllers", type=str, nargs="+", default=["pd", "adaptive_bias", "adaptive_gain"], choices=["pd", "adaptive_bias", "adaptive_gain"])
    p.add_argument("--overlay_controller", type=str, default="", choices=["", "pd", "adaptive_bias", "adaptive_gain"])

    # Shared controller interface / historical PD settings
    p.add_argument("--wn", type=float, default=5.0)
    p.add_argument("--zeta", type=float, default=0.7)
    p.add_argument("--deadband", type=float, default=0.05)
    p.add_argument("--min_side_action", type=float, default=0.55)
    p.add_argument("--max_side_action", type=float, default=1.0)
    p.add_argument("--side_sign", type=int, default=0, help="0=auto detect, +1 or -1 to force")
    p.add_argument("--zero_side_on_contact", type=str2bool, default=False)

    # Adaptive-controller settings
    p.add_argument("--adaptive_ref_wn", type=float, default=6.0)
    p.add_argument("--adaptive_ref_zeta", type=float, default=1.0)
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

    # Transfer-only actuator mismatch to make adaptation meaningful under explicit uncertainty
    p.add_argument("--transfer_side_gain", type=float, default=1.0)
    p.add_argument("--transfer_side_bias", type=float, default=0.0)
    p.add_argument("--transfer_side_delay_steps", type=int, default=0)
    p.add_argument("--transfer_main_gain", type=float, default=1.0)
    p.add_argument("--transfer_main_bias", type=float, default=0.0)

    # Episode / environment / plots
    p.add_argument("--max_transfer_steps", type=int, default=1000)
    p.add_argument("--regime", type=str, default="custom", choices=["custom","nominal","light_mismatch","delay1","biased","windy","hard"], help="Preset transfer regime; custom leaves manual flags unchanged.")
    p.add_argument("--eval_enable_wind", type=str2bool, default=False)
    p.add_argument("--eval_wind_power", type=float, default=15.0)
    p.add_argument("--eval_turbulence_power", type=float, default=1.5)
    p.add_argument("--render_overlay", type=str2bool, default=False)
    p.add_argument("--realtime", type=str2bool, default=True)
    p.add_argument("--save_gif", type=str2bool, default=True)
    p.add_argument("--num_gifs", type=int, default=10, help="How many initializations to save/render as overlay GIFs")
    p.add_argument("--output_dir", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    args = apply_regime_preset(args)
    controller_names: List[str] = []
    for name in args.controllers:
        key = str(name).strip().lower()
        if key not in controller_names:
            controller_names.append(key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg = build_cfg_from_checkpoint(ckpt, override_seed=args.seed)
    cfg.enable_wind = bool(args.eval_enable_wind)
    cfg.wind_power = float(args.eval_wind_power)
    cfg.turbulence_power = float(args.eval_turbulence_power)
    set_seed(cfg.seed)

    red_env = make_env(cfg, render_mode=None, apply_training_reward_shaping=False)()
    obs_dim = red_env.observation_space.shape[0]
    act_dim = red_env.action_space.shape[0]
    agent = build_agent_from_checkpoint(cfg, ckpt, device, obs_dim, act_dim)
    obs_rms = build_obs_rms_from_checkpoint(ckpt, obs_dim)

    full_env = TimeLimit(
        LunarLander(
            render_mode=None,
            continuous=True,
            gravity=cfg.gravity,
            enable_wind=cfg.enable_wind,
            wind_power=cfg.wind_power,
            turbulence_power=cfg.turbulence_power,
        ),
        max_episode_steps=args.max_transfer_steps,
    )

    side_sign = detect_side_sign(cfg, cfg.seed) if args.side_sign == 0 else int(np.sign(args.side_sign))
    controllers = {name: build_transfer_controller(name, cfg, args, side_sign) for name in controller_names}

    explicit_seed_list = load_seed_list(args.seed_list_json, args.seed_list_txt)
    if explicit_seed_list is not None:
        eval_seed_list = [int(s) for s in explicit_seed_list]
    else:
        eval_seed_list = [int(cfg.seed + init_idx) for init_idx in range(args.num_inits)]

    ckpt_path = Path(args.checkpoint)
    ctrl_tag = shorten_tag("-".join(controller_names), max_len=32)
    timestamp_tag = time.strftime('%Y%m%d_%H%M%S')
    if explicit_seed_list is not None:
        seed_src = Path(args.seed_list_json or args.seed_list_txt)
        default_out_root = seed_src.parent
        seed_tag = shorten_tag(seed_src.stem, max_len=40)
        default_out = default_out_root / f"bank_eval_{ctrl_tag}_{seed_tag}_{timestamp_tag}"
    else:
        ckpt_tag = shorten_tag(ckpt_path.stem, max_len=28)
        default_out = ckpt_path.parent / f"bank_eval_{ctrl_tag}_{ckpt_tag}_n{len(eval_seed_list)}_{timestamp_tag}"
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else default_out)
    traj_dir = ensure_dir(out_dir / "trajectories")
    plots_dir = ensure_dir(out_dir / "plots")

    overlay_controller = str(args.overlay_controller).strip().lower()
    if not overlay_controller:
        overlay_controller = "adaptive_gain" if "adaptive_gain" in controller_names else controller_names[0]
    if overlay_controller not in controller_names:
        overlay_controller = controller_names[0]

    print("\n[Bank comparison setup loaded]")
    print(f"Checkpoint                 : {args.checkpoint}")
    print(f"Epoch                      : {ckpt.get('epoch', 'N/A')}")
    print(f"Global step                : {ckpt.get('global_step', 'N/A')}")
    print(f"Base seed                  : {cfg.seed}")
    print(f"Num initializations        : {len(eval_seed_list)}")
    print(f"Explicit seed list         : {bool(explicit_seed_list)}")
    print(f"Controllers                : {controller_names}")
    print(f"Overlay controller         : {overlay_controller}")
    print(f"Num GIF overlays           : {min(max(0, int(args.num_gifs)), int(len(eval_seed_list)))}")
    print(f"Theta limit [deg]          : {cfg.theta_limit_deg}")
    print(f"Use obs norm               : {cfg.use_obs_norm and obs_rms is not None}")
    print(f"Leg contacts in obs        : {cfg.include_leg_contacts_in_obs}")
    print(f"Controller deadband        : {args.deadband}")
    print(f"Min/Max side action        : {args.min_side_action}/{args.max_side_action}")
    print(f"Auto/selected sign         : {side_sign}")
    print(f"Zero side on contact       : {args.zero_side_on_contact}")
    print(f"Adaptive ref wn/zeta       : {args.adaptive_ref_wn}/{args.adaptive_ref_zeta}")
    print(f"Adaptive lambda_s          : {args.adaptive_lambda_s}")
    print(f"Adaptive gammas            : gp={args.adaptive_gamma_p} gd={args.adaptive_gamma_d} gb={args.adaptive_gamma_b}")
    print(f"Adaptive sigmas            : sp={args.adaptive_sigma_p} sd={args.adaptive_sigma_d} sb={args.adaptive_sigma_b}")
    print(f"Adaptive kp range          : [{args.adaptive_kp_min}, {args.adaptive_kp_max}]")
    print(f"Adaptive kd range          : [{args.adaptive_kd_min}, {args.adaptive_kd_max}]")
    print(f"Adaptive bias range        : [{args.adaptive_bias_min}, {args.adaptive_bias_max}]")
    print(f"Freeze adapt on contact    : {args.freeze_adaptation_on_contact}")
    print(f"Regime                    : {args.regime}")
    print(f"Transfer side gain/bias    : {args.transfer_side_gain}/{args.transfer_side_bias}")
    print(f"Transfer side delay steps  : {args.transfer_side_delay_steps}")
    print(f"Transfer main gain/bias    : {args.transfer_main_gain}/{args.transfer_main_bias}")
    print(f"Max transfer steps         : {args.max_transfer_steps}")
    if hasattr(cfg, "settle_mode_enabled"):
        print(f"Settle mode enabled        : {getattr(cfg, 'settle_mode_enabled', None)}")
        print(f"Settle force side off      : {getattr(cfg, 'settle_force_side_off', None)}")
        print(f"Settle enter/exit          : {getattr(cfg, 'settle_enter_steps', None)}/{getattr(cfg, 'settle_exit_steps', None)}")
    print(f"Eval wind enabled          : {cfg.enable_wind}")
    print(f"Eval wind power            : {cfg.wind_power}")
    print(f"Eval turbulence power      : {cfg.turbulence_power}")
    print(f"Output dir                 : {out_dir}")

    metadata = {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch", None),
        "global_step": ckpt.get("global_step", None),
        "base_seed": int(cfg.seed),
        "num_inits": int(len(eval_seed_list)),
        "seed_list_json": str(args.seed_list_json),
        "seed_list_txt": str(args.seed_list_txt),
        "explicit_seed_list": [int(s) for s in eval_seed_list],
        "theta_limit_deg": float(cfg.theta_limit_deg),
        "use_obs_norm": bool(cfg.use_obs_norm and obs_rms is not None),
        "include_leg_contacts_in_obs": bool(cfg.include_leg_contacts_in_obs),
        "controller_names": controller_names,
        "overlay_controller": overlay_controller,
        "controller_specs": {name: controller_summary_dict(controller, name) for name, controller in controllers.items()},
        "regime": str(args.regime),
        "transfer_side_gain": float(args.transfer_side_gain),
        "transfer_side_bias": float(args.transfer_side_bias),
        "transfer_side_delay_steps": int(args.transfer_side_delay_steps),
        "transfer_main_gain": float(args.transfer_main_gain),
        "transfer_main_bias": float(args.transfer_main_bias),
        "zero_side_on_contact": bool(args.zero_side_on_contact),
        "max_transfer_steps": int(args.max_transfer_steps),
        "settle_mode_enabled": bool(getattr(cfg, "settle_mode_enabled", False)),
        "settle_force_side_off": bool(getattr(cfg, "settle_force_side_off", False)),
        "settle_enter_steps": int(getattr(cfg, "settle_enter_steps", 0)),
        "settle_exit_steps": int(getattr(cfg, "settle_exit_steps", 0)),
    }
    save_json(metadata, out_dir / "metadata.json")

    reduced_episode_rows: List[Dict[str, Any]] = []
    reduced_rollouts: List[RolloutResult] = []
    transfer_rollouts_by_controller: Dict[str, List[RolloutResult]] = {name: [] for name in controller_names}
    controller_episode_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in controller_names}
    controller_obs_summary_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in controller_names}
    wide_episode_rows: List[Dict[str, Any]] = []
    gif_pairs: List[Tuple[int, int, RolloutResult, RolloutResult]] = []

    for init_idx, init_seed in enumerate(eval_seed_list):
        reduced = run_reduced_episode(red_env, agent, obs_rms, device, args.deterministic, init_seed)
        reduced_rollouts.append(reduced)
        save_trajectory_csv(reduced.trajectory, traj_dir / f"reduced_init{init_idx:03d}_seed{init_seed}.csv")

        wide_row: Dict[str, Any] = {
            "init_index": init_idx,
            "seed": init_seed,
            "reduced_return": reduced.return_value,
            "reduced_cost": reduced.cost,
            "reduced_length": reduced.length,
            "reduced_success": reduced.success,
            "reduced_crashed": reduced.crashed,
        }
        reduced_episode_rows.append({
            "init_index": init_idx,
            "seed": init_seed,
            "reduced_return": reduced.return_value,
            "reduced_cost": reduced.cost,
            "reduced_length": reduced.length,
            "reduced_success": reduced.success,
            "reduced_crashed": reduced.crashed,
        })

        for controller_name in controller_names:
            controller = controllers[controller_name]
            actuator_mismatch = TransferActuatorMismatch(
                side_gain=args.transfer_side_gain,
                side_bias=args.transfer_side_bias,
                side_delay_steps=args.transfer_side_delay_steps,
                main_gain=args.transfer_main_gain,
                main_bias=args.transfer_main_bias,
            )
            transfer = run_transfer_episode(
                full_env,
                agent,
                controller,
                actuator_mismatch,
                obs_rms,
                cfg.include_leg_contacts_in_obs,
                cfg.theta_limit_deg,
                cfg,
                device,
                args.deterministic,
                init_seed,
                args.max_transfer_steps,
            )
            transfer_rollouts_by_controller[controller_name].append(transfer)

            if controller_name == overlay_controller and len(gif_pairs) < max(0, int(args.num_gifs)):
                gif_pairs.append((init_idx, init_seed, reduced, transfer))

            metrics = compare_trajectories(reduced, transfer)
            row = {
                "init_index": init_idx,
                "seed": init_seed,
                "controller_name": controller_name,
                **controller_summary_dict(controller, controller_name),
                "reduced_return": reduced.return_value,
                "transfer_return": transfer.return_value,
                "reduced_cost": reduced.cost,
                "transfer_cost": transfer.cost,
                "reduced_length": reduced.length,
                "transfer_length": transfer.length,
                "reduced_success": reduced.success,
                "transfer_success": transfer.success,
                "reduced_crashed": reduced.crashed,
                "transfer_crashed": transfer.crashed,
                **metrics,
            }
            controller_episode_rows[controller_name].append(floatify_dict(row))

            wide_row.update({
                f"{controller_name}_return": transfer.return_value,
                f"{controller_name}_cost": transfer.cost,
                f"{controller_name}_length": transfer.length,
                f"{controller_name}_success": transfer.success,
                f"{controller_name}_crashed": transfer.crashed,
                f"{controller_name}_precontact_rms_x": metrics["precontact_rms_x"],
                f"{controller_name}_precontact_rms_y": metrics["precontact_rms_y"],
                f"{controller_name}_precontact_rms_vx": metrics["precontact_rms_vx"],
                f"{controller_name}_precontact_rms_vy": metrics["precontact_rms_vy"],
                f"{controller_name}_precontact_rms_theta_rad": metrics["precontact_rms_theta_rad"],
            })

            obs_summary = {
                "init_index": init_idx,
                "seed": init_seed,
                "controller_name": controller_name,
                **controller_summary_dict(controller, controller_name),
                **{f"reduced_{k}": v for k, v in rollout_obs_summary(reduced.trajectory).items()},
                **{f"transfer_{k}": v for k, v in rollout_obs_summary(transfer.trajectory).items()},
            }
            controller_obs_summary_rows[controller_name].append(floatify_dict(obs_summary))

            save_trajectory_csv(transfer.trajectory, traj_dir / f"{controller_name}_init{init_idx:03d}_seed{init_seed}.csv")
            per_title = f"ctrl={controller_name} | init={init_idx} seed={init_seed} | wn={args.wn:.3f}, zeta={args.zeta:.3f}"
            plot_xy_trajectory(reduced, transfer, plots_dir / f"xy_{controller_name}_init{init_idx:03d}.png", per_title)
            plot_actions(reduced, transfer, plots_dir / f"actions_{controller_name}_init{init_idx:03d}.png", per_title)
            plot_tracking_error(transfer, plots_dir / f"tracking_{controller_name}_init{init_idx:03d}.png", per_title)
            if controller_name_is_adaptive(controller_name):
                plot_adaptive_internal(transfer, plots_dir / f"adaptive_internal_{controller_name}_init{init_idx:03d}.png", per_title)

        wide_episode_rows.append(floatify_dict(wide_row))
        brief = " | ".join([f"{name}: ret={transfer_rollouts_by_controller[name][-1].return_value:.2f}, succ={transfer_rollouts_by_controller[name][-1].success}" for name in controller_names])
        print(f"[Init {init_idx+1:03d}/{len(eval_seed_list):03d}] seed={init_seed} reduced_ret={reduced.return_value:.2f} || {brief}")

    save_episode_summary_csv(wide_episode_rows, out_dir / "episode_metrics_wide.csv")

    controller_summaries: Dict[str, Dict[str, Any]] = {}
    for controller_name in controller_names:
        rows = controller_episode_rows[controller_name]
        obs_rows = controller_obs_summary_rows[controller_name]
        rollouts = transfer_rollouts_by_controller[controller_name]
        summary = aggregate_episode_rows(rows)
        spec = controller_summary_dict(controllers[controller_name], controller_name)
        summary.update(spec)
        controller_summaries[controller_name] = floatify_dict(summary)

        print(f"\n[Aggregate summary | {controller_name}]")
        for k, v in summary.items():
            print(f"{k:>32s} : {v}")

        save_episode_summary_csv(rows, out_dir / f"episode_metrics_{controller_name}.csv")
        save_json(floatify_dict(summary), out_dir / f"aggregate_summary_{controller_name}.json")
        save_observation_summary_csv(obs_rows, out_dir / f"observation_summary_{controller_name}.csv")

        plot_observation_stats(obs_rows, plots_dir / f"obs_stats_reduced_{controller_name}.png", f"Reduced obs mean/std | ctrl={controller_name}", "reduced")
        plot_observation_stats(obs_rows, plots_dir / f"obs_stats_transfer_{controller_name}.png", f"Transfer obs mean/std | ctrl={controller_name}", "transfer")
        plot_obs_mean_std_combined(obs_rows, plots_dir / f"obs_stats_combined_{controller_name}.png", f"Observation mean/std across runs | ctrl={controller_name}")
        plot_summary_bars(rows, plots_dir / f"aggregate_bars_{controller_name}.png", f"Aggregate metrics | ctrl={controller_name}")
        plot_xy_aggregate(reduced_rollouts, rollouts, plots_dir / f"aggregate_xy_mean_std_{controller_name}.png", f"2D trajectories across runs | ctrl={controller_name}")
        plot_timeseries_mean_std(reduced_rollouts, rollouts, [("x", "x"), ("y", "y"), ("vx", "vx"), ("vy", "vy"), ("theta_rad", "theta [rad]")], plots_dir / f"aggregate_state_mean_std_{controller_name}.png", f"State mean ± std across runs | ctrl={controller_name}")
        plot_timeseries_mean_std(reduced_rollouts, rollouts, [("action_main", "main"), ("theta_ref_deg", "theta_ref [deg]"), ("side_cmd", "controller side")], plots_dir / f"aggregate_action_mean_std_{controller_name}.png", f"Action mean ± std across runs | ctrl={controller_name}")
        plot_timeseries_mean_std(reduced_rollouts, rollouts, [("theta_error_deg", "theta error [deg]"), ("omega_raw", "omega [rad/s]"), ("effort", "control effort")], plots_dir / f"aggregate_tracking_mean_std_{controller_name}.png", f"Tracking mean ± std across runs | ctrl={controller_name}")
        plot_final_conclusion(rows, plots_dir / f"final_conclusion_summary_{controller_name}.png", f"Final summary across {len(rows)} runs | ctrl={controller_name}")
        if controller_name_is_adaptive(controller_name):
            plot_single_rollouts_mean_std(rollouts, [("kp_hat", "kp_hat"), ("kd_hat", "kd_hat"), ("d_hat", "d_hat"), ("v_unsat", "u unsat"), ("plant_side_cmd", "plant side")], plots_dir / f"aggregate_adaptive_internal_{controller_name}.png", f"Adaptive internals mean ± std | ctrl={controller_name}")

    save_json(controller_summaries, out_dir / "aggregate_summary_by_controller.json")
    plot_controller_comparison_bars(controller_summaries, plots_dir / "controller_comparison_bars.png", f"Controller comparison across {len(eval_seed_list)} runs")
    percentage_rows: List[Dict[str, Any]] = []
    for controller_name, summary in controller_summaries.items():
        percentage_rows.append(floatify_dict({
            "controller_name": controller_name,
            "success_pct_of_reduced": summary.get("success_pct_of_reduced", np.nan),
            "return_pct_of_reduced": summary.get("return_pct_of_reduced", np.nan),
            "cost_efficiency_pct_vs_reduced": summary.get("cost_efficiency_pct_vs_reduced", np.nan),
            "crash_efficiency_pct_vs_reduced": summary.get("crash_efficiency_pct_vs_reduced", np.nan),
            "length_efficiency_pct_vs_reduced": summary.get("length_efficiency_pct_vs_reduced", np.nan),
        }))
    if percentage_rows:
        save_episode_summary_csv(percentage_rows, out_dir / "performance_percentages_vs_reduced.csv")
        save_json({row["controller_name"]: {k: v for k, v in row.items() if k != "controller_name"} for row in percentage_rows}, out_dir / "performance_percentages_vs_reduced.json")


    if gif_pairs and (args.render_overlay or args.save_gif):
        for gif_idx, (init_idx, init_seed, reduced_gif, transfer_gif) in enumerate(gif_pairs, start=1):
            gif_path = plots_dir / f"overlay_{overlay_controller}_init{init_idx:03d}_seed{init_seed}.gif" if args.save_gif else None
            print(f"[Overlay {gif_idx}/{len(gif_pairs)}] ctrl={overlay_controller} init={init_idx} seed={init_seed}")
            replay_overlay(
                reduced_gif,
                transfer_gif,
                title=f"Reduced vs {overlay_controller} | init={init_idx} seed={init_seed}",
                realtime=args.realtime,
                save_gif_path=gif_path,
            )

    red_env.close()
    full_env.close()


if __name__ == "__main__":
    main()
