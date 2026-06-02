#!/usr/bin/env python3
"""
Train and evaluate a reduced-order Lunar Lander policy (Lander-R) with PPO or PPO-Lagrangian.

Design choices in this script:
- Reduced environment keeps the Box2D LunarLander rigid body, legs, contacts, and stock render.
- This is NOT a pure point-mass simulator. It is a reduced-observation / reduced-dynamics wrapper
  built on top of the full Box2D LunarLander.
- The default policy observation is [x, y, vx, vy, c_left, c_right], i.e., translational state plus
  the two leg-contact bits, so the agent is not blind to touchdown/contact events.
- The policy outputs two continuous commands in normalized space [-1, 1]^2:
    a[0]: main-engine command in stock continuous LunarLander units
          (-1..0 => off, 0..1 => 50%..100% throttle)
    a[1]: desired angle command theta* normalized to [-1, 1], mapped to [-theta_limit, theta_limit]
- Reduced dynamics are created by instantaneously setting the lander's attitude to theta*
  and zeroing the angular velocity, i.e. reduced training ignores theta dynamics.
- The same normalized PPO action used for theta* is also used directly for the reduced side-thruster
  translation path, but the side engine only fires under the official LunarLander continuous-action
  fire rule: no side thrust for |action[1]| <= 0.5, otherwise thrust magnitude is clipped to [0.5, 1.0].
- Settle mode is kept only as a light-touch helper so Box2D can still reach the official
  success condition based on `not self.lander.awake` near touchdown.
- Unlike the previous settle logic, touchdown gating is aligned more closely with the
  official LunarLander sleep behavior, but this training-oriented variant also adds
  small discovery aids so PPO does not get trapped in hover / soft-touch local optima.
  Specifically, training can apply: (1) a penalty for time-limit endings that are not
  real sleep-based landings, (2) a one-time quiet-touchdown bonus on first both-leg
  contact, and (3) an optional one-time bonus when settle mode first engages.
- Best-checkpoint selection is success-first, then return, so a true landing policy is
  preferred over a higher-return hover policy.

Requirements
------------
Python 3.10+
Install packages:
    pip install numpy matplotlib torch gymnasium[box2d] pygame

Usage examples
--------------
Train from scratch (cold start):
    python train_reduced_lander.py --mode train

Unconstrained PPO (lambda fixed to 0):
    python train_reduced_lander.py --mode train --use_safety 0 --lambda_init 0 --lambda_lr 0

Warm start / fine-tune from a checkpoint:
    python train_reduced_lander.py --mode train --warm_start_path "C:\\...\\model.pt"

Evaluate a saved policy with render:
    python train_reduced_lander.py --mode eval --checkpoint "C:\\...\\model.pt" --render 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows / Conda OpenMP workaround. This is unrelated to the landing logic,
# but it prevents the libomp / libiomp double-load abort seen near the end of
# long runs in some environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SquashedNormal:
    """
    Tanh-squashed diagonal Gaussian for bounded continuous control.

    This keeps the action that PPO stores / scores identical to the action sent to
    the environment, avoiding the sample-then-clamp mismatch from a plain Normal.
    """

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        self.eps = eps

    def rsample(self) -> torch.Tensor:
        z = self.base_dist.rsample()
        return torch.tanh(z)

    def sample(self) -> torch.Tensor:
        z = self.base_dist.sample()
        return torch.tanh(z)

    def deterministic_sample(self) -> torch.Tensor:
        return torch.tanh(self.loc)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        safe_value = torch.clamp(value, -1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = torch.atanh(safe_value)
        log_prob = self.base_dist.log_prob(pre_tanh) - torch.log(1.0 - safe_value.pow(2) + self.eps)
        return log_prob.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # Exact entropy of the squashed distribution is not analytic in closed form.
        # We use the base Gaussian entropy, which is a common approximation for
        # entropy regularization / logging in tanh-squashed policies.
        return self.base_dist.entropy().sum(dim=-1)

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

# Prefer the local lunar_lander.py if it sits next to this script.
try:
    from lunar_lander import (
        LunarLander,
        VIEWPORT_W,
        VIEWPORT_H,
        SCALE,
        LEG_DOWN,
        FPS,
        SIDE_ENGINE_POWER,
        SIDE_ENGINE_AWAY,
    )  # local file in your project folder
except ImportError:
    from gymnasium.envs.box2d.lunar_lander import (
        LunarLander,
        VIEWPORT_W,
        VIEWPORT_H,
        SCALE,
        LEG_DOWN,
        FPS,
        SIDE_ENGINE_POWER,
        SIDE_ENGINE_AWAY,
    )


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    # General
    mode: str = "train"  # train or eval
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = r"C:\Users\rabiei\My Research\LunarLander_v1"
    run_name_prefix: str = "reduced_lander"

    # Environment / reduced model
    gravity: float = -10.0
    enable_wind: bool = False
    wind_power: float = 0.0
    turbulence_power: float = 0.0
    theta_limit_deg: float = 35.0
    max_episode_steps: int = 1000
    safety_x_bound: float = 0.90   # unsafe if |x| > this (normalized LunarLander x)
    include_leg_contacts_in_obs: bool = True

    # Legacy reduced side-thruster settings kept for CLI/checkpoint compatibility.
    # The direct-theta settle environment uses PPO action[1] directly; only
    # reduced_zero_side_on_contact still affects runtime behavior here.
    reduced_inner_kp: float = 2.0
    reduced_inner_kd: float = 0.5
    reduced_min_side_action: float = 0.55
    reduced_max_side_action: float = 1.0
    reduced_side_deadband: float = 0.05
    reduced_zero_side_on_contact: bool = False
    reduced_side_sign: int = 0  # 0 = auto-detect from stock LunarLander sign convention

    # Touchdown / settle-mode logic to make Box2D sleep reachable while keeping
    # official LunarLander success based on `not self.lander.awake`.
    settle_mode_enabled: bool = True
    settle_force_side_off: bool = True
    settle_require_leg_contact: bool = True          # legacy compatibility: require at least one leg if both-leg mode is off
    settle_require_both_legs_contact: bool = True    # safer default for touchdown than any-leg contact
    settle_enter_steps: int = 3
    settle_exit_steps: int = 2
    settle_enter_main_threshold: float = 0.10        # relaxed from 0.0 so settle can engage near touchdown
    settle_exit_main_threshold: float = 0.10
    settle_enter_side_threshold: float = 0.50        # kept for compatibility; effective side-off check follows official deadzone
    settle_exit_side_threshold: float = 0.55
    settle_enter_vx_threshold: float = 0.20
    settle_enter_vy_threshold: float = 0.20
    settle_enter_omega_threshold: float = 0.20
    settle_exit_vx_threshold: float = 0.35
    settle_exit_vy_threshold: float = 0.35
    settle_exit_omega_threshold: float = 0.35

    # Training-only reward shaping to make true sleep-based landing easier to discover.
    timeout_penalty_on_truncation: float = 50.0
    touchdown_discovery_bonus: float = 25.0
    touchdown_bonus_vx_threshold: float = 0.35
    touchdown_bonus_vy_threshold: float = 0.35
    touchdown_bonus_omega_threshold: float = 0.35
    settle_enter_bonus: float = 10.0
    best_model_select_by_success_then_return: bool = True

    # PPO / PPO-Lagrangian
    total_steps: int = 500_000
    steps_per_epoch: int = 4096
    num_envs: int = 8
    update_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    cost_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    cost_vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    target_kl: float = 0.03

    # Safety / Lagrange multiplier
    use_safety: bool = True
    lambda_init: float = 0.0
    lambda_lr: float = 5e-3
    cost_limit: float = 3.0    # mean episodic cost target

    # Policy / network
    hidden_sizes: Tuple[int, int] = (256, 256)
    activation: str = "tanh"

    # Action std behavior in NORMALIZED action units [-1, 1]
    # Action 0: main engine command; Action 1: normalized theta* command.
    std_mode: str = "anneal"   # fixed | anneal | learned
    init_action_std: Tuple[float, float] = (0.35, 0.20)
    final_action_std: Tuple[float, float] = (0.08, 0.05)

    # Normalization
    use_obs_norm: bool = True
    obs_clip: float = 10.0

    # Evaluation / logging
    eval_every_epochs: int = 10
    eval_episodes: int = 10
    render: bool = False
    show_plots: bool = True
    deterministic_eval: bool = True

    # Warm start / fine tuning
    warm_start_path: str = ""
    checkpoint: str = ""         # for eval mode
    resume_optimizer: bool = False
    true_resume: bool = False     # resume schedule/epoch/global_step from checkpoint


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v in ["1", "true", "True", "yes", "y"]:
        return True
    if v in ["0", "false", "False", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool from {v}")


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def detect_stock_side_sign(
    gravity: float,
    enable_wind: bool,
    wind_power: float,
    turbulence_power: float,
    seed: int = 0,
) -> int:
    env = LunarLander(
        render_mode=None,
        continuous=True,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
    )
    try:
        obs, _ = env.reset(seed=seed)
        theta0 = float(obs[4])
        obs1, _, _, _, _ = env.step(np.array([0.0, 1.0], dtype=np.float32))
        theta1 = float(obs1[4])
        dtheta = (theta1 - theta0 + math.pi) % (2.0 * math.pi) - math.pi
        return 1 if dtheta >= 0.0 else -1
    finally:
        env.close()


class RunningMeanStd:
    """Numerically stable running mean/std for observation normalization."""

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(x, -clip, clip)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "var": self.var, "count": np.array([self.count], dtype=np.float64)}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(np.asarray(state["count"]).reshape(-1)[0])


# -----------------------------
# Reduced Lander Environment
# -----------------------------

class ReducedLanderEnv(LunarLander):
    """
    Reduced model (Lander-R) built on top of stock-like Box2D LunarLander.

    Key idea for transfer:
    - The policy still outputs [main_cmd, theta_cmd_norm].
    - theta* defines the attitude directly, so reduced training ignores theta dynamics.
    - PPO action[1] is also used directly as the reduced side-thruster command.
    - The official continuous LunarLander side-engine fire rule is preserved:
      no side thrust for |action[1]| <= 0.5, and clipped magnitude in [0.5, 1.0] otherwise.
    - The reduced environment applies the translational effect and fuel cost of that side
      command, but removes its rotational effect by applying the side impulse at the center
      of mass and re-imposing theta* outside settle mode.
    - Settle mode is retained only as a helper so the body can still sleep and trigger the
      official success condition based on `not self.lander.awake`.
    - Reward shaping stays aligned with official LunarLander for the base task, but this
      training-oriented variant can add small discovery aids: a one-time quiet-touchdown
      bonus, a one-time settle-entry bonus, and a time-limit penalty applied externally
      during training when the episode ends without a real sleep-based landing.
    - The main settle changes are: both-leg contact by default, a slightly relaxed
      main-thrust threshold, and checking whether the side engine is effectively OFF under
      the official side-engine deadzone.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 0.0,
        turbulence_power: float = 0.0,
        theta_limit_deg: float = 35.0,
        safety_x_bound: float = 0.90,
        include_leg_contacts_in_obs: bool = True,
        reduced_inner_kp: float = 2.0,
        reduced_inner_kd: float = 0.5,
        reduced_min_side_action: float = 0.55,
        reduced_max_side_action: float = 1.0,
        reduced_side_deadband: float = 0.05,
        reduced_zero_side_on_contact: bool = False,
        reduced_side_sign: int = 1,
        settle_mode_enabled: bool = True,
        settle_force_side_off: bool = True,
        settle_require_leg_contact: bool = True,
        settle_require_both_legs_contact: bool = True,
        settle_enter_steps: int = 3,
        settle_exit_steps: int = 2,
        settle_enter_main_threshold: float = 0.10,
        settle_exit_main_threshold: float = 0.10,
        settle_enter_side_threshold: float = 0.50,
        settle_exit_side_threshold: float = 0.55,
        settle_enter_vx_threshold: float = 0.20,
        settle_enter_vy_threshold: float = 0.20,
        settle_enter_omega_threshold: float = 0.20,
        settle_exit_vx_threshold: float = 0.35,
        settle_exit_vy_threshold: float = 0.35,
        settle_exit_omega_threshold: float = 0.35,
        apply_training_reward_shaping: bool = False,
        touchdown_discovery_bonus: float = 25.0,
        touchdown_bonus_vx_threshold: float = 0.35,
        touchdown_bonus_vy_threshold: float = 0.35,
        touchdown_bonus_omega_threshold: float = 0.35,
        settle_enter_bonus: float = 10.0,
    ):
        super().__init__(
            render_mode=render_mode,
            continuous=True,
            gravity=gravity,
            enable_wind=enable_wind,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )
        self.theta_limit_rad = math.radians(theta_limit_deg)
        self.safety_x_bound = float(safety_x_bound)
        self.include_leg_contacts_in_obs = bool(include_leg_contacts_in_obs)

        # Kept only so old CLI arguments / checkpoints still load cleanly.
        self.reduced_inner_kp = float(reduced_inner_kp)
        self.reduced_inner_kd = float(reduced_inner_kd)
        self.reduced_min_side_action = float(reduced_min_side_action)
        self.reduced_max_side_action = float(reduced_max_side_action)
        self.reduced_side_deadband = float(reduced_side_deadband)
        self.reduced_zero_side_on_contact = bool(reduced_zero_side_on_contact)
        self.reduced_side_sign = int(reduced_side_sign)

        self.settle_mode_enabled = bool(settle_mode_enabled)
        self.settle_force_side_off = bool(settle_force_side_off)
        self.settle_require_leg_contact = bool(settle_require_leg_contact)
        self.settle_require_both_legs_contact = bool(settle_require_both_legs_contact)
        self.settle_enter_steps = max(1, int(settle_enter_steps))
        self.settle_exit_steps = max(1, int(settle_exit_steps))
        self.settle_enter_main_threshold = float(settle_enter_main_threshold)
        self.settle_exit_main_threshold = float(settle_exit_main_threshold)
        self.settle_enter_side_threshold = float(settle_enter_side_threshold)
        self.settle_exit_side_threshold = float(settle_exit_side_threshold)
        self.settle_enter_vx_threshold = float(settle_enter_vx_threshold)
        self.settle_enter_vy_threshold = float(settle_enter_vy_threshold)
        self.settle_enter_omega_threshold = float(settle_enter_omega_threshold)
        self.settle_exit_vx_threshold = float(settle_exit_vx_threshold)
        self.settle_exit_vy_threshold = float(settle_exit_vy_threshold)
        self.settle_exit_omega_threshold = float(settle_exit_omega_threshold)

        self.apply_training_reward_shaping = bool(apply_training_reward_shaping)
        self.touchdown_discovery_bonus = float(touchdown_discovery_bonus)
        self.touchdown_bonus_vx_threshold = float(touchdown_bonus_vx_threshold)
        self.touchdown_bonus_vy_threshold = float(touchdown_bonus_vy_threshold)
        self.touchdown_bonus_omega_threshold = float(touchdown_bonus_omega_threshold)
        self.settle_enter_bonus = float(settle_enter_bonus)

        if self.include_leg_contacts_in_obs:
            low = np.array([-2.5, -2.5, -10.0, -10.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([2.5, 2.5, 10.0, 10.0, 1.0, 1.0], dtype=np.float32)
        else:
            low = np.array([-2.5, -2.5, -10.0, -10.0], dtype=np.float32)
            high = np.array([2.5, 2.5, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.last_theta_cmd = 0.0
        self.last_main_cmd = -1.0
        self.last_side_cmd = 0.0
        self._reduced_prev_shaping = None
        self.settle_mode = False
        self._settle_ready_counter = 0
        self._settle_exit_counter = 0
        self._prev_leg_contact_both = False
        self._touchdown_bonus_given = False
        self._settle_bonus_given = False

    def _current_full_state(self) -> np.ndarray:
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = np.array([
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ], dtype=np.float32)
        return state

    def _reduce_obs(self, full_state: np.ndarray) -> np.ndarray:
        if self.include_leg_contacts_in_obs:
            return np.asarray([full_state[0], full_state[1], full_state[2], full_state[3], full_state[6], full_state[7]], dtype=np.float32)
        return np.asarray(full_state[:4], dtype=np.float32)

    def _compute_cost(self, full_state: np.ndarray, terminated: bool) -> float:
        x = float(full_state[0])
        crashed_or_oob = bool(self.game_over or abs(x) >= 1.0)
        unsafe_x = bool(abs(x) > self.safety_x_bound)
        return float(crashed_or_oob or unsafe_x)

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _side_power_from_cmd(self, side_cmd: float) -> float:
        if abs(side_cmd) <= 0.5:
            return 0.0
        return float(np.clip(abs(side_cmd), 0.5, 1.0))

    def _direct_side_cmd_from_theta_action(self, theta_action_norm: float, c_left: float, c_right: float):
        raw_side_cmd = float(np.clip(theta_action_norm, -1.0, 1.0))
    
        if self.reduced_zero_side_on_contact and (c_left > 0.5 and c_right > 0.5):
            gated_side_cmd = 0.0
        else:
            gated_side_cmd = raw_side_cmd
    
        return gated_side_cmd, {
            "mode": "direct_theta_action",
            "theta_action_norm": float(theta_action_norm),
            "raw_side_cmd": float(raw_side_cmd),
            "contact_gate_applied": bool(
                self.reduced_zero_side_on_contact and (c_left > 0.5 and c_right > 0.5)
            ),
            "side_fire_rule": "official_continuous_deadzone_abs_le_0p5_off",
            "side_cmd": float(gated_side_cmd),
        }

    def _leg_contact_flags(self, full_state: np.ndarray) -> Tuple[bool, bool, bool]:
        left = bool(full_state[6] > 0.5)
        right = bool(full_state[7] > 0.5)
        return left, right, bool(left or right)

    def _settle_contact_condition(self, full_state: np.ndarray) -> bool:
        left, right, any_contact = self._leg_contact_flags(full_state)
        if self.settle_require_both_legs_contact:
            return bool(left and right)
        if self.settle_require_leg_contact:
            return bool(any_contact)
        return True

    def _side_engine_effectively_on(self, side_cmd: float) -> bool:
        # Follow the official continuous-action side-engine fire rule:
        # |side_cmd| <= 0.5 means the side engine is OFF.
        return bool(self._side_power_from_cmd(side_cmd) > 0.0)

    def _enter_ready_to_settle(self, full_state: np.ndarray, main_cmd: float, side_cmd: float, omega_raw: float) -> bool:
        if not self._settle_contact_condition(full_state):
            return False
        side_effectively_off = not self._side_engine_effectively_on(side_cmd)
        return bool(
            main_cmd <= self.settle_enter_main_threshold
            and side_effectively_off
            and abs(float(full_state[2])) < self.settle_enter_vx_threshold
            and abs(float(full_state[3])) < self.settle_enter_vy_threshold
            and abs(float(omega_raw)) < self.settle_enter_omega_threshold
        )

    def _leave_settle_mode(self, full_state: np.ndarray, main_cmd: float, side_cmd: float, omega_raw: float) -> bool:
        side_effectively_on = self._side_engine_effectively_on(side_cmd)
        return bool(
            (not self._settle_contact_condition(full_state))
            or (main_cmd > self.settle_exit_main_threshold)
            or side_effectively_on
            or (abs(float(full_state[2])) > self.settle_exit_vx_threshold)
            or (abs(float(full_state[3])) > self.settle_exit_vy_threshold)
            or (abs(float(omega_raw)) > self.settle_exit_omega_threshold)
        )

    def _update_settle_mode(self, full_state: np.ndarray, main_cmd: float, side_cmd: float, omega_raw: float) -> Tuple[bool, bool, bool]:
        if not self.settle_mode_enabled:
            self.settle_mode = False
            self._settle_ready_counter = 0
            self._settle_exit_counter = 0
            return False, False, False

        entered = False
        exited = False
        ready_to_settle = self._enter_ready_to_settle(full_state, main_cmd, side_cmd, omega_raw)

        if self.settle_mode:
            if self._leave_settle_mode(full_state, main_cmd, side_cmd, omega_raw):
                self._settle_exit_counter += 1
            else:
                self._settle_exit_counter = 0
            if self._settle_exit_counter >= self.settle_exit_steps:
                self.settle_mode = False
                self._settle_exit_counter = 0
                self._settle_ready_counter = 0
                exited = True
        else:
            if ready_to_settle:
                self._settle_ready_counter += 1
            else:
                self._settle_ready_counter = 0
            if self._settle_ready_counter >= self.settle_enter_steps:
                self.settle_mode = True
                self._settle_ready_counter = 0
                self._settle_exit_counter = 0
                entered = True

        return self.settle_mode, entered, exited

    def _apply_side_translation_only(self, side_cmd: float) -> float:
        s_power = self._side_power_from_cmd(side_cmd)
        if s_power <= 0.0:
            return 0.0

        direction = float(np.sign(side_cmd))
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        ox = tip[0] * dispersion[0] + side[0] * (3.0 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
        oy = -tip[1] * dispersion[0] - side[1] * (3.0 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
        impulse = (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power)
        self.lander.ApplyLinearImpulse(impulse, self.lander.worldCenter, True)
        return float(s_power)

    def _compute_stock_style_reward(self, full_state: np.ndarray, main_cmd: float, s_power: float) -> float:
        x, y, vx, vy, angle, _, left_leg, right_leg = [float(v) for v in full_state]
        shaping = (
            -100.0 * math.sqrt(x * x + y * y)
            -100.0 * math.sqrt(vx * vx + vy * vy)
            + 10.0 * left_leg
            + 10.0 * right_leg
        )
        shaping -= 100.0 * abs(angle)

        if self._reduced_prev_shaping is None:
            reward = 0.0
        else:
            reward = shaping - float(self._reduced_prev_shaping)
        self._reduced_prev_shaping = float(shaping)

        if main_cmd > 0.0:
            m_power = float((np.clip(main_cmd, 0.0, 1.0) + 1.0) * 0.5)
        else:
            m_power = 0.0
        reward -= m_power * 0.30
        reward -= s_power * 0.03
        return float(reward)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        full_obs, info = super().reset(seed=seed, options=options)
        self.last_theta_cmd = 0.0
        self.last_main_cmd = -1.0
        self.last_side_cmd = 0.0
        self._reduced_prev_shaping = None
        self.settle_mode = False
        self._settle_ready_counter = 0
        self._settle_exit_counter = 0
        self._prev_leg_contact_both = False
        self._touchdown_bonus_given = False
        self._settle_bonus_given = False
        full_state = self._current_full_state()
        reduced_obs = self._reduce_obs(full_state)
        info = dict(info)
        info["full_obs"] = np.asarray(full_obs, dtype=np.float32)
        info["full_state"] = np.asarray(full_state, dtype=np.float32)
        info["theta_cmd_rad"] = self.last_theta_cmd
        info["theta_cmd_deg"] = math.degrees(self.last_theta_cmd)
        info["main_cmd"] = self.last_main_cmd
        info["side_cmd"] = self.last_side_cmd
        info["leg_contact_left"] = float(full_state[6])
        info["leg_contact_right"] = float(full_state[7])
        info["leg_contact_any"] = bool((full_state[6] > 0.5) or (full_state[7] > 0.5))
        info["leg_contact_both"] = bool((full_state[6] > 0.5) and (full_state[7] > 0.5))
        info["cost"] = 0.0
        info["settle_mode"] = bool(self.settle_mode)
        info["settle_ready_counter"] = int(self._settle_ready_counter)
        info["settle_exit_counter"] = int(self._settle_exit_counter)
        info["settle_entered"] = False
        info["settle_exited"] = False
        info["theta_imposed_pre_step"] = False
        info["theta_imposed_post_step"] = False
        return reduced_obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        main_cmd = float(action[0])
        theta_cmd = float(action[1] * self.theta_limit_rad)

        pre_state = self._current_full_state()
        omega_raw = float(pre_state[5]) * FPS / 20.0
        c_left = float(pre_state[6])
        c_right = float(pre_state[7])
        raw_side_cmd, ctrl_dbg = self._direct_side_cmd_from_theta_action(action[1], c_left, c_right)

        settle_mode, settle_entered, settle_exited = self._update_settle_mode(pre_state, main_cmd, raw_side_cmd, omega_raw)
        if settle_mode and self.settle_force_side_off:
            side_cmd = 0.0
        else:
            side_cmd = raw_side_cmd

        self.last_main_cmd = main_cmd
        self.last_theta_cmd = theta_cmd
        self.last_side_cmd = side_cmd

        theta_imposed_pre_step = False
        if not settle_mode:
            self.lander.angle = theta_cmd
            self.lander.angularVelocity = 0.0
            theta_imposed_pre_step = True

        s_power = self._apply_side_translation_only(side_cmd)

        full_obs_raw, _, terminated, truncated, info = super().step(np.array([main_cmd, 0.0], dtype=np.float32))

        theta_imposed_post_step = False
        if not terminated and self.lander.awake and not settle_mode:
            self.lander.angle = theta_cmd
            self.lander.angularVelocity = 0.0
            theta_imposed_post_step = True

        full_state = self._current_full_state()
        reward = self._compute_stock_style_reward(full_state, main_cmd, s_power)

        leg_contact_any = bool((full_state[6] > 0.5) or (full_state[7] > 0.5))
        leg_contact_both = bool((full_state[6] > 0.5) and (full_state[7] > 0.5))
        info = dict(info)

        training_touchdown_bonus_applied = 0.0
        training_settle_bonus_applied = 0.0
        omega_raw_now = float(full_state[5]) * FPS / 20.0
        if self.apply_training_reward_shaping:
            new_both_legs = bool(leg_contact_both and (not self._prev_leg_contact_both))
            low_touch = bool(
                abs(float(full_state[2])) < self.touchdown_bonus_vx_threshold
                and abs(float(full_state[3])) < self.touchdown_bonus_vy_threshold
                and abs(float(omega_raw_now)) < self.touchdown_bonus_omega_threshold
            )
            if (not self._touchdown_bonus_given) and new_both_legs and low_touch:
                reward += self.touchdown_discovery_bonus
                training_touchdown_bonus_applied = float(self.touchdown_discovery_bonus)
                self._touchdown_bonus_given = True
            if (not self._settle_bonus_given) and settle_entered:
                reward += self.settle_enter_bonus
                training_settle_bonus_applied = float(self.settle_enter_bonus)
                self._settle_bonus_given = True
        self._prev_leg_contact_both = bool(leg_contact_both)

        is_success = False
        if self.game_over or abs(float(full_state[0])) >= 1.0:
            terminated = True
            reward = -100.0
        if not self.lander.awake:
            terminated = True
            reward = +100.0
            is_success = True

        reduced_obs = self._reduce_obs(full_state)
        cost = self._compute_cost(full_state, terminated)

        ctrl_dbg = dict(ctrl_dbg)
        ctrl_dbg["raw_side_cmd"] = float(raw_side_cmd)
        ctrl_dbg["applied_side_cmd"] = float(side_cmd)
        ctrl_dbg["settle_mode"] = bool(settle_mode)
        ctrl_dbg["settle_entered"] = bool(settle_entered)
        ctrl_dbg["settle_exited"] = bool(settle_exited)

        info["full_obs"] = np.asarray(full_obs_raw, dtype=np.float32)
        info["full_state"] = np.asarray(full_state, dtype=np.float32)
        info["theta_cmd_rad"] = theta_cmd
        info["theta_cmd_deg"] = math.degrees(theta_cmd)
        info["main_cmd"] = main_cmd
        info["side_cmd"] = float(side_cmd)
        info["raw_side_cmd"] = float(raw_side_cmd)
        info["side_power"] = float(s_power)
        info["controller_debug"] = ctrl_dbg
        info["leg_contact_left"] = float(full_state[6])
        info["leg_contact_right"] = float(full_state[7])
        info["leg_contact_any"] = leg_contact_any
        info["leg_contact_both"] = leg_contact_both
        info["cost"] = cost
        info["is_success"] = bool(is_success)
        info["crashed"] = bool(self.game_over)
        info["awake"] = bool(self.lander.awake)
        info["settle_mode"] = bool(self.settle_mode)
        info["settle_ready_counter"] = int(self._settle_ready_counter)
        info["settle_exit_counter"] = int(self._settle_exit_counter)
        info["settle_entered"] = bool(settle_entered)
        info["settle_exited"] = bool(settle_exited)
        info["theta_imposed_pre_step"] = bool(theta_imposed_pre_step)
        info["theta_imposed_post_step"] = bool(theta_imposed_post_step)
        info["training_touchdown_bonus"] = float(training_touchdown_bonus_applied)
        info["training_settle_enter_bonus"] = float(training_settle_bonus_applied)
        return reduced_obs, float(reward), bool(terminated), bool(truncated), info



def make_env(cfg: Config, render_mode: Optional[str] = None, apply_training_reward_shaping: bool = False):
    def thunk():
        env = ReducedLanderEnv(
            render_mode=render_mode,
            gravity=cfg.gravity,
            enable_wind=cfg.enable_wind,
            wind_power=cfg.wind_power,
            turbulence_power=cfg.turbulence_power,
            theta_limit_deg=cfg.theta_limit_deg,
            safety_x_bound=cfg.safety_x_bound,
            include_leg_contacts_in_obs=cfg.include_leg_contacts_in_obs,
            reduced_inner_kp=cfg.reduced_inner_kp,
            reduced_inner_kd=cfg.reduced_inner_kd,
            reduced_min_side_action=cfg.reduced_min_side_action,
            reduced_max_side_action=cfg.reduced_max_side_action,
            reduced_side_deadband=cfg.reduced_side_deadband,
            reduced_zero_side_on_contact=cfg.reduced_zero_side_on_contact,
            reduced_side_sign=cfg.reduced_side_sign,
            settle_mode_enabled=cfg.settle_mode_enabled,
            settle_force_side_off=cfg.settle_force_side_off,
            settle_require_leg_contact=cfg.settle_require_leg_contact,
            settle_require_both_legs_contact=cfg.settle_require_both_legs_contact,
            settle_enter_steps=cfg.settle_enter_steps,
            settle_exit_steps=cfg.settle_exit_steps,
            settle_enter_main_threshold=cfg.settle_enter_main_threshold,
            settle_exit_main_threshold=cfg.settle_exit_main_threshold,
            settle_enter_side_threshold=cfg.settle_enter_side_threshold,
            settle_exit_side_threshold=cfg.settle_exit_side_threshold,
            settle_enter_vx_threshold=cfg.settle_enter_vx_threshold,
            settle_enter_vy_threshold=cfg.settle_enter_vy_threshold,
            settle_enter_omega_threshold=cfg.settle_enter_omega_threshold,
            settle_exit_vx_threshold=cfg.settle_exit_vx_threshold,
            settle_exit_vy_threshold=cfg.settle_exit_vy_threshold,
            settle_exit_omega_threshold=cfg.settle_exit_omega_threshold,
            apply_training_reward_shaping=apply_training_reward_shaping,
            touchdown_discovery_bonus=cfg.touchdown_discovery_bonus,
            touchdown_bonus_vx_threshold=cfg.touchdown_bonus_vx_threshold,
            touchdown_bonus_vy_threshold=cfg.touchdown_bonus_vy_threshold,
            touchdown_bonus_omega_threshold=cfg.touchdown_bonus_omega_threshold,
            settle_enter_bonus=cfg.settle_enter_bonus,
        )
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        return env
    return thunk


# -----------------------------
# PPO / PPO-Lagrangian model
# -----------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, int], activation: str = "tanh"):
        super().__init__()
        act = make_activation(activation)
        layers: List[nn.Module] = []
        prev = input_dim
        for hs in hidden_sizes:
            layers += [nn.Linear(prev, hs), act.__class__()]
            prev = hs
        self.net = nn.Sequential(*layers)
        self.output_dim = prev

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Config, action_scale: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32)

        self.shared = MLP(obs_dim, cfg.hidden_sizes, cfg.activation)
        self.policy_head = nn.Linear(self.shared.output_dim, act_dim)
        self.value_head = nn.Linear(self.shared.output_dim, 1)
        self.cost_value_head = nn.Linear(self.shared.output_dim, 1)

        init_std = torch.tensor(cfg.init_action_std, dtype=torch.float32)
        self.std_mode = cfg.std_mode.lower()
        if self.std_mode == "learned":
            self.log_std = nn.Parameter(torch.log(init_std))
        else:
            self.register_buffer("log_std_buffer", torch.log(init_std))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def set_action_std(self, std_values: np.ndarray) -> None:
        if self.std_mode == "learned":
            return
        std_values = np.asarray(std_values, dtype=np.float32)
        self.log_std_buffer.copy_(torch.log(torch.tensor(std_values, dtype=torch.float32, device=self.log_std_buffer.device)))

    def current_action_std(self) -> torch.Tensor:
        if self.std_mode == "learned":
            return torch.exp(self.log_std)
        return torch.exp(self.log_std_buffer)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        pre_tanh_mean = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        cost_value = self.cost_value_head(h).squeeze(-1)
        return pre_tanh_mean, value, cost_value

    def distribution(self, obs: torch.Tensor) -> Tuple[SquashedNormal, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_tanh_mean, value, cost_value = self.forward(obs)
        std = self.current_action_std().expand_as(pre_tanh_mean)
        dist = SquashedNormal(pre_tanh_mean, std)
        mean_action = torch.tanh(pre_tanh_mean)
        return dist, mean_action, value, cost_value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        dist, mean_action, value, cost_value = self.distribution(obs)
        if deterministic:
            action = mean_action
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, cost_value, mean_action, self.current_action_std()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, mean_action, value, cost_value = self.distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value, cost_value, mean_action, self.current_action_std()


# -----------------------------
# Rollout Buffer
# -----------------------------


class VectorizedRolloutBuffer:
    def __init__(self, obs_dim: int, act_dim: int, rollout_steps: int, num_envs: int, gamma: float, cost_gamma: float, gae_lambda: float):
        self.obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs, act_dim), dtype=np.float32)
        self.logprobs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.costs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.cost_values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.cost_returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.gamma = gamma
        self.cost_gamma = cost_gamma
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.ptr = 0

    def store(self, obs, actions, logprobs, rewards, costs, dones, values, cost_values):
        assert self.ptr < self.rollout_steps
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = rewards
        self.costs[self.ptr] = costs
        self.dones[self.ptr] = dones
        self.values[self.ptr] = values
        self.cost_values[self.ptr] = cost_values
        self.ptr += 1

    def finish_path(self, last_values: np.ndarray, last_cost_values: np.ndarray):
        assert self.ptr == self.rollout_steps

        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        last_cost_gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_values = np.asarray(last_values, dtype=np.float32)
                next_cost_values = np.asarray(last_cost_values, dtype=np.float32)
            else:
                next_values = self.values[t + 1]
                next_cost_values = self.cost_values[t + 1]

            next_nonterminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + self.gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + self.values[t]

            cost_delta = self.costs[t] + self.cost_gamma * next_cost_values * next_nonterminal - self.cost_values[t]
            last_cost_gae = cost_delta + self.cost_gamma * self.gae_lambda * next_nonterminal * last_cost_gae
            self.cost_advantages[t] = last_cost_gae
            self.cost_returns[t] = self.cost_advantages[t] + self.cost_values[t]

    def get(self):
        assert self.ptr == self.rollout_steps
        adv = self.advantages.reshape(-1)
        cost_adv = self.cost_advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        cost_adv = (cost_adv - cost_adv.mean()) / (cost_adv.std() + 1e-8)
        data = dict(
            obs=self.obs.reshape(-1, self.obs.shape[-1]),
            actions=self.actions.reshape(-1, self.actions.shape[-1]),
            logprobs=self.logprobs.reshape(-1),
            advantages=adv,
            cost_advantages=cost_adv,
            returns=self.returns.reshape(-1),
            cost_returns=self.cost_returns.reshape(-1),
            rewards=self.rewards.reshape(-1),
            costs=self.costs.reshape(-1),
        )
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}


def discounted_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out


# -----------------------------
# Plotting / saving helpers
# -----------------------------

def moving_average(xs: List[float], window: int = 20) -> np.ndarray:
    if len(xs) == 0:
        return np.array([])
    xs = np.asarray(xs, dtype=np.float32)
    if len(xs) < window:
        return xs
    kernel = np.ones(window, dtype=np.float32) / window
    valid = np.convolve(xs, kernel, mode="valid")
    pad = np.concatenate([xs[: window - 1], valid])
    return pad


@dataclass
class TrainingHistory:
    epoch: List[int] = field(default_factory=list)
    global_step: List[int] = field(default_factory=list)
    mean_episode_return: List[float] = field(default_factory=list)
    mean_episode_cost: List[float] = field(default_factory=list)
    success_rate: List[float] = field(default_factory=list)
    crash_rate: List[float] = field(default_factory=list)
    lambda_value: List[float] = field(default_factory=list)
    eval_return: List[float] = field(default_factory=list)
    eval_cost: List[float] = field(default_factory=list)
    eval_success: List[float] = field(default_factory=list)
    eval_crashed: List[float] = field(default_factory=list)
    policy_std_main: List[float] = field(default_factory=list)
    policy_std_theta_deg: List[float] = field(default_factory=list)
    rollout_std_main: List[float] = field(default_factory=list)
    rollout_std_theta_deg: List[float] = field(default_factory=list)


@dataclass
class RunPaths:
    run_dir: Path
    checkpoint_path: Path
    norm_path: Path
    best_checkpoint_path: Path
    best_norm_path: Path
    best_metrics_path: Path
    config_path: Path
    csv_path: Path
    learning_curve_path: Path
    action_std_path: Path



def build_run_paths(cfg: Config) -> RunPaths:
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    warm_tag = "warm" if cfg.warm_start_path else "cold"
    safety_tag = f"safe{int(cfg.use_safety)}"
    lam_lr_tag = f"lamlr{cfg.lambda_lr:g}"
    lr_tag = f"lr{cfg.learning_rate:g}"
    std_tag = f"std{cfg.std_mode}"
    env_tag = f"nenv{cfg.num_envs}"
    norm_tag = f"norm{int(cfg.use_obs_norm)}"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.run_name_prefix}_{safety_tag}_{lam_lr_tag}_{lr_tag}_{std_tag}_{env_tag}_{warm_tag}_{norm_tag}_seed{cfg.seed}_{stamp}"
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        checkpoint_path=run_dir / f"{run_name}_model.pt",
        norm_path=run_dir / f"{run_name}_obsnorm.npz",
        best_checkpoint_path=run_dir / "best_model.pt",
        best_norm_path=run_dir / "best_obsnorm.npz",
        best_metrics_path=run_dir / "best_metrics.json",
        config_path=run_dir / f"{run_name}_config.json",
        csv_path=run_dir / f"{run_name}_history.csv",
        learning_curve_path=run_dir / f"{run_name}_learning_curves.png",
        action_std_path=run_dir / f"{run_name}_action_std.png",
    )



def save_obs_norm(rms: RunningMeanStd, path: Path) -> None:
    np.savez(path, **rms.state_dict())



def load_obs_norm(path: str | Path) -> RunningMeanStd:
    arr = np.load(path)
    rms = RunningMeanStd(shape=arr["mean"].shape)
    rms.load_state_dict({"mean": arr["mean"], "var": arr["var"], "count": arr["count"]})
    return rms


def infer_obs_norm_path_from_model_path(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    if model_path.name.endswith("_model.pt"):
        return model_path.with_name(model_path.name.replace("_model.pt", "_obsnorm.npz"))
    return model_path.with_suffix(".obsnorm.npz")


def maybe_load_companion_obs_norm(obs_rms: Optional[RunningMeanStd], model_path: str | Path) -> bool:
    if obs_rms is None:
        return False
    obsnorm_path = infer_obs_norm_path_from_model_path(model_path)
    if not obsnorm_path.exists():
        return False
    loaded = load_obs_norm(obsnorm_path)
    obs_rms.load_state_dict(loaded.state_dict())
    print(f"[INFO] Loaded obs norm from: {obsnorm_path}")
    return True



def save_history_csv(history: TrainingHistory, path: Path) -> None:
    keys = list(asdict(history).keys())
    rows = zip(*[getattr(history, k) for k in keys])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)



def plot_history(history: TrainingHistory, paths: RunPaths, show: bool = True) -> None:
    epochs = np.asarray(history.epoch)

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, history.mean_episode_return, label="Train return")
    ax1.plot(epochs, moving_average(history.mean_episode_return, 10), label="Train return (MA10)")
    if len(history.eval_return) == len(epochs):
        ax1.plot(epochs, history.eval_return, label="Eval return")
    ax1.set_title("Episode return")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, history.mean_episode_cost, label="Train cost")
    if len(history.eval_cost) == len(epochs):
        ax2.plot(epochs, history.eval_cost, label="Eval cost")
    ax2.plot(epochs, history.lambda_value, label="Lambda")
    ax2.set_title("Safety cost / lambda")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(epochs, history.success_rate, label="Success rate")
    ax3.plot(epochs, history.crash_rate, label="Crash rate")
    ax3.set_title("Success / crash rate")
    ax3.set_xlabel("Epoch")
    ax3.legend()
    ax3.grid(True)

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(epochs, history.global_step, label="Global steps")
    ax4.set_title("Training progress")
    ax4.set_xlabel("Epoch")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(paths.learning_curve_path, dpi=180, bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.policy_std_main, label="Policy std main")
    plt.plot(epochs, history.rollout_std_main, label="Rollout std main")
    plt.plot(epochs, history.policy_std_theta_deg, label="Policy std theta* [deg]")
    plt.plot(epochs, history.rollout_std_theta_deg, label="Rollout std theta* [deg]")
    plt.title("Action exploration std")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(paths.action_std_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")


# -----------------------------
# Training / evaluation
# -----------------------------

def schedule_action_std(cfg: Config, progress: float) -> np.ndarray:
    """
    progress: 0.0 at start, 1.0 at end.
    """
    init_std = np.asarray(cfg.init_action_std, dtype=np.float32)
    final_std = np.asarray(cfg.final_action_std, dtype=np.float32)
    mode = cfg.std_mode.lower()
    if mode == "fixed":
        return init_std
    if mode == "anneal":
        return init_std + (final_std - init_std) * progress
    if mode == "learned":
        return init_std  # only used before the model is loaded/updated; ignored after first call
    raise ValueError(f"Unknown std_mode: {cfg.std_mode}")


@torch.no_grad()
def run_policy_episode(
    env: gym.Env,
    agent: ActorCritic,
    obs_rms: Optional[RunningMeanStd],
    device: torch.device,
    deterministic: bool = True,
    render: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    obs, info = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    ep_cost = 0.0
    steps = 0
    success = 0.0
    crashed = 0.0

    while not done:
        obs_in = obs.copy()
        if obs_rms is not None:
            obs_in = obs_rms.normalize(obs_in)
        obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _, _, _, _, _ = agent.get_action(obs_t, deterministic=deterministic)
        action_np = action.squeeze(0).detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        ep_return += reward
        ep_cost += float(info.get("cost", 0.0))
        steps += 1
        done = terminated or truncated
        if render:
            env.render()
        if done:
            success = float(info.get("is_success", False))
            crashed = float(info.get("crashed", False))

    return {
        "return": ep_return,
        "cost": ep_cost,
        "length": steps,
        "success": success,
        "crashed": crashed,
    }


@torch.no_grad()
def evaluate_policy(
    cfg: Config,
    agent: ActorCritic,
    obs_rms: Optional[RunningMeanStd],
    device: torch.device,
    num_episodes: int = 10,
    render: bool = False,
) -> Dict[str, float]:
    env = make_env(cfg, render_mode="human" if render else None, apply_training_reward_shaping=False)()
    metrics = []
    for i in range(num_episodes):
        eval_seed = cfg.seed + 10_000 + i
        metrics.append(
            run_policy_episode(
                env,
                agent,
                obs_rms,
                device,
                deterministic=cfg.deterministic_eval,
                render=render,
                seed=eval_seed,
            )
        )
    env.close()
    out = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0].keys()}
    return out




def close_envs(envs: List[gym.Env]) -> None:
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def train(cfg: Config) -> None:
    if cfg.num_envs < 1:
        raise ValueError("--num_envs must be >= 1")
    if cfg.steps_per_epoch % cfg.num_envs != 0:
        raise ValueError(
            f"steps_per_epoch ({cfg.steps_per_epoch}) must be divisible by num_envs ({cfg.num_envs}) "
            "so each environment contributes equally to every PPO epoch."
        )

    device = torch.device(cfg.device)
    paths = build_run_paths(cfg)

    with open(paths.config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    envs = [make_env(cfg, render_mode=None, apply_training_reward_shaping=True)() for _ in range(cfg.num_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    action_scale = np.array([1.0, math.radians(cfg.theta_limit_deg)], dtype=np.float32)
    rollout_steps = cfg.steps_per_epoch // cfg.num_envs

    obs_rms = RunningMeanStd(shape=(obs_dim,)) if cfg.use_obs_norm else None

    agent = ActorCritic(obs_dim, act_dim, cfg, action_scale=action_scale).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    global_step = 0
    start_epoch = 0
    lambda_value = max(0.0, float(cfg.lambda_init)) if cfg.use_safety else 0.0
    next_reset_seed = int(cfg.seed)

    if cfg.warm_start_path:
        ckpt = torch.load(cfg.warm_start_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            agent.load_state_dict(ckpt["model_state_dict"])
        else:
            agent.load_state_dict(ckpt)
            ckpt = {}

        resumed_optimizer = False
        if cfg.resume_optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            resumed_optimizer = True

        loaded_obs_norm = False
        if cfg.use_obs_norm and obs_rms is not None:
            if ckpt.get("obs_rms", None) is not None:
                obs_rms.load_state_dict(ckpt["obs_rms"])
                loaded_obs_norm = True
                print("[INFO] Loaded obs norm from checkpoint payload.")
            else:
                loaded_obs_norm = maybe_load_companion_obs_norm(obs_rms, cfg.warm_start_path)

        if cfg.true_resume:
            global_step = int(ckpt.get("global_step", 0))
            start_epoch = int(ckpt.get("epoch", 0))
            if cfg.use_safety:
                lambda_value = float(ckpt.get("lambda_value", cfg.lambda_init))
            if "next_reset_seed" in ckpt:
                next_reset_seed = int(ckpt["next_reset_seed"])
            else:
                next_reset_seed = int(cfg.seed + max(1, global_step))
            print(f"[INFO] Resumed training from: {cfg.warm_start_path}")
            print(f"[INFO] Resume mode keeps epoch/global_step ({start_epoch}, {global_step}).")
        else:
            global_step = 0
            start_epoch = 0
            lambda_value = max(0.0, float(cfg.lambda_init)) if cfg.use_safety else 0.0
            next_reset_seed = int(cfg.seed)
            print(f"[INFO] Warm-started for fine-tuning from: {cfg.warm_start_path}")
            print("[INFO] Fine-tune mode resets epoch/global_step/std schedule to start fresh.")

        if cfg.use_obs_norm and obs_rms is not None and not loaded_obs_norm:
            print("[WARN] No obs norm found in checkpoint payload or companion file.")
        if resumed_optimizer:
            print("[INFO] Optimizer state resumed from checkpoint.")
        else:
            print("[INFO] Optimizer state reset for fine-tuning.")

    # Seed the Python / NumPy / Torch RNGs from the current reset stream.
    # This avoids the warm-start 'Groundhog Day' issue by continuing from a fresh deterministic seed.
    set_seed(next_reset_seed)

    def allocate_reset_seed() -> int:
        nonlocal next_reset_seed
        seed = int(next_reset_seed)
        next_reset_seed += 1
        return seed

    history = TrainingHistory()
    best_score = (-float("inf"), -float("inf"))

    obs = np.zeros((cfg.num_envs, obs_dim), dtype=np.float32)
    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_costs = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_lens = np.zeros(cfg.num_envs, dtype=np.int32)

    for env_idx, env in enumerate(envs):
        obs_i, _ = env.reset(seed=allocate_reset_seed())
        obs[env_idx] = obs_i
    if obs_rms is not None:
        obs_rms.update(obs)

    recent_returns: List[float] = []
    recent_costs: List[float] = []
    recent_success: List[float] = []
    recent_crashes: List[float] = []

    epochs = max(1, cfg.total_steps // cfg.steps_per_epoch)

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            progress = min(1.0, global_step / max(1, cfg.total_steps))
            if cfg.std_mode.lower() in ["fixed", "anneal"]:
                agent.set_action_std(schedule_action_std(cfg, progress))

            buffer = VectorizedRolloutBuffer(
                obs_dim=obs_dim,
                act_dim=act_dim,
                rollout_steps=rollout_steps,
                num_envs=cfg.num_envs,
                gamma=cfg.gamma,
                cost_gamma=cfg.cost_gamma,
                gae_lambda=cfg.gae_lambda,
            )
            rollout_actions = []

            for _ in range(rollout_steps):
                obs_in = obs.copy()
                if obs_rms is not None:
                    obs_in = obs_rms.normalize(obs_in, clip=cfg.obs_clip)

                obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device)
                action_t, logp_t, _, value_t, cost_value_t, _, _ = agent.get_action(obs_t, deterministic=False)

                actions = action_t.detach().cpu().numpy()
                logprobs = logp_t.detach().cpu().numpy()
                values = value_t.detach().cpu().numpy()
                cost_values = cost_value_t.detach().cpu().numpy()

                next_obs = np.zeros_like(obs)
                rewards = np.zeros(cfg.num_envs, dtype=np.float32)
                costs = np.zeros(cfg.num_envs, dtype=np.float32)
                dones = np.zeros(cfg.num_envs, dtype=np.float32)

                for env_idx, env in enumerate(envs):
                    obs_i, reward, terminated, truncated, info = env.step(actions[env_idx])
                    done = bool(terminated or truncated)

                    if truncated and (not terminated) and float(cfg.timeout_penalty_on_truncation) != 0.0:
                        if (not bool(info.get("is_success", False))) and (not bool(info.get("crashed", False))):
                            reward = float(reward) - float(cfg.timeout_penalty_on_truncation)
                            info = dict(info)
                            info["timeout_penalty_applied"] = float(cfg.timeout_penalty_on_truncation)

                    rewards[env_idx] = float(reward)
                    costs[env_idx] = float(info.get("cost", 0.0))
                    dones[env_idx] = float(done)
                    rollout_actions.append(
                        np.array(
                            [actions[env_idx, 0], actions[env_idx, 1] * cfg.theta_limit_deg],
                            dtype=np.float32,
                        )
                    )

                    ep_returns[env_idx] += float(reward)
                    ep_costs[env_idx] += float(costs[env_idx])
                    ep_lens[env_idx] += 1
                    global_step += 1

                    if done:
                        recent_returns.append(float(ep_returns[env_idx]))
                        recent_costs.append(float(ep_costs[env_idx]))
                        recent_success.append(float(info.get("is_success", False)))
                        recent_crashes.append(float(info.get("crashed", False)))

                        reset_obs, _ = env.reset(seed=allocate_reset_seed())
                        next_obs[env_idx] = reset_obs
                        ep_returns[env_idx] = 0.0
                        ep_costs[env_idx] = 0.0
                        ep_lens[env_idx] = 0
                    else:
                        next_obs[env_idx] = obs_i

                buffer.store(
                    obs=obs_in,
                    actions=actions,
                    logprobs=logprobs,
                    rewards=rewards,
                    costs=costs,
                    dones=dones,
                    values=values,
                    cost_values=cost_values,
                )

                if obs_rms is not None:
                    obs_rms.update(next_obs)
                obs = next_obs

            obs_in = obs.copy()
            if obs_rms is not None:
                obs_in = obs_rms.normalize(obs_in, clip=cfg.obs_clip)
            with torch.no_grad():
                obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device)
                _, _, _, last_value_t, last_cost_value_t, _, _ = agent.get_action(obs_t, deterministic=True)

            buffer.finish_path(
                last_values=last_value_t.detach().cpu().numpy(),
                last_cost_values=last_cost_value_t.detach().cpu().numpy(),
            )

            data = buffer.get()
            b_obs = data["obs"].to(device)
            b_actions = data["actions"].to(device)
            b_logprobs = data["logprobs"].to(device)
            b_advantages = data["advantages"].to(device)
            b_cost_advantages = data["cost_advantages"].to(device)
            b_returns = data["returns"].to(device)
            b_cost_returns = data["cost_returns"].to(device)

            mean_ep_cost = float(np.mean(recent_costs[-20:])) if recent_costs else float(np.mean(ep_costs))
            if cfg.use_safety and cfg.lambda_lr > 0.0:
                lambda_value = max(0.0, lambda_value + cfg.lambda_lr * (mean_ep_cost - cfg.cost_limit))
            else:
                lambda_value = 0.0

            batch_size = b_obs.shape[0]
            inds = np.arange(batch_size)
            approx_kl_epoch = 0.0
            stop_early = False

            for _ in range(cfg.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, cfg.minibatch_size):
                    mb_inds = inds[start : start + cfg.minibatch_size]

                    new_logprob, entropy, value, cost_value, _, _ = agent.evaluate_actions(b_obs[mb_inds], b_actions[mb_inds])
                    ratio = torch.exp(new_logprob - b_logprobs[mb_inds])

                    combined_adv = b_advantages[mb_inds] - float(lambda_value) * b_cost_advantages[mb_inds]
                    pg_loss1 = -combined_adv * ratio
                    pg_loss2 = -combined_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    value_loss = 0.5 * F.mse_loss(value, b_returns[mb_inds])
                    cost_value_loss = 0.5 * F.mse_loss(cost_value, b_cost_returns[mb_inds])
                    entropy_loss = entropy.mean()

                    loss = (
                        policy_loss
                        + cfg.vf_coef * value_loss
                        + cfg.cost_vf_coef * cost_value_loss
                        - cfg.ent_coef * entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    with torch.no_grad():
                        approx_kl_epoch = torch.mean(b_logprobs[mb_inds] - new_logprob).item()
                    if approx_kl_epoch > cfg.target_kl:
                        stop_early = True
                        break

                if stop_early:
                    break

            eval_metrics = {"return": np.nan, "cost": np.nan}
            if ((epoch - start_epoch + 1) % cfg.eval_every_epochs == 0) or (epoch == start_epoch + epochs - 1):
                eval_metrics = evaluate_policy(cfg, agent, obs_rms, device, num_episodes=cfg.eval_episodes, render=False)

            current_std = agent.current_action_std().detach().cpu().numpy()
            rollout_actions_arr = np.asarray(rollout_actions, dtype=np.float32)
            rollout_std_main = float(np.std(rollout_actions_arr[:, 0])) if len(rollout_actions_arr) else 0.0
            rollout_std_theta_deg = float(np.std(rollout_actions_arr[:, 1])) if len(rollout_actions_arr) else 0.0

            mean_ep_return = float(np.mean(recent_returns[-20:])) if recent_returns else float(np.mean(ep_returns))
            mean_ep_cost = float(np.mean(recent_costs[-20:])) if recent_costs else float(np.mean(ep_costs))
            success_rate = float(np.mean(recent_success[-20:])) if recent_success else 0.0
            crash_rate = float(np.mean(recent_crashes[-20:])) if recent_crashes else 0.0

            history.epoch.append(epoch + 1)
            history.global_step.append(global_step)
            history.mean_episode_return.append(mean_ep_return)
            history.mean_episode_cost.append(mean_ep_cost)
            history.success_rate.append(success_rate)
            history.crash_rate.append(crash_rate)
            history.lambda_value.append(float(lambda_value))
            history.eval_return.append(float(eval_metrics["return"]))
            history.eval_cost.append(float(eval_metrics["cost"]))
            history.eval_success.append(float(eval_metrics.get("success", np.nan)))
            history.eval_crashed.append(float(eval_metrics.get("crashed", np.nan)))
            history.policy_std_main.append(float(current_std[0]))
            history.policy_std_theta_deg.append(float(current_std[1] * cfg.theta_limit_deg))
            history.rollout_std_main.append(rollout_std_main)
            history.rollout_std_theta_deg.append(rollout_std_theta_deg)

            print(
                f"[Epoch {epoch+1:04d}] steps={global_step} "
                f"ret={mean_ep_return:8.2f} cost={mean_ep_cost:6.2f} "
                f"succ={success_rate:5.2f} crash={crash_rate:5.2f} "
                f"lambda={lambda_value:7.4f} KL={approx_kl_epoch:8.5f} "
                f"std_main={current_std[0]:6.3f} std_theta_deg={current_std[1]*cfg.theta_limit_deg:6.3f}"
            )

            ckpt = {
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_rms": obs_rms.state_dict() if obs_rms is not None else None,
                "config": asdict(cfg),
                "global_step": global_step,
                "epoch": epoch + 1,
                "lambda_value": float(lambda_value),
                "next_reset_seed": int(next_reset_seed),
            }
            torch.save(ckpt, paths.checkpoint_path)
            if obs_rms is not None:
                save_obs_norm(obs_rms, paths.norm_path)

            eval_return = float(eval_metrics.get("return", np.nan))
            eval_success = float(eval_metrics.get("success", np.nan))
            eval_crashed = float(eval_metrics.get("crashed", np.nan))
            if np.isfinite(eval_return):
                current_score = ((eval_success if np.isfinite(eval_success) else -float("inf")), eval_return)
                if (not cfg.best_model_select_by_success_then_return and np.isfinite(eval_return)):
                    current_score = (-float("inf"), eval_return)
                if current_score > best_score:
                    best_score = current_score
                    torch.save(ckpt, paths.best_checkpoint_path)
                    if obs_rms is not None:
                        save_obs_norm(obs_rms, paths.best_norm_path)
                    with open(paths.best_metrics_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "epoch": int(epoch + 1),
                            "global_step": int(global_step),
                            "eval_return": eval_return,
                            "eval_cost": float(eval_metrics.get("cost", np.nan)),
                            "eval_success": eval_success,
                            "eval_crashed": eval_crashed,
                            "train_success_rate": float(success_rate),
                            "train_crash_rate": float(crash_rate),
                            "best_score": [float(best_score[0]), float(best_score[1])],
                        }, f, indent=2)

            save_history_csv(history, paths.csv_path)

        plot_history(history, paths, show=cfg.show_plots)
        print("\nTraining finished.")
        print(f"Model saved to:      {paths.checkpoint_path}")
        if obs_rms is not None:
            print(f"Obs norm saved to:   {paths.norm_path}")
        print(f"Curves saved to:     {paths.learning_curve_path}")
        print(f"Action std plot to:  {paths.action_std_path}")
        print(f"History CSV saved:   {paths.csv_path}")
        if best_score[1] > -float("inf"):
            print(f"Best model saved to: {paths.best_checkpoint_path}")
            if obs_rms is not None:
                print(f"Best obs norm to:    {paths.best_norm_path}")
            print(f"Best metrics saved:  {paths.best_metrics_path}")
    finally:
        close_envs(envs)


def evaluate_from_checkpoint(cfg: Config) -> None:
    if not cfg.checkpoint:
        raise ValueError("For --mode eval you must pass --checkpoint PATH")

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg_dict = ckpt.get("config", {})
    # Use the checkpoint's core model sizes unless you intentionally changed them.
    # Use the checkpoint's core model sizes unless you intentionally changed them.
    merged = asdict(cfg)
    
    # Add eval-specific arguments to the exclusion set so CLI args override the checkpoint
    excluded_keys = {
        "mode", "render", "show_plots", "checkpoint", 
        "warm_start_path", "resume_optimizer", "true_resume",
        "eval_episodes", "max_episode_steps", "deterministic_eval"
    }
    
    for key, value in ckpt_cfg_dict.items():
        if key not in excluded_keys:
            merged[key] = value
    cfg = Config(**merged)
    # merged = asdict(cfg)
    # for key, value in ckpt_cfg_dict.items():
    #     if key not in {"mode", "render", "show_plots", "checkpoint", "warm_start_path", "resume_optimizer", "true_resume"}:
    #         merged[key] = value
    # cfg = Config(**merged)

    env = make_env(cfg, render_mode="human" if cfg.render else None, apply_training_reward_shaping=False)()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    agent = ActorCritic(obs_dim, act_dim, cfg, action_scale=np.array([1.0, math.radians(cfg.theta_limit_deg)], dtype=np.float32)).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    obs_rms = None
    if cfg.use_obs_norm and ckpt.get("obs_rms", None) is not None:
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        obs_rms.load_state_dict(ckpt["obs_rms"])

    metrics = evaluate_policy(cfg, agent, obs_rms, device, num_episodes=cfg.eval_episodes, render=cfg.render)
    print("\nEvaluation summary")
    for k, v in metrics.items():
        print(f"  {k:>8s}: {v:.4f}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train or evaluate reduced-order Lunar Lander with PPO / PPO-Lagrangian")

    # General
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\rabiei\My Research\LunarLander_v1")
    parser.add_argument("--run_name_prefix", type=str, default="reduced_lander")

    # Environment
    parser.add_argument("--gravity", type=float, default=-10.0)
    parser.add_argument("--enable_wind", type=str2bool, default=False)
    parser.add_argument("--wind_power", type=float, default=0.0)
    parser.add_argument("--turbulence_power", type=float, default=0.0)
    parser.add_argument("--theta_limit_deg", type=float, default=35.0)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--safety_x_bound", type=float, default=0.90)
    parser.add_argument("--include_leg_contacts_in_obs", type=str2bool, default=True)
    parser.add_argument("--reduced_inner_kp", type=float, default=2.0)
    parser.add_argument("--reduced_inner_kd", type=float, default=0.5)
    parser.add_argument("--reduced_min_side_action", type=float, default=0.55)
    parser.add_argument("--reduced_max_side_action", type=float, default=1.0)
    parser.add_argument("--reduced_side_deadband", type=float, default=0.05)
    parser.add_argument("--reduced_zero_side_on_contact", type=str2bool, default=False)
    parser.add_argument("--reduced_side_sign", type=int, default=0)
    parser.add_argument("--settle_mode_enabled", type=str2bool, default=True)
    parser.add_argument("--settle_force_side_off", type=str2bool, default=True)
    parser.add_argument("--settle_require_leg_contact", type=str2bool, default=True)
    parser.add_argument("--settle_require_both_legs_contact", type=str2bool, default=True)
    parser.add_argument("--settle_enter_steps", type=int, default=3)
    parser.add_argument("--settle_exit_steps", type=int, default=2)
    parser.add_argument("--settle_enter_main_threshold", type=float, default=0.10)
    parser.add_argument("--settle_exit_main_threshold", type=float, default=0.10)
    parser.add_argument("--settle_enter_side_threshold", type=float, default=0.50)
    parser.add_argument("--settle_exit_side_threshold", type=float, default=0.55)
    parser.add_argument("--settle_enter_vx_threshold", type=float, default=0.20)
    parser.add_argument("--settle_enter_vy_threshold", type=float, default=0.20)
    parser.add_argument("--settle_enter_omega_threshold", type=float, default=0.20)
    parser.add_argument("--settle_exit_vx_threshold", type=float, default=0.35)
    parser.add_argument("--settle_exit_vy_threshold", type=float, default=0.35)
    parser.add_argument("--settle_exit_omega_threshold", type=float, default=0.35)
    parser.add_argument("--timeout_penalty_on_truncation", type=float, default=50.0)
    parser.add_argument("--touchdown_discovery_bonus", type=float, default=25.0)
    parser.add_argument("--touchdown_bonus_vx_threshold", type=float, default=0.35)
    parser.add_argument("--touchdown_bonus_vy_threshold", type=float, default=0.35)
    parser.add_argument("--touchdown_bonus_omega_threshold", type=float, default=0.35)
    parser.add_argument("--settle_enter_bonus", type=float, default=10.0)
    parser.add_argument("--best_model_select_by_success_then_return", type=str2bool, default=True)


    # PPO
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--steps_per_epoch", type=int, default=4096)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--update_epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cost_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--cost_vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--target_kl", type=float, default=0.03)

    # Safety / lambda
    parser.add_argument("--use_safety", type=str2bool, default=True)
    parser.add_argument("--lambda_init", type=float, default=0.0)
    parser.add_argument("--lambda_lr", type=float, default=5e-3)
    parser.add_argument("--cost_limit", type=float, default=3.0)

    # Policy
    parser.add_argument("--hidden_sizes", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "elu"])
    parser.add_argument("--std_mode", type=str, default="anneal", choices=["fixed", "anneal", "learned"])
    parser.add_argument("--init_action_std", type=float, nargs=2, default=[0.35, 0.20])
    parser.add_argument("--final_action_std", type=float, nargs=2, default=[0.08, 0.05])

    # Normalization and eval/logging
    parser.add_argument("--use_obs_norm", type=str2bool, default=True)
    parser.add_argument("--obs_clip", type=float, default=10.0)
    parser.add_argument("--eval_every_epochs", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--render", type=str2bool, default=False)
    parser.add_argument("--show_plots", type=str2bool, default=True)
    parser.add_argument("--deterministic_eval", type=str2bool, default=True)

    # Warm start / eval
    parser.add_argument("--warm_start_path", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume_optimizer", type=str2bool, default=False)
    parser.add_argument("--true_resume", type=str2bool, default=False)

    args = parser.parse_args()
    cfg = Config(
        mode=args.mode,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        run_name_prefix=args.run_name_prefix,
        gravity=args.gravity,
        enable_wind=args.enable_wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power,
        theta_limit_deg=args.theta_limit_deg,
        max_episode_steps=args.max_episode_steps,
        safety_x_bound=args.safety_x_bound,
        include_leg_contacts_in_obs=args.include_leg_contacts_in_obs,
        reduced_inner_kp=args.reduced_inner_kp,
        reduced_inner_kd=args.reduced_inner_kd,
        reduced_min_side_action=args.reduced_min_side_action,
        reduced_max_side_action=args.reduced_max_side_action,
        reduced_side_deadband=args.reduced_side_deadband,
        reduced_zero_side_on_contact=args.reduced_zero_side_on_contact,
        reduced_side_sign=args.reduced_side_sign,
        settle_mode_enabled=args.settle_mode_enabled,
        settle_force_side_off=args.settle_force_side_off,
        settle_require_leg_contact=args.settle_require_leg_contact,
        settle_require_both_legs_contact=args.settle_require_both_legs_contact,
        settle_enter_steps=args.settle_enter_steps,
        settle_exit_steps=args.settle_exit_steps,
        settle_enter_main_threshold=args.settle_enter_main_threshold,
        settle_exit_main_threshold=args.settle_exit_main_threshold,
        settle_enter_side_threshold=args.settle_enter_side_threshold,
        settle_exit_side_threshold=args.settle_exit_side_threshold,
        settle_enter_vx_threshold=args.settle_enter_vx_threshold,
        settle_enter_vy_threshold=args.settle_enter_vy_threshold,
        settle_enter_omega_threshold=args.settle_enter_omega_threshold,
        settle_exit_vx_threshold=args.settle_exit_vx_threshold,
        settle_exit_vy_threshold=args.settle_exit_vy_threshold,
        settle_exit_omega_threshold=args.settle_exit_omega_threshold,
        timeout_penalty_on_truncation=args.timeout_penalty_on_truncation,
        touchdown_discovery_bonus=args.touchdown_discovery_bonus,
        touchdown_bonus_vx_threshold=args.touchdown_bonus_vx_threshold,
        touchdown_bonus_vy_threshold=args.touchdown_bonus_vy_threshold,
        touchdown_bonus_omega_threshold=args.touchdown_bonus_omega_threshold,
        settle_enter_bonus=args.settle_enter_bonus,
        best_model_select_by_success_then_return=args.best_model_select_by_success_then_return,
        total_steps=args.total_steps,
        steps_per_epoch=args.steps_per_epoch,
        num_envs=args.num_envs,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        cost_gamma=args.cost_gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        cost_vf_coef=args.cost_vf_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        use_safety=args.use_safety,
        lambda_init=args.lambda_init,
        lambda_lr=args.lambda_lr,
        cost_limit=args.cost_limit,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        std_mode=args.std_mode,
        init_action_std=tuple(args.init_action_std),
        final_action_std=tuple(args.final_action_std),
        use_obs_norm=args.use_obs_norm,
        obs_clip=args.obs_clip,
        eval_every_epochs=args.eval_every_epochs,
        eval_episodes=args.eval_episodes,
        render=args.render,
        show_plots=args.show_plots,
        deterministic_eval=args.deterministic_eval,
        warm_start_path=args.warm_start_path,
        checkpoint=args.checkpoint,
        resume_optimizer=args.resume_optimizer,
        true_resume=args.true_resume,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate_from_checkpoint(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
