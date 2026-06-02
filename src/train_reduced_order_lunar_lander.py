from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
try:
    from lunar_lander import LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN, FPS, MAIN_ENGINE_POWER, MAIN_ENGINE_Y_LOCATION
except ImportError:
    from gymnasium.envs.box2d.lunar_lander import LunarLander, VIEWPORT_W, VIEWPORT_H, SCALE, LEG_DOWN, FPS, MAIN_ENGINE_POWER, MAIN_ENGINE_Y_LOCATION

@dataclass
class Config:
    mode: str = 'train'
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir: str = 'runs_reduced_order_lunar_lander'
    run_name: str = 'reduced_order_lunar_lander'
    theta_limit_deg: float = 35.0
    gravity: float = -10.0
    enable_wind: bool = False
    wind_power: float = 0.0
    turbulence_power: float = 0.0
    max_episode_steps: int = 1000
    timeout_penalty: float = 0.0
    step_penalty: float = 0.02
    include_prev_theta_in_obs: bool = True
    include_contacts_in_obs: bool = True
    lateral_stress_reset: bool = False
    stress_x_min: float = -0.15
    stress_x_max: float = 0.15
    stress_vx_min: float = -0.8
    stress_vx_max: float = 0.8
    variation_lambda: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    total_steps: int = 500000
    num_envs: int = 8
    steps_per_rollout: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 256
    learning_rate: float = 0.0003
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden_size: int = 256
    std_mode: str = 'anneal'
    init_std_main: float = 0.35
    init_std_theta: float = 0.25
    final_std_main: float = 0.08
    final_std_theta: float = 0.05
    obs_norm: bool = True
    obs_clip: float = 10.0
    eval_every_rollouts: int = 10
    eval_episodes: int = 20
    final_eval_episodes: int = 333
    fixed_eval_seed: int = 20260527
    fixed_eval_seed_file: str = ''
    deterministic_eval: bool = True
    checkpoint: str = ''
    warm_start_path: str = ''

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def str2bool(x):
    if isinstance(x, bool):
        return x
    if str(x).lower() in ['1', 'true', 'yes', 'y']:
        return True
    if str(x).lower() in ['0', 'false', 'no', 'n']:
        return False
    raise argparse.ArgumentTypeError(f'Cannot parse boolean value: {x}')

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_json(path: Path, obj) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

class RunningMeanStd:

    def __init__(self, shape: Sequence[int], epsilon: float=0.0001):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

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
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = float(total_count)

    def normalize(self, x: np.ndarray, clip: float) -> np.ndarray:
        z = (x - self.mean) / np.sqrt(self.var + 1e-08)
        return np.clip(z, -clip, clip).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {'mean': self.mean, 'var': self.var, 'count': np.array([self.count], dtype=np.float64)}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(state['mean'], dtype=np.float64)
        self.var = np.asarray(state['var'], dtype=np.float64)
        self.count = float(np.asarray(state['count']).reshape(-1)[0])

class ReducedOrderLunarLanderEnv(LunarLander):

    def __init__(self, render_mode: Optional[str]=None, theta_limit_deg: float=35.0, gravity: float=-10.0, enable_wind: bool=False, wind_power: float=0.0, turbulence_power: float=0.0, variation_lambda: float=0.0, step_penalty: float=0.02, include_prev_theta_in_obs: bool=True, include_contacts_in_obs: bool=True, lateral_stress_reset: bool=False, stress_x_min: float=-0.15, stress_x_max: float=0.15, stress_vx_min: float=-0.8, stress_vx_max: float=0.8):
        super().__init__(render_mode=render_mode, continuous=True, gravity=gravity, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
        self.theta_limit_rad = math.radians(theta_limit_deg)
        self.variation_lambda = float(variation_lambda)
        self.step_penalty = float(step_penalty)
        self.include_prev_theta_in_obs = bool(include_prev_theta_in_obs)
        self.include_contacts_in_obs = bool(include_contacts_in_obs)
        self.lateral_stress_reset = bool(lateral_stress_reset)
        self.stress_x_min = float(stress_x_min)
        self.stress_x_max = float(stress_x_max)
        self.stress_vx_min = float(stress_vx_min)
        self.stress_vx_max = float(stress_vx_max)
        self.lateral_stress_x0 = 0.0
        self.lateral_stress_vx0 = 0.0
        self.prev_theta_star = 0.0
        self.last_theta_star = 0.0
        self.episode_task_return = 0.0
        self.episode_train_return = 0.0
        self.episode_variation = 0.0
        self.episode_len = 0
        from gymnasium import spaces
        obs_low_parts = [np.array([-2.5, -2.5, -10.0, -10.0], dtype=np.float32)]
        obs_high_parts = [np.array([2.5, 2.5, 10.0, 10.0], dtype=np.float32)]
        if self.include_prev_theta_in_obs:
            obs_low_parts.append(np.array([-self.theta_limit_rad], dtype=np.float32))
            obs_high_parts.append(np.array([self.theta_limit_rad], dtype=np.float32))
        if self.include_contacts_in_obs:
            obs_low_parts.append(np.array([0.0, 0.0], dtype=np.float32))
            obs_high_parts.append(np.array([1.0, 1.0], dtype=np.float32))
        self.observation_space = spaces.Box(low=np.concatenate(obs_low_parts), high=np.concatenate(obs_high_parts), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        self.prev_theta_star = 0.0
        self.last_theta_star = 0.0
        self.episode_task_return = 0.0
        self.episode_train_return = 0.0
        self.episode_variation = 0.0
        self.episode_len = 0
        obs8, info = super().reset(seed=seed, options=options)
        self.prev_theta_star = 0.0
        self.last_theta_star = 0.0
        self.episode_task_return = 0.0
        self.episode_train_return = 0.0
        self.episode_variation = 0.0
        self.episode_len = 0
        if self.lateral_stress_reset:
            self._apply_lateral_stress_reset()
        obs8 = self._stock_state_with_imposed_angle(theta_star=0.0)
        if self.lateral_stress_reset:
            self.prev_shaping = self._shaping_from_state(obs8)
        return (self._reduced_obs_from_stock(obs8), info)

    def _apply_lateral_stress_reset(self) -> None:
        x0 = float(self.np_random.uniform(self.stress_x_min, self.stress_x_max))
        vx0 = float(self.np_random.uniform(self.stress_vx_min, self.stress_vx_max))
        center_x = VIEWPORT_W / SCALE / 2
        half_width = VIEWPORT_W / SCALE / 2
        target_x = center_x + x0 * half_width
        target_vx = vx0 * FPS / half_width
        dx = target_x - float(self.lander.position.x)
        for body in [self.lander] + list(self.legs):
            body.position = (float(body.position.x + dx), float(body.position.y))
            body.linearVelocity = (float(target_vx), float(body.linearVelocity.y))
            body.awake = True
        self.lateral_stress_x0 = x0
        self.lateral_stress_vx0 = vx0

    def _shaping_from_state(self, state: np.ndarray) -> float:
        return float(-100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7])

    def _impose_attitude(self, theta_star: float) -> None:
        if self.lander is None:
            return
        self.lander.angle = float(theta_star)
        self.lander.angularVelocity = 0.0

    def _stock_state_with_imposed_angle(self, theta_star: float) -> np.ndarray:
        self._impose_attitude(theta_star)
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = np.array([(pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2), (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2), vel.x * (VIEWPORT_W / SCALE / 2) / FPS, vel.y * (VIEWPORT_H / SCALE / 2) / FPS, float(theta_star), 0.0, 1.0 if self.legs[0].ground_contact else 0.0, 1.0 if self.legs[1].ground_contact else 0.0], dtype=np.float32)
        return state

    def _reduced_obs_from_stock(self, state8: np.ndarray) -> np.ndarray:
        parts = [state8[0:4]]
        if self.include_prev_theta_in_obs:
            parts.append(np.array([self.last_theta_star], dtype=np.float32))
        if self.include_contacts_in_obs:
            parts.append(state8[6:8])
        return np.concatenate(parts).astype(np.float32)

    def step(self, action):
        assert self.lander is not None, 'Call reset before step.'
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        main_cmd = float(action[0])
        theta_star = float(action[1]) * self.theta_limit_rad
        if self.enable_wind and (not (self.legs[0].ground_contact or self.legs[1].ground_contact)):
            wind_mag = math.tanh(math.sin(0.02 * self.wind_idx) + math.sin(math.pi * 0.01 * self.wind_idx)) * self.wind_power
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)
            torque_mag = math.tanh(math.sin(0.02 * self.torque_idx) + math.sin(math.pi * 0.01 * self.torque_idx)) * self.turbulence_power
            self.torque_idx += 1
        self._impose_attitude(theta_star)
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
        m_power = 0.0
        if main_cmd > 0.0:
            m_power = (np.clip(main_cmd, 0.0, 1.0) + 1.0) * 0.5
            ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
        s_power = 0.0
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self._impose_attitude(theta_star)
        state = self._stock_state_with_imposed_angle(theta_star)
        shaping = -100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]
        task_reward = 0.0
        if self.prev_shaping is not None:
            task_reward = float(shaping - self.prev_shaping)
        self.prev_shaping = shaping
        task_reward -= float(m_power * 0.3)
        terminated = False
        success = False
        crash = False
        out_of_bounds = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            crash = bool(self.game_over)
            out_of_bounds = bool(abs(state[0]) >= 1.0)
            task_reward = -100.0
        if not self.lander.awake:
            terminated = True
            success = True
            task_reward = +100.0
        variation_cost = abs(theta_star - self.prev_theta_star)
        train_reward = task_reward - self.variation_lambda * variation_cost - self.step_penalty
        self.episode_task_return += float(task_reward)
        self.episode_train_return += float(train_reward)
        self.episode_variation += float(variation_cost)
        self.episode_len += 1
        theta_tol = math.radians(2.0)
        dtheta_tol = math.radians(1.0)
        vel_tol = 0.05
        quiet_theta_star = abs(theta_star) < theta_tol
        quiet_theta_variation = variation_cost < dtheta_tol
        both_legs_contact = bool(state[6] > 0.5 and state[7] > 0.5)
        low_velocity = bool(math.sqrt(float(state[2]) ** 2 + float(state[3]) ** 2) < vel_tol)
        lander_awake = bool(self.lander.awake)
        self.prev_theta_star = theta_star
        self.last_theta_star = theta_star
        info = {'task_reward': float(task_reward), 'train_reward': float(train_reward), 'variation_cost': float(variation_cost), 'theta_star': float(theta_star), 'theta_star_norm': float(action[1]), 'main_cmd': float(main_cmd), 'm_power': float(m_power), 's_power': float(s_power), 'step_penalty': float(self.step_penalty), 'lateral_stress_reset': bool(self.lateral_stress_reset), 'lateral_stress_x0': float(self.lateral_stress_x0), 'lateral_stress_vx0': float(self.lateral_stress_vx0), 'success': bool(success), 'crash': bool(crash), 'out_of_bounds': bool(out_of_bounds), 'lander_awake': lander_awake, 'not_awake_success_condition': bool(not lander_awake), 'both_legs_contact': both_legs_contact, 'low_velocity': low_velocity, 'quiet_theta_star': bool(quiet_theta_star), 'quiet_theta_variation': bool(quiet_theta_variation)}
        if terminated:
            info['episode'] = {'task_return': float(self.episode_task_return), 'train_return': float(self.episode_train_return), 'variation': float(self.episode_variation), 'length': int(self.episode_len), 'success': bool(success), 'crash': bool(crash), 'out_of_bounds': bool(out_of_bounds)}
        return (self._reduced_obs_from_stock(state), float(train_reward), terminated, False, info)

class TimeLimitReducedOrder:

    def __init__(self, env: ReducedOrderLunarLanderEnv, max_episode_steps: int, timeout_penalty: float=0.0):
        self.env = env
        self.max_episode_steps = int(max_episode_steps)
        self.timeout_penalty = float(timeout_penalty)
        self.elapsed_steps = 0
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        self.elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps and (not terminated):
            truncated = True
            info = dict(info)
            penalty = float(self.timeout_penalty)
            reward = float(reward) - penalty
            self.env.episode_train_return -= penalty
            info['time_limit'] = True
            info['train_reward'] = float(info.get('train_reward', reward + penalty)) - penalty
            info['timeout_penalty'] = penalty
            info['episode'] = {'task_return': float(self.env.episode_task_return), 'train_return': float(self.env.episode_train_return), 'variation': float(self.env.episode_variation), 'length': int(self.env.episode_len), 'success': False, 'crash': False, 'out_of_bounds': False}
        return (obs, reward, terminated, truncated, info)

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

def make_env(cfg: Config, render_mode: Optional[str]=None) -> TimeLimitReducedOrder:
    env = ReducedOrderLunarLanderEnv(render_mode=render_mode, theta_limit_deg=cfg.theta_limit_deg, gravity=cfg.gravity, enable_wind=cfg.enable_wind, wind_power=cfg.wind_power, turbulence_power=cfg.turbulence_power, variation_lambda=cfg.variation_lambda, step_penalty=cfg.step_penalty, include_prev_theta_in_obs=cfg.include_prev_theta_in_obs, include_contacts_in_obs=cfg.include_contacts_in_obs, lateral_stress_reset=cfg.lateral_stress_reset, stress_x_min=cfg.stress_x_min, stress_x_max=cfg.stress_x_max, stress_vx_min=cfg.stress_vx_min, stress_vx_max=cfg.stress_vx_max)
    return TimeLimitReducedOrder(env, cfg.max_episode_steps, cfg.timeout_penalty)

class SquashedNormal:

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, eps: float=1e-06):
        self.loc = loc
        self.scale = scale
        self.base = Normal(loc, scale)
        self.eps = eps

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.base.sample())

    def rsample(self) -> torch.Tensor:
        return torch.tanh(self.base.rsample())

    def deterministic(self) -> torch.Tensor:
        return torch.tanh(self.loc)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        v = torch.clamp(value, -1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = torch.atanh(v)
        logp = self.base.log_prob(pre_tanh) - torch.log(1.0 - v.pow(2) + self.eps)
        return logp.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.base.entropy().sum(dim=-1)

class ActorCritic(nn.Module):

    def __init__(self, obs_dim: int, hidden_size: int=256, std_mode: str='anneal'):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.std_mode = std_mode
        if std_mode == 'learned':
            self.log_std = nn.Parameter(torch.log(torch.tensor([0.25, 0.15], dtype=torch.float32)))
        else:
            self.register_buffer('fixed_std', torch.tensor([0.25, 0.15], dtype=torch.float32))

    def set_action_std(self, std: Sequence[float]) -> None:
        std_t = torch.tensor(std, dtype=torch.float32, device=next(self.parameters()).device)
        if self.std_mode == 'learned':
            with torch.no_grad():
                self.log_std.copy_(torch.log(torch.clamp(std_t, min=0.001)))
        else:
            self.fixed_std = std_t

    def dist(self, obs: torch.Tensor) -> SquashedNormal:
        mu = self.actor(obs)
        if self.std_mode == 'learned':
            std = torch.exp(self.log_std).expand_as(mu)
        else:
            std = self.fixed_std.to(obs.device).expand_as(mu)
        return SquashedNormal(mu, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor]=None, deterministic: bool=False):
        dist = self.dist(obs)
        if action is None:
            action = dist.deterministic() if deterministic else dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(obs)
        return (action, logp, entropy, value)

def make_fixed_eval_seeds(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2 ** 31 - 1, size=n, dtype=np.int64)

def load_or_create_eval_seeds(cfg: Config, run_dir: Path) -> np.ndarray:
    if cfg.fixed_eval_seed_file:
        path = Path(cfg.fixed_eval_seed_file)
    else:
        path = run_dir / f'fixed_stock_reset_seeds_{cfg.final_eval_episodes}.npy'
    if path.exists():
        seeds = np.load(path).astype(np.int64)
        if len(seeds) < cfg.final_eval_episodes:
            raise ValueError(f'Seed file {path} has only {len(seeds)} seeds.')
        return seeds[:cfg.final_eval_episodes]
    seeds = make_fixed_eval_seeds(cfg.final_eval_episodes, cfg.fixed_eval_seed)
    np.save(path, seeds)
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['episode_index', 'reset_seed'])
        for i, s in enumerate(seeds):
            writer.writerow([i, int(s)])
    return seeds

def save_initial_observations_for_seeds(cfg: Config, seeds: np.ndarray, run_dir: Path) -> None:
    env = make_env(cfg)
    rows = []
    for i, s in enumerate(seeds):
        obs, _ = env.reset(seed=int(s))
        rows.append([i, int(s), float(env.env.lateral_stress_x0), float(env.env.lateral_stress_vx0)] + [float(x) for x in obs])
    env.close()
    csv_path = run_dir / 'fixed_stock_reset_initial_observations.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        obs_cols = [f'obs_{j}' for j in range(len(rows[0]) - 4)] if rows else []
        writer.writerow(['episode_index', 'reset_seed', 'stress_x0', 'stress_vx0'] + obs_cols)
        writer.writerows(rows)

@torch.no_grad()
def evaluate(cfg: Config, model: ActorCritic, obs_rms: Optional[RunningMeanStd], seeds: Sequence[int], run_dir: Optional[Path]=None, prefix: str='eval', save_trajectory: bool=False) -> Dict[str, float]:
    env = make_env(cfg)
    device = cfg.device
    model.eval()
    episode_rows = []
    trajectory_rows = []
    for ep, reset_seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(reset_seed))
        done = False
        disc = 1.0
        discounted_task = 0.0
        discounted_variation = 0.0
        discounted_train = 0.0
        step_id = 0
        ep_info = None
        dist_values = []
        speed_values = []
        abs_x_values = []
        y_values = []
        m_power_values = []
        main_cmd_values = []
        theta_abs_values = []
        theta_var_values = []
        both_contact_values = []
        low_velocity_values = []
        lander_awake_values = []
        while not done:
            obs_in = obs
            if obs_rms is not None:
                obs_in = obs_rms.normalize(obs_in, cfg.obs_clip)
            obs_t = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = model.get_action_and_value(obs_t, deterministic=cfg.deterministic_eval)
            act = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(act)
            x_val = float(next_obs[0])
            y_val = float(next_obs[1])
            vx_val = float(next_obs[2])
            vy_val = float(next_obs[3])
            dist_to_pad = math.sqrt(x_val * x_val + y_val * y_val)
            speed = math.sqrt(vx_val * vx_val + vy_val * vy_val)
            dist_values.append(dist_to_pad)
            speed_values.append(speed)
            abs_x_values.append(abs(x_val))
            y_values.append(y_val)
            m_power_values.append(float(info.get('m_power', 0.0)))
            main_cmd_values.append(float(info.get('main_cmd', 0.0)))
            theta_abs_values.append(abs(float(info.get('theta_star', 0.0))))
            theta_var_values.append(float(info.get('variation_cost', 0.0)))
            both_contact_values.append(1.0 if bool(info.get('both_legs_contact', False)) else 0.0)
            low_velocity_values.append(1.0 if bool(info.get('low_velocity', False)) else 0.0)
            lander_awake_values.append(1.0 if bool(info.get('lander_awake', True)) else 0.0)
            done = bool(terminated or truncated)
            discounted_task += disc * float(info.get('task_reward', reward))
            discounted_variation += disc * float(info.get('variation_cost', 0.0))
            discounted_train += disc * float(reward)
            if save_trajectory:
                trajectory_rows.append({'episode': ep, 'reset_seed': int(reset_seed), 'step': step_id, 'obs': obs.tolist(), 'main_cmd': float(act[0]), 'theta_star_norm': float(act[1]), 'theta_star': float(info.get('theta_star', 0.0)), 'task_reward': float(info.get('task_reward', reward)), 'variation_cost': float(info.get('variation_cost', 0.0)), 'train_reward': float(reward), 'distance_to_pad': float(dist_to_pad), 'speed': float(speed), 'abs_x': float(abs(x_val)), 'y': float(y_val), 'success': bool(info.get('success', False)), 'crash': bool(info.get('crash', False)), 'out_of_bounds': bool(info.get('out_of_bounds', False)), 'lander_awake': bool(info.get('lander_awake', True)), 'both_legs_contact': bool(info.get('both_legs_contact', False)), 'low_velocity': bool(info.get('low_velocity', False)), 'quiet_theta_star': bool(info.get('quiet_theta_star', False)), 'quiet_theta_variation': bool(info.get('quiet_theta_variation', False)), 'm_power': float(info.get('m_power', 0.0)), 'main_cmd_info': float(info.get('main_cmd', 0.0))})
            obs = next_obs
            disc *= cfg.gamma
            step_id += 1
            if done:
                ep_info = info.get('episode', {})
        if dist_values:
            min_distance_to_pad = float(np.min(dist_values))
            mean_distance_to_pad = float(np.mean(dist_values))
            final_distance_to_pad = float(dist_values[-1])
            mean_speed = float(np.mean(speed_values))
            final_speed = float(speed_values[-1])
            min_abs_x = float(np.min(abs_x_values))
            final_abs_x = float(abs_x_values[-1])
            final_y = float(y_values[-1])
            near_pad_fraction_0p2 = float(np.mean(np.asarray(dist_values) < 0.2))
            near_pad_fraction_0p1 = float(np.mean(np.asarray(dist_values) < 0.1))
            final_m_power = float(m_power_values[-1])
            mean_m_power = float(np.mean(m_power_values))
            last50_mean_m_power = float(np.mean(m_power_values[-min(50, len(m_power_values)):]))
            final_main_cmd = float(main_cmd_values[-1])
            positive_main_fraction = float(np.mean(np.asarray(main_cmd_values) > 0.0))
            final_theta_abs = float(theta_abs_values[-1])
            mean_theta_abs = float(np.mean(theta_abs_values))
            final_theta_variation = float(theta_var_values[-1])
            mean_theta_variation = float(np.mean(theta_var_values))
            final_both_legs_contact = float(both_contact_values[-1])
            both_legs_contact_fraction = float(np.mean(both_contact_values))
            ever_both_legs_contact = float(np.max(both_contact_values))
            low_velocity_fraction = float(np.mean(low_velocity_values))
            final_lander_awake = float(lander_awake_values[-1])
            landing_candidate = float(final_distance_to_pad < 0.1 and final_speed < 0.05 and final_both_legs_contact > 0.5)
        else:
            min_distance_to_pad = float('nan')
            mean_distance_to_pad = float('nan')
            final_distance_to_pad = float('nan')
            mean_speed = float('nan')
            final_speed = float('nan')
            min_abs_x = float('nan')
            final_abs_x = float('nan')
            final_y = float('nan')
            near_pad_fraction_0p2 = float('nan')
            near_pad_fraction_0p1 = float('nan')
            final_m_power = float('nan')
            mean_m_power = float('nan')
            last50_mean_m_power = float('nan')
            final_main_cmd = float('nan')
            positive_main_fraction = float('nan')
            final_theta_abs = float('nan')
            mean_theta_abs = float('nan')
            final_theta_variation = float('nan')
            mean_theta_variation = float('nan')
            final_both_legs_contact = float('nan')
            both_legs_contact_fraction = float('nan')
            ever_both_legs_contact = float('nan')
            low_velocity_fraction = float('nan')
            final_lander_awake = float('nan')
            landing_candidate = float('nan')
        episode_rows.append({'episode': ep, 'reset_seed': int(reset_seed), 'task_return': float(ep_info.get('task_return', np.nan)), 'train_return': float(ep_info.get('train_return', np.nan)), 'variation': float(ep_info.get('variation', np.nan)), 'length': int(ep_info.get('length', step_id)), 'success': bool(ep_info.get('success', False)), 'crash': bool(ep_info.get('crash', False)), 'out_of_bounds': bool(ep_info.get('out_of_bounds', False)), 'discounted_task_return': float(discounted_task), 'discounted_variation': float(discounted_variation), 'discounted_train_return': float(discounted_train), 'min_distance_to_pad': min_distance_to_pad, 'mean_distance_to_pad': mean_distance_to_pad, 'final_distance_to_pad': final_distance_to_pad, 'mean_speed': mean_speed, 'final_speed': final_speed, 'min_abs_x': min_abs_x, 'final_abs_x': final_abs_x, 'final_y': final_y, 'near_pad_fraction_0p2': near_pad_fraction_0p2, 'near_pad_fraction_0p1': near_pad_fraction_0p1, 'final_m_power': final_m_power, 'mean_m_power': mean_m_power, 'last50_mean_m_power': last50_mean_m_power, 'final_main_cmd': final_main_cmd, 'positive_main_fraction': positive_main_fraction, 'final_theta_abs': final_theta_abs, 'mean_theta_abs': mean_theta_abs, 'final_theta_variation': final_theta_variation, 'mean_theta_variation': mean_theta_variation, 'final_both_legs_contact': final_both_legs_contact, 'both_legs_contact_fraction': both_legs_contact_fraction, 'ever_both_legs_contact': ever_both_legs_contact, 'low_velocity_fraction': low_velocity_fraction, 'final_lander_awake': final_lander_awake, 'landing_candidate': landing_candidate})
    env.close()
    summary = {'episodes': len(episode_rows), 'success_rate': float(np.mean([r['success'] for r in episode_rows])) if episode_rows else 0.0, 'crash_rate': float(np.mean([r['crash'] for r in episode_rows])) if episode_rows else 0.0, 'out_of_bounds_rate': float(np.mean([r['out_of_bounds'] for r in episode_rows])) if episode_rows else 0.0, 'mean_task_return': float(np.nanmean([r['task_return'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_train_return': float(np.nanmean([r['train_return'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_variation': float(np.nanmean([r['variation'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_length': float(np.mean([r['length'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_discounted_task_return': float(np.mean([r['discounted_task_return'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_discounted_variation': float(np.mean([r['discounted_variation'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_discounted_train_return': float(np.mean([r['discounted_train_return'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_min_distance_to_pad': float(np.nanmean([r['min_distance_to_pad'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_mean_distance_to_pad': float(np.nanmean([r['mean_distance_to_pad'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_distance_to_pad': float(np.nanmean([r['final_distance_to_pad'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_speed': float(np.nanmean([r['mean_speed'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_speed': float(np.nanmean([r['final_speed'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_min_abs_x': float(np.nanmean([r['min_abs_x'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_abs_x': float(np.nanmean([r['final_abs_x'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_y': float(np.nanmean([r['final_y'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_near_pad_fraction_0p2': float(np.nanmean([r['near_pad_fraction_0p2'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_near_pad_fraction_0p1': float(np.nanmean([r['near_pad_fraction_0p1'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_m_power': float(np.nanmean([r['final_m_power'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_m_power': float(np.nanmean([r['mean_m_power'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_last50_m_power': float(np.nanmean([r['last50_mean_m_power'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_main_cmd': float(np.nanmean([r['final_main_cmd'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_positive_main_fraction': float(np.nanmean([r['positive_main_fraction'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_theta_abs': float(np.nanmean([r['final_theta_abs'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_theta_abs': float(np.nanmean([r['mean_theta_abs'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_theta_variation': float(np.nanmean([r['final_theta_variation'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_theta_variation_step': float(np.nanmean([r['mean_theta_variation'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_both_legs_contact': float(np.nanmean([r['final_both_legs_contact'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_both_legs_contact_fraction': float(np.nanmean([r['both_legs_contact_fraction'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_ever_both_legs_contact': float(np.nanmean([r['ever_both_legs_contact'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_low_velocity_fraction': float(np.nanmean([r['low_velocity_fraction'] for r in episode_rows])) if episode_rows else float('nan'), 'mean_final_lander_awake': float(np.nanmean([r['final_lander_awake'] for r in episode_rows])) if episode_rows else float('nan'), 'landing_candidate_rate': float(np.nanmean([r['landing_candidate'] for r in episode_rows])) if episode_rows else float('nan')}
    if run_dir is not None:
        ep_csv = run_dir / f'{prefix}_episodes.csv'
        with ep_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
            writer.writeheader()
            writer.writerows(episode_rows)
        write_json(run_dir / f'{prefix}_summary.json', summary)
        if save_trajectory and trajectory_rows:
            traj_csv = run_dir / f'{prefix}_trajectory.csv'
            with traj_csv.open('w', newline='', encoding='utf-8') as f:
                fieldnames = list(trajectory_rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(trajectory_rows)
    model.train()
    return summary

def linear_std(cfg: Config, progress: float) -> Tuple[float, float]:
    progress = float(np.clip(progress, 0.0, 1.0))
    if cfg.std_mode == 'fixed':
        return (cfg.init_std_main, cfg.init_std_theta)
    return (cfg.init_std_main + progress * (cfg.final_std_main - cfg.init_std_main), cfg.init_std_theta + progress * (cfg.final_std_theta - cfg.init_std_theta))

def save_checkpoint(path: Path, model: ActorCritic, optimizer, obs_rms: Optional[RunningMeanStd], cfg: Config, meta: dict) -> None:
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer is not None else None, 'config': asdict(cfg), 'meta': meta}
    if obs_rms is not None:
        payload['obs_rms'] = obs_rms.state_dict()
    torch.save(payload, path)

def load_checkpoint(path: str, model: ActorCritic, optimizer=None, obs_rms: Optional[RunningMeanStd]=None, device: str='cpu') -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if obs_rms is not None and ckpt.get('obs_rms') is not None:
        obs_rms.load_state_dict(ckpt['obs_rms'])
    return ckpt

def train(cfg: Config) -> Path:
    set_seed(cfg.seed)
    run_dir = ensure_dir(Path(cfg.save_dir) / f"{cfg.run_name}_lam{cfg.variation_lambda:g}_{time.strftime('%Y%m%d_%H%M%S')}")
    write_json(run_dir / 'config.json', asdict(cfg))
    fixed_eval_seeds = load_or_create_eval_seeds(cfg, run_dir)
    save_initial_observations_for_seeds(cfg, fixed_eval_seeds, run_dir)
    envs = [make_env(cfg) for _ in range(cfg.num_envs)]
    obs_list = []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=cfg.seed + 1000 * i)
        obs_list.append(obs)
    obs = np.stack(obs_list).astype(np.float32)
    obs_dim = obs.shape[1]
    model = ActorCritic(obs_dim, cfg.hidden_size, cfg.std_mode).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-05)
    obs_rms = RunningMeanStd((obs_dim,)) if cfg.obs_norm else None
    if cfg.warm_start_path:
        load_checkpoint(cfg.warm_start_path, model, optimizer=None, obs_rms=obs_rms, device=cfg.device)
    num_updates = max(1, cfg.total_steps // (cfg.num_envs * cfg.steps_per_rollout))
    history = []
    global_step = 0
    for update in range(1, num_updates + 1):
        progress = (update - 1) / max(1, num_updates - 1)
        model.set_action_std(linear_std(cfg, progress))
        obs_buf = np.zeros((cfg.steps_per_rollout, cfg.num_envs, obs_dim), dtype=np.float32)
        action_buf = np.zeros((cfg.steps_per_rollout, cfg.num_envs, 2), dtype=np.float32)
        logp_buf = np.zeros((cfg.steps_per_rollout, cfg.num_envs), dtype=np.float32)
        reward_buf = np.zeros((cfg.steps_per_rollout, cfg.num_envs), dtype=np.float32)
        done_buf = np.zeros((cfg.steps_per_rollout, cfg.num_envs), dtype=np.float32)
        value_buf = np.zeros((cfg.steps_per_rollout, cfg.num_envs), dtype=np.float32)
        rollout_episodes = []
        for t in range(cfg.steps_per_rollout):
            if obs_rms is not None:
                obs_rms.update(obs)
                obs_in = obs_rms.normalize(obs, cfg.obs_clip)
            else:
                obs_in = obs
            obs_buf[t] = obs_in
            obs_t = torch.tensor(obs_in, dtype=torch.float32, device=cfg.device)
            with torch.no_grad():
                action_t, logp_t, _, value_t = model.get_action_and_value(obs_t)
            actions = action_t.cpu().numpy().astype(np.float32)
            action_buf[t] = actions
            logp_buf[t] = logp_t.cpu().numpy()
            value_buf[t] = value_t.cpu().numpy()
            next_obs = []
            rewards = []
            dones = []
            for i, env in enumerate(envs):
                o2, r, terminated, truncated, info = env.step(actions[i])
                done = bool(terminated or truncated)
                rewards.append(float(r))
                dones.append(float(done))
                if done:
                    if 'episode' in info:
                        rollout_episodes.append(info['episode'])
                    o2, _ = env.reset()
                next_obs.append(o2)
            reward_buf[t] = np.asarray(rewards, dtype=np.float32)
            done_buf[t] = np.asarray(dones, dtype=np.float32)
            obs = np.stack(next_obs).astype(np.float32)
            global_step += cfg.num_envs
        if obs_rms is not None:
            obs_in = obs_rms.normalize(obs, cfg.obs_clip)
        else:
            obs_in = obs
        with torch.no_grad():
            next_value = model.value(torch.tensor(obs_in, dtype=torch.float32, device=cfg.device)).cpu().numpy()
        advantages = np.zeros_like(reward_buf, dtype=np.float32)
        lastgaelam = np.zeros(cfg.num_envs, dtype=np.float32)
        for t in reversed(range(cfg.steps_per_rollout)):
            if t == cfg.steps_per_rollout - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_values = value_buf[t + 1]
            delta = reward_buf[t] + cfg.gamma * next_values * next_nonterminal - value_buf[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + value_buf
        b_obs = torch.tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=cfg.device)
        b_actions = torch.tensor(action_buf.reshape(-1, 2), dtype=torch.float32, device=cfg.device)
        b_logp = torch.tensor(logp_buf.reshape(-1), dtype=torch.float32, device=cfg.device)
        b_adv = torch.tensor(advantages.reshape(-1), dtype=torch.float32, device=cfg.device)
        b_returns = torch.tensor(returns.reshape(-1), dtype=torch.float32, device=cfg.device)
        b_values = torch.tensor(value_buf.reshape(-1), dtype=torch.float32, device=cfg.device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-08)
        batch_size = cfg.steps_per_rollout * cfg.num_envs
        inds = np.arange(batch_size)
        approx_kl = 0.0
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb = inds[start:start + cfg.minibatch_size]
                _, new_logp, entropy, new_value = model.get_action_and_value(b_obs[mb], b_actions[mb])
                logratio = new_logp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - logratio).mean().item()
                pg_loss1 = -b_adv[mb] * ratio
                pg_loss2 = -b_adv[mb] * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * (new_value - b_returns[mb]).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            if cfg.target_kl > 0 and approx_kl > cfg.target_kl:
                break
        ep_task = [e.get('task_return', np.nan) for e in rollout_episodes]
        ep_train = [e.get('train_return', np.nan) for e in rollout_episodes]
        ep_var = [e.get('variation', np.nan) for e in rollout_episodes]
        ep_success = [e.get('success', False) for e in rollout_episodes]
        row = {'update': update, 'global_step': global_step, 'mean_rollout_task_return': float(np.nanmean(ep_task)) if ep_task else float('nan'), 'mean_rollout_train_return': float(np.nanmean(ep_train)) if ep_train else float('nan'), 'mean_rollout_variation': float(np.nanmean(ep_var)) if ep_var else float('nan'), 'rollout_success_rate': float(np.mean(ep_success)) if ep_success else float('nan'), 'approx_kl': float(approx_kl), 'std_main': float(linear_std(cfg, progress)[0]), 'std_theta': float(linear_std(cfg, progress)[1])}
        if update % cfg.eval_every_rollouts == 0 or update == num_updates:
            eval_seeds = fixed_eval_seeds[:min(cfg.eval_episodes, len(fixed_eval_seeds))]
            summary = evaluate(cfg, model, obs_rms, eval_seeds, run_dir=run_dir, prefix=f'eval_update_{update:04d}')
            row.update({f'eval_{k}': v for k, v in summary.items()})
            print(f"update {update:04d}/{num_updates} step {global_step} eval_success={summary['success_rate']:.3f} eval_return={summary['mean_task_return']:.2f} eval_disc={summary['mean_discounted_task_return']:.2f} eval_var={summary['mean_discounted_variation']:.3f} eval_dist={summary['mean_final_distance_to_pad']:.3f} eval_min_dist={summary['mean_min_distance_to_pad']:.3f} eval_final_speed={summary['mean_final_speed']:.3f}")
        else:
            print(f"update {update:04d}/{num_updates} step {global_step} rollout_task={row['mean_rollout_task_return']:.2f} rollout_var={row['mean_rollout_variation']:.3f}")
        history.append(row)
        with (run_dir / 'training_history.csv').open('w', newline='', encoding='utf-8') as f:
            fieldnames = sorted({k for r in history for k in r.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        save_checkpoint(run_dir / 'last_model.pt', model, optimizer, obs_rms, cfg, {'update': update})
    for env in envs:
        env.close()
    save_checkpoint(run_dir / 'final_model.pt', model, optimizer, obs_rms, cfg, {'update': num_updates, 'global_step': global_step})
    final_summary = evaluate(cfg, model, obs_rms, fixed_eval_seeds, run_dir=run_dir, prefix='final_fixed333', save_trajectory=False)
    print('Final fixed evaluation:', json.dumps(final_summary, indent=2))
    return run_dir

def eval_only(cfg: Config) -> None:
    if not cfg.checkpoint:
        raise ValueError('--checkpoint is required in eval mode')
    run_dir = ensure_dir(Path(cfg.save_dir) / f"eval_{time.strftime('%Y%m%d_%H%M%S')}")
    tmp_env = make_env(cfg)
    obs, _ = tmp_env.reset(seed=cfg.seed)
    obs_dim = len(obs)
    tmp_env.close()
    model = ActorCritic(obs_dim, cfg.hidden_size, cfg.std_mode).to(cfg.device)
    obs_rms = RunningMeanStd((obs_dim,)) if cfg.obs_norm else None
    ckpt = load_checkpoint(cfg.checkpoint, model, optimizer=None, obs_rms=obs_rms, device=cfg.device)
    if 'config' in ckpt:
        pass
    fixed_eval_seeds = load_or_create_eval_seeds(cfg, run_dir)
    save_initial_observations_for_seeds(cfg, fixed_eval_seeds, run_dir)
    summary = evaluate(cfg, model, obs_rms, fixed_eval_seeds, run_dir=run_dir, prefix='eval_fixed333', save_trajectory=False)
    print(json.dumps(summary, indent=2))

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, default=Config.mode, choices=['train', 'eval'])
    p.add_argument('--seed', type=int, default=Config.seed)
    p.add_argument('--device', type=str, default=Config.device)
    p.add_argument('--save_dir', type=str, default=Config.save_dir)
    p.add_argument('--run_name', type=str, default=Config.run_name)
    p.add_argument('--theta_limit_deg', type=float, default=Config.theta_limit_deg)
    p.add_argument('--gravity', type=float, default=Config.gravity)
    p.add_argument('--enable_wind', type=str2bool, default=Config.enable_wind)
    p.add_argument('--wind_power', type=float, default=Config.wind_power)
    p.add_argument('--turbulence_power', type=float, default=Config.turbulence_power)
    p.add_argument('--max_episode_steps', type=int, default=Config.max_episode_steps)
    p.add_argument('--timeout_penalty', type=float, default=Config.timeout_penalty)
    p.add_argument('--step_penalty', type=float, default=Config.step_penalty)
    p.add_argument('--include_prev_theta_in_obs', type=str2bool, default=Config.include_prev_theta_in_obs)
    p.add_argument('--include_contacts_in_obs', type=str2bool, default=Config.include_contacts_in_obs)
    p.add_argument('--lateral_stress_reset', type=str2bool, default=Config.lateral_stress_reset)
    p.add_argument('--stress_x_min', type=float, default=Config.stress_x_min)
    p.add_argument('--stress_x_max', type=float, default=Config.stress_x_max)
    p.add_argument('--stress_vx_min', type=float, default=Config.stress_vx_min)
    p.add_argument('--stress_vx_max', type=float, default=Config.stress_vx_max)
    p.add_argument('--variation_lambda', type=float, default=Config.variation_lambda)
    p.add_argument('--gamma', type=float, default=Config.gamma)
    p.add_argument('--gae_lambda', type=float, default=Config.gae_lambda)
    p.add_argument('--total_steps', type=int, default=Config.total_steps)
    p.add_argument('--num_envs', type=int, default=Config.num_envs)
    p.add_argument('--steps_per_rollout', type=int, default=Config.steps_per_rollout)
    p.add_argument('--update_epochs', type=int, default=Config.update_epochs)
    p.add_argument('--minibatch_size', type=int, default=Config.minibatch_size)
    p.add_argument('--learning_rate', type=float, default=Config.learning_rate)
    p.add_argument('--clip_coef', type=float, default=Config.clip_coef)
    p.add_argument('--ent_coef', type=float, default=Config.ent_coef)
    p.add_argument('--vf_coef', type=float, default=Config.vf_coef)
    p.add_argument('--max_grad_norm', type=float, default=Config.max_grad_norm)
    p.add_argument('--target_kl', type=float, default=Config.target_kl)
    p.add_argument('--hidden_size', type=int, default=Config.hidden_size)
    p.add_argument('--std_mode', type=str, default=Config.std_mode, choices=['fixed', 'anneal', 'learned'])
    p.add_argument('--init_std_main', type=float, default=Config.init_std_main)
    p.add_argument('--init_std_theta', type=float, default=Config.init_std_theta)
    p.add_argument('--final_std_main', type=float, default=Config.final_std_main)
    p.add_argument('--final_std_theta', type=float, default=Config.final_std_theta)
    p.add_argument('--obs_norm', type=str2bool, default=Config.obs_norm)
    p.add_argument('--obs_clip', type=float, default=Config.obs_clip)
    p.add_argument('--eval_every_rollouts', type=int, default=Config.eval_every_rollouts)
    p.add_argument('--eval_episodes', type=int, default=Config.eval_episodes)
    p.add_argument('--final_eval_episodes', type=int, default=Config.final_eval_episodes)
    p.add_argument('--fixed_eval_seed', type=int, default=Config.fixed_eval_seed)
    p.add_argument('--fixed_eval_seed_file', type=str, default=Config.fixed_eval_seed_file)
    p.add_argument('--deterministic_eval', type=str2bool, default=Config.deterministic_eval)
    p.add_argument('--checkpoint', type=str, default=Config.checkpoint)
    p.add_argument('--warm_start_path', type=str, default=Config.warm_start_path)
    args = p.parse_args()
    return Config(**vars(args))

def main():
    cfg = parse_args()
    if cfg.mode == 'train':
        run_dir = train(cfg)
        print(f'Saved run to: {run_dir}')
    else:
        eval_only(cfg)
if __name__ == '__main__':
    main()
