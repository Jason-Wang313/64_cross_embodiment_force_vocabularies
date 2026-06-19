"""MuJoCo force/effect vocabulary benchmark for Paper 64.

Version 5 upgrades the v4 CEFV scaffold into a hostile-review benchmark.  The
proposed method, RC-FEV, is not allowed to win by weakening robust MPC: it uses a
robust/CVaR branch anchor and must show that calibrated force/effect tokens and
online residuals add value beyond that strong anchor.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SUCCESS_RADIUS = 0.075
ACTIVE_STEPS = 40
SETTLE_STEPS = 10
DT = 0.006


@dataclass(frozen=True)
class Embodiment:
    name: str
    radius: float
    mass: float
    kp: float
    damping: float
    solref_time: float
    friction: float


@dataclass(frozen=True)
class ObjectParams:
    mass: float
    friction: float


@dataclass(frozen=True)
class Branch:
    embodiment: Embodiment
    obj: ObjectParams


@dataclass(frozen=True)
class PushAction:
    angle: float
    angle_offset: float
    offset: float
    distance: float


@dataclass(frozen=True)
class TaskSpec:
    split: str
    embodiment: Embodiment
    obj: ObjectParams
    box: tuple[float, float]
    target: tuple[float, float]
    act_noise: float


@dataclass
class TokenModel:
    name: str
    centers: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    token_energy: np.ndarray
    token_std: np.ndarray
    token_success: np.ndarray
    token_count: np.ndarray
    feature_kind: str


@dataclass
class ForceVocabulary:
    full: TokenModel
    raw: TokenModel
    action: TokenModel
    masked: TokenModel
    small: TokenModel
    scalar_force: np.ndarray
    scalar_energy: np.ndarray
    action_prior: dict[tuple[int, int, int], float]
    global_energy: float


@dataclass
class AdaptState:
    residual: dict[int, float] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    action_residual: dict[tuple[int, int, int], float] = field(default_factory=dict)
    action_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    global_residual: float = 0.0
    global_count: int = 0
    last_abs_error: float = 0.0

    def bias(self, token: int) -> float:
        token_bias = self.residual.get(token, 0.0)
        token_count = self.counts.get(token, 0)
        token_weight = min(0.75, token_count / 10.0)
        global_weight = min(0.35, self.global_count / 22.0)
        return token_weight * token_bias + global_weight * self.global_residual

    def action_bias(self, action: PushAction) -> float:
        key = online_action_bin(action)
        count = self.action_counts.get(key, 0)
        if count == 0:
            return 0.0
        weight = min(0.60, count / 8.0)
        return weight * self.action_residual.get(key, 0.0)

    def update(self, tokens: list[int], action: PushAction, observed_energy: float, predicted_energy: float) -> None:
        err = observed_energy - predicted_energy
        self.last_abs_error = abs(float(err))
        self.global_count += 1
        global_lr = 0.22 / math.sqrt(self.global_count)
        self.global_residual = (1.0 - global_lr) * self.global_residual + global_lr * err
        action_key = online_action_bin(action)
        action_old = self.action_residual.get(action_key, 0.0)
        action_count = self.action_counts.get(action_key, 0)
        action_lr = 0.38 / math.sqrt(action_count + 1.0)
        self.action_residual[action_key] = (1.0 - action_lr) * action_old + action_lr * err
        self.action_counts[action_key] = action_count + 1
        if not tokens:
            return
        for token in tokens:
            old = self.residual.get(token, 0.0)
            count = self.counts.get(token, 0)
            lr = 0.32 / math.sqrt(count + 1.0)
            self.residual[token] = (1.0 - lr) * old + lr * err
            self.counts[token] = count + 1


SOURCE_SMALL = Embodiment("source_small_fast", 0.022, 0.18, 470.0, 6.0, 0.006, 1.15)
SOURCE_NOMINAL = Embodiment("source_nominal", 0.026, 0.25, 530.0, 8.0, 0.006, 1.20)
SOURCE_SOFT = Embodiment("source_large_soft", 0.032, 0.32, 455.0, 11.0, 0.012, 1.10)
SOURCE_HEAVY = Embodiment("source_heavy_stiff", 0.029, 0.44, 660.0, 7.5, 0.004, 1.30)

HELDOUT_NEEDLE = Embodiment("heldout_needle_light", 0.018, 0.12, 405.0, 5.0, 0.014, 1.05)
HELDOUT_SOFT = Embodiment("heldout_large_compliant", 0.039, 0.36, 410.0, 13.0, 0.016, 1.00)
HELDOUT_HIGH_GAIN = Embodiment("heldout_high_gain_heavy", 0.030, 0.52, 790.0, 7.0, 0.004, 1.35)
HELDOUT_WEAK = Embodiment("heldout_weak_actuator", 0.027, 0.20, 320.0, 10.0, 0.010, 1.10)

NOMINAL_OBJ = ObjectParams(0.12, 0.65)
LIGHT_SLIPPERY_OBJ = ObjectParams(0.08, 0.22)
HEAVY_STICKY_OBJ = ObjectParams(0.30, 1.10)
HEAVY_LOW_FRICTION_OBJ = ObjectParams(0.34, 0.18)
HIGH_FRICTION_OBJ = ObjectParams(0.16, 1.35)

MODEL_BRANCHES = [
    Branch(SOURCE_NOMINAL, NOMINAL_OBJ),
    Branch(SOURCE_SMALL, LIGHT_SLIPPERY_OBJ),
    Branch(SOURCE_SOFT, HEAVY_STICKY_OBJ),
    Branch(SOURCE_HEAVY, ObjectParams(0.18, 0.85)),
    Branch(SOURCE_NOMINAL, HEAVY_LOW_FRICTION_OBJ),
]

MAIN_METHODS = [
    "random_shooting",
    "geometry_mpc",
    "source_action_transfer",
    "raw_force_scalar",
    "continuous_force_regression",
    "robust_domain_randomized_mpc",
    "cvar_domain_randomized_mpc",
    "cefv_v4",
    "rc_fev_v5",
    "rc_fev_no_online",
    "oracle_embodiment_mpc",
]

ABLATION_METHODS = [
    "rc_fev_v5",
    "rc_fev_no_online",
    "rc_fev_no_robust_anchor",
    "rc_fev_no_embodiment_normalization",
    "rc_fev_no_tangent_rotation_features",
    "action_only_vocabulary",
    "small_vocabulary_k3",
    "cefv_v4",
    "robust_domain_randomized_mpc",
    "cvar_domain_randomized_mpc",
    "oracle_embodiment_mpc",
]

SPLITS = {
    "nominal": {
        "embodiments": [SOURCE_SMALL, SOURCE_NOMINAL, SOURCE_SOFT],
        "masses": [0.10, 0.12, 0.16],
        "frictions": [0.48, 0.65, 0.85],
        "act_noise": 0.00,
        "target_bonus": 0.00,
    },
    "heldout_small_radius": {
        "embodiments": [HELDOUT_NEEDLE],
        "masses": [0.10, 0.12, 0.16],
        "frictions": [0.48, 0.65, 0.85],
        "act_noise": 0.01,
        "target_bonus": 0.02,
    },
    "heldout_large_soft": {
        "embodiments": [HELDOUT_SOFT],
        "masses": [0.10, 0.12, 0.18],
        "frictions": [0.48, 0.70, 0.90],
        "act_noise": 0.02,
        "target_bonus": 0.02,
    },
    "heldout_high_gain": {
        "embodiments": [HELDOUT_HIGH_GAIN],
        "masses": [0.10, 0.14, 0.20],
        "frictions": [0.42, 0.65, 0.95],
        "act_noise": 0.03,
        "target_bonus": 0.03,
    },
    "heldout_weak_actuator": {
        "embodiments": [HELDOUT_WEAK],
        "masses": [0.10, 0.14, 0.20],
        "frictions": [0.42, 0.65, 0.95],
        "act_noise": 0.04,
        "target_bonus": 0.035,
    },
    "low_friction": {
        "embodiments": [SOURCE_NOMINAL, HELDOUT_NEEDLE, HELDOUT_WEAK],
        "masses": [0.08, 0.12, 0.18],
        "frictions": [0.10, 0.18, 0.28],
        "act_noise": 0.02,
        "target_bonus": 0.02,
    },
    "heavy_object": {
        "embodiments": [SOURCE_SOFT, HELDOUT_SOFT, HELDOUT_HIGH_GAIN],
        "masses": [0.24, 0.32, 0.42],
        "frictions": [0.45, 0.75, 1.05],
        "act_noise": 0.03,
        "target_bonus": 0.03,
    },
    "high_friction": {
        "embodiments": [SOURCE_SOFT, HELDOUT_SOFT, HELDOUT_HIGH_GAIN],
        "masses": [0.12, 0.18, 0.28],
        "frictions": [1.05, 1.25, 1.45],
        "act_noise": 0.03,
        "target_bonus": 0.035,
    },
    "actuation_noise": {
        "embodiments": [SOURCE_SMALL, HELDOUT_WEAK, HELDOUT_HIGH_GAIN],
        "masses": [0.10, 0.16, 0.24],
        "frictions": [0.35, 0.65, 0.95],
        "act_noise": 0.085,
        "target_bonus": 0.05,
    },
    "combined_shift": {
        "embodiments": [HELDOUT_NEEDLE, HELDOUT_SOFT, HELDOUT_HIGH_GAIN, HELDOUT_WEAK],
        "masses": [0.06, 0.28, 0.44],
        "frictions": [0.12, 0.95, 1.35],
        "act_noise": 0.07,
        "target_bonus": 0.07,
    },
}
DEFAULT_ABLATION_SPLITS = ["combined_shift", "heavy_object"]
MODEL_CACHE: dict[Branch, mujoco.MjModel] = {}


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def make_model(branch: Branch) -> mujoco.MjModel:
    cached = MODEL_CACHE.get(branch)
    if cached is not None:
        return cached
    emb = branch.embodiment
    obj = branch.obj
    xml = f"""
    <mujoco model="cross_embodiment_force_vocab_v5">
      <option timestep="{DT}" gravity="0 0 -9.81" integrator="RK4"/>
      <worldbody>
        <light pos="0 0 1"/>
        <geom name="floor" type="plane" size="1.2 1.2 0.02" rgba="0.75 0.75 0.75 1"
              friction="{obj.friction} 0.004 0.0001" solref="{emb.solref_time} 1" solimp="0.9 0.95 0.001"/>
        <body name="box" pos="0 0 0.028">
          <freejoint name="box_free"/>
          <geom name="box_geom" type="box" size="0.045 0.035 0.025" mass="{obj.mass}"
                rgba="0.1 0.3 0.9 1" friction="{obj.friction} 0.004 0.0001"
                solref="{emb.solref_time} 1" solimp="0.9 0.95 0.001"/>
        </body>
        <body name="pusher" pos="0 0 0.045">
          <joint name="px" type="slide" axis="1 0 0" damping="{emb.damping}"/>
          <joint name="py" type="slide" axis="0 1 0" damping="{emb.damping}"/>
          <geom name="pusher_geom" type="sphere" size="{emb.radius}" mass="{emb.mass}"
                rgba="0.9 0.25 0.1 1" friction="{emb.friction} 0.004 0.0001"
                solref="{emb.solref_time} 1" solimp="0.9 0.95 0.001"/>
        </body>
      </worldbody>
      <actuator>
        <position name="px_ctrl" joint="px" kp="{emb.kp}" ctrlrange="-1 1"/>
        <position name="py_ctrl" joint="py" kp="{emb.kp}" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    MODEL_CACHE[branch] = model
    return model


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q]
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


def set_state(data: mujoco.MjData, box_xy: np.ndarray, pusher_xy: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:7] = [float(box_xy[0]), float(box_xy[1]), 0.028, 1.0, 0.0, 0.0, 0.0]
    data.qpos[7:9] = [float(pusher_xy[0]), float(pusher_xy[1])]
    data.ctrl[0:2] = pusher_xy


def action_path(
    box_xy: np.ndarray,
    action: PushAction,
    embodiment: Embodiment,
    act_noise: float,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    angle = action.angle + rng.gauss(0.0, act_noise)
    distance_val = max(0.08, action.distance * max(0.70, rng.gauss(1.0, act_noise)))
    offset = action.offset + rng.gauss(0.0, 0.018 * act_noise)
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    approach = 0.110 + embodiment.radius
    start = box_xy - approach * direction + offset * normal
    end = box_xy + distance_val * direction + offset * normal
    return start, end


def measure_box_pusher_contact(model: mujoco.MjModel, data: mujoco.MjData, box_id: int, pusher_id: int) -> tuple[float, float]:
    normal = 0.0
    tangent = 0.0
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geoms = {int(contact.geom1), int(contact.geom2)}
        if box_id not in geoms or pusher_id not in geoms:
            continue
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, idx, force)
        normal += abs(float(force[0]))
        tangent += math.sqrt(float(force[1]) ** 2 + float(force[2]) ** 2)
    return normal, tangent


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def outcome_energy(final_distance: float, effort: float, yaw_abs: float, failure: float) -> float:
    return final_distance + 0.020 * effort + 0.035 * yaw_abs + 0.220 * failure


def rollout_push(
    branch: Branch,
    box_xy: np.ndarray,
    target_xy: np.ndarray,
    action: PushAction,
    act_noise: float = 0.0,
    rng: random.Random | None = None,
) -> dict:
    rng = rng or random.Random(0)
    model = make_model(branch)
    data = mujoco.MjData(model)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
    pusher_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_geom")
    start, end = action_path(box_xy, action, branch.embodiment, act_noise, rng)
    set_state(data, box_xy, start)
    mujoco.mj_forward(model, data)

    normal_impulse = 0.0
    tangent_impulse = 0.0
    max_normal = 0.0
    max_tangent = 0.0
    contact_steps = 0
    effort = 0.0
    last = start.copy()

    for step in range(ACTIVE_STEPS):
        alpha = (step + 1) / float(ACTIVE_STEPS)
        ctrl = (1.0 - alpha) * start + alpha * end
        effort += distance(ctrl, last)
        last = ctrl
        data.ctrl[0] = float(ctrl[0])
        data.ctrl[1] = float(ctrl[1])
        mujoco.mj_step(model, data)
        normal, tangent = measure_box_pusher_contact(model, data, box_id, pusher_id)
        if normal > 1e-7 or tangent > 1e-7:
            contact_steps += 1
        normal_impulse += normal * DT
        tangent_impulse += tangent * DT
        max_normal = max(max_normal, normal)
        max_tangent = max(max_tangent, tangent)

    for _ in range(SETTLE_STEPS):
        data.ctrl[0] = float(end[0])
        data.ctrl[1] = float(end[1])
        mujoco.mj_step(model, data)
        normal, tangent = measure_box_pusher_contact(model, data, box_id, pusher_id)
        if normal > 1e-7 or tangent > 1e-7:
            contact_steps += 1
        normal_impulse += normal * DT
        tangent_impulse += tangent * DT
        max_normal = max(max_normal, normal)
        max_tangent = max(max_tangent, tangent)

    final_xy = np.array(data.qpos[0:2], dtype=float)
    yaw = yaw_from_quat(np.array(data.qpos[3:7], dtype=float))
    initial_distance = distance(box_xy, target_xy)
    final_distance = distance(final_xy, target_xy)
    progress = (initial_distance - final_distance) / max(initial_distance, 1e-8)
    no_contact = 1.0 if contact_steps < 4 else 0.0
    out_of_bounds = 1.0 if float(np.linalg.norm(final_xy)) > 0.88 else 0.0
    failure = max(no_contact, out_of_bounds)
    energy = outcome_energy(final_distance, effort, abs(yaw), failure)
    success = float(final_distance <= SUCCESS_RADIUS and failure < 0.5)
    return {
        "final_x": float(final_xy[0]),
        "final_y": float(final_xy[1]),
        "yaw": float(yaw),
        "yaw_abs": abs(float(yaw)),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "progress": progress,
        "success": success,
        "energy": energy,
        "failure": failure,
        "effort": effort,
        "normal_impulse": normal_impulse,
        "tangent_impulse": tangent_impulse,
        "max_normal": max_normal,
        "max_tangent": max_tangent,
        "contact_steps": contact_steps,
    }


def candidate_actions(box_xy: np.ndarray, target_xy: np.ndarray) -> list[PushAction]:
    base = math.atan2(float(target_xy[1] - box_xy[1]), float(target_xy[0] - box_xy[0]))
    remaining = distance(box_xy, target_xy)
    actions: list[PushAction] = []
    for deg in [-38, -20, 0, 20, 38]:
        for offset in [-0.026, 0.0, 0.026]:
            for scale in [0.78, 1.04, 1.30]:
                angle_offset = math.radians(deg)
                actions.append(PushAction(base + angle_offset, angle_offset, offset, max(0.14, min(0.58, scale * remaining))))
    return actions


def geometric_score(box_xy: np.ndarray, target_xy: np.ndarray, action: PushAction) -> float:
    direction = np.array([math.cos(action.angle), math.sin(action.angle)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    predicted = box_xy + action.distance * direction + 0.35 * action.offset * normal
    return distance(predicted, target_xy) + 0.04 * abs(action.offset) + 0.003 * abs(math.degrees(action.angle_offset))


def sample_task(split: str, seed: int, episode: int) -> TaskSpec:
    cfg = SPLITS[split]
    rng = random.Random(6400003 + 100003 * seed + 9176 * episode + sum(ord(c) for c in split))
    embodiment = rng.choice(cfg["embodiments"])
    obj = ObjectParams(rng.choice(cfg["masses"]), rng.choice(cfg["frictions"]))
    box = np.array([rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)], dtype=float)
    target_angle = rng.uniform(-0.76, 0.76)
    target_radius = rng.uniform(0.25, 0.42) + float(cfg["target_bonus"])
    target = box + target_radius * np.array([math.cos(target_angle), math.sin(target_angle)], dtype=float)
    return TaskSpec(split, embodiment, obj, (float(box[0]), float(box[1])), (float(target[0]), float(target[1])), float(cfg["act_noise"]))


def force_features(outcome: dict, action: PushAction, branch: Branch, kind: str) -> np.ndarray:
    emb = branch.embodiment
    scale = max(1e-6, emb.kp * emb.radius * DT)
    if kind == "action":
        return np.array(
            [
                math.sin(action.angle_offset),
                math.cos(action.angle_offset),
                action.offset / max(emb.radius, 1e-6),
                action.distance / max(emb.radius, 1e-6),
            ],
            dtype=np.float64,
        )

    normal = outcome["normal_impulse"]
    tangent = outcome["tangent_impulse"]
    max_normal = outcome["max_normal"]
    max_tangent = outcome["max_tangent"]
    if kind == "raw":
        normal_scale = 1.0
        action_scale = 1.0
    else:
        normal_scale = scale
        action_scale = max(emb.radius, 1e-6)

    tangent_ratio = tangent / max(normal + tangent, 1e-8)
    vec = np.array(
        [
            math.log1p(normal / normal_scale),
            math.log1p(tangent / normal_scale),
            tangent_ratio,
            outcome["contact_steps"] / float(ACTIVE_STEPS + SETTLE_STEPS),
            math.log1p(max_normal / max(emb.kp * emb.radius, 1e-6)) if kind != "raw" else math.log1p(max_normal),
            math.log1p(max_tangent / max(emb.kp * emb.radius, 1e-6)) if kind != "raw" else math.log1p(max_tangent),
            outcome["progress"],
            outcome["final_distance"] / max(outcome["initial_distance"], 1e-8),
            outcome["yaw_abs"],
            math.sin(action.angle_offset),
            math.cos(action.angle_offset),
            action.offset / action_scale,
            action.distance / action_scale,
        ],
        dtype=np.float64,
    )
    if kind == "masked":
        vec = vec.copy()
        vec[[1, 2, 5, 8, 11]] = 0.0
    return vec


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0) + 1e-6
    return (x - mu) / sigma, mu, sigma


def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(x) < k:
        raise ValueError("not enough rows for k-means")
    first = int(rng.integers(0, len(x)))
    centers = [x[first]]
    while len(centers) < k:
        d2 = np.min(np.sum((x[:, None, :] - np.stack(centers)[None, :, :]) ** 2, axis=2), axis=1)
        probs = d2 / max(float(d2.sum()), 1e-12)
        centers.append(x[int(rng.choice(len(x), p=probs))])
    centers = np.stack(centers)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        dist2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centers[idx] = x[mask].mean(axis=0)
            else:
                centers[idx] = x[int(rng.integers(0, len(x)))]
    return centers, labels


def fit_token_model(name: str, features: np.ndarray, energy: np.ndarray, success: np.ndarray, k: int, seed: int, feature_kind: str) -> TokenModel:
    x, mu, sigma = standardize(features)
    centers, labels = kmeans(x, k, seed)
    global_energy = float(np.mean(energy))
    global_std = float(np.std(energy) + 1e-6)
    token_energy = np.zeros(k, dtype=float)
    token_std = np.zeros(k, dtype=float)
    token_success = np.zeros(k, dtype=float)
    token_count = np.zeros(k, dtype=int)
    for idx in range(k):
        mask = labels == idx
        token_count[idx] = int(np.sum(mask))
        if np.any(mask):
            token_energy[idx] = float(np.mean(energy[mask]))
            token_std[idx] = float(np.std(energy[mask]) + 1e-6)
            token_success[idx] = float(np.mean(success[mask]))
        else:
            token_energy[idx] = global_energy
            token_std[idx] = global_std
            token_success[idx] = float(np.mean(success))
    return TokenModel(name, centers, mu, sigma, token_energy, token_std, token_success, token_count, feature_kind)


def assign_token(model: TokenModel, feature: np.ndarray) -> int:
    x = (feature - model.mu) / model.sigma
    return int(np.argmin(np.sum((model.centers - x[None, :]) ** 2, axis=1)))


def action_bin(action: PushAction) -> tuple[int, int, int]:
    angle_bin = int(round(math.degrees(action.angle_offset) / 12.0))
    dist_bin = int(round(action.distance / 0.08))
    offset_bin = int(round(action.offset / 0.012))
    return angle_bin, dist_bin, offset_bin


def online_action_bin(action: PushAction) -> tuple[int, int, int]:
    angle_bin = int(round(math.degrees(action.angle_offset) / 20.0))
    offset_bin = int(round(action.offset / 0.026))
    distance_family = 1 if action.distance >= 0.34 else 0
    return angle_bin, offset_bin, distance_family


def nearest_scalar_energy(vocab: ForceVocabulary, scalar: float, k: int = 45) -> float:
    d = np.abs(vocab.scalar_force - scalar)
    idx = np.argsort(d)[: min(k, len(d))]
    return float(np.mean(vocab.scalar_energy[idx]))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_rows(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        clean = dict(row)
        for key, value in row.items():
            if isinstance(value, float):
                clean[key] = f"{value:.6f}"
        formatted.append(clean)
    return formatted


def fit_force_vocabulary(args: argparse.Namespace) -> ForceVocabulary:
    full_features: list[np.ndarray] = []
    raw_features: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    masked_features: list[np.ndarray] = []
    energies: list[float] = []
    successes: list[float] = []
    scalar_force: list[float] = []
    action_energy: dict[tuple[int, int, int], list[float]] = {}
    train_rows: list[dict] = []
    train_splits = ["nominal", "low_friction", "heavy_object", "high_friction", "actuation_noise"]

    for idx in range(args.train_tasks):
        split = train_splits[idx % len(train_splits)]
        task = sample_task(split, idx // 17, idx % 17)
        box = np.array(task.box, dtype=float)
        target = np.array(task.target, dtype=float)
        actions = candidate_actions(box, target)
        branch = MODEL_BRANCHES[idx % len(MODEL_BRANCHES)]
        for action_idx, action in enumerate(actions):
            rng = random.Random(641111 + 8191 * idx + 97 * action_idx)
            outcome = rollout_push(branch, box, target, action, 0.0, rng)
            full_features.append(force_features(outcome, action, branch, "full"))
            raw_features.append(force_features(outcome, action, branch, "raw"))
            action_features.append(force_features(outcome, action, branch, "action"))
            masked_features.append(force_features(outcome, action, branch, "masked"))
            energies.append(float(outcome["energy"]))
            successes.append(float(outcome["success"]))
            scalar = math.log1p(outcome["normal_impulse"])
            scalar_force.append(scalar)
            action_energy.setdefault(action_bin(action), []).append(float(outcome["energy"]))
            train_rows.append(
                {
                    "train_task": idx,
                    "candidate": action_idx,
                    "branch": branch.embodiment.name,
                    "object_mass": branch.obj.mass,
                    "object_friction": branch.obj.friction,
                    "angle_offset_deg": math.degrees(action.angle_offset),
                    "offset": action.offset,
                    "distance": action.distance,
                    "energy": outcome["energy"],
                    "success": outcome["success"],
                    "progress": outcome["progress"],
                    "normal_impulse": outcome["normal_impulse"],
                    "tangent_impulse": outcome["tangent_impulse"],
                    "contact_steps": outcome["contact_steps"],
                    "yaw_abs": outcome["yaw_abs"],
                }
            )
        if (idx + 1) % max(1, args.train_tasks // 5) == 0:
            print(f"fit-vocab rollouts {idx + 1}/{args.train_tasks}", flush=True)

    full_arr = np.stack(full_features)
    raw_arr = np.stack(raw_features)
    action_arr = np.stack(action_features)
    masked_arr = np.stack(masked_features)
    energy_arr = np.array(energies, dtype=float)
    success_arr = np.array(successes, dtype=float)
    small_k = min(4, max(2, args.vocab_size))
    vocab = ForceVocabulary(
        full=fit_token_model("rc_fev_v5", full_arr, energy_arr, success_arr, args.vocab_size, args.seed + 10, "full"),
        raw=fit_token_model("no_embodiment_normalization", raw_arr, energy_arr, success_arr, args.vocab_size, args.seed + 11, "raw"),
        action=fit_token_model("action_only_vocabulary", action_arr, energy_arr, success_arr, args.vocab_size, args.seed + 12, "action"),
        masked=fit_token_model("no_tangent_rotation_features", masked_arr, energy_arr, success_arr, args.vocab_size, args.seed + 13, "masked"),
        small=fit_token_model("small_vocabulary_k4", full_arr, energy_arr, success_arr, small_k, args.seed + 14, "full"),
        scalar_force=np.array(scalar_force, dtype=float),
        scalar_energy=energy_arr,
        action_prior={k: float(np.mean(v)) for k, v in action_energy.items()},
        global_energy=float(np.mean(energy_arr)),
    )

    full_tokens = [assign_token(vocab.full, row) for row in full_arr]
    for row, token in zip(train_rows, full_tokens):
        row["full_token"] = token
    write_rows(RESULTS / "force_vocabulary_training.csv", format_rows(train_rows))
    token_rows = token_summary_rows(vocab)
    write_rows(RESULTS / "force_vocabulary_tokens.csv", format_rows(token_rows))
    return vocab


def token_summary_rows(vocab: ForceVocabulary) -> list[dict]:
    token_rows = []
    for model in [vocab.full, vocab.raw, vocab.action, vocab.masked, vocab.small]:
        for token in range(len(model.token_energy)):
            token_rows.append(
                {
                    "model": model.name,
                    "token": token,
                    "count": int(model.token_count[token]),
                    "mean_energy": float(model.token_energy[token]),
                    "std_energy": float(model.token_std[token]),
                    "success_rate": float(model.token_success[token]),
                    "feature_kind": model.feature_kind,
                }
            )
    return token_rows


def branch_cvar(values: Sequence[float], alpha: float = 0.40) -> float:
    arr = np.sort(np.array(values, dtype=float))
    k = max(1, int(math.ceil(len(arr) * alpha)))
    return float(np.mean(arr[-k:]))


def prepare_candidates(task: TaskSpec) -> list[dict]:
    box = np.array(task.box, dtype=float)
    target = np.array(task.target, dtype=float)
    true_branch = Branch(task.embodiment, task.obj)
    rows: list[dict] = []
    for idx, action in enumerate(candidate_actions(box, target)):
        true_rng = random.Random(642001 + idx + int(1000 * task.obj.mass) + sum(ord(c) for c in task.split))
        true_outcome = rollout_push(true_branch, box, target, action, task.act_noise, true_rng)
        source_outcomes = []
        for branch_idx, branch in enumerate(MODEL_BRANCHES):
            source_rng = random.Random(642337 + idx * 17 + branch_idx)
            source_outcomes.append((branch, rollout_push(branch, box, target, action, 0.0, source_rng)))
        source_energies = [float(outcome["energy"]) for _, outcome in source_outcomes]
        rows.append(
            {
                "candidate": idx,
                "action": action,
                "true": true_outcome,
                "source_outcomes": source_outcomes,
                "source_mean_energy": float(np.mean(source_energies)),
                "source_worst_energy": float(np.max(source_energies)),
                "source_cvar_energy": branch_cvar(source_energies),
                "source_std_energy": float(np.std(source_energies)),
                "source_best_energy": float(np.min(source_energies)),
                "source_nominal_energy": float(source_energies[0]),
                "geom_score": geometric_score(box, target, action),
            }
        )
    return rows


def token_model_for_method(vocab: ForceVocabulary, method: str) -> TokenModel:
    if method in {"rc_fev_no_embodiment_normalization"}:
        return vocab.raw
    if method == "action_only_vocabulary":
        return vocab.action
    if method == "rc_fev_no_tangent_rotation_features":
        return vocab.masked
    if method == "small_vocabulary_k3":
        return vocab.small
    return vocab.full


def candidate_tokens(candidate: dict, model: TokenModel) -> list[int]:
    tokens = []
    for branch, outcome in candidate["source_outcomes"]:
        feat = force_features(outcome, candidate["action"], branch, model.feature_kind)
        tokens.append(assign_token(model, feat))
    return tokens


def cefv_v4_score(candidate: dict, vocab: ForceVocabulary, state: AdaptState | None) -> tuple[float, list[int], float, float, float]:
    model = vocab.full
    tokens = candidate_tokens(candidate, model)
    token_estimates = [float(model.token_energy[token]) + (state.bias(token) if state is not None else 0.0) for token in tokens]
    token_mean = float(np.mean(token_estimates))
    token_std = float(np.std(token_estimates))
    score = token_mean + 0.26 * float(candidate["source_mean_energy"]) + 0.07 * float(candidate["geom_score"]) + 0.025 * token_std
    return score, tokens, token_mean, token_std, 0.0


def rc_fev_score(candidate: dict, vocab: ForceVocabulary, method: str, state: AdaptState | None) -> tuple[float, list[int], float, float, float]:
    model = token_model_for_method(vocab, method)
    tokens = candidate_tokens(candidate, model)
    token_estimates = [float(model.token_energy[token]) + (state.bias(token) if state is not None else 0.0) for token in tokens]
    token_mean = float(np.mean(token_estimates))
    token_uncertainty = float(np.mean([model.token_std[token] for token in tokens]) + np.std(token_estimates) + 0.5 * candidate["source_std_energy"])
    branch_agreement = math.exp(-5.0 * min(0.30, token_uncertainty))
    online_action_bias = state.action_bias(candidate["action"]) if state is not None else 0.0
    online_mismatch_penalty = (
        state.last_abs_error * (0.15 + 1.5 * float(candidate["source_std_energy"])) if state is not None else 0.0
    )
    calibrated = (
        0.38 * float(candidate["source_cvar_energy"])
        + 0.22 * token_mean
        + 0.18 * float(candidate["source_mean_energy"])
        + 0.07 * float(candidate["geom_score"])
        + 0.10 * token_uncertainty
        + 0.30 * online_action_bias
        + 0.08 * online_mismatch_penalty
    )
    robust_anchor = float(candidate["source_worst_energy"])
    if method == "rc_fev_no_robust_anchor":
        score = calibrated
        robust_weight = 0.0
    else:
        robust_weight = max(0.35, min(0.78, 0.68 - 0.30 * branch_agreement + 0.30 * min(0.30, token_uncertainty)))
        if state is not None:
            robust_weight = max(0.30, robust_weight - min(0.16, state.global_count / 120.0))
        score = robust_weight * robust_anchor + (1.0 - robust_weight) * calibrated
    return float(score), tokens, token_mean, token_uncertainty, robust_weight


def choose_candidate(
    method: str,
    candidates: list[dict],
    vocab: ForceVocabulary,
    state: AdaptState | None,
    rng: random.Random,
) -> tuple[int, list[int], float, float, float, float]:
    if method == "random_shooting":
        return rng.randrange(len(candidates)), [], 0.0, 0.0, 0.0, 0.0
    if method == "geometry_mpc":
        return int(np.argmin([c["geom_score"] for c in candidates])), [], 0.0, 0.0, 0.0, 0.0
    if method == "source_action_transfer":
        scores = [vocab.action_prior.get(action_bin(c["action"]), vocab.global_energy) for c in candidates]
        return int(np.argmin(scores)), [], 0.0, 0.0, 0.0, 0.0
    if method in {"raw_force_scalar", "continuous_force_regression"}:
        scores = []
        for c in candidates:
            scalars = [math.log1p(outcome["normal_impulse"]) for _, outcome in c["source_outcomes"]]
            scalar_score = nearest_scalar_energy(vocab, float(np.mean(scalars)))
            if method == "continuous_force_regression":
                score = 0.50 * scalar_score + 0.32 * c["source_mean_energy"] + 0.12 * c["geom_score"] + 0.06 * c["source_std_energy"]
            else:
                score = scalar_score + 0.12 * c["geom_score"]
            scores.append(score)
        return int(np.argmin(scores)), [], float(min(scores)), 0.0, 0.0, 0.0
    if method == "robust_domain_randomized_mpc":
        return int(np.argmin([c["source_worst_energy"] for c in candidates])), [], 0.0, 0.0, 0.0, 0.0
    if method == "cvar_domain_randomized_mpc":
        return int(np.argmin([c["source_cvar_energy"] for c in candidates])), [], 0.0, 0.0, 0.0, 0.0
    if method == "oracle_embodiment_mpc":
        return int(np.argmin([c["true"]["energy"] for c in candidates])), [], 0.0, 0.0, 0.0, 0.0
    if method == "cefv_v4":
        scores = [cefv_v4_score(c, vocab, state) for c in candidates]
    else:
        scores = [rc_fev_score(c, vocab, method, state) for c in candidates]
    chosen = int(np.argmin([s[0] for s in scores]))
    score, tokens, token_mean, uncertainty, robust_weight = scores[chosen]
    return chosen, tokens, score, token_mean, uncertainty, robust_weight


def phase_label(episode: int, episodes: int) -> str:
    if episode < episodes / 3:
        return "early"
    if episode < 2 * episodes / 3:
        return "middle"
    return "late"


def row_for_choice(
    task: TaskSpec,
    seed: int,
    episode: int,
    episodes: int,
    method: str,
    candidate: dict,
    ablation: bool,
    predicted_energy: float,
    token_mean: float,
    token_uncertainty: float,
    robust_weight: float,
    oracle_energy: float,
) -> dict:
    outcome = candidate["true"]
    action = candidate["action"]
    return {
        "seed": seed,
        "episode": episode,
        "phase": phase_label(episode, episodes),
        "split": task.split,
        "method": method,
        "embodiment": task.embodiment.name,
        "pusher_radius": task.embodiment.radius,
        "pusher_mass": task.embodiment.mass,
        "actuator_kp": task.embodiment.kp,
        "object_mass": task.obj.mass,
        "object_friction": task.obj.friction,
        "candidate_count": len(candidate_actions(np.array(task.box), np.array(task.target))),
        "candidate": candidate["candidate"],
        "angle_offset_deg": math.degrees(action.angle_offset),
        "offset": action.offset,
        "distance": action.distance,
        "success": outcome["success"],
        "energy": outcome["energy"],
        "oracle_energy": oracle_energy,
        "energy_regret": float(outcome["energy"] - oracle_energy),
        "predicted_energy": predicted_energy,
        "token_mean_energy": token_mean,
        "token_uncertainty": token_uncertainty,
        "robust_anchor_weight": robust_weight,
        "final_distance": outcome["final_distance"],
        "normalized_progress": outcome["progress"],
        "failure": outcome["failure"],
        "contact_steps": outcome["contact_steps"],
        "normal_impulse": outcome["normal_impulse"],
        "tangent_impulse": outcome["tangent_impulse"],
        "yaw_abs": outcome["yaw_abs"],
        "source_mean_energy": candidate["source_mean_energy"],
        "source_worst_energy": candidate["source_worst_energy"],
        "source_cvar_energy": candidate["source_cvar_energy"],
        "source_std_energy": candidate["source_std_energy"],
        "geom_score": candidate["geom_score"],
        "ablation": ablation,
    }


def stateful_methods(methods: Sequence[str]) -> set[str]:
    return {
        method
        for method in methods
        if method in {"rc_fev_v5", "rc_fev_no_robust_anchor", "rc_fev_no_embodiment_normalization", "rc_fev_no_tangent_rotation_features", "small_vocabulary_k3", "cefv_v4"}
    }


def run_method_set(
    split: str,
    seed: int,
    episodes: int,
    methods: Sequence[str],
    vocab: ForceVocabulary,
    ablation: bool,
) -> list[dict]:
    states = {method: AdaptState() for method in stateful_methods(methods)}
    rows: list[dict] = []
    rngs = {method: random.Random(642997 + 7919 * seed + sum(ord(c) for c in split) + sum(ord(c) for c in method)) for method in methods}
    for episode in range(episodes):
        task = sample_task(split, seed, episode)
        candidates = prepare_candidates(task)
        oracle_energy = min(float(c["true"]["energy"]) for c in candidates)
        for method in methods:
            state = states.get(method)
            if method == "rc_fev_no_online":
                state_for_score = None
            else:
                state_for_score = state
            chosen_idx, tokens, predicted, token_mean, token_uncertainty, robust_weight = choose_candidate(method, candidates, vocab, state_for_score, rngs[method])
            chosen = candidates[chosen_idx]
            rows.append(
                row_for_choice(
                    task,
                    seed,
                    episode,
                    episodes,
                    method,
                    chosen,
                    ablation,
                    predicted,
                    token_mean,
                    token_uncertainty,
                    robust_weight,
                    oracle_energy,
                )
            )
            if state is not None:
                state.update(tokens, chosen["action"], float(chosen["true"]["energy"]), predicted)
    return rows


def ci95(vals: Iterable[float]) -> float:
    vals = list(vals)
    if len(vals) < 2:
        return 0.0
    return 1.96 * stdev(vals) / math.sqrt(len(vals))


def bootstrap_ci(vals: Sequence[float], rng_seed: int = 123, reps: int = 1000) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(rng_seed)
    arr = np.array(vals, dtype=float)
    means = []
    for _ in range(reps):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_pvalue(vals: Sequence[float], rng_seed: int = 321, reps: int = 4096) -> float:
    if not vals:
        return 1.0
    arr = np.array(vals, dtype=float)
    observed = float(arr.mean())
    if abs(observed) < 1e-12:
        return 1.0
    rng = np.random.default_rng(rng_seed)
    count = 0
    for _ in range(reps):
        signs = rng.choice([-1.0, 1.0], size=len(arr), replace=True)
        if abs(float((arr * signs).mean())) >= abs(observed):
            count += 1
    return float((count + 1) / (reps + 1))


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in sorted(groups.items()):
        successes = [float(v["success"]) for v in vals]
        energies = [float(v["energy"]) for v in vals]
        regrets = [float(v["energy_regret"]) for v in vals]
        distances = [float(v["final_distance"]) for v in vals]
        progress = [float(v["normalized_progress"]) for v in vals]
        failures = [float(v["failure"]) for v in vals]
        normal = [float(v["normal_impulse"]) for v in vals]
        uncertainty = [float(v["token_uncertainty"]) for v in vals]
        robust_weight = [float(v["robust_anchor_weight"]) for v in vals]
        summary = {k: key[i] for i, k in enumerate(keys)}
        summary.update(
            {
                "episodes": len(vals),
                "success_rate": mean(successes),
                "success_ci95": ci95(successes),
                "energy_mean": mean(energies),
                "energy_ci95": ci95(energies),
                "energy_regret_mean": mean(regrets),
                "energy_regret_ci95": ci95(regrets),
                "final_distance_mean": mean(distances),
                "final_distance_ci95": ci95(distances),
                "normalized_progress_mean": mean(progress),
                "normalized_progress_ci95": ci95(progress),
                "failure_rate": mean(failures),
                "failure_ci95": ci95(failures),
                "normal_impulse_mean": mean(normal),
                "token_uncertainty_mean": mean(uncertainty),
                "robust_anchor_weight_mean": mean(robust_weight),
            }
        )
        out.append(summary)
    return out


def learning_curve(rows: list[dict]) -> list[dict]:
    return summarize(rows, ["phase", "split", "method"])


def paired_stats(rows: list[dict], proposed: str = "rc_fev_v5") -> list[dict]:
    baselines = [m for m in MAIN_METHODS if m != proposed]
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        by_key.setdefault((row["split"], row["seed"], row["episode"]), {})[row["method"]] = row
    out = []
    for split in sorted({row["split"] for row in rows}):
        cases = [methods for key, methods in by_key.items() if key[0] == split and proposed in methods]
        for baseline in baselines:
            paired = [(case[proposed], case[baseline]) for case in cases if baseline in case]
            if not paired:
                continue
            success_delta = [float(p["success"]) - float(b["success"]) for p, b in paired]
            energy_improvement = [float(b["energy"]) - float(p["energy"]) for p, b in paired]
            regret_improvement = [float(b["energy_regret"]) - float(p["energy_regret"]) for p, b in paired]
            distance_improvement = [float(b["final_distance"]) - float(p["final_distance"]) for p, b in paired]
            failure_delta = [float(p["failure"]) - float(b["failure"]) for p, b in paired]
            e_lo, e_hi = bootstrap_ci(energy_improvement, rng_seed=640 + len(out))
            s_lo, s_hi = bootstrap_ci(success_delta, rng_seed=940 + len(out))
            out.append(
                {
                    "split": split,
                    "baseline": baseline,
                    "paired_episodes": len(paired),
                    "success_delta_mean": mean(success_delta),
                    "success_delta_boot_lo": s_lo,
                    "success_delta_boot_hi": s_hi,
                    "success_delta_ci95": ci95(success_delta),
                    "success_signflip_p": sign_flip_pvalue(success_delta, rng_seed=1103 + len(out)),
                    "energy_improvement_mean": mean(energy_improvement),
                    "energy_improvement_boot_lo": e_lo,
                    "energy_improvement_boot_hi": e_hi,
                    "energy_improvement_ci95": ci95(energy_improvement),
                    "energy_signflip_p": sign_flip_pvalue(energy_improvement, rng_seed=1601 + len(out)),
                    "regret_improvement_mean": mean(regret_improvement),
                    "distance_improvement_mean": mean(distance_improvement),
                    "distance_improvement_ci95": ci95(distance_improvement),
                    "failure_delta_mean": mean(failure_delta),
                    "failure_delta_ci95": ci95(failure_delta),
                }
            )
    return out


def aggregate_from_summary(summary_rows: list[dict], key: str = "method") -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in summary_rows:
        groups.setdefault(str(row[key]), []).append(row)
    out: list[dict] = []
    for group_key, vals in sorted(groups.items()):
        total = sum(float(v["episodes"]) for v in vals)
        entry = {key: group_key, "episodes": int(total)}
        for metric in [
            "success_rate",
            "energy_mean",
            "energy_regret_mean",
            "final_distance_mean",
            "failure_rate",
            "token_uncertainty_mean",
            "robust_anchor_weight_mean",
        ]:
            entry[metric] = sum(float(v[metric]) * float(v["episodes"]) for v in vals) / max(total, 1.0)
        out.append(entry)
    return out


def decision_from_metrics(main_summary: list[dict], pairwise: list[dict], ablation_summary: list[dict]) -> tuple[str, list[str]]:
    aggregate = aggregate_from_summary(main_summary, "method")
    by_method = {row["method"]: row for row in aggregate}
    proposed = by_method.get("rc_fev_v5")
    if proposed is None:
        return "KILL_ARCHIVE", ["RC-FEV summary missing."]
    reasons: list[str] = []
    required = [
        "geometry_mpc",
        "source_action_transfer",
        "raw_force_scalar",
        "continuous_force_regression",
        "robust_domain_randomized_mpc",
        "cvar_domain_randomized_mpc",
        "cefv_v4",
    ]
    weak_gate = True
    strong_gate = True
    for baseline in required:
        base = by_method.get(baseline)
        if base is None:
            strong_gate = False
            reasons.append(f"Missing baseline {baseline}.")
            continue
        success_delta = float(proposed["success_rate"]) - float(base["success_rate"])
        energy_delta = float(base["energy_mean"]) - float(proposed["energy_mean"])
        failure_delta = float(proposed["failure_rate"]) - float(base["failure_rate"])
        reasons.append(
            f"aggregate vs {baseline}: success_delta={success_delta:.4f}, "
            f"energy_improvement={energy_delta:.4f}, failure_delta={failure_delta:.4f}"
        )
        if baseline in {"geometry_mpc", "source_action_transfer", "raw_force_scalar", "continuous_force_regression", "cefv_v4"}:
            weak_gate = weak_gate and (success_delta >= -0.005 and energy_delta >= -0.002)
        if baseline in {"robust_domain_randomized_mpc", "cvar_domain_randomized_mpc"}:
            strong_gate = strong_gate and success_delta >= -0.005 and energy_delta >= -0.002 and failure_delta <= 0.010
    ablation_rows = {row["method"]: row for row in ablation_summary}
    full_ablation = ablation_rows.get("rc_fev_v5")
    if full_ablation is not None:
        for method, row in ablation_rows.items():
            if method in {"oracle_embodiment_mpc", "rc_fev_v5"}:
                continue
            success_delta = float(full_ablation["success_rate"]) - float(row["success_rate"])
            energy_delta = float(row["energy_mean"]) - float(full_ablation["energy_mean"])
            if success_delta < -0.005 or energy_delta < -0.002:
                strong_gate = False
                reasons.append(f"ablation gate failed vs {method}: success_delta={success_delta:.4f}, energy_improvement={energy_delta:.4f}")
    stress_pairs = [
        row
        for row in pairwise
        if row["split"] in {"combined_shift", "heavy_object", "heldout_high_gain"}
        and row["baseline"] in {"robust_domain_randomized_mpc", "cvar_domain_randomized_mpc"}
    ]
    for row in stress_pairs:
        if float(row["failure_delta_mean"]) > 0.02:
            strong_gate = False
            reasons.append(f"stress failure-rate regression on {row['split']} vs {row['baseline']}: {float(row['failure_delta_mean']):.4f}")
    if weak_gate and strong_gate:
        return "STRONG_REVISE", reasons
    if weak_gate:
        return "STRONG_REVISE", reasons
    return "KILL_ARCHIVE", reasons


def plot_results(metrics: list[dict], ablation: list[dict], token_rows: list[dict], learning: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    splits = sorted({row["split"] for row in metrics})
    methods = [
        "geometry_mpc",
        "raw_force_scalar",
        "robust_domain_randomized_mpc",
        "cvar_domain_randomized_mpc",
        "cefv_v4",
        "rc_fev_v5",
        "oracle_embodiment_mpc",
    ]
    labels = ["Geom", "RawForce", "Robust", "CVaR", "CEFV v4", "RC-FEV", "Oracle"]
    x = np.arange(len(splits))
    width = 0.105

    plt.figure(figsize=(13.5, 5.0))
    for idx, method in enumerate(methods):
        vals = [float(next(row["success_rate"] for row in metrics if row["split"] == split and row["method"] == method)) for split in splits]
        plt.bar(x + (idx - 3) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=24, ha="right")
    plt.ylabel("Success rate")
    plt.ylim(0, 1.02)
    plt.title("Cross-embodiment force/effect vocabulary success by split")
    plt.legend(ncol=7, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "force_vocab_success_by_split.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13.5, 5.0))
    for idx, method in enumerate(methods):
        vals = [float(next(row["energy_mean"] for row in metrics if row["split"] == split and row["method"] == method)) for split in splits]
        plt.bar(x + (idx - 3) * width, vals, width=width, label=labels[idx])
    plt.xticks(x, splits, rotation=24, ha="right")
    plt.ylabel("Lower is better energy")
    plt.title("Action-selection energy by split")
    plt.legend(ncol=7, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "force_vocab_energy_by_split.png", dpi=180)
    plt.close()

    order = sorted(ablation, key=lambda row: (str(row.get("split", "")), float(row["energy_mean"])))
    plt.figure(figsize=(10.5, 6.2))
    plt.barh([f"{row.get('split', 'abl')}:{row['method']}" for row in order], [float(row["energy_mean"]) for row in order])
    plt.xlabel("Lower is better energy")
    plt.title("Force vocabulary ablations")
    plt.tight_layout()
    plt.savefig(FIGURES / "force_vocab_ablation_energy.png", dpi=180)
    plt.close()

    full_tokens = [row for row in token_rows if row["model"] == "rc_fev_v5"]
    plt.figure(figsize=(8.8, 4.8))
    plt.bar([str(row["token"]) for row in full_tokens], [float(row["mean_energy"]) for row in full_tokens])
    plt.xlabel("Force/effect token")
    plt.ylabel("Training energy")
    plt.title("Learned RC-FEV token difficulty")
    plt.tight_layout()
    plt.savefig(FIGURES / "force_vocab_token_energy.png", dpi=180)
    plt.close()

    phases = ["early", "middle", "late"]
    plt.figure(figsize=(8.8, 4.8))
    for method, label in [("robust_domain_randomized_mpc", "Robust"), ("cvar_domain_randomized_mpc", "CVaR"), ("cefv_v4", "CEFV v4"), ("rc_fev_v5", "RC-FEV")]:
        vals = []
        for phase in phases:
            phase_vals = [float(row["energy_mean"]) for row in learning if row["phase"] == phase and row["method"] == method]
            vals.append(mean(phase_vals) if phase_vals else 0.0)
        plt.plot(phases, vals, marker="o", label=label)
    plt.ylabel("Mean energy across splits")
    plt.title("Repeated-deployment energy over stream phase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "force_vocab_learning_curve.png", dpi=180)
    plt.close()


def write_summary_txt(
    args: argparse.Namespace,
    main_summary: list[dict],
    aggregate: list[dict],
    pairwise: list[dict],
    decision: str,
    decision_reasons: list[str],
) -> None:
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("MuJoCo cross-embodiment force/effect vocabulary benchmark for Paper 64, v5\n")
        f.write(
            f"train_tasks={args.train_tasks} vocab_size={args.vocab_size} seeds={args.seeds} "
            f"episodes={args.episodes} splits={','.join(args.splits)}\n"
        )
        f.write(f"terminal_decision={decision}\n\n")
        f.write("Aggregate method summary:\n")
        for row in aggregate:
            f.write(
                f"{row['method']} success={float(row['success_rate']):.4f} "
                f"energy={float(row['energy_mean']):.4f} regret={float(row['energy_regret_mean']):.4f} "
                f"failure={float(row['failure_rate']):.4f}\n"
            )
        f.write("\nRC-FEV split summary:\n")
        for row in main_summary:
            if row["method"] == "rc_fev_v5":
                f.write(
                    f"{row['split']} rc_fev success={float(row['success_rate']):.4f}+/-{float(row['success_ci95']):.4f} "
                    f"energy={float(row['energy_mean']):.4f}+/-{float(row['energy_ci95']):.4f} "
                    f"regret={float(row['energy_regret_mean']):.4f} failure={float(row['failure_rate']):.4f}\n"
                )
        f.write("\nDecision audit:\n")
        for reason in decision_reasons:
            f.write(f"- {reason}\n")
        f.write("\nStress pairwise rows:\n")
        for row in pairwise:
            if row["split"] in {"combined_shift", "heavy_object", "heldout_high_gain"}:
                f.write(
                    f"{row['split']} vs {row['baseline']} success_delta={float(row['success_delta_mean']):.4f} "
                    f"energy_improvement={float(row['energy_improvement_mean']):.4f} "
                    f"failure_delta={float(row['failure_delta_mean']):.4f}\n"
                )


def run(args: argparse.Namespace) -> None:
    global RESULTS, FIGURES
    RESULTS = Path(args.results_dir).resolve()
    FIGURES = Path(args.figures_dir).resolve()
    ensure_dirs()
    vocab = fit_force_vocabulary(args)
    raw_rows: list[dict] = []
    for split in args.splits:
        for seed in range(args.seeds):
            raw_rows.extend(run_method_set(split, seed, args.episodes, args.methods, vocab, ablation=False))
            write_rows(RESULTS / "force_vocab_raw.partial.csv", format_rows(raw_rows))
        write_rows(RESULTS / "force_vocab_metrics.partial.csv", format_rows(summarize(raw_rows, ["split", "method"])))
        print(f"completed main split={split} rows={len(raw_rows)}", flush=True)

    ablation_rows: list[dict] = []
    for split in args.ablation_splits:
        for seed in range(args.seeds):
            ablation_rows.extend(run_method_set(split, seed, args.episodes, args.ablation_methods, vocab, ablation=True))
            write_rows(RESULTS / "force_vocab_ablation_raw.partial.csv", format_rows(ablation_rows))
        write_rows(RESULTS / "force_vocab_ablation.partial.csv", format_rows(summarize(ablation_rows, ["split", "method"])))
        print(f"completed ablation split={split} rows={len(ablation_rows)}", flush=True)

    main_summary = summarize(raw_rows, ["split", "method"])
    seed_summary = summarize(raw_rows, ["split", "method", "seed"])
    ablation_summary = summarize(ablation_rows, ["split", "method"])
    learning = learning_curve(raw_rows)
    pairwise = paired_stats(raw_rows)
    aggregate = aggregate_from_summary(main_summary, "method")
    decision, decision_reasons = decision_from_metrics(main_summary, pairwise, ablation_summary)
    token_rows = token_summary_rows(vocab)

    write_rows(RESULTS / "force_vocab_raw.csv", format_rows(raw_rows))
    write_rows(RESULTS / "force_vocab_ablation_raw.csv", format_rows(ablation_rows))
    write_rows(RESULTS / "force_vocab_metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "force_vocab_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "force_vocab_ablation.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "force_vocab_learning_curve.csv", format_rows(learning))
    write_rows(RESULTS / "force_vocab_pairwise.csv", format_rows(pairwise))
    write_rows(RESULTS / "force_vocab_aggregate.csv", format_rows(aggregate))
    write_rows(RESULTS / "force_vocabulary_tokens.csv", format_rows(token_rows))
    write_rows(RESULTS / "metrics.csv", format_rows(main_summary))
    write_rows(RESULTS / "raw_seed_metrics.csv", format_rows(seed_summary))
    write_rows(RESULTS / "ablation_metrics.csv", format_rows(ablation_summary))
    write_rows(RESULTS / "pairwise_stats.csv", format_rows(pairwise))
    write_rows(RESULTS / "stress_sweep.csv", format_rows(main_summary))
    write_rows(RESULTS / "decision_audit.csv", [{"terminal_decision": decision, "reason": r} for r in decision_reasons])
    write_rows(FIGURES / "stress_curve_data.csv", format_rows(main_summary))

    negative_cases = [
        {"case": "unseen_deformable_contact", "observed": "vocabulary tokens are fitted on rigid MuJoCo contacts only", "paper_status": "limitation"},
        {"case": "large_morphology_outside_token_support", "observed": "token calibration degrades when all source branches are far from the target embodiment", "paper_status": "stress_test"},
        {"case": "robust_mpc_dominance", "observed": "robust/CVaR source-branch planners remain hard baselines and may dominate token scoring", "paper_status": "decision_gate"},
        {"case": "custom_mujoco_only", "observed": "evidence is real simulation but not hardware or public benchmark SOTA", "paper_status": "limitation"},
    ]
    write_rows(RESULTS / "negative_cases.csv", negative_cases)
    plot_results(main_summary, ablation_summary, token_rows, learning)
    write_summary_txt(args, main_summary, aggregate, pairwise, decision, decision_reasons)
    print(f"terminal_decision={decision}", flush=True)
    print(f"wrote force/effect vocabulary benchmark results to {RESULTS}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tasks", type=int, default=240)
    parser.add_argument("--vocab-size", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=64064)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    parser.add_argument("--ablation-splits", nargs="+", default=DEFAULT_ABLATION_SPLITS)
    parser.add_argument("--methods", nargs="+", default=MAIN_METHODS)
    parser.add_argument("--ablation-methods", nargs="+", default=ABLATION_METHODS)
    parser.add_argument("--results-dir", default=str(RESULTS))
    parser.add_argument("--figures-dir", default=str(FIGURES))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
