"""
Does a policy react to an incoming missile?  (diagnostic only, no training)

For every enemy file (or for one file given with --policy) the UCAV fires a missile at that aircraft from about 4 km.
While the missile flies, the aircraft's own observation is read and its policy output is computed twice:
  1. with the real observation, and
  2. with the incoming-missile block of the observation (obs[42:48]) replaced by "no missile".
The script also counts how many of the staged shots miss (the target escaped).

How to read the result: the number of missed missiles is the measure of escape. The output change only shows how sensitive the
network is to that block of the observation; a network that never saw a missile in training also changes its output when its inputs
change, so a large change does not prove purposeful evasion. A count of 0 missed means a well-aimed missile is not escaped.

    python tools\\check_enemy_reaction.py                                   # all files in models\\enemies
    python tools\\check_enemy_reaction.py --policy models\\ucav_bvr_eco\\ucav_bvr_eco_v1.zip   # a single file
    python tools\\check_enemy_reaction.py --n 30                            # 30 staged shots per file

Use the same UCAV_* settings as in the run you want to explain (for the final model: $env:UCAV_COMBAT="bvr").
The staged test is simple (the shooter holds a straight course); it measures reaction, not combat skill.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))
os.environ.setdefault("UCAV_RENDER_MODE", "none")

import numpy as np  # noqa: E402
import torch  # noqa: E402  (torch is imported before Panda3D)
from stable_baselines3 import PPO  # noqa: E402

from combat.ucav_config import UCAV_CONFIG as DC, describe_settings  # noqa: E402
from combat.ucav_env import UCAVEnv  # noqa: E402
from combat.enemy_pool import PolicyEnemy  # noqa: E402

THR = slice(42, 48)                                   # incoming-missile block of the 51-number observation
NO_THREAT = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0], np.float32)
FIRE = np.array([0.0, 0.0, 0.0, 0.0, 1.0], np.float32)
IDLE = np.array([0.0, 0.0, 0.0, 0.0, -1.0], np.float32)
CHANNELS = ("elevator", "aileron", "rudder", "throttle")


def stage_shot(env, model, name, seed, distance):
    """Fire one UCAV missile at the policy's aircraft; return (list of |difference| per step, missile_hit)."""
    env.set_forced_opponent(PolicyEnemy(model, name, "attacker", False))
    env.reset(seed=seed)
    p, e = env.sim.player, env.sim.enemy
    e.position = p.position + p.get_forward_vector() * distance
    diffs, hit, action = [], False, FIRE
    for _ in range(220):
        _, _, te, tr, info = env.step(action)
        action = IDLE
        obs = env._obs_for("red")
        if obs[42] > 0.5:                              # a missile is inbound
            real, _ = model.predict(obs, deterministic=True)
            masked = obs.copy()
            masked[THR] = NO_THREAT
            blind, _ = model.predict(masked, deterministic=True)
            diffs.append(np.abs(np.asarray(real[:4]) - np.asarray(blind[:4])))
        if te or tr:
            hit = bool(info["episode_record"]["missile_hit"])
            break
    return diffs, hit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enemy-dir", default=str(ROOT / "models" / "enemies"))
    ap.add_argument("--policy", default=None, help="test one policy file instead of the enemy folder")
    ap.add_argument("--n", type=int, default=20, help="staged shots per policy")
    ap.add_argument("--distance", type=float, default=4000.0, help="shooting distance in metres")
    a = ap.parse_args()

    DC["REPLAY_PROB"] = 0.0
    files = [Path(a.policy)] if a.policy else sorted(Path(a.enemy_dir).glob("*.zip"))
    if not files:
        sys.exit(f"no policy files found ({a.enemy_dir})")
    print(describe_settings())
    env = UCAVEnv(stage=4)
    print(f"{'policy':34s} {'missiles that missed':>21s}   sensitivity: mean |change in output| when a missile is inbound")
    print(f"{'':34s} {'':>21s}   " + "  ".join(f"{c:>8s}" for c in CHANNELS))
    for f in files:
        model = PPO.load(str(f), device="cpu")
        all_d, misses = [], 0
        for k in range(a.n):
            d, hit = stage_shot(env, model, f.stem, 1000 + k, a.distance)
            all_d.extend(d)
            misses += 0 if hit else 1
        if all_d:
            m = np.mean(np.stack(all_d), axis=0)
            cols = "  ".join(f"{x:8.3f}" for x in m)
        else:
            cols = "  ".join(f"{'n/a':>8s}" for _ in CHANNELS)
        print(f"{f.stem:34s} {misses:3d} of {a.n:<3d} missed        {cols}")
    print("\nThe missed count is the measure of escape. The sensitivity columns do not prove evasion (see the top of this file).")


if __name__ == "__main__":
    main()
