"""
Fixed-seed evaluation of the autonomous UCAV.  Results are hit-based: no win / loss / draw.

  python evaluate_ucav.py --policy models/ucav/ucav_final.zip                    # stage-4 combat vs every enemy in models/enemies
  python evaluate_ucav.py --policy models/ucav/ucav_final.zip --episodes 100
  python evaluate_ucav.py --policy models/ucav/ucav_final.zip --skills           # all four stage tests (retention / forgetting check)

Per enemy: target-hit rate, mission success (hit + return home), UCAV-hit rate, self-failure (crash / zone / stall),
missiles fired per episode, hits per missile, launch requests blocked by the envelope, enemy missiles defeated,
mean time to hit.  Episodes where the ENEMY left the zone or crashed are invalid and reported separately.
Same seeds for every enemy, so numbers are comparable.  Stick = mean action, fire channel sampled.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["UCAV_RENDER_MODE"] = "none"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

import torch  # noqa: E402,F401
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from combat.ucav_config import UCAV_CONFIG as DC, describe_settings  # noqa: E402
from combat.enemy_pool import load_enemy, act_policy  # noqa: E402
from combat.ucav_env import UCAVEnv  # noqa: E402

DC["REPLAY_PROB"] = 0.0          # evaluation tests exactly one stage's task


def run(env, model, enemy, stage, episodes, seed, stochastic):
    env.set_stage(stage)
    env.set_forced_opponent(enemy)
    recs = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, first = False, None
        while not done:
            obs, _, te, tr, info = env.step(act_policy(model, obs, stochastic))
            done = te or tr
            if first is None and info["events"]["launch"]:
                first = (float(info["time"]), float(info["distance"]))       # when and from how far the UCAV first fired
        rec = dict(info["episode_record"])
        rec["first_shot_t"], rec["first_shot_R"] = (first if first else (None, None))
        recs.append(rec)
    return recs


def stats(recs):
    valid = [r for r in recs if r["valid"]]
    n = max(len(valid), 1)
    f = lambda fn: sum(1 for r in valid if fn(r)) / n
    shots = sum(r["launch_attempts"] for r in valid)
    hits = [r for r in valid if r["missile_hit"]]
    fired = [r for r in valid if r.get("first_shot_t") is not None]
    es = sum(r["enemy_shots"] for r in valid)
    return dict(
        episodes=len(recs), invalid_enemy_left=len(recs) - len(valid),
        target_hit=f(lambda r: r["missile_hit"]), mission_success=f(lambda r: r["success"]),
        hit_all=sum(1 for r in recs if r["missile_hit"]) / max(len(recs), 1),     # hits over ALL episodes (incl. enemy-left)
        ucav_hit=f(lambda r: r["failure_reason"] == "UCAV_LOST"),
        self_fail=f(lambda r: r["failure_reason"] in ("UCAV_CRASH", "ZONE_VIOLATION", "FLIGHT_ENVELOPE")),
        survived=f(lambda r: r["result"] == "survived"),
        advantage_frac=float(np.mean([r["advantage_frac"] for r in valid])) if valid else 0.0,
        missiles_per_ep=shots / n, hits_per_missile=len(hits) / max(shots, 1),
        blocked_per_ep=sum(r["blocked_launches"] for r in valid) / n,
        enemy_missiles_per_ep=es / n, defended=sum(r["enemy_missiles_defeated"] for r in valid) / max(es, 1),
        mean_reward=float(np.mean([r["episode_reward"] for r in valid])) if valid else 0.0,
        time_to_hit=float(np.mean([r["episode_duration"] for r in hits])) if hits else float("nan"),
        hold_ok=f(lambda r: r["result"] == "hold_ok"), pursuit_ok=f(lambda r: r["result"] == "pursuit_ok"),
        # how the fight is fought: how soon and from how far the UCAV fires its first missile
        early_shot=(sum(1 for r in fired if r["first_shot_t"] < 3.0) / len(fired)) if fired else float("nan"),
        first_shot_t=float(np.median([r["first_shot_t"] for r in fired])) if fired else float("nan"),
        first_shot_R=float(np.median([r["first_shot_R"] for r in fired])) if fired else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--enemy-dir", default=str(ROOT / "models" / "enemies"))
    ap.add_argument("--enemies", nargs="*", default=None, help="enemy file stems to use (default: all)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123456)
    ap.add_argument("--stage", type=int, default=4, choices=[1, 2, 3, 4])
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--skills", action="store_true", help="run the four stage tests (retention check)")
    ap.add_argument("--out", default=None, help="write a JSON report")
    a = ap.parse_args()

    model = PPO.load(a.policy, device="cpu")
    env = UCAVEnv(stage=1)
    files = sorted(Path(a.enemy_dir).glob("*.zip"))
    if a.enemies:
        files = [f for f in files if f.stem in a.enemies]
    enemies = [load_enemy(f, stochastic=a.stochastic) for f in files]
    t0, report = time.time(), {}
    print(f"policy {a.policy}  episodes {a.episodes}  seeds {a.seed}+  enemies: {[e.name for e in enemies]}")
    print(describe_settings())

    if a.skills:
        print(f"{'stage':28s} {'key metric':34s} detail")
        for s in (1, 2, 3, 4):
            if s == 1:
                recs = run(env, model, None, 1, a.episodes, a.seed, a.stochastic)
                st = stats(recs)
                line = f"hold success {st['hold_ok']:.2f}  self-fail {st['self_fail']:.2f}"
            else:
                recs = []
                per = max(a.episodes // max(len(enemies), 1), 1)
                for e in enemies:
                    recs += run(env, model, e, s, per, a.seed, a.stochastic)
                st = stats(recs)
                line = {2: f"pursuit success {st['pursuit_ok']:.2f}  advantage {st['advantage_frac']:.2f}  self-fail {st['self_fail']:.2f}",
                        3: f"survived {st['survived']:.2f}  defended {st['defended']:.2f}  self-fail {st['self_fail']:.2f}",
                        4: f"target hit {st['target_hit']:.2f}  hits/missile {st['hits_per_missile']:.2f}  UCAV hit {st['ucav_hit']:.2f}"}[s]
            report[f"stage{s}"] = st
            names = {1: "1 flight", 2: "2 pursuit", 3: "3 defence", 4: "4 combat"}
            print(f"{names[s]:28s} {line}", flush=True)
    else:
        head = (f"{'enemy':22s} {'kind':8s} hit   hitAll mission ucavhit selffail msl/ep hit/msl blocked/ep "
                f"enemy_msl defended  ttHit  early<3s t1st  R1st invalid")
        print(head)
        for e in enemies:
            st = stats(run(env, model, e, a.stage, a.episodes, a.seed, a.stochastic))
            report[e.name] = st
            print(f"{e.name:22s} {e.kind:8s} {st['target_hit']:4.2f}  {st['hit_all']:6.2f} {st['mission_success']:6.2f}  {st['ucav_hit']:6.2f} "
                  f"{st['self_fail']:7.2f} {st['missiles_per_ep']:6.2f} {st['hits_per_missile']:7.2f} {st['blocked_per_ep']:10.2f} "
                  f"{st['enemy_missiles_per_ep']:9.2f} {st['defended']:8.2f} {st['time_to_hit']:6.1f}  {st['early_shot']:7.2f} {st['first_shot_t']:5.1f} {st['first_shot_R']:5.0f}  {st['invalid_enemy_left']}",
                  flush=True)
        if report:
            keys = ("target_hit", "hit_all", "ucav_hit", "self_fail", "hits_per_missile")
            print("MEAN over enemies:  " + "  ".join(f"{k} {np.mean([v[k] for v in report.values()]):.2f}" for k in keys))
    print(f"({time.time() - t0:.0f}s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(policy=a.policy, episodes=a.episodes, seed=a.seed, results=report), indent=2))
    env.close()


if __name__ == "__main__":
    main()
