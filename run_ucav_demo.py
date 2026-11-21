"""
Real-time visual demo of the autonomous UCAV against a learned enemy.

  python run_ucav_demo.py --policy models/ucav/ucav_final.zip --debug
  python run_ucav_demo.py --policy models/ucav/ucav_final.zip --enemy attacker_final --stage 4
  python run_ucav_demo.py --policy models/ucav/ucav_stage2_done.zip --stage 2          # pursuit only, nobody armed
  python run_ucav_demo.py --policy models/ucav/ucav_stage3_done.zip --stage 3          # defence: enemy fires, UCAV unarmed
  python run_ucav_demo.py --policy models/ucav/ucav_stage1_done.zip --stage 1          # flight only

--enemy: file stem inside --enemy-dir (default: a random enemy from the population each episode).
Keys: C = cycle camera (auto / chase / cockpit)   ESC = quit.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "simulator"))
ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--enemy", default=None)
ap.add_argument("--enemy-dir", default=str(ROOT / "models" / "enemies"))
ap.add_argument("--stage", type=int, default=4, choices=[1, 2, 3, 4])
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--episodes", type=int, default=0, help="0 = loop forever")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--scenario", default=None, choices=["tail_chase", "head_on", "beam", "enemy_on_six", "neutral"])
ap.add_argument("--debug", action="store_true")
ap.add_argument("--offscreen", default=None, help="test mode: software-render PNGs into this directory")
ap.add_argument("--shot-every", type=int, default=25)
args = ap.parse_args()

os.environ["UCAV_RENDER_MODE"] = "offscreen" if args.offscreen else "human"
sys.path.insert(0, str(ROOT))

import torch  # noqa: F401,E402  (before Panda3D on Windows)
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from panda3d.core import Filename, loadPrcFileData  # noqa: E402
if not args.offscreen:
    loadPrcFileData("", "window-title Autonomous UCAV")
    loadPrcFileData("", "win-size 1280 720")
    loadPrcFileData("", "sync-video 0")
from combat.enemy_pool import load_enemy, act_policy  # noqa: E402
from combat.ucav_env import UCAVEnv  # noqa: E402
from combat.ucav_config import UCAV_CONFIG as DC, describe_settings  # noqa: E402

DC["REPLAY_PROB"] = 0.0     # replay of earlier stages is a training aid only; a demo shows one stage's task
print(describe_settings(), flush=True)

LABEL = {"target_hit": "TARGET HIT", "ucav_hit": "UCAV HIT", "ucav_failed": "MISSION FAILED", "survived": "SURVIVED",
         "pursuit_ok": "CLOSE ON TARGET", "pursuit_fail": "NOT CLOSE ENOUGH", "hold_ok": "HOLD OK", "hold_fail": "HOLD FAILED",
         "no_hit": "NO HIT", "target_hit_no_return": "TARGET HIT - NOT RETURNED", "mutual_hit": "BOTH AIRCRAFT HIT",
         "enemy_out": "ENEMY LEFT THE AREA"}
GOOD = ("target_hit", "survived", "pursuit_ok", "hold_ok")


def main():
    model = PPO.load(args.policy, device="cpu")
    env = UCAVEnv(stage=args.stage, enemy_dir=args.enemy_dir)
    if args.enemy:
        env.set_forced_opponent(load_enemy(Path(args.enemy_dir) / f"{args.enemy}.zip"))
    env.forced_scenario = args.scenario
    sim = env.sim
    save_dir = Path(args.offscreen) if args.offscreen else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    frame_no, t_last = [0], [time.perf_counter()]
    state = {"info": None, "a": None}
    acos = lambda x: math.degrees(math.acos(max(-1.0, min(1.0, x))))

    def hud(env_):
        info = state["info"] or env_.last_info
        p, e = sim.player, sim.enemy
        my, em = env_._msl("ucav"), env_._msl("enemy")
        left = (f"UCAV   SPD {p.current_speed:4.0f} m/s   ALT {p.altitude:5.0f} m   G {p.g_force:3.1f}   AOA {info['aoa_deg']:4.1f}\n"
                f"       HDG {p.yaw % 360:3.0f}   PITCH {p.pitch:4.0f}   ROLL {p.roll:4.0f}\n"
                f"ZONE   {100 * (p.position - env_.home).length() / env_.R:3.0f}% of radius  (leaving = mission failed)\n")
        if env_.foe_present:
            R = (e.position - p.position).length()
            left += (f"ENEMY  {info['enemy']}   RNG {R:6.0f} m   ATA {acos(info['cos_ata']):4.0f} deg   "
                     f"CLOSURE {info['closure_rate']:+5.0f} m/s   {'DESTROYED' if e.crashed else ''}\n"
                     f"       SPD {e.current_speed:4.0f} m/s   ALT {e.altitude:5.0f} m\n")
            if env_.stage_cfg["own_weapons"]:
                left += f"MISSILES {env_.ammo_blue}   enemy missiles {env_.ammo_red}\n"
            elif env_.stage_cfg["enemy_weapons"]:
                left += f"UCAV UNARMED   enemy missiles {env_.ammo_red}\n"
            else:
                left += f"TIME CLOSE TO TARGET {100 * info['advantage_frac']:3.0f}%\n"
        else:
            left += f"HOLD TARGET  speed {env_.v_ref:.0f} m/s   alt {env_.h_ref:.0f} m\n"
        if my is not None:
            left += (f"MISSILE  SPD {my['speed']:4.0f} m/s  RNG TO TARGET {(my['target'].position - my['pos']).length():5.0f} m  "
                     f"T+{my['age']:4.1f}s\n")
        if em is not None:
            left += f"!! MISSILE INBOUND {(em['pos'] - p.position).length():5.0f} m\n"
        left += f"seed {info['seed']}  scenario {info['scenario']}  stage {info['episode_stage']}"
        if args.debug:
            top = sorted(info["reward_parts"].items(), key=lambda kv: -abs(kv[1]))[:6]
            a = state["a"]
            left += ("\n--- DEBUG ---\nstick [elev ail rud thr fire]: "
                     + (" ".join(f"{x:+.2f}" for x in a) if a is not None else "-")
                     + f"   ep return {env_.ep_return:7.1f}   blocked requests {env_.blocked_blue}\nreward parts: "
                     + "  ".join(f"{k}:{v:+.1f}" for k, v in top))
        if em is not None:
            b, c = "MISSILE WARNING", (1, 0.25, 0.25, 1)
        elif my is not None:
            b, c = "MISSILE AWAY", (1, 1, 0.4, 1)
        elif env_.killed and env_.rtb_required:
            b, c = "TARGET HIT - RETURN TO ZONE", (0.5, 0.9, 1, 1)
        else:
            b, c = "", (1, 1, 1, 1)
        sim.set_hud(left, b, c)

    def frame(env_):
        hud(env_)
        ok = sim.render_frame(env_.DT)
        frame_no[0] += 1
        if save_dir is not None:
            if frame_no[0] % args.shot_every == 0 and frame_no[0] // args.shot_every < 600:
                sim.win.saveScreenshot(Filename.fromOsSpecific(str(save_dir / f"f{frame_no[0]:05d}.png")))
        else:
            my = env_._msl("ucav")
            slow = 0.45 if (my is not None and (my["target"].position - my["pos"]).length() < 700) else 1.0
            wait = env_.DT / (args.speed * slow) - (time.perf_counter() - t_last[0])
            if wait > 0:
                time.sleep(wait)
            t_last[0] = time.perf_counter()
        return ok

    ep, hits, shown = 0, 0, 0
    while args.episodes == 0 or ep < args.episodes:
        seed = args.seed if (args.seed is not None and ep == 0) else None
        obs, info = env.reset(seed=seed)
        state["info"] = info
        print(f"\n=== Episode {ep + 1}  seed={info['seed']}  scenario={info['scenario']}  enemy={info['enemy']} ===")
        done, window_open = False, True
        while not done and window_open:
            action = act_policy(model, obs, False)
            state["a"] = action
            obs, r, te, tr, info = env.step(action, frame_callback=frame)
            state["info"] = info
            done = te or tr
            ev = info["events"]
            t = info["time"]
            if ev["launch"]:
                print(f"[t={t:5.1f}s] MISSILE LAUNCHED  range={info['distance']:.0f} m  ATA={acos(info['cos_ata']):.0f} deg")
            if ev["enemy_launch"]:
                print(f"[t={t:5.1f}s] ENEMY MISSILE LAUNCHED")
            if ev["enemy_miss"]:
                print(f"[t={t:5.1f}s] enemy missile defeated")
            if ev["own_miss"]:
                print(f"[t={t:5.1f}s] our missile missed")
            if ev["own_hit"]:
                print(f"[t={t:5.1f}s] TARGET HIT")
            window_open = sim.window_open()
        if not window_open:
            break
        label = LABEL.get(info["result"], info["result"].upper())
        hits += int(info["killed"])
        shown += 1
        extra = f"   target hits so far: {hits}/{shown}" if env.stage_cfg["own_weapons"] else ""
        print(f"RESULT: {label}  ({info['termination_reason']})  t={info['time']:.1f}s{extra}")
        sim.set_big_text(label, (0.4, 1, 0.4, 1) if info["result"] in GOOD else (1, 0.35, 0.3, 1))
        env.tail(2.5 if info["termination_reason"] in ("target_hit", "shot_down", "mission_complete") else 1.5, frame)
        sim.set_big_text("")
        ep += 1
    env.close()


if __name__ == "__main__":
    main()
