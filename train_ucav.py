"""
Autonomous UCAV combat training: continuous raw-stick PPO, four stages, varied learned enemies.  Laptop-safe.

  python train_ucav.py                                  # stages 1 -> 4 in ONE run, 3 envs
  python train_ucav.py --n-envs 2 --cooldown 0.3        # cooler / lighter on RAM
  python train_ucav.py --timesteps 400000 --stop-after-stage 1
  python train_ucav.py --resume models/ucav/ucav_final.zip --resume-vecnorm models/ucav/ucav_vecnormalize_final.pkl --start-stage 3

Only the UCAV learns.  The enemy is part of the environment: each episode draws a frozen LEARNED policy from
models/enemies/*.zip (see combat/enemy_pool.py).  Copy the snapshots / final model of an earlier run there.

Stages (one skill per stage, firing only in stage 4):
  1 flight   learn to fly
  2 pursuit  reach and hold a position of advantage behind the enemy (nobody has weapons)
  3 defence  defeat the enemy's missiles, stay in the zone (enemy armed, UCAV unarmed)
  4 combat   fire when it makes sense (the episode ends at the hit)
Every stage keeps the earlier rewards, and 20% of episodes replay an earlier stage (against forgetting).
Stages 1-3 stop with a NO-GO message if their criterion is not met within the step budget (--force-promote overrides).
"""
import argparse
import collections
import datetime
import math
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["UCAV_RENDER_MODE"] = "none"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

import torch  # noqa: E402  (before Panda3D on Windows)
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import get_schedule_fn  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # noqa: E402
from combat.ucav_config import UCAV_CONFIG as DC, describe_settings  # noqa: E402
from combat.metrics import MetricsTracker  # noqa: E402

SELF_FAIL = ("UCAV_CRASH", "ZONE_VIOLATION", "FLIGHT_ENVELOPE")


def lower_priority():
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(10)
    except Exception:
        pass


def build_env(stage, enemy_dir):
    lower_priority()
    from combat.ucav_env import UCAVEnv
    return Monitor(UCAVEnv(stage=stage, enemy_dir=str(enemy_dir)))


class StageCallback(BaseCallback):
    def __init__(self, tracker, args):
        super().__init__()
        self.tr, self.args = tracker, args
        self.stage = args.start_stage
        self.stage_start = 0
        self.ep_id = 0
        self.recs = collections.deque(maxlen=400)       # valid episodes of the CURRENT stage only
        self.t0 = time.time()
        self.t_print = 0.0
        self.t_roll = time.time()
        self.no_go = False
        self.passed = set()

    def _frac(self, recs, fn):
        return sum(1 for r in recs if fn(r)) / max(len(recs), 1)

    def _on_training_start(self):
        self.training_env.env_method("set_stage", self.stage)
        self.stage_start = self.num_timesteps
        if self.stage == 4:
            self.model.lr_schedule = get_schedule_fn(DC["CURRICULUM"]["LR_STAGE4"])

    def _on_rollout_start(self):
        self.t_roll = time.time()

    def _on_rollout_end(self):
        if self.args.cooldown > 0:
            time.sleep(self.args.cooldown * (time.time() - self.t_roll))

    def _on_step(self):
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            rec = self.locals["infos"][i].get("episode_record")
            if rec is None:
                continue
            self.ep_id += 1
            rec["episode_id"] = self.ep_id
            row = self.tr.add(rec, self.num_timesteps, self.stage)
            if rec["valid"] and rec["stage"] == self.stage and not rec["replay"]:
                self.recs.append(rec)
            if row and time.time() - self.t_print > 20:
                self.t_print = time.time()
                self._print(row)
            if not self._check_stage():
                return False
        return True

    def _print(self, row):
        recent = list(self.recs)[-DC["CURRICULUM"]["WINDOW"]:]
        sps = self.num_timesteps / max(time.time() - self.t0, 1)
        ok = self._frac(recent, lambda r: r["success"])
        fail = self._frac(recent, lambda r: r["failure_reason"] in SELF_FAIL)
        hit = self._frac(recent, lambda r: r["ucav_survival"] is False and r["failure_reason"] == "UCAV_LOST")
        shots = sum(r["launch_attempts"] for r in recent)
        kills = sum(1 for r in recent if r["missile_hit"])
        es = sum(r["enemy_shots"] for r in recent)
        dfd = sum(r["enemy_missiles_defeated"] for r in recent) / max(es, 1)
        adv = sum(r["advantage_frac"] for r in recent) / max(len(recent), 1)
        print(f"[S{self.stage}] ep {self.ep_id:6d} step {self.num_timesteps:9d} | {sps:5.0f} sps | R {row['mean_reward']:7.1f} "
              f"success {ok:.2f} selffail {fail:.2f} ucav_hit {hit:.2f} close {adv:.2f} "
              f"shots/ep {shots / max(len(recent), 1):.2f} hit/shot {kills / max(shots, 1):.2f} defended {dfd:.2f}", flush=True)

    def _criterion(self):
        W = DC["CURRICULUM"]["WINDOW"]
        recs = list(self.recs)[-W:]
        if len(recs) < W:
            return False, f"only {len(recs)}/{W} valid stage-{self.stage} episodes so far"
        cfg = DC["CURRICULUM"][f"S{self.stage}"]
        fail = self._frac(recs, lambda r: r["failure_reason"] in SELF_FAIL)
        if self.stage == 1:
            s = self._frac(recs, lambda r: r["success"])
            return s >= cfg["success"], f"hold success {s:.2f} (need {cfg['success']})"
        if self.stage == 2:
            s = self._frac(recs, lambda r: r["success"])
            return (s >= cfg["success"] and fail <= cfg["fail_max"],
                    f"pursuit success {s:.2f} (need {cfg['success']}), self-fail {fail:.2f} (max {cfg['fail_max']})")
        if self.stage == 3:
            shots = sum(r["enemy_shots"] for r in recs)
            if shots < cfg["min_enemy_shots"]:
                return False, f"only {shots} enemy missiles seen (need {cfg['min_enemy_shots']})"
            surv = self._frac(recs, lambda r: r["success"])
            dfd = sum(r["enemy_missiles_defeated"] for r in recs) / shots
            return (surv >= cfg["survive"] and dfd >= cfg["defended"] and fail <= cfg["fail_max"],
                    f"survived {surv:.2f} (need {cfg['survive']}), defended {dfd:.2f} (need {cfg['defended']}), "
                    f"self-fail {fail:.2f} (max {cfg['fail_max']})")
        return False, "final stage"

    def _check_stage(self):
        if self.stage >= 4:
            return True
        cfg = DC["CURRICULUM"][f"S{self.stage}"]
        steps = self.num_timesteps - self.stage_start
        if steps < cfg["min_steps"] and steps < cfg["max_steps"]:
            return True
        ok, msg = self._criterion()
        if ok or (steps >= cfg["max_steps"] and self.args.force_promote):
            print(f"==> STAGE {self.stage} {'PASSED' if ok else 'FORCE-PROMOTED'} after {steps} steps: {msg}", flush=True)
            if ok:
                self.passed.add(self.stage)
            out = Path(self.args.out_dir)
            self.model.save(str(out / f"ucav_stage{self.stage}_done.zip"))
            self.training_env.save(str(out / f"ucav_stage{self.stage}_done_vecnormalize.pkl"))
            if self.args.stop_after_stage == self.stage:
                print(f"==> STOPPED AS REQUESTED after stage {self.stage}.", flush=True)
                return False
            self.stage += 1
            self.stage_start = self.num_timesteps
            self.recs.clear()
            if self.stage == 4:
                self.model.lr_schedule = get_schedule_fn(DC["CURRICULUM"]["LR_STAGE4"])
            self.training_env.env_method("set_stage", self.stage)
            return True
        if steps >= cfg["max_steps"]:
            print(f"\n==> NO-GO: stage {self.stage} missed its criterion within {cfg['max_steps']} steps: {msg}\n"
                  f"    Checkpoints are saved. Inspect logs/ucav/episodes.jsonl (reward_parts) before continuing.", flush=True)
            self.no_go = True
            return False
        return True

    def _on_training_end(self):
        if self.no_go or self.stage >= 4 or self.stage in self.passed:
            return
        ok, msg = self._criterion()
        print(f"==> TRAINING ENDED INSIDE STAGE {self.stage} after {self.num_timesteps - self.stage_start} steps "
              f"({'criterion met: ' if ok else 'not yet: '}{msg})", flush=True)


def make_model(venv, args):
    P = DC["PPO"]
    model = PPO("MlpPolicy", venv, device="cpu", seed=args.seed, verbose=0,
                learning_rate=P["learning_rate"], n_steps=P["n_steps"], batch_size=P["batch_size"],
                n_epochs=P["n_epochs"], gamma=P["gamma"], gae_lambda=P["gae_lambda"],
                clip_range=P["clip_range"], ent_coef=P["ent_coef"], vf_coef=P["vf_coef"],
                max_grad_norm=P["max_grad_norm"],
                policy_kwargs=dict(net_arch=dict(pi=P["net_arch"], vf=P["net_arch"]),
                                   activation_fn=torch.nn.Tanh, log_std_init=P["log_std_init"]))
    with torch.no_grad():                    # fire starts as a rare event (stage 4 only ever uses it)
        model.policy.action_net.bias[4] = DC["FIRE_INIT_MEAN"]
        model.policy.log_std[4] = math.log(DC["FIRE_INIT_STD"])
    return model


def main():
    L = DC["LAPTOP"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=L["TOTAL_TIMESTEPS"])
    ap.add_argument("--n-envs", type=int, default=L["N_ENVS"])
    ap.add_argument("--checkpoint-every", type=int, default=L["CHECKPOINT_EVERY"])
    ap.add_argument("--cooldown", type=float, default=L["COOLDOWN"])
    ap.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--stop-after-stage", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-vecnorm", default=None)
    ap.add_argument("--force-promote", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "ucav"))
    ap.add_argument("--log-dir", default=str(ROOT / "logs" / "ucav"))
    ap.add_argument("--enemy-dir", default=str(ROOT / "models" / "enemies"))
    ap.add_argument("--dummy", action="store_true", help="single process (debugging)")
    args = ap.parse_args()

    torch.set_num_threads(L["TORCH_THREADS"])
    lower_priority()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # never silently destroy a previous result: back up an existing ucav_final.zip before this run can overwrite it
    prev = out / "ucav_final.zip"
    if prev.exists():
        stamp = datetime.datetime.fromtimestamp(prev.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        bk = out / f"ucav_final_before_{stamp}.zip"
        if not bk.exists():
            shutil.copy2(prev, bk)
            pv = out / "ucav_vecnormalize_final.pkl"
            if pv.exists():
                shutil.copy2(pv, out / f"ucav_vecnormalize_before_{stamp}.pkl")
        print(f"NOTE: existing ucav_final.zip was backed up as {bk.name}. This run overwrites ucav_final.zip when it stops "
              f"(including on Ctrl+C); use --out-dir for experiments.", flush=True)
    enemy_dir = Path(args.enemy_dir)
    enemy_dir.mkdir(parents=True, exist_ok=True)
    n_enemy = len([p for p in enemy_dir.glob("*.zip")])
    if args.start_stage >= 2 and n_enemy == 0:
        print(f"WARNING: no enemy policies in {enemy_dir}. The enemy will be the NULL stick (no controller), "
              f"which is not a useful opponent. Copy trained policy .zip files there.", flush=True)

    fns = [(lambda st=args.start_stage: build_env(st, enemy_dir)) for _ in range(args.n_envs)]
    venv = DummyVecEnv(fns[:1]) if args.dummy else SubprocVecEnv(fns, start_method="spawn")
    P = DC["PPO"]
    if args.resume_vecnorm:
        venv = VecNormalize.load(args.resume_vecnorm, venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=P["gamma"])
    model = PPO.load(args.resume, env=venv, device="cpu") if args.resume else make_model(venv, args)

    tracker = MetricsTracker(args.log_dir)
    ckpt = CheckpointCallback(save_freq=max(args.checkpoint_every // max(args.n_envs, 1), 1), save_path=str(out),
                              name_prefix="ucav", save_vecnormalize=True)
    cb = StageCallback(tracker, args)
    print(f"UCAV training: {args.timesteps} steps, {args.n_envs} envs, start stage {args.start_stage}, "
          f"{n_enemy} enemy policies in {enemy_dir}", flush=True)
    print(describe_settings(), flush=True)
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=[ckpt, cb], progress_bar=False,
                    reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("interrupted - saving", flush=True)
    model.save(str(out / "ucav_final.zip"))
    venv.save(str(out / "ucav_vecnormalize_final.pkl"))
    tag = f"ucav_last_stage{cb.stage}_{model.num_timesteps}steps"       # unambiguous name: which stage, how many steps
    model.save(str(out / f"{tag}.zip"))
    venv.save(str(out / f"{tag}_vecnormalize.pkl"))
    status = ("curriculum complete" if cb.stage >= 4 else "stage criterion passed" if cb.stage in cb.passed
              else "NO-GO" if cb.no_go else "incomplete")
    print(f"saved {out}/ucav_final.zip ({time.time() - t0:.0f}s, stage {cb.stage}: {status})", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
