"""Structured per-episode records (JSONL) and rolling training metrics (CSV)  - spec items 31, 32."""
from __future__ import annotations
import json, collections, csv, time
from pathlib import Path


def episode_record(env, info, episode_id):
    ic = info["init_cond"]
    return dict(
        episode_id=episode_id, episode_seed=info["seed"], level=info["level"], scenario=info["scenario"],
        initial_range=round(ic["range"], 1), initial_bearing=round(ic["bearing"], 1),
        initial_altitude=round(ic["altitude"], 1), initial_speed=round(ic["speed"], 1),
        enemy_behavior=dict(style=info["enemy_style"], aggression=round(env.brain.aggression, 2),
                            react_time=round(env.brain.react_time, 2), ammo=env.brain.missiles),
        maneuver_sequence=[],
        n_maneuvers=info["n_decisions"], max_range=None, minimum_range=round(info["min_range"], 1),
        launch_attempts=info["shots_fired"], launch_success=info["shots_fired"] > 0,
        missile_hit=bool(info["hit"]), missile_miss=info["shots_fired"] - int(info["hit"]) > 0,
        returned_after_hit=bool(info["hit"] and info["termination_reason"] == "mission_complete"),
        zone_violation=info["termination_reason"] == "zone_exit", zone_max_frac=round(info["zone_max_frac"], 3),
        mission_success=bool(info["success"]), failure_reason=info["failure_reason"],
        termination=info["termination_reason"], episode_reward=round(env.ep_return, 2),
        episode_duration=round(info["time"], 1), ucav_survival=info["termination_reason"] not in ("ucav_shot_down", "ground_collision", "midair_collision", "stall"),
        enemy_survival=not info["hit"], enemy_shots=info["enemy_shots"],
        enemy_missiles_defeated=info["enemy_missiles_defeated"],
        launch_range=None, reward_parts={k: round(v, 2) for k, v in info["reward_parts"].items()},
    )


class MetricsTracker:
    def __init__(self, out_dir, window=100, log_every=20):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = open(self.dir / "episodes.jsonl", "a", encoding="utf-8")
        self.csv_path = self.dir / "training_metrics.csv"
        self.win = collections.deque(maxlen=window)
        self.n = 0; self.log_every = log_every; self.t0 = time.time(); self.header = False

    def add(self, rec, timesteps, level):
        self.n += 1
        self.jsonl.write(json.dumps(rec) + "\n"); self.jsonl.flush()
        self.win.append(rec)
        if self.n % self.log_every == 0:
            row = self.summary(); row.update(episodes=self.n, timesteps=timesteps, level=level,
                                             wall_s=round(time.time() - self.t0))
            new = not self.csv_path.exists()
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row)); 
                if new: w.writeheader()
                w.writerow(row)
            return row
        return None

    def summary(self):
        w = list(self.win); n = max(len(w), 1)
        f = lambda k: sum(1 for r in w if r[k]) / n
        m = lambda k: sum(r[k] for r in w) / n
        return dict(
            mean_reward=round(m("episode_reward"), 2), mission_success=round(f("mission_success"), 3),
            survival=round(f("ucav_survival"), 3), launch_rate=round(f("launch_success"), 3),
            hit_rate=round(f("missile_hit"), 3),
            miss_rate=round(sum(1 for r in w if r["launch_attempts"] > 0 and not r["missile_hit"]) / n, 3),
            zone_violation=round(f("zone_violation"), 3),
            avg_duration=round(m("episode_duration"), 1), avg_maneuvers=round(m("n_maneuvers"), 1),
            enemy_missiles_defeated=round(m("enemy_missiles_defeated"), 2),
            defensive_success=round(sum(r["enemy_missiles_defeated"] for r in w) / max(sum(r["enemy_shots"] for r in w), 1), 3),
        )
