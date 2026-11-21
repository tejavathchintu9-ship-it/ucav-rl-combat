"""
UCAVEnv - autonomous UCAV combat against a varied population of learned enemies (Gymnasium).

Design rules (the point of this file):
  * NO scripted pilot, FSM, maneuver library, launch gate or return-to-base autopilot exists here.
  * Both aircraft are flown through the SAME raw channels by learned policies:
        action = [elevator, aileron, rudder, throttle, fire]   (continuous, [-1, 1])
  * The opponent is any object with .act(obs) -> action (a frozen learned snapshot from the league,
    see combat/league.py).  Stage 1 has no opponent; the fallback "null stick" opponent applies no
    control at all (it is the absence of a controller, used only if no learned policy exists yet).
  * Firing is an ungated policy output.  There is no launch bonus, so launching is worth exactly
    what it produces (hit / miss / spent ammunition).
  * Zone boundary, ground, stall and mid-air collision are physical task outcomes, not behaviours.
  * Both sides see the world through the same egocentric observation function (_obs_for), so a
    policy can be dropped into either seat (self-play).

Physics and missiles are the unchanged simulator (simulator/updated_chase_jet_sim_v5.py) via
combat/combat_sim.py (proportional-navigation missiles, seeker gimbal limit, proximity fuse).
"""
from __future__ import annotations

import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

try:                                   # must precede Panda3D on Windows (DLL clash); harmless elsewhere
    import torch  # noqa: F401
except Exception:                      # pragma: no cover
    pass

from panda3d.core import loadPrcFileData

_MODE = os.environ.get("UCAV_RENDER_MODE", "none")
if _MODE == "none":
    loadPrcFileData("", "window-type none")
elif _MODE == "offscreen":
    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", "load-display p3tinydisplay")
    loadPrcFileData("", "win-size 1280 720")
loadPrcFileData("", "audio-library-name null")

import gymnasium as gym                # noqa: E402
from gymnasium import spaces           # noqa: E402
from panda3d.core import Vec3          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))
import combat.combat_sim as _cs        # noqa: E402
from combat.combat_sim import CombatSim, HOME, ZONE_RADIUS, MSL_LIFE   # noqa: E402
from combat.config import RANDOMIZATION_CONFIG as RC                   # noqa: E402
from combat.ucav_config import UCAV_CONFIG as DC                       # noqa: E402
from combat.enemy_pool import EnemyPool, NullOpponent                  # noqa: E402

base = _cs.base                        # the unchanged v5 simulator module
def apply_physics_overrides():
    """Apply ucav_config.MISSILE to the simulator's module constants (None = leave the simulator default)."""
    if DC["MISSILE"].get("a_max") is not None:
        _cs.MSL_A_MAX = float(DC["MISSILE"]["a_max"])
    if DC["MISSILE"].get("coast_decel") is not None:
        _cs.MSL_COAST_DECEL = float(DC["MISSILE"]["coast_decel"])
    if DC["MISSILE"].get("life") is not None:
        _cs.MSL_LIFE = float(DC["MISSILE"]["life"])      # the observation keeps its own fixed scale (MSL_LIFE imported above)


apply_physics_overrides()
OBS_DIM = 51


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def unit(v, fallback=None):
    n = v.length()
    if n < 1e-9:
        return Vec3(fallback) if fallback is not None else Vec3(0, 1, 0)
    return v / n


class UCAVEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 50}
    DT = DC["DT"]
    DECISION_STEPS = DC["DECISION_STEPS"]

    def __init__(self, stage: int = 1, seed: Optional[int] = None, enemy_dir: Optional[str] = None,
                 zone_radius: Optional[float] = None):
        super().__init__()
        self.stage = int(stage)
        self.ep_stage = int(stage)
        self.R_ = DC["REWARD"]
        self.gamma = DC["PPO"]["gamma"]
        zr = float(zone_radius if zone_radius is not None else DC["ZONE"]["RADIUS"])
        self.sim = CombatSim(zone_radius=zr, home=HOME)
        self.simulator = self.sim
        self.R = zr
        self.home = Vec3(HOME)
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        self.pool = EnemyPool(enemy_dir) if enemy_dir else None
        self.forced_opponent = None        # evaluation / demo can pin an opponent object
        self.forced_scenario = None
        self.episode_counter = 0
        self.episode_seed = None
        self.opponent = None
        self.opp_name, self.opp_kind = "none", "none"
        self.last_info = {}
        self._hist = deque(maxlen=int(DC["HOLD_TOLERANCE"]["last_seconds"] * 10))
        self.reset(seed=seed)

    # ------------------------------------------------------------------ control hooks
    def set_stage(self, stage: int):
        self.stage = int(max(1, min(4, stage)))

    def set_enemy_dir(self, enemy_dir: Optional[str]):
        self.pool = EnemyPool(enemy_dir) if enemy_dir else None

    def set_forced_opponent(self, opponent):
        self.forced_opponent = opponent

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _frame(a):
        fwd, up = a.get_forward_vector(), a.get_up_vector()
        return fwd, up, unit(fwd.cross(up), Vec3(1, 0, 0))

    @staticmethod
    def _aoa(a):
        v = a.velocity
        if v.length() < 1e-6:
            return 0.0
        v = v.normalized()
        f = a.get_forward_vector()
        ang = math.degrees(math.acos(clamp(v.dot(f), -1, 1)))
        return -ang if v.z < f.z else ang

    def _beta_gamma(self, a):
        v = a.velocity
        L = v.length()
        if L < 1e-6:
            return 0.0, 0.0
        vh = v / L
        _, _, right = self._frame(a)
        return (math.degrees(math.asin(clamp(vh.dot(right), -1, 1))),
                math.degrees(math.asin(clamp(vh.z, -1, 1))))

    def _msl(self, owner):
        for m in self.sim.missiles:
            if m["owner"] == owner:
                return m
        return None

    @staticmethod
    def _place(a, pos, speed, roll, pitch, yaw):
        a.position = Vec3(pos)
        a.roll, a.pitch, a.yaw = roll, pitch, yaw
        a.roll_rate = a.pitch_rate = a.yaw_rate = a.turn_rate = a.target_turn_rate = 0.0
        a.pitch_input = a.roll_input = a.yaw_input = a.speed_input = 0.0
        a.sync_graphics()
        a.velocity = a.get_forward_vector() * speed
        a.target_speed = a.current_speed = a.airspeed = float(speed)
        a.altitude = max(0.0, pos.z)
        a.g_force = 1.0
        a.engine_on = True
        a.crashed = False

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2 ** 31))
        self.episode_seed = int(seed)
        rng = np.random.default_rng(self.episode_seed)
        self.rng = rng
        # anti-forgetting: a share of episodes replays an EARLIER stage's task
        ep_stage = self.stage
        if self.stage > 1 and rng.random() < DC["REPLAY_PROB"]:
            ep_stage = int(rng.integers(1, self.stage))
        self.ep_stage = ep_stage
        st = DC["STAGES"][ep_stage]
        self.stage_cfg = st
        self.MAX_SECONDS = st["seconds"]
        self.rtb_required = bool(st.get("rtb", False)) and st["own_weapons"]
        sim = self.sim
        sim.reset_scene()
        p, e = sim.player, sim.enemy

        self.v_ref = self.h_ref = 0.0       # stage-1 hold references (0 = "not provided")
        self.red_v_ref = self.red_h_ref = 0.0
        self.scenario = "flight" if not st["enemy"] else "n/a"
        self.opponent, self.opp_name, self.opp_kind = None, "none", "none"

        if not st["enemy"]:
            self._spawn_hold(rng, st)
        else:
            self._spawn_engagement(rng, st)
            self.opponent, self.opp_name, self.opp_kind = self._pick_opponent(rng, st)
        self.opp_active = self.opponent is not None
        self.foe_present = self.opp_active

        # stage 1 is a flight-only task: no weapons and nothing to shoot at
        self.ammo_blue = DC["AMMO"] if (self.foe_present and st["own_weapons"]) else 0
        self.ammo_red = st["enemy_ammo"] if (self.foe_present and st["enemy_weapons"]) else 0
        self.blocked_blue = self.blocked_red = 0
        self.adv_steps = 0
        hs = DC["ENEMY_VARIATION"]["hold_steps"]
        self.enemy_hold = int(rng.integers(hs[0], hs[1] + 1))
        self._red_counter = 0
        self._lock = {"blue": 0.0, "red": 0.0}
        self.shots_blue = self.shots_red = 0
        self.red_missiles_missed = 0
        self.blue_missiles_missed = 0
        self.killed = False
        self.kill_t = None
        self.t = 0.0
        self.step_count = 0
        self.n_decisions = 0
        self.ended = False
        self.end_reason = None
        self.result = None
        self.valid = True
        self.success = False
        self.failure_reason = None
        self.min_range = 1e9
        self.zone_max_frac = 0.0
        self.ep_return = 0.0
        self.reward_parts = {}
        self._a_blue = np.zeros(4, np.float32)
        self._a_red = np.zeros(4, np.float32)
        self._prev_a = np.zeros(4, np.float32)
        self._prev_a_red = np.zeros(4, np.float32)
        self._hist.clear()
        self.closure = 0.0
        self.prev_range = self._range()
        self.prev_zfrac = (p.position - self.home).length() / self.R
        self._zone_pot_prev = self._zone_potential(self.prev_zfrac)
        self._phi_prev = self._potential(p, e) if self.foe_present else 0.0
        self._defense_prev = 0.0
        obs = self._obs_for("blue")
        info = self._info(0.0, {})
        return obs, info

    def _spawn_hold(self, rng, st):
        sim = self.sim
        p, e = sim.player, sim.enemy
        self.v_ref = float(rng.uniform(*st["v_target"]))
        self.h_ref = float(rng.uniform(*st["alt_target"]))
        alt = float(np.clip(self.h_ref + rng.uniform(-300, 300), 700, 2600))
        spd = float(np.clip(self.v_ref + rng.uniform(-1, 1) * st["init_speed_dev"], 150, 380))
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(*st.get("start_radius", (0.0, 1500.0)))
        pos = Vec3(self.home.x + r * math.cos(a), self.home.y + r * math.sin(a), alt)
        self._place(p, pos, spd, float(rng.uniform(-1, 1) * st["init_roll"]),
                    float(rng.uniform(-1, 1) * st["init_pitch"]), float(rng.uniform(0, 360)))
        self._place(e, Vec3(self.home.x + 9000, self.home.y, 1500.0), 150.0, 0.0, 0.0, 0.0)
        e.crashed = True                   # parked, harmless, never in range
        sim.enemy_destroyed = True         # presentation only: hides its bracket / radar dot in the demo
        e.root.hide()

    def _spawn_engagement(self, rng, st):
        sim = self.sim
        p, e = sim.player, sim.enemy
        geoms = RC["GEOMETRIES"]
        names = [n for n in geoms if st.get("scenarios") is None or n in st["scenarios"]]
        probs = np.asarray([geoms[n][0] for n in names], float)
        probs /= probs.sum()
        scenario = self.forced_scenario or str(rng.choice(names, p=probs))
        self.scenario = scenario
        _, drange, brange, jitter = geoms[scenario]
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(*RC["UCAV_START_RADIUS"])
        upos = Vec3(self.home.x + r * math.cos(a), self.home.y + r * math.sin(a),
                    float(rng.uniform(*RC["INITIAL_ALTITUDE_RANGE"])))
        uyaw = float(rng.uniform(*RC["INITIAL_HEADING_RANGE"]))
        uspd = float(rng.uniform(*RC["UCAV_SPEED_RANGE"]))
        fwd_h = Vec3(-math.sin(math.radians(uyaw)), math.cos(math.radians(uyaw)), 0)

        def rot(v, deg):
            c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            return Vec3(v.x * c - v.y * s, v.x * s + v.y * c, 0)

        sign = rng.choice([-1, 1]) if scenario == "beam" else 1
        bearing = float(rng.uniform(*brange) * sign)
        off, dist = rot(fwd_h, bearing), float(rng.uniform(*drange)) * float(DC.get("START_DISTANCE_SCALE", 1.0))
        if scenario == "enemy_on_six":
            eyaw = math.degrees(math.atan2(off.x, -off.y))      # nose pointing at the UCAV
        elif scenario == "head_on":
            eyaw = uyaw + 180 + rng.uniform(-jitter, jitter)
        elif scenario == "tail_chase":
            eyaw = uyaw + rng.uniform(-jitter, jitter)
        else:
            eyaw = float(rng.uniform(0, 360))
        ealt_rng = st.get("opp_alt", RC["ENEMY_ALTITUDE_RANGE"])
        espd = float(rng.uniform(*st.get("opp_speed", RC["ENEMY_SPEED_RANGE"])))
        epos = Vec3(upos.x + off.x * dist, upos.y + off.y * dist, float(rng.uniform(*ealt_rng)))
        rel_h = Vec3(epos.x - self.home.x, epos.y - self.home.y, 0)
        lim = RC["ENEMY_MAX_START_ZONE_FRAC"] * self.R
        if rel_h.length() > lim:
            k = lim / rel_h.length()
            epos = Vec3(self.home.x + rel_h.x * k, self.home.y + rel_h.y * k, epos.z)
        if st.get("safe_enemy_heading"):
            # an enemy must not leave the zone within the horizon; this only picks its start
            # heading (initial-state randomisation), it never controls the aircraft afterwards
            for _ in range(24):
                d = Vec3(-math.sin(math.radians(eyaw)), math.cos(math.radians(eyaw)), 0)
                end = Vec3(epos.x + d.x * espd * self.MAX_SECONDS, epos.y + d.y * espd * self.MAX_SECONDS, 0)
                if Vec3(end.x - self.home.x, end.y - self.home.y, 0).length() < 0.92 * self.R:
                    break
                to_home = Vec3(self.home.x - epos.x, self.home.y - epos.y, 0)
                eyaw = math.degrees(math.atan2(-to_home.x, to_home.y)) + float(rng.uniform(-50, 50))
        self._place(p, upos, uspd, float(rng.uniform(-20, 20)), float(rng.uniform(-6, 6)), uyaw)
        self._place(e, epos, espd, 0.0, 0.0, float(eyaw))
        self.red_v_ref, self.red_h_ref = espd, float(epos.z)
        self.init_range = (epos - upos).length()
        self.init_bearing = bearing

    def _pick_opponent(self, rng, st):
        if self.forced_opponent is not None:
            fo = self.forced_opponent
            return fo, getattr(fo, "name", "forced"), getattr(fo, "kind", "forced")
        opp = self.pool.sample(rng, st, DC["ENEMY_VARIATION"]["stochastic_prob"]) if self.pool else None
        if opp is None:
            n = NullOpponent()
            return n, n.name, n.kind
        return opp, opp.name, opp.kind

    # ------------------------------------------------------------------ observation
    def _range(self):
        return (self.sim.enemy.position - self.sim.player.position).length()

    def _obs_for(self, side: str):
        """Egocentric observation for one seat.  Used verbatim for blue (learner) and red (opponent)."""
        sim = self.sim
        if side == "blue":
            me, foe, my_o, foe_o = sim.player, sim.enemy, "ucav", "enemy"
            ammo, last = self.ammo_blue, self._prev_a
            v_ref, h_ref = self.v_ref, self.h_ref
            foe_alive = self.foe_present and not foe.crashed
            killed = self.killed
        else:
            me, foe, my_o, foe_o = sim.enemy, sim.player, "enemy", "ucav"
            ammo, last = self.ammo_red, self._prev_a_red
            v_ref, h_ref = self.red_v_ref, self.red_h_ref
            foe_alive = not foe.crashed
            killed = False
        c = lambda x: float(clamp(x, -1.0, 1.0))
        fwd, up, right = self._frame(me)
        beta, gamma = self._beta_gamma(me)
        zrel = me.position - self.home
        zr = zrel.length()
        out_v = (zrel.dot(me.velocity) / max(zr, 1.0)) / 300.0
        hd = unit(self.home - me.position)
        t_left = 1.0 - min(1.0, self.t / max(self.MAX_SECONDS, 1.0))
        own = [
            me.current_speed / 800.0, (me.altitude - 1800.0) / 1800.0, me.pitch / 90.0, me.roll / 90.0,
            self._aoa(me) / 20.0, beta / 30.0, (me.g_force - 1.0) / 5.5,
            me.roll_rate / base.ROLL_SENSITIVITY, me.pitch_rate / base.PITCH_SENSITIVITY,
            me.yaw_rate / base.YAW_SENSITIVITY, gamma / 45.0, (me.target_speed - me.current_speed) / 100.0,
            (v_ref - me.current_speed) / 200.0 if v_ref else 0.0,
            (h_ref - me.altitude) / 1000.0 if h_ref else 0.0,
        ]
        if foe_alive:
            rel = foe.position - me.position
            R = max(rel.length(), 1e-6)
            d = rel / R
            vrel = foe.velocity - me.velocity
            closure = -rel.dot(vrel) / R
            om = rel.cross(vrel) / (R * R)                       # LOS angular rate (rad/s)
            ff = foe.get_forward_vector()
            foe_f = [
                d.dot(right), d.dot(fwd), d.dot(up),
                R / 10000.0, min(R, 3000.0) / 3000.0, closure / 300.0, foe.current_speed / 800.0,
                (foe.position.z - me.position.z) / 2000.0,
                ff.dot(right), ff.dot(fwd), ff.dot(up),
                clamp((-d).dot(ff), -1, 1),
                om.dot(right) / 0.3, om.dot(fwd) / 0.3, om.dot(up) / 0.3,
            ]
        else:
            foe_f = [0.0] * 15
        zone = [zr / self.R, out_v, hd.dot(right), hd.dot(fwd), hd.dot(up)]
        m_own = self._msl(my_o)
        weap = [
            ammo / max(DC["AMMO"], 1),
            1.0 if m_own is not None else 0.0,
            (m_own["age"] / MSL_LIFE) if m_own is not None else 0.0,
            (min((m_own["target"].position - m_own["pos"]).length() / 5000.0, 1.0)) if m_own is not None else 1.0,
        ]
        m_in = self._msl(foe_o)
        if m_in is not None:
            mr = m_in["pos"] - me.position
            mR = max(mr.length(), 1e-6)
            md = mr / mR
            mclos = -(mr.dot(m_in["vel"] - me.velocity)) / mR
            thr = [1.0, mR / 6000.0, min(mR / max(mclos, 50.0), 20.0) / 20.0,
                   md.dot(right), md.dot(fwd), md.dot(up)]
        else:
            thr = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        misc = [t_left, 1.0 if foe_alive else 0.0, 1.0 if killed else 0.0]
        arr = np.asarray(own + [float(x) for x in last] + foe_f + zone + weap + thr + misc, np.float32)
        if arr.shape != (OBS_DIM,):
            raise RuntimeError(f"observation has {arr.shape}, expected ({OBS_DIM},)")
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0).astype(np.float32)

    def _obs(self):
        return self._obs_for("blue")

    # ------------------------------------------------------------------ physics interface
    def _apply_stick(self, actor, a, dt):
        s, (lo, hi) = DC["STICK"], DC["SPEED_LIMITS"]
        actor.pitch_input = float(a[0]) * s["pitch"]
        actor.roll_input = float(a[1]) * s["roll"]
        actor.yaw_input = float(a[2]) * s["yaw"]
        actor.speed_input = 0.0            # the sim's own throttle is a 3-state sign; use the magnitude here
        actor.target_speed = clamp(actor.target_speed + float(a[3]) * s["throttle_rate"] * dt, lo, hi)

    def _opponent_action(self):
        a = np.asarray(self.opponent.act(self._obs_for("red"), self.rng), np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        if a.size != 5:
            raise ValueError(f"opponent returned {a.size} actions, expected 5")
        return a

    # ------------------------------------------------------------------ step
    def step(self, action, frame_callback=None):
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.size != 5:
            raise ValueError(f"expected 5 actions, got {raw.size}")
        a = np.clip(raw, -1.0, 1.0)
        self._a_blue = a[:4].copy()
        fire_b = bool(a[4] > DC["FIRE_THRESHOLD"])
        fire_r = False
        if self.opp_active and not self.ended:
            if self._red_counter % self.enemy_hold == 0:      # enemy reaction delay: re-decides every few steps
                ar = self._opponent_action()
                self._a_red = ar[:4].copy()
                fire_r = bool(ar[4] > DC["FIRE_THRESHOLD"])
            self._red_counter += 1
        ev = {k: False for k in ("launch", "enemy_launch", "own_hit", "own_miss", "ucav_hit", "enemy_miss",
                                 "blocked", "enemy_blocked")}
        for k in range(self.DECISION_STEPS):
            self._substep(fire_b and k == 0, fire_r and k == 0, ev)
            if frame_callback is not None and frame_callback(self) is False:
                break
            if self.ended:
                break
        self._prev_a_red = self._a_red.copy()
        R = self._range() if self.foe_present else self.prev_range
        self.closure = (self.prev_range - R) / (self.DT * self.DECISION_STEPS)
        truncated = False
        if not self.ended and self.t >= self.MAX_SECONDS and not (self.killed and self.rtb_required):
            truncated = True
            self._timeout_result()
        reward = self._reward(ev, a, truncated)
        self.prev_range = R
        self._prev_a = a[:4].copy()
        self.n_decisions += 1
        self.ep_return += reward
        terminated = self.ended
        obs = self._obs_for("blue")
        info = self._info(reward, ev)
        if terminated or truncated:
            self.episode_counter += 1
            info["episode_record"] = self._episode_record(info)
        return obs, float(reward), terminated, truncated, info

    def _timeout_result(self):
        s = self.ep_stage
        self.end_reason = "timeout"
        if s == 1:
            self.end_reason = "hold_complete"
            self.success = self._hold_ok()
            self.result = "hold_ok" if self.success else "hold_fail"
            if not self.success:
                self.failure_reason = "HOLD_ERROR"
        elif s == 2:
            frac = self.adv_steps / max(self.n_decisions, 1)
            self.success = frac >= DC["ADVANTAGE"]["success_frac"]
            self.result = "pursuit_ok" if self.success else "pursuit_fail"
            if not self.success:
                self.failure_reason = "NOT_CLOSE"
        elif s == 3:
            self.success = True
            self.result = "survived"
        else:
            self.success = False
            self.result = "no_hit"
            self.failure_reason = "TIMEOUT"

    def _in_envelope(self, shooter, target):
        """Instantaneous launch conditions: range window, seeker cone, and (optional) the shooter's own flight limits."""
        env = DC["ENVELOPE"]
        rel = target.position - shooter.position
        R = rel.length()
        if R < 1.0:
            return False
        cos = clamp(unit(rel).dot(unit(shooter.velocity, shooter.get_forward_vector())), -1, 1)
        if not (env["range"][0] <= R <= env["range"][1] and cos >= env["cos_ata_min"]):
            return False
        if env.get("aoa_max") is not None and abs(self._aoa(shooter)) > env["aoa_max"]:
            return False
        if env.get("g_max") is not None and shooter.g_force > env["g_max"]:
            return False
        if env.get("bank_max") is not None and abs(shooter.roll) > env["bank_max"]:
            return False
        return True

    def _envelope_ok(self, shooter, target, side):
        """Physical weapon envelope.  Same rule for UCAV and enemy.  `lock_time` = seconds the conditions must hold continuously."""
        env = DC["ENVELOPE"]
        if not env["INTERLOCK"]:
            return True
        return self._in_envelope(shooter, target) and self._lock[side] >= env.get("lock_time", 0.0)

    def _substep(self, fire_b, fire_r, ev, tail=False):
        sim, dt = self.sim, self.DT
        p, e = sim.player, sim.enemy
        if not p.crashed:
            self._apply_stick(p, self._a_blue, dt)
        if self.opp_active and not e.crashed:
            self._apply_stick(e, self._a_red, dt)
        if not tail:
            # seeker acquisition timers: continuous time the launch conditions have held, for each side
            self._lock["blue"] = (self._lock["blue"] + dt) if (self.foe_present and not e.crashed and not p.crashed
                                                               and self._in_envelope(p, e)) else 0.0
            self._lock["red"] = (self._lock["red"] + dt) if (self.opp_active and not e.crashed and not p.crashed
                                                             and self._in_envelope(e, p)) else 0.0
            if (fire_b and not p.crashed and self.ammo_blue > 0 and self._msl("ucav") is None
                    and self.foe_present and not e.crashed):
                if self._envelope_ok(p, e, "blue"):
                    sim.launch_missile("ucav")
                    self.ammo_blue -= 1
                    self.shots_blue += 1
                    ev["launch"] = True
                else:
                    self.blocked_blue += 1
                    ev["blocked"] = True
            if (fire_r and self.opp_active and not e.crashed and not p.crashed and self.ammo_red > 0
                    and self._msl("enemy") is None):
                if self._envelope_ok(e, p, "red"):
                    sim.launch_missile("enemy")
                    self.ammo_red -= 1
                    self.shots_red += 1
                    ev["enemy_launch"] = True
                else:
                    self.blocked_red += 1
                    ev["enemy_blocked"] = True
        if not p.crashed:
            sim.update_actor_physics(p, dt)
        if not e.crashed:
            sim.update_actor_physics(e, dt)
        sim.update_missiles(dt)
        self.t += dt
        self.step_count += 1

        killed_now = blue_dead = False
        for evt in sim.events:
            if evt["owner"] == "ucav":
                if evt["type"] == "hit":
                    ev["own_hit"] = True
                    killed_now = True
                else:
                    ev["own_miss"] = True
                    self.blue_missiles_missed += 1
            else:
                if evt["type"] == "hit":
                    ev["ucav_hit"] = True
                    blue_dead = True
                else:
                    ev["enemy_miss"] = True
                    self.red_missiles_missed += 1
        sim.events = []
        if tail or self.ended:
            return
        if killed_now:
            self.killed = True
            self.kill_t = self.t
        if blue_dead and killed_now:
            self._finish("mutual_kill", "mutual_hit")
        elif blue_dead:
            self._finish("shot_down", "ucav_hit")
        elif killed_now and not self.rtb_required:
            self._finish("target_hit", "target_hit", success=True)
        if self.ended:
            return

        zr = (p.position - self.home).length()
        self.zone_max_frac = max(self.zone_max_frac, zr / self.R)
        if self.foe_present and not e.crashed:
            R = (e.position - p.position).length()
            self.min_range = min(self.min_range, R)
        else:
            R = 1e9
        if zr > self.R:
            self._finish("zone_exit", "ucav_failed")
        elif p.position.z < 25.0:
            self._finish("ground_collision", "ucav_failed")
        elif p.current_speed < 70.0:
            self._finish("stall", "ucav_failed")
        elif R < 30.0:
            self._finish("midair_collision", "ucav_failed")
        elif self.opp_active and not e.crashed:
            ze = (e.position - self.home).length()
            if ze > self.R:
                self._finish("opp_zone_exit", "enemy_out", valid=False)
            elif e.position.z < 25.0:
                self._finish("opp_ground_collision", "enemy_out", valid=False)
            elif e.current_speed < 70.0:
                self._finish("opp_stall", "enemy_out", valid=False)
        if self.ended:
            return
        st = self.stage_cfg
        if self.killed and self.rtb_required:
            since = self.t - self.kill_t
            if since >= DC["ZONE"]["RTB_MIN_SECONDS"] and zr / self.R <= DC["ZONE"]["RTB_COMPLETE_FRACTION"]:
                self._finish("mission_complete", "target_hit", success=True)
            elif since >= DC["ZONE"]["RTB_TIMEOUT_SECONDS"]:
                self._finish("hit_but_not_returned", "target_hit_no_return")
        elif (st["own_weapons"] and not self.killed and self.ammo_blue == 0 and self._msl("ucav") is None):
            self._finish("out_of_missiles", "no_hit")
        elif (self.ep_stage == 3 and self.shots_red > 0 and self.ammo_red == 0 and self._msl("enemy") is None):
            self._finish("enemy_missiles_spent", "survived", success=True)

    def _finish(self, reason, result, success=False, valid=True):
        self.ended = True
        self.end_reason = reason
        self.result = result
        self.success = bool(success)
        self.valid = bool(valid)
        if not self.success:
            self.failure_reason = {
                "zone_exit": "ZONE_VIOLATION", "shot_down": "UCAV_LOST", "ground_collision": "UCAV_CRASH",
                "midair_collision": "UCAV_CRASH", "stall": "FLIGHT_ENVELOPE", "mutual_kill": "MUTUAL_HIT",
                "hit_but_not_returned": "DID_NOT_RETURN", "out_of_missiles": "OUT_OF_MISSILES",
            }.get(reason, "ENEMY_OUT" if reason.startswith("opp_") else reason.upper())

    def _hold_ok(self):
        t = DC["HOLD_TOLERANCE"]
        if len(self._hist) < self._hist.maxlen * 0.9:
            return False
        h = np.asarray(self._hist, np.float32)
        return bool(h[:, 0].mean() < t["roll"] and h[:, 1].mean() < t["speed"] and h[:, 2].mean() < t["alt"])

    # ------------------------------------------------------------------ reward shaping pieces
    @staticmethod
    def _zone_potential(zfrac):
        w = DC["ZONE"]["WARN"]
        risk = max(0.0, (zfrac - w) / (1.0 - w))
        return -(risk ** 2)

    def _potential(self, me, foe):
        """Phi(s) = K * f_D  (Li et al. 2022, Eq. 13-17, 23), from `me`'s point of view."""
        c = self.R_
        rel = foe.position - me.position
        R = rel.length()
        if R < 1.0:
            return 0.0
        rhat = rel / R
        vu, vt = unit(me.velocity, me.get_forward_vector()), unit(foe.velocity, foe.get_forward_vector())
        phi_u = math.acos(clamp(vu.dot(rhat), -1, 1))
        phi_t = math.acos(clamp(vt.dot(rhat), -1, 1))
        f_phi = (2 * math.pi - phi_u - phi_t) / (2 * math.pi)
        Rg = c["Rg"]
        f_R = 1.0 if R <= Rg else math.exp(-((R - Rg) ** 2) / (2 * c["sigma_R"] ** 2))
        VB = foe.current_speed
        Vs = VB if R <= Rg else VB + (c["Vmax"] - VB) * (1 - math.exp((Rg - R) / Rg))
        V = me.current_speed
        f_V = (V / max(Vs, 1.0)) * math.exp(-2 * abs(V - Vs) / max(Vs, 1.0))
        dH = me.position.z - foe.position.z
        if c["hz"] <= dH <= c["hz"] + c["sigma_h"]:
            f_h = 1.0
        elif dH < c["hz"]:
            f_h = math.exp(-((dH - c["hz"]) ** 2) / (2 * c["sigma_h"] ** 2))
        else:
            f_h = math.exp(-((dH - c["hz"] - c["sigma_h"]) ** 2) / (2 * c["sigma_h"] ** 2))
        fD = 2.0 * (c["w_phi"] * f_phi + c["w_R"] * f_R + c["w_V"] * f_V + c["w_h"] * f_h) - 1.0
        return c["phi_scale"] * fD

    def _defense_potential(self):
        m = self._msl("enemy")
        if m is None:
            return 0.0
        p = self.sim.player
        delta = m["pos"] - p.position
        dist = max(delta.length(), 1.0)
        closing = -(delta.dot(m["vel"] - p.velocity)) / dist
        return -math.exp(-max(dist / max(closing, 50.0), 0.0) / 4.0)

    # ------------------------------------------------------------------ reward
    def _reward(self, ev, a, truncated):
        c = self.R_
        p, e = self.sim.player, self.sim.enemy
        parts = {}

        def add(k, v):
            if v:
                parts[k] = parts.get(k, 0.0) + v

        s = self.ep_stage
        fq = 1.0 if s == 1 else 0.3          # agility is the task once the enemy appears
        beta, gamma = self._beta_gamma(p)
        aoa = abs(self._aoa(p))
        add("time", c["time"])
        rr = ((p.roll_rate / base.ROLL_SENSITIVITY) ** 2 + (p.pitch_rate / base.PITCH_SENSITIVITY) ** 2
              + (p.yaw_rate / base.YAW_SENSITIVITY) ** 2) / 3.0
        add("rate", fq * c["rate"] * rr)
        add("smooth", fq * c["smooth"] * float(np.sum((a[:4] - self._prev_a) ** 2)) / 4.0)
        add("effort", c["effort"] * float(np.sum(a[:4] ** 2)))
        add("beta", c["beta"] * min((abs(beta) / 10.0) ** 2, 4.0))
        if abs(beta) > c["beta_hard_deg"]:
            add("beta_hard", c["beta_hard"])
        if aoa > c["aoa_free_deg"]:
            add("aoa", c["aoa"] * (aoa - c["aoa_free_deg"]) / 5.0)
        if aoa > c["aoa_hard_deg"]:
            add("aoa_hard", c["aoa_hard"])
        if p.g_force > c["g_free"]:
            add("g", c["g"] * (p.g_force - c["g_free"]) / 3.0)
        if p.altitude < c["low_alt_m"]:
            add("low_alt", c["low_alt"] * (c["low_alt_m"] - p.altitude) / c["low_alt_m"])
        if p.current_speed < c["low_speed_ms"]:
            add("low_speed", c["low_speed"] * (c["low_speed_ms"] - p.current_speed) / c["low_speed_ms"])

        zfrac = (p.position - self.home).length() / self.R
        if s == 1:
            dv, dh = abs(p.current_speed - self.v_ref), abs(p.altitude - self.h_ref)
            self._hist.append((abs(p.roll), dv, dh))
            add("hold_roll", c["hold_roll"] / (1.0 + abs(p.roll) / c["hold_roll_deg"]))
            add("hold_roll_error", -c["hold_roll_error_penalty"] * math.sqrt(abs(p.roll) / c["hold_roll_error_scale"]))
            add("hold_speed", c["hold_speed"] / (1.0 + dv / 100.0))
            add("hold_alt", c["hold_alt"] / (1.0 + dh / 1000.0) + c["hold_alt_wide"] / (1.0 + dh / 3000.0))
            add("hold_speed_error", -c["hold_speed_error_penalty"] * math.sqrt(dv / c["hold_speed_error_scale"]))
            add("hold_alt_error", -c["hold_alt_error_penalty"] * math.sqrt(dh / c["hold_alt_error_scale"]))
            g_star = clamp((self.h_ref - p.altitude) / c["hold_gamma_gain_m"],
                           -c["hold_gamma_max_deg"], c["hold_gamma_max_deg"])
            add("hold_gamma", c["hold_gamma"] * math.exp(-((gamma - g_star) / c["hold_gamma_deg"]) ** 2))
        else:
            alive_foe = self.foe_present and not e.crashed
            rebal = bool(DC["DEFENCE_REBALANCE"])
            phi_mult = c["phi_mult_defence"] if (rebal and s >= 3) else 1.0
            phi_new = phi_mult * self._potential(p, e) if (alive_foe and not self.ended) else 0.0
            add("pursuit_potential", self.gamma * phi_new - self._phi_prev)
            self._phi_prev = phi_new
            if alive_foe:
                rel = e.position - p.position
                R = rel.length()
                cos_ata = clamp(unit(rel).dot(p.get_forward_vector()), -1, 1)
                cos_e = clamp(unit(-rel).dot(e.get_forward_vector()), -1, 1)
                adv = DC["ADVANTAGE"]
                bonus_scale = ((c["pursuit_scale_defence"] if s >= 3 else 1.0) * (0.0 if self._msl("enemy") is not None else 1.0)) if rebal else 1.0
                if s == 4:
                    bonus_scale *= c.get("pursuit_scale_combat", 1.0)
                if adv["success_range"][0] <= R <= adv["success_range"][1] and cos_ata >= adv["success_cos"]:
                    self.adv_steps += 1                      # stage-2 success measure: close and pointing at the enemy
                # graded version of the same target gives a gradient toward it (rewarding only the exact
                # condition is almost never discovered by a policy that can merely fly straight)
                lo_r, hi_r = adv["range"]
                g_r = 1.0 if lo_r <= R <= hi_r else math.exp(-(((lo_r - R) if R < lo_r else (R - hi_r)) / 2500.0) ** 2)
                g_a = clamp((cos_ata - 0.2) / (adv["cos_ata_min"] - 0.2), 0.0, 1.0)
                g_s = clamp((-cos_e + 0.1) / (-adv["cos_enemy_max"] + 0.1), 0.0, 1.0)
                add("adv_bonus", bonus_scale * c["adv_bonus"] * g_r * g_a * g_s)       # rear-cone shaping (secondary)
                # primary stage-2 reward = the success measure itself: close to the enemy and pointing at it
                slo, shi = adv["success_range"]
                p_r = 1.0 if slo <= R <= shi else math.exp(-(((slo - R) if R < slo else (R - shi)) / 2500.0) ** 2)
                p_a = clamp((cos_ata - 0.2) / (adv["success_cos"] - 0.2), 0.0, 1.0)
                add("prox_bonus", bonus_scale * c["prox_bonus"] * p_r * p_a)
                if R < c["overshoot_range"]:
                    add("overshoot", c["overshoot"] * (1.0 - R / c["overshoot_range"]))
                if R < c["closure_range"]:       # manage closure: arriving fast means flying past the target
                    add("closure_control", c["closure_penalty"] * clamp((self.closure - c["closure_free"]) / 150.0, 0.0, 1.0))
                lo, hi = c["wez_range"]
                if lo <= R <= hi and cos_ata >= c["wez_cos"]:
                    add("wez_bonus", bonus_scale * c["wez_bonus"])
                if self.opp_active and self.stage_cfg["enemy_weapons"] and lo <= R <= hi and cos_e >= c["wez_cos"]:
                    add("wez_threat", c["wez_threat"])
            if ev["launch"] and c.get("launch_cost", 0.0):
                add("launch_cost", c["launch_cost"])
            if ev["own_hit"]:
                add("hit", c["hit"])
                if c.get("unused_missile_bonus", 0.0):
                    add("unused_missiles", c["unused_missile_bonus"] * max(self.ammo_blue, 0))
            if ev["own_miss"]:
                add("miss", c["miss"])
            if ev["blocked"]:
                add("blocked_request", c["blocked_request"])
            if self._msl("enemy") is not None:
                add("missile_inbound", c["missile_inbound"])
            dphi = self._defense_potential()
            if ev["ucav_hit"]:
                dphi = -1.0
            add("defense_progress", c["defense_progress"] * (dphi - self._defense_prev))
            self._defense_prev = dphi
            if ev["enemy_miss"]:
                add("defended", c["enemy_missile_defeated"])
            if ev["ucav_hit"]:
                add("shot_down", c["shot_down"])
            if self.killed and self.rtb_required:
                add("rtb_progress", c["rtb_progress"] * (self.prev_zfrac - zfrac) * (1.0 if zfrac > 0.3 else 0.0))
        self.prev_zfrac = zfrac
        warn = DC["ZONE"]["WARN"]
        if zfrac > warn:
            add("zone_warn", c["zone_warn"] * ((zfrac - warn) / (1.0 - warn)) ** 2)
        zp = self._zone_potential(zfrac)
        add("zone_safety_progress", c["zone_safety_potential"] * (self.gamma * zp - self._zone_pot_prev))
        self._zone_pot_prev = zp
        er = self.end_reason
        if er == "zone_exit":
            add("zone_exit", c["zone_exit"])
        elif er in ("ground_collision", "midair_collision"):
            add("crash", c["crash"])
        elif er == "stall":
            add("stall", c["stall"])
        elif er == "mission_complete":
            add("mission_complete", c["mission_complete"])
        elif er == "hit_but_not_returned":
            add("no_return", c["no_return"])
        elif er == "out_of_missiles":
            add("out_of_missiles", c["out_of_missiles"])
        elif er in ("timeout", "enemy_missiles_spent"):
            if s == 3:
                add("survived", c["survived"])
            elif s == 4:
                add("timeout", c["timeout"])
        for k, v in parts.items():
            self.reward_parts[k] = self.reward_parts.get(k, 0.0) + v
        return float(sum(parts.values()))

    # ------------------------------------------------------------------ info / records
    def _info(self, reward, ev):
        p, e = self.sim.player, self.sim.enemy
        if self.foe_present and not e.crashed:
            rel = e.position - p.position
            R = rel.length()
            cos_ata = clamp(unit(rel).dot(p.get_forward_vector()), -1, 1)
            cos_e = clamp(unit(-rel).dot(e.get_forward_vector()), -1, 1)
        else:
            R, cos_ata, cos_e = float("inf"), 0.0, 0.0
        info = dict(
            distance=R, cos_ata=cos_ata, cos_enemy_ata=cos_e, closure_rate=self.closure,
            airspeed=p.current_speed, altitude=p.altitude, aoa_deg=self._aoa(p), g_force=p.g_force,
            roll=p.roll, pitch=p.pitch, heading=p.yaw, zone_frac=(p.position - self.home).length() / self.R,
            enemy_speed=e.current_speed, missiles=self.ammo_blue, enemy_missiles=self.ammo_red,
            enemy_missile_inbound=self._msl("enemy") is not None, own_missile_active=self._msl("ucav") is not None,
            time=self.t, stage=self.stage, episode_stage=self.ep_stage, scenario=self.scenario, seed=self.episode_seed,
            enemy=self.opp_name, enemy_kind=self.opp_kind,
            success=self.success, result=self.result, termination_reason=self.end_reason,
            failure_reason=self.failure_reason, shots=self.shots_blue, enemy_shots=self.shots_red,
            blocked=self.blocked_blue, killed=self.killed, min_range=self.min_range,
            zone_max_frac=self.zone_max_frac, reward_parts=dict(self.reward_parts),
            n_decisions=self.n_decisions, events=ev, v_ref=self.v_ref, h_ref=self.h_ref,
            advantage_frac=self.adv_steps / max(self.n_decisions, 1),
        )
        self.last_info = info
        return info

    def _episode_record(self, info):
        """Keys follow combat.metrics.MetricsTracker so existing logging keeps working."""
        reason = self.end_reason
        self_fail = reason in ("ground_collision", "midair_collision", "stall", "zone_exit", "shot_down")
        return dict(
            episode_id=self.episode_counter, episode_seed=self.episode_seed, stage=self.ep_stage, level=self.ep_stage,
            replay=self.ep_stage != self.stage, scenario=self.scenario, enemy=self.opp_name, enemy_kind=self.opp_kind,
            result=self.result, success=bool(self.success), mission_success=bool(self.success),
            valid=bool(getattr(self, "valid", True)), failure_reason=self.failure_reason, termination=reason,
            missile_hit=bool(self.killed), launch_attempts=self.shots_blue, launch_success=self.shots_blue > 0,
            missile_miss=self.blue_missiles_missed > 0, blocked_launches=self.blocked_blue,
            enemy_blocked=self.blocked_red, enemy_shots=self.shots_red,
            enemy_missiles_defeated=self.red_missiles_missed,
            advantage_frac=round(self.adv_steps / max(self.n_decisions, 1), 3),
            zone_violation=reason == "zone_exit", zone_max_frac=round(self.zone_max_frac, 3),
            ucav_survival=not self_fail, episode_reward=round(self.ep_return, 2),
            episode_duration=round(self.t, 1), n_maneuvers=self.n_decisions,
            minimum_range=round(self.min_range, 1) if self.min_range < 1e8 else None,
            initial_range=round(getattr(self, "init_range", 0.0), 1),
            reward_parts={k: round(v, 2) for k, v in self.reward_parts.items()},
        )


    # ------------------------------------------------------------------ demo helpers
    def tail(self, seconds, frame_callback=None):
        """Keep the world running after the episode ended (explosions, fly-out) - visual demo only."""
        was = self.ended
        ev = {k: False for k in ("launch", "enemy_launch", "own_hit", "own_miss", "ucav_hit", "enemy_miss", "blocked", "enemy_blocked")}
        for _ in range(int(seconds / self.DT)):
            self._substep(False, False, ev, tail=True)
            if frame_callback is not None and frame_callback(self) is False:
                break
        self.ended = was

    def close(self):
        try:
            self.sim.destroy()
        except Exception:
            pass


def make_env(**kw):
    return UCAVEnv(**kw)
