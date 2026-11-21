"""
Tests for combat/ucav_env.py.  Run from the 'ucav rl' directory:   python -m pytest tests/test_ucav_env.py -q
They pin down: nothing scripted is used, weapons exist only where the stage says so, the weapon envelope blocks
physically impossible launches, results are hit-based, and the enemy moves only by its policy's output.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("UCAV_RENDER_MODE", "none")
# The tests check the DEFAULT settings.  Preset variables left over in the shell (UCAV_REWARD=economy, UCAV_COMBAT=bvr, ...)
# would change them, so clear them before the project is imported.
for _k in ("UCAV_COMBAT", "UCAV_REWARD", "UCAV_ENVELOPE", "UCAV_COAST_DECEL", "UCAV_START_SCALE"):
    os.environ.pop(_k, None)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

import numpy as np
import pytest
from panda3d.core import Vec3

from combat.ucav_config import UCAV_CONFIG as DC
from combat.ucav_env import UCAVEnv, OBS_DIM
from combat.enemy_pool import NullOpponent

DC["REPLAY_PROB"] = 0.0
FIRE = np.array([0, 0, 0, 0, 1.0], np.float32)
IDLE = np.array([0, 0, 0, 0, -1.0], np.float32)


@pytest.fixture(scope="module")
def env():
    e = UCAVEnv(stage=4, seed=1)
    e.set_forced_opponent(NullOpponent())
    yield e
    e.close()


def test_no_scripted_modules_are_imported():
    bad = [m for m in ("combat.enemy_brain", "combat.maneuvers", "combat.tactical_pilot", "combat.combat_env",
                       "combat.stick_env") if m in sys.modules]
    assert not bad, bad


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_reset_step_api(env, stage):
    env.set_stage(stage)
    obs, info = env.reset(seed=11)
    assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
    for _ in range(30):
        obs, r, te, tr, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r)
        if te or tr:
            assert "episode_record" in info
            break


def test_determinism(env):
    env.set_stage(2)
    acts = np.random.default_rng(5).uniform(-1, 1, (15, 5)).astype(np.float32)
    runs = []
    for _ in range(2):
        obs, _ = env.reset(seed=77)
        tr = [obs.copy()]
        for a in acts:
            obs, _, te, trn, _ = env.step(a)
            tr.append(obs.copy())
            if te or trn:
                break
        runs.append(tr)
    n = min(len(runs[0]), len(runs[1]))
    np.testing.assert_allclose(np.stack(runs[0][:n]), np.stack(runs[1][:n]), atol=1e-5)


def test_weapons_only_where_the_stage_says(env):
    for stage, own, enemy in ((1, 0, 0), (2, 0, 0), (3, 0, 4), (4, DC["AMMO"], 2)):
        env.set_stage(stage)
        env.reset(seed=5)
        assert env.ammo_blue == own and env.ammo_red == enemy, (stage, env.ammo_blue, env.ammo_red)
    env.set_stage(3)
    env.reset(seed=6)
    for _ in range(20):
        _, _, te, tr, info = env.step(FIRE)
        assert not info["events"]["launch"]
        if te or tr:
            break


def _align(env, behind):
    p, e = env.sim.player, env.sim.enemy
    f = p.get_forward_vector()
    e.position = p.position + f * (-3000.0 if behind else 3000.0)


@pytest.fixture
def gate():
    """Temporarily switch the old hard launch window on (UCAV_ENVELOPE=gate)."""
    saved = dict(DC["ENVELOPE"])
    DC["ENVELOPE"].update(INTERLOCK=True, range=(800.0, 7000.0), cos_ata_min=0.70, lock_time=0.0)
    yield
    DC["ENVELOPE"].clear()
    DC["ENVELOPE"].update(saved)


def test_envelope_blocks_impossible_launches_and_allows_valid_ones(env, gate):
    env.set_stage(4)
    env.reset(seed=3)
    _align(env, behind=True)                       # target 3 km behind: seeker cannot see it
    n0 = env.ammo_blue
    _, _, _, _, info = env.step(FIRE)
    assert info["events"]["blocked"] and not info["events"]["launch"] and env.ammo_blue == n0
    env.reset(seed=3)
    _align(env, behind=False)                      # target 3 km dead ahead
    _, _, _, _, info = env.step(FIRE)
    assert info["events"]["launch"] and env.ammo_blue == n0 - 1


def test_no_launch_bonus_and_miss_costs_more(env):
    assert DC["REWARD"]["miss"] <= -10.0
    env.set_stage(4)
    env.reset(seed=4)
    seen = set()
    for _ in range(100):
        _, _, te, tr, info = env.step(FIRE)
        seen |= set(info["reward_parts"])
        if te or tr:
            break
    assert not any("launch" in k for k in seen), seen


class ConstantRoll:
    name, kind = "const_roll", "test"

    def __init__(self):
        self.seen = 0

    def act(self, obs, rng=None):
        self.seen += 1
        assert obs.shape == (OBS_DIM,)
        return np.array([0, 1.0, 0, 0, -1], np.float32)


def test_enemy_moves_only_by_its_policy_output(env):
    env.set_stage(3)
    opp = ConstantRoll()
    env.set_forced_opponent(opp)
    try:
        env.reset(seed=8)
        r0 = env.sim.enemy.roll
        for _ in range(15):
            env.step(IDLE)
        assert opp.seen > 0 and env.sim.enemy.roll > r0 + 15.0
    finally:
        env.set_forced_opponent(NullOpponent())


def test_seat_symmetry_of_observation(env):
    env.set_stage(3)
    env.set_forced_opponent(ConstantRoll())
    try:
        env.reset(seed=9)
        for _ in range(10):
            env.step(np.array([0.2, -0.3, 0.1, 0.0, -1.0], np.float32))
        sim = env.sim
        red = env._obs_for("red").copy()
        p, e = sim.player, sim.enemy
        for f in ("position", "velocity", "roll", "pitch", "yaw", "roll_rate", "pitch_rate", "yaw_rate",
                  "current_speed", "target_speed", "airspeed", "altitude", "g_force"):
            a, b = getattr(p, f), getattr(e, f)
            setattr(p, f, type(b)(b) if isinstance(b, Vec3) else b)
            setattr(e, f, type(a)(a) if isinstance(a, Vec3) else a)
        env.ammo_blue, env.ammo_red = env.ammo_red, env.ammo_blue
        env._prev_a, env._prev_a_red = env._prev_a_red, env._prev_a
        env.red_v_ref, env.v_ref = env.v_ref, env.red_v_ref
        env.red_h_ref, env.h_ref = env.h_ref, env.red_h_ref
        np.testing.assert_allclose(env._obs_for("blue"), red, atol=1e-5)
    finally:
        env.set_forced_opponent(NullOpponent())


def test_physical_outcomes_use_hit_based_results(env):
    env.set_stage(2)
    env.reset(seed=12)
    env.sim.player.position = Vec3(env.sim.player.position.x, env.sim.player.position.y, 5.0)
    _, _, te, _, info = env.step(IDLE)
    assert te and info["termination_reason"] == "ground_collision" and info["result"] == "ucav_failed"
    env.reset(seed=13)
    p = env.sim.player
    p.position = Vec3(env.home.x + env.R + 500.0, env.home.y, p.position.z)
    _, _, te, _, info = env.step(IDLE)
    assert te and info["termination_reason"] == "zone_exit"
    allowed = {"hold_ok", "hold_fail", "pursuit_ok", "pursuit_fail", "survived", "target_hit", "target_hit_no_return",
               "ucav_hit", "ucav_failed", "no_hit", "mutual_hit", "enemy_out"}
    assert info["result"] in allowed and not any(w in info["result"] for w in ("win", "loss", "draw"))


def test_out_of_missiles_ends_stage4_without_hit(env):
    env.set_stage(4)
    env.reset(seed=14)
    env.ammo_blue = 0
    _, _, te, _, info = env.step(IDLE)
    assert te and info["termination_reason"] == "out_of_missiles" and info["result"] == "no_hit" and not info["success"]


def test_stage3_ends_as_survived_when_enemy_missiles_are_spent(env):
    env.set_stage(3)
    env.reset(seed=15)
    env.shots_red, env.ammo_red = 1, 0
    _, _, te, _, info = env.step(IDLE)
    assert te and info["result"] == "survived" and info["success"]


def test_replay_mixes_in_earlier_stages(env):
    DC["REPLAY_PROB"] = 1.0
    try:
        env.set_stage(4)
        seen = set()
        for i in range(12):
            env.reset(seed=100 + i)
            seen.add(env.ep_stage)
        assert seen <= {1, 2, 3} and len(seen) >= 2
    finally:
        DC["REPLAY_PROB"] = 0.0


@pytest.fixture
def realistic():
    """Temporarily switch to the realistic launch conditions (400-2000 m, 23 deg cone, 1 s lock, own flight limits)."""
    saved = dict(DC["ENVELOPE"])
    DC["ENVELOPE"].update(INTERLOCK=True, range=(400.0, 2000.0), cos_ata_min=0.92, lock_time=1.0, aoa_max=15.0, g_max=5.5, bank_max=60.0)
    yield
    DC["ENVELOPE"].clear()
    DC["ENVELOPE"].update(saved)


def test_realistic_range_window(env, realistic):
    env.set_stage(4)
    for dist, expect_launch in ((3000.0, False), (300.0, False), (1500.0, True)):
        env.reset(seed=21)
        p, e = env.sim.player, env.sim.enemy
        e.position = p.position + p.get_forward_vector() * dist
        env._lock["blue"] = 5.0                      # pretend the lock is already held; this isolates the range test
        _, _, _, _, info = env.step(FIRE)
        assert bool(info["events"]["launch"]) == expect_launch, (dist, info["events"])


def test_realistic_lock_time_delays_release(env, realistic):
    env.set_stage(4)
    env.reset(seed=22)
    p, e = env.sim.player, env.sim.enemy
    e.position = p.position + p.get_forward_vector() * 1500.0
    first = None
    for k in range(30):                              # 30 decisions = 3 s of holding the target in the cone
        e.position = p.position + p.get_forward_vector() * 1500.0
        _, _, te, tr, info = env.step(FIRE)
        if info["events"]["launch"]:
            first = info["time"]
            break
        if te or tr:
            break
    assert first is not None and first >= 1.0, f"released at {first}s, before the 1.0 s seeker lock"


def test_default_has_no_launch_gate(env):
    e = DC["ENVELOPE"]
    assert e["INTERLOCK"] is False and e["lock_time"] == 0.0 and e["aoa_max"] is None


def test_without_a_gate_the_launch_is_the_policys_decision(env):
    """No rule intervenes: a launch request with ammo and no missile in flight is released at ANY geometry (physics decides the result)."""
    env.set_stage(4)
    env.reset(seed=31)
    p, e = env.sim.player, env.sim.enemy
    e.position = p.position - p.get_forward_vector() * 9000.0       # enemy far BEHIND: a useless shot, but nothing blocks it
    n0 = env.ammo_blue
    _, _, _, _, info = env.step(FIRE)
    assert info["events"]["launch"] and not info["events"]["blocked"] and env.ammo_blue == n0 - 1


def test_start_distance_scale_moves_the_enemy_farther(env):
    import numpy as np
    env.set_stage(4)
    DC["REPLAY_PROB"] = 0.0
    def mean_init(scale):
        DC["START_DISTANCE_SCALE"] = scale
        out = []
        for sd in range(40, 70):
            env.reset(seed=sd)
            out.append(env.init_range)
        return float(np.mean(out))
    saved = DC.get("START_DISTANCE_SCALE", 1.0)
    try:
        assert mean_init(1.6) > mean_init(1.0) + 800.0
    finally:
        DC["START_DISTANCE_SCALE"] = saved


def test_missile_energy_override_reaches_the_simulator():
    import combat.combat_sim as cs
    from combat.ucav_env import apply_physics_overrides
    saved_cfg, saved_cs = DC["MISSILE"]["coast_decel"], cs.MSL_COAST_DECEL
    saved_life_cfg, saved_life = DC["MISSILE"]["life"], cs.MSL_LIFE
    saved_a_cfg, saved_a = DC["MISSILE"]["a_max"], cs.MSL_A_MAX
    try:
        DC["MISSILE"]["coast_decel"] = 40.0
        DC["MISSILE"]["life"] = 9.0
        DC["MISSILE"]["a_max"] = 140.0
        apply_physics_overrides()
        assert cs.MSL_COAST_DECEL == 40.0 and cs.MSL_LIFE == 9.0 and cs.MSL_A_MAX == 140.0
    finally:
        DC["MISSILE"]["a_max"] = saved_a_cfg
        cs.MSL_A_MAX = saved_a
        DC["MISSILE"]["coast_decel"] = saved_cfg
        DC["MISSILE"]["life"] = saved_life_cfg
        cs.MSL_COAST_DECEL = saved_cs
        cs.MSL_LIFE = saved_life


def test_launch_cost_is_charged_once_per_launch(env):
    env.set_stage(4)
    DC["REPLAY_PROB"] = 0.0
    saved = DC["REWARD"].get("launch_cost", 0.0)
    DC["REWARD"]["launch_cost"] = -5.0
    try:
        env.reset(seed=77)
        p, e = env.sim.player, env.sim.enemy
        e.position = p.position + p.get_forward_vector() * 3000.0
        _, _, _, _, info = env.step(FIRE)
        assert info["events"]["launch"]
        assert abs(env.reward_parts.get("launch_cost", 0.0) + 5.0) < 1e-9
    finally:
        DC["REWARD"]["launch_cost"] = saved


def test_economy_terms_are_off_by_default():
    assert DC["REWARD"]["launch_cost"] == 0.0 and DC["REWARD"]["unused_missile_bonus"] == 0.0


def test_describe_settings_states_the_active_setup():
    from combat.ucav_config import describe_settings
    text = describe_settings()
    assert "launch window = none" in text and "coast deceleration" in text and "start distances" in text


def test_base_game_enemy_ai_is_disabled(env):
    """The base simulator file still contains the original game's scripted enemy routine (update_enemy_ai).
    It runs only inside the base game's update task, which CombatSim removes, so the enemy moves only through its policy."""
    assert not env.sim.taskMgr.getTasksNamed("update_task")
