"""
Configuration for the UCAV-vs-enemy autonomous combat system (combat/ucav_env.py).

Only the UCAV learns.  The enemy is part of the environment: a frozen, learned policy drawn per episode from a
varied enemy population (combat/enemy_pool.py).  Nothing here selects a flight action; it only parameterises
stages, rewards, the weapon envelope and training.

Stages (one skill added per stage; weapons only in stage 4):
  1 flight    : learn to fly                                   (no enemy, no weapons)
  2 pursuit   : stay close to the enemy with the nose on it    (enemy flies, nobody has weapons)
  3 defence   : defeat the enemy's missiles, stay in the zone  (enemy shoots, UCAV has no weapons)
  4 combat    : fire when it makes sense (the episode ends at the hit)     (both armed)
"""

UCAV_CONFIG = dict(
    DT=0.02,
    DECISION_STEPS=5,                       # 5 x 0.02 s -> 10 Hz policy
    SPEED_LIMITS=(120.0, 780.0),
    STICK=dict(pitch=1.0, roll=1.0, yaw=1.0, throttle_rate=60.0),
    FIRE_THRESHOLD=0.5,
    # Missile lateral-acceleration limit (m/s^2) for BOTH sides' missiles.  None = the simulator's own value (400).
    # The main results use None.  A lower value (for example 140) makes evasion a matter of timing; with 400 a learned
    # policy still passed stage 3.  The value changes the physics, so a model must be evaluated with the value it was trained with.
    MISSILE=dict(a_max=None, coast_decel=None, life=None),   # coast_decel: m/s^2 speed loss after burnout; None = simulator default (7: a missile keeps ~full speed to the end of its 22 s life)
    # Distances at which an engagement starts are multiplied by this (1.0 = the original table 2.2-9 km; 1.6 = about 3.5-14 km, clipped by the zone).
    START_DISTANCE_SCALE=1.0,
    # Optional, off by default: from stage 3 on, scale the pursuit rewards down, switch them off while an enemy missile is
    # inbound, and pay more for defence (see REWARD_REBALANCED).
    DEFENCE_REBALANCE=False,
    AMMO=2,                                 # UCAV missiles (stage 4)
    ZONE=dict(RADIUS=12000.0, WARN=0.50, RTB_COMPLETE_FRACTION=0.50, RTB_MIN_SECONDS=3.0, RTB_TIMEOUT_SECONDS=45.0),

    # Launch conditions.  Default: no gate (INTERLOCK False): a launch needs only ammunition and no own missile in flight.
    # When to fire is decided by the policy; the missile physics (seeker cone, motor, guidance, range) decides whether the shot
    # hits, and a wasted shot costs a missile and the miss penalty (as in Hong-Peng 2023, where a launch outside the missile's
    # off-axis limit is simply a miss).  UCAV_ENVELOPE=gate switches on a hard window (800-7000 m, about 45 deg), which was used
    # for the first stage-4 model.  The other presets are listed at the end of this file.  Extra keys: None / 0 = not checked.
    ENVELOPE=dict(INTERLOCK=False, range=(800.0, 7000.0), cos_ata_min=0.70, lock_time=0.0, aoa_max=None, g_max=None, bank_max=None),

    STAGES={
        1: dict(name="flight", seconds=30.0, enemy=False, own_weapons=False, enemy_weapons=False, enemy_ammo=0,
                v_target=(200.0, 320.0), alt_target=(1200.0, 2200.0),
                init_roll=60.0, init_pitch=20.0, init_speed_dev=60.0),
        2: dict(name="pursuit", seconds=60.0, enemy=True, own_weapons=False, enemy_weapons=False, enemy_ammo=0,
                scenarios=None, safe_enemy_heading=True, enemy_kinds=("passive", "attacker"), passive_prob=0.9),
        3: dict(name="defence", seconds=90.0, enemy=True, own_weapons=False, enemy_weapons=True, enemy_ammo=4,
                scenarios=None, enemy_kinds=("attacker",), passive_prob=0.0),
        4: dict(name="combat", seconds=150.0, enemy=True, own_weapons=True, enemy_weapons=True, enemy_ammo=2,
                scenarios=None, enemy_kinds=("passive", "attacker"), passive_prob=0.15),   # episode ends at the hit
    },
    # advantage position (stage 2 goal): UCAV behind the enemy, nose on it, at a controlled range
    ADVANTAGE=dict(range=(1000.0, 3500.0), cos_ata_min=0.85, cos_enemy_max=-0.50,   # rear-cone target: SHAPING only
               # stage-2 success: stay close to the enemy and point at it
               success_range=(300.0, 3000.0), success_cos=0.50, success_frac=0.40),
    ENEMY_VARIATION=dict(hold_steps=(1, 3), stochastic_prob=0.5),   # reaction delay (decisions) / sampled actions
    REPLAY_PROB=0.20,                       # share of episodes drawn from EARLIER stages (anti-forgetting)

    CURRICULUM=dict(
        WINDOW=100,
        S1=dict(min_steps=100_000, max_steps=800_000, success=0.90),
        S2=dict(min_steps=300_000, max_steps=2_000_000, success=0.50, fail_max=0.10),
        S3=dict(min_steps=500_000, max_steps=2_500_000, survive=0.80, defended=0.70, min_enemy_shots=20, fail_max=0.10),
        S4=dict(),
        LR_STAGE4=1.0e-4,
    ),
    HOLD_TOLERANCE=dict(roll=8.0, speed=25.0, alt=150.0, last_seconds=10.0),

    REWARD=dict(
        time=-0.01,
        rate=-0.03, smooth=-0.02, effort=-0.001,
        beta=-0.02, beta_hard=-0.5, beta_hard_deg=30.0,
        aoa=-0.04, aoa_free_deg=15.0, aoa_hard=-0.5, aoa_hard_deg=20.0,
        g=-0.05, g_free=9.0,
        low_alt=-0.5, low_alt_m=600.0, low_speed=-0.5, low_speed_ms=150.0,
        hold_roll=0.10, hold_roll_deg=20.0, hold_speed=0.20, hold_alt=0.20, hold_alt_wide=0.10,
        hold_gamma=0.10, hold_gamma_deg=8.0, hold_gamma_gain_m=80.0, hold_gamma_max_deg=15.0,
        hold_roll_error_penalty=0.08, hold_roll_error_scale=8.0,
        hold_speed_error_penalty=0.10, hold_speed_error_scale=25.0,
        hold_alt_error_penalty=0.05, hold_alt_error_scale=150.0,
        # pursuit (stage 2 onward, kept afterwards): potential (Li et al. 2022) + position-of-advantage bonus
        phi_scale=10.0, w_phi=0.4, w_R=0.3, w_V=0.1, w_h=0.2,
        Rg=2000.0, sigma_R=3000.0, hz=0.0, sigma_h=200.0, Vmax=450.0,
        prox_bonus=0.50, adv_bonus=0.15, overshoot=-0.3, overshoot_range=800.0, collision=-100.0,
        closure_penalty=-0.1, closure_range=2200.0, closure_free=80.0,
        wez_bonus=0.10, wez_range=(800.0, 5000.0), wez_cos=0.94, wez_threat=-0.10,
        # defence (stage 3 onward)
        missile_inbound=-0.25, defense_progress=8.0, enemy_missile_defeated=8.0, shot_down=-100.0,
        survived=20.0,
        # stage 4 only: multiplier on the pursuit rewards (proximity / rear-cone / weapon-zone bonuses).  1.0 = unchanged.
        # The realistic-launch preset lowers it so that hovering close to the enemy cannot out-earn a hit.
        pursuit_scale_combat=1.0,
        # only used when DEFENCE_REBALANCE is True: smaller pursuit rewards from stage 3, and none while a missile is inbound
        pursuit_scale_defence=0.2, phi_mult_defence=0.3,
        # weapons (stage 4): no launch bonus, a miss is expensive, blocked requests cost a little
        hit=100.0, miss=-10.0, blocked_request=-0.05, out_of_missiles=-10.0, timeout=-20.0,
        # missile economy (0 = off): a cost for every launch, and a bonus per missile still unused at the hit
        launch_cost=0.0, unused_missile_bonus=0.0,
        crash=-100.0, stall=-50.0, zone_exit=-100.0,
        zone_warn=-6.0, zone_safety_potential=8.0,
        rtb_progress=50.0, mission_complete=30.0, no_return=-20.0,
    ),

    PPO=dict(learning_rate=3e-4, n_steps=1024, batch_size=512, n_epochs=5, gamma=0.995,
             gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
             net_arch=[256, 256], log_std_init=-1.0),
    FIRE_INIT_MEAN=-0.5, FIRE_INIT_STD=0.4,
    LAPTOP=dict(N_ENVS=3, TORCH_THREADS=1, CHECKPOINT_EVERY=500_000, TOTAL_TIMESTEPS=8_000_000, COOLDOWN=0.0),
)

# values used only when DEFENCE_REBALANCE is True
REWARD_REBALANCED = dict(defense_progress=20.0, enemy_missile_defeated=30.0, survived=60.0)
if UCAV_CONFIG["DEFENCE_REBALANCE"]:
    UCAV_CONFIG["REWARD"].update(REWARD_REBALANCED)

# ---------------------------------------------------------------------------------------------------------------------
# OPTIONAL launch-condition presets (opt-in; default = no gate).  PowerShell:  $env:UCAV_ENVELOPE="short"
# then run train / evaluate / demo in that same window.  Tighten in steps, each step resumed from the previous model:
#
#   (default)  NO gate: launch is the policy's own decision, physics decides hit or miss
#   gate       the hard window (800-7000 m, ~45 deg); needed to evaluate models trained with it (models/comparison/gated_baseline_8M.zip)
#   none       same as the default (kept for compatibility)
#   short      range 400-2000 m only (the cone stays ~45 deg, no lock time, no flight-limit checks)
#   medium     + nose within ~32 deg (cos 0.85), target held in the cone 0.5 s
#   realistic  + nose within ~23 deg (cos 0.92), held 1.0 s, and the launching aircraft must itself be flying within limits
#              (angle of attack <= 15 deg, load factor <= 5.5 g, bank <= 60 deg)
#
# The cone, angle-of-attack, g and bank values are the launch limits used in the original version of the project.  Every preset also lowers the
# stage-4 pursuit rewards (pursuit_scale_combat 0.1): with the proximity bonus at full size, hovering close to the enemy out-earned a hit.
# A model must be trained AND evaluated/demonstrated with the same setting.
# Note: in short training runs "short" learned to fire, while "realistic" did not within about 250k stage-4 steps.
# ---------------------------------------------------------------------------------------------------------------------
import os as _os
_preset = _os.environ.get("UCAV_ENVELOPE", "").strip().lower()
if _preset == "gate":
    UCAV_CONFIG["ENVELOPE"]["INTERLOCK"] = True              # the old hard launch window: 800-7000 m, ~45 deg (rule, not learned)
if _preset in ("short", "medium", "realistic"):
    UCAV_CONFIG["ENVELOPE"]["INTERLOCK"] = True              # these presets are gated windows by definition
if _preset == "none":
    # NO launch conditions at all: a launch needs only ammunition (2) and no own missile in flight.  When to fire is then
    # decided purely by the policy and the reward (hit +100, miss -10).  Rewards are unchanged (pursuit_scale_combat stays 1.0).
    UCAV_CONFIG["ENVELOPE"]["INTERLOCK"] = False
if _preset in ("short", "medium", "realistic"):
    UCAV_CONFIG["ENVELOPE"].update(range=(400.0, 2000.0))
    UCAV_CONFIG["REWARD"]["pursuit_scale_combat"] = 0.1
if _preset == "medium":
    UCAV_CONFIG["ENVELOPE"].update(cos_ata_min=0.85, lock_time=0.5)
if _preset == "realistic":
    UCAV_CONFIG["ENVELOPE"].update(cos_ata_min=0.92, lock_time=1.0, aoa_max=15.0, g_max=5.5, bank_max=60.0)

# ---------------------------------------------------------------------------------------------------------------------
# ENVIRONMENT-REALISM presets (no rule is added: they change PHYSICS so that firing at once from far away stops being the best move).
#   $env:UCAV_COMBAT="bvr"     missile bleeds energy after burnout (coast_decel 40) AND engagements start farther apart (x1.6)
#   $env:UCAV_COAST_DECEL=50   set the missile energy loss directly (m/s^2; the simulator default is 7)
#   $env:UCAV_START_SCALE=1.4  set the start-distance multiplier directly
# Why: with the simulator's defaults a missile keeps almost full speed and a constant 400 m/s^2 of turning for its whole 22 s life,
# so a shot from 5-7 km is almost as lethal as one from 2 km, and hit probability does not depend on range.  Firing at once is then the
# best policy against any enemy that does not evade.  A model must be trained AND evaluated with the same setting.
# ---------------------------------------------------------------------------------------------------------------------
if _os.environ.get("UCAV_COMBAT", "").strip().lower() == "bvr":
    UCAV_CONFIG["MISSILE"]["coast_decel"] = 40.0
    UCAV_CONFIG["START_DISTANCE_SCALE"] = 1.6
if _os.environ.get("UCAV_COAST_DECEL", "").strip():
    UCAV_CONFIG["MISSILE"]["coast_decel"] = float(_os.environ["UCAV_COAST_DECEL"])
if _os.environ.get("UCAV_START_SCALE", "").strip():
    UCAV_CONFIG["START_DISTANCE_SCALE"] = float(_os.environ["UCAV_START_SCALE"])
# $env:UCAV_MISSILE_AMAX=140   lateral acceleration limit of every missile in m/s^2 (default 400, about 41 g).  In a physics test, a hard break did not defeat a
# well-aimed missile at 400, whatever its timing; at 140 a hard break started about 4 s before impact did, so escaping becomes a timing skill.
if _os.environ.get("UCAV_MISSILE_AMAX", "").strip():
    UCAV_CONFIG["MISSILE"]["a_max"] = float(_os.environ["UCAV_MISSILE_AMAX"])
# $env:UCAV_MISSILE_LIFE=9   flight time of every missile in seconds (default 22).  With the 5 s motor boost the missile covers about 3 km
# by itself, so only a shorter flight time makes a long shot useless: life 9 s gives a reach of about 5 km, 7 s about 4 km (straight-line estimate).
if _os.environ.get("UCAV_MISSILE_LIFE", "").strip():
    UCAV_CONFIG["MISSILE"]["life"] = float(_os.environ["UCAV_MISSILE_LIFE"])

# ---------------------------------------------------------------------------------------------------------------------
# REWARD preset: missiles are scarce (2) and precious.   $env:UCAV_REWARD="economy"
#   miss -25 (was -10), spending both missiles without a hit -60 (was -10), every launch costs -5, +25 for each missile still unused when the hit lands.
# This changes what the best move IS: a hasty low-probability shot now costs more than it can win.  It does NOT tell the policy where to be or when
# to fire.  It only produces patience if hit probability really differs between a prepared and an unprepared shot (see UCAV_COMBAT=bvr above).
# ---------------------------------------------------------------------------------------------------------------------
if _os.environ.get("UCAV_REWARD", "").strip().lower() == "economy":
    UCAV_CONFIG["REWARD"].update(miss=-25.0, out_of_missiles=-60.0, launch_cost=-5.0, unused_missile_bonus=25.0)

# ---------------------------------------------------------------------------------------------------------------------
# Check the preset variables.  A value that is not recognised (for example UCAV_COMBAT="medium": the launch-window presets belong to
# UCAV_ENVELOPE) would otherwise be ignored without any message.
# ---------------------------------------------------------------------------------------------------------------------
import sys as _sys
_VALID = {"UCAV_COMBAT": ("bvr",), "UCAV_REWARD": ("economy",), "UCAV_ENVELOPE": ("gate", "none", "short", "medium", "realistic")}
for _var, _ok in _VALID.items():
    _val = _os.environ.get(_var, "").strip().lower()
    if _val and _val not in _ok:
        print(f"WARNING: {_var}='{_val}' is not recognised and is ignored. Valid values for {_var}: {', '.join(_ok)}.", file=_sys.stderr, flush=True)


def describe_settings():
    """One line that states which settings are active (printed by the training, evaluation and demo scripts)."""
    e, r, m = UCAV_CONFIG["ENVELOPE"], UCAV_CONFIG["REWARD"], UCAV_CONFIG["MISSILE"]
    if e["INTERLOCK"]:
        window = f"hard window {e['range'][0]:.0f}-{e['range'][1]:.0f} m, cone cos>={e['cos_ata_min']}, lock {e['lock_time']} s"
    else:
        window = "none (the policy decides)"
    return (f"settings: launch window = {window} | missile coast deceleration = {m['coast_decel'] if m['coast_decel'] is not None else 'default (7)'} m/s^2, life = {m['life'] if m['life'] is not None else 'default (22)'} s, turn limit = {m['a_max'] if m['a_max'] is not None else 'default (400)'} m/s^2 | "
            f"start distances x{UCAV_CONFIG['START_DISTANCE_SCALE']} | reward: miss {r['miss']}, out of missiles {r['out_of_missiles']}, launch cost {r['launch_cost']}")
