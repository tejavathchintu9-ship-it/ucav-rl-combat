"""
Start situations for the combat environment (used by combat/ucav_env.py).

Only the randomisation of the episode start is defined here. The aircraft physics
(simulator/updated_chase_jet_sim_v5.py) is not affected by it. All stages, rewards and
training settings are in combat/ucav_config.py.
"""

# ---- episode randomisation ---------------------------------------------------
RANDOMIZATION_CONFIG = dict(
    UCAV_START_RADIUS=(0.0, 4000.0),        # m from zone centre
    INITIAL_ALTITUDE_RANGE=(900.0, 2600.0),   # UCAV
    ENEMY_ALTITUDE_RANGE=(800.0, 3000.0),
    UCAV_SPEED_RANGE=(180.0, 350.0),
    ENEMY_SPEED_RANGE=(150.0, 260.0),
    UCAV_BANK_RANGE=(-20.0, 20.0),
    UCAV_PITCH_RANGE=(-6.0, 6.0),
    INITIAL_HEADING_RANGE=(0.0, 360.0),
    ENEMY_MAX_START_ZONE_FRAC=0.70,
    # geometry classes: name -> (probability, distance range, bearing range deg, aspect jitter deg)
    GEOMETRIES={
        "tail_chase":   (0.25, (3500, 7000), (-40, 40),   40),
        "head_on":      (0.15, (5000, 9000), (-25, 25),   30),
        "beam":         (0.20, (3500, 7500), (70, 110),  180),
        "enemy_on_six": (0.20, (2200, 4200), (155, 205),  0),
        "neutral":      (0.20, (4000, 9000), (-180, 180), 180),
    },
)
