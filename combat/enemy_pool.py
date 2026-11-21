"""
Enemy population.  The enemy is NOT controlled by us and is never scripted: each episode draws one frozen LEARNED
policy (a trained PPO .zip) from a directory, and varies how it behaves (reaction delay, sampled vs mean actions).
The UCAV cannot know in advance which enemy it faces, as in real combat.

Put enemy policies in  models/enemies/*.zip  (any trained policy with the same 51-value observation works, for
example the snapshots and final model of an earlier training run).  Names containing 'hold' are tagged passive
(a straight-and-level flier); everything else is tagged attacker.
If the directory is empty a NullOpponent (no controller at all) is used and a warning is printed by the trainer.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class NullOpponent:
    """No controller: neutral stick, no throttle change, no fire."""
    name, kind = "null_stick", "null"

    def act(self, obs, rng=None):
        return np.array([0.0, 0.0, 0.0, 0.0, -1.0], np.float32)


def act_policy(model, obs, stochastic=False):
    """Stick = mean action (or sampled when stochastic).  Fire is a rare-event output, so it is always sampled."""
    mean, _ = model.predict(obs, deterministic=True)
    mean = np.asarray(mean, np.float32).reshape(-1)
    samp, _ = model.predict(obs, deterministic=False)
    samp = np.asarray(samp, np.float32).reshape(-1)
    return samp if stochastic else np.concatenate([mean[:4], samp[4:5]])


class PolicyEnemy:
    def __init__(self, model, name, kind, stochastic=False):
        self.model, self.name, self.kind, self.stochastic = model, name, kind, bool(stochastic)

    def act(self, obs, rng=None):
        return act_policy(self.model, obs, self.stochastic)


def load_enemy(path, kind=None, stochastic=False):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    p = Path(path)
    k = kind or ("passive" if "hold" in p.stem.lower() else "attacker")
    return PolicyEnemy(PPO.load(str(p), device="cpu"), p.stem, k, stochastic)


class EnemyPool:
    def __init__(self, enemy_dir, cache=4):
        self.dir = Path(enemy_dir)
        self.cache_size = cache
        self._cache = {}

    def files(self):
        return sorted(p for p in self.dir.glob("*.zip") if not p.name.startswith("_tmp"))

    @staticmethod
    def kind_of(path):
        return "passive" if "hold" in Path(path).stem.lower() else "attacker"

    def _model(self, path):
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        key = (str(path), os.path.getmtime(path))
        if key not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = PPO.load(str(path), device="cpu")
        return self._cache[key]

    def sample(self, rng, stage_cfg, stochastic_prob=0.5):
        files = self.files()
        if not files:
            return None
        passive = [f for f in files if self.kind_of(f) == "passive"]
        attack = [f for f in files if self.kind_of(f) == "attacker"]
        want_passive = passive and (not attack or rng.random() < stage_cfg.get("passive_prob", 0.0))
        pool = passive if want_passive else (attack or passive)
        path = pool[int(rng.integers(0, len(pool)))]
        try:
            return PolicyEnemy(self._model(path), path.stem, self.kind_of(path), rng.random() < stochastic_prob)
        except Exception:
            return None
