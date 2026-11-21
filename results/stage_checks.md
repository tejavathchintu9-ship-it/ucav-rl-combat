# Stage-by-stage checks (run_ucav_demo.py, 5 episodes each, each model in its own stage)

| Model | Command stage | Results of the 5 episodes |
|---|---|---|
| untrained network (6,144 steps) | `--stage 1` | HOLD FAILED, HOLD FAILED, HOLD FAILED, HOLD FAILED, HOLD FAILED |
| stage-1 model (`stage1_done`) | `--stage 1` | HOLD OK x 5 |
| stage-2 model (`stage2_done`) | `--stage 2` | CLOSE ON TARGET, CLOSE ON TARGET, CLOSE ON TARGET, NOT CLOSE ENOUGH, CLOSE ON TARGET |
| stage-3 model (`stage3_done`) | `--stage 3` | SURVIVED x 5 (each episode: all 4 enemy missiles ended without hitting; episodes 16 to 61 s) |
| stage-3 model (`stage3_done`) | `--stage 4` | NO HIT (out of missiles), NO HIT (timeout), TARGET HIT, NO HIT (timeout), NO HIT (out of missiles) |
| untrained network | `--stage 4` (8 episodes, `UCAV_COMBAT=bvr`) | 1 hit, 2 shot down, 5 ended with both missiles wasted; 10 of 12 launches were more than 60 degrees off the nose (worst 165 degrees) |

Observation: the untrained network cannot hold level flight, each saved stage model passes its own stage, and a model whose fire output was never trained (stage 3) does not fight well in stage 4.
