# Training milestones (from the training console output; the full per-run logs are logs\<run>\training_metrics.csv on the training machine)

Hardware: a normal laptop CPU, 3 parallel environments, about 750 to 1,300 steps per second.

## Stages 1 to 3 (default physics)
```
[S1] ep     20 step      5121 |  1251 sps | R  -168.6 success 0.00 selffail 0.50 ucav_hit 0.00 ...
==> STAGE 1 PASSED after 100119 steps: hold success 0.95 (need 0.9)
==> STAGE 2 PASSED after 300684 steps: pursuit success 0.80 (need 0.5), self-fail 0.00 (max 0.1)
==> NO-GO: stage 3 missed its criterion within 2500000 steps: survived 0.37 (need 0.8), defended 0.00 (need 0.7), self-fail 0.00 (max 0.1)
    (first attempt, 3,502 s)
==> STAGE 3 PASSED after 1189410 steps: survived 0.80 (need 0.8), defended 0.81 (need 0.7), self-fail 0.00 (max 0.1)
    (second attempt, resumed from the stage-2 model with the same code)
```

## First stage 4 (hard launch window, trained to 8.4 million total steps)
```
saved models\ucav/ucav_final.zip (15193s, stage 4: curriculum complete)
```
This run's stage-4 training window ended around: success 0.93, shots per episode 1.0, hits per shot 0.95 to 0.98, self-failure 0.00 to 0.05.

## Final model (UCAV_COMBAT=bvr, UCAV_REWARD=economy; resumed from the stage-3 model, 5,000,000 additional steps)
```
[S4] ep     20 step   1598310 |  140808 sps | R    98.5 success 0.69 selffail 0.00 ucav_hit 0.06 close 0.14 shots/ep 1.06 hit/shot 0.65 defended 0.88
[S4] ep   1000 step   1976127 |   3466 sps | R   131.2 success 0.74 selffail 0.00 ucav_hit 0.04 close 0.11 shots/ep 1.24 hit/shot 0.60 defended 0.92
[S4] ep   2000 step   2340174 |   2208 sps | R   123.8 success 0.78 selffail 0.00 ucav_hit 0.09 close 0.11 shots/ep 1.05 hit/shot 0.74 defended 0.82
[S4] ep   4000 step   3032004 |   1566 sps | R   136.8 success 0.82 selffail 0.00 ucav_hit 0.05 close 0.09 shots/ep 1.14 hit/shot 0.72 defended 0.83
[S4] ep   8000 step   4343556 |   1296 sps | R    76.5 success 0.87 selffail 0.01 ucav_hit 0.02 close 0.09 shots/ep 1.06 hit/shot 0.82 defended 0.76
[S4] ep  14480 step   6129408 |   1161 sps | R   124.1 success 0.98 selffail 0.00 ucav_hit 0.01 close 0.11 shots/ep 1.06 hit/shot 0.92 defended 0.69
[S4] ep  16400 step   6582036 |   1139 sps | R   133.2 success 0.99 selffail 0.00 ucav_hit 0.00 close 0.10 shots/ep 1.03 hit/shot 0.96 defended 0.71
saved models\ucav_bvr_eco/ucav_final.zip (5792s, stage 4: curriculum complete)
```
Observation: hits per shot rose from 0.65 to 0.96 while shots per episode stayed near 1, and the UCAV was shot down in 0% to 9% of episodes in the training window.

## Other runs
| Run | Steps | Wall time | Note |
|---|---|---|---|
| Earlier "learned launch" model (no launch gate, default physics) | +3,000,000 from the 8.4M model | 3,802 s | comparison model |
| Short-range window experiment | +3,000,000 | 3,222 s | comparison only |
| Second short-range run (meant to be the medium window; see the note in `evaluation_tables.md`) | +3,000,000 | 3,371 s | comparison only |
| Untrained reference | 6,144 (requested 5,000) | 5 s | "TRAINING ENDED INSIDE STAGE 1 after 6144 steps"; steps are rounded up to whole updates of 3 x 1,024 |
