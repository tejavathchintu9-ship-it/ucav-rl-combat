# Evaluation tables (verbatim output of evaluate_ucav.py)

Fixed seeds 123456 and up, stage 4, four enemy files: attacker_anchor, attacker_final, evader_stage3, hold_policy (the gated and 3-enemy runs predate evader_stage3 or the `hitAll` column).
Column meaning is in README.md section 6. The enemy's sampling is not seeded, so repeating a run changes numbers by a few hundredths.

## A. FINAL model, run 1 (settings: UCAV_COMBAT=bvr, UCAV_REWARD=economy; 100 episodes per enemy)
```
enemy                  kind     hit   hitAll mission ucavhit selffail msl/ep hit/msl blocked/ep enemy_msl defended  ttHit  early<3s t1st  R1st invalid
attacker_anchor        attacker 0.92    0.92   0.92    0.02    0.00   1.21    0.76       0.00      1.72     0.71   17.4     0.24   4.6  6917  0
attacker_final         attacker 0.99    0.95   0.99    0.00    0.00   1.03    0.96       0.00      0.79     0.71   17.5     0.11   5.7  6207  4
evader_stage3          attacker 0.92    0.92   0.92    0.00    0.00   1.10    0.84       0.00      0.73     0.86   17.5     0.15   6.5  5708  0
hold_policy            passive  0.94    0.94   0.94    0.01    0.00   1.05    0.90       0.00      1.72     0.67   14.7     0.26   4.7  6789  0
MEAN over enemies:  target_hit 0.94  hit_all 0.93  ucav_hit 0.01  self_fail 0.00  hits_per_missile 0.86
```

## B. FINAL model, run 2 (same settings, same seeds, 100 episodes per enemy)
```
attacker_anchor        attacker 0.94    0.94   0.94    0.01    0.00   1.14    0.82       0.00      1.72     0.72   16.9     0.23   4.7  6320  0
attacker_final         attacker 1.00    0.92   1.00    0.00    0.00   1.03    0.97       0.00      0.76     0.79   17.7     0.17   5.4  6375  8
evader_stage3          attacker 0.98    0.98   0.98    0.00    0.00   1.04    0.94       0.00      0.70     0.79   17.4     0.09   6.5  5670  0
hold_policy            passive  0.94    0.94   0.94    0.01    0.00   0.99    0.95       0.00      1.69     0.67   15.0     0.23   4.7  6421  0
MEAN over enemies:  target_hit 0.96  hit_all 0.94  ucav_hit 0.01  self_fail 0.00  hits_per_missile 0.92
```

## C. Earlier "learned launch" model (trained with default physics, no economy reward), evaluated under UCAV_COMBAT=bvr, 100 episodes per enemy
```
attacker_anchor        attacker 0.76    0.75   0.76    0.05    0.00   1.44    0.52       0.00      1.75     0.77   18.8     0.65   1.3  7695  1
attacker_final         attacker 0.93    0.91   0.93    0.01    0.00   1.39    0.67       0.00      0.80     0.74   20.1     0.59   1.8  7295  2
evader_stage3          attacker 0.89    0.88   0.89    0.00    0.00   1.44    0.62       0.00      1.25     0.91   20.2     0.53   2.4  6551  1
hold_policy            passive  0.86    0.86   0.86    0.05    0.00   1.39    0.62       0.00      1.66     0.77   17.6     0.63   0.9  7706  0
MEAN over enemies:  target_hit 0.86  hit_all 0.85  ucav_hit 0.03  self_fail 0.00  hits_per_missile 0.61
```

## D. Same earlier model under its own (default) physics, 100 episodes per enemy (older script, no hitAll column)
```
enemy                  kind     hit   mission ucavhit selffail msl/ep hit/msl blocked/ep enemy_msl defended  ttHit  invalid
attacker_anchor        attacker 0.86    0.86    0.11    0.00   1.19    0.72       0.00      1.71     0.74   11.8  0
attacker_final         attacker 0.97    0.97    0.02    0.00   1.17    0.83       0.00      0.57     0.75   12.0  0
evader_stage3          attacker 0.94    0.94    0.02    0.00   1.29    0.73       0.00      0.91     0.80   13.1  0
hold_policy            passive  0.84    0.84    0.10    0.00   1.19    0.71       0.00      1.67     0.78   12.1  0
MEAN over enemies:  target_hit 0.90  mission_success 0.90  ucav_hit 0.06  self_fail 0.00  hits_per_missile 0.75
```

## E. UNTRAINED network (6,144 steps in stage 1), stage 4, UCAV_COMBAT=bvr, 50 episodes per enemy
```
attacker_anchor        attacker 0.22    0.22   0.22    0.52    0.00   1.08    0.20       0.00      1.54     0.49   26.0     0.23   9.3  6277  0
attacker_final         attacker 0.30    0.26   0.30    0.07    0.02   1.59    0.19       0.00      1.14     0.90   22.1     0.26   9.8  7818  6
evader_stage3          attacker 0.13    0.12   0.13    0.23    0.04   1.66    0.08       0.00      1.13     0.60   30.8     0.24   8.3  5907  3
hold_policy            passive  0.20    0.20   0.20    0.48    0.02   1.18    0.17       0.00      1.48     0.51   24.7     0.21   5.7  6463  0
MEAN over enemies:  target_hit 0.21  hit_all 0.20  ucav_hit 0.33  self_fail 0.02  hits_per_missile 0.16
```

## F. First stage-4 model with the hard launch window (trained to 8.4M steps; evaluated with the window on, 30 episodes per enemy, older script)
```
enemy                  kind     hit   mission ucavhit selffail msl/ep hit/msl blocked/ep enemy_msl defended  ttHit  invalid
attacker_anchor        attacker 0.93    0.93    0.03    0.03   1.03    0.90      13.50      0.70     0.52   14.9  0
attacker_final         attacker 0.93    0.93    0.07    0.00   1.00    0.93      16.17      0.17     0.20   13.1  1
evader_stage3          attacker 0.80    0.80    0.07    0.00   1.00    0.80      57.07      0.30     0.22   25.3  0
hold_policy            passive  0.93    0.93    0.07    0.00   1.00    0.93      17.90      0.50     0.33   13.0  0
MEAN over enemies:  target_hit 0.90  mission_success 0.90  ucav_hit 0.06  self_fail 0.01  hits_per_missile 0.89
```

## G. Same hard-window model, evaluated before evader_stage3 existed (3 enemies, 100 episodes; success included returning to the zone, a requirement later removed)
```
attacker_anchor        attacker 0.95    0.94    0.04    0.02   1.01    0.94       9.85      0.64     0.62   16.5  0
attacker_final         attacker 1.00    1.00    0.00    0.00   1.00    1.00      12.22      0.10     0.60   15.5  3
hold_policy            passive  0.95    0.92    0.03    0.03   1.01    0.94      12.50      0.52     0.58   15.4  0
MEAN over enemies:  target_hit 0.97  mission_success 0.95  ucav_hit 0.02  self_fail 0.02  hits_per_missile 0.96
```
Retention check of that model (`--skills`, 50 episodes):
```
1 flight       hold success 1.00  self-fail 0.00
2 pursuit      pursuit success 0.00  advantage 0.00  self-fail 0.00
3 defence      survived 0.92  defended 0.91  self-fail 0.03
4 combat       target hit 1.00  hits/missile 1.00  UCAV hit 0.00
```

## H. Launch-window experiments (comparison only), 100 episodes per enemy, older script
Short range (400 to 2,000 m window):
```
attacker_anchor        attacker 0.93    0.93    0.04    0.00   1.05    0.89      69.85      0.25     0.64   19.9  0
attacker_final         attacker 0.98    0.98    0.00    0.00   1.05    0.93      44.51      0.02     0.00   14.3  57
evader_stage3          attacker 0.91    0.91    0.06    0.00   1.02    0.89      57.64      0.08     0.25   15.0  0
hold_policy            passive  0.89    0.89    0.03    0.00   1.04    0.86      77.52      0.17     0.41   18.5  0
MEAN over enemies:  target_hit 0.93  mission_success 0.93  ucav_hit 0.03  self_fail 0.00  hits_per_missile 0.89
```
Second short-range run, trained for another 3,000,000 steps from the first short-range model. It was meant to be the "medium" window (about 32 degree cone, 0.5 s lock), but the variable for
that setting was entered in the wrong place (UCAV_COMBAT instead of UCAV_ENVELOPE), so UCAV_ENVELOPE=short from the previous run was still active:
```
attacker_anchor        attacker 0.68    0.68    0.09    0.03   0.82    0.83      64.89      0.23     0.57   25.0  0
attacker_final         attacker 1.00    1.00    0.00    0.00   1.03    0.97      53.03      0.03     0.00   17.4  67
evader_stage3          attacker 0.91    0.91    0.05    0.00   1.02    0.89      76.74      0.07     0.29   20.9  0
hold_policy            passive  0.80    0.80    0.05    0.00   0.90    0.89      71.28      0.08     0.25   28.0  0
MEAN over enemies:  target_hit 0.85  mission_success 0.85  ucav_hit 0.05  self_fail 0.01  hits_per_missile 0.90
```
In these two runs the `invalid` count for attacker_final is the number of episodes in which that enemy left the zone before the UCAV caught it; those are not counted in `hit`.
