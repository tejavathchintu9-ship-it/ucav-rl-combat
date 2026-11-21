# Autonomous UCAV Air Combat using Reinforcement Learning

This project trains an unmanned combat aerial vehicle (UCAV) in a 3D flight and missile simulation using reinforcement learning (PPO).
The UCAV decides its own stick and throttle commands and when to fire, and the enemy aircraft is also flown by a trained policy.
This README explains how to install the software, how to run the visual simulation, how to evaluate a trained model, and how to train
(the complete run, each stage on its own, and a short fresh run). The project definition, implementation and results are in sections 9 to 12;
the full description is in `PROJECT_REPORT.md`.

---------------------------------------------------------------------------------------------------------

## 1. Software and hardware required

| Item | Requirement |
|---|---|
| Operating system | Windows 10 or 11 (used for all runs here); Linux and macOS also work |
| Python | 3.12 or 3.13, 64-bit |
| Python packages | `panda3d` 1.10.x (3D simulation), `gymnasium` (environment interface), `stable-baselines3` 2.x (PPO), `torch` 2.x (CPU build is enough), `numpy`, `pytest` (tests) |
| Hardware | a normal laptop CPU; no GPU is used. Training runs 3 environments in parallel |
| Display | needed only for the visual simulation (a window with OpenGL). Training and evaluation run without a window |
| Disk space | about 100 MB for the project (35 MB of it is 3D models); each trained model is about 2 MB |

Versions used while developing: Python 3.12 and 3.13, panda3d 1.10.16, gymnasium 1.3, stable-baselines3 2.9, torch 2.x (CPU), numpy 2.x.

## 2. Installation

Open PowerShell in the project folder.

```
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and activate again.
A smaller CPU-only PyTorch can be installed first with `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
On Linux or macOS activate with `source venv/bin/activate`.

Check the installation:

```
python -c "import panda3d, gymnasium, stable_baselines3, torch, numpy; print('all packages found')"
python -m pytest tests -q
```

The second command runs the 25 unit tests and should report `25 passed`. The tests clear the `UCAV_*` environment variables themselves,
so they pass even if a preset is set in the window (`Get-ChildItem Env:UCAV*` lists what is set).

## 3. Project folders, models and settings

```
README.md  PROJECT_REPORT.md  SOURCES_AND_REFERENCES.md  requirements.txt  collect_models.ps1
train_ucav.py        training (all four stages in one run, or one stage at a time)
evaluate_ucav.py     fixed-seed evaluation table (no window)
run_ucav_demo.py     3D visual simulation with a text log of launches, hits and defences
combat/              ucav_env.py (environment), ucav_config.py (all settings), enemy_pool.py, combat_sim.py, config.py, metrics.py
simulator/           updated_chase_jet_sim_v5.py (aircraft physics and 3D scene)
assets/              3D models (aircraft, enemy, missile)
tests/               test_ucav_env.py (25 unit tests)
models/              trained policies (below)
results/             evaluation tables, training milestones, stage checks and demo logs (the data behind the report)
```

Trained policies are kept in `models/`. `collect_models.ps1 -Source "<working folder>"` copies them there from the folder where they were trained
(add `-MakeZip` to build `..\ucav_rl_submission.zip`, and `-IncludeEpisodeLogs` to include the per-episode training log).

| File | Content | Setting to use with it |
|---|---|---|
| `models/final/FINAL_bvr_eco_v1.zip` | the final model (stage 4) | `$env:UCAV_COMBAT="bvr"` |
| `models/stages/stage1_done.zip`, `stage2_done.zip`, `stage3_done.zip` | the model saved when each stage was passed | none; run with `--stage 1`, `2`, `3` |
| `models/enemies/*.zip` | opponent policies: `attacker_anchor`, `attacker_final`, `hold_policy` (flies straight and level), `evader_stage3`. They come from earlier training runs of this project | not applicable |
| `models/comparison/untrained_5k.zip` | a network trained for only 6,144 steps | `$env:UCAV_COMBAT="bvr"`, stage 4 |
| `models/comparison/learned_launch_default_physics.zip` | an earlier model, trained with the default missile physics | none |
| `models/comparison/gated_baseline_8M.zip` | the first stage-4 model, trained with a hard launch window | `$env:UCAV_ENVELOPE="gate"` |

The settings are environment variables. Set them in the same PowerShell window for training, evaluation and the demo, and use the setting a model was trained with.
The training, evaluation and demo scripts print one `settings:` line at the start that states the active launch window, missile energy loss, start distances and reward values;
check it before every run. A value that is not recognised (for example `UCAV_COMBAT="medium"`; the launch windows belong to `UCAV_ENVELOPE`) produces a `WARNING` line.
Clear one with `Remove-Item Env:UCAV_COMBAT`.

| Variable | Effect |
|---|---|
| `UCAV_COMBAT=bvr` | missiles lose speed after burnout (coast deceleration 40 m/s^2 instead of 7) and engagements start 1.6 times farther apart |
| `UCAV_REWARD=economy` | missile economy in the reward: a miss costs -25 (instead of -10), spending both missiles without a hit -60 (instead of -10), every launch -5, and +25 for each unused missile at a hit |
| `UCAV_ENVELOPE=gate` | a hard launch window (800 to 7,000 m, about 45 degrees). The default has no launch window |
| `UCAV_ENVELOPE=short`, `medium`, `realistic` | stricter launch windows for comparison experiments (`short` was run; `medium` and `realistic` are implemented but I did not complete a run with them) |
| `UCAV_COAST_DECEL=50`, `UCAV_START_SCALE=1.4` | set the two `bvr` values directly |
| `UCAV_MISSILE_AMAX=140` | lateral acceleration limit of every missile in m/s^2 (default 400, about 41 g). In a physics test a hard break did not defeat a well-aimed missile at 400, whatever its timing; at 140 a hard break about 4 s before impact did, so escaping becomes a matter of timing |
| `UCAV_MISSILE_LIFE=9` | flight time of every missile in seconds (default 22). The motor boost alone covers about 3 km, so a shorter life is what shortens the missile's reach (9 s gives about 5 km, 7 s about 4 km, straight-line estimates) and makes the UCAV close in before it fires. A model must be trained with the value it is run with |

## 4. Running the visual simulation

A window opens, and the terminal prints every launch (range and angle off the nose), hit, miss and defeated enemy missile.

```
$env:UCAV_COMBAT="bvr"
python run_ucav_demo.py --policy models\final\FINAL_bvr_eco_v1.zip --episodes 10 --debug
```

The policy flies by itself; there is nothing to control. Keys: **C** changes the camera (auto, chase, cockpit), **ESC** quits.
The window shows the aircraft, missiles with smoke trails, explosions, a text panel (range, angle, closure, missiles left, inbound-missile warning),
target brackets and a top-down radar. `--debug` adds the stick commands of the policy and the reward terms.

Useful options: `--enemy <file name without .zip>` (one fixed opponent, for example `evader_stage3`),
`--scenario tail_chase|head_on|beam|enemy_on_six|neutral`, `--stage 1|2|3|4` (default 4), `--seed N`, `--speed 2` (faster playback),
`--episodes 0` (run until the window is closed), `--enemy-dir <folder>`, and `--offscreen <folder>` (saves PNG frames when there is no display).

**Running each stage on its own.** The stage belongs to the environment, not to the model file, and the default is stage 4 (the only stage with weapons):

```
python run_ucav_demo.py --policy models\stages\stage1_done.zip --stage 1 --episodes 5 --debug    # flight only, result HOLD OK
python run_ucav_demo.py --policy models\stages\stage2_done.zip --stage 2 --episodes 5 --debug    # chase, nobody armed, CLOSE ON TARGET
python run_ucav_demo.py --policy models\stages\stage3_done.zip --stage 3 --episodes 5 --debug    # enemy fires, UCAV unarmed, SURVIVED
python run_ucav_demo.py --policy models\final\FINAL_bvr_eco_v1.zip --stage 4 --episodes 10 --debug  # full combat
```

Result labels: `TARGET HIT`, `NO HIT` (timeout or out of missiles), `UCAV HIT` (shot down), `MISSION FAILED` (crash or left the zone), `BOTH AIRCRAFT HIT`,
`ENEMY LEFT THE AREA` (invalid episode), and for stages 1 to 3: `HOLD OK/FAILED`, `CLOSE ON TARGET` / `NOT CLOSE ENOUGH`, `SURVIVED`.

## 5. Evaluating a model

```
$env:UCAV_COMBAT="bvr"
python evaluate_ucav.py --policy models\final\FINAL_bvr_eco_v1.zip --episodes 100
python evaluate_ucav.py --policy models\final\FINAL_bvr_eco_v1.zip --skills --episodes 50
```

The first command plays the same fixed start situations (seeds 123456 and up) against every enemy in `models/enemies` and prints one row per enemy
(about 5 minutes for 100 episodes per enemy, no window). The second re-tests the skill of each of the four stages. Other options: `--enemies <names>`,
`--stage N`, `--stochastic`, `--out report.json`.

| Column | Meaning |
|---|---|
| `hit` | share of valid episodes in which the UCAV's missile hit the enemy |
| `hitAll` | the same over all episodes (episodes in which the enemy left the zone count as misses) |
| `ucavhit`, `selffail` | UCAV shot down / UCAV crashed, stalled or left the zone |
| `msl/ep`, `hit/msl` | missiles fired per episode / hits per missile fired |
| `enemy_msl`, `defended` | enemy missiles per episode / share that did not hit (this includes enemy shots from poor geometry, so it overstates real defence) |
| `ttHit` | median time to the hit in seconds |
| `early<3s`, `t1st`, `R1st` | share of episodes in which the first shot came within 3 s / median time of the first shot / median range at the first shot in metres |
| `invalid` | episodes excluded because the enemy crashed or left the area |

`blocked/ep` is 0 unless a launch window is switched on. The enemy's action sampling is not seeded, so repeating an evaluation changes the numbers by a few hundredths.

## 6. Training

All training runs on the CPU with 3 environments in parallel (`--n-envs`). Every update uses 3 x 1,024 steps, so a requested step count is rounded up to a multiple of 3,072.
Stages 2 to 4 need the enemy policies in `models\enemies` (use `--enemy-dir` for another folder). Each training run writes:

* `<out-dir>\ucav_final.zip`: the model at the end of the run (an existing `ucav_final.zip` is first backed up as `ucav_final_before_<date>.zip`), and
  `ucav_last_stage<N>_<steps>steps.zip`, which names the stage and step count;
* `<out-dir>\ucav_stage<N>_done.zip` (with `_vecnormalize.pkl`): the model saved when stage N is passed;
* `<out-dir>\ucav_<steps>_steps.zip`: a checkpoint every 500,000 steps (`--checkpoint-every` changes this);
* `<log-dir>\training_metrics.csv` and `episodes.jsonl`: the training log.

A new `--out-dir` should be used for every experiment, so earlier models are not touched.

### 6.1 Complete run, all stages from scratch

Stages 1 to 3 (default physics, no environment variables set):

```
Remove-Item Env:UCAV_COMBAT, Env:UCAV_REWARD, Env:UCAV_ENVELOPE -ErrorAction SilentlyContinue
python train_ucav.py --stop-after-stage 3 --out-dir models\run --log-dir logs\run
```

The stages follow one another automatically; a stage is passed when its test is met and the next one starts from the same network. The messages `==> STAGE 1 PASSED ...`,
`==> STAGE 2 PASSED ...` and `==> STAGE 3 PASSED ...` mark the progress. If a stage does not meet its test within its step budget, the run stops with `NO-GO`
and keeps its checkpoints; stage 3 failed once in my runs and passed in the next, so in that case I resume from the stage-2 model (section 6.2) with another `--seed`.
This part takes about 40 minutes.

Stage 4 for the final model, from the stage-3 model, with the two settings used for it:

```
$env:UCAV_COMBAT="bvr"; $env:UCAV_REWARD="economy"
python train_ucav.py --start-stage 4 --resume models\run\ucav_stage3_done.zip --resume-vecnorm models\run\ucav_stage3_done_vecnormalize.pkl --out-dir models\run_final --log-dir logs\run_final --timesteps 5000000
```

This takes about 1.6 hours (5 million steps; `--timesteps` counts additional steps when resuming). The result is `models\run_final\ucav_final.zip`.

All four stages with default settings can also be run in a single command: `python train_ucav.py --out-dir models\full --log-dir logs\full` (default 8 million steps in total, about 4 hours).

### 6.2 Stage-wise training

Each stage can be trained and checked separately. Stage N is resumed from the model of stage N-1.

```
# Stage 1 (flight, no enemy): passes at about 100,000 steps, about 2 minutes
python train_ucav.py --stop-after-stage 1 --out-dir models\s1 --log-dir logs\s1
python run_ucav_demo.py --policy models\s1\ucav_stage1_done.zip --stage 1 --episodes 5 --debug

# Stage 2 (chase, nobody armed): passes about 300,000 steps later, about 6 minutes
python train_ucav.py --start-stage 2 --stop-after-stage 2 --resume models\s1\ucav_stage1_done.zip --resume-vecnorm models\s1\ucav_stage1_done_vecnormalize.pkl --out-dir models\s2 --log-dir logs\s2
python run_ucav_demo.py --policy models\s2\ucav_stage2_done.zip --stage 2 --episodes 5 --debug

# Stage 3 (defence, enemy armed): up to 30 minutes or more
python train_ucav.py --start-stage 3 --stop-after-stage 3 --resume models\s2\ucav_stage2_done.zip --resume-vecnorm models\s2\ucav_stage2_done_vecnormalize.pkl --out-dir models\s3 --log-dir logs\s3
python run_ucav_demo.py --policy models\s3\ucav_stage3_done.zip --stage 3 --episodes 5 --debug

# Stage 4 (combat): see section 6.1 for the final model
python train_ucav.py --start-stage 4 --resume models\s3\ucav_stage3_done.zip --resume-vecnorm models\s3\ucav_stage3_done_vecnormalize.pkl --out-dir models\s4 --log-dir logs\s4 --timesteps 5000000
```

The pass marks are: stage 1 hold success 0.90 (not before 100,000 steps), stage 2 pursuit success 0.50 with self-failure at most 0.10, stage 3 survived 0.80 and defended 0.70.
`--force-promote` continues to the next stage after a NO-GO, and `--seed N` changes the random seed.

### 6.3 Fresh training for a chosen number of steps, then testing and running it

To train a new network from nothing for a small number of steps (here 10,000) and look at the result:

```
python train_ucav.py --timesteps 10000 --out-dir models\fresh10k --log-dir logs\fresh10k
python run_ucav_demo.py --policy models\fresh10k\ucav_final.zip --stage 1 --episodes 5 --debug
$env:UCAV_COMBAT="bvr"
python evaluate_ucav.py --policy models\fresh10k\ucav_final.zip --episodes 50
python run_ucav_demo.py --policy models\fresh10k\ucav_final.zip --episodes 8 --debug
```

The run takes about 20 seconds and trains 12,288 steps (the next multiple of 3,072); it ends with the message `TRAINING ENDED INSIDE STAGE 1`, because stage 1 cannot be passed before 100,000 steps.
The stage-1 demo shows `HOLD FAILED` (the aircraft cannot hold level flight yet), and the stage-4 evaluation and demo show a network that fires at random
and is often shot down. The same commands with a larger `--timesteps` give a model with more training.

### 6.4 Learning curve from checkpoints

Training stage 1 to its end takes about 2 minutes and the checkpoints show how flight is learned:

```
python train_ucav.py --stop-after-stage 1 --checkpoint-every 25000 --out-dir models\curve --log-dir logs\curve
Get-ChildItem models\curve\*_steps.zip
foreach ($m in Get-ChildItem models\curve\*_steps.zip) { Write-Host "=== $($m.Name)"; python run_ucav_demo.py --policy $m.FullName --stage 1 --episodes 5 --speed 4 | Select-String "RESULT" }
python run_ucav_demo.py --policy models\curve\ucav_stage1_done.zip --stage 1 --episodes 5 | Select-String "RESULT"
```

The early checkpoints give `HOLD FAILED`, and the later ones and `stage1_done` give `HOLD OK`.

### 6.5 Checking whether a policy escapes missiles

`python tools\check_enemy_reaction.py` fires staged missiles at each enemy file (or at one file given with `--policy`) and counts how many miss; use the same `UCAV_*` settings as in the run being explained.
The count of missed missiles is the measure of escape. The extra columns show only how sensitive the network is to the incoming-missile inputs, which is not proof of evasion.

### 6.6 Reading the training log

`[S4] ep 16400 step 6582036 | 1139 sps | R 133.2 success 0.99 selffail 0.00 ucav_hit 0.00 close 0.10 shots/ep 1.03 hit/shot 0.96 defended 0.71`
shows the stage, number of episodes, total steps, speed, mean reward, share of successes, share of self-failures (crash, stall, zone exit), share of episodes in which the UCAV was shot down,
share of time close to the target, missiles per episode, hits per missile, and the share of enemy missiles that did not hit.
In stage 4 I look for `shots/ep` around 1 to 2 and `hit/shot` rising above 0.7. If `shots/ep` and `success` fall to about 0 while the reward stays positive,
the agent is staying close to the enemy instead of firing, and the run should be stopped.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| Crash or DLL error at start on Windows | `torch` must be imported before Panda3D; the provided scripts do this |
| Warning about missing enemy policies | put the enemy `.zip` files in `models\enemies` |
| A setting seems to have no effect | read the `settings:` line printed at the start and the `WARNING` lines. `UCAV_COMBAT` accepts only `bvr`, `UCAV_REWARD` only `economy`, `UCAV_ENVELOPE` one of `gate`, `short`, `medium`, `realistic`. Variables set earlier in the same window stay active until removed |
| A model behaves differently from the results below | it was trained with another setting; check `Get-ChildItem Env:UCAV*` against the table in section 3 |
| Tests fail on `launch_cost` or similar | a `UCAV_*` variable is set; the current tests clear them, so use the tests of this folder |
| Evaluation numbers differ a little between runs | the enemy actions are sampled; differences below about 0.05 are noise |
| Training is slow or the laptop gets hot | `--n-envs 2 --cooldown 0.3`; close other programs; do not run the demo window during training |
| Stage 3 `NO-GO` | resume from the stage-2 model with another `--seed` |
| No window (remote desktop or server) | use `evaluate_ucav.py`, or `run_ucav_demo.py --offscreen <folder>` |

---------------------------------------------------------------------------------------------------------

## 8. Documents in this folder

`PROJECT_REPORT.md` (project description, method, implementation, results, limitations), `SOURCES_AND_REFERENCES.md` (software and research papers used, and what was taken from each),
`results/` (evaluation tables, training milestones, stage checks, demo logs).

## 9. The project

**Problem.** An air-combat controller usually consists of scripted parts: a state machine that switches between behaviours, a library of manoeuvres, and a rule that decides when to fire.
The aim of this project is to replace all of that with learning: a UCAV that flies, chases an enemy, defeats incoming missiles and decides when to fire, using only a trained neural network.

**What "pure reinforcement learning" means here.** The stick, throttle and fire commands of the UCAV, and the behaviour of every enemy aircraft, come from trained policies.
There is no scripted pilot, no state machine, no manoeuvre library and no rule that decides when to fire. A launch needs only unused ammunition and no own missile in flight,
and the missile physics decides whether a shot hits. The base simulator file still contains the original game's scripted enemy routine (`update_enemy_ai`), but it is switched off: `CombatSim` removes the base game's update task, so
the enemy moves only through its policy output, and a unit test checks this. The environment, the reward and the training stages are designed by me, as in the research papers I used.

**Approach.** The task is learned in four stages, each adding one skill to the same network: flying, chasing, defending against missiles, and finally combat with weapons.
The enemy is a trained policy drawn at random from a small population at the start of each episode, so the UCAV never knows in advance who it will meet.
The result is judged by physical outcomes (a missile hit, being shot down, a crash), and each model is compared with an untrained network.

### Autonomy of the control, learning and working systems

| Part | What it is | How to check |
|---|---|---|
| Control of the UCAV | A neural network gives 5 outputs every 0.1 s. They are clipped to -1..1 and go straight to the aircraft inputs (`_apply_stick` in `combat/ucav_env.py`): elevator, aileron and rudder as control inputs, throttle as a change of the speed setpoint, and fire when the output exceeds 0.5. No autopilot, rule or manoeuvre routine sits between the network and the aircraft. The only fixed parts are the physical limits (speed 120 to 780 m/s) and the aircraft dynamics of the simulator | run the demo with `--debug`: the printed stick values are the network outputs |
| Control of the enemy | A loaded policy network, with the same interface. The original game's scripted enemy routine is switched off | `python -m pytest tests -q -k "enemy"` |
| Learning system | PPO (Stable-Baselines3) updates the network from experience in the simulation. `train_ucav.py` only runs the schedule: it moves to the next stage when the stage test is passed. The reward and the stage tests are designed by me | `train_ucav.py`, `combat/ucav_config.py` |
| Working system after training | The saved network is loaded and frozen. Its only input is the observation, and no human input is used during an episode: the demo window accepts only **C** (camera) and **ESC** (quit), and no key reaches the aircraft | `combat/combat_sim.py` (`setup_controls`) |
| Missiles | Fixed physics (proportional navigation), part of the environment | `combat/combat_sim.py` |

The unit tests check the main points: no scripted module is imported, the enemy moves only through its policy output, weapons exist only in stage 4, a launch request is released at any geometry when ammunition is left
(no firing rule), and the original scripted enemy routine is disabled (`python -m pytest tests -q`).

## 10. Implementation

* **Simulation.** A Panda3D 3D simulation with arcade-style flight dynamics (no gravity), a 12 km safety zone, and missiles that are flown for real: 5 s motor boost to 720 m/s,
  proportional-navigation guidance, 70 degree seeker cone, 25 m hit radius, 22 s life. The policy decides 10 times per second.
* **Observation and action.** 51 numbers (own flight state, last action, enemy direction, range, closure and aspect, zone, own missile state, inbound enemy missile, time left)
  and 5 outputs (elevator, aileron, rudder, throttle, fire; fire is requested when it exceeds 0.5).
* **Enemy.** Four trained policies (`attacker_anchor`, `attacker_final`, `evader_stage3`, `hold_policy`) with a randomised reaction delay of 1 to 3 decisions and sampled or mean actions.
  Start situations are drawn from five geometries (tail chase, head-on, beam, enemy on six, neutral).
* **Stages.** Flight (no enemy), pursuit (enemy, nobody armed), defence (enemy armed with 4 missiles, UCAV unarmed), combat (both armed with 2 missiles). The network of each stage starts from the previous one,
  and 20% of the episodes of later stages replay earlier stages.
* **Reward.** Outcome terms (hit +100, shot down -100, crash or leaving the zone -100, stall -50, timeout -20, miss -10 or -25), safety terms, a potential-based engagement term
  (angle, range, speed and height with weights 0.4, 0.3, 0.1, 0.2), defence terms, and the missile-economy terms of the `economy` setting.
* **Algorithm.** PPO (Stable-Baselines3): learning rate 3e-4 (1e-4 in stage 4), 1,024 steps per environment per update, batch 512, 5 epochs, gamma 0.995, GAE lambda 0.95, clip 0.2,
  256 x 256 tanh networks, reward normalization.
* **Code.** `combat/ucav_env.py` is the environment, `combat/ucav_config.py` holds every setting, `train_ucav.py` runs the staged training, `evaluate_ucav.py` and `run_ucav_demo.py` test and show a model.

## 11. Results

All results are fixed-seed evaluations against the four enemy policies; the tables are in `results/evaluation_tables.md`, and a longer discussion is in `PROJECT_REPORT.md`.

| Model | Episodes per enemy | Target hit | Hit over all episodes | UCAV shot down | Hits per missile | First shot within 3 s | Median first shot |
|---|---|---|---|---|---|---|---|
| Final model (`bvr` + `economy`), two evaluation runs | 100 | 0.94 / 0.96 | 0.93 / 0.94 | 0.01 | 0.86 / 0.92 | about 0.18 | about 5.3 s |
| Earlier model without the economy training, same physics | 100 | 0.86 | 0.85 | 0.03 | 0.61 | about 0.60 | about 1.6 s |
| Untrained network (6,144 steps) | 50 | 0.21 | 0.20 | 0.33 | 0.16 | about 0.24 | about 8 s |

Training: stage 1 was passed after 100,119 steps, stage 2 after 300,684 steps and stage 3 after 1,189,410 steps; the final model then trained for 5,000,000 steps in stage 4 (1.6 hours),
about 6.6 million steps in total. Stage checks (5 episodes each): the untrained network fails the stage-1 hold test in 5 of 5 episodes, the stage-1 model passes 5 of 5, the stage-2 model closes on the target in 4 of 5
and the stage-3 model survives 5 of 5. In 30 demo episodes of the final model there were 27 hits.

## 12. Limitations

* Each result comes from a single training run, so another random seed could give different numbers (stage 3 already differed between two runs).
* The four enemy policies come from earlier work on this project, and many of them fire early; their ability to evade was not tested separately.
* Sensing is perfect (no noise or delay), the physics is arcade-style without gravity, only one enemy is fought at a time, and there is no human weapon-release authority.
* The reward and the settings of the `bvr` and `economy` presets (energy loss 40 m/s^2, start distances 1.6 times larger) were chosen by hand and not tuned.
* A random network still hits about a fifth of the time because the missile model is forgiving, so a model should be judged by hits per missile, shot-down rate, missiles per episode and time of the first shot, and not by the hit rate alone.
