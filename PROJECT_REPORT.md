# Autonomous UCAV Air Combat using Reinforcement Learning: Project Report

## Abstract

I built a simulated unmanned combat aerial vehicle (UCAV) whose flying, chasing, missile defence and missile firing are all decided by a neural network trained with reinforcement learning (PPO).
The enemy aircraft is flown by a trained policy of the same kind. No scripted pilot, state machine, manoeuvre library or firing rule is used. The task is learned in four stages (fly, chase, defend, fire)
on a normal laptop CPU; the final model needed about 6.6 million training steps. In fixed-seed tests of 100 episodes against each of four enemy policies it hits the enemy in 94% to 96% of episodes,
is shot down in 1%, and hits with 86% to 92% of the missiles it fires. A network trained for only 6,144 steps hits in 21% of episodes, is shot down in 33% and hits with 16% of its missiles.
Each result comes from a single training run, and the reward, the training stages and the simulation settings were designed by me.

## 1. Introduction and objective

Conventional autonomous air-combat controllers are built from scripted parts: a finite-state machine that switches between behaviours, a library of manoeuvres, and a rule for weapon release.
The objective of this project is a UCAV that does all of this by learning. It must fly safely, chase an enemy it knows nothing about in advance, defeat incoming missiles, align itself and decide when to fire,
using only the output of a trained network. The enemy is not controlled by me during an episode and fires at the UCAV.

"Pure reinforcement learning" is used here in this sense: every control and firing decision of both aircraft comes from a trained policy, and nothing scripted sits in the decision path.
As in all the papers I used, the environment, the reward and the training stages are specified by the designer (section 7).

## 2. Background

The work follows the common approach of the recent UCAV air-combat literature: a deep reinforcement learning agent trained in a simulated engagement, with shaped rewards and a curriculum from simple to harder tasks
(Hong-Peng 2023; Li et al. 2022; Wang et al. 2022; Liang et al. 2024). Where the papers differ, I took from them as follows: continuous control actions, hits decided by missile physics and a launch that is not blocked by a rule
(Hong-Peng 2023); a four-factor engagement reward with fixed weights and training in sub-tasks with weight transfer (Li et al. 2022); a Gymnasium environment with a boundary and an adversary (Search 2025).
The classical alternative, a layered controller with a manoeuvre planner (Ure and Inalhan 2012), is what the learned policy replaces. The exact use of each source is listed in `SOURCES_AND_REFERENCES.md`.

## 3. System design and implementation

### 3.1 Simulation
The simulation is a Panda3D 3D scene with arcade-style flight dynamics (no gravity) and a 12 km radius safety zone around a home point. The agent decides 10 times per second.
Missiles are flown for real: a 5 s motor boost to 720 m/s, proportional-navigation guidance (gain 4), a 70 degree seeker cone, a 25 m hit radius, a 22 s life and a 400 m/s^2 lateral limit.
A collision with the ground, a stall below 70 m/s and leaving the zone end an episode as failures. An episode also ends at a hit, when the missiles are used up, or at the time limit.

### 3.2 Observation and action
The observation has 51 numbers: the own flight state (speed, altitude, attitude, rates, angle of attack, load factor), the last action, the direction, range, closure, speed, relative altitude and aspect of the enemy,
the distance and direction to the zone centre, the own missile state, an inbound enemy missile (range, time to impact, direction) and the time left. The enemy's observation is built by the same function,
so any trained policy can fly either aircraft.

The action has 5 numbers between -1 and 1: elevator, aileron, rudder, throttle and fire. A launch is requested when the fire output exceeds 0.5. A launch is released whenever ammunition is left
(2 missiles) and no own missile is in flight; no rule on range or angle intervenes, and the missile physics decides whether the shot hits.

### 3.3 Enemy and start situations
At the start of each episode one frozen trained policy is drawn from four files (`attacker_anchor`, `attacker_final`, `evader_stage3`, `hold_policy`; the "hold" policy flies straight and level and is drawn 15% of the time in stage 4).
Each enemy re-decides every 1 to 3 decisions and uses sampled or mean actions at random, so the same file behaves slightly differently each time. These policies come from earlier training runs of this project:
three from an earlier self-play version of the system, and `evader_stage3`, a copy of my stage-3 model. Start situations are drawn from five geometries: tail chase (25%), head-on (15%), beam (20%),
enemy on six (20%) and neutral (20%), with random distance (2.2 to 9 km, multiplied by 1.6 in the `bvr` setting), altitude, speed and heading.

### 3.4 The four stages
One network is carried through all stages, and each stage adds one skill.

| Stage | Task | Enemy | UCAV weapons | Enemy weapons | Passed when |
|---|---|---|---|---|---|
| 1 flight | hold speed and altitude, recover from a bad start, stay in the zone | none | none | none | hold success 0.90 (not before 100,000 steps) |
| 2 pursuit | stay 300 to 3,000 m from the enemy with the nose within 60 degrees of it | passive or attacker | none | none | pursuit success 0.50, self-failure at most 0.10 |
| 3 defence | defeat enemy missiles and stay inside the zone | attacker | none | 4 missiles | survived 0.80 and defended 0.70 |
| 4 combat | hit the enemy; chase, defence and firing all matter | passive or attacker | 2 missiles | 2 missiles | final stage |

Each stage starts from the weights of the previous one, and 20% of the episodes in later stages replay earlier stages to limit forgetting.

### 3.5 Reward
* Outcome terms: hit +100, shot down -100, crash or leaving the zone -100, stall -50, timeout -20, out of missiles -10, a miss -10 (-25 with the economy setting), and a small time cost.
* Safety and flight terms: penalties for extreme rates, sideslip, angle of attack, load factor, low altitude or speed, and a warning term near the zone edge.
* Engagement shaping (from stage 2): a potential-based term from angle, range, speed and height with weights 0.4, 0.3, 0.1 and 0.2, a proximity bonus, a small weapon-zone bonus and penalties for overshooting and closing too fast.
* Defence terms (stages 3 and 4): a cost while an enemy missile is inbound, a term for the time to impact, and a bonus when an enemy missile misses.
* Missile economy (`UCAV_REWARD=economy`): a miss -25, spending both missiles without a hit -60, a launch cost of -5 and +25 for each unused missile at a hit.

### 3.6 Learning algorithm
PPO from Stable-Baselines3: learning rate 3e-4 (1e-4 in stage 4), 1,024 steps per environment per update with 3 environments, batch size 512, 5 epochs, gamma 0.995, GAE lambda 0.95, clip range 0.2,
actor and critic networks of 256 x 256 units with tanh activation, a Gaussian policy with initial log standard deviation -1, and reward normalization.

### 3.7 Code
`combat/ucav_env.py` is the environment (stages, rewards, enemy handling), `combat/ucav_config.py` holds every setting, `combat/enemy_pool.py` loads the enemy policies, `combat/combat_sim.py` and `simulator/` are the simulation,
`train_ucav.py` runs the staged training, `evaluate_ucav.py` produces the evaluation tables and `run_ucav_demo.py` shows a model in the 3D window. 25 unit tests check, among other things, that weapons exist only in stage 4,
that the enemy moves only through its policy output, and that results are hit-based.

## 4. Development of the approach

| Step | What I found | What I changed |
|---|---|---|
| Original code | a scripted enemy, a +15 reward for any launch, and an episode that continued after the kill | trained enemy policies, no launch reward, an episode ends at the hit |
| Earlier version with self-play | it worked technically, but both sides fired in the first second and results were counted as win, loss or draw | results based on physical outcomes (hit, shot down, crash); the staged curriculum (fly, chase, defend, fire) |
| Stage 2 | success stayed at 0.00 for over a million steps: the strict "behind the enemy" target and a penalty for closing made the agent keep its distance | success redefined as staying close and pointing at the enemy, with a reward that pays exactly that; stage 2 was then passed after about 300,000 steps |
| Stage 3 | one run ended in NO-GO after 2.5 million steps (survived 0.37, defended 0.00), and a second run with the same code passed after 1.19 million steps | treated as run-to-run variation; I resumed from the stage-2 model |
| Launch window | an early version released a launch only inside a hard window (800 to 7,000 m, about 45 degrees) | removed, so that the policy decides when to fire. The hit rate stayed at 0.90, but hits per missile fell from 0.89 to 0.75 |
| Firing at the start | with the default missile model a shot from 6 km is almost as lethal as one from 2 km, so a hasty shot pays | missiles lose speed after burnout, engagements start 1.6 times farther apart, and the missile-economy reward (the `bvr` and `economy` settings) |

For the last row, first shots within the first 3 seconds fell from 0.60 to 0.18 of the episodes and hits per missile rose from 0.61 to 0.86 to 0.92 (same physics, section 6.2). I changed the physics and the reward together,
so I cannot say how much each contributed.

## 5. Training

| Run | Result |
|---|---|
| Stage 1 | passed after 100,119 steps (hold success 0.95), about 2 minutes |
| Stage 2 | passed after 300,684 steps (pursuit success 0.80, self-failure 0.00) |
| Stage 3 | first attempt NO-GO after 2,500,000 steps; second attempt passed after 1,189,410 steps (survived 0.80, defended 0.81, self-failure 0.00) |
| Stage 4, first version (hard launch window) | trained to 8.4 million total steps (15,193 s, about 4.2 hours); kept as a comparison model |
| Stage 4, final model (`bvr` and `economy`) | 5,000,000 steps from the stage-3 model in 5,792 s (about 1.6 hours); in the training window, success rose from 0.69 to 0.99 and hits per shot from 0.65 to 0.96 |

Progress of the final run (stage 4, step: success, hits per shot, shots per episode): 1.60M: 0.69, 0.65, 1.06; 3.03M: 0.82, 0.72, 1.14; 4.34M: 0.87, 0.82, 1.06; 6.13M: 0.98, 0.92, 1.06; 6.58M: 0.99, 0.96, 1.03.
More lines are in `results/training_milestones.md`, and the raw logs are in `results/logs/`. The step count stored in each model file agrees with these figures (stage 1: 100,119; stage 2: 400,803; stage 3: 1,590,213; final model: 6,591,429).
The CSV logs average over all training episodes, including the 20% that replay earlier stages without weapons, so their launch and hit rates are lower (about 0.8 and 0.75) than the stage-4 figures quoted here.

## 6. Testing and results

Protocol: `evaluate_ucav.py` with fixed seeds (123456 and up), stage 4, against each of the four enemy policies. The final model is run with `UCAV_COMBAT=bvr`. Complete tables are in `results/evaluation_tables.md`.

### 6.1 Final model (100 episodes per enemy, two evaluation runs because the enemy's action sampling is not seeded)

| | Run 1 | Run 2 |
|---|---|---|
| Target hit (valid episodes) | 0.94 | 0.96 |
| Target hit over all episodes | 0.93 | 0.94 |
| UCAV shot down | 0.01 | 0.01 |
| Self-failure (crash, stall, zone exit) | 0.00 | 0.00 |
| Hits per missile | 0.86 | 0.92 |
| Missiles per episode | 1.10 | 1.05 |
| First shot within 3 s | 0.19 | 0.18 |
| Median time of the first shot | 5.4 s | 5.3 s |
| Median range at the first shot | 6.4 km | 6.2 km |

### 6.2 Earlier model and final model under the same physics (100 episodes per enemy, `bvr` physics)

| | Earlier model (no economy training) | Final model |
|---|---|---|
| Target hit | 0.86 | 0.94 to 0.96 |
| Hits per missile | 0.61 | 0.86 to 0.92 |
| Missiles per episode | 1.42 | 1.05 to 1.10 |
| First shot within 3 s | 0.60 | 0.18 to 0.19 |
| Median time of the first shot | 1.6 s | 5.3 to 5.4 s |
| UCAV shot down | 0.03 | 0.01 |

### 6.3 Untrained network and final model

| | Untrained (6,144 steps), 50 episodes per enemy | Final model, 100 episodes per enemy |
|---|---|---|
| Target hit | 0.21 | 0.94 to 0.96 |
| UCAV shot down | 0.33 | 0.01 |
| Hits per missile | 0.16 | 0.86 to 0.92 |
| Self-failure | 0.02 | 0.00 |

The untrained network requests a launch in about 9% of its decisions and in the demo it fired in all directions, including backwards (up to 165 degrees off the nose); 10 of its 12 launches were more than 60 degrees off the nose.
It still hits about a fifth of the time, because the missile model is forgiving. For this reason I judge models by hits per missile and shot-down rate, and not by the hit rate alone.

### 6.4 Stage-by-stage checks (5 episodes each, each model in its own stage)

| Model | Stage | Result |
|---|---|---|
| untrained (6,144 steps) | 1 | HOLD FAILED in 5 of 5 |
| stage-1 model | 1 | HOLD OK in 5 of 5 |
| stage-2 model | 2 | CLOSE ON TARGET in 4 of 5 |
| stage-3 model | 3 | SURVIVED in 5 of 5 |
| stage-3 model run in stage 4 | 4 | 1 hit in 5 (its fire output was never trained) |

A retention check of the first stage-4 model (`--skills`, 50 episodes) gave a stage-1 hold success of 1.00, survived 0.92 and defended 0.91 in stage 3, and a stage-4 hit rate of 1.00,
but a stage-2 pursuit success of 0.00: the final policy does not try to stay close to the enemy, because its priority in stage 4 is the shot.

### 6.5 Demo of the final model
In 30 demo episodes (two runs, logs in `results/demo_logs_final_model.txt`) the final model made 27 hits; one episode ended with the UCAV shot down, one with a timeout and one with both missiles missed.
The first shot came within 3 s in only 4 of 30 episodes and between 4 and 15 s in 22 of them; it was fired from 3 to 8 km in 25 of 30 episodes and within 25 degrees of the nose in 21 of 30.
In 16 of 30 episodes at least one enemy missile ended without a hit. The sequence "enemy fires, UCAV evades, closes in, fires, hits" can be seen in many episodes.

### 6.6 Launch-window experiments (comparison only)

| Setting (100 episodes per enemy) | Target hit | Hits per missile | Enemy left the zone (invalid episodes) |
|---|---|---|---|
| Hard window 800 to 7,000 m (3 enemies) | 0.97 | 0.96 | 3 of 300 |
| Short range 400 to 2,000 m | 0.93 | 0.89 | 57 of 100 against one enemy |
| Short range, trained for another 3 million steps (second run) | 0.85 | 0.90 | 67 of 100 against one enemy |

The last row was meant to test a stricter window (a 32 degree cone and a 0.5 s lock), but the variable for that setting was entered in the wrong place in that session,
so the run used the short-range window again (the logged launch angles of up to 42 degrees also exceed that cone). It is therefore a second run of the short-range setting, and it shows
how much two runs of the same setting can differ (0.93 and 0.85). The `medium` and `realistic` windows are implemented in `combat/ucav_config.py`, but I did not complete a run with them.

A short launch window gave more realistic firing ranges but a lower hit rate once the escapes are counted, because the enemy had time to leave the zone.
I kept the learned launch (no window) as the main system.

## 7. Discussion

**What is learned and what is designed.**

| Learned by reinforcement learning | Designed by me |
|---|---|
| the stick, throttle and fire decisions of the UCAV | the reward (outcome, safety, engagement shaping, missile economy) |
| the behaviour of every enemy policy | the four stages, their pass marks and the 20% replay |
| when and from where to fire | the physics, the missile model, the zone and the missile count |
| chasing, evading and aiming | the energy loss (40 m/s^2) and the start distances (1.6 times) of the `bvr` setting, chosen by hand and not tuned |
| | the start situations and the enemy population |

**Autonomy of the control, learning and working systems.** Control: a neural network produces 5 outputs every 0.1 s, which are clipped to the range -1 to 1 and applied directly to the aircraft inputs (elevator, aileron and rudder as inputs,
throttle as a change of the speed setpoint, fire above 0.5); no autopilot, rule or manoeuvre routine lies between the network and the aircraft, and the only fixed parts are the physical limits (speed 120 to 780 m/s) and the aircraft dynamics
of the simulator. The enemy is controlled in the same way by a loaded policy. Learning: PPO updates the network from experience; the training script only runs the schedule and moves to the next stage when a stage test is passed.
Working system: the saved network is frozen, its only input is the observation, and no human input is used during an episode (the demo window accepts only the camera key and the exit key). The missiles are fixed physics.

**Checks that nothing is scripted.** The package contains no state machine, manoeuvre library or firing rule. The base simulator file still contains the original game's scripted enemy routine (`update_enemy_ai`),
but it is called only from the base game's update task, which `CombatSim` removes at start-up; a unit test confirms that this task does not exist, so the enemy aircraft moves only through its policy output.
The old scripted-enemy tables of the original configuration were removed from `combat/config.py`, which now contains only the randomisation of the start situations.

**Limitations.**
* Each result comes from a single training run; another seed could give different numbers (stage 3 already differed between two runs).
* The four enemy policies come from earlier work on this project and many of them fire early; their ability to evade was not tested separately.
* Sensing is perfect (no noise or delay), the physics is arcade-style without gravity, one enemy is fought at a time, and there is no human weapon-release authority.
* The `defended` figure counts every enemy missile that does not hit, including shots fired from poor geometry, so it overstates real defence.
* The reward shaping is hand-built, like in Wang et al. (2022) and Li et al. (2022); I did not test whether a sparse win-or-loss reward would also learn.

## 8. Conclusion

A UCAV controlled only by a trained policy learned to fly, chase, defend against missiles and fire, in four stages on a laptop CPU, against enemies that are also trained policies and that it cannot predict.
Compared with an untrained network and with an earlier version of the same agent, the final model hits more often, wastes fewer missiles, is shot down less, and waits several seconds before its first shot.
The main weaknesses are the single training run and the limited set of enemy policies. Natural extensions are additional training seeds, enemies trained independently of the UCAV
(for example by self-play against the new model), sensor noise, and a recurrent policy to infer the enemy's intent.

## 9. Reproducing the results

Installation, training (complete, stage by stage and fresh short runs), evaluation and the visual simulation are described in `README.md`. The recorded results are in `results/`.
