# Sources and references

This file lists what the simulation is built from, which ideas, methods and parameters I took from which research paper, and what I developed myself.
I adapted the ideas and did not reproduce the papers: my simulator, task and numbers differ from each of them, so my results cannot be compared directly with theirs.

Legend: **[D]** used directly (library or formula) | **[A]** adapted (an idea or parameter taken and implemented in my own way) |
**[B]** background or contrast (read, not implemented) | **[N]** developed in this project.

---------------------------------------------------------------------------------------------------------

## 1. Simulation sources

| Part | Source |
|---|---|
| Flight simulation, scene, camera, clouds (`simulator/updated_chase_jet_sim_v5.py`) | The Panda3D flight simulation the project started from, with arcade-style flight dynamics (no gravity); used unchanged **[D]** |
| Combat layer (`combat/combat_sim.py`): safety zone, missiles, explosions, HUD, radar | Part of the base project files, used as provided **[D]**. Missile model: 5 s motor boost to 720 m/s, coast, proportional navigation (gain 4), lateral limit 400 m/s^2, seeker cone 70 degrees, hit radius 25 m, life 22 s |
| Missile guidance law | Proportional navigation, a classical guidance law (Zarchan, 2002); it is also used in Hong-Peng (2023), Eq. 6 **[D]** |
| 3D models in `assets/` (`ba_f-15ex_eagle_ii.glb`, `enemy.glb`, `aim-9_missile.glb`) | The author and licence of these three files are not recorded in the project files; they should be confirmed before the project is redistributed |
| 3D engine | Panda3D, open source, https://www.panda3d.org **[D]** |
| Environment interface | Gymnasium (Towers et al., 2024) **[D]** |
| Learning library and PPO implementation | Stable-Baselines3 (Raffin et al., 2021) with PyTorch (Paszke et al., 2019) and NumPy (Harris et al., 2020) **[D]** |

## 2. What I took from each paper

**Zhang Hong-Peng (2023), automatic curriculum reinforcement learning without handcrafted reward** (arXiv:2307.06152)
* **[A]** Continuous control actions instead of a discrete list of manoeuvres (the paper argues that real action spaces are continuous).
* **[A]** A hit is decided by missile physics (miss distance: 30 m in the paper, 25 m here), not by entering an attack zone.
* **[A]** No launch gate: in the paper a launch outside the missile's off-axis limit is simply a miss. I do the same: a launch needs only unused ammunition and no own missile in flight.
* **[A]** Learning from easy to hard, and opponents that are earlier learned agents (in the paper each test agent fights 36 randomly chosen past agents; in mine a population of frozen policies is drawn per episode).
* **[A]** PPO with 256 x 256 tanh networks (its Table 2); start distances of 4 to 16 km, which support my long-range setting.
* **[B]** Contrast: its reward is only +1, -1 or 0. Mine is shaped (like Wang et al. and Li et al.), so my reward is not "reward-free".

**Wang Yuan et al. (2022), UCAV maneuvering decision based on heuristic reinforcement learning** (Computational Intelligence and Neuroscience, 1477078)
* **[A]** A hand-built reward made of angle, distance, speed and height terms (the family my engagement shaping belongs to), and a -50 penalty when position or speed leave the allowed limits (my counterparts are the zone-exit, crash and stall penalties).
* **[A]** The idea of a geometric attack condition (angle off the nose, distance window, height difference). I use it only as a small weapon-zone bonus (800 to 5,000 m, cos 0.94), never as a launch rule or a win condition.
* **[B]** Contrast: there the UCAV is judged to win as soon as the angle, height, distance and speed factors are all 1 (the weapon launch condition is reached), and the enemy uses a rolling time-domain maneuver decision method or random maneuvers, not a learned policy. In my system real missiles are flown and the enemy is learned.

**Li et al. (2022), UCAV maneuver decision in short-range combat with MS-DDQN** (Defence Technology 18, 1697-1714)
* **[A]** A shaped engagement reward from angle, distance, speed and height with fixed weights (its Table 4 lists 0.4, 0.3, 0.1 and 0.2). I use the same four values for angle, range, speed and height (`w_phi`, `w_R`, `w_V`, `w_h` in `combat/ucav_config.py`), in a potential-based form scaled by 10.
* **[A]** Training in sub-tasks from simple to harder ones, passing the network weights of the first task as the starting point of the next. My four stages do the same: each stage starts from the weights of the previous stage.
* **[A]** The notion of an attack distance (2,000 m in the paper), reflected in my short-range experiment.
* **[B]** Contrast: a discrete manoeuvre library with a DDQN learner; mine is continuous control with PPO.

**Liang et al. (2024), Meta-TD3-context for UCAV air combat** (AIPR 2024)
* **[B]** Background on deep reinforcement learning for UCAV combat, evaluation against a policy-based adversary, and the simple-to-complex scenario idea in its related work.
* **[B]** Its recurrent (GRU) context idea is not implemented; I list it as future work for inferring the enemy's intent.

**Search (2025), Tunnel: training environment for high-performance aircraft reinforcement learning** (arXiv:2505.01953)
* **[A]** A Gymnasium-standard environment with boundary, adversary and sensing primitives and configurable state, action and reward. My `UCAVEnv` follows this structure, with the safety zone as the boundary; PPO is the learner.
* **[B]** Its discussion of the fly-by-wire flight control layer (in my simulator the stick commands also pass through built-in rate control) and of how observation and action design matter for live air combat (my perfect-sensing limitation).

**Ure and Inalhan (2012), autonomous control of unmanned combat air vehicles** (IEEE Control Systems Magazine, doi 10.1109/MCS.2012.2205532)
* **[B]** The classical layered autonomy design: a tasking layer, a path and manoeuvre planner with a manoeuvre database, and feedback control (sliding mode). My system replaces the planner and the manoeuvre library with one learned policy; this paper is the scripted or hierarchical alternative I compare against.

**Bookstaber (undated manuscript), Unmanned Combat Aerial Vehicles: what men do in aircraft and why machines can do it better**
* **[B]** Missiles are designed with about a factor-of-three margin in lateral acceleration (up to 80 G); a UCAV evades with a rolling high-G turn started a few seconds before impact; weapon release will require confirmation by a human operator.
  I used this to interpret my missile-evasion tests (default 400 m/s^2, about 41 g) and in the limitations.

**Prakash (2022), AI and the future of air combat in India** (Journal of Defence Studies 16(4), 285-293) and **Kumar (2020), a brief review on unmanned combat aerial vehicles** (SSRN 3593220)
* **[B]** Strategic context, UCAV capabilities and software layers, and the ethical concern about autonomous weapons. They motivate the project and the limitation that there is no human weapon-release authority.

## 3. Methods and algorithms (by concept)

| Concept | Where in the project | Source |
|---|---|---|
| PPO (clipped objective) | `train_ucav.py` via Stable-Baselines3 | Schulman et al. (2017) **[D]**; also the learner in Hong-Peng (2023) and Search (2025) |
| Generalized advantage estimation (lambda 0.95), Adam optimizer | inside the Stable-Baselines3 PPO | Schulman et al. (2016); Kingma and Ba (2015) **[D]** |
| Reward scaling (not observation scaling) | `VecNormalize(norm_obs=False, norm_reward=True)` | Stable-Baselines3 **[D]** |
| Potential-based reward shaping (policy-invariant) | `phi_scale`, `w_phi`, `w_R`, `w_V`, `w_h` | Ng, Harada and Russell (1999) for the theory **[D]**; the four-factor form and the weights from Li et al. (2022) **[A]** |
| Staged curriculum with weight transfer | the four stages in `train_ucav.py` | Li et al. (2022), Hong-Peng (2023); general idea in Bengio et al. (2009) **[A]** |
| Learned opponents drawn per episode | `combat/enemy_pool.py` | Hong-Peng (2023) **[A]** |
| Hit decided by physics, no launch gate | `ENVELOPE` default (`INTERLOCK=False`) | Hong-Peng (2023) **[A]**; contrast Wang et al. (2022) |
| Proportional-navigation missile | `combat/combat_sim.py` | Zarchan (2002) **[D]** |
| Gymnasium environment with boundary and adversary | `combat/ucav_env.py` | Search (2025) **[A]** |
| Continuous stick, throttle and fire outputs | the policy output (5 numbers) | Hong-Peng (2023) **[A]** |

## 4. Developed in this project **[N]**

* The four-stage skill curriculum (fly, chase, defend, fire) with measurable pass marks, 20% replay of earlier stages, and weapons only in stage 4.
* A proximity success measure for stage 2 that is consistent with its reward (an earlier strict "behind the enemy" target did not learn).
* The missile-economy reward (`UCAV_REWARD=economy`) and the missile energy-loss and farther-start setting (`UCAV_COMBAT=bvr`).
* Optional launch-window settings (`gate`, `short`, `medium`, `realistic`); `gate` and `short` were used in comparison experiments.
* The evaluation columns `hitAll`, `early<3s`, `t1st` and `R1st`, the comparison with an untrained network, and unit tests that check that nothing in the decision path is scripted.

## 5. References

**Research papers**
1. Zhang Hong-Peng. *Maneuver Decision-Making Through Automatic Curriculum Reinforcement Learning Without Handcrafted Reward functions.* arXiv:2307.06152, July 2023.
2. Wang Yuan, Zhang Xiwen, Zhou Rong, Tang Shangqin, Zhou Huan, Ding Wei. *Research on UCAV Maneuvering Decision Method Based on Heuristic Reinforcement Learning.* Computational Intelligence and Neuroscience, vol. 2022, Article ID 1477078. doi:10.1155/2022/1477078.
3. Yong-feng Li, Jing-ping Shi, Wei Jiang, Wei-guo Zhang, Yong-xi Lyu. *Autonomous maneuver decision-making for a UCAV in short-range aerial combat based on an MS-DDQN algorithm.* Defence Technology 18 (2022) 1697-1714.
4. Siyuan Liang, Xiaodong Pu, Danian Zhao, Guodong Duan. *Autonomous Air Combat Decision Making Method for UCAVs Based on Meta-learning and Deep Reinforcement Learning.* AIPR 2024, Xiamen, 20-22 September 2024. doi:10.1145/3703935.3703999.
5. Gregory F. Search. *Training Environment for High Performance Aircraft Reinforcement Learning.* arXiv:2505.01953, May 2025.
6. N. Kemal Ure, Gokhan Inalhan. *Autonomous Control of Unmanned Combat Air Vehicles: Design of a multimodal control and flight planning framework for agile maneuvering.* IEEE Control Systems Magazine, October 2012. doi:10.1109/MCS.2012.2205532.
7. David Bookstaber. *Unmanned Combat Aerial Vehicles: What men do in aircraft and why machines can do it better.* Undated manuscript.
8. Bulbul Prakash. *AI and the Future of Air Combat in India.* Journal of Defence Studies, vol. 16, no. 4, October-December 2022, pp. 285-293.
9. Shashank Kumar. *A brief review on Unmanned Combat Aerial Vehicle (U.C.A.V).* SSRN 3593220, 2020.

**Standard methods and software**
10. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
11. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016 (arXiv:1506.02438).
12. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015 (arXiv:1412.6980).
13. A. Y. Ng, D. Harada, S. Russell. *Policy invariance under reward transformations: Theory and application to reward shaping.* ICML 1999.
14. Y. Bengio, J. Louradour, R. Collobert, J. Weston. *Curriculum learning.* ICML 2009.
15. A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann. *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research 22(268), 2021.
16. M. Towers et al. *Gymnasium: A Standard Interface for Reinforcement Learning Environments.* arXiv:2407.17032, 2024.
17. A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
18. C. R. Harris et al. *Array programming with NumPy.* Nature 585, 2020.
19. P. Zarchan. *Tactical and Strategic Missile Guidance.* AIAA Progress in Astronautics and Aeronautics, 4th edition, 2002.
20. Panda3D open-source game engine, https://www.panda3d.org.
