Trained policies (.zip) are placed in this folder. collect_models.ps1 copies them from the working folder (see README.md, section 3).

models/
  enemies/      opponent policies, one is drawn at random each episode (training and evaluation):
                  attacker_anchor.zip, attacker_final.zip, hold_policy.zip, evader_stage3.zip
                (a file whose name contains "hold" flies straight and level; every other file is an attacker)
  final/        FINAL_bvr_eco_v1.zip  (+ _vecnormalize.pkl, only needed to resume training)
  stages/       stage1_done.zip, stage2_done.zip, stage3_done.zip (+ stage3_done_vecnormalize.pkl)
  comparison/   learned_launch_default_physics.zip, gated_baseline_8M.zip, untrained_5k.zip

The environment variables each model needs are listed in README.md (section 3).
