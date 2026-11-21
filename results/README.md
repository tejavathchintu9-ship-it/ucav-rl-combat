# results/

Result data recorded from the training and test runs on the laptop (copied from the terminal output).

| File | Content |
|---|---|
| `evaluation_tables.md` | all evaluation tables (final model, comparisons, untrained network, launch-window experiments), verbatim |
| `training_milestones.md` | stage pass messages, the final training run's progress lines, times and steps |
| `stage_checks.md` | each model run in its own stage, and the untrained network |
| `demo_logs_final_model.txt` | the event logs of 30 demo episodes of the final model |
| `logs/final_run_training_metrics.csv` | raw log of the final stage-4 run (one row per logging window of about 20 episodes: steps 1.6 million to 6.59 million) |
| `logs/stages_1_to_3_and_first_stage4_training_metrics.csv` | raw log of the first run (stages 1 to 3, including the first attempt at stage 3 that ended in NO-GO, and the first stage-4 run up to 8.4 million steps) and of the resumed run |

The numbers in `PROJECT_REPORT.md` are taken from these files.

**Reading the CSV logs.** The columns (`mission_success`, `launch_rate`, `hit_rate`, `miss_rate`, `survival`, ...) are averages over all training episodes in the window, including the 20% of episodes
that replay earlier stages. Those episodes have no weapons, so `launch_rate` is at most about 0.8 and `hit_rate` is about 0.75 to 0.80 at the end of the final run. The success and hits-per-shot figures in
`PROJECT_REPORT.md` (0.99 and 0.96 at the end) are for stage-4 episodes only, as printed in the console during training; 0.75 divided by 0.8 is about 0.94, which agrees with them.

**Training steps recorded inside the model files** (read from each `.zip`): stage 1 100,119; stage 2 400,803; stage 3 1,590,213; final model 6,591,429; first stage-4 model 8,400,771;
earlier learned-launch model 11,402,115; untrained network 6,144.
