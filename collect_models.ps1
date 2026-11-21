# Copies the trained model files from the working folder into this clean project.
# Optional -MakeZip then builds ..\ucav_rl_submission.zip (one folder ucav_rl_project with code, assets, models and results; no venv or caches).
#
#   .\collect_models.ps1 -Source "C:\Users\user\Downloads\ucav_project\new rl works\rl works\ucav rl"
#   .\collect_models.ps1 -Source "<old folder>" -MakeZip
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [switch]$MakeZip,
    [switch]$IncludeEpisodeLogs
)
$here = (Get-Location).Path

function Copy-Model($rel, $dest) {
    $src = Join-Path $Source $rel
    if (Test-Path $src) {
        $dst = Join-Path $here $dest
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Copy-Item $src $dst -Force
        Write-Host "copied   $rel  ->  $dest"
    } else {
        Write-Host "MISSING  $rel  (skipped)" -ForegroundColor Yellow
    }
}

$enemySrc = Join-Path $Source "models\enemies"
if (Test-Path $enemySrc) {
    New-Item -ItemType Directory -Force -Path (Join-Path $here "models\enemies") | Out-Null
    Copy-Item (Join-Path $enemySrc "*.zip") (Join-Path $here "models\enemies") -Force
    Write-Host "copied   models\enemies\*.zip"
} else {
    Write-Host "MISSING  models\enemies  (skipped)" -ForegroundColor Yellow
}

Copy-Model "models\ucav_bvr_eco\ucav_bvr_eco_v1.zip"              "models\final\FINAL_bvr_eco_v1.zip"
Copy-Model "models\ucav_bvr_eco\ucav_bvr_eco_v1_vecnormalize.pkl" "models\final\FINAL_bvr_eco_v1_vecnormalize.pkl"
Copy-Model "models\ucav\ucav_stage1_done.zip"                     "models\stages\stage1_done.zip"
Copy-Model "models\ucav\ucav_stage2_done.zip"                     "models\stages\stage2_done.zip"
Copy-Model "models\ucav\ucav_stage3_done.zip"                     "models\stages\stage3_done.zip"
Copy-Model "models\ucav\ucav_stage3_done_vecnormalize.pkl"        "models\stages\stage3_done_vecnormalize.pkl"
Copy-Model "models\ucav_learned_launch\ucav_final.zip"            "models\comparison\learned_launch_default_physics.zip"
Copy-Model "models\ucav\ucav_best_8M.zip"                         "models\comparison\gated_baseline_8M.zip"
Copy-Model "models\ucav_fresh5k\ucav_final.zip"                   "models\comparison\untrained_5k.zip"

# raw training logs (training_metrics.csv is small; episodes.jsonl can be several MB, copied only with -IncludeEpisodeLogs)
Copy-Model "logs\ucav_bvr_eco\training_metrics.csv"       "results\logs\final_run_training_metrics.csv"
Copy-Model "logs\ucav\training_metrics.csv"               "results\logs\stages_1_to_3_and_first_stage4_training_metrics.csv"
Copy-Model "logs\ucav_learned_launch\training_metrics.csv" "results\logs\learned_launch_training_metrics.csv"
Copy-Model "logs\ucav_fresh5k\training_metrics.csv"       "results\logs\untrained_5k_training_metrics.csv"
if ($IncludeEpisodeLogs) {
    Copy-Model "logs\ucav_bvr_eco\episodes.jsonl"         "results\logs\final_run_episodes.jsonl"
}

$settings = @"
FINAL_bvr_eco_v1.zip               needs  UCAV_COMBAT=bvr  and  UCAV_REWARD=economy   (evaluate and demo: UCAV_COMBAT=bvr)
learned_launch_default_physics.zip no preset (default physics); evaluate it with UCAV_COMBAT=bvr only to compare with the final model
gated_baseline_8M.zip              needs  UCAV_ENVELOPE=gate  (it was trained with the hard launch window 800-7000 m)
stage1_done / stage2_done / stage3_done   trained with the default physics (no preset); run them with --stage 1 / 2 / 3
untrained_5k.zip                   6,144 training steps in stage 1, no skills; evaluate with UCAV_COMBAT=bvr (stage 4)
"@
$settings | Out-File -Encoding utf8 (Join-Path $here "models\SETTINGS.txt")
Write-Host "wrote    models\SETTINGS.txt"

if ($MakeZip) {
    $parent = Split-Path $here
    $tmp = Join-Path $parent "ucav_rl_submission_tmp"
    $stage = Join-Path $tmp "ucav_rl_project"
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    # exclude only the top-level venv and logs folders (results\logs must stay)
    robocopy $here $stage /E /XD (Join-Path $here "venv") (Join-Path $here "logs") "__pycache__" ".pytest_cache" /XF *.pyc | Out-Null
    $zip = Join-Path $parent "ucav_rl_submission.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path $stage -DestinationPath $zip
    Remove-Item $tmp -Recurse -Force
    Write-Host "created  $zip"
}
