# AERIS Windows runbook

Open PowerShell in the existing checkout. All commands preserve historical bundles.

```powershell
Set-Location 'C:\MSAIS\EGN6216-AI SYSTEMS\AHU_PROJECT\aeris-data-warmup_updated'
$py = 'C:\Users\mhars\miniconda3\envs\aeris-data\python.exe'
git status --short --branch
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
docker version
docker compose version
docker compose config --quiet
.\deployment\Run-Aeris.ps1 check
.\deployment\Run-Aeris.ps1 start
```

Docker Desktop must be running. The tested host CUDA runtime is PyTorch
2.9.1+cu128 on the RTX 4070 Laptop GPU. Docker uses the tested CPU path;
GPU passthrough in containers has not been verified. Do not install the CPU
requirements over the working host CUDA environment. The session's isolated
UI/evaluation environment is under the Codex task's `work\env`; install
Streamlit 1.50.0 and Matplotlib 3.10.7 in a separate environment if recreating it.

- AERIS UI: http://127.0.0.1:8510
- API readiness: http://127.0.0.1:8011/readyz
- Live metrics: http://127.0.0.1:8011/metrics
- Feedback cohort summaries: http://127.0.0.1:8011/summary
- Prometheus targets: http://127.0.0.1:9091/targets
- Prometheus alerts: http://127.0.0.1:9091/alerts
- Grafana dashboard: http://127.0.0.1:3001/d/aeris-operations

Grafana has anonymous read-only viewing for the loopback demo. Its dashboard
and data source are provisioned from deployment/grafana; no manual panel setup
is needed. Ports 8501 and 8000 were already occupied on this laptop, so the
host mappings above intentionally differ from the professor example. Service
scrapes use app:8001 inside Compose, independent of host mappings.

The application mounts local `data` read-only. Its SQLite ledger is in the
`aeris_feedback` named volume at `/state/ledger.sqlite3`. Prometheus and Grafana
also use named volumes. A normal startup loads the selected complete bundle;
it does not fit models. `data/deployment_selection.json` selects a version.
If absent, the verified October `v1_attention` bundle is the fallback default.
Missing or corrupted selected bundles produce not-ready/error states; there
is no silent substitution after an explicit selection fails.

## Reproduce experiments and evaluation

```powershell
& $py scripts\09_experiment.py audit
& $py scripts\13_data_audit.py
& $py scripts\09_experiment.py pilot
& $py scripts\09_experiment.py search
& $py scripts\10_evaluate.py
```

The final evidence plots use Matplotlib 3.10.7. In the isolated environment:

```powershell
$uiPy = 'C:\Users\mhars\Documents\Codex\2026-11-16\files-pasted-by-the-user-implement\work\env\Scripts\python.exe'
& $uiPy scripts\15_plot_evidence.py
& $uiPy scripts\18_report.py
```

The report command requires completed selection/evaluation and the final live
test file. Preserve its evidence order rather than presenting draft outputs as
completed runs.

Completed trials are reused by name; their configurations and model hashes
are retained under experiments/end_to_end_20261116 and data/aeris_models.
For a genuinely new experiment, version the run directory/name rather than
overwriting old results. The search uses the historical CUDA interpreter;
it never selects hyperparameters against target scores. The 50-epoch anchor
architecture uses historical defaults and logs the additional fusion sweep.
Evaluation writes the selection first, then opens descriptive later-source
and retrospective-target scores. It does not establish a sealed holdout.

Explicit container workflow commands:

```powershell
docker compose --profile workflow run --rm workflow scripts/workflow.py train --version v2 --run-name my_cpu_v2_run --epochs 50 --device cpu
docker compose --profile workflow run --rm workflow scripts/workflow.py evaluate
docker compose --profile workflow run --rm workflow scripts/workflow.py preprocess
```

Preprocessing deliberately refuses to overwrite the frozen READY data.
Use a separate data mount with the existing raw files and no prepared outputs
for a fresh rebuild. Keep old data and split manifests unchanged. The executed
one-epoch container_cpu_smoke_v2_20261116 run is an infrastructure smoke test,
not a finalist or a replacement for 50-epoch comparisons.

## Controlled traffic, feedback and monitoring

```powershell
docker compose exec -T app python scripts/11_deployment_test.py --api http://127.0.0.1:8001
docker compose exec -T app python -m unittest discover -s tests -v
docker compose exec -T prometheus promtool check config /etc/prometheus/prometheus.yml
docker compose exec -T prometheus promtool check rules /etc/prometheus/alerts.yml
Invoke-RestMethod http://127.0.0.1:8011/summary
Invoke-RestMethod 'http://127.0.0.1:9091/api/v1/query?query=aeris_eligible_predictions'
```

The traffic script submits only **test** records, including malformed and
unsupported examples, duplicate submissions and 100 timed requests. In the UI,
choose test, click Diagnose, then enter optional agreement/satisfaction/override
and notes before Save review record. Note the stable prediction ID. Reruns
and repeat saves must not increase eligible predictions or feedback coverage.
Test and genuine cohort metrics remain separate; no test label counts as
independent ground truth. A complete manual browser-render check is still
required because computer-use URL verification stopped automated browser work.

```powershell
& $py scripts\12_failure_demo.py
& $py scripts\19_verify_stack.py
```

This injects a missing bundle, waits for AerisUnavailable to fire, restores
the exact prior selection in a finally block, verifies recovery, restarts the
app, and compares feedback summaries. It writes failure_demo.json. Do not
interrupt power during the exercise; if interrupted, restore the prior bundle
with the rollback command below. Alert evidence is a controlled demonstration,
not a claim of semester-long monitoring or external notification delivery.
The second script switches to the earlier v1_attention bundle, verifies a real
prediction, restores the exact prior selection, and checks artifact hashes,
service health, dashboard provisioning, real Grafana totals and no-data ratios.
Both scripts generate test-cohort requests.

## Switch, rollback and stop

```powershell
.\deployment\Run-Aeris.ps1 rollback -Bundle v1_attention
.\deployment\Run-Aeris.ps1 status
docker compose restart app
Invoke-RestMethod http://127.0.0.1:8011/summary
.\deployment\Run-Aeris.ps1 stop
```

Choose another existing verified bundle name to switch back. The command
changes only the selection file; all earlier fitted artifacts remain intact.
`docker compose down` retains volumes. Do not add `-v` if feedback/monitoring
history must be retained.

Genuine-feedback training is an explicit offline candidate step using
scripts/14_feedback_candidate.py with a reviewed alignment manifest and ledger
path. It rejects opinions, test/simulated records, duplicate or dependent split
cases, and fewer than five independent cases per class in train and validation.
No currently collected records satisfy this gate. A successful candidate still
does not promote itself; inspect parent and reserved-feedback comparisons and
then explicitly select a passed bundle. A rejected candidate must never be
selected for demonstration as an approved improvement.

When genuine eligible labels exist, export a transactionally consistent ledger
snapshot and run the reviewed candidate on the host:

```powershell
New-Item -ItemType Directory -Force .\data\operator_feedback | Out-Null
docker compose exec -T app python -c "import sqlite3; s=sqlite3.connect('/state/ledger.sqlite3'); d=sqlite3.connect('/state/review_snapshot.sqlite3'); s.backup(d); d.close(); s.close()"
docker compose cp app:/state/review_snapshot.sqlite3 .\data\operator_feedback\review_snapshot.sqlite3
& $py scripts\14_feedback_candidate.py --db .\data\operator_feedback\review_snapshot.sqlite3 --manifest .\data\reviewed_feedback_manifest.json
```

The manifest must name `reviewed_by`, `parent_bundle`, a new `output_bundle`,
and `records` containing `prediction_id`, `window_id`, `independent_case_id`
and `partition` (`train` or `validation`). Alignment is checked against reserved
target-feedback sensor windows. This command currently rejects the zero eligible
labels; do not fabricate a manifest or treat test reviews as genuine labels.
The successful genuine-label update remains unverified until those data exist.
