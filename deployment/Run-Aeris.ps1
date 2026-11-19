param([ValidateSet('check','start','stop','status','evaluate','test','dashboard','rollback')][string]$Action='start',[string]$Bundle='v1_attention')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot)
switch ($Action) {
 'check' { git --version; docker version; docker compose version; docker compose config --quiet; Test-Path .\data\aeris_windows_v1\READY.json; Test-Path ".\data\aeris_models\$Bundle\READY.json" }
 'start' { docker compose up -d --build; docker compose ps; Write-Output 'UI http://127.0.0.1:8510 | Prometheus http://127.0.0.1:9091 | Grafana http://127.0.0.1:3001/d/aeris-operations' }
 'stop' { docker compose down }
 'status' { docker compose ps; Invoke-RestMethod http://127.0.0.1:8011/readyz; Invoke-RestMethod http://127.0.0.1:8011/summary }
 'evaluate' { docker compose --profile workflow run --rm workflow scripts/workflow.py evaluate }
 'test' { docker compose exec app python scripts/11_deployment_test.py --api http://127.0.0.1:8001 }
 'dashboard' { Start-Process 'http://127.0.0.1:3001/d/aeris-operations' }
 'rollback' {
    if ($Bundle -notmatch '^[A-Za-z0-9_-]+$' -or -not (Test-Path ".\data\aeris_models\$Bundle\READY.json")) { throw 'Choose an existing complete bundle name.' }
    @{bundle=$Bundle;reason='Explicit operator switch or rollback'} | ConvertTo-Json | Set-Content -Encoding utf8 .\data\deployment_selection.json
    docker compose restart app
    Write-Output 'Verify readyz and run a test prediction; the service checks model hashes before use.'
 }
}
