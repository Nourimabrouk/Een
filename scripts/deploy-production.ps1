# Een Unity Mathematics - One-shot Production Deployment (Docker Compose)
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ComposeDir = Join-Path $Root 'deployment'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Error 'Docker is required'; exit 1
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
  Write-Host 'curl not found, health check will use Invoke-WebRequest'
}

if (-not $env:API_PORT) { $env:API_PORT = '8000' }
if (-not $env:REDIS_PORT) { $env:REDIS_PORT = '6379' }
if (-not $env:DB_PORT) { $env:DB_PORT = '5432' }
if (-not $env:PROMETHEUS_PORT) { $env:PROMETHEUS_PORT = '9090' }
if (-not $env:GRAFANA_PORT) { $env:GRAFANA_PORT = '3000' }
if (-not $env:DB_PASSWORD) { $env:DB_PASSWORD = 'een_unity_123' }

Write-Host 'Building images...'
docker compose -f (Join-Path $ComposeDir 'compose.yaml') build --no-cache

Write-Host 'Starting stack...'
docker compose -f (Join-Path $ComposeDir 'compose.yaml') up -d

Write-Host 'Waiting for API health...'
for ($i = 0; $i -lt 30; $i++) {
  try {
    Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$($env:API_PORT)/health" | Out-Null
    Write-Host 'API is healthy'
    break
  } catch {
    Start-Sleep -Seconds 2
  }
}

Write-Host 'Containers:'
docker ps --filter name=een-

Write-Host "Done. Visit http://localhost and http://localhost:$($env:API_PORT)/docs"

