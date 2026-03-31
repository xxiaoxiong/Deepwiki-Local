# DeepWiki Docker image build & export script (Windows PowerShell)
# Usage: .\export-image.ps1
# Output: deepwiki-latest.tar  -> copy to Ubuntu, then: docker load < deepwiki-latest.tar

param(
    [string]$ImageName = "deepwiki",
    [string]$Tag = "latest",
    [string]$OutputFile = "deepwiki-latest.tar"
)

$ErrorActionPreference = "Stop"
$FullImageName = "${ImageName}:${Tag}"

Write-Host "========================================"
Write-Host " DeepWiki - Build & Export Docker Image"
Write-Host "========================================"
Write-Host ""

# Step 1: Build image (embedding model baked in)
Write-Host "[1/2] Building image $FullImageName ..." -ForegroundColor Yellow
Write-Host "      First build downloads the embedding model - this may take 10-30 min." -ForegroundColor Gray
docker build -t $FullImageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed. Check errors above." -ForegroundColor Red
    exit 1
}
Write-Host "Build succeeded!" -ForegroundColor Green
Write-Host ""

# Step 2: Export image to tar (no gzip needed, works natively on Windows)
Write-Host "[2/2] Exporting image to $OutputFile ..." -ForegroundColor Yellow
Write-Host "      Image is ~3-5 GB, export may take a few minutes..." -ForegroundColor Gray
docker save -o $OutputFile $FullImageName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Export failed." -ForegroundColor Red
    exit 1
}

$SizeMB = [math]::Round((Get-Item $OutputFile).Length / 1MB, 1)
Write-Host "Export done: $OutputFile ($SizeMB MB)" -ForegroundColor Green
Write-Host ""
Write-Host "========================================"
Write-Host " Next steps on Ubuntu server:"
Write-Host "========================================"
Write-Host "1. Copy these files to Ubuntu:"
Write-Host "   - $OutputFile"
Write-Host "   - docker-compose.yml"
Write-Host "   - .env  (set OPENAI_BASE_URL to your LAN LLM, e.g. http://192.168.x.x:8000/v1)"
Write-Host ""
Write-Host "2. Run on Ubuntu:"
Write-Host "   docker load < deepwiki-latest.tar"
Write-Host "   docker compose up -d"
Write-Host ""
Write-Host "3. Access:"
Write-Host "   Frontend : http://<server-ip>:3001"
Write-Host "   Backend  : http://<server-ip>:8001"
