Write-Host "Stopping DeepWiki services..." -ForegroundColor Cyan

# 精确匹配本地地址端口，避免误杀无关进程
function Kill-Port {
    param([int]$Port, [string]$Label)
    # 只匹配本地地址列（第二列）严格等于 0.0.0.0:PORT 或 [::]:PORT 或 127.0.0.1:PORT
    $lines = netstat -ano | Select-String ("^\\s+TCP\\s+[\\d\\.\\[\\]:]+:" + $Port + "\\s")
    $killed = @{}
    foreach ($line in $lines) {
        $parts = $line.ToString().Trim() -split '\s+'
        $procId = $parts[-1]
        if ($procId -match '^\d+$' -and $procId -ne '0' -and -not $killed.ContainsKey($procId)) {
            $killed[$procId] = $true
            try {
                $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
                if ($proc) {
                    Write-Host "Killing $Label`: $($proc.Name) (PID: $procId)" -ForegroundColor Yellow
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                }
            } catch {}
        }
    }
}

Kill-Port -Port 8001 -Label "backend"
Kill-Port -Port 3000 -Label "frontend"
Kill-Port -Port 3001 -Label "port 3001"

Start-Sleep -Seconds 1

$check8001 = netstat -ano | Select-String ("^\\s+TCP\\s+[\\d\\.\\[\\]:]+:8001\\s")
$check3000 = netstat -ano | Select-String ("^\\s+TCP\\s+[\\d\\.\\[\\]:]+:3000\\s")
$check3001 = netstat -ano | Select-String ("^\\s+TCP\\s+[\\d\\.\\[\\]:]+:3001\\s")

if ($check8001 -or $check3000 -or $check3001) {
    Write-Host "Some ports are still in use!" -ForegroundColor Red
    if ($check8001) { Write-Host "8001 still open" }
    if ($check3000) { Write-Host "3000 still open" }
    if ($check3001) { Write-Host "3001 still open" }
} else {
    Write-Host "All DeepWiki services stopped successfully!" -ForegroundColor Green
}
