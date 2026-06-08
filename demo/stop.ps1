# demo/stop.ps1 — Gradio 데모 서버 + frpc 터널 일괄 종료
# 사용법: PowerShell에서  ./demo/stop.ps1   또는  powershell -File demo/stop.ps1
# 다른 파이썬 작업은 건드리지 않고 demo/app.py 프로세스만 종료합니다.

$killed = 0

# 1) demo/app.py 파이썬 프로세스만 콕 집어 종료
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'demo[/\\]app\.py' } |
    ForEach-Object {
        Write-Host ("kill python  PID={0}  port={1}" -f $_.ProcessId,
            ($(if ($_.CommandLine -match '--port\s+(\d+)') { $Matches[1] } else { '?' })))
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $killed++
    }

# 2) frpc 터널 전부 종료 (부모가 죽어도 안 따라 죽는 좀비들)
Get-Process -Name 'frpc_windows_amd64_v0.3' -ErrorAction SilentlyContinue |
    ForEach-Object {
        Write-Host ("kill frpc    PID={0}" -f $_.Id)
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
        $killed++
    }

Start-Sleep -Milliseconds 500

# 3) 결과 확인
$py   = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'demo[/\\]app\.py' }
$frpc = Get-Process -Name 'frpc_windows_amd64_v0.3' -ErrorAction SilentlyContinue

if (-not $py -and -not $frpc) {
    Write-Host "OK - all clean (demo server + frpc terminated, $killed killed)" -ForegroundColor Green
} else {
    Write-Host "WARNING - something still running:" -ForegroundColor Yellow
    if ($py)   { $py   | Select-Object ProcessId, CommandLine | Format-List }
    if ($frpc) { Write-Host ("frpc still: " + ($frpc.Id -join ', ')) }
}
