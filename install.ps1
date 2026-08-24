# haste one-command install (Windows):
#   irm https://raw.githubusercontent.com/NodeNestor/haste/master/install.ps1 | iex
$ErrorActionPreference = "Stop"
$repo = "NodeNestor/haste"
$asset = "haste-windows-x86_64.exe"
$dir = Join-Path $env:USERPROFILE ".local\bin"
New-Item -ItemType Directory -Force $dir | Out-Null
$rel = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers @{ "User-Agent" = "haste-install" }
$url = ($rel.assets | Where-Object name -eq $asset).browser_download_url
if (-not $url) { throw "latest release has no asset $asset" }
$exe = Join-Path $dir "haste.exe"
if (Test-Path $exe) { Move-Item $exe "$exe.old" -Force }
Invoke-WebRequest $url -OutFile $exe -Headers @{ "User-Agent" = "haste-install" }
Remove-Item "$exe.old" -Force -ErrorAction SilentlyContinue
$path = [Environment]::GetEnvironmentVariable("Path", "User")
if ($path -notlike "*$dir*") {
    [Environment]::SetEnvironmentVariable("Path", "$path;$dir", "User")
    Write-Host "haste $($rel.tag_name) installed to $exe (added $dir to PATH — restart your shell)"
} else {
    Write-Host "haste $($rel.tag_name) installed to $exe"
}
