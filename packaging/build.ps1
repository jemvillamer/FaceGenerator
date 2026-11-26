<#
.SYNOPSIS
    Build the desktop app and its Windows installer.

.DESCRIPTION
    1. Makes the Windows icon (packaging\facegen.ico) from packaging\logo.png.
    2. Bundles the app with PyInstaller into dist\FaceGenerator\.
    3. Runs the bundled app's self-test: loads the model and generates one
       face, so a broken bundle is caught before it ships.
    4. Builds the installer with Inno Setup: dist\installer\FaceGeneratorSetup-<version>.exe

    Needs the Python 3.7 venv (.venv37) with requirements.txt and
    requirements-build.txt installed, and Inno Setup 6
    (winget install JRSoftware.InnoSetup).

.EXAMPLE
    .\packaging\build.ps1
    .\packaging\build.ps1 -Version 1.1.0 -IncludeModel
#>
param(
    # Shown in the installer and in Windows' installed-apps list.
    [string]$Version = "1.0.0",
    # Ship the 325 MB model inside the app so it works offline from the first
    # launch (otherwise it is downloaded on first launch). Must be in cache\.
    [switch]$IncludeModel,
    # Stop after the PyInstaller bundle (no self-test, no installer).
    [switch]$SkipInstaller,
    # Don't run the bundled app's self-test.
    [switch]$SkipSelfTest
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv37\Scripts\python.exe"

# Run a program and fail on a non-zero exit code. With "Stop", Windows
# PowerShell 5.1 turns anything a program prints on stderr (PyInstaller's
# progress lines) into a fatal error when output is redirected, so the call
# runs with "Continue" and is judged by its exit code instead.
function Invoke-Checked([string]$what, [scriptblock]$command) {
    $ErrorActionPreference = "Continue"
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)." }
}

if (-not (Test-Path $python)) {
    throw "Python 3.7 venv not found at $python. See README.md > Setup."
}
try { Invoke-Checked "import PyInstaller" { & $python -c "import PyInstaller" 2>$null } }
catch {
    throw "PyInstaller is not installed. Run: .venv37\Scripts\python.exe -m pip install -r requirements-build.txt"
}

Push-Location $root
try {
    Write-Host "== Making the app icon" -ForegroundColor Cyan
    Invoke-Checked "make_icon.py" { & $python packaging\make_icon.py }

    Write-Host "== Bundling with PyInstaller (takes a few minutes)" -ForegroundColor Cyan
    if ($IncludeModel) { $env:FACEGEN_INCLUDE_MODEL = "1" } else { $env:FACEGEN_INCLUDE_MODEL = "0" }
    Invoke-Checked "PyInstaller" {
        & $python -m PyInstaller --noconfirm --clean --log-level WARN `
            --distpath dist --workpath build packaging\facegen.spec
    }
    $exe = Join-Path $root "dist\FaceGenerator\FaceGenerator.exe"
    $size = (Get-ChildItem dist\FaceGenerator -Recurse | Measure-Object Length -Sum).Sum / 1GB
    Write-Host ("   Built {0} ({1:N2} GB)" -f $exe, $size)

    if ($SkipInstaller) { return }

    if (-not $SkipSelfTest) {
        Write-Host "== Self-test: loading the model and generating one face" -ForegroundColor Cyan
        $testDir = Join-Path ([IO.Path]::GetTempPath()) ("facegen-selftest-" + [guid]::NewGuid())
        $proc = Start-Process -FilePath $exe -ArgumentList "--self-test", "`"$testDir`"" -Wait -PassThru
        $resultFile = Join-Path $testDir "self_test.json"
        if (-not (Test-Path $resultFile)) {
            throw "Self-test wrote no result (exit code $($proc.ExitCode)). See %LOCALAPPDATA%\FaceGenerator\logs\facegen.log"
        }
        $result = Get-Content $resultFile -Raw | ConvertFrom-Json
        if (-not $result.ok) {
            throw "Self-test failed:`n$($result.error)`nLog: $($result.log_file)"
        }
        Write-Host ("   Passed in {0} s (InsightFace age check: {1})" -f $result.seconds, $result.age_check)
        if (-not $result.age_check) {
            Write-Warning "The age check is not available in the bundle; ages will be less accurate."
        }
        Remove-Item $testDir -Recurse -Force
    }

    Write-Host "== Building the installer with Inno Setup" -ForegroundColor Cyan
    $iscc = @(
        (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source,
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $iscc) {
        throw "Inno Setup 6 not found. Install it with: winget install JRSoftware.InnoSetup"
    }
    Invoke-Checked "Inno Setup" { & $iscc /Q "/DAppVersion=$Version" packaging\installer.iss }
    Write-Host "   Built $(Join-Path $root "dist\installer\FaceGeneratorSetup-$Version.exe")" -ForegroundColor Green
}
finally {
    Remove-Item Env:\FACEGEN_INCLUDE_MODEL -ErrorAction SilentlyContinue
    Pop-Location
}
