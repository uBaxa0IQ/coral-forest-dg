$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath 'build')) { New-Item -ItemType Directory -Path 'build' | Out-Null }
    $nirTex=Get-Command pdflatex -ErrorAction Stop
    $nirVersion=(& $nirTex.Source --version | Select-Object -First 1)
    $nirOptions=@('-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory=build')
    if ($nirVersion -match 'MiKTeX') { $nirOptions=@('--disable-installer')+$nirOptions }
    foreach ($nirPass in 1..3) {
        & $nirTex.Source @nirOptions 'report.tex' *> "build/pass-$nirPass.log"
        if ($LASTEXITCODE -ne 0) { Get-Content "build/pass-$nirPass.log" -Tail 30; throw "LaTeX pass $nirPass failed" }
    }
    Copy-Item -LiteralPath 'build/report.pdf' -Destination 'report.pdf' -Force
    Write-Output (Join-Path $PSScriptRoot 'report.pdf')
} finally { Pop-Location }
