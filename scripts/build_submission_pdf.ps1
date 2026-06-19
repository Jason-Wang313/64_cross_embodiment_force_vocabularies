param(
    [string]$PaperNumber = "64"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Paper = Join-Path $Root "paper"
$Downloads = Join-Path $env:USERPROFILE "Downloads"
$OutPdf = Join-Path $Downloads "$PaperNumber.pdf"

Push-Location $Paper
try {
    pdflatex -interaction=nonstopmode main.tex | Out-Host
    bibtex main | Out-Host
    pdflatex -interaction=nonstopmode main.tex | Out-Host
    pdflatex -interaction=nonstopmode main.tex | Out-Host
}
finally {
    Pop-Location
}

$SourcePdf = Join-Path $Paper "main.pdf"
if (!(Test-Path $SourcePdf)) {
    throw "Build did not create $SourcePdf"
}

Copy-Item -LiteralPath $SourcePdf -Destination $OutPdf -Force
$Hash = Get-FileHash -Algorithm SHA256 -LiteralPath $OutPdf
[pscustomobject]@{
    pdf = $OutPdf
    sha256 = $Hash.Hash
} | Format-List
