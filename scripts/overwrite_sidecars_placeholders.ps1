param(
  [Parameter(Mandatory = $true)]
  [string]$Directory,

  [string]$Caption = "acapella, dry vocals"
)

$ErrorActionPreference = "Stop"

$txts = Get-ChildItem -LiteralPath $Directory -File -Filter "*.wav.txt"

$nl = [Environment]::NewLine
$content = @(
  ("caption: " + $Caption),
  "genre: ",
  "bpm: ",
  "key: ",
  "signature: ",
  "is_instrumental: false",
  "lyrics:",
  ""
) -join $nl
$content = $content + $nl

$n = 0
foreach ($f in $txts) {
  # Best-effort backup, but avoid failing on long Windows paths.
  $bak = ($f.FullName + ".bak2")
  if (!(Test-Path -LiteralPath $bak)) {
    try {
      Copy-Item -LiteralPath $f.FullName -Destination $bak -Force -ErrorAction Stop
    } catch {
      # Ignore backup failures (commonly path length).
    }
  }
  Set-Content -LiteralPath $f.FullName -Value $content -Encoding utf8
  $n++
}

Write-Host ("overwritten=" + $n)
