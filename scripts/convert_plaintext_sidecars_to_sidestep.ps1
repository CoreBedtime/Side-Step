param(
  [Parameter(Mandatory = $true)]
  [string]$Directory,

  [string]$Caption = "acapella, dry vocals",

  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$audioExts = @(".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aiff", ".aif")

$audio = Get-ChildItem -LiteralPath $Directory -File |
  Where-Object { $audioExts -contains $_.Extension.ToLowerInvariant() }

$converted = 0
$skipped = 0
$missing = 0

foreach ($a in $audio) {
  $sidecar = ($a.FullName + ".txt") # e.g. clip.wav -> clip.wav.txt
  if (!(Test-Path -LiteralPath $sidecar)) {
    $missing++
    continue
  }

  $orig = Get-Content -LiteralPath $sidecar -Raw

  $looksOptionA = $orig.TrimStart().ToLowerInvariant().StartsWith("caption:")
  if ($looksOptionA -and -not $Overwrite) {
    $skipped++
    continue
  }

  $bak = ($sidecar + ".plain.bak")
  if (!(Test-Path -LiteralPath $bak)) {
    Set-Content -LiteralPath $bak -Value $orig -Encoding utf8
  }

  $lyrics = $orig.Trim([char]0xFEFF).Trim()

  $lines = @(
    ("caption: " + $Caption),
    "genre: ",
    "bpm: ",
    "key: ",
    "signature: ",
    "is_instrumental: false",
    "lyrics:",
    $lyrics
  )

  $new = ($lines -join "`n") + "`n"
  Set-Content -LiteralPath $sidecar -Value $new -Encoding utf8
  $converted++
}

Write-Host "Converted sidecars."
Write-Host ("audio_found=" + $audio.Count)
Write-Host ("converted=" + $converted)
Write-Host ("skipped=" + $skipped)
Write-Host ("missing_sidecar=" + $missing)
