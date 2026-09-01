param(
  [Parameter(Mandatory = $true)]
  [string]$Directory
)

$ErrorActionPreference = "Stop"

$files = Get-ChildItem -LiteralPath $Directory -File -Filter "*.wav.txt"
$renamed = 0
$skipped = 0

foreach ($f in $files) {
  # Convert "...wav.txt" -> "...txt" (remove the extra ".wav" segment)
  $targetName = ($f.Name -replace '\.wav\.txt$', '.txt')
  $targetPath = Join-Path -Path $Directory -ChildPath $targetName

  if (Test-Path -LiteralPath $targetPath) {
    $skipped++
    continue
  }

  Move-Item -LiteralPath $f.FullName -Destination $targetPath
  $renamed++
}

Write-Host ("renamed=" + $renamed)
Write-Host ("skipped_existing_txt=" + $skipped)
