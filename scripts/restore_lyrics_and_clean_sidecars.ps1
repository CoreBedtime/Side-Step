param(
  [Parameter(Mandatory = $true)]
  [string]$Directory,

  [string]$Caption = "acapella, dry vocals"
)

$ErrorActionPreference = "Stop"

function Is-DashSeparator {
  param([string]$S)
  $t = ($S -replace '\s', '')
  if ($t.Length -lt 10) { return $false }
  return ($t -match '^[-_]+$')
}

function Extract-LyricsFromAnyFormat {
  param([string]$Text)

  $lines = $Text -split "`r?`n"

  # If it's already Option-A, use only the block after `lyrics:`
  $lyricsIdx = $null
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\s*lyrics\s*:\s*$') { $lyricsIdx = $i; break }
  }

  if ($lyricsIdx -ne $null -and ($lyricsIdx + 1) -le ($lines.Count - 1)) {
    return $lines[($lyricsIdx + 1)..($lines.Count - 1)]
  }
  return $lines
}

function Clean-Lyrics {
  param([string[]]$Lines)

  $out = New-Object System.Collections.Generic.List[string]

  foreach ($l in $Lines) {
    $t = $l.Trim([char]0xFEFF)

    # Drop obvious sidecar headers if they leaked into lyrics
    if ($t -match '^(caption|genre|bpm|key|signature|is_instrumental)\s*:') { continue }
    if ($t -match '^\s*lyrics\s*:\s*$') { continue }

    # Drop separators
    if (Is-DashSeparator -S $t) { continue }

    $out.Add($t)
  }

  # Trim leading/trailing blanks
  while ($out.Count -gt 0 -and $out[0].Trim() -eq '') { $out.RemoveAt(0) }
  while ($out.Count -gt 0 -and $out[$out.Count - 1].Trim() -eq '') { $out.RemoveAt($out.Count - 1) }

  # Drop first line if it looks like a title/notes header:
  # commonly "Song - Artist - notes..." or similar.
  if ($out.Count -gt 1) {
    $h = $out[0].Trim()
    if ($h -match '\s-\s') {
      $out.RemoveAt(0)
      while ($out.Count -gt 0 -and $out[0].Trim() -eq '') { $out.RemoveAt(0) }
    }
  }

  # Collapse multiple blank lines
  $collapsed = New-Object System.Collections.Generic.List[string]
  $prevBlank = $false
  foreach ($l in $out) {
    $isBlank = ($l.Trim() -eq '')
    if ($isBlank) {
      if ($prevBlank) { continue }
      $prevBlank = $true
      $collapsed.Add('')
    } else {
      $prevBlank = $false
      $collapsed.Add($l)
    }
  }

  return ($collapsed.ToArray() -join "`n")
}

$txts = Get-ChildItem -LiteralPath $Directory -File -Filter "*.wav.txt"
$updated = 0
$missingBackup = 0

foreach ($f in $txts) {
  $plainBak = ($f.FullName + ".plain.bak")
  $beforeCleanBak = ($f.FullName + ".before_clean.bak")

  $sourcePath = $null
  if (Test-Path -LiteralPath $plainBak) { $sourcePath = $plainBak }
  elseif (Test-Path -LiteralPath $beforeCleanBak) { $sourcePath = $beforeCleanBak }

  if ($null -eq $sourcePath) {
    $missingBackup++
    continue
  }

  $srcText = Get-Content -LiteralPath $sourcePath -Raw
  $lyricsLines = Extract-LyricsFromAnyFormat -Text $srcText
  $lyrics = Clean-Lyrics -Lines $lyricsLines

  $nl = [Environment]::NewLine
  $out = @(
    ("caption: " + $Caption),
    "genre: ",
    "bpm: ",
    "key: ",
    "signature: ",
    "is_instrumental: false",
    "lyrics:",
    $lyrics
  ) -join $nl

  Set-Content -LiteralPath $f.FullName -Value ($out + $nl) -Encoding utf8
  $updated++
}

Write-Host ("updated=" + $updated)
Write-Host ("missing_backup=" + $missingBackup)
