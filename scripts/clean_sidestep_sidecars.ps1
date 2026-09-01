param(
  [Parameter(Mandatory = $true)]
  [string]$Directory,

  [string]$Caption = "acapella, dry vocals",

  [switch]$OverwriteCaption
)

$ErrorActionPreference = "Stop"

function Get-LyricsLinesFromSidecar {
  param([string]$Text)

  $lines = $Text -split "`r?`n"
  $lyricsIdx = $null
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match '^\\s*lyrics\\s*:\\s*$') { $lyricsIdx = $i; break }
  }

  if ($lyricsIdx -ne $null) {
    if ($lyricsIdx + 1 -le $lines.Count - 1) {
      return $lines[($lyricsIdx + 1)..($lines.Count - 1)]
    }
    return @()
  }
  return $lines
}

function Clean-LyricsLines {
  param([string[]]$Lines)

  function Is-DashSeparator {
    param([string]$S)
    $t = ($S -replace '\\s', '')
    if ($t.Length -lt 10) { return $false }
    return ($t -match '^[-_]+$')
  }

  $clean = New-Object System.Collections.Generic.List[string]
  foreach ($l in $Lines) {
    $t = $l.Trim([char]0xFEFF)
    if (Is-DashSeparator -S $t) { continue } # dashed separators
    $clean.Add($t)
  }

  # If the lyrics blob accidentally includes Option-A headers, strip them
  # (including repeated headers and any blank lines before real lyrics).
  while ($clean.Count -gt 0) {
    $head = $clean[0].Trim()
    if ($head -eq '') { $clean.RemoveAt(0); continue }
    if (Is-DashSeparator -S $head) { $clean.RemoveAt(0); continue }
    if ($head -match '^(caption|genre|bpm|key|signature|is_instrumental|lyrics)\\s*:') {
      $clean.RemoveAt(0)
      continue
    }
    break
  }

  # Trim leading/trailing blank lines
  while ($clean.Count -gt 0 -and $clean[0].Trim() -eq '') { $clean.RemoveAt(0) }
  while ($clean.Count -gt 0 -and $clean[$clean.Count - 1].Trim() -eq '') { $clean.RemoveAt($clean.Count - 1) }

  # Drop first line if it looks like a header/title line (common in transcripts)
  if ($clean.Count -gt 1) {
    $h = $clean[0].Trim()
    if ($h -match '\\s-\\s') { $clean.RemoveAt(0) }
  }

  # Collapse multiple blank lines
  $collapsed = New-Object System.Collections.Generic.List[string]
  $prevBlank = $false
  foreach ($l in $clean) {
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

  return $collapsed.ToArray()
}

$txts = Get-ChildItem -LiteralPath $Directory -File -Filter '*.wav.txt'
$updated = 0

foreach ($f in $txts) {
  $raw = Get-Content -LiteralPath $f.FullName -Raw

  $bak = ($f.FullName + '.before_clean.bak')
  if (!(Test-Path -LiteralPath $bak)) {
    Set-Content -LiteralPath $bak -Value $raw -Encoding utf8
  }

  $lyricsLines = Get-LyricsLinesFromSidecar -Text $raw
  $lyricsLines = Clean-LyricsLines -Lines $lyricsLines
  $lyrics = ($lyricsLines -join "`n")

  $captionOut = $Caption
  if (-not $OverwriteCaption) {
    # Preserve existing caption if present and non-empty
    foreach ($l in ($raw -split "`r?`n")) {
      if ($l -match '^\\s*caption\\s*:\\s*(.*)$') {
        $cap = $Matches[1].Trim()
        if ($cap) { $captionOut = $cap }
        break
      }
    }
  }

  $nl = [Environment]::NewLine
  $out = @(
    ('caption: ' + $captionOut),
    'genre: ',
    'bpm: ',
    'key: ',
    'signature: ',
    'is_instrumental: false',
    'lyrics:',
    $lyrics
  ) -join $nl

  Set-Content -LiteralPath $f.FullName -Value ($out + $nl) -Encoding utf8
  $updated++
}

Write-Host ('updated=' + $updated)
