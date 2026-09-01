param(
  [Parameter(Mandatory = $true)]
  [string]$Directory,

  [string]$Trigger = "d5vm4h1"
)

$ErrorActionPreference = "Stop"

function Fix-Mojibake {
  param([string]$Text)

  if ($null -eq $Text) { return "" }

  # Common UTF-8 mis-decoding sequences seen in lyrics exports.
  # Use explicit char codes to avoid source-encoding issues in this script file.
  $t = $Text

  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x2122)), '’' # â€™
  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x02DC)), '‘' # â€˜
  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x0153)), '“' # â€œ
  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x009D)), '”' # â€�
  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x201C)), '”' # fallback variant
  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x201D)), '”' # fallback variant
  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x2013)), '–' # â€“ (variant)
  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x2014)), '—' # â€” (variant)
  $t = $t -replace ([string]::Concat([char]0x00E2, [char]0x20AC, [char]0x00A6)), '…' # â€¦

  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00A9)), 'é' # Ã©
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00A8)), 'è' # Ã¨
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00AA)), 'ê' # Ãª
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00AB)), 'ë' # Ã«
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00A1)), 'á' # Ã¡
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00A0)), 'à' # Ã 
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00A2)), 'â' # Ã¢
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00B6)), 'ö' # Ã¶
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00BC)), 'ü' # Ã¼
  $t = $t -replace ([string]::Concat([char]0x00C3, [char]0x00B1)), 'ñ' # Ã±
  return $t
}

function Parse-OptionA {
  param([string]$Text)

  $lines = $Text -split "`r?`n"
  $data = @{}
  $lyricsLines = @()
  $inLyrics = $false

  foreach ($l in $lines) {
    if ($inLyrics) {
      $lyricsLines += $l
      continue
    }

    if ($l -match '^\s*lyrics\s*:\s*$') {
      $inLyrics = $true
      continue
    }

    if ($l -match '^\s*([^:]+)\s*:\s*(.*)$') {
      $k = $Matches[1].Trim().ToLowerInvariant()
      $v = $Matches[2]
      $data[$k] = $v
    }
  }

  return @{
    data = $data
    lyrics = ($lyricsLines -join "`n").TrimEnd()
  }
}

function Write-OptionA {
  param(
    [string]$Path,
    [hashtable]$Data,
    [string]$Lyrics
  )

  function Get-Val {
    param([string]$Key, [string]$Default = "")
    if ($Data.ContainsKey($Key) -and $null -ne $Data[$Key]) { return [string]$Data[$Key] }
    return $Default
  }

  $nl = [Environment]::NewLine
  $lines = @(
    ("caption: " + (Get-Val -Key "caption" -Default "")),
    ("genre: " + (Get-Val -Key "genre" -Default "")),
    ("bpm: " + (Get-Val -Key "bpm" -Default "")),
    ("key: " + (Get-Val -Key "key" -Default "")),
    ("signature: " + (Get-Val -Key "signature" -Default "")),
    ("is_instrumental: " + (Get-Val -Key "is_instrumental" -Default "false")),
    ("custom_tag: " + (Get-Val -Key "custom_tag" -Default "")),
    "lyrics:",
    $Lyrics
  ) -join $nl

  Set-Content -LiteralPath $Path -Value ($lines + $nl) -Encoding utf8
}

$txts = Get-ChildItem -LiteralPath $Directory -File -Filter "*.wav.txt"
$updated = 0

foreach ($f in $txts) {
  $raw = Get-Content -LiteralPath $f.FullName -Raw
  $parsed = Parse-OptionA -Text $raw
  $data = $parsed.data
  $lyrics = $parsed.lyrics

  # Ensure caption exists (don’t change it if already set)
  if (-not ($data.ContainsKey("caption") -and $data["caption"].Trim())) {
    $data["caption"] = "acapella, dry vocals"
  }

  # Add/overwrite trigger word
  $data["custom_tag"] = $Trigger

  # Fix mojibake in lyrics only
  $lyricsFixed = Fix-Mojibake -Text $lyrics

  Write-OptionA -Path $f.FullName -Data $data -Lyrics $lyricsFixed
  $updated++
}

Write-Host ("updated=" + $updated)
