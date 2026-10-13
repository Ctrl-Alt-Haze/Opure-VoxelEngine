<#
  Opure-VoxelEngine :: model fetcher
  Downloads a HuggingFace safetensors model + tokenizer directly into models/.
  Pure PowerShell + curl.exe: no python, no pip, no torch.

  Notes on robustness, learned the hard way against the HF CDN:
    * A browser-ish User-Agent is sent; the default curl UA is sometimes
      rejected outright.
    * huggingface.co returns transient 401/429 when requests arrive in a burst,
      so the metadata query is best-effort and the file downloads retry hard.
    * Downloads resume (-C -), so an interrupted 500 MB fetch is not restarted.
#>
[CmdletBinding()]
param(
    [string]$Repo = 'HuggingFaceTB/SmolLM-135M',
    [string]$OutDir = '',
    [string]$WeightsFile = 'model.safetensors',
    [switch]$SkipWeights
)

$ErrorActionPreference = 'Stop'
$Script:UA = 'Opure-VoxelEngine/1.0 (native C++20/DX12 LLM runtime)'

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    # $PSScriptRoot is not bound yet inside param(), so resolve the default here.
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $OutDir = Join-Path (Split-Path -Parent $scriptDir) 'models'
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Test-RetryableHttp {
    param([string]$Code, [int]$Attempt, [int]$MaxAttempts, [string]$Label)
    $ok = $Code -match '^(200|206)$'
    if (-not $ok -and $Attempt -lt $MaxAttempts) {
        $backoff = [Math]::Min($Attempt * 3, 20)
        Write-Host ("           {0}: http={1}, retry {2}/{3} in {4}s" -f $Label, $Code, $Attempt, $MaxAttempts, $backoff)
        Start-Sleep -Seconds $backoff
    }
    return $ok
}

function Get-HFFile {
    param([string]$Repo, [string]$File, [string]$OutDir, [switch]$Optional)

    $name = Split-Path $File -Leaf
    $dest = Join-Path $OutDir $name

    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0)) {
        Write-Host ("  [cached] {0,-30} {1,14:N0} bytes" -f $name, (Get-Item $dest).Length)
        return $true
    }

    $url = "https://huggingface.co/$Repo/resolve/main/$File"
    $tmp = "$dest.part"
    $max = 6

    for ($i = 1; $i -le $max; $i++) {
        Write-Host ("  [get]    {0}  (attempt {1}/{2})" -f $name, $i, $max)

        # -C - resumes a partial .part file; --retry-all-errors covers transient
        # TLS/connection resets as well as HTTP error codes.
        & curl.exe -L --fail --silent --show-error `
            --retry 4 --retry-delay 3 --retry-all-errors `
            -A $Script:UA -C - -o $tmp $url 2>$null
        $rc = $LASTEXITCODE

        if ($rc -eq 0 -and (Test-Path $tmp) -and (Get-Item $tmp).Length -gt 0) {
            Move-Item -Force $tmp $dest
            Write-Host ("  [ok]     {0,-30} {1,14:N0} bytes" -f $name, (Get-Item $dest).Length)
            return $true
        }

        if ($i -eq $max) {
            if (Test-Path $tmp) { Remove-Item -Force $tmp }
            if ($Optional) {
                Write-Host ("  [skip]   {0,-30} (not available in repo)" -f $name)
                return $false
            }
            throw "failed to download $url (curl exit $rc)"
        }
        Start-Sleep -Seconds ([Math]::Min($i * 3, 20))
    }
    return $false
}

Write-Host "Opure-VoxelEngine model fetch"
Write-Host ("  repo   : {0}" -f $Repo)
Write-Host ("  dest   : {0}"  -f $OutDir)
Write-Host ""

# --- Best-effort metadata probe -------------------------------------------
$available = @{}
$api = "https://huggingface.co/api/models/$Repo?blobs=true"
for ($i = 1; $i -le 4 -and $available.Count -eq 0; $i++) {
    $metaJson = & curl.exe -L --fail --silent -A $Script:UA $api 2>$null
    if ($LASTEXITCODE -eq 0 -and $metaJson) {
        try {
            $meta = $metaJson | ConvertFrom-Json
            foreach ($s in $meta.siblings) { $available[$s.rfilename] = $s.size }
        } catch {
            Write-Host ("  [warn]   HF API returned unparsable JSON (attempt {0})" -f $i)
        }
    } else {
        Write-Host ("  [warn]   HF API unavailable (attempt {0}); proceeding blind" -f $i)
        Start-Sleep -Seconds ($i * 3)
    }
}
if ($available.Count -gt 0) {
    Write-Host ("  [meta]   {0} files listed in repo" -f $available.Count)
}

function Test-Listed([string]$f) {
    if ($available.Count -eq 0) { return $true }   # metadata unavailable: try anyway
    return $available.ContainsKey($f)
}

# --- config.json is mandatory ---------------------------------------------
if (-not (Test-Listed 'config.json')) { throw "repo '$Repo' has no config.json" }
Get-HFFile -Repo $Repo -File 'config.json' -OutDir $OutDir | Out-Null

# --- tokenizer side-files --------------------------------------------------
foreach ($f in @('tokenizer.json', 'generation_config.json', 'special_tokens_map.json')) {
    if (Test-Listed $f) { Get-HFFile -Repo $Repo -File $f -OutDir $OutDir -Optional | Out-Null }
}

# --- weights ---------------------------------------------------------------
if (-not $SkipWeights) {
    if (Test-Listed $WeightsFile) {
        Get-HFFile -Repo $Repo -File $WeightsFile -OutDir $OutDir | Out-Null
    } else {
        throw "weights file '$WeightsFile' not present in repo '$Repo'"
    }
}

Write-Host ""
Write-Host ("Done. Contents of {0}:" -f $OutDir)
$total = 0L
Get-ChildItem $OutDir | Sort-Object Name | ForEach-Object {
    Write-Host ("  {0,-32} {1,16:N0} bytes" -f $_.Name, $_.Length)
    $total += $_.Length
}
Write-Host ("  {0,-32} {1,16:N0} bytes" -f 'TOTAL', $total)