<#
  Opure-VoxelEngine :: shader build
  Compiles shaders/*.usf to DXIL with DXC and emits a single self-contained C++
  header (generated/Opure_ShaderBlob.h) holding the bytecode inline. The runtime
  therefore never shells out to dxc and ships as a single executable.

  Root-constant ABI is shared with src/core/Opure_Dx12.h (kMaxRootConstants = 32).
#>
[CmdletBinding()]
param(
    [string]$DxcPath   = '',
    [string]$ShaderDir = '',
    [string]$OutFile   = '',
    [string]$Target    = 'cs_6_6'
)

$ErrorActionPreference = 'Stop'

# $MyInvocation.MyCommand.Path is not bound yet inside param(), so all path
# defaults are resolved here rather than in the parameter list.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($DxcPath))   { $DxcPath = 'dxc.exe' }
if ([string]::IsNullOrWhiteSpace($ShaderDir)) { $ShaderDir = Join-Path $scriptDir '..\shaders' }
if ([string]::IsNullOrWhiteSpace($OutFile))   { $OutFile = Join-Path $scriptDir '..\generated\Opure_ShaderBlob.h' }

# Resolve dxc: explicit path, PATH lookup, then common install locations.
function Resolve-Dxc {
    param([string]$Candidate)
    if ($Candidate -and (Test-Path $Candidate)) { return (Resolve-Path $Candidate).Path }
    $cmd = Get-Command $Candidate -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $fallbacks = @(
        (Get-ChildItem 'C:\VulkanSDK\*\Bin\dxc.exe' -ErrorAction SilentlyContinue | Select-Object -Last 1),
        (Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\bin\*\x64\dxc.exe' -ErrorAction SilentlyContinue | Select-Object -Last 1)
    )
    foreach ($f in $fallbacks) { if ($f) { return $f.FullName } }
    return $null
}

$dxc = Resolve-Dxc -Candidate $DxcPath
if (-not $dxc) {
    Write-Error @'
dxc.exe was not found.

Install the Vulkan SDK (which ships dxc) or the Windows SDK, or pass -DxcPath.
Without DXC there is no way to produce shader bytecode, so the build stops here.
'@
    exit 1
}

# ---------------------------------------------------------------------------
#  Manifest: source file -> list of entry points to compile.
#  A file can contribute more than one pipeline (the attention kernel is emitted
#  twice, once per head dimension, so both get a fully occupied thread group).
#  Keeping this explicit means a typo in an entry point is a hard error instead
#  of a silent "no such function" that only shows up as a garbage PSO at runtime.
# ---------------------------------------------------------------------------
$manifest = [ordered]@{
    'Opure_Debug'                 = @('DebugMain', 'DebugRootConstMain', 'DebugSrvMain')
    'Opure_Embedding'             = @('EmbeddingMain')
    'Opure_RMSNorm'               = @('RMSNormMain')
    'Opure_RoPE'                  = @('RoPEMain')
    'Opure_GEMV'                  = @('GEMVMain')
    'Opure_GEMV_Quant'            = @('GEMVQuantMain')
    'Opure_FlashAttention_Decode' = @(
        @{ Entry = 'AttentionMain32';  Defines = @('OPURE_ATT_THREADS=32')  },
        @{ Entry = 'AttentionMain64';  Defines = @('OPURE_ATT_THREADS=64')  },
        @{ Entry = 'AttentionMain96';  Defines = @('OPURE_ATT_THREADS=96')  },
        @{ Entry = 'AttentionMain128'; Defines = @('OPURE_ATT_THREADS=128') },
        @{ Entry = 'AttentionMain160'; Defines = @('OPURE_ATT_THREADS=160') },
        @{ Entry = 'AttentionMain192'; Defines = @('OPURE_ATT_THREADS=192') },
        @{ Entry = 'AttentionMain224'; Defines = @('OPURE_ATT_THREADS=224') },
        @{ Entry = 'AttentionMain256'; Defines = @('OPURE_ATT_THREADS=256') }
    )
    'Opure_SpeculativeDecode' = @(
        @{ Entry = 'SpeculativeVerifyMain32';  Defines = @('OPURE_ATT_THREADS=32')  },
        @{ Entry = 'SpeculativeVerifyMain64';  Defines = @('OPURE_ATT_THREADS=64')  },
        @{ Entry = 'SpeculativeVerifyMain96';  Defines = @('OPURE_ATT_THREADS=96')  },
        @{ Entry = 'SpeculativeVerifyMain128'; Defines = @('OPURE_ATT_THREADS=128') },
        @{ Entry = 'SpeculativeVerifyMain160'; Defines = @('OPURE_ATT_THREADS=160') },
        @{ Entry = 'SpeculativeVerifyMain192'; Defines = @('OPURE_ATT_THREADS=192') },
        @{ Entry = 'SpeculativeVerifyMain224'; Defines = @('OPURE_ATT_THREADS=224') },
        @{ Entry = 'SpeculativeVerifyMain256'; Defines = @('OPURE_ATT_THREADS=256') }
    )
    'Opure_SwiGLU'                = @('SwiGLUMain')
    'Opure_Add'                   = @('AddMain')
    'Opure_AddRes'                = @('AddResMain')
    'Opure_ArgMax'                = @('ArgMaxMain')
    'Opure_SignSGD_Update'        = @('SignSGDMain', 'CommitMain')
}

$ShaderDir = (Resolve-Path $ShaderDir).Path
$outDir    = Split-Path -Parent $OutFile
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

$tmp = Join-Path $env:TEMP ("opure_shader_" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$blobs = [ordered]@{}
$failures = 0

foreach ($name in $manifest.Keys) {
    $src = Join-Path $ShaderDir "$name.usf"
    if (-not (Test-Path $src)) {
        Write-Host ("  [MISS]  {0}.usf not found" -f $name) -ForegroundColor Red
        $failures++
        continue
    }

    foreach ($entry in $manifest[$name]) {
        # A manifest entry is either a bare entry-point name, or a hashtable of
        # @{ Entry = 'Name'; Defines = @('K=V', ...) }. The hashtable form exists
        # for entry points that are specialised by a preprocessor define -- the
        # attention kernel compiles once per supported head_dim, and hand-writing
        # eight near-identical bodies in the .usf was how the 64 and 128 variants
        # drifted apart in the first place.
        $defines = @()
        if ($entry -is [hashtable]) {
            $symbol  = $entry.Entry
            $defines = @($entry.Defines)
        }
        else {
            $symbol  = [string]$entry
        }
        $dxil = Join-Path $tmp "$symbol.dxil"

        # -enable-16bit-types : native fp16 math for the packed-precision paths.
        # -O3                 : decode is latency-critical; this is not optional.
        # -I                  : shaders include Opure_Common.hlsl.
        $args = @('-T', $Target, '-E', $symbol, '-enable-16bit-types', '-O3',
                  '-I', $ShaderDir)
        foreach ($d in $defines) { $args += @('-D', $d) }
        $args += @('-Fo', $dxil, $src)

        # Windows PowerShell surfaces native stderr as error records, which
        # ErrorActionPreference='Stop' would turn into a false failure on a
        # mere warning. Relax it for the call and judge by the exit code.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $output = & $dxc @args 2>&1
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $prev

        foreach ($line in $output) {
            if ("$line" -match 'error')   { Write-Host "        $line" -ForegroundColor Red }
            elseif ("$line" -match 'warning') { Write-Host "        $line" -ForegroundColor Yellow }
        }

        if ($exitCode -ne 0 -or -not (Test-Path $dxil)) {
            Write-Host ("  [FAIL]  {0}.usf::{1}" -f $name, $symbol) -ForegroundColor Red
            $failures++
            continue
        }

        $bytes = [System.IO.File]::ReadAllBytes($dxil)
        $blobs[$symbol] = $bytes
        Write-Host ("  [ok]    {0,-34} {1,7} bytes  <- {2}.usf::{3}" -f $symbol, $bytes.Length, $name, $symbol)
    }
}

Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue

if ($failures -gt 0) {
    Write-Host ''
    Write-Host "$failures shader(s) failed to compile." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
#  Emit the embedded blob header
# ---------------------------------------------------------------------------
$sb = [System.Text.StringBuilder]::new()
[void]$sb.AppendLine("// ============================================================================")
[void]$sb.AppendLine("//  Opure_ShaderBlob.h  -- GENERATED, do not edit.")
[void]$sb.AppendLine("//  Produced by scripts/build_shaders.ps1. Contains DXIL bytecode compiled")
[void]$sb.AppendLine("//  with DXC for target '$Target'.")
[void]$sb.AppendLine("// ============================================================================")
[void]$sb.AppendLine("#pragma once")
[void]$sb.AppendLine("#include <cstdint>")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("namespace opure {")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("struct ShaderBlob")
[void]$sb.AppendLine("{")
[void]$sb.AppendLine("    const uint8_t* code;")
[void]$sb.AppendLine("    uint32_t       size;")
[void]$sb.AppendLine("};")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("// Stable shader identifiers, used to index into ShaderBlobTable.")
[void]$sb.AppendLine("enum class ShaderId : uint32_t")
[void]$sb.AppendLine("{")

$idx = 0
foreach ($name in $blobs.Keys) {
    [void]$sb.AppendLine(("    {0} = {1}," -f $name, $idx))
    $idx++
}
[void]$sb.AppendLine("};")
[void]$sb.AppendLine("")

# Byte arrays
foreach ($name in $blobs.Keys) {
    $bytes = $blobs[$name]
    [void]$sb.AppendLine("inline constexpr uint8_t g_dxil_${name}[] = {")
    for ($i = 0; $i -lt $bytes.Length; $i += 16) {
        $chunk = $bytes[$i..([Math]::Min($i + 15, $bytes.Length - 1))]
        $hex = ($chunk | ForEach-Object { '0x{0:x2}' -f $_ }) -join ','
        if ($i + 16 -lt $bytes.Length) { $hex += ',' }
        [void]$sb.AppendLine("    $hex")
    }
    [void]$sb.AppendLine("};")
    [void]$sb.AppendLine("")
}

# Lookup table
[void]$sb.AppendLine("inline constexpr ShaderBlob kShaderTable[] = {")
foreach ($name in $blobs.Keys) {
    [void]$sb.AppendLine(("    {{ g_dxil_{0}, sizeof(g_dxil_{0}) }}," -f $name))
}
[void]$sb.AppendLine("};")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("inline constexpr uint32_t kShaderCount = sizeof(kShaderTable) / sizeof(kShaderTable[0]);")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("} // namespace opure")
[void]$sb.AppendLine("")

[System.IO.File]::WriteAllText($OutFile, $sb.ToString())

$total = ($blobs.Values | ForEach-Object { $_.Length } | Measure-Object -Sum).Sum
Write-Host ''
Write-Host ("  {0} shaders, {1:N0} bytes of DXIL -> {2}" -f $blobs.Count, $total, (Split-Path -Leaf $OutFile))