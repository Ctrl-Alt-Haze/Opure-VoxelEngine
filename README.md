# Opure-VoxelEngine

A native C++20 / Direct3D 12 LLM inference runtime with idle-time plasticity.
No Python, no PyTorch, no LibTorch, no CUDA driver runtime — weights are
structured VRAM allocations and all forward/inference work happens in DXIL
compute shaders.

The model actually runs. `opure_harness.exe` runs SmolLM-135M end to end on the
GPU and compares its logits against an independent scalar CPU reference, to
`max_abs = 3.0e-05`.

---

## STATUS: end-to-end inference VERIFIED

Everything below was observed on this machine, not assumed.

### Verified working

| Component | Evidence |
|---|---|
| Toolchain | MSVC 14.44.35207 (VS 2022 Build Tools), Windows SDK 10.0.26100, DXC 1.9.0.5399, CMake 4.4.3 |
| Device | `NVIDIA GeForce RTX 5070 Ti`, 15.62 GB VRAM |
| Upload/readback | `CreateCommittedResource` round-trip verified bit-exact |
| GPU timing | Timestamp query heap, frequency calibrated against `QueryPerformanceCounter` |
| Shaders | 16/16 `.usf` compile to DXIL for `cs_6_6 -enable-16bit-types -O3` (76,488 bytes) |
| Pipelines | All 16 compute PSOs created |
| Dispatch | Writes verified; `GetDeviceRemovedReason() == 0` after every dispatch |
| GEMV throughput | 576×576 in 0.0014 ms → **~958 GB/s effective** |
| `Opure_Streamer` | 272 tensors mapped zero-copy from a 513 MB file; every range cross-checked against `shape × dtype`; all 270 planned weight offsets verified against their tensor |
| `Opure_Tokenizer` | 49152-token byte-level BPE; round-trips CJK, accents, emoji-class bytes, digits and specials |
| `Opure_Graph` | config parse, layout plan, GQA/head_dim validation, VRAM admission gate |
| `Opure_VoxelEngine` | full 30-layer forward pass, paged KV cache, chunked prefill, single-token decode, greedy generation |
| Plasticity | `SignSGD_Update.usf` + `CommitMain`, verified against a CPU reference, decay-only and stepped |
| Build hygiene | `/W4 /WX /permissive-`, zero warnings, zero third-party deps |

### The model runs and agrees with the reference

```
end-to-end model:
  engine init                PASS  engine ready, prompt 1 tokens
  forward pass               PASS  deviceRemovedReason = 0x00000000
  plasticity (engine, q_proj) PASS  max_abs=0.000e+00
  KV V  <- v_proj            PASS  max_abs=0.000e+00
  attention  <- KV V         PASS  max_abs=0.000e+00
  KV K  <- rotated k         PASS  max_abs=0.000e+00
  logits vs CPU reference    PASS  max_abs=2.956e-05  rms=4.983e-06  (tol 2e-02)
  layer divergence           PASS  residual matches the reference through 6 layers
  semantic sanity            PASS  top prediction 28 "," (matches CPU reference)
                              8 tokens -> ", I'm sorry,"
```

48 checks, all green. The harness exits **0 only if every one passes**.

### Verification strategy

Three independent layers, because each catches a class the others cannot:

1. **Per-kernel parity.** Every shader is compared against an independent scalar
   CPU reference written in C++, so a self-consistent-but-wrong kernel cannot pass.
2. **Whole-model parity.** The 30-layer graph's logits are compared against a
   full CPU forward pass of the same weights, plus a per-layer residual trace
   that localises any divergence to a layer index.
3. **Structural gates.** Poison the activation banks, then step the dispatch
   budget through a layer one pass at a time and read the region each pass is
   supposed to write. Any pass that leaves its output unwritten is a failure.
   This is the check that finds a mis-specified GEMV, a wrong thread index, or a
   missing memory barrier — none of which a final-value comparison can see.

---

## Nine bugs that cost the most time

Recording these because they are all non-obvious and all cost real debugging.

### 1. `SV_GroupIndex` is the *thread* index, not the group index — twice

The GEMV and both attention kernels originally did `const uint o = gid;` where
`gid` was `SV_GroupIndex`. That silently collapses an entire matrix into its
diagonal: every thread in the group computes a *different* row, so only thread 0
writes an output and the result is a deterministic-looking wrong number.

The same mistake survived in the **embedding** kernel and, later, in
**SignSGD**. Both use `SV_GroupIndex` as an element/row index while being
dispatched as `ceil(n / 64)` or `ceil(n / 256)` groups — so every group rewrites
the same first 64 (or 256) words and the rest of the row stays whatever was in
the buffer. The residual stream was half real data and half uninitialised VRAM,
which is what made the model's logits differ between runs of the same binary.

Both are invisible for any element count a single group covers, so the rule is:
**a test for such a kernel must use more elements than one group holds.** The
SignSGD test uses 300 rows precisely so it cannot regress.

### 2. Root constants silently deliver bytes, not dwords

`SetComputeRoot32BitConstants` + `cbuffer : register(b0) { uint P[32]; }` compiled
and ran without error but every `P[]` read back as **zero**. Dumping the DXIL
showed why:

```
cbufferLoadLegacy.i32(i32 59, %handle, i32 0)   // P[0]
cbufferLoadLegacy.i32(i32 59, %handle, i32 1)   // P[1]  <- byte offset 1, not 4
cbufferLoadLegacy.i32(i32 59, %handle, i32 2)   // P[2]  <- byte offset 2, not 8
```

This runtime addresses a `uint` array with consecutive **byte** offsets. Since
SRV reads were independently proven working, parameters were moved to a
`StructuredBuffer<uint>` instead — see `Opure_Common.hlsl`.

**Do not "fix" this by going back to root constants.** The parameter block is a
deliberate workaround and is load-bearing.

### 3. Per-dispatch parameter slots are mandatory, not an optimisation

The GPU samples the parameter buffer at **execution** time, not record time. With
one shared slot, every dispatch in a command list sees only the last parameters
written. The fix offsets the root descriptor's GPU virtual address by a slot
stride — root descriptors are raw addresses, so this needs no descriptor heap:

```cpp
list_->SetComputeRootShaderResourceView(2 * kSlotCount + 2,
                                        dev_->ParamSlotVa(dev_->ParamSlot()));
```

`TestParamSlots` records two dispatches with different parameters into a single
command list and asserts they see different values. Without the per-dispatch
address, this test fails.

### 4. A chain of dependent dispatches needs an explicit UAV barrier

`CommandScope::Barrier()` existed from the start and was called from exactly zero
places. Every dispatch in the graph reads UAVs the previous dispatch wrote — the
RMSNorm writes a bank through its UAV and the next GEMV reads the same bank
through an SRV — with nothing ordering the two.

The symptom was `q_proj` returning `|q|` up to **3.1e4** for an input whose
maximum was `0.40`, and logits that changed between runs of the same binary: a
dot product against activation words the graph had never written. All dispatches
now go through one `GO()` helper that emits the barrier unconditionally.

### 5. Every rectangular GEMV had `inDim` and `outDim` swapped

The per-kernel parity test used square matrices, so passing the two dimensions
in the wrong order was invisible. In the graph, `k_proj`, `v_proj`, `gate_proj`,
`up_proj` and `down_proj` all had them backwards: the kernel then read `inDim-1`
activation words past the end of its input and wrote only `min(inDim, outDim)`
outputs. `down_proj` produced a maximum of 53.6 where ~2 was expected.

**A parity test that only covers the square case is not testing the contract.**
The engine now states the expected shapes next to each call.

### 6. The residual add was reading the bank it was writing

`rb = LB.rb1;` was assigned *before* the `AddRes` parameters were built, so the
add's source bank was the destination bank — both an SRV/UAV alias and a read of
uninitialised memory. The layer-0 residual was off by `5e1` and grew to `1.8e17`
by layer 5. Capture the source before advancing the destination.

### 7. RoPE wrote the KV cache at the activation stride

Inside its key/V block RoPE uses two different offsets: `base = kvHead*headDim +
pair*2` for reading the packed QKV activation bank, and `inHead = pair*2` for
the channel offset *inside* one cache slot. It used `base` for both, so heads 1
and 2 landed past the end of their own slot and in the gap before the next head.
Attention then read an unwritten slot and was wrong for 2 of every 3 heads.

The tell was that the first 64 cached words matched and the rest did not.

### 8. `ExecuteAndWait()` on an already-closed command list kills the device

`Device::ExecuteAndWait` closed the list only `if (listOpen_)` but then submitted
it unconditionally. `UploadToBuffer` already ends in its own submit, so the extra
call handed the queue a closed command list; D3D12 reports this as
`DXGI_ERROR_DRIVER_INTERNAL_ERROR` plus a hang, and `ExecuteCommandLists` returns
`void`, so there is no HRESULT to check. A redundant `ExecuteAndWait` is now a
no-op.

### 9. The GPT-2 byte alphabet must be indexed by byte, not codepoint

`bytes_to_unicode` assigns non-printable bytes to codepoints starting at
U+0100. The obvious guard — "skip codepoints already used" — is a trap,
because the `taken[]` bitmap is indexed by **byte value** while the candidate is
a **codepoint**. Probing `taken[next - 0x100]` silently shifts every multi-byte
character (em-dash, CJK) onto a wrong codepoint, and the tokenizer then decodes
them as replacement characters while every ASCII test still passes.

The correct form needs no collision check at all: printable ranges are all below
0x100 and the assigned codepoints start at 0x100 and only increase.

---

## Two more structural traps

**The bank schedule must satisfy liveness, not just SRV/UAV aliasing.** The
residual stream `x` has to survive the whole layer, because both the opening
RMSNorm and the attention residual add read it. A schedule that avoids aliasing
but writes `x`'s bank mid-layer destroys `x` and produces an all-zero output —
which is exactly what `gate_proj` did in an earlier revision. The schedule in
`MakeLayerBanks` was **solved by enumerating assignments** under both constraints
and is asserted at every layer in `RecordForward`, because hand-reasoning it
already produced the invalid bank index `-1` once.

**Three banks, not two.** With a ping-pong pair, `o_proj` must read the
attention output's bank and therefore writes the opposite one — which is exactly
where the residual already lives, so the add would have to read and write the
same resource. A third bank breaks the cycle; the residual rotates and each add
reads two distinct banks and writes the third.

---

## SDK divergences on this machine

Every one of these was found empirically and is documented in
`src/core/Opure_Dx12.cpp`.

| Area | Divergence |
|---|---|
| `CreateHeap` | Needs a `D3D12_HEAP_DESC` wrapper |
| Heap sizes | `SizeInBytes == 0` is rejected |
| Buffer views | Require `StructureByteStride == 4`; stride 0/2 and RAW both remove the device with `DXGI_ERROR_INVALID_CALL` |
| Root descriptors | Bound by **GPU virtual address**, not a descriptor |
| Shader model | `D3D_SHADER_MODEL_6_6` |
| Root signature | 12 root parameters: 5 UAVs, 6 SRVs, 1 parameter SRV |
| Barriers | Explicit `ResourceBarrier` required for UAV transitions — see bug 4 |
| `CheckFeatureSupport` | Fails for `D3D12_FEATURE_SHADER_MODEL` on this runtime; downgraded to a warning |
| `CreateCommandList1` | On Device4 there is no allocator parameter |
| `GetDeviceRemovedReason` | No `ID3D12Debug` available, so this is the only device-health signal |

Other hard-won notes:

- **Never pre-reserve heaps.** Committed resources directly from heap properties;
  pre-reservation caused `DXGI_ERROR_DEVICE_REMOVED`.
- `dot4addI8` is absent in this DXC — the INT8 GEMV uses manual `asint` plus
  arithmetic-shift sign extension.
- `SV_GroupIndex` is a scalar `uint`, not `uint3`.
- This DXC **mis-compiles dynamic groupshared indexing inside `[unroll]` loops**.
  All wave sums are written out explicitly instead of looped.
- `h` is a CUDA type, not HLSL — use `mad()`, not `fmaf()`.
- `$PSScriptRoot` / `$MyInvocation.MyCommand.Path` are null inside `param()`
  defaults in both PowerShell scripts; defaults must resolve in the body.
- PowerShell's `ConvertFrom-Json` cannot parse `tokenizer.json`, whose vocabulary
  contains both `A` and `a`; the project's own parser is case-sensitive and
  tolerates duplicate keys.

---

## Memory layout

Five persistent buffers. Every tensor is a 32-bit **word** range inside one of
them; ranges are passed through the parameter block, so nothing needs descriptor
churn as the model grows.

| Slot | SRV / UAV | Resource | Contents |
|---|---|---|---|
| 0 | `t0` / `u0` | weights | all model weights, safetensors layout |
| 1 | `t1` / `u1` | bank A | activation scratch + plasticity staging tail |
| 2 | `t2` / `u2` | bank B | activation scratch + plasticity staging tail |
| 3 | `t3` / `u3` | kv | KV cache, `[2][layer][kvHead][seq][dim]`, plus a 2-word ArgMax tail |
| 4 | `t4` / `u4` | bank C | residual stream, third bank, logits |

**Bank discipline.** No dispatch may bind one resource as both SRV and UAV, and
no pass may write the bank the live residual is in. Both are asserted at run
time, per layer, rather than left to a comment.

**The ArgMax result needs a tail.** Its two result words go in the KV buffer past
the end of the cache proper. Without that reservation the argmax lands on the last
real cache entry and corrupts attention for the next token.

**Plasticity staging.** `SignSGD` reads `O_W` and would write `O_WU`; that is one
resource bound as both SRV and UAV, which is exactly what the bank rules forbid.
The update is instead computed into the tail of an activation bank — scratch the
inference graph never reads — and copied back by a second dispatch. Both
dispatches bind exactly one UAV, and the net effect on the weights is an in-place,
state-free update with no optimiser state and no extra allocation.

---

## Design decisions

- **Weights keep their natural safetensors `[out, in]` row-major layout.** No
  transpose at load time — the file streams straight into VRAM, and the GEMV
  reads consecutive `in` elements within a wave, which is fully coalesced.
- **One 256-thread group per output row** for GEMV. A one-wave-per-row design
  would yield only 1536 warps for a 49152-wide LM head and be bandwidth-starved;
  a full group per row saturates memory during autoregressive decode.
- **One token per command list** for decode; the CPU waits once.
- **Decode computes only the new token.** Re-running the whole sequence per
  generated token is quadratic *and* exhausted the parameter slots after three
  tokens of output. The KV cache exists precisely so a decode step is O(1)
  passes. Prefill is chunked so a long prompt cannot overflow the slot budget.
- **Verification is a CPU reference, not a downloaded one**, plus a per-layer
  residual trace and the poison-and-walk structural gate.

---

## Phase 2 (Qwen 30B/32B) is not physically possible on this hardware

This is a hardware limit, not a scheduling problem:

| Precision | Size | Fits in 15.62 GB VRAM? |
|---|---|---|
| BF16 | ~65 GB | No |
| INT8 | ~33 GB | No |
| INT4 | ~17 GB | No |

Host RAM is 32 GB, and ~3.4 GB of VRAM is already consumed by the compositor. The
INT8 GEMV kernel (`Opure_GEMV_Quant.usf`) is written and compiles, but a 32B
model cannot be resident on this GPU. Phase 2 would need different hardware
(≥48 GB VRAM for INT8, or multi-GPU tensor parallelism across NVLink).

---

## Build and run

MSVC is not on `PATH` by default. Prepend the toolchain first.

```powershell
cd Opure-VoxelEngine
$env:Path = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;" + $env:Path

# 1. Compile shaders -> build\generated\Opure_ShaderBlob.h  (must precede the build)
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_shaders.ps1 `
    -DxcPath 'C:\VulkanSDK\1.4.357.0\Bin\dxc.exe' `
    -OutFile '.\build\generated\Opure_ShaderBlob.h'

# 2. Configure (once)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DOPURE_DXC_PATH="C:/VulkanSDK/1.4.357.0/Bin/dxc.exe"

# 3. Build
cmake --build build --config Release --target opure_harness

# 4. Run the gate
.\build\Release\opure_harness.exe
$LASTEXITCODE   # 0 = every check passed
```

Fetch the Phase 1 model (curl only, resumable, no Python):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\fetch_model.ps1
```

---

## Not yet implemented

- **Unreal Engine integration.** `ue5/OpureVoxelEngine.Build.cs` is authored but
  has never been compiled, because Unreal Engine is not installed on this
  machine. Everything it would link against — `OpureVoxelEngine`, `OpureStreamer`
  and `OpureTokenizer` — builds and passes here.
- **Multi-token prefill batching.** Prefill is chunked to respect the parameter
  slot budget, but every token is still computed by its own full 30-layer pass.
  A real GEMM prefill would collapse that into a handful of batched launches.
- **KV cache eviction.** The cache is sized for `max_position_embeddings` and
  never trimmed; a conversation longer than the window will not fit.
- **INT8 / INT4 GEMV is compiled but not wired in.** `Opure_GEMV_Quant.usf` has
  no per-dispatch caller, and the weight buffer is laid out for F32.
