# Inference runtime for a C++ OBS plugin on Windows and macOS

Resolves [#5](https://github.com/kreuzhofer/obs-ai-director/issues/5). Researched 2026-10-13 against primary sources (ONNX Runtime repo/docs/releases, Google AI Edge docs, Apple developer docs, Microsoft Learn, and the source of existing OBS plugins). Versions quoted are the latest as of that date.

## Recommendation

**Embed ONNX Runtime (ORT), built from source in CI at a pinned tag, with telemetry compiled out, statically linked into the plugin.**

- **macOS (arm64 only):** ORT + **CoreML EP** (`MLProgram` format, `MLComputeUnits=ALL` or `CPUAndNeuralEngine`), with the CPU EP as fallback.
- **Windows x64:** ORT **CPU EP** as the baseline that must always work. **DirectML EP** is an optional GPU path, gated behind a setting, with `DirectML.dll` loaded from the plugin's own directory. Don't ship the CUDA/TensorRT EPs in v1: they add hundreds of MB, and a face/head-pose model doesn't need them.
- **Model format:** `.onnx` files, downloaded separately. One runtime and one model format serve face/head pose now, and gestures and voice-adjacent models (VAD) later.
- **Don't** use MediaPipe C++ as the runtime. Its *models* (BlazeFace, Face Landmarker, Hand Landmarker) stay usable as ONNX conversions; obs-backgroundremoval already ships `mediapipe.onnx`.
- **Runner-up:** LiteRT 2.x (the new TFLite). Pick it only if the model ticket lands on `.tflite`-only models that don't convert cleanly to ONNX.

Two non-obvious constraints drive the "build it yourself" part:

1. **Official ORT binaries now phone home.** Since ORT added POSIX 1DS telemetry (July 2026) and made 1DS the default for native builds (ORT 1.31, Oct 2026), the official macOS and Windows release binaries contain an HTTPS uploader to `mobile.events.data.microsoft.com`. That conflicts with the project rule "no network at runtime, no telemetry". A direct CMake build of ORT defaults to `onnxruntime_USE_TELEMETRY=OFF`.
2. **Shared DLLs collide inside `obs-plugins/64bit`.** obs-backgroundremoval's maintainers removed their DirectML backend because shipping `DirectML.dll` with the plugin was "the main cause of incompatibility errors". They now statically link a source-built ORT on Windows.

## Comparison

| | ONNX Runtime | MediaPipe (C++ Framework / Tasks) | LiteRT 2.x (ex-TFLite) | Native per-OS (Apple Vision / Core ML; Windows ML) |
|---|---|---|---|---|
| **macOS acceleration** | CoreML EP (ANE/GPU/CPU). In 1.31 it joins device-based EP selection (`PREFER_NPU` → ANE) | **No desktop GPU on macOS** (CPU only) | Metal GPU accelerator (prebuilt macOS arm64); ANE/CoreML not listed | Best possible: Core ML / Vision use ANE+GPU directly |
| **Windows acceleration** | CPU (MLAS); DirectML EP (any DX12 GPU, "sustained engineering"); CUDA / TensorRT-RTX as separate plugin EPs or via Windows ML | "Running MediaPipe on Windows is experimental"; docs build with GPU disabled | WebGPU (Direct3D) GPU accelerator; Intel/Qualcomm NPU | Windows ML = Microsoft-maintained ORT with EPs delivered by Windows Update (Windows App SDK) |
| **Binary size (per platform)** | Prebuilt shared: macOS arm64 `libonnxruntime` 44.8 MB; Win CPU `onnxruntime.dll` ~17 MB (+18.5 MB `DirectML.dll` for DML); CUDA package 300-390 MB zipped. Static and reduced-op builds can be much smaller | Large Bazel monorepo build; no supported prebuilt C++ libs | macOS `CLiteRT_mac.xcframework.zip` 12.6 MB (compressed) | ~0 (OS frameworks) |
| **Build/packaging in an OBS plugin** | CMake. Proven in several OBS plugins (static via vcpkg, or prebuilt via FetchContent). Requires care with telemetry, DLL placement and rpaths | Bazel only; doesn't fit obs-plugintemplate's CMake; C++ isn't a Tasks platform | CMake SDK (`litert_cc_api`), prebuilt `libLiteRt` for Win x64 + macOS arm64; no OBS prior art found | Two codebases and two model formats; Windows ML targets packaged/unpackaged *apps*, and an OBS plugin doesn't own the host process |
| **Prior art in OBS plugins** | obs-backgroundremoval (Windows/Linux), obs-localvocal, obs-detect | None as runtime; models reused as ONNX | None found | obs-backgroundremoval on macOS (Core ML, since Aug 2026) |
| **Fit for later voice/gesture** | Same runtime runs hand-landmark ONNX and VAD models; ASR would likely be whisper.cpp (as obs-localvocal does), independent of this choice | Has Gesture Recognizer / Hand Landmarker / Audio Classifier tasks, but not in C++ on desktop | Runs MediaPipe `.tflite` models natively | Per-OS again |
| **Verdict** | **Use** | Reject as runtime | Runner-up | Reject as primary; Apple Vision is a useful baseline for the prototype |

## Evidence

### ONNX Runtime

**Release state.** The latest release is v1.31.0 (2026-10-09). Release assets: `onnxruntime-osx-arm64-1.31.0.tgz` (36.5 MB), `onnxruntime-win-x64-1.31.0.zip` (82.9 MB), `onnxruntime-win-x64-gpu_cuda12/13` (393 / 309 MB). There is no macOS x86_64 or universal2 package. ([release v1.31.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0))
- Since 1.24: "x86_64 binaries for macOS/iOS are no longer provided and minimum macOS is raised to 14.0". 1.23.0 was the last release with `osx-x86_64` / `osx-universal2` assets. ([v1.24.1 notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1), [v1.23.0 assets](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0)) obs-localvocal pins 1.20.1 for this reason. Its CMake comment reads "v1.20.1 is the last version to support MacOS 12 v1.23.0 is the last version to support x86_64 on MacOS". ([FetchOnnxruntime.cmake](https://github.com/locaal-ai/obs-localvocal/blob/master/cmake/FetchOnnxruntime.cmake))
- Measured from the v1.31.0 macOS tarball: `libonnxruntime.1.31.0.dylib` is 44,830,288 bytes and links `CoreML.framework` and `Metal.framework`, so the CoreML EP is built in.

**Telemetry: the most important finding.**
- [Privacy.md](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md): "for other Windows builds including our official releases, supported Linux architectures, macOS, Android, and iOS, it uses the cross-platform 1DS telemetry SDK … Telemetry is turned **ON** by default in the official builds." It sends events "to Microsoft's telemetry backend over HTTPS".
- [PR #27379 "Add POSIX telemetry"](https://github.com/microsoft/onnxruntime/pull/27379) (merged 2026-07-24) added 1DS to macOS and Linux. [PR #32384 "Default native builds to 1DS"](https://github.com/microsoft/onnxruntime/pull/32384) (merged 2026-10-08, in 1.31) made it the default backend for supported native builds, including Windows.
- Verified locally: `strings` on the official macOS 1.31.0 dylib contains `https://mobile.events.data.microsoft.com/OneCollector/1.0` and `telemetry_1ds.cc`. `otool -L` shows it links `CFNetwork` and `Network.framework`.
- Opt-outs per Privacy.md: build with `--no_telemetry`, or set `ORT_DISABLE_TELEMETRY=1` "before ONNX Runtime initializes", or call the C API to suppress events. The API route comes with a caveat: "ONNX Runtime may already have emitted a minimal initialization event before the API can be called."
- In ORT's [`cmake/CMakeLists.txt`](https://github.com/microsoft/onnxruntime/blob/main/cmake/CMakeLists.txt), `option(onnxruntime_USE_TELEMETRY "Build with Telemetry" OFF)`, so a **direct CMake build** (the way obs-backgroundremoval builds it) has no telemetry. The `build.py` driver defaults it on.
- Consequence: shipping Microsoft's prebuilt binaries would violate "no network at runtime, no telemetry" unless the plugin sets the env var for the whole OBS process before ORT loads. That's fragile, and it affects other plugins in the same process. Build from source instead.

**CoreML EP** ([docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)):
- Requires macOS 10.15+. `MLProgram` format requires macOS 12+.
- `MLComputeUnits`: `CPUOnly | CPUAndNeuralEngine | CPUAndGPU | ALL`.
- `ModelCacheDirectory` caches compiled models, but "ONNX Runtime does NOT have a mechanism to track model changes and does not delete the cache entries". The plugin must version its cache key when downloaded weights change.
- `RequireStaticInputShapes`: dynamic shapes are allowed but slower. Export models with static input shapes.
- 1.31 adds CoreML to device-based EP selection: "NPU selection requires … macOS 13 or later" ([v1.31.0 notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.31.0), [#31975](https://github.com/microsoft/onnxruntime/pull/31975)).

**DirectML EP** ([docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)):
- "Note: DirectML is in sustained engineering." New work has moved to Windows ML.
- Requires a DirectX 12 device and Windows 10 1903+.
- The session must use `ORT_SEQUENTIAL` with `enable_mem_pattern=false`, and "only one thread may call `Run` at a time" per session.
- It still receives fixes in 1.31 (DML upload-heap validation, Resize ROI).
- On NuGet, the newest `Microsoft.ML.OnnxRuntime.DirectML` is 1.24.4 (published 2026-03-17), while core ORT is at 1.31. Checked via the NuGet flat-container API on 2026-10-13. Inspected sizes: `onnxruntime.dll` 17.3 MB in that package, and `DirectML.dll` (x64) 18.5 MB in `Microsoft.AI.DirectML` 1.15.4. To track newer ORT with DML you build ORT with `onnxruntime_USE_DML=ON` (CMake option, default OFF).

**CUDA / WebGPU.**
- Since 1.25 the CUDA EP ships as a separately released **plugin EP** ([v1.25.0 notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.25.0), [plugin-ep-cuda/v0.3.0](https://github.com/microsoft/onnxruntime/releases/tag/plugin-ep-cuda/v0.3.0)). Core and plugin should be built "from the same ONNX Runtime revision" (1.31 notes).
- A WebGPU plugin EP (Dawn: D3D12 on Windows, Metal on macOS) is also released separately ([plugin-ep-webgpu/v0.4.0](https://github.com/microsoft/onnxruntime/releases/tag/plugin-ep-webgpu/v0.4.0)). It ships no prebuilt binary assets, and its notes focus on LLM kernels.
- Both are overkill for v1. WebGPU is the most interesting future "one GPU path on both OSes" option.

**Size reduction** ([custom build docs](https://onnxruntime.ai/docs/build/custom.html)):
- `--include_ops_by_config` keeps only the kernels our models need.
- An *extended* minimal build supports CoreML ("EPs that compile kernels such as NNAPI and CoreML"). A *basic* minimal build requires ORT-format models and only statically-registered EPs (CPU).
- `MinSizeRel` gives the smallest binary.

### Prior art: OBS plugins

**obs-backgroundremoval** ([repo](https://github.com/royshil/obs-backgroundremoval), GPL-3.0, active):
- `buildspec.props` pins `onnxruntime_git_tag=v1.28.0`. ORT is a git submodule (`vendor/onnxruntime`).
- On Windows, CI configures ORT with plain CMake: `-Donnxruntime_BUILD_SHARED_LIB=OFF -Donnxruntime_USE_VCPKG=ON …`, using a vcpkg triplet with `VCPKG_LIBRARY_LINKAGE static` and `VCPKG_CRT_LINKAGE dynamic` (`x64-windows-static-md-obs-ort`). The result is linked into the plugin DLL ([build-windows.yml](https://github.com/royshil/obs-backgroundremoval/blob/main/.github/workflows/build-windows.yml), [triplet](https://github.com/royshil/obs-backgroundremoval/blob/main/vcpkg-triplets/x64-windows-static-md-obs-ort.cmake)).
- No DML/CUDA flags are passed on Windows, so it currently runs **CPU only** there. The CUDA/ROCm EPs are only probed on Linux ([CMakeLists.txt](https://github.com/royshil/obs-backgroundremoval/blob/main/CMakeLists.txt), [ort-session-utils.cpp](https://github.com/royshil/obs-backgroundremoval/blob/main/src/ort-utils/ort-session-utils.cpp)).
- Why they dropped DirectML: maintainer in [#655](https://github.com/royshil/obs-backgroundremoval/issues/655): "We know that including DirectML.dll in the plugin package is the main cause of incompatibility errors. This is why we removed the DirectML backend." A user fork re-added DML for Windows 11 + ORT 1.28. Related: [#759](https://github.com/royshil/obs-backgroundremoval/issues/759), [#724](https://github.com/royshil/obs-backgroundremoval/issues/724).
- **macOS dropped ORT entirely** in [PR #968 "Replace backend on macOS"](https://github.com/royshil/obs-backgroundremoval/pull/968) (merged 2026-08-31): "Replaces the macOS background-removal backend with a Core ML implementation using the MediaPipe Selfie Segmentation landscape model … Removes the macOS dependency on ONNX Runtime and OpenCV." The model is shipped as a compiled `.mlpackage` via the Xcode generator.
- Stated direction ([#984](https://github.com/royshil/obs-backgroundremoval/issues/984)): "For macOS, CoreML should always be used … For Windows, DirectML should always be used. It is the only way to get acceptable compatibility between Windows systems." They also plan an ncnn + Vulkan backend for Linux .deb packages. A Dawn/WebGPU idea ([#963](https://github.com/royshil/obs-backgroundremoval/issues/963)) was folded into #984.
- README: "On Windows, we plan to support WinML acceleration."
- Release sizes with bundled models: `windows-x64.zip` 49.8 MB, `macos-universal.pkg` 53.6 MB ([v1.4.1](https://github.com/royshil/obs-backgroundremoval/releases/tag/1.4.1)).

**obs-localvocal / obs-detect** (locaal-ai):
- Both use FetchContent on Microsoft's **prebuilt** ORT archives.
- On Windows they install `onnxruntime.dll` and `onnxruntime_providers_shared.dll` straight into `obs-plugins/64bit`, the directory shared by every plugin. On macOS they bundle the dylib into the plugin's `Frameworks` and rewrite the install name ([FetchOnnxruntime.cmake](https://github.com/locaal-ai/obs-localvocal/blob/master/cmake/FetchOnnxruntime.cmake)).
- This is the simplest route. It is also the DLL-collision pattern to avoid, and with ORT ≥ the July/Oct 2026 telemetry changes it would pull in 1DS.

**obs-face-tracker** ([repo](https://github.com/norihiro/obs-face-tracker), GPL-2.0) uses statically linked dlib (HOG/CNN face detection) and no NN runtime. It's prior art for face tracking in OBS, but not for head pose with modern models.

### MediaPipe

- The Tasks API is offered for **Android, Web, Python, iOS**. C++ appears only under "MediaPipe Framework" ([solutions guide](https://developers.google.com/edge/mediapipe/solutions/guide)).
- Framework install docs ([install](https://developers.google.com/edge/mediapipe/framework/getting_started/install)): "Running MediaPipe on Windows is experimental". On macOS, "desktop GPU is currently not supported". Desktop GPU is Linux-only with EGL. The build is Bazel (Windows: "Bazel 6.5.0 or higher").
- Relevant tasks for later: Face Landmarker, Hand Landmarker, Gesture Recognizer, Audio Classifier (no speech recognition task).
- Conclusion: MediaPipe is the wrong *runtime* for a CMake OBS plugin on Windows and macOS, but a good *model source*.

### LiteRT (formerly TensorFlow Lite)

- [Overview](https://developers.google.com/edge/litert/overview): the C++ `CompiledModel` API is the recommended path, built with CMake.
- Accelerators: macOS = XNNPACK CPU + Metal/WebGPU GPU. Windows = XNNPACK + WebGPU/OpenCL GPU + Intel/Qualcomm NPU.
- [C++ SDK](https://developers.google.com/edge/litert/next/cpp_sdk) v2.3.0: prebuilt `libLiteRt` for Windows x86_64 (GPU via WebGPU/Direct3D) and macOS arm64 (GPU via Metal).
- GitHub [v2.3.0](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.3.0) (2026-10-07) assets: `litert_cc_sdk.zip` (0.3 MB headers/wrappers) and `CLiteRT_mac.xcframework.zip` (12.6 MB).
- Pros: smaller than ORT, runs MediaPipe `.tflite` models with no conversion, one GPU story on both OSes.
- Cons: no Core ML/ANE path listed for macOS, no OBS prior art found, and the 2.x C++ SDK is young. I didn't verify whether LiteRT has its own telemetry.

### Native per-OS alternatives

- **Apple Vision**: `VNFaceObservation` exposes head pose directly. `yaw` since macOS 10.14 and `pitch` since macOS 12.0 ([yaw](https://developer.apple.com/documentation/vision/vnfaceobservation/yaw), [pitch](https://developer.apple.com/documentation/vision/vnfaceobservation/pitch)); `roll` similarly. That gives a zero-download head-pose source on macOS. It has no Windows counterpart, so it would create asymmetric Attention behaviour. Use it as a benchmark or baseline in the prototype, not as the v1 design.
- **Windows ML** ([overview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)): "the Windows-supported and maintained copy of ONNX Runtime", with "execution providers that Windows installs and keeps up to date via Windows Update". CPU and DirectML GPU work on all supported versions. Hardware-specific EPs (e.g. NVIDIA) need Windows 11 24H2+. It ships via the Windows App SDK, framework-dependent or self-contained. Code stays ORT-API-compatible, so it's a later swap-in on Windows, not a separate design. Open question: how it behaves inside a third-party host process (obs64.exe) that doesn't initialise the Windows App SDK.

## Implications and open questions for the map

- **Telemetry and network rule:** build ORT ourselves with `onnxruntime_USE_TELEMETRY=OFF` (or `--no_telemetry`). Add a CI check that greps the built binary for `events.data.microsoft.com`.
- **macOS is arm64-only** if we track current ORT (≥1.24, macOS ≥14). Intel Macs would need ORT ≤1.23 or a CPU-only universal build. This feeds the packaging ticket.
- **DirectML.dll placement:** if DML ships, load it by full path from the plugin's data dir (or delay-load) and keep the CPU EP as an always-working fallback. Re-check obs-backgroundremoval's #984 outcome before implementing.
- **Model format pressure:** obs-backgroundremoval's macOS move to native Core ML suggests the CoreML EP path can lag behind native performance or polish. Benchmark on the M2 Max (ORT + CoreML EP vs. Core ML/Vision directly) in the prototype ticket before locking in.
- **Weights downloaded separately:** ORT and LiteRT load model files from a path at runtime. A native Core ML path would instead need `.mlpackage` download plus on-device compile (`MLModel.compileModel`), which is not covered here.
